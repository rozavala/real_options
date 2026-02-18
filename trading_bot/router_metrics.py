"""
Router Metrics Tracking

Tracks LLM routing metrics including:
- Success/failure rates by provider
- Fallback frequency
- Response latencies
"""

import json
import os
import logging
import threading
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional
from threading import Lock

logger = logging.getLogger(__name__)

_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data'
)

def set_data_dir(data_dir: str):
    """Set the data directory for router metrics (called by orchestrator)."""
    global _data_dir
    _data_dir = data_dir
    logger.info(f"RouterMetrics data_dir set to: {data_dir}")

def _get_metrics_file():
    return os.path.join(_data_dir, 'router_metrics.json')


class RouterMetrics:
    """Singleton class for tracking router metrics."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._metrics = self._load_metrics()
        self._save_lock = Lock()

    def _load_metrics(self) -> dict:
        """Load metrics from disk."""
        try:
            metrics_file = _get_metrics_file()
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load router metrics: {e}")

        return {
            'requests': defaultdict(lambda: {'success': 0, 'failure': 0}),
            'fallbacks': defaultdict(lambda: defaultdict(int)),
            'latencies': defaultdict(list),
            'last_reset': datetime.now(timezone.utc).isoformat()
        }

    def _save_to_disk(self, data):
        """Worker function to save metrics to disk."""
        with self._save_lock:
            try:
                metrics_file = _get_metrics_file()
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                with open(metrics_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                logger.error(f"Failed to save router metrics: {e}")

    def _save_metrics(self):
        """Persist metrics to disk asynchronously."""
        # Convert defaultdicts to regular dicts for JSON serialization
        save_data = {
            'requests': dict(self._metrics.get('requests', {})),
            'fallbacks': {k: dict(v) for k, v in self._metrics.get('fallbacks', {}).items()},
            'latencies': dict(self._metrics.get('latencies', {})),
            'error_types': {k: dict(v) for k, v in self._metrics.get('error_types', {}).items()},
            'last_reset': self._metrics.get('last_reset', datetime.now(timezone.utc).isoformat())
        }

        # Fire and forget thread
        threading.Thread(target=self._save_to_disk, args=(save_data,)).start()

    def record_request(
        self,
        role: str,
        provider: str,
        success: bool,
        latency_ms: Optional[float] = None,
        was_fallback: bool = False,
        primary_provider: Optional[str] = None,
        error_type: Optional[str] = None
    ):
        """
        Record a routing request outcome.

        Args:
            role: Agent role (e.g., 'master', 'agronomist')
            provider: Provider that handled the request
            success: Whether the request succeeded
            latency_ms: Response time in milliseconds
            was_fallback: True if this was a fallback attempt
            primary_provider: Original provider that failed (if was_fallback)
            error_type: Classification of failure (timeout/rate_limit/parse_error/api_error)
        """
        key = f"{role}:{provider}"

        if 'requests' not in self._metrics:
            self._metrics['requests'] = {}
        if key not in self._metrics['requests']:
            self._metrics['requests'][key] = {'success': 0, 'failure': 0}

        if success:
            self._metrics['requests'][key]['success'] += 1
        else:
            self._metrics['requests'][key]['failure'] += 1

        # Track error types for failed requests
        if not success and error_type:
            if 'error_types' not in self._metrics:
                self._metrics['error_types'] = {}
            if key not in self._metrics['error_types']:
                self._metrics['error_types'][key] = {}
            if error_type not in self._metrics['error_types'][key]:
                self._metrics['error_types'][key][error_type] = 0
            self._metrics['error_types'][key][error_type] += 1

        # Track fallback chain
        if was_fallback and primary_provider:
            if 'fallbacks' not in self._metrics:
                self._metrics['fallbacks'] = {}
            fallback_key = f"{primary_provider}->{provider}"
            if role not in self._metrics['fallbacks']:
                self._metrics['fallbacks'][role] = {}
            if fallback_key not in self._metrics['fallbacks'][role]:
                self._metrics['fallbacks'][role][fallback_key] = 0
            self._metrics['fallbacks'][role][fallback_key] += 1

        # Track latencies (keep last 100 per role)
        if latency_ms is not None:
            if 'latencies' not in self._metrics:
                self._metrics['latencies'] = {}
            if key not in self._metrics['latencies']:
                self._metrics['latencies'][key] = []
            self._metrics['latencies'][key].append({
                'ms': latency_ms,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            # Keep only last 100
            self._metrics['latencies'][key] = self._metrics['latencies'][key][-100:]

        # Auto-save periodically (every 10 requests)
        total_requests = sum(
            v.get('success', 0) + v.get('failure', 0)
            for v in self._metrics.get('requests', {}).values()
        )
        if total_requests % 10 == 0:
            self._save_metrics()

    def record_fallback(self, role: str, primary_provider: str, fallback_provider: str, error: str):
        """
        Explicitly records a fallback event.
        """
        fallback_key = f"{primary_provider}->{fallback_provider}"

        if 'fallbacks' not in self._metrics:
            self._metrics['fallbacks'] = {}
        if role not in self._metrics['fallbacks']:
            self._metrics['fallbacks'][role] = {}
        if fallback_key not in self._metrics['fallbacks'][role]:
            self._metrics['fallbacks'][role][fallback_key] = 0

        self._metrics['fallbacks'][role][fallback_key] += 1

        # Store recent error details
        if 'recent_errors' not in self._metrics:
            self._metrics['recent_errors'] = []

        self._metrics['recent_errors'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'role': role,
            'primary': primary_provider,
            'fallback': fallback_provider,
            'error': str(error)[:200]  # Truncate for storage
        })
        # Keep last 50
        self._metrics['recent_errors'] = self._metrics['recent_errors'][-50:]

        logger.warning(f"Fallback event: {role} {primary_provider} -> {fallback_provider}. Reason: {error}")

    def get_summary(self) -> dict:
        """Get a summary of routing metrics."""
        summary = {
            'total_requests': 0,
            'total_successes': 0,
            'total_failures': 0,
            'fallback_count': 0,
            'by_provider': {},
            'by_role': {},
            'top_fallback_chains': [],
            'last_reset': self._metrics.get('last_reset')
        }

        # Aggregate by provider and role
        for key, counts in self._metrics.get('requests', {}).items():
            role, provider = key.split(':') if ':' in key else ('unknown', key)

            success = counts.get('success', 0)
            failure = counts.get('failure', 0)

            summary['total_requests'] += success + failure
            summary['total_successes'] += success
            summary['total_failures'] += failure

            # By provider
            if provider not in summary['by_provider']:
                summary['by_provider'][provider] = {'success': 0, 'failure': 0}
            summary['by_provider'][provider]['success'] += success
            summary['by_provider'][provider]['failure'] += failure

            # By role
            if role not in summary['by_role']:
                summary['by_role'][role] = {'success': 0, 'failure': 0}
            summary['by_role'][role]['success'] += success
            summary['by_role'][role]['failure'] += failure

        # Count fallbacks
        for role_fallbacks in self._metrics.get('fallbacks', {}).values():
            for chain, count in role_fallbacks.items():
                summary['fallback_count'] += count

        # Top fallback chains
        all_chains = []
        for role, chains in self._metrics.get('fallbacks', {}).items():
            for chain, count in chains.items():
                all_chains.append({'role': role, 'chain': chain, 'count': count})

        summary['top_fallback_chains'] = sorted(
            all_chains,
            key=lambda x: x['count'],
            reverse=True
        )[:10]

        # Calculate success rates
        if summary['total_requests'] > 0:
            summary['overall_success_rate'] = summary['total_successes'] / summary['total_requests']
        else:
            summary['overall_success_rate'] = 1.0

        for provider, counts in summary['by_provider'].items():
            total = counts['success'] + counts['failure']
            counts['success_rate'] = counts['success'] / total if total > 0 else 1.0

        return summary

    def reset(self):
        """Reset all metrics."""
        self._metrics = {
            'requests': {},
            'fallbacks': {},
            'latencies': {},
            'last_reset': datetime.now(timezone.utc).isoformat()
        }
        self._save_metrics()
        logger.info("Router metrics reset")


# Module-level singleton accessor
def get_router_metrics() -> RouterMetrics:
    """Get the singleton RouterMetrics instance."""
    return RouterMetrics()
