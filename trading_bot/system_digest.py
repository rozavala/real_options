"""
System Health Digest — daily post-close summary of system health.

Reads ~15 data files per commodity and synthesizes them into a single JSON
artifact that an LLM or human can read cold to understand what's broken,
degrading, or needs attention.

Zero risk to trading loop: reads only, no IB connections, no LLM calls.
"""

import gzip
import hashlib
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Guarded imports — digest still works if these fail
try:
    from trading_bot.timestamps import parse_ts_column
    _HAS_TIMESTAMPS = True
except ImportError:
    _HAS_TIMESTAMPS = False

try:
    from trading_bot.weighted_voting import DOMAIN_WEIGHTS, TriggerType
    _HAS_WEIGHTS = True
except ImportError:
    _HAS_WEIGHTS = False

try:
    from trading_bot.agent_names import normalize_agent_name
    _HAS_AGENT_NAMES = True
except ImportError:
    _HAS_AGENT_NAMES = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read_json(path: str) -> Optional[dict]:
    """Read a JSON file, returning None on any error."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to read JSON {path}: {e}")
        return None


def _safe_read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file, returning empty DataFrame on any error."""
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, on_bad_lines='warn')
    except Exception as e:
        logger.debug(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()


def _safe_float(val) -> Optional[float]:
    """Convert value to float, returning None on failure."""
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_data_dir(config: dict, ticker: str) -> str:
    """Resolve per-commodity data directory."""
    base = config.get('data_dir', 'data')
    return os.path.join(base, ticker)


def _load_yesterday_digest(config: dict) -> Optional[dict]:
    """Load the most recent archived digest for delta comparison."""
    digest_dir = os.path.join('logs', 'digests')
    if not os.path.isdir(digest_dir):
        return None
    try:
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        # Find most recent archive that isn't today
        candidates = sorted(
            [f for f in os.listdir(digest_dir)
             if f.endswith('.json.gz') and not f.startswith(today_str)],
            reverse=True
        )
        if not candidates:
            return None
        path = os.path.join(digest_dir, candidates[0])
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load yesterday's digest: {e}")
        return None


def _parse_timestamp_column(df: pd.DataFrame, col: str = 'timestamp') -> pd.DataFrame:
    """Parse timestamp column using project's parser, with fallback."""
    if col not in df.columns:
        return df
    if _HAS_TIMESTAMPS:
        df[col] = parse_ts_column(df[col], errors='coerce')
    else:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    return df


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


def _filter_today(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    """Filter DataFrame to today's entries (UTC)."""
    if df.empty or ts_col not in df.columns:
        return pd.DataFrame()
    today = _today_utc_str()
    mask = df[ts_col].dt.strftime('%Y-%m-%d') == today
    return df[mask]


# ---------------------------------------------------------------------------
# v1.0 Per-Commodity Builders
# ---------------------------------------------------------------------------

def _build_feedback_loop(data_dir: str) -> dict:
    """Feedback loop health from enhanced_brier.json."""
    try:
        path = os.path.join(data_dir, 'enhanced_brier.json')
        data = _safe_read_json(path)
        if not data:
            return {
                'status': 'no_data',
                'resolution_rate': None,
                'total_predictions': 0,
                'resolved': 0,
                'pending': 0,
                'thresholds': {'healthy': 0.75, 'critical': 0.50},
            }

        predictions = data.get('predictions', [])
        total = len(predictions)
        if total == 0:
            return {
                'status': 'empty',
                'resolution_rate': None,
                'total_predictions': 0,
                'resolved': 0,
                'pending': 0,
                'thresholds': {'healthy': 0.75, 'critical': 0.50},
            }

        # Pending = no actual_outcome AND no resolved_at
        pending = sum(
            1 for p in predictions
            if p.get('actual_outcome') is None and p.get('resolved_at') is None
        )
        resolved = total - pending
        resolution_rate = resolved / total if total > 0 else 0.0

        if resolution_rate >= 0.75:
            status = 'healthy'
        elif resolution_rate >= 0.50:
            status = 'degraded'
        else:
            status = 'critical'

        return {
            'status': status,
            'resolution_rate': round(resolution_rate, 3),
            'total_predictions': total,
            'resolved': resolved,
            'pending': pending,
            'thresholds': {'healthy': 0.75, 'critical': 0.50},
        }
    except Exception as e:
        logger.debug(f"_build_feedback_loop error: {e}")
        return {'status': 'error', 'error': str(e)}


def _build_agent_calibration(data_dir: str) -> dict:
    """Agent Brier scores and weight multipliers."""
    try:
        from trading_bot.brier_scoring import BrierScoreTracker
        history_file = os.path.join(data_dir, 'agent_accuracy.csv')
        tracker = BrierScoreTracker(history_file=history_file)
        scores = tracker.scores or {}
        agents = {}
        for agent, score in scores.items():
            agents[agent] = {
                'brier_score': round(score, 4),
                'weight_multiplier': round(tracker.get_agent_weight_multiplier(agent), 3),
            }
        return {
            'agents': agents,
            'avg_brier': round(sum(scores.values()) / len(scores), 4) if scores else None,
            'tracked_count': len(scores),
        }
    except Exception as e:
        logger.debug(f"_build_agent_calibration error: {e}")
        return {'agents': {}, 'avg_brier': None, 'tracked_count': 0, 'error': str(e)}


def _build_cognitive_layer(ch_df: pd.DataFrame) -> dict:
    """Council decision summary from council_history.csv (today only)."""
    try:
        today_df = _filter_today(ch_df)
        if today_df.empty:
            return {
                'decisions_today': 0,
                'bull_pct': None, 'bear_pct': None, 'neutral_pct': None,
                'avg_confidence': None, 'avg_weighted_score': None,
                'strategies_used': [],
            }

        n = len(today_df)

        # Direction breakdown
        direction_col = 'master_direction' if 'master_direction' in today_df.columns else None
        bull_pct = bear_pct = neutral_pct = None
        if direction_col:
            dirs = today_df[direction_col].str.upper().fillna('')
            bull_pct = round((dirs.str.contains('BULL')).sum() / n, 3)
            bear_pct = round((dirs.str.contains('BEAR')).sum() / n, 3)
            neutral_pct = round(1.0 - bull_pct - bear_pct, 3)

        # Confidence
        avg_confidence = None
        if 'master_confidence' in today_df.columns:
            conf_vals = pd.to_numeric(today_df['master_confidence'], errors='coerce')
            avg_confidence = round(conf_vals.mean(), 3) if not conf_vals.isna().all() else None

        # Weighted score
        avg_weighted_score = None
        if 'weighted_score' in today_df.columns:
            ws_vals = pd.to_numeric(today_df['weighted_score'], errors='coerce')
            avg_weighted_score = round(ws_vals.mean(), 3) if not ws_vals.isna().all() else None

        # Strategies
        strategies = []
        if 'strategy' in today_df.columns:
            strategies = today_df['strategy'].dropna().unique().tolist()

        return {
            'decisions_today': n,
            'bull_pct': bull_pct,
            'bear_pct': bear_pct,
            'neutral_pct': neutral_pct,
            'avg_confidence': avg_confidence,
            'avg_weighted_score': avg_weighted_score,
            'strategies_used': strategies,
        }
    except Exception as e:
        logger.debug(f"_build_cognitive_layer error: {e}")
        return {'decisions_today': 0, 'error': str(e)}


def _build_sentinel_efficiency(data_dir: str) -> dict:
    """Sentinel alert/trade efficiency from sentinel_stats.json."""
    try:
        path = os.path.join(data_dir, 'sentinel_stats.json')
        data = _safe_read_json(path)
        if not data:
            return {'sentinels': {}, 'total_alerts': 0, 'total_trades_triggered': 0}

        sentinels_data = data.get('sentinels', data)
        result = {}
        total_alerts = 0
        total_trades = 0

        for name, stats in sentinels_data.items():
            if not isinstance(stats, dict):
                continue
            alerts = stats.get('total_alerts', 0)
            trades = stats.get('trades_triggered', 0)
            snr = round(trades / alerts, 4) if alerts > 0 else None
            result[name] = {
                'total_alerts': alerts,
                'trades_triggered': trades,
                'signal_to_noise': snr,
            }
            total_alerts += alerts
            total_trades += trades

        return {
            'sentinels': result,
            'total_alerts': total_alerts,
            'total_trades_triggered': total_trades,
        }
    except Exception as e:
        logger.debug(f"_build_sentinel_efficiency error: {e}")
        return {'sentinels': {}, 'error': str(e)}


def _build_efficiency(data_dir: str, config: dict) -> dict:
    """LLM cost efficiency from budget_state.json + router_metrics.json + semantic cache."""
    try:
        budget = _safe_read_json(os.path.join(data_dir, 'budget_state.json')) or {}
        router = _safe_read_json(os.path.join(data_dir, 'router_metrics.json')) or {}

        # Semantic cache stats
        cache_stats = None
        try:
            from trading_bot.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cache_stats = cache.get_stats()
        except Exception:
            pass

        return {
            'daily_spend_usd': _safe_float(budget.get('daily_spend', 0)),
            'request_count': budget.get('request_count', 0),
            'router_metrics': {
                k: v for k, v in router.items()
                if k in ('total_requests', 'total_errors', 'avg_latency_ms', 'provider_stats')
            } if router else {},
            'semantic_cache': cache_stats,
        }
    except Exception as e:
        logger.debug(f"_build_efficiency error: {e}")
        return {'error': str(e)}


def _build_risk_rails(data_dir: str, config: dict) -> dict:
    """Risk status from drawdown_state.json + var_state.json."""
    try:
        drawdown = _safe_read_json(os.path.join(data_dir, 'drawdown_state.json')) or {}
        # VaR is shared (project root data/), not per-commodity
        var_path = os.path.join(os.path.dirname(data_dir), 'var_state.json')
        var_data = _safe_read_json(var_path) or {}

        dd_config = config.get('drawdown_circuit_breaker', {})

        return {
            'drawdown': {
                'status': drawdown.get('status', 'unknown'),
                'current_pct': _safe_float(drawdown.get('current_drawdown_pct')),
                'thresholds': {
                    'warning_pct': dd_config.get('warning_pct'),
                    'halt_pct': dd_config.get('halt_pct'),
                    'panic_pct': dd_config.get('panic_pct'),
                },
            },
            'var': {
                'utilization': _safe_float(var_data.get('utilization')),
                'var_95': _safe_float(var_data.get('var_95')),
                'var_99': _safe_float(var_data.get('var_99')),
                'enforcement_mode': config.get('compliance', {}).get('var_enforcement_mode', 'unknown'),
            },
        }
    except Exception as e:
        logger.debug(f"_build_risk_rails error: {e}")
        return {'error': str(e)}


# ---------------------------------------------------------------------------
# v1.1 Per-Commodity Builders
# ---------------------------------------------------------------------------

# Legacy column → canonical agent name mapping
_LEGACY_COLUMN_MAP = {
    'meteorologist_summary': 'agronomist',
    'meteorologist_sentiment': 'agronomist',
    'fundamentalist_summary': 'inventory',
    'fundamentalist_sentiment': 'inventory',
}


def _build_decision_traces(ch_df: pd.DataFrame, max_traces: int = 5) -> list:
    """Last N council decisions with vote breakdowns."""
    try:
        if ch_df.empty:
            return []

        # Sort by timestamp descending, take last N
        df = ch_df.sort_values('timestamp', ascending=False).head(max_traces)

        traces = []
        for _, row in df.iterrows():
            # Parse vote_breakdown JSON
            top_contributors = []
            try:
                vb_raw = row.get('vote_breakdown', '')
                if isinstance(vb_raw, str) and vb_raw.strip():
                    vb = json.loads(vb_raw)
                    if isinstance(vb, dict):
                        # Sort by absolute contribution, take top 2
                        sorted_agents = sorted(
                            vb.items(), key=lambda x: abs(float(x[1])) if isinstance(x[1], (int, float, str)) else 0,
                            reverse=True
                        )[:2]
                        for agent, weight in sorted_agents:
                            # Get agent's key argument from summary columns
                            summary_col = f"{agent}_summary"
                            # Check legacy mapping
                            for legacy, canonical in _LEGACY_COLUMN_MAP.items():
                                if canonical == agent and legacy in row.index:
                                    summary_col = legacy
                                    break
                            argument = str(row.get(summary_col, ''))[:150] if summary_col in row.index else ''
                            top_contributors.append({
                                'agent': agent,
                                'weight': _safe_float(weight),
                                'key_argument': argument,
                            })
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            # Contrarian views from dissent_acknowledged
            dissent = str(row.get('dissent_acknowledged', ''))[:200] if 'dissent_acknowledged' in row.index else ''

            traces.append({
                'timestamp': str(row.get('timestamp', '')),
                'direction': row.get('master_direction', ''),
                'confidence': _safe_float(row.get('master_confidence')),
                'strategy': row.get('strategy', ''),
                'top_contributors': top_contributors,
                'contrarian_view': dissent,
                'realized_pnl': _safe_float(row.get('realized_pnl')),
            })

        return traces
    except Exception as e:
        logger.debug(f"_build_decision_traces error: {e}")
        return []


def _build_data_freshness(data_dir: str) -> dict:
    """Per-sentinel data freshness from state.json sentinel_health namespace."""
    try:
        from trading_bot.state_manager import StateManager
        # Temporarily set data dir for StateManager read
        sentinel_health = StateManager.load_state_raw(namespace="sentinel_health")

        if not sentinel_health:
            return {'sentinels': {}, 'status': 'no_data'}

        now = datetime.now(timezone.utc).timestamp()
        result = {}

        for sentinel_name, entry in sentinel_health.items():
            if not isinstance(entry, dict):
                continue
            ts = entry.get('timestamp')
            data = entry.get('data', entry)
            interval_seconds = None
            if isinstance(data, dict):
                interval_seconds = data.get('interval_seconds')

            last_check_minutes_ago = None
            is_stale = False
            if ts:
                elapsed_seconds = now - float(ts)
                last_check_minutes_ago = round(elapsed_seconds / 60, 1)
                if interval_seconds and interval_seconds > 0:
                    is_stale = elapsed_seconds > (2 * interval_seconds)

            result[sentinel_name] = {
                'last_check_minutes_ago': last_check_minutes_ago,
                'check_interval_seconds': interval_seconds,
                'is_stale': is_stale,
            }

        stale_count = sum(1 for v in result.values() if v.get('is_stale'))
        status = 'healthy' if stale_count == 0 else ('degraded' if stale_count <= 2 else 'critical')

        return {
            'sentinels': result,
            'stale_count': stale_count,
            'status': status,
        }
    except Exception as e:
        logger.debug(f"_build_data_freshness error: {e}")
        return {'sentinels': {}, 'status': 'error', 'error': str(e)}


def _build_regime_context(data_dir: str) -> dict:
    """Current fundamental regime from fundamental_regime.json."""
    try:
        path = os.path.join(data_dir, 'fundamental_regime.json')
        data = _safe_read_json(path)
        if not data:
            return {'regime': 'UNKNOWN', 'confidence': None, 'updated_at': None}

        return {
            'regime': data.get('regime', 'UNKNOWN'),
            'confidence': _safe_float(data.get('confidence')),
            'updated_at': data.get('updated_at'),
        }
    except Exception as e:
        logger.debug(f"_build_regime_context error: {e}")
        return {'regime': 'UNKNOWN', 'error': str(e)}


def _build_agent_contribution(ch_df: pd.DataFrame) -> dict:
    """30-day agent agreement rate with master decision."""
    try:
        if ch_df.empty or 'timestamp' not in ch_df.columns:
            return {'agents': {}}

        # Filter to last 30 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent = ch_df[ch_df['timestamp'] >= cutoff] if not ch_df.empty else ch_df
        if recent.empty:
            return {'agents': {}}

        master_col = 'master_direction'
        if master_col not in recent.columns:
            return {'agents': {}}

        master_dirs = recent[master_col].str.upper().fillna('')

        # Sentiment columns to check
        sentiment_cols = {
            'agronomist_sentiment': 'agronomist',
            'macro_sentiment': 'macro',
            'geopolitical_sentiment': 'geopolitical',
            'supply_chain_sentiment': 'supply_chain',
            'inventory_sentiment': 'inventory',
            'sentiment_sentiment': 'sentiment',
            'technical_sentiment': 'technical',
            'volatility_sentiment': 'volatility',
            # Legacy columns
            'meteorologist_sentiment': 'agronomist',
            'fundamentalist_sentiment': 'inventory',
        }

        agents = {}
        seen_canonical = set()
        for col, canonical in sentiment_cols.items():
            if canonical in seen_canonical:
                continue
            if col not in recent.columns:
                # Try legacy fallback
                continue
            seen_canonical.add(canonical)
            agent_dirs = recent[col].str.upper().fillna('')
            valid = (master_dirs != '') & (agent_dirs != '')
            if valid.sum() == 0:
                continue
            agreement = ((master_dirs[valid] == agent_dirs[valid]).sum() / valid.sum())
            agents[canonical] = {
                'agreement_rate_with_master': round(agreement, 3),
                'samples': int(valid.sum()),
            }

        return {'agents': agents}
    except Exception as e:
        logger.debug(f"_build_agent_contribution error: {e}")
        return {'agents': {}, 'error': str(e)}


# ---------------------------------------------------------------------------
# v1.0 Account-Wide Builders
# ---------------------------------------------------------------------------

def _build_portfolio(config: dict) -> dict:
    """Portfolio overview from daily_equity.csv."""
    try:
        base_dir = config.get('data_dir', 'data')
        equity_path = os.path.join(base_dir, 'daily_equity.csv')
        df = _safe_read_csv(equity_path)
        if df.empty or 'total_value_usd' not in df.columns:
            return {'status': 'no_data'}

        df['total_value_usd'] = pd.to_numeric(df['total_value_usd'], errors='coerce')
        df = df.dropna(subset=['total_value_usd'])
        if df.empty:
            return {'status': 'no_data'}

        nlv = df['total_value_usd'].iloc[-1]

        # Daily P&L
        daily_pnl = None
        if len(df) >= 2:
            daily_pnl = round(nlv - df['total_value_usd'].iloc[-2], 2)

        # LTD return
        ltd_return = None
        if len(df) >= 2:
            first_val = df['total_value_usd'].iloc[0]
            if first_val and first_val > 0:
                ltd_return = round((nlv - first_val) / first_val * 100, 2)

        # Max drawdown 30d
        max_dd_30d = None
        recent = df.tail(30)
        if len(recent) >= 2:
            peak = recent['total_value_usd'].expanding().max()
            drawdown = (recent['total_value_usd'] - peak) / peak * 100
            max_dd_30d = round(drawdown.min(), 2)

        # VaR from shared var_state.json
        var_path = os.path.join(base_dir, 'var_state.json')
        var_data = _safe_read_json(var_path) or {}

        return {
            'nlv_usd': round(nlv, 2),
            'daily_pnl_usd': daily_pnl,
            'ltd_return_pct': ltd_return,
            'max_drawdown_30d_pct': max_dd_30d,
            'var_95': _safe_float(var_data.get('var_95')),
            'var_99': _safe_float(var_data.get('var_99')),
            'equity_data_points': len(df),
        }
    except Exception as e:
        logger.debug(f"_build_portfolio error: {e}")
        return {'status': 'error', 'error': str(e)}


def _build_changes(config: dict, active_tickers: list, yesterday_digest: Optional[dict]) -> dict:
    """Delta analysis: what changed since yesterday's digest."""
    try:
        changes = []

        if not yesterday_digest:
            return {'changes': [], 'note': 'No previous digest for comparison'}

        # Compare per-commodity decision counts
        yesterday_commodities = yesterday_digest.get('commodities', {})
        for ticker in active_tickers:
            data_dir = _get_data_dir(config, ticker)
            ch_path = os.path.join(data_dir, 'council_history.csv')
            ch_df = _safe_read_csv(ch_path)
            if not ch_df.empty and 'timestamp' in ch_df.columns:
                ch_df = _parse_timestamp_column(ch_df)
                today_count = len(_filter_today(ch_df))
            else:
                today_count = 0

            yest_count = yesterday_commodities.get(ticker, {}).get('cognitive_layer', {}).get('decisions_today', 0)
            if today_count != yest_count:
                changes.append({
                    'component': f'{ticker}/decisions',
                    'yesterday': yest_count,
                    'today': today_count,
                })

        # Compare portfolio
        today_portfolio = _build_portfolio(config)
        yest_portfolio = yesterday_digest.get('portfolio', {})
        if today_portfolio.get('nlv_usd') and yest_portfolio.get('nlv_usd'):
            delta = today_portfolio['nlv_usd'] - yest_portfolio['nlv_usd']
            if abs(delta) > 0.01:
                changes.append({
                    'component': 'portfolio/nlv',
                    'yesterday': yest_portfolio['nlv_usd'],
                    'today': today_portfolio['nlv_usd'],
                    'delta': round(delta, 2),
                })

        return {'changes': changes}
    except Exception as e:
        logger.debug(f"_build_changes error: {e}")
        return {'changes': [], 'error': str(e)}


def _build_rolling_trends(config: dict) -> dict:
    """7d/30d equity trends from daily_equity.csv."""
    try:
        base_dir = config.get('data_dir', 'data')
        equity_path = os.path.join(base_dir, 'daily_equity.csv')
        df = _safe_read_csv(equity_path)
        if df.empty or 'total_value_usd' not in df.columns:
            return {'status': 'no_data'}

        df['total_value_usd'] = pd.to_numeric(df['total_value_usd'], errors='coerce')
        df = df.dropna(subset=['total_value_usd'])

        result = {}

        # 7d delta
        if len(df) >= 7:
            result['equity_delta_7d'] = round(df['total_value_usd'].iloc[-1] - df['total_value_usd'].iloc[-7], 2)
        # 30d delta
        if len(df) >= 30:
            result['equity_delta_30d'] = round(df['total_value_usd'].iloc[-1] - df['total_value_usd'].iloc[-30], 2)

        # Avg daily P&L (last 30 entries)
        recent = df.tail(30)
        if len(recent) >= 2:
            daily_returns = recent['total_value_usd'].diff().dropna()
            result['avg_daily_pnl'] = round(daily_returns.mean(), 2)

            # Sharpe estimate (annualized, assuming ~252 trading days)
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
                result['sharpe_estimate'] = round(sharpe, 2)

        return result
    except Exception as e:
        logger.debug(f"_build_rolling_trends error: {e}")
        return {'status': 'error', 'error': str(e)}


def _build_improvement_opportunities(commodity_blocks: dict) -> list:
    """Deterministic threshold-based improvement flags."""
    opportunities = []
    try:
        for ticker, block in commodity_blocks.items():
            # Feedback loop health
            fb = block.get('feedback_loop', {})
            if fb.get('status') == 'critical':
                opportunities.append({
                    'priority': 'HIGH',
                    'component': f'{ticker}/feedback_loop',
                    'observation': f"Resolution rate {fb.get('resolution_rate', 'N/A')} below critical threshold 0.50",
                    'suggestion': 'Check Brier reconciliation pipeline — predictions may not be resolving',
                })
            elif fb.get('status') == 'degraded':
                opportunities.append({
                    'priority': 'MEDIUM',
                    'component': f'{ticker}/feedback_loop',
                    'observation': f"Resolution rate {fb.get('resolution_rate', 'N/A')} below healthy threshold 0.75",
                    'suggestion': 'Review pending predictions for stale entries',
                })

            # Noisy sentinels (SNR < 0.05 AND >= 10 alerts)
            se = block.get('sentinel_efficiency', {})
            for s_name, s_data in se.get('sentinels', {}).items():
                if (s_data.get('signal_to_noise') is not None
                        and s_data['signal_to_noise'] < 0.05
                        and s_data.get('total_alerts', 0) >= 10):
                    opportunities.append({
                        'priority': 'MEDIUM',
                        'component': f'{ticker}/sentinel/{s_name}',
                        'observation': f"SNR {s_data['signal_to_noise']:.4f} with {s_data['total_alerts']} alerts — mostly noise",
                        'suggestion': 'Increase trigger threshold or add debounce',
                    })

            # Weak agents (Brier < 0.10 AND >= 30 samples via tracker)
            cal = block.get('agent_calibration', {})
            for agent, agent_data in cal.get('agents', {}).items():
                brier = agent_data.get('brier_score')
                if brier is not None and brier < 0.10:
                    opportunities.append({
                        'priority': 'LOW',
                        'component': f'{ticker}/agent/{agent}',
                        'observation': f"Brier score {brier:.4f} — near-zero predictive signal",
                        'suggestion': 'Review agent prompt or consider downweighting',
                    })

            # Stale data sources
            freshness = block.get('data_freshness', {})
            if freshness.get('stale_count', 0) > 2:
                opportunities.append({
                    'priority': 'HIGH',
                    'component': f'{ticker}/data_freshness',
                    'observation': f"{freshness['stale_count']} sentinels have stale data",
                    'suggestion': 'Check sentinel connectivity and API keys',
                })

    except Exception as e:
        logger.debug(f"_build_improvement_opportunities error: {e}")

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    opportunities.sort(key=lambda x: priority_order.get(x.get('priority', 'LOW'), 3))
    return opportunities


# ---------------------------------------------------------------------------
# v1.1 Account-Wide Builders
# ---------------------------------------------------------------------------

def _build_config_snapshot(config: dict, active_tickers: list) -> dict:
    """Allowlisted config extraction — no secrets, no raw config copy."""
    try:
        snapshot = {}

        # Agent base weights for scheduled triggers
        if _HAS_WEIGHTS:
            try:
                scheduled_weights = DOMAIN_WEIGHTS.get(TriggerType.SCHEDULED, {})
                snapshot['agent_base_weights_scheduled'] = dict(scheduled_weights)
            except Exception:
                snapshot['agent_base_weights_scheduled'] = None
        else:
            snapshot['agent_base_weights_scheduled'] = None

        # Risk management
        snapshot['risk'] = config.get('risk_management', {})

        # Drawdown circuit breaker
        snapshot['drawdown'] = config.get('drawdown_circuit_breaker', {})

        # Execution: base + per-commodity overrides
        base_tuning = config.get('strategy_tuning', {})
        execution = {
            'base': {k: v for k, v in base_tuning.items()
                     if k not in ('order_type',)},  # Keep all non-secret fields
        }
        for ticker in active_tickers:
            overrides = config.get('commodity_overrides', {}).get(ticker, {}).get('strategy_tuning', {})
            if overrides:
                execution[ticker] = overrides
        snapshot['execution'] = execution

        # Iron condor specifics
        snapshot['iron_condor'] = {
            'short_strikes_from_atm': base_tuning.get('iron_condor_short_strikes_from_atm'),
            'wing_strikes_apart': base_tuning.get('iron_condor_wing_strikes_apart'),
        }

        # Sentinel thresholds
        sentinels_cfg = config.get('sentinels', {})
        snapshot['sentinel_thresholds'] = {
            'weather_frost_temp_c': sentinels_cfg.get('weather', {}).get('triggers', {}).get('frost_temp_c'),
            'price_pct_change_threshold': sentinels_cfg.get('price', {}).get('pct_change_threshold'),
            'microstructure_spread_std_threshold': sentinels_cfg.get('microstructure', {}).get('spread_std_threshold'),
        }

        # Semantic cache
        snapshot['semantic_cache'] = config.get('semantic_cache', {})

        # Strategy
        snapshot['strategy'] = config.get('strategy', {})

        # Brier scoring
        snapshot['brier'] = {
            'enhanced_weight': config.get('brier_scoring', {}).get('enhanced_weight'),
            'note': 'decay HALF_LIFE_DAYS=14 is hardcoded in brier_scoring.py',
        }

        # LLM budget
        snapshot['llm_budget_daily_usd'] = config.get('cost_management', {}).get('daily_budget_usd', 15.0)

        return snapshot
    except Exception as e:
        logger.debug(f"_build_config_snapshot error: {e}")
        return {'error': str(e)}


def _build_error_telemetry(config: dict, active_tickers: list) -> dict:
    """Error categorization from order_events.csv across all commodities."""
    try:
        base_dir = config.get('data_dir', 'data')
        all_errors = {
            'liquidity_reject': 0,
            'margin_reject': 0,
            'order_timeout': 0,
            'order_error': 0,
            'trading_execution': 0,
        }
        per_commodity = {}

        for ticker in active_tickers:
            data_dir = _get_data_dir(config, ticker)
            events_path = os.path.join(data_dir, 'order_events.csv')
            df = _safe_read_csv(events_path)
            commodity_errors = {
                'liquidity_reject': 0,
                'margin_reject': 0,
                'order_timeout': 0,
                'order_error': 0,
                'trading_execution': 0,
            }

            if not df.empty and 'event_type' in df.columns:
                # Parse timestamps and filter to today
                if 'timestamp' in df.columns:
                    df = _parse_timestamp_column(df)
                    df = _filter_today(df)

                if not df.empty:
                    events = df['event_type'].str.lower().fillna('')
                    for _, evt in events.items():
                        if 'liquidity' in evt or 'spread' in evt:
                            commodity_errors['liquidity_reject'] += 1
                        elif 'margin' in evt:
                            commodity_errors['margin_reject'] += 1
                        elif 'timeout' in evt:
                            commodity_errors['order_timeout'] += 1
                        elif 'execution' in evt or 'fill' in evt:
                            commodity_errors['trading_execution'] += 1
                        elif 'error' in evt or 'reject' in evt or 'fail' in evt:
                            commodity_errors['order_error'] += 1

            per_commodity[ticker] = commodity_errors
            for k in all_errors:
                all_errors[k] += commodity_errors[k]

        # Error reporter state (account-wide) — extract counts only, no raw messages
        reporter_state = _safe_read_json(os.path.join(base_dir, 'error_reporter_state.json'))
        reporter_summary = None
        if reporter_state and isinstance(reporter_state, dict):
            reporter_summary = {
                'total_errors_reported': reporter_state.get('total_errors_reported', 0),
                'last_report_time': reporter_state.get('last_report_time'),
            }

        total = sum(all_errors.values())
        high_impact = all_errors['trading_execution'] > 0 or total > 5

        return {
            'totals': all_errors,
            'per_commodity': per_commodity,
            'total_errors_today': total,
            'high_impact': high_impact,
            'error_reporter': reporter_summary,
        }
    except Exception as e:
        logger.debug(f"_build_error_telemetry error: {e}")
        return {'totals': {}, 'total_errors_today': 0, 'high_impact': False, 'error': str(e)}


def _build_executive_summary(digest: dict) -> str:
    """Template-based executive summary."""
    try:
        parts = []

        # Portfolio status
        portfolio = digest.get('portfolio', {})
        if portfolio.get('nlv_usd'):
            pnl = portfolio.get('daily_pnl_usd')
            pnl_str = f"{'+' if pnl >= 0 else '-'}${abs(pnl):.2f}" if pnl is not None else "N/A"
            parts.append(f"Portfolio NLV ${portfolio['nlv_usd']:,.2f} (daily P&L {pnl_str})")

        # Per-commodity summaries
        for ticker, block in digest.get('commodities', {}).items():
            cog = block.get('cognitive_layer', {})
            decisions = cog.get('decisions_today', 0)
            regime = block.get('regime_context', {}).get('regime', 'UNKNOWN')
            parts.append(f"{ticker}: {decisions} decisions, regime={regime}")

        # Feedback loop health
        degraded = []
        for ticker, block in digest.get('commodities', {}).items():
            fb = block.get('feedback_loop', {})
            if fb.get('status') in ('degraded', 'critical'):
                degraded.append(f"{ticker} ({fb['status']}: resolution_rate={fb.get('resolution_rate')})")
        if degraded:
            parts.append(f"Degraded feedback loops: {', '.join(degraded)}")

        # High-priority issues
        opportunities = digest.get('improvement_opportunities', [])
        high_priority = [o for o in opportunities if o.get('priority') == 'HIGH']
        if high_priority:
            parts.append(f"{len(high_priority)} HIGH-priority issue(s) requiring attention")
        else:
            parts.append("No high-priority issues")

        # Health score
        health = digest.get('system_health_score', {})
        if health.get('overall') is not None:
            parts.append(f"System health score: {health['overall']:.2f}/1.00")

        return " | ".join(parts)
    except Exception as e:
        logger.debug(f"_build_executive_summary error: {e}")
        return f"Summary generation failed: {e}"


def _build_system_health_score(digest: dict) -> dict:
    """
    Deterministic composite health score.

    Formula (documented, no magic):
      overall = 0.35 * feedback_norm + 0.25 * (1 - avg_brier_norm) + 0.25 * exec_quality + 0.15 * sentinel_snr_avg

    Per-component normalization:
      feedback = min(1.0, resolution_rate / 0.75)
      brier = max(0, 1 - avg_brier / 0.5)
      exec_quality = 1.0 - min(1.0, error_rate)
      sentinel_snr = avg of non-null SNR values, default 0.5 if none
    """
    try:
        commodities = digest.get('commodities', {})

        # Average feedback resolution rate across commodities
        fb_rates = []
        for block in commodities.values():
            rate = block.get('feedback_loop', {}).get('resolution_rate')
            if rate is not None:
                fb_rates.append(rate)
        feedback_norm = min(1.0, (sum(fb_rates) / len(fb_rates)) / 0.75) if fb_rates else 0.5

        # Average Brier score across commodities
        brier_vals = []
        for block in commodities.values():
            avg_b = block.get('agent_calibration', {}).get('avg_brier')
            if avg_b is not None:
                brier_vals.append(avg_b)
        brier_norm = max(0, 1 - (sum(brier_vals) / len(brier_vals)) / 0.5) if brier_vals else 0.5

        # Execution quality from error telemetry
        telemetry = digest.get('error_telemetry', {})
        total_errors = telemetry.get('total_errors_today', 0)
        # Normalize: 0 errors = 1.0, 10+ errors = 0.0
        error_rate = min(1.0, total_errors / 10.0)
        exec_quality = 1.0 - error_rate

        # Average sentinel SNR across commodities
        snr_vals = []
        for block in commodities.values():
            for s in block.get('sentinel_efficiency', {}).get('sentinels', {}).values():
                snr = s.get('signal_to_noise')
                if snr is not None:
                    snr_vals.append(snr)
        sentinel_snr_avg = sum(snr_vals) / len(snr_vals) if snr_vals else 0.5

        overall = (
            0.35 * feedback_norm
            + 0.25 * brier_norm
            + 0.25 * exec_quality
            + 0.15 * sentinel_snr_avg
        )

        return {
            'overall': round(overall, 4),
            'components': {
                'feedback_health': round(feedback_norm, 4),
                'prediction_accuracy': round(brier_norm, 4),
                'execution_quality': round(exec_quality, 4),
                'sentinel_efficiency': round(sentinel_snr_avg, 4),
            },
            'weights': {
                'feedback_health': 0.35,
                'prediction_accuracy': 0.25,
                'execution_quality': 0.25,
                'sentinel_efficiency': 0.15,
            },
            'formula': 'overall = 0.35*feedback + 0.25*(1-brier) + 0.25*exec_quality + 0.15*sentinel_snr',
        }
    except Exception as e:
        logger.debug(f"_build_system_health_score error: {e}")
        return {'overall': None, 'error': str(e)}


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_system_digest(config: dict) -> Optional[dict]:
    """
    Generate daily System Health Digest.

    Synchronous — call via asyncio.to_thread() from the orchestrator.
    Reads ~15 data files per commodity, produces a single JSON summary.
    No IB connections, no LLM calls.

    Args:
        config: Full application config dict

    Returns:
        Digest dict if successful, None on failure
    """
    try:
        logger.info("--- Generating System Health Digest ---")
        now = datetime.now(timezone.utc)

        # 1. Determine active tickers
        active_tickers = config.get('commodities', [])
        if not active_tickers:
            ticker = config.get('commodity', {}).get('ticker', 'KC')
            active_tickers = [ticker]

        # 2. Load yesterday's digest for delta comparison
        yesterday_digest = _load_yesterday_digest(config)

        # 3. Build per-commodity sections
        commodity_blocks = {}
        for ticker in active_tickers:
            data_dir = _get_data_dir(config, ticker)
            if not os.path.isdir(data_dir):
                logger.warning(f"Data directory missing for {ticker}: {data_dir}")
                commodity_blocks[ticker] = {'status': 'no_data_directory'}
                continue

            # Load council_history.csv ONCE per commodity (micro concern #1)
            ch_path = os.path.join(data_dir, 'council_history.csv')
            ch_df = _safe_read_csv(ch_path)
            if not ch_df.empty and 'timestamp' in ch_df.columns:
                ch_df = _parse_timestamp_column(ch_df)

            commodity_blocks[ticker] = {
                # v1.0
                'feedback_loop': _build_feedback_loop(data_dir),
                'agent_calibration': _build_agent_calibration(data_dir),
                'cognitive_layer': _build_cognitive_layer(ch_df),
                'sentinel_efficiency': _build_sentinel_efficiency(data_dir),
                'efficiency': _build_efficiency(data_dir, config),
                'risk_rails': _build_risk_rails(data_dir, config),
                # v1.1
                'decision_traces': _build_decision_traces(ch_df),
                'data_freshness': _build_data_freshness(data_dir),
                'regime_context': _build_regime_context(data_dir),
                'agent_contribution': _build_agent_contribution(ch_df),
            }

        # 4. Build account-wide sections
        portfolio = _build_portfolio(config)
        changes = _build_changes(config, active_tickers, yesterday_digest)
        rolling_trends = _build_rolling_trends(config)
        improvement_opportunities = _build_improvement_opportunities(commodity_blocks)

        # 5. Build v1.1 account-wide sections
        config_snapshot = _build_config_snapshot(config, active_tickers)
        error_telemetry = _build_error_telemetry(config, active_tickers)

        # 6. Assemble digest
        digest = {
            'schema_version': '1.1',
            'generated_at': now.isoformat(),
            'active_tickers': active_tickers,
            'commodities': commodity_blocks,
            'portfolio': portfolio,
            'changes': changes,
            'rolling_trends': rolling_trends,
            'improvement_opportunities': improvement_opportunities,
            'config_snapshot': config_snapshot,
            'error_telemetry': error_telemetry,
        }

        # 7. Add digest_id (content-hash, post-assembly)
        digest_hash = hashlib.sha256(
            json.dumps(digest, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        digest['digest_id'] = f"{now.strftime('%Y-%m-%d')}_{digest_hash}"

        # 8. Add executive_summary + system_health_score (post-assembly)
        digest['system_health_score'] = _build_system_health_score(digest)
        digest['executive_summary'] = _build_executive_summary(digest)

        # 9. Write to data/system_health_digest.json
        base_dir = config.get('data_dir', 'data')
        output_path = os.path.join(base_dir, 'system_health_digest.json')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(digest, f, indent=2, default=str)

        # 10. Archive to logs/digests/{date}.json.gz
        archive_dir = os.path.join('logs', 'digests')
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, f"{now.strftime('%Y-%m-%d')}.json.gz")
        with gzip.open(archive_path, 'wt') as f:
            json.dump(digest, f, indent=2, default=str)

        logger.info(
            f"System Health Digest generated: {digest['digest_id']} "
            f"({len(active_tickers)} commodities, "
            f"{len(improvement_opportunities)} improvement opportunities)"
        )

        return digest

    except Exception as e:
        logger.error(f"System Health Digest generation failed: {e}", exc_info=True)
        return None
