#!/usr/bin/env python3
"""Automated Error-to-GitHub-Issue Reporter.

Scans log files for ERROR/CRITICAL entries, classifies and deduplicates them,
sanitizes sensitive data, and creates GitHub issues for tracking.

Runs as a standalone cron job (every hour). Completely decoupled from
the trading orchestrator — never imports or affects trading-path code.

Usage:
    python scripts/error_reporter.py            # Normal mode
    python scripts/error_reporter.py --dry-run  # Parse only, no issues created

Exit codes:
    0 — Completed (issues may or may not have been created)
    1 — Fatal error (config missing, lock contention, etc.)
"""

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ErrorReporter")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    timestamp: str
    logger_name: str
    level: str  # "ERROR" or "CRITICAL"
    message: str
    source_file: str  # Which log file this came from


@dataclass
class ErrorSignature:
    category: str  # "ib_connection", "llm_api", etc.
    fingerprint: str  # MD5 of (category + normalized_message)
    sample_message: str  # First occurrence, sanitized
    count: int
    first_seen: str
    last_seen: str
    level: str  # Highest severity seen


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

# Category → list of compiled regex patterns
ERROR_PATTERNS: dict[str, list[re.Pattern]] = {
    "ib_connection": [
        re.compile(r"IB.*connection failed", re.IGNORECASE),
        re.compile(r"reqPositionsAsync timed out", re.IGNORECASE),
        re.compile(r"completed orders request timed out", re.IGNORECASE),
        re.compile(r"DISCONNECTED", re.IGNORECASE),
        re.compile(r"Connection pool.*exhaust", re.IGNORECASE),
        re.compile(r"IB Gateway.*not reachable", re.IGNORECASE),
    ],
    "llm_api": [
        re.compile(r"CriticalRPCError", re.IGNORECASE),
        re.compile(r"All.*providers? exhausted", re.IGNORECASE),
        re.compile(r"Rate limit slot timeout", re.IGNORECASE),
        re.compile(r"LLM.*call failed", re.IGNORECASE),
        re.compile(r"API key.*invalid", re.IGNORECASE),
    ],
    "parse_error": [
        re.compile(r"Could not parse", re.IGNORECASE),
        re.compile(r"JSONDecodeError", re.IGNORECASE),
        re.compile(r"Schema validation failed", re.IGNORECASE),
        re.compile(r"Failed to parse.*response", re.IGNORECASE),
    ],
    "file_io": [
        re.compile(r"Failed to save.*state", re.IGNORECASE),
        re.compile(r"PermissionError", re.IGNORECASE),
        re.compile(r"Error writing to", re.IGNORECASE),
        re.compile(r"FileNotFoundError", re.IGNORECASE),
    ],
    "trading_execution": [
        re.compile(r"Compliance.*failed", re.IGNORECASE),
        re.compile(r"Error closing position", re.IGNORECASE),
        re.compile(r"FLASH CRASH", re.IGNORECASE),
        re.compile(r"Drawdown PANIC", re.IGNORECASE),
        re.compile(r"Order.*rejected", re.IGNORECASE),
    ],
    "budget": [
        re.compile(r"Budget.*hit", re.IGNORECASE),
        re.compile(r"BudgetThrottledError", re.IGNORECASE),
        re.compile(r"Capital.*exceeded", re.IGNORECASE),
    ],
    "data_integrity": [
        re.compile(r"Reconciliation.*failed", re.IGNORECASE),
        re.compile(r"corrupted timestamps", re.IGNORECASE),
        re.compile(r"CSV.*corrupt", re.IGNORECASE),
        re.compile(r"State file.*corrupt", re.IGNORECASE),
    ],
}

# Patterns for transient errors that should be skipped entirely
TRANSIENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"RSS.*timed?\s*out", re.IGNORECASE),
    re.compile(r"rate limit.*fallback", re.IGNORECASE),
    re.compile(r"weather.*fetch.*retry", re.IGNORECASE),
    re.compile(r"Retrying in \d+ seconds", re.IGNORECASE),
    re.compile(r"Temporary failure.*name resolution", re.IGNORECASE),
    # IB Gateway transient connectivity (restarts, brief disconnects)
    re.compile(r"Not connected", re.IGNORECASE),
    re.compile(r"DISCONNECTED.*reconnect", re.IGNORECASE),
    re.compile(r"Connect call failed", re.IGNORECASE),
    re.compile(r"Connection refused", re.IGNORECASE),
    # LLM provider transient errors (429 quota, 529 overloaded)
    re.compile(r"RESOURCE_EXHAUSTED", re.IGNORECASE),
    re.compile(r"429.*quota exceeded", re.IGNORECASE),
    re.compile(r"overloaded_error", re.IGNORECASE),
    re.compile(r"Error code: 529", re.IGNORECASE),
    # Fallback successes (system recovered, not an issue)
    re.compile(r"FALLBACK SUCCESS", re.IGNORECASE),
]


def classify_error(message: str) -> str | None:
    """Classify an error message into a category. Returns None for transient errors."""
    for pattern in TRANSIENT_PATTERNS:
        if pattern.search(message):
            return None  # Skip transient

    for category, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(message):
                return category

    return "uncategorized"


# ---------------------------------------------------------------------------
# Fingerprint normalization
# ---------------------------------------------------------------------------

# Patterns to normalize variable parts of messages before hashing
_NORMALIZE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[\.\d]*[Z]?"), "<TS>"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE), "<UUID>"),
    (re.compile(r"\b(orderId|permId|conId|reqId|order_id)[=: ]*\d+"), r"\1=<ID>"),
    (re.compile(r"\b\d+\.\d+s\b"), "<N>s"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?"), "<HOST>"),
    (re.compile(r"PID[=: ]*\d+"), "PID=<ID>"),
]


def normalize_for_fingerprint(message: str) -> str:
    """Strip variable parts from a message before hashing."""
    result = message
    for pattern, replacement in _NORMALIZE_RULES:
        result = pattern.sub(replacement, result)
    return result


def compute_fingerprint(category: str, message: str) -> str:
    """Compute a stable fingerprint for deduplication."""
    normalized = normalize_for_fingerprint(message)
    raw = f"{category}:{normalized}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Log sanitizer
# ---------------------------------------------------------------------------

_SANITIZE_RULES: list[tuple[re.Pattern, str]] = [
    # API keys / tokens / secrets (various formats)
    (re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)[=: ]+\S+"), r"\1=<REDACTED>"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "<REDACTED>"),
    (re.compile(r"AIza[a-zA-Z0-9_-]{35}"), "<REDACTED>"),
    # IB account numbers (U1234567 or DU1234567)
    (re.compile(r"\b[DU]{0,2}U\d{5,8}\b"), "<ACCT>"),
    # Dollar amounts ($1,234.56 or 1234.56 USD)
    (re.compile(r"\$[\d,]+\.?\d*"), "<AMT>"),
    (re.compile(r"[\d,]+\.?\d*\s*USD"), "<AMT>"),
    # Position sizes (bare numbers in context of lots/contracts)
    (re.compile(r"\b\d+\s*(lots?|contracts?|shares?)\b", re.IGNORECASE), "<N> \\1"),
    # Strike prices in option context
    (re.compile(r"(?i)(strike|premium|price)[=: ]*[\d.]+"), r"\1=<PRICE>"),
    # File paths with usernames
    (re.compile(r"/home/\w+/"), "/home/<USER>/"),
    (re.compile(r"/Users/\w+/"), "/Users/<USER>/"),
    # Env var values for known sensitive vars
    (re.compile(r"(?i)(GEMINI_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|XAI_API_KEY|"
                r"PUSHOVER_USER_KEY|PUSHOVER_API_TOKEN|FLEX_TOKEN|FRED_API_KEY|"
                r"NASDAQ_API_KEY|X_BEARER_TOKEN|GITHUB_TOKEN|IB_PASSWORD)[=: ]+\S+"),
     r"\1=<REDACTED>"),
]


def sanitize_message(message: str) -> str:
    """Redact sensitive data from a log message."""
    result = message
    for pattern, replacement in _SANITIZE_RULES:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# Log file parsing
# ---------------------------------------------------------------------------

# Log format: "2026-02-16 10:30:45,123 - LoggerName - ERROR - The message"
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+-\s+(\S+)\s+-\s+(ERROR|CRITICAL)\s+-\s+(.*)$"
)

# Pattern for rotated log files (date stamp in name) — skip these
_ROTATED_LOG_RE = re.compile(r"-\d{4}-\d{2}-\d{2}T")


def discover_log_files(log_dir: str) -> list[str]:
    """Find all active (non-rotated) .log files in the given directory."""
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []
    result = []
    for f in sorted(log_path.iterdir()):
        if f.suffix == ".log" and not _ROTATED_LOG_RE.search(f.name):
            result.append(str(f))
    return result


def parse_log_file(
    filepath: str, start_offset: int = 0
) -> tuple[list[LogEntry], int]:
    """Parse ERROR/CRITICAL entries from a log file starting at byte offset.

    Returns (entries, new_offset). Handles log rotation (file shrunk).
    """
    entries: list[LogEntry] = []
    try:
        file_size = os.path.getsize(filepath)
    except OSError:
        return entries, start_offset

    # Rotation detection: file smaller than stored offset → reset
    if file_size < start_offset:
        start_offset = 0

    if file_size == start_offset:
        return entries, start_offset  # Nothing new

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            f.seek(start_offset)
            # Skip first partial line if we seeked into the middle of a line
            if start_offset > 0:
                # Check if we're at a line boundary by reading the byte before
                f.seek(start_offset - 1)
                prev_char = f.read(1)
                if prev_char != "\n":
                    # We're mid-line, skip to the next line
                    f.readline()
                # else: we're at a line boundary, no skip needed
            for line in f:
                m = _LOG_LINE_RE.match(line.rstrip())
                if m:
                    entries.append(LogEntry(
                        timestamp=m.group(1),
                        logger_name=m.group(2),
                        level=m.group(3),
                        message=m.group(4),
                        source_file=filepath,
                    ))
            new_offset = f.tell()
    except OSError as e:
        logger.warning(f"Could not read {filepath}: {e}")
        return entries, start_offset

    return entries, new_offset


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

STATE_VERSION = 1


def _default_state() -> dict:
    return {
        "version": STATE_VERSION,
        "last_run": "",
        "log_offsets": {},
        "reported_signatures": {},
        "daily_counters": {"date": "", "issues_created": 0, "critical_issues_created": 0},
        "accumulated_errors": {},
    }


def load_state(state_path: str) -> dict:
    """Load reporter state from disk. Returns default state on any failure."""
    try:
        with open(state_path, "r") as f:
            state = json.load(f)
        if state.get("version") != STATE_VERSION:
            logger.warning("State version mismatch, starting fresh")
            return _default_state()
        return state
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return _default_state()


def save_state(state: dict, state_path: str) -> None:
    """Atomically save state: temp file + fsync + os.replace."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state_dir = os.path.dirname(state_path)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=state_dir or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Deduplication and rate limiting
# ---------------------------------------------------------------------------

def is_deduped(fingerprint: str, state: dict, cooldown_hours: int) -> bool:
    """Check if this fingerprint was reported recently."""
    reported = state.get("reported_signatures", {})
    if fingerprint not in reported:
        return False
    cooldown_until = reported[fingerprint].get("cooldown_until", "")
    if not cooldown_until:
        return False
    try:
        until = datetime.fromisoformat(cooldown_until)
        return datetime.now(timezone.utc) < until
    except (ValueError, TypeError):
        return False


def check_rate_limit(state: dict, is_critical: bool, max_normal: int, max_critical: int) -> bool:
    """Returns True if we're within rate limits (OK to create issue)."""
    counters = state.get("daily_counters", {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if counters.get("date") != today:
        # New day, reset counters
        state["daily_counters"] = {"date": today, "issues_created": 0, "critical_issues_created": 0}
        counters = state["daily_counters"]

    if is_critical:
        return counters.get("critical_issues_created", 0) < max_critical
    else:
        return counters.get("issues_created", 0) < max_normal


def record_issue_created(state: dict, fingerprint: str, category: str,
                         issue_number: int, cooldown_hours: int, is_critical: bool) -> None:
    """Update state after creating an issue."""
    now = datetime.now(timezone.utc)
    cooldown_until = (now + timedelta(hours=cooldown_hours)).isoformat()

    state.setdefault("reported_signatures", {})[fingerprint] = {
        "category": category,
        "reported_at": now.isoformat(),
        "issue_number": issue_number,
        "cooldown_until": cooldown_until,
    }

    counters = state.setdefault("daily_counters", {})
    today = now.strftime("%Y-%m-%d")
    if counters.get("date") != today:
        counters.update({"date": today, "issues_created": 0, "critical_issues_created": 0})

    if is_critical:
        counters["critical_issues_created"] = counters.get("critical_issues_created", 0) + 1
    else:
        counters["issues_created"] = counters.get("issues_created", 0) + 1


# ---------------------------------------------------------------------------
# Issue formatting
# ---------------------------------------------------------------------------

_IMPACT_NOTES = {
    "ib_connection": "IB connection issues can prevent order execution and position monitoring.",
    "llm_api": "LLM API failures degrade decision quality — the council may produce incomplete analysis.",
    "parse_error": "Parse errors may cause agents to produce fallback (conservative) decisions.",
    "file_io": "File I/O errors can cause state loss or logging gaps.",
    "trading_execution": "Trading execution errors may indicate failed orders or compliance blocks.",
    "budget": "Budget errors indicate capital allocation limits have been reached.",
    "data_integrity": "Data integrity errors may produce incorrect reconciliation or corrupt state.",
    "uncategorized": "Investigate to determine impact on trading operations.",
}


def format_critical_issue(sig: ErrorSignature, env_name: str = "DEV", env_label: str = "env:dev") -> tuple[str, str, list[str]]:
    """Format a CRITICAL-level issue. Returns (title, body, labels)."""
    title = f"[CRITICAL] {sig.category}: {_truncate(sig.sample_message, 80)}"
    impact = _IMPACT_NOTES.get(sig.category, _IMPACT_NOTES["uncategorized"])

    body = f"""## Error Report

| Field | Value |
|-------|-------|
| **Category** | `{sig.category}` |
| **Severity** | {sig.level} |
| **Environment** | {env_name} |
| **First seen** | {sig.first_seen} |
| **Last seen** | {sig.last_seen} |
| **Occurrences** | {sig.count} |

### Sample Log Entry

```
{sig.sample_message}
```

### Impact

{impact}

> The `claude-fix` label will be added after triage — Claude will automatically attempt a fix.

---
*Automated report by `scripts/error_reporter.py` — fingerprint: `{sig.fingerprint[:12]}`*
"""
    labels = ["automated", "error-report", "priority:critical", env_label]
    return title, body, labels


def format_summary_issue(
    date_str: str,
    signatures: list[ErrorSignature],
    total_count: int,
    env_name: str = "DEV",
    env_label: str = "env:dev",
) -> tuple[str, str, list[str]]:
    """Format a daily summary issue. Returns (title, body, labels)."""
    categories = sorted(set(s.category for s in signatures))
    title = f"[Daily Summary] [{env_name}] {date_str}: {total_count} errors across {len(categories)} categories"

    # Build per-category breakdown
    sections = []
    for cat in categories:
        cat_sigs = [s for s in signatures if s.category == cat]
        cat_count = sum(s.count for s in cat_sigs)
        lines = [f"### `{cat}` — {cat_count} occurrences\n"]
        for sig in sorted(cat_sigs, key=lambda s: -s.count)[:5]:
            lines.append(f"- **{sig.count}x** `{_truncate(sig.sample_message, 100)}`")
        sections.append("\n".join(lines))

    body = f"""## Daily Error Summary — {date_str}

**Environment:** {env_name}
**Total errors:** {total_count}
**Categories:** {', '.join(f'`{c}`' for c in categories)}

{chr(10).join(sections)}

> The `claude-fix` label will be added after triage — Claude will automatically attempt a fix.

---
*Automated report by `scripts/error_reporter.py`*
"""
    labels = ["automated", "error-report", "daily-summary", env_label]
    return title, body, labels


def _truncate(s: str, maxlen: int) -> str:
    return s if len(s) <= maxlen else s[:maxlen - 3] + "..."


# ---------------------------------------------------------------------------
# GitHub issue creation
# ---------------------------------------------------------------------------

class GitHubIssueCreator:
    """Creates GitHub issues via the REST API using urllib (no extra deps)."""

    API_BASE = "https://api.github.com"

    def __init__(self, owner: str, repo: str, token: str):
        self.owner = owner
        self.repo = repo
        self.token = token

    def create_issue(self, title: str, body: str, labels: list[str]) -> int | None:
        """Create a GitHub issue. Returns the issue number, or None on failure."""
        url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/issues"
        payload = json.dumps({
            "title": title,
            "body": body,
            "labels": labels,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                issue_number = result.get("number")
                logger.info(f"Created issue #{issue_number}: {title}")
                return issue_number
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")[:500]
            logger.error(f"GitHub API error {e.code}: {body_text}")
            return None
        except Exception as e:
            logger.error(f"GitHub API request failed: {e}")
            return None

    def ensure_labels_exist(self, labels: list[str]) -> None:
        """Create labels if they don't exist (best-effort)."""
        label_colors = {
            "automated": "c5def5",
            "error-report": "f9d0c4",
            "priority:critical": "B60205",
            "daily-summary": "0075ca",
            "env:dev": "1D76DB",
            "env:prod": "D93F0B",
        }
        for label in labels:
            url = f"{self.API_BASE}/repos/{self.owner}/{self.repo}/labels"
            payload = json.dumps({
                "name": label,
                "color": label_colors.get(label, "ededed"),
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=10)
            except urllib.error.HTTPError:
                pass  # Label already exists (422) or other non-fatal error
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Lock file (prevent concurrent runs)
# ---------------------------------------------------------------------------

class LockFile:
    """Simple PID-based lockfile."""

    def __init__(self, path: str):
        self.path = path
        self._held = False

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns False if another process holds it."""
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    pid = int(f.read().strip())
                # Check if PID is still running
                try:
                    os.kill(pid, 0)
                    return False  # Process is alive, lock is held
                except OSError:
                    pass  # Stale lock, proceed to overwrite

            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w") as f:
                f.write(str(os.getpid()))
            self._held = True
            return True
        except Exception as e:
            logger.warning(f"Lock acquisition failed: {e}")
            return False

    def release(self) -> None:
        if self._held:
            try:
                os.unlink(self.path)
            except OSError:
                pass
            self._held = False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, dry_run: bool = False) -> int:
    """Main pipeline: scan logs → classify → dedup → create issues.

    Returns the number of issues created (0 in dry-run mode).
    """
    er_config = config.get("error_reporter", {})
    if not er_config.get("enabled", False):
        logger.info("Error reporter is disabled in config")
        return 0

    project_root = Path(__file__).resolve().parent.parent
    log_dir = str(project_root / er_config.get("log_directory", "logs"))
    state_path = str(project_root / "data" / "error_reporter_state.json")
    cooldown_hours = er_config.get("dedup_cooldown_hours", 24)
    max_normal = er_config.get("max_issues_per_day", 3)
    max_critical = er_config.get("max_critical_issues_per_day", 5)
    summary_threshold = er_config.get("daily_summary_threshold", 5)
    default_labels = er_config.get("default_labels", ["automated", "error-report"])

    # Environment detection (set in .env on each droplet)
    env_name = os.getenv("ENV_NAME", "DEV").upper()
    env_label = f"env:{env_name.lower()}"

    # GitHub setup
    github_owner = er_config.get("github_owner", "")
    github_repo = er_config.get("github_repo", "")
    token_env = er_config.get("github_token_env", "GITHUB_ERROR_REPORTER_TOKEN")
    github_token = os.environ.get(token_env, "")

    if not dry_run and not github_token:
        logger.warning(f"No GitHub token found in ${token_env}, exiting gracefully")
        return 0

    github: GitHubIssueCreator | None = None
    if not dry_run and github_token and github_owner and github_repo:
        github = GitHubIssueCreator(github_owner, github_repo, github_token)

    # Load state
    state = load_state(state_path)

    # Discover and parse log files
    log_files = discover_log_files(log_dir)
    if not log_files:
        logger.info(f"No log files found in {log_dir}")
        save_state(state, state_path)
        return 0

    all_entries: list[LogEntry] = []
    offsets = state.get("log_offsets", {})

    for lf in log_files:
        prev_offset = offsets.get(lf, 0)
        entries, new_offset = parse_log_file(lf, prev_offset)
        offsets[lf] = new_offset
        all_entries.extend(entries)

    state["log_offsets"] = offsets

    if not all_entries:
        logger.info("No new ERROR/CRITICAL entries found")
        save_state(state, state_path)
        return 0

    logger.info(f"Found {len(all_entries)} new error entries across {len(log_files)} log files")

    # Classify and group by fingerprint
    signatures: dict[str, ErrorSignature] = {}

    for entry in all_entries:
        category = classify_error(entry.message)
        if category is None:
            continue  # Transient, skip

        sanitized = sanitize_message(entry.message)
        fp = compute_fingerprint(category, entry.message)

        if fp in signatures:
            sig = signatures[fp]
            sig.count += 1
            sig.last_seen = entry.timestamp
            if entry.level == "CRITICAL" and sig.level != "CRITICAL":
                sig.level = "CRITICAL"
        else:
            signatures[fp] = ErrorSignature(
                category=category,
                fingerprint=fp,
                sample_message=sanitized,
                count=1,
                first_seen=entry.timestamp,
                last_seen=entry.timestamp,
                level=entry.level,
            )

    if not signatures:
        logger.info("All errors were transient, nothing to report")
        save_state(state, state_path)
        return 0

    # Process signatures
    issues_created = 0
    accumulated_for_summary: list[ErrorSignature] = []

    for fp, sig in signatures.items():
        if is_deduped(fp, state, cooldown_hours):
            logger.info(f"Skipping deduped: {sig.category} ({fp[:12]})")
            continue

        if sig.level == "CRITICAL":
            # CRITICAL → immediate issue
            if not check_rate_limit(state, is_critical=True, max_normal=max_normal, max_critical=max_critical):
                logger.warning("Critical issue rate limit reached")
                continue

            title, body, labels = format_critical_issue(sig, env_name=env_name, env_label=env_label)
            if dry_run:
                logger.info(f"[DRY RUN] Would create issue: {title}")
                logger.info(f"[DRY RUN] Labels: {labels}")
            else:
                if github:
                    github.ensure_labels_exist(labels)
                    issue_num = github.create_issue(title, body, labels)
                    if issue_num:
                        record_issue_created(state, fp, sig.category, issue_num,
                                             cooldown_hours, is_critical=True)
                        issues_created += 1
        else:
            # ERROR — accumulate for summary if count >= threshold, or
            # create individual issue if count is very high
            if sig.count >= summary_threshold:
                accumulated_for_summary.append(sig)
            # Even below threshold, track it for potential future summary
            state.setdefault("accumulated_errors", {})[fp] = asdict(sig)

    # Create daily summary if we have enough accumulated errors
    if accumulated_for_summary:
        total_count = sum(s.count for s in accumulated_for_summary)
        if not check_rate_limit(state, is_critical=False, max_normal=max_normal, max_critical=max_critical):
            logger.warning("Daily summary rate limit reached")
        else:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Check dedup for summary (use a special fingerprint)
            summary_fp = compute_fingerprint("daily_summary", today)
            if not is_deduped(summary_fp, state, cooldown_hours):
                title, body, labels = format_summary_issue(today, accumulated_for_summary, total_count, env_name=env_name, env_label=env_label)
                if dry_run:
                    logger.info(f"[DRY RUN] Would create summary issue: {title}")
                else:
                    if github:
                        github.ensure_labels_exist(labels)
                        issue_num = github.create_issue(title, body, labels)
                        if issue_num:
                            record_issue_created(state, summary_fp, "daily_summary",
                                                 issue_num, cooldown_hours, is_critical=False)
                            issues_created += 1
                            # Clear accumulated errors after creating summary
                            state["accumulated_errors"] = {}

    save_state(state, state_path)
    return issues_created


def main() -> int:
    """Entry point."""
    dry_run = "--dry-run" in sys.argv

    config = load_config()
    if not config:
        logger.error("Failed to load config")
        return 1

    er_config = config.get("error_reporter", {})
    if not er_config.get("enabled", False):
        logger.info("Error reporter disabled, exiting")
        return 0

    # Lockfile
    project_root = Path(__file__).resolve().parent.parent
    lock_path = str(project_root / "data" / ".error_reporter.lock")
    lock = LockFile(lock_path)
    if not lock.acquire():
        logger.warning("Another error_reporter instance is running, exiting")
        return 0

    try:
        issues_created = run_pipeline(config, dry_run=dry_run)

        # Send Pushover notification if issues were created
        if issues_created > 0 and er_config.get("notify_on_issue_created", True):
            try:
                from notifications import send_pushover_notification
                notif_config = config.get("notifications", {})
                send_pushover_notification(
                    config=notif_config,
                    title="Error Reporter: Issues Created",
                    message=f"Created {issues_created} GitHub issue(s) from log errors.",
                    priority=0,
                )
            except Exception as e:
                logger.warning(f"Pushover notification failed: {e}")

        if dry_run:
            logger.info("Dry run complete — no issues were created")
        elif issues_created:
            logger.info(f"Created {issues_created} issue(s)")
        else:
            logger.info("No new issues needed")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
