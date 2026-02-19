"""Tests for the automated error-to-GitHub-issue reporter."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import from scripts directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from error_reporter import (
    ErrorSignature,
    GitHubIssueCreator,
    LockFile,
    LogEntry,
    check_rate_limit,
    classify_error,
    compute_fingerprint,
    discover_log_files,
    format_critical_issue,
    format_summary_issue,
    is_deduped,
    load_state,
    normalize_for_fingerprint,
    parse_log_file,
    record_issue_created,
    run_pipeline,
    sanitize_message,
    save_state,
)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassifyError:
    def test_ib_connection_patterns(self):
        assert classify_error("IB Gateway connection failed after 3 retries") == "ib_connection"
        assert classify_error("reqPositionsAsync timed out after 10s") == "ib_connection"
        assert classify_error("Client state: DISCONNECTED") == "ib_connection"

    def test_llm_api_patterns(self):
        assert classify_error("CriticalRPCError: Gemini returned 503") == "llm_api"
        assert classify_error("All 4 providers exhausted for agronomist") == "llm_api"
        assert classify_error("Rate limit slot timeout after 30s") == "llm_api"

    def test_parse_error_patterns(self):
        assert classify_error("Could not parse Devil's Advocate response") == "parse_error"
        assert classify_error("JSONDecodeError in council output") == "parse_error"
        assert classify_error("Schema validation failed for signal") == "parse_error"

    def test_file_io_patterns(self):
        assert classify_error("Failed to save drawdown state: disk full") == "file_io"
        assert classify_error("PermissionError: [Errno 13] Permission denied") == "file_io"
        assert classify_error("Error writing to council_history.csv") == "file_io"

    def test_trading_execution_patterns(self):
        assert classify_error("Compliance check failed: VaR exceeded") == "trading_execution"
        assert classify_error("Error closing position for KC Mar 2026") == "trading_execution"
        assert classify_error("FLASH CRASH detected: price moved 5%") == "trading_execution"
        assert classify_error("Drawdown PANIC: -12% unrealized") == "trading_execution"

    def test_budget_patterns(self):
        assert classify_error("Budget limit hit for weekly allocation") == "budget"
        assert classify_error("BudgetThrottledError: max trades reached") == "budget"

    def test_data_integrity_patterns(self):
        assert classify_error("Reconciliation with IBKR failed: mismatch") == "data_integrity"
        assert classify_error("CSV file corrupt: unexpected EOF") == "data_integrity"

    def test_uncategorized(self):
        assert classify_error("Something went terribly wrong") == "uncategorized"

    def test_transient_detection(self):
        """Transient errors return None (should be skipped)."""
        assert classify_error("RSS feed timed out after 5s") is None
        assert classify_error("RSS fetch timeout for reuters") is None
        assert classify_error("rate limit exceeded, using fallback provider") is None
        assert classify_error("weather fetch retry #2 of 3") is None
        assert classify_error("Retrying in 5 seconds") is None

    def test_transient_ib_noise(self):
        """IB operational noise should be skipped entirely."""
        assert classify_error("completed orders request timed out") is None
        assert classify_error("API connection failed: TimeoutError()") is None
        assert classify_error("client id 28 already in use? Retry") is None

    def test_transient_llm_noise(self):
        """LLM provider transient errors should be skipped."""
        assert classify_error("503 UNAVAILABLE. This model is currently experiencing high demand") is None
        assert classify_error("currently experiencing high demand. Please try again") is None
        assert classify_error("Gemini timed out after 60.0s") is None
        assert classify_error("usage limits reached for this billing period") is None

    def test_transient_operational_noise(self):
        """Operational errors handled by circuit breakers should be skipped."""
        assert classify_error("EMERGENCY_LOCK acquisition timed out (300s) for MacroContagionSentinel") is None
        assert classify_error("Drawdown guard check failed (fail-closed): timeout") is None
        assert classify_error("CIRCUIT BREAKER: gemini tripped for 1.0h") is None

    def test_case_insensitivity(self):
        assert classify_error("ib gateway CONNECTION FAILED") == "ib_connection"
        assert classify_error("jsondecodeError in output") == "parse_error"


# ---------------------------------------------------------------------------
# Sanitization tests
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_api_keys(self):
        msg = "api_key=sk-abc123xyz789foo token=AIzaSyD12345678901234567890"
        result = sanitize_message(msg)
        assert "sk-abc123xyz789" not in result
        assert "AIzaSyD123456789" not in result
        assert "<REDACTED>" in result

    def test_ib_account_numbers(self):
        msg = "Account U1234567 has position in KC"
        result = sanitize_message(msg)
        assert "U1234567" not in result
        assert "<ACCT>" in result

    def test_du_account_numbers(self):
        msg = "Paper account DU9876543 disconnected"
        result = sanitize_message(msg)
        assert "DU9876543" not in result
        assert "<ACCT>" in result

    def test_dollar_amounts(self):
        msg = "P&L is $1,234.56 and margin is $50,000"
        result = sanitize_message(msg)
        assert "$1,234.56" not in result
        assert "$50,000" not in result
        assert "<AMT>" in result

    def test_file_paths(self):
        msg = "Error in /home/rodrigo/real_options/data/state.json"
        result = sanitize_message(msg)
        assert "rodrigo" not in result
        assert "/home/<USER>/" in result

    def test_env_var_values(self):
        msg = "OPENAI_API_KEY=sk-realkey123 was loaded"
        result = sanitize_message(msg)
        assert "sk-realkey123" not in result
        assert "OPENAI_API_KEY=<REDACTED>" in result

    def test_strike_prices(self):
        msg = "Option with strike=285.50 expired"
        result = sanitize_message(msg)
        assert "285.50" not in result
        assert "strike=<PRICE>" in result

    def test_position_sizes(self):
        msg = "Holding 5 contracts of KC"
        result = sanitize_message(msg)
        assert "<N> contracts" in result

    def test_no_false_positives(self):
        """Normal messages should pass through mostly unchanged."""
        msg = "Council vote complete: BULLISH consensus"
        result = sanitize_message(msg)
        assert "Council vote complete" in result
        assert "BULLISH consensus" in result


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_stability_across_timestamps(self):
        """Same error with different timestamps produces same fingerprint."""
        msg1 = "2026-02-16T10:30:00 IB connection failed for orderId=123"
        msg2 = "2026-02-16T11:45:00 IB connection failed for orderId=456"
        fp1 = compute_fingerprint("ib_connection", msg1)
        fp2 = compute_fingerprint("ib_connection", msg2)
        assert fp1 == fp2

    def test_stability_across_ips(self):
        msg1 = "Connection to 192.168.1.1:7497 refused"
        msg2 = "Connection to 10.0.0.5:7497 refused"
        fp1 = compute_fingerprint("ib_connection", msg1)
        fp2 = compute_fingerprint("ib_connection", msg2)
        assert fp1 == fp2

    def test_different_errors_different_fingerprints(self):
        fp1 = compute_fingerprint("ib_connection", "IB connection failed")
        fp2 = compute_fingerprint("llm_api", "CriticalRPCError from Gemini")
        assert fp1 != fp2

    def test_normalization_uuids(self):
        result = normalize_for_fingerprint("trace=550e8400-e29b-41d4-a716-446655440000 failed")
        assert "<UUID>" in result
        assert "550e8400" not in result

    def test_normalization_pids(self):
        result = normalize_for_fingerprint("PID=12345 crashed")
        assert "PID=<ID>" in result

    def test_normalization_durations(self):
        result = normalize_for_fingerprint("took 45.2s to respond")
        assert "<N>s" in result


# ---------------------------------------------------------------------------
# Log parsing tests
# ---------------------------------------------------------------------------

class TestParseLogFile:
    def test_normal_parsing(self, tmp_path):
        log_content = (
            "2026-02-16 10:00:00,123 - Orchestrator - INFO - Starting cycle\n"
            "2026-02-16 10:00:01,456 - Orchestrator - ERROR - Something failed\n"
            "2026-02-16 10:00:02,789 - Sentinel - CRITICAL - Flash crash detected\n"
            "2026-02-16 10:00:03,012 - Orchestrator - INFO - Cycle complete\n"
        )
        log_file = tmp_path / "test.log"
        log_file.write_text(log_content)

        entries, offset = parse_log_file(str(log_file))
        assert len(entries) == 2
        assert entries[0].level == "ERROR"
        assert entries[0].logger_name == "Orchestrator"
        assert entries[1].level == "CRITICAL"
        assert entries[1].logger_name == "Sentinel"
        assert offset == len(log_content)

    def test_offset_tracking(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-02-16 10:00:00,123 - A - ERROR - First error\n"
        )
        entries1, offset1 = parse_log_file(str(log_file))
        assert len(entries1) == 1

        # Append more content
        with open(log_file, "a") as f:
            f.write("2026-02-16 10:01:00,123 - B - ERROR - Second error\n")

        entries2, offset2 = parse_log_file(str(log_file), offset1)
        assert len(entries2) == 1
        assert entries2[0].message == "Second error"
        assert offset2 > offset1

    def test_rotation_detection(self, tmp_path):
        """If file is smaller than stored offset, reset to 0."""
        log_file = tmp_path / "test.log"
        log_file.write_text(
            "2026-02-16 10:00:00,123 - A - ERROR - After rotation\n"
        )
        # Pretend we had a much larger offset before rotation
        entries, offset = parse_log_file(str(log_file), start_offset=999999)
        assert len(entries) == 1
        assert entries[0].message == "After rotation"

    def test_partial_line_skip(self, tmp_path):
        """When seeking into middle of file, skip the first (partial) line."""
        log_file = tmp_path / "test.log"
        content = (
            "2026-02-16 10:00:00,123 - A - ERROR - First line\n"
            "2026-02-16 10:00:01,456 - B - ERROR - Second line\n"
        )
        log_file.write_text(content)
        # Seek to middle of first line
        entries, _ = parse_log_file(str(log_file), start_offset=10)
        assert len(entries) == 1
        assert entries[0].message == "Second line"

    def test_empty_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("")
        entries, offset = parse_log_file(str(log_file))
        assert len(entries) == 0
        assert offset == 0

    def test_missing_file(self):
        entries, offset = parse_log_file("/nonexistent/path.log")
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# Log discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverLogFiles:
    def test_discovers_log_files(self, tmp_path):
        (tmp_path / "orchestrator.log").write_text("content")
        (tmp_path / "sentinels.log").write_text("content")
        (tmp_path / "dashboard.log").write_text("content")
        files = discover_log_files(str(tmp_path))
        assert len(files) == 3

    def test_skips_rotated_logs(self, tmp_path):
        (tmp_path / "orchestrator.log").write_text("current")
        (tmp_path / "orchestrator-2026-02-15T12:34:56.log").write_text("old")
        files = discover_log_files(str(tmp_path))
        assert len(files) == 1
        assert "orchestrator.log" in files[0]

    def test_skips_non_log_files(self, tmp_path):
        (tmp_path / "orchestrator.log").write_text("content")
        (tmp_path / "notes.txt").write_text("content")
        (tmp_path / "data.json").write_text("content")
        files = discover_log_files(str(tmp_path))
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = discover_log_files(str(tmp_path))
        assert len(files) == 0

    def test_nonexistent_directory(self):
        files = discover_log_files("/nonexistent/dir")
        assert len(files) == 0


# ---------------------------------------------------------------------------
# Deduplication and rate limiting tests
# ---------------------------------------------------------------------------

class TestDedup:
    def test_new_fingerprint_not_deduped(self):
        state = {"reported_signatures": {}}
        assert not is_deduped("abc123", state, cooldown_hours=24)

    def test_recent_fingerprint_deduped(self):
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=12)).isoformat()
        state = {
            "reported_signatures": {
                "abc123": {"cooldown_until": future}
            }
        }
        assert is_deduped("abc123", state, cooldown_hours=24)

    def test_expired_fingerprint_not_deduped(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        state = {
            "reported_signatures": {
                "abc123": {"cooldown_until": past}
            }
        }
        assert not is_deduped("abc123", state, cooldown_hours=24)


class TestRateLimiting:
    def test_within_normal_limit(self):
        state = {"daily_counters": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "issues_created": 2,
            "critical_issues_created": 0,
        }}
        assert check_rate_limit(state, is_critical=False, max_normal=3, max_critical=5)

    def test_at_normal_limit(self):
        state = {"daily_counters": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "issues_created": 3,
            "critical_issues_created": 0,
        }}
        assert not check_rate_limit(state, is_critical=False, max_normal=3, max_critical=5)

    def test_critical_separate_limit(self):
        state = {"daily_counters": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "issues_created": 3,
            "critical_issues_created": 4,
        }}
        # Normal limit exceeded, but critical still has room
        assert check_rate_limit(state, is_critical=True, max_normal=3, max_critical=5)

    def test_new_day_resets(self):
        state = {"daily_counters": {
            "date": "2020-01-01",  # Old date
            "issues_created": 999,
            "critical_issues_created": 999,
        }}
        assert check_rate_limit(state, is_critical=False, max_normal=3, max_critical=5)
        # Counters should have been reset
        assert state["daily_counters"]["issues_created"] == 0


# ---------------------------------------------------------------------------
# Issue formatting tests
# ---------------------------------------------------------------------------

class TestFormatIssues:
    def test_format_critical_issue(self):
        sig = ErrorSignature(
            category="ib_connection",
            fingerprint="abc123def456",
            sample_message="IB connection failed after 3 retries",
            count=5,
            first_seen="2026-02-16 10:00:00,123",
            last_seen="2026-02-16 10:30:00,456",
            level="CRITICAL",
        )
        title, body, labels = format_critical_issue(sig)
        assert "[CRITICAL]" in title
        assert "ib_connection" in title
        assert "ib_connection" in body
        assert "CRITICAL" in body
        assert "5" in body  # count
        assert "abc123def456"[:12] in body
        assert "priority:critical" in labels
        assert "automated" in labels

    def test_format_summary_issue(self):
        sigs = [
            ErrorSignature("ib_connection", "fp1", "conn failed", 10,
                           "10:00", "10:30", "ERROR"),
            ErrorSignature("llm_api", "fp2", "API timeout", 5,
                           "10:00", "10:15", "ERROR"),
        ]
        title, body, labels = format_summary_issue("2026-02-16", sigs, 15)
        assert "[Daily Summary]" in title
        assert "2026-02-16" in title
        assert "15 errors" in title
        assert "2 categories" in title
        assert "ib_connection" in body
        assert "llm_api" in body
        assert "daily-summary" in labels


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        original = {
            "version": 1,
            "last_run": "",
            "log_offsets": {"logs/orchestrator.log": 12345},
            "reported_signatures": {
                "fp1": {"category": "ib_connection", "cooldown_until": "2026-02-17T00:00:00"}
            },
            "daily_counters": {"date": "2026-02-16", "issues_created": 2, "critical_issues_created": 1},
            "accumulated_errors": {},
        }
        save_state(original, state_path)
        loaded = load_state(state_path)

        assert loaded["log_offsets"] == original["log_offsets"]
        assert loaded["reported_signatures"] == original["reported_signatures"]
        assert loaded["daily_counters"] == original["daily_counters"]
        assert loaded["last_run"] != ""  # save_state sets this

    def test_load_missing_file(self, tmp_path):
        state = load_state(str(tmp_path / "nonexistent.json"))
        assert state["version"] == 1
        assert state["log_offsets"] == {}

    def test_load_corrupt_file(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json{{{")
        state = load_state(str(state_path))
        assert state["version"] == 1  # Falls back to default

    def test_atomic_write(self, tmp_path):
        """State file should be written atomically (no partial writes)."""
        state_path = str(tmp_path / "state.json")
        state = {"version": 1, "last_run": "", "log_offsets": {},
                 "reported_signatures": {}, "daily_counters": {},
                 "accumulated_errors": {}}
        save_state(state, state_path)
        # File should exist and be valid JSON
        with open(state_path) as f:
            loaded = json.load(f)
        assert loaded["version"] == 1


# ---------------------------------------------------------------------------
# GitHub API mock tests
# ---------------------------------------------------------------------------

class TestGitHubIssueCreator:
    @patch("error_reporter.urllib.request.urlopen")
    def test_create_issue_success(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"number": 42}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        creator = GitHubIssueCreator("owner", "repo", "token123")
        result = creator.create_issue("Test", "Body", ["bug"])
        assert result == 42

        # Verify the request was made correctly
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "repos/owner/repo/issues" in req.full_url
        assert req.get_header("Authorization") == "Bearer token123"
        payload = json.loads(req.data)
        assert payload["title"] == "Test"
        assert payload["labels"] == ["bug"]

    @patch("error_reporter.urllib.request.urlopen")
    def test_create_issue_api_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="", code=422, msg="Unprocessable", hdrs={}, fp=MagicMock(read=lambda: b"error")
        )
        creator = GitHubIssueCreator("owner", "repo", "token123")
        result = creator.create_issue("Test", "Body", [])
        assert result is None


# ---------------------------------------------------------------------------
# Lock file tests
# ---------------------------------------------------------------------------

class TestLockFile:
    def test_acquire_and_release(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        lock = LockFile(lock_path)
        assert lock.acquire()
        assert os.path.exists(lock_path)
        lock.release()
        assert not os.path.exists(lock_path)

    def test_stale_lock_overwritten(self, tmp_path):
        lock_path = str(tmp_path / "test.lock")
        # Write a PID that doesn't exist
        with open(lock_path, "w") as f:
            f.write("999999999")
        lock = LockFile(lock_path)
        assert lock.acquire()  # Should succeed — stale lock
        lock.release()


# ---------------------------------------------------------------------------
# Integration: run_pipeline with mocked GitHub
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_pipeline_disabled(self):
        config = {"error_reporter": {"enabled": False}}
        assert run_pipeline(config) == 0

    def test_pipeline_no_logs(self, tmp_path):
        config = {
            "error_reporter": {
                "enabled": True,
                "log_directory": str(tmp_path / "empty_logs"),
                "github_owner": "test",
                "github_repo": "test",
                "github_token_env": "FAKE_TOKEN",
            }
        }
        os.makedirs(tmp_path / "empty_logs", exist_ok=True)
        assert run_pipeline(config, dry_run=True) == 0

    def test_pipeline_dry_run(self, tmp_path, monkeypatch):
        # Create a log file with errors
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "test.log"
        log_file.write_text(
            "2026-02-16 10:00:00,123 - Test - CRITICAL - IB connection failed badly\n"
            "2026-02-16 10:00:01,456 - Test - ERROR - Something failed\n"
        )

        # Create data dir for state
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "error_reporter": {
                "enabled": True,
                "log_directory": str(log_dir),
                "github_owner": "test",
                "github_repo": "test",
                "github_token_env": "FAKE_TOKEN",
                "dedup_cooldown_hours": 24,
                "max_issues_per_day": 3,
                "max_critical_issues_per_day": 5,
                "daily_summary_threshold": 5,
            }
        }

        # Patch __file__ in the error_reporter module so project_root resolves to tmp_path
        import error_reporter
        fake_script = tmp_path / "scripts" / "error_reporter.py"
        fake_script.parent.mkdir(parents=True, exist_ok=True)
        fake_script.touch()
        monkeypatch.setattr(error_reporter, "__file__", str(fake_script))

        result = run_pipeline(config, dry_run=True)
        # Dry run returns 0 issues created
        assert result == 0


class TestRecordIssueCreated:
    def test_records_normal_issue(self):
        state = {"reported_signatures": {}, "daily_counters": {}}
        record_issue_created(state, "fp1", "ib_connection", 42, 24, is_critical=False)
        assert "fp1" in state["reported_signatures"]
        assert state["reported_signatures"]["fp1"]["issue_number"] == 42
        assert state["daily_counters"]["issues_created"] == 1
        assert state["daily_counters"]["critical_issues_created"] == 0

    def test_records_critical_issue(self):
        state = {"reported_signatures": {}, "daily_counters": {}}
        record_issue_created(state, "fp1", "trading_execution", 99, 24, is_critical=True)
        assert state["daily_counters"]["critical_issues_created"] == 1
        assert state["daily_counters"]["issues_created"] == 0


import urllib.error
