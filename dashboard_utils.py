"""
Shared utilities for the Coffee Bot Real Options dashboard.
Contains data loading, caching, and decision grading logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import random
import json
import re
import time
import logging
import pytz
from datetime import datetime, timedelta, timezone
import yfinance as yf
import sys
import asyncio
import warnings

logger = logging.getLogger(__name__)

# Suppress "coroutine was never awaited" warnings from Streamlit's execution model
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

# --- FIX: Ensure Event Loop Exists BEFORE importing IB ---
# Streamlit runs scripts in a separate thread which may not have an event loop.
# ib_insync (via eventkit) requires one at import time.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB

# Path setup for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from performance_analyzer import get_trade_ledger_df
# Decision signals: lightweight summary of Council decisions
from trading_bot.decision_signals import set_data_dir as set_signals_dir
from config_loader import load_config
from trading_bot.utils import configure_market_data_type
from trading_bot.timestamps import parse_ts_column

# Set module-level data paths for the active commodity (dashboard doesn't go through orchestrator init)
_dashboard_ticker = os.environ.get("COMMODITY_TICKER", "KC")
_dashboard_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', _dashboard_ticker)
set_signals_dir(_dashboard_data_dir)

from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
set_brier_bridge_dir(_dashboard_data_dir)

# === MULTI-COMMODITY PATH RESOLUTION ===
def _resolve_data_path(filename: str) -> str:
    """Resolve a data file path with multi-commodity fallback.

    3-tier fallback:
    1. data/{COMMODITY_TICKER}/{filename} — per-commodity isolated path
    2. data/{filename} — legacy pre-migration path
    3. Returns data/{COMMODITY_TICKER}/{filename} — targeted path for file creation
    """
    ticker = os.environ.get("COMMODITY_TICKER", "KC")
    commodity_path = os.path.join("data", ticker, filename)
    legacy_path = os.path.join("data", filename)

    if os.path.exists(commodity_path):
        return commodity_path
    if os.path.exists(legacy_path):
        return legacy_path
    return commodity_path  # Default to commodity path for creation


def _resolve_data_path_for(filename: str, ticker: str) -> str:
    """Resolve a data file path for a specific commodity ticker.

    Same 3-tier fallback as _resolve_data_path but with explicit ticker
    (no env var dependency), enabling commodity switching at runtime.
    """
    commodity_path = os.path.join("data", ticker, filename)
    legacy_path = os.path.join("data", filename)
    if os.path.exists(commodity_path):
        return commodity_path
    if os.path.exists(legacy_path):
        return legacy_path
    return commodity_path


# === CONFIGURATION ===
# E4 FIX: Dynamic starting capital handled in get_starting_capital function
STATE_FILE_PATH = _resolve_data_path('state.json')
_ticker_for_log = os.environ.get("COMMODITY_TICKER", "KC").lower()
ORCHESTRATOR_LOG_PATH = f'logs/orchestrator_{_ticker_for_log}.log'
COUNCIL_HISTORY_PATH = _resolve_data_path('council_history.csv')
DAILY_EQUITY_PATH = _resolve_data_path('daily_equity.csv')


# Dynamic path getters (re-evaluate each call for in-app commodity switching)
def _get_state_file_path():
    return _resolve_data_path('state.json')

def _get_orchestrator_log_path():
    ticker = os.environ.get("COMMODITY_TICKER", "KC").lower()
    return f'logs/orchestrator_{ticker}.log'

def _get_council_history_path():
    return _resolve_data_path('council_history.csv')

def _get_daily_equity_path():
    return _resolve_data_path('daily_equity.csv')


# === TRADE JOURNAL ===

@st.cache_data(ttl=120)
def load_trade_journal(ticker: str = None) -> list:
    """Load trade journal entries for a commodity."""
    ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
    journal_path = _resolve_data_path_for('trade_journal.json', ticker)
    if not os.path.exists(journal_path):
        return []
    try:
        with open(journal_path, 'r') as f:
            return json.load(f)
    except Exception:
        return []


def find_journal_entry(journal: list, contract: str, pnl: float) -> dict | None:
    """Find the journal entry matching a council decision by contract and P&L.

    Returns the best match or None. Matches on contract name and approximate
    P&L (within $1 tolerance), picking the closest PnL if multiple match.
    """
    if not journal or not contract or pnl is None:
        return None
    try:
        pnl = float(pnl)
    except (ValueError, TypeError):
        return None

    candidates = [
        e for e in journal
        if e.get('contract') == contract and abs(e.get('pnl', 0) - pnl) < 1.0
    ]
    if not candidates:
        return None
    # Pick closest PnL match
    return min(candidates, key=lambda e: abs(e.get('pnl', 0) - pnl))


# === DATA LOADING FUNCTIONS ===

def get_starting_capital(config: dict) -> float:
    """Get starting capital from config, not hardcoded."""
    from config import get_active_profile
    profile = get_active_profile(config)

    # Priority: Env Var > Config > Profile > Default
    env_cap = os.getenv("INITIAL_CAPITAL")
    if env_cap:
        return float(env_cap)

    return config.get('account', {}).get(
        'starting_capital',
        profile.default_starting_capital if hasattr(profile, 'default_starting_capital') else 50000.0
    )

@st.cache_data
def get_config():
    """Loads and caches the application configuration."""
    config = load_config()
    if config is None:
        st.error("Fatal: Could not load config.json.")
        return {}
    # COMMODITY_TICKER override is now handled in load_config() itself
    return config


def get_commodity_profile(config: dict = None) -> dict:
    """
    Returns the commodity profile for the active symbol.
    Derives defaults from the CommodityProfile dataclass.
    """
    if config is None:
        config = get_config()

    symbol = config.get('symbol', 'KC')
    profiles = config.get('commodity_profile', {})

    # If config.json has a commodity_profile section, use it
    if symbol in profiles:
        return profiles[symbol]

    # Derive defaults from the CommodityProfile dataclass
    from config import get_commodity_profile as get_cp
    try:
        cp = get_cp(symbol)
        return {
            'name': cp.name,
            'price_unit': cp.contract.unit if hasattr(cp.contract, 'unit') else 'USD',
            'stop_parse_range': list(cp.stop_parse_range) if hasattr(cp, 'stop_parse_range') else [0, 100000],
            'typical_price_range': list(cp.typical_price_range) if hasattr(cp, 'typical_price_range') else [0, 100000],
        }
    except Exception:
        return {
            'name': symbol,
            'price_unit': 'USD',
            'stop_parse_range': [0, 100000],
            'typical_price_range': [0, 100000],
        }


@st.cache_data(ttl=60)
def load_trade_data(ticker: str = None):
    """Loads and caches the consolidated trade ledger."""
    try:
        ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
        data_dir = os.path.join("data", ticker)
        df = get_trade_ledger_df(data_dir)
        if not df.empty:
            df['timestamp'] = parse_ts_column(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to load trade_ledger.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _load_legacy_council_history(data_dir: str) -> pd.DataFrame:
    """Loads and caches legacy council history files (long TTL)."""
    legacy_dfs = []
    if os.path.exists(data_dir):
        legacy_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith('council_history_legacy') and f.endswith('.csv')
        ]
        for legacy_file in legacy_files:
            try:
                legacy_df = pd.read_csv(legacy_file, on_bad_lines='warn')
                if not legacy_df.empty:
                    legacy_dfs.append(legacy_df)
            except Exception as e:
                logger.warning(f"Could not load legacy file {legacy_file}: {e}")

    if legacy_dfs:
        combined = pd.concat(legacy_dfs, ignore_index=True)
        # OPTIMIZATION: Parse timestamps once on load, so it's cached with datetime objects
        if 'timestamp' in combined.columns:
            combined['timestamp'] = parse_ts_column(combined['timestamp'])
        return combined
    return pd.DataFrame()


@st.cache_data(ttl=60)
def load_council_history(ticker: str = None):
    """
    Loads council_history.csv for decision analysis.
    Also loads any archived legacy files and concatenates them.
    """
    try:
        ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
        council_path = _resolve_data_path_for('council_history.csv', ticker)
        dataframes = []
        data_dir = os.path.dirname(council_path)

        # Load main file
        if os.path.exists(council_path):
            df = pd.read_csv(council_path, on_bad_lines='warn')
            if not df.empty:
                # OPTIMIZATION: Parse timestamps immediately for live data
                if 'timestamp' in df.columns:
                    df['timestamp'] = parse_ts_column(df['timestamp'])
                dataframes.append(df)

        # Load any legacy/archived files (cached separately)
        legacy_df = _load_legacy_council_history(data_dir)
        if not legacy_df.empty:
            dataframes.append(legacy_df)

        if not dataframes:
            return pd.DataFrame()

        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Remove duplicates (same timestamp + contract)
        if 'timestamp' in combined_df.columns and 'contract' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'contract'], keep='last')

        return combined_df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    except Exception as e:
        st.error(f"Failed to load council_history.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_equity_data(ticker: str = None):
    """Loads daily_equity.csv for equity curve visualization."""
    try:
        ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
        equity_path = _resolve_data_path_for('daily_equity.csv', ticker)
        if os.path.exists(equity_path):
            df = pd.read_csv(equity_path)
            if not df.empty:
                df['timestamp'] = parse_ts_column(df['timestamp'])
                df = df.sort_values('timestamp')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load daily_equity.csv: {e}")
        return pd.DataFrame()


def tail_file(filepath: str, n_lines: int = 50, block_size: int = 4096) -> list[str]:
    """
    Reads the last n_lines of a file efficiently without reading the entire file.

    This function seeks to the end of the file and reads backwards in blocks
    until it finds enough newlines, making it much faster for large log files
    than readlines().
    """
    if not os.path.exists(filepath):
        return [f"Error: File {filepath} not found"]

    try:
        with open(filepath, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                return []

            # Read backwards
            pos = file_size
            lines_found = 0

            while pos > 0 and lines_found < n_lines + 1:
                step = min(block_size, pos)
                pos -= step
                f.seek(pos)
                block = f.read(step)
                lines_found += block.count(b'\n')

            # seek to the calculated position
            f.seek(pos)
            content = f.read()

            # Decode safely (ignore partial multibyte chars at start of block)
            text = content.decode('utf-8', errors='replace')
            lines = text.splitlines(keepends=True)

            return lines[-n_lines:]

    except Exception as e:
        return [f"Error reading file: {e}"]


@st.cache_data(ttl=60)
def load_log_data():
    """Finds all relevant log files and returns last 50 lines of each."""
    try:
        list_of_logs = glob.glob('logs/*.log')
        if not list_of_logs:
            return {}

        logs_content = {}
        for log_path in list_of_logs:
            filename = os.path.basename(log_path)
            # Only skip RotatingFileHandler backups (e.g., orchestrator.log.1, .log.2)
            if re.match(r'.*\.log\.\d+$', filename):
                continue
            name = filename.split('.')[0].capitalize()
            logs_content[name] = tail_file(log_path, n_lines=50)

        return logs_content
    except Exception:
        return {}


# === ASYNC HELPERS ===

def _ensure_event_loop():
    """Ensures there is a valid asyncio event loop in the current thread."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


# === SHARED STATE CACHING ===

@st.cache_data(ttl=2)
def _load_shared_state() -> dict:
    """
    Cached access to state.json to prevent multiple disk reads per rerun.
    Short TTL (2s) ensures freshness while batching reads within a single script run.
    """
    try:
        from trading_bot.state_manager import StateManager
        return StateManager._load_raw_sync()
    except Exception:
        # Fallback if import fails or StateManager errors
        state_path = _get_state_file_path()
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}


# === SYSTEM HEALTH FUNCTIONS ===

def get_system_heartbeat():
    """
    Checks file modification timestamps to determine system health.
    Returns dict with status info for Orchestrator and State.
    """
    heartbeat = {
        'orchestrator_status': 'OFFLINE',
        'orchestrator_last_pulse': None,
        'state_status': 'OFFLINE',
        'state_last_pulse': None,
        'alert_threshold_minutes': 10
    }

    # Check Orchestrator log
    log_path = _get_orchestrator_log_path()
    if os.path.exists(log_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        heartbeat['orchestrator_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['orchestrator_status'] = 'ONLINE' if minutes_since < heartbeat['alert_threshold_minutes'] else 'STALE'

    # Check State file
    state_path = _get_state_file_path()
    if os.path.exists(state_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(state_path))
        heartbeat['state_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['state_status'] = 'ONLINE' if minutes_since < heartbeat['alert_threshold_minutes'] else 'STALE'

    return heartbeat


# === TASK SCHEDULE TRACKER ===

ACTIVE_SCHEDULE_PATH = _resolve_data_path('active_schedule.json')
TASK_COMPLETIONS_PATH = _resolve_data_path('task_completions.json')

def _get_active_schedule_path():
    return _resolve_data_path('active_schedule.json')

def _get_task_completions_path():
    return _resolve_data_path('task_completions.json')

# Legacy labels: fallback for old active_schedule.json files without 'label' field.
# New schedules include per-instance labels (e.g., "Signal: Early Session (04:00 ET)")
# directly in the JSON via orchestrator's ScheduledTask.
_LEGACY_TASK_LABELS = {
    'start_monitoring':               '🟢 Start Position Monitoring',
    'process_deferred_triggers':      '📬 Process Deferred Triggers',
    'run_position_audit_cycle':       '🔍 Position Audit',
    'guarded_generate_orders':        '🧠 Generate & Execute Orders',
    'close_stale_positions':          '🔒 Close Stale Positions',
    'close_stale_positions_fallback': '🔒 Close Stale (Fallback)',
    'emergency_hard_close':           '🚨 Emergency Hard Close',
    'cancel_and_stop_monitoring':     '🔴 End-of-Day Shutdown',
    'log_equity_snapshot':            '📊 Log Equity Snapshot',
    'run_brier_reconciliation':       '🎯 Brier Reconciliation',
    'reconcile_and_analyze':          '📈 Reconcile & Analyze',
}


@st.cache_data(ttl=30)
def load_task_schedule_status() -> dict:
    """
    Loads the active schedule and task completions, then computes
    per-task status for today.

    Returns:
        dict with keys:
        - 'tasks': list of dicts, each with:
            - 'time_et': scheduled time string (HH:MM)
            - 'name': function name
            - 'label': human-readable label
            - 'status': one of 'completed', 'overdue', 'skipped', 'upcoming'
            - 'completed_at': ISO timestamp or None
        - 'summary': dict with 'total', 'completed', 'upcoming', 'overdue', 'skipped'
        - 'schedule_env': environment name
        - 'available': bool — whether data files exist
    """
    result = {
        'tasks': [],
        'summary': {'total': 0, 'completed': 0, 'upcoming': 0, 'overdue': 0, 'skipped': 0},
        'schedule_env': 'Unknown',
        'available': False,
    }

    # Load active schedule
    schedule_path = _get_active_schedule_path()
    if not os.path.exists(schedule_path):
        return result

    try:
        with open(schedule_path, 'r') as f:
            schedule_data = json.load(f)
    except Exception:
        return result

    # === WEEKEND / HOLIDAY AWARENESS ===
    # The orchestrator writes the full weekday schedule to active_schedule.json
    # regardless of day. On non-trading days, all tasks should show as 'inactive'
    # rather than 'overdue' — matching the orchestrator's own weekend skip logic.
    ny_tz_check = pytz.timezone('America/New_York')
    now_ny_check = datetime.now(timezone.utc).astimezone(ny_tz_check)
    _is_trading_day = now_ny_check.weekday() < 5  # Mon-Fri

    if _is_trading_day:
        # Also check US holidays (consistent with trading_bot/utils.py is_trading_day)
        try:
            import holidays as holidays_lib
            us_holidays = holidays_lib.US(years=now_ny_check.year, observed=True)
            if now_ny_check.date() in us_holidays:
                _is_trading_day = False
        except ImportError:
            pass  # holidays lib not available — weekday check is sufficient

    if not _is_trading_day:
        # Non-trading day: return all tasks as 'inactive' with metadata
        inactive_tasks = []
        for task_entry in schedule_data.get('tasks', []):
            time_str = task_entry['time_et']
            name = task_entry.get('name', '')
            task_id = task_entry.get('id', name)
            label = task_entry.get('label') or _LEGACY_TASK_LABELS.get(name, name)
            inactive_tasks.append({
                'time_et': time_str,
                'name': name,
                'task_id': task_id,
                'label': label,
                'status': 'inactive',
                'completed_at': None,
            })

        # Calculate next trading day for display
        next_trading = now_ny_check + timedelta(days=1)
        while next_trading.weekday() >= 5:
            next_trading += timedelta(days=1)
        # Note: doesn't check holidays for next day — acceptable simplification

        return {
            'tasks': inactive_tasks,
            'summary': {
                'total': len(inactive_tasks),
                'completed': 0,
                'upcoming': 0,
                'overdue': 0,
                'skipped': 0,
                'inactive': len(inactive_tasks),
            },
            'schedule_env': schedule_data.get('env', 'Unknown'),
            'available': True,
            'is_trading_day': False,
            'next_trading_day': next_trading.strftime('%A, %b %d'),
        }

    # Load completions
    completions = {}
    try:
        completions_path = _get_task_completions_path()
        if os.path.exists(completions_path):
            with open(completions_path, 'r') as f:
                tracker_data = json.load(f)

            # Only use completions from today (NY timezone)
            ny_tz = pytz.timezone('America/New_York')
            today_str = datetime.now(timezone.utc).astimezone(ny_tz).strftime('%Y-%m-%d')

            if tracker_data.get('trading_date') == today_str:
                completions = tracker_data.get('completions', {})
    except Exception:
        pass

    # Current time in ET
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)
    now_minutes = now_ny.hour * 60 + now_ny.minute

    # After market close (14:00 ET), tasks that never completed are
    # "skipped" (informational) rather than "overdue" (alarming).
    # This avoids visual noise from intentionally skipped tasks like
    # guarded_generate_orders past cutoff. (Flight Director advisory)
    market_closed = now_ny.hour >= 14

    tasks = []
    completed_count = 0
    upcoming_count = 0
    overdue_count = 0
    skipped_count = 0

    for task_entry in schedule_data.get('tasks', []):
        time_str = task_entry['time_et']
        name = task_entry.get('name', '')
        task_id = task_entry.get('id', name)
        label = task_entry.get('label') or _LEGACY_TASK_LABELS.get(name, name)

        # Parse scheduled time to minutes since midnight
        parts = time_str.split(':')
        task_minutes = int(parts[0]) * 60 + int(parts[1])

        # Determine status — check both task_id and legacy name for transition period
        completed_at = completions.get(task_id) or completions.get(name)
        if completed_at:
            status = 'completed'
            completed_count += 1
        elif task_minutes > now_minutes:
            status = 'upcoming'
            upcoming_count += 1
        elif market_closed:
            # Market is closed and task never completed — end-of-day state.
            # This is informational, not alarming (e.g., order gen past cutoff).
            status = 'skipped'
            skipped_count += 1
        else:
            # Scheduled time has passed but market is still open —
            # something may be wrong.
            status = 'overdue'
            overdue_count += 1

        tasks.append({
            'time_et': time_str,
            'name': name,
            'task_id': task_id,
            'label': label,
            'status': status,
            'completed_at': completed_at,
        })

    result['tasks'] = tasks
    result['summary'] = {
        'total': len(tasks),
        'completed': completed_count,
        'upcoming': upcoming_count,
        'overdue': overdue_count,
        'skipped': skipped_count,
    }
    result['schedule_env'] = schedule_data.get('env', 'Unknown')
    result['available'] = True
    result['is_trading_day'] = True

    return result


def get_ib_connection_health() -> dict:
    """
    Returns IB connection health metrics from state.
    """
    try:
        state = _load_shared_state()

        # Check for recent heartbeats or connection events
        sensors = state.get("sensors", {})

        return {
            "sentinel_ib": sensors.get("sentinel_ib_status", {}).get("data", "UNKNOWN"),
            "micro_ib": sensors.get("micro_ib_status", {}).get("data", "UNKNOWN"),
            "last_successful_connection": sensors.get("last_ib_success", {}).get("data"),
            "reconnect_backoff": sensors.get("reconnect_backoff", {}).get("data", 0)
        }
    except Exception:
        return {
            "sentinel_ib": "UNKNOWN",
            "micro_ib": "UNKNOWN",
            "last_successful_connection": None,
            "reconnect_backoff": 0
        }


def extract_sentinel_status(report: any, sentinel_name: str) -> dict:
    """
    Safely extracts sentinel status from various report formats.
    Handles: None, str, dict, nested dict with 'data' key
    """
    result = {
        'data': None,
        'is_stale': True,
        'stale_minutes': 999,
        'sentiment': 'NEUTRAL',
        'confidence': 0.5,
        'error': None
    }

    try:
        if report is None:
            result['error'] = 'No data'
            return result

        if isinstance(report, str):
            result['data'] = report
            result['is_stale'] = 'STALE' in report.upper()
            match = re.search(r'STALE.*?(\d+)\s*min', report, re.IGNORECASE)
            if match:
                result['stale_minutes'] = int(match.group(1))
            return result

        if isinstance(report, dict):
            result['data'] = report.get('data', report)
            result['is_stale'] = report.get('is_stale', False)
            result['stale_minutes'] = report.get('stale_minutes', 0)
            result['sentiment'] = report.get('sentiment', 'NEUTRAL')
            result['confidence'] = report.get('confidence', 0.5)

            if 'STALE' in str(result['data']).upper() and not result['is_stale']:
                result['is_stale'] = True
            return result

        result['data'] = str(report)
        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def get_sentinel_status():
    """
    Reads sentinel operational health from state.json (sentinel_health namespace).

    Returns a dict of sentinel names -> status info with staleness detection.
    Commodity-agnostic: sentinel list is derived from state data, not hardcoded.
    """

    # Complete sentinel registry with display metadata
    # 'availability' helps the dashboard group sentinels logically
    SENTINEL_REGISTRY = {
        'WeatherSentinel':          {'display': 'Weather',          'availability': '24/7',          'icon': '🌦️'},
        'LogisticsSentinel':        {'display': 'Logistics',        'availability': '24/7',          'icon': '🚢'},
        'NewsSentinel':             {'display': 'News',             'availability': '24/7',          'icon': '📰'},
        'XSentimentSentinel':       {'display': 'X Sentiment',      'availability': 'Market-Adjacent','icon': '📡'},
        'PredictionMarketSentinel': {'display': 'Prediction Mkt',   'availability': '24/7',          'icon': '🎯'},
        'MacroContagionSentinel':   {'display': 'Macro Contagion',  'availability': '24/7',          'icon': '🌐'},
        'PriceSentinel':            {'display': 'Price',            'availability': 'Market Hours',  'icon': '📈'},
        'MicrostructureSentinel':   {'display': 'Microstructure',   'availability': 'Market Hours',  'icon': '🔬'},
    }

    result = {}

    try:
        state = _load_shared_state()
        health_ns = state.get('sentinel_health', {})

        for name, meta in SENTINEL_REGISTRY.items():
            entry = health_ns.get(name, {})
            data = entry.get('data', {}) if isinstance(entry, dict) else {}
            timestamp = entry.get('timestamp', 0) if isinstance(entry, dict) else 0

            status = data.get('status', 'Unknown')
            interval = data.get('interval_seconds', 0)
            error = data.get('error')
            last_check = data.get('last_check_utc')

            # Staleness detection: stale if no check within 2x expected interval
            is_stale = False
            minutes_since = None
            if timestamp > 0:
                import time
                seconds_since = time.time() - timestamp
                minutes_since = round(seconds_since / 60)
                # Stale if more than 2x the check interval has elapsed
                if interval > 0 and seconds_since > (interval * 2):
                    is_stale = True

            result[name] = {
                'status': status,
                'display_name': meta['display'],
                'availability': meta['availability'],
                'icon': meta['icon'],
                'last_check_utc': last_check,
                'minutes_since_check': minutes_since,
                'is_stale': is_stale,
                'interval_seconds': interval,
                'error': error,
            }

    except Exception:
        # Fallback: return registry with Unknown status
        for name, meta in SENTINEL_REGISTRY.items():
            result[name] = {
                'status': 'Unknown',
                'display_name': meta['display'],
                'availability': meta['availability'],
                'icon': meta['icon'],
                'last_check_utc': None,
                'minutes_since_check': None,
                'is_stale': False,
                'interval_seconds': 0,
                'error': None,
            }

    return result


@st.cache_data(ttl=10)
def load_deduplicator_metrics() -> dict:
    """Load trigger deduplication metrics with caching."""
    try:
        with open(_resolve_data_path('deduplicator_state.json'), 'r') as f:
            data = json.load(f)
            metrics = data.get('metrics', {})
            total = metrics.get('total_triggers', 0)
            processed = metrics.get('processed', 0)

            return {
                'total_triggers': total,
                'processed': processed,
                'filtered': total - processed,
                'efficiency': processed / total if total > 0 else 1.0,
            }
    except Exception:
        return {'total_triggers': 0, 'processed': 0, 'efficiency': 1.0}


# === LIVE DATA FUNCTIONS (IB Connection) ===

@st.cache_data(ttl=60)
def fetch_live_dashboard_data(_config):
    """
    Consolidated fetcher for Account Summary, Daily P&L, and Active Futures Data.
    Uses underscore prefix for config to prevent Streamlit from hashing it.
    """
    ib = IB()
    ticker = _config.get('commodity', {}).get('ticker', _config.get('symbol', 'KC'))
    data = {
        "NetLiquidation": 0.0,
        "MaintMarginReq": 0.0,
        "DailyPnL": 0.0,
        "DailyPnLPct": 0.0,
        f"{ticker}_DailyChange": 0.0,
        f"{ticker}_Price": 0.0,
        "MarketData": pd.DataFrame()
    }

    try:
        loop = _ensure_event_loop()

        # Use run_until_complete with connectAsync to ensure proper awaiting
        loop.run_until_complete(ib.connectAsync(
            _config['connection']['host'],
            _config['connection']['port'],
            clientId=random.randint(1000, 9999)
        ))
        configure_market_data_type(ib)

        # Account Summary
        summary = ib.accountSummary()
        for item in summary:
            if item.tag == "NetLiquidation":
                data["NetLiquidation"] = float(item.value)
            elif item.tag == "MaintMarginReq":
                data["MaintMarginReq"] = float(item.value)

        # PnL
        account = ib.managedAccounts()[0] if ib.managedAccounts() else None
        if account:
            ib.reqPnL(account)
            ib.sleep(1)
            pnl = ib.pnl()
            if pnl:
                data["DailyPnL"] = pnl[0].dailyPnL or 0.0
                if data["NetLiquidation"] > 0:
                    data["DailyPnLPct"] = (data["DailyPnL"] / data["NetLiquidation"]) * 100

    except Exception:
        # st.warning(f"Could not connect to IB: {e}")
        pass
    finally:
        if ib.isConnected():
            ib.disconnect()
            # FLIGHT DIRECTOR FIX: Force blocking sleep for Gateway cleanup
            # Streamlit runs in a thread, so time.sleep is safe and necessary here.
            import time
            time.sleep(3.0)

    return data


@st.cache_data(ttl=60)
def fetch_all_live_data(_config: dict) -> dict:
    """
    Single consolidated IB data fetch for ALL dashboard pages.

    FLIGHT DIRECTOR AMENDMENTS:
    1. Creates FRESH event loop for thread safety
    2. Uses RANDOM ClientID to prevent tab collisions
    3. Proper cleanup with blocking sleep for Gateway
    4. OPTIMIZATION: Uses asyncio.gather for parallel fetching (Bolt)
    """
    result = {
        'net_liquidation': 0.0,
        'unrealized_pnl': 0.0,
        'realized_pnl': 0.0,
        'daily_pnl': 0.0,
        'maint_margin': 0.0,
        'open_positions': [],
        'pending_orders': [],
        'portfolio_items': [],
        'account_summary': {},
        'connection_status': 'DISCONNECTED',
        'last_fetch_time': datetime.now(timezone.utc),
        'error': None
    }

    loop = _ensure_event_loop()
    ib = IB()

    try:
        # Random ClientID prevents collision if two browser tabs refresh simultaneously
        client_id = random.randint(1000, 9999)

        # 1. Connect (Sync wrapper around async)
        loop.run_until_complete(ib.connectAsync(
            _config.get('connection', {}).get('host', '127.0.0.1'),
            _config.get('connection', {}).get('port', 7497),
            clientId=client_id,
            timeout=15
        ))

        if not ib.isConnected():
            result['connection_status'] = 'FAILED'
            result['error'] = 'Connection timeout'
            return result

        result['connection_status'] = 'CONNECTED'

        # Configure market data type
        configure_market_data_type(ib)

        # 2. Start PnL Subscription (Early)
        # Allows PnL data to arrive while we fetch other data
        accounts = ib.managedAccounts()
        if accounts:
            ib.reqPnL(accounts[0])

        # 3. Fetch Portfolio (Synchronous/Blocking)
        # In-memory dictionary lookup, very fast.
        result['portfolio_items'] = ib.portfolio()

        # 4. Async Fetch for Parallelizable Data with Robust Error Handling
        async def _fetch_rest():
            # Wait for PnL data to arrive (parallel with other fetches)
            async def wait_pnl():
                await asyncio.sleep(1.0)
                return ib.pnl()

            # Launch parallel tasks with timeouts
            # accountSummaryAsync returns list[AccountValue]
            # reqPositionsAsync returns list[Position]
            # reqAllOpenOrdersAsync returns list[Trade] (fix: need to extract .order)

            t1 = asyncio.wait_for(ib.accountSummaryAsync(), timeout=15)
            t2 = asyncio.wait_for(ib.reqPositionsAsync(), timeout=15)
            t3 = asyncio.wait_for(ib.reqAllOpenOrdersAsync(), timeout=15)
            t4 = wait_pnl()

            # Use return_exceptions=True so one failure doesn't kill all data
            return await asyncio.gather(t1, t2, t3, t4, return_exceptions=True)

        # Execute async gather
        async_results = loop.run_until_complete(_fetch_rest())

        # 5. Unpack Results with Type Safety
        # Summary
        if isinstance(async_results[0], list):
            summary_data = async_results[0]
        else:
            logger.warning(f"Account Summary fetch failed: {async_results[0]}")
            summary_data = []

        # Positions
        if isinstance(async_results[1], list):
            result['open_positions'] = async_results[1]
        else:
            logger.warning(f"Positions fetch failed: {async_results[1]}")
            result['open_positions'] = []

        # Orders (Fix: Extract .order from Trade objects if needed)
        if isinstance(async_results[2], list):
            trades = async_results[2]
            # reqAllOpenOrdersAsync returns Trade objects, but openOrders() returned Order objects.
            # We map Trade -> Trade.order to maintain compatibility.
            result['pending_orders'] = [t.order for t in trades] if trades and hasattr(trades[0], 'order') else trades
        else:
            logger.warning(f"Open Orders fetch failed: {async_results[2]}")
            result['pending_orders'] = []

        # PnL
        if isinstance(async_results[3], list):
            pnl_data = async_results[3]
        else:
            logger.warning(f"PnL fetch failed: {async_results[3]}")
            pnl_data = []

        # Process Account Summary
        if summary_data:
            for av in summary_data:
                result['account_summary'][av.tag] = av.value
                if av.tag == 'NetLiquidation':
                    result['net_liquidation'] = float(av.value)
                elif av.tag == 'UnrealizedPnL':
                    result['unrealized_pnl'] = float(av.value)
                elif av.tag == 'RealizedPnL':
                    result['realized_pnl'] = float(av.value)
                elif av.tag == 'MaintMarginReq':
                    result['maint_margin'] = float(av.value)

        # Process PnL
        if pnl_data:
            raw_pnl = pnl_data[0].dailyPnL
            import math
            result['daily_pnl'] = 0.0 if (raw_pnl is None or math.isnan(raw_pnl)) else raw_pnl

    except Exception as e:
        result['connection_status'] = 'ERROR'
        result['error'] = str(e)
        logger.error(f"IB fetch failed: {e}")

    finally:
        if ib.isConnected():
            ib.disconnect()
            # CRITICAL: Blocking sleep allows Gateway to cleanup TCP state
            time.sleep(3.0)

    return result


@st.cache_data(ttl=300)
def fetch_todays_benchmark_data():
    """Fetches today's performance for SPY and Commodity Futures from Yahoo Finance."""
    try:
        # Commodity-agnostic: derive ticker from config
        config = get_config()
        commodity_ticker = config.get('commodity', {}).get('ticker', 'KC')
        yf_commodity = f"{commodity_ticker}=F"
        tickers = ['SPY', yf_commodity]

        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)['Close']
        if data.empty:
            return {}
        changes = {}
        for ticker in tickers:
            try:
                if isinstance(data, pd.DataFrame) and ticker in data.columns:
                    series = data[ticker].dropna()
                else:
                    if len(tickers) == 1:
                        series = data.dropna()
                    else:
                        continue
                if len(series) >= 2:
                    changes[ticker] = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100
                else:
                    changes[ticker] = 0.0
            except (ValueError, TypeError, KeyError, IndexError):
                changes[ticker] = 0.0
        return changes
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_benchmark_data(start_date, end_date):
    """Fetches S&P 500 (SPY) and Commodity Futures from Yahoo Finance for a date range."""
    try:
        # Commodity-agnostic: derive ticker from config
        config = get_config()
        commodity_ticker = config.get('commodity', {}).get('ticker', 'KC')
        yf_commodity = f"{commodity_ticker}=F"
        tickers = ['SPY', yf_commodity]

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date + timedelta(days=1),
            progress=False,
            auto_adjust=True
        )['Close']
        if data.empty:
            return pd.DataFrame()
        normalized = data.apply(lambda x: (x / x.dropna().iloc[0]) - 1) * 100
        return normalized
    except Exception:
        return pd.DataFrame()


# === DECISION GRADING FUNCTIONS (Key for The Scorecard) ===

def grade_decision_quality(council_df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """
    Categorizes every AI decision as WIN, LOSS, or PENDING.
    Also ensures 'pnl' column is populated for visualization.

    CRITICAL: Volatility trades use volatility_outcome field as source of truth.
    Thresholds calibrated to IV regime (~35%): Straddle=1.8%, Condor=1.5%

    OPTIMIZATION: Vectorized implementation (~30x speedup over iterrows)
    """
    if council_df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid modifying original
    graded_df = council_df.copy()

    # Initialize outcome to PENDING
    graded_df['outcome'] = 'PENDING'

    # Ensure pnl is numeric
    graded_df['pnl'] = pd.to_numeric(graded_df.get('pnl_realized', np.nan), errors='coerce')

    # --- VOLATILITY TRADES ---
    vol_mask = graded_df['prediction_type'] == 'VOLATILITY'
    if vol_mask.any():
        big_move = graded_df['volatility_outcome'] == 'BIG_MOVE'
        stayed_flat = graded_df['volatility_outcome'] == 'STAYED_FLAT'
        straddle = graded_df['strategy_type'] == 'LONG_STRADDLE'
        condor = graded_df['strategy_type'] == 'IRON_CONDOR'

        # Win conditions
        win_mask = vol_mask & ((big_move & straddle) | (stayed_flat & condor))
        graded_df.loc[win_mask, 'outcome'] = 'WIN'

        # Loss conditions
        loss_mask = vol_mask & ((big_move & ~straddle) | (stayed_flat & ~condor))
        graded_df.loc[loss_mask, 'outcome'] = 'LOSS'

    # --- DIRECTIONAL TRADES ---
    # Treat NaN prediction_type as DIRECTIONAL (legacy support)
    dir_mask = (graded_df['prediction_type'] == 'DIRECTIONAL') | (graded_df['prediction_type'].isna())

    if dir_mask.any():
        # Define masks
        pnl_active = graded_df['pnl'] != 0
        non_neutral_mask = graded_df['master_decision'] != 'NEUTRAL'

        # 1. PnL Logic (Primary)
        # Only apply to non-neutral decisions to match original logic
        graded_df.loc[dir_mask & non_neutral_mask & pnl_active & (graded_df['pnl'] > 0), 'outcome'] = 'WIN'
        graded_df.loc[dir_mask & non_neutral_mask & pnl_active & (graded_df['pnl'] < 0), 'outcome'] = 'LOSS'

        # 2. Trend Logic (Secondary)
        pending_mask = graded_df['outcome'] == 'PENDING'
        if 'actual_trend_direction' not in graded_df.columns:
            graded_df['actual_trend_direction'] = None
        trend_mask = graded_df['actual_trend_direction'].notna()

        candidates = dir_mask & non_neutral_mask & pending_mask & trend_mask

        if candidates.any():
            bullish = graded_df['master_decision'] == 'BULLISH'
            bearish = graded_df['master_decision'] == 'BEARISH'
            up = graded_df['actual_trend_direction'] == 'UP'
            down = graded_df['actual_trend_direction'] == 'DOWN'

            graded_df.loc[candidates & bullish & up, 'outcome'] = 'WIN'
            graded_df.loc[candidates & bullish & down, 'outcome'] = 'LOSS'
            graded_df.loc[candidates & bearish & down, 'outcome'] = 'WIN'
            graded_df.loc[candidates & bearish & up, 'outcome'] = 'LOSS'

    # Filter out rows that shouldn't be displayed (NEUTRAL directional with no position)
    # Keep VOLATILITY trades even if master_decision is NEUTRAL
    if 'prediction_type' in graded_df.columns:
        graded_df = graded_df[
            (graded_df['prediction_type'] == 'VOLATILITY') |
            (graded_df['master_decision'] != 'NEUTRAL') |
            (graded_df['outcome'] != 'PENDING')
        ]
    else:
        graded_df = graded_df[
            (graded_df['master_decision'] != 'NEUTRAL') |
            (graded_df['outcome'] != 'PENDING')
        ]

    return graded_df


def calculate_confusion_matrix(graded_df: pd.DataFrame) -> dict:
    """
    Calculates confusion matrix for decision quality analysis.

    Supports both DIRECTIONAL and VOLATILITY trade evaluation.

    For DIRECTIONAL trades:
      - TP: BULLISH decision + Market UP = WIN
      - FP: BULLISH decision + Market DOWN = LOSS
      - TN: BEARISH decision + Market DOWN = WIN
      - FN: BEARISH decision + Market UP = LOSS

    For VOLATILITY trades:
      - TP: Correct strategy for outcome (Straddle+BigMove OR Condor+Flat)
      - FP: Wrong strategy for outcome
      - TN/FN: Not directly applicable (mapped to TP/FP)

    Returns:
        dict with matrix values, metrics, and volatility-specific counts
    """
    result = {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
        'total': 0,
        # Volatility-specific metrics for dashboard display
        'vol_wins': 0,
        'vol_losses': 0,
        'vol_total': 0
    }

    if graded_df.empty:
        return result

    # Filter to only graded decisions (not PENDING)
    graded = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()

    if graded.empty:
        return result

    # --- Process VOLATILITY TRADES ---
    if 'prediction_type' in graded.columns:
        vol_trades = graded[graded['prediction_type'] == 'VOLATILITY']
        if not vol_trades.empty:
            vol_wins = len(vol_trades[vol_trades['outcome'] == 'WIN'])
            vol_losses = len(vol_trades[vol_trades['outcome'] == 'LOSS'])

            result['vol_wins'] = vol_wins
            result['vol_losses'] = vol_losses
            result['vol_total'] = vol_wins + vol_losses

            # Map volatility trades to confusion matrix
            # WIN = Correct prediction (True Positive)
            # LOSS = Incorrect prediction (False Positive)
            result['true_positive'] += vol_wins
            result['false_positive'] += vol_losses

    # --- Process DIRECTIONAL TRADES ---
    # Filter to only directional trades with BULLISH/BEARISH decisions
    if 'prediction_type' in graded.columns:
        dir_trades = graded[graded['prediction_type'] != 'VOLATILITY']
    else:
        dir_trades = graded.copy()

    dir_trades = dir_trades[dir_trades['master_decision'].isin(['BULLISH', 'BEARISH'])]

    if not dir_trades.empty:
        tp = len(dir_trades[(dir_trades['master_decision'] == 'BULLISH') & (dir_trades['outcome'] == 'WIN')])
        fp = len(dir_trades[(dir_trades['master_decision'] == 'BULLISH') & (dir_trades['outcome'] == 'LOSS')])
        tn = len(dir_trades[(dir_trades['master_decision'] == 'BEARISH') & (dir_trades['outcome'] == 'WIN')])
        fn = len(dir_trades[(dir_trades['master_decision'] == 'BEARISH') & (dir_trades['outcome'] == 'LOSS')])

        result['true_positive'] += tp
        result['false_positive'] += fp
        result['true_negative'] += tn
        result['false_negative'] += fn

    # --- Calculate Final Metrics ---
    tp = result['true_positive']
    fp = result['false_positive']
    tn = result['true_negative']
    fn = result['false_negative']
    total = tp + fp + tn + fn

    result['total'] = total
    result['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    result['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    result['accuracy'] = (tp + tn) / total if total > 0 else 0.0

    return result


def calculate_agent_scores(council_df: pd.DataFrame, live_price: float = None) -> dict:
    """
    Calculates accuracy scores for each sub-agent based on actual outcomes.

    VECTORIZED OPTIMIZATION: Uses pandas masking instead of iterrows for 50x+ speedup.
    MEMORY OPTIMIZATION: Subsets dataframe columns before copying to avoid duplicating large text fields.

    ARCHITECTURE NOTES:
    - Agent Accuracy = Did the agent correctly predict market behavior?
    - Trade Success = Did the Master's trade make money?

    CRITICAL DISTINCTION FOR VOLATILITY TRADES:
    - master_decision is ALWAYS 'NEUTRAL' for vol trades (by design)
    - Master is scored on STRATEGY success, not direction
    - Volatility agent is scored on PREDICTION accuracy
    - Skip NEUTRAL for other agents (means "no opinion")

    Supports both DIRECTIONAL (Up/Down) and VOLATILITY (Big Move/Flat) grading.
    """
    import numpy as np

    agents = [
        'meteorologist_sentiment',
        'macro_sentiment',
        'geopolitical_sentiment',
        'fundamentalist_sentiment',
        'sentiment_sentiment',
        'technical_sentiment',
        'volatility_sentiment',
        'master_decision'
    ]

    scores = {agent: {'correct': 0, 'total': 0, 'accuracy': 0.0} for agent in agents}

    if council_df.empty:
        return scores

    # OPTIMIZATION: Only copy necessary columns to avoid overhead from large text fields
    # Use set to avoid duplicates (e.g. master_decision is in agents list too)
    needed_cols = {
        'prediction_type', 'strategy_type', 'volatility_outcome',
        'volatility_sentiment', 'master_decision', 'entry_price',
        'actual_trend_direction'
    }
    needed_cols.update(agents)

    # Intersect with available columns
    available_cols = [c for c in needed_cols if c in council_df.columns]

    # Work on a subset copy to minimize memory/time
    # Avoid SettingWithCopy warnings and preserve original df
    df = council_df[available_cols].copy()

    # === PRE-PROCESSING: Fill missing prediction_type ===
    if 'prediction_type' not in df.columns:
        df['prediction_type'] = np.nan

    vol_strategies = ['LONG_STRADDLE', 'IRON_CONDOR']
    dir_strategies = ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']

    # Vectorized fill
    if 'strategy_type' in df.columns:
        df.loc[df['strategy_type'].isin(vol_strategies) & df['prediction_type'].isna(), 'prediction_type'] = 'VOLATILITY'
        df.loc[df['strategy_type'].isin(dir_strategies) & df['prediction_type'].isna(), 'prediction_type'] = 'DIRECTIONAL'

    # ================================================================
    # SECTION 1: VOLATILITY TRADES (Vectorized)
    # ================================================================
    vol_df = df[df['prediction_type'] == 'VOLATILITY'].copy()

    if not vol_df.empty and 'volatility_outcome' in vol_df.columns:
        # Filter only rows with valid outcomes
        vol_df = vol_df[vol_df['volatility_outcome'].notna()]

        if not vol_df.empty:
            # --- A. Score Master Strategist (STRATEGY SUCCESS) ---
            # Win conditions: (Straddle + Big Move) OR (Condor + Flat)
            wins = (
                ((vol_df['strategy_type'] == 'LONG_STRADDLE') & (vol_df['volatility_outcome'] == 'BIG_MOVE')) |
                ((vol_df['strategy_type'] == 'IRON_CONDOR') & (vol_df['volatility_outcome'] == 'STAYED_FLAT'))
            )

            scores['master_decision']['total'] += len(vol_df)
            scores['master_decision']['correct'] += wins.sum()

            # --- B. Score Volatility Agent (PREDICTION ACCURACY) ---
            if 'volatility_sentiment' in vol_df.columns:
                # Normalize sentiment strings
                vol_sent = vol_df['volatility_sentiment'].astype(str).str.upper().str.strip()

                # Filter out NEUTRAL/invalid
                valid_mask = ~vol_sent.isin(['NEUTRAL', 'NONE', '', 'NAN', 'N/A'])
                valid_vol = vol_df[valid_mask].copy()
                valid_sent = vol_sent[valid_mask]

                if not valid_vol.empty:
                    # Prediction logic
                    predicted_high = valid_sent.isin(['HIGH', 'BULLISH', 'VOLATILE'])
                    predicted_low = valid_sent.isin(['LOW', 'BEARISH', 'QUIET', 'RANGE_BOUND'])

                    correct_vol = (
                        ((valid_vol['volatility_outcome'] == 'BIG_MOVE') & predicted_high) |
                        ((valid_vol['volatility_outcome'] == 'STAYED_FLAT') & predicted_low)
                    )

                    scores['volatility_sentiment']['total'] += len(valid_vol)
                    scores['volatility_sentiment']['correct'] += correct_vol.sum()

    # ================================================================
    # SECTION 2: DIRECTIONAL TRADES (Vectorized)
    # ================================================================
    dir_df = df[df['prediction_type'] == 'DIRECTIONAL'].copy()

    if not dir_df.empty:
        # Fallback: Infer actual_trend_direction from live_price if missing
        if live_price is not None and 'entry_price' in dir_df.columns:
            # Check if actual_trend_direction column exists, create if not
            if 'actual_trend_direction' not in dir_df.columns:
                dir_df['actual_trend_direction'] = np.nan

            # Ensure object type to avoid FutureWarning when setting string values
            if dir_df['actual_trend_direction'].dtype != 'object':
                 dir_df['actual_trend_direction'] = dir_df['actual_trend_direction'].astype(object)

            missing_actual = dir_df['actual_trend_direction'].isna() | (dir_df['actual_trend_direction'] == '') | (dir_df['actual_trend_direction'] == 'NEUTRAL')

            if missing_actual.any():
                # Ensure numeric entry_price
                entries = pd.to_numeric(dir_df.loc[missing_actual, 'entry_price'], errors='coerce')

                # Apply thresholds (0.5% move)
                up_mask = (live_price > entries * 1.005)
                down_mask = (live_price < entries * 0.995)

                # Fill inferred values using index alignment
                dir_df.loc[up_mask.index[up_mask], 'actual_trend_direction'] = 'UP'
                dir_df.loc[down_mask.index[down_mask], 'actual_trend_direction'] = 'DOWN'

        # Filter for valid actual trends
        if 'actual_trend_direction' in dir_df.columns:
            valid_trend = dir_df['actual_trend_direction'].isin(['UP', 'DOWN'])
            scored_dir = dir_df[valid_trend].copy()

            if not scored_dir.empty:
                actual_up = scored_dir['actual_trend_direction'] == 'UP'
                actual_down = scored_dir['actual_trend_direction'] == 'DOWN'

                for agent in agents:
                    # Skip master_decision for vol strategies (redundant given df filtering but safe to keep)
                    # Actually, we already filtered to prediction_type == 'DIRECTIONAL', so Master applies here too
                    # UNLESS it's a vol strategy mislabeled. But we trust prediction_type logic above.

                    if agent not in scored_dir.columns:
                        continue

                    # Normalize sentiment
                    agent_sent = scored_dir[agent].astype(str).str.upper().str.strip()

                    # Filter NEUTRAL
                    valid_agent_mask = ~agent_sent.isin(['NEUTRAL', 'NONE', '', 'NAN', 'N/A'])

                    if not valid_agent_mask.any():
                        continue

                    # Calculate correctness
                    # Bullish prediction matches UP, Bearish matches DOWN
                    is_bullish = agent_sent[valid_agent_mask].isin(['BULLISH', 'LONG', 'UP'])

                    # Using the filtered subset
                    subset_actual_up = actual_up[valid_agent_mask]
                    subset_actual_down = actual_down[valid_agent_mask]

                    correct_bull = is_bullish & subset_actual_up
                    correct_bear = (~is_bullish) & subset_actual_down

                    total_correct = correct_bull | correct_bear

                    scores[agent]['total'] += len(total_correct)
                    scores[agent]['correct'] += total_correct.sum()

    # ================================================================
    # SECTION 3: Calculate Final Accuracy Percentages
    # ================================================================
    for agent in agents:
        if scores[agent]['total'] > 0:
            scores[agent]['accuracy'] = scores[agent]['correct'] / scores[agent]['total']

    return scores


def calculate_rolling_win_rate(graded_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates rolling win rate over the specified window.

    Returns:
        DataFrame with timestamp and rolling_win_rate columns
    """
    if graded_df.empty:
        return pd.DataFrame()

    graded = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()
    if graded.empty:
        return pd.DataFrame()

    graded = graded.sort_values('timestamp')
    graded['is_win'] = (graded['outcome'] == 'WIN').astype(int)
    graded['rolling_win_rate'] = graded['is_win'].rolling(window=window, min_periods=1).mean() * 100

    return graded[['timestamp', 'rolling_win_rate']]


def calculate_learning_metrics(graded_df: pd.DataFrame, windows: list[int] = None) -> dict:
    """
    Calculates time-series learning metrics for the Learning Curve section.

    Uses trade-sequence x-axis (not calendar time) since trades are irregularly spaced.
    Only includes resolved trades (WIN/LOSS) — PENDING trades are excluded.

    Args:
        graded_df: Output of grade_decision_quality() with 'outcome' column.
        windows: Rolling window sizes for win rate. Defaults to [10, 20, 30].

    Returns:
        dict with keys:
        - 'trade_series': DataFrame with trade_num, timestamp, rolling win rates,
          cumulative P&L, process_score, skill_pct columns
        - 'has_data': bool — whether there's enough data to plot
        - 'total_resolved': int — number of resolved trades
    """
    if windows is None:
        windows = [10, 20, 30]

    result = {'trade_series': pd.DataFrame(), 'has_data': False, 'total_resolved': 0}

    if graded_df.empty:
        return result

    resolved = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()
    if len(resolved) < 3:
        return result

    resolved = resolved.sort_values('timestamp').reset_index(drop=True)
    resolved['trade_num'] = range(1, len(resolved) + 1)
    resolved['is_win'] = (resolved['outcome'] == 'WIN').astype(int)

    # Rolling win rates at each window size
    for w in windows:
        resolved[f'win_rate_{w}'] = (
            resolved['is_win'].rolling(window=w, min_periods=max(3, w // 4)).mean() * 100
        )

    # Cumulative P&L
    pnl_col = 'pnl' if 'pnl' in resolved.columns else 'pnl_realized'
    if pnl_col in resolved.columns:
        resolved['cum_pnl'] = pd.to_numeric(resolved[pnl_col], errors='coerce').fillna(0).cumsum()
    else:
        resolved['cum_pnl'] = 0.0

    # Process quality: confidence * |weighted_score| (same formula as Process vs Outcome)
    if 'master_confidence' in resolved.columns and 'weighted_score' in resolved.columns:
        w_score = pd.to_numeric(resolved['weighted_score'], errors='coerce').abs().fillna(0)
        conf = pd.to_numeric(resolved['master_confidence'], errors='coerce').fillna(0.5)
        resolved['process_score'] = conf * w_score

        # Normalize to 0-1
        max_ps = resolved['process_score'].max()
        if max_ps > 0:
            resolved['process_score_norm'] = resolved['process_score'] / max_ps
        else:
            resolved['process_score_norm'] = 0.5

        # Classify quadrants: SKILL = good process + win
        median_ps = resolved['process_score_norm'].median()
        good_process = resolved['process_score_norm'] >= median_ps
        good_outcome = resolved['is_win'] == 1

        # Rolling SKILL % over the balanced (middle) window
        w_mid = windows[len(windows) // 2] if len(windows) > 1 else windows[0]
        resolved['is_skill'] = (good_process & good_outcome).astype(int)
        resolved['skill_pct'] = (
            resolved['is_skill'].rolling(window=w_mid, min_periods=max(3, w_mid // 4)).mean() * 100
        )

        # Rolling process score (outcome-independent signal strength)
        resolved['rolling_process'] = (
            resolved['process_score_norm'].rolling(window=w_mid, min_periods=max(3, w_mid // 4)).mean() * 100
        )
    else:
        resolved['process_score_norm'] = 0.5
        resolved['skill_pct'] = np.nan
        resolved['rolling_process'] = np.nan

    keep_cols = ['trade_num', 'timestamp', 'cum_pnl', 'process_score_norm', 'skill_pct', 'rolling_process']
    keep_cols += [f'win_rate_{w}' for w in windows]

    result['trade_series'] = resolved[keep_cols]
    result['has_data'] = True
    result['total_resolved'] = len(resolved)

    return result


def calculate_rolling_agent_accuracy(council_df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calculates per-agent rolling accuracy over a trade-sequence window.

    Handles both DIRECTIONAL (sentiment vs actual_trend_direction) and
    VOLATILITY (volatility_sentiment vs volatility_outcome) scoring.

    Args:
        council_df: Raw council history DataFrame (pre-grading).
        window: Rolling window size for accuracy calculation.

    Returns:
        DataFrame with columns: trade_num, timestamp, and one column per agent
        containing rolling accuracy (0-100%). Empty DataFrame if insufficient data.
    """
    if council_df.empty:
        return pd.DataFrame()

    agents = [
        'meteorologist_sentiment', 'macro_sentiment', 'geopolitical_sentiment',
        'fundamentalist_sentiment', 'sentiment_sentiment', 'technical_sentiment',
        'volatility_sentiment', 'master_decision'
    ]

    df = council_df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Ensure prediction_type exists
    if 'prediction_type' not in df.columns:
        df['prediction_type'] = np.nan
    vol_strategies = ['LONG_STRADDLE', 'IRON_CONDOR']
    dir_strategies = ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']
    if 'strategy_type' in df.columns:
        df.loc[df['strategy_type'].isin(vol_strategies) & df['prediction_type'].isna(), 'prediction_type'] = 'VOLATILITY'
        df.loc[df['strategy_type'].isin(dir_strategies) & df['prediction_type'].isna(), 'prediction_type'] = 'DIRECTIONAL'

    # --- Score each agent per row ---
    for agent in agents:
        if agent not in df.columns:
            df[f'{agent}_correct'] = np.nan
            continue

        correct = pd.Series(np.nan, index=df.index)

        # DIRECTIONAL scoring
        dir_mask = (df['prediction_type'] == 'DIRECTIONAL')
        if 'actual_trend_direction' in df.columns:
            has_actual = dir_mask & df['actual_trend_direction'].isin(['UP', 'DOWN'])
            if has_actual.any():
                sent = df.loc[has_actual, agent].astype(str).str.upper().str.strip()
                valid = ~sent.isin(['NEUTRAL', 'NONE', '', 'NAN', 'N/A'])

                is_bullish = sent.isin(['BULLISH', 'LONG', 'UP'])
                actual_up = df.loc[has_actual, 'actual_trend_direction'] == 'UP'
                actual_down = df.loc[has_actual, 'actual_trend_direction'] == 'DOWN'

                agent_correct = ((is_bullish & actual_up) | (~is_bullish & actual_down))
                correct.loc[has_actual] = np.where(valid, agent_correct.astype(float), np.nan)

        # VOLATILITY scoring (only for volatility_sentiment and master_decision)
        if agent in ('volatility_sentiment', 'master_decision'):
            vol_mask = (df['prediction_type'] == 'VOLATILITY')
            if 'volatility_outcome' in df.columns:
                has_outcome = vol_mask & df['volatility_outcome'].notna()
                if has_outcome.any():
                    if agent == 'master_decision':
                        # Master scored on strategy success
                        if 'strategy_type' in df.columns:
                            straddle_win = (
                                (df.loc[has_outcome, 'strategy_type'] == 'LONG_STRADDLE') &
                                (df.loc[has_outcome, 'volatility_outcome'] == 'BIG_MOVE')
                            )
                            condor_win = (
                                (df.loc[has_outcome, 'strategy_type'] == 'IRON_CONDOR') &
                                (df.loc[has_outcome, 'volatility_outcome'] == 'STAYED_FLAT')
                            )
                            correct.loc[has_outcome] = (straddle_win | condor_win).astype(float)
                    else:
                        # Volatility agent scored on prediction
                        sent = df.loc[has_outcome, agent].astype(str).str.upper().str.strip()
                        valid = ~sent.isin(['NEUTRAL', 'NONE', '', 'NAN', 'N/A'])
                        predicted_high = sent.isin(['HIGH', 'BULLISH', 'VOLATILE'])
                        predicted_low = sent.isin(['LOW', 'BEARISH', 'QUIET', 'RANGE_BOUND'])
                        big_move = df.loc[has_outcome, 'volatility_outcome'] == 'BIG_MOVE'
                        stayed_flat = df.loc[has_outcome, 'volatility_outcome'] == 'STAYED_FLAT'
                        agent_correct = ((big_move & predicted_high) | (stayed_flat & predicted_low))
                        correct.loc[has_outcome] = np.where(valid, agent_correct.astype(float), np.nan)

        df[f'{agent}_correct'] = correct

    # --- Build rolling accuracy ---
    # Only keep rows where at least one agent has a score
    correct_cols = [f'{a}_correct' for a in agents]
    has_any = df[correct_cols].notna().any(axis=1)
    scored = df[has_any].copy().reset_index(drop=True)

    if len(scored) < 3:
        return pd.DataFrame()

    scored['trade_num'] = range(1, len(scored) + 1)

    result = scored[['trade_num', 'timestamp']].copy()
    for agent in agents:
        col = f'{agent}_correct'
        if col in scored.columns:
            result[agent] = (
                scored[col].rolling(window=window, min_periods=max(3, window // 4)).mean() * 100
            )

    return result


def calculate_confidence_calibration(graded_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Calculates confidence calibration data: binned confidence vs actual win rate.

    Perfect calibration = 45-degree line (70% confidence should win 70% of the time).
    Above the line = under-confident. Below = overconfident.

    Args:
        graded_df: Output of grade_decision_quality() with 'outcome' column.
        n_bins: Number of confidence bins.

    Returns:
        DataFrame with columns: bin_center, actual_win_rate, count, bin_label.
        Empty DataFrame if insufficient data.
    """
    if graded_df.empty or 'master_confidence' not in graded_df.columns:
        return pd.DataFrame()

    resolved = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])].copy()
    if len(resolved) < 5:
        return pd.DataFrame()

    resolved['confidence'] = pd.to_numeric(resolved['master_confidence'], errors='coerce')
    resolved = resolved[resolved['confidence'].notna() & (resolved['confidence'] > 0)]
    if resolved.empty:
        return pd.DataFrame()

    resolved['is_win'] = (resolved['outcome'] == 'WIN').astype(int)

    # Create bins — use quantile-based if data is clustered, else uniform
    conf_min = resolved['confidence'].min()
    conf_max = resolved['confidence'].max()

    # Uniform bins across the confidence range
    bins = np.linspace(max(conf_min - 0.01, 0), min(conf_max + 0.01, 1.0), n_bins + 1)
    resolved['bin'] = pd.cut(resolved['confidence'], bins=bins, include_lowest=True)

    calibration = resolved.groupby('bin', observed=True).agg(
        actual_win_rate=('is_win', 'mean'),
        count=('is_win', 'count'),
        avg_confidence=('confidence', 'mean')
    ).reset_index()

    calibration['actual_win_rate'] *= 100
    calibration['bin_center'] = calibration['avg_confidence'] * 100
    calibration['bin_label'] = calibration['bin'].astype(str)

    # Expected win rate = bin midpoint (what a perfectly calibrated system would achieve)
    # .astype(float) needed because pd.cut produces categorical dtype
    calibration['expected_win_rate'] = calibration['bin'].apply(
        lambda b: (b.left + b.right) / 2 * 100 if hasattr(b, 'left') else 50
    ).astype(float)

    return calibration[['bin_center', 'actual_win_rate', 'expected_win_rate', 'count', 'bin_label']]


# === PORTFOLIO FUNCTIONS ===

def fetch_portfolio_data(_config, trade_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches current portfolio positions from IB and enriches with trade ledger data.
    """
    ib = IB()
    portfolio_data = []

    try:
        loop = _ensure_event_loop()

        # Use run_until_complete with connectAsync to ensure proper awaiting
        loop.run_until_complete(ib.connectAsync(
            _config['connection']['host'],
            _config['connection']['port'],
            clientId=random.randint(1000, 9999)
        ))
        configure_market_data_type(ib)

        positions = ib.portfolio()

        for pos in positions:
            if pos.position == 0:
                continue

            # Find matching trade in ledger for additional context
            symbol = pos.contract.localSymbol
            matching_trade = None
            if not trade_df.empty:
                matches = trade_df[trade_df['local_symbol'] == symbol]
                if not matches.empty:
                    matching_trade = matches.iloc[-1]

            portfolio_data.append({
                'Symbol': symbol,
                'Position': pos.position,
                'Avg Cost': pos.averageCost,
                'Market Value': pos.marketValue,
                'Unrealized P&L': pos.unrealizedPNL,
                'Combo ID': matching_trade['position_id'] if matching_trade is not None else 'Unknown',
                'Days Held': (datetime.now() - matching_trade['timestamp']).days if matching_trade is not None else 0
            })

    except Exception:
        # st.error(f"Error fetching portfolio: {e}")
        pass
    finally:
        if ib.isConnected():
            ib.disconnect()

    return pd.DataFrame(portfolio_data)


# === VISUALIZATION HELPERS ===

def create_sparkline_data(values: list, timestamps: list = None) -> dict:
    """
    Prepares data for sparkline visualization.
    """
    return {
        'values': values,
        'timestamps': timestamps or list(range(len(values))),
        'min': min(values) if values else 0,
        'max': max(values) if values else 0,
        'current': values[-1] if values else 0
    }


def get_status_color(status: str) -> str:
    """Returns appropriate color for status indicators."""
    status_colors = {
        'ONLINE': 'green',
        'OFFLINE': 'red',
        'STALE': 'orange',
        'WIN': 'green',
        'LOSS': 'red',
        'PENDING': 'gray',
        'BULLISH': 'green',
        'BEARISH': 'red',
        'NEUTRAL': 'gray'
    }
    return status_colors.get(status, 'gray')


# === POSITION DISPLAY NAMES ===

STRATEGY_ABBREVIATIONS = {
    'IRON_CONDOR': 'IC',
    'LONG_STRADDLE': 'LS',
    'BULL_CALL_SPREAD': 'BCS',
    'BEAR_PUT_SPREAD': 'BPS',
    'LONG_STRANGLE': 'LG',
    'CALENDAR_SPREAD': 'CAL',
    'BUTTERFLY': 'BF',
    'DIAGONAL': 'DIAG',
}


def build_thesis_display_name(thesis: dict) -> str:
    """
    Builds a human-readable display name for a thesis.

    Format: "{STRATEGY_ABBREV} {CONTRACT} • {ENTRY_DATE}"
    Example: "IC KOK6 • Jan 30"

    Commodity-agnostic: Derives contract from supporting_data or
    falls back to truncated UUID.

    Args:
        thesis: Dictionary from get_active_theses() with keys:
                position_id, strategy_type, entry_timestamp,
                and optionally supporting_data with contract info.

    Returns:
        Human-readable string for display.
    """
    parts = []

    # 1. Strategy abbreviation
    strategy = thesis.get('strategy_type', 'UNKNOWN')
    abbrev = STRATEGY_ABBREVIATIONS.get(strategy, strategy[:3].upper())
    parts.append(abbrev)

    # 2. Contract name — try to extract from supporting_data or position metadata
    contract_name = None

    # Try supporting_data.contract (most reliable if present)
    supporting = thesis.get('supporting_data', {})
    if isinstance(supporting, dict):
        contract_name = supporting.get('contract') or supporting.get('contract_name')

        # Try to extract from leg symbols if contract not directly available
        if not contract_name:
            legs = supporting.get('legs', [])
            if legs and isinstance(legs[0], dict):
                first_symbol = legs[0].get('local_symbol', '')
                # Extract root: "KOK6 C3.275" → "KOK6"
                if first_symbol:
                    contract_name = first_symbol.strip().split(' ')[0]

    if contract_name:
        parts.append(contract_name)

    # 3. Entry date
    entry_ts = thesis.get('entry_timestamp')
    if entry_ts:
        try:
            if isinstance(entry_ts, str):
                entry_ts = datetime.fromisoformat(entry_ts)
            parts.append(f"• {entry_ts.strftime('%b %d')}")
        except (ValueError, TypeError):
            pass

    display = ' '.join(parts)

    # Fallback: if we only got the abbreviation, add truncated UUID
    if len(parts) <= 1:
        pid = thesis.get('position_id', 'Unknown')
        display = f"{abbrev} {pid[:8]}"

    return display


# === THESIS STATUS FUNCTIONS ===

def get_active_theses() -> list[dict]:
    """
    Retrieves all active trade theses from TMS.
    Returns a list of thesis dictionaries with computed fields.
    """
    try:
        from trading_bot.tms import TransactiveMemory
        tms = TransactiveMemory()

        if not tms.collection:
            return []

        # Query all active theses
        results = tms.collection.get(
            where={"active": "true"},
            include=['documents', 'metadatas']
        )

        theses = []
        now = datetime.now(timezone.utc)

        for doc, meta in zip(results.get('documents', []), results.get('metadatas', [])):
            try:
                thesis = json.loads(doc)

                # Compute age
                entry_time = datetime.fromisoformat(thesis.get('entry_timestamp', ''))
                age_hours = (now - entry_time).total_seconds() / 3600

                # Parse supporting_data for display name construction
                raw_supporting = thesis.get('supporting_data', {})

                thesis_dict = {
                    'position_id': meta.get('trade_id', 'Unknown'),
                    'strategy_type': thesis.get('strategy_type', 'Unknown'),
                    'guardian_agent': thesis.get('guardian_agent', 'Unknown'),
                    'primary_rationale': thesis.get('primary_rationale', '')[:50] + '...',
                    'entry_regime': thesis.get('entry_regime', 'Unknown'),
                    'invalidation_triggers': thesis.get('invalidation_triggers', []),
                    'entry_price': raw_supporting.get('entry_price', 0) if isinstance(raw_supporting, dict) else 0,
                    'confidence': raw_supporting.get('confidence', 0) if isinstance(raw_supporting, dict) else 0,
                    'age_hours': age_hours,
                    'entry_timestamp': entry_time,
                    'supporting_data': raw_supporting,
                }
                thesis_dict['display_name'] = build_thesis_display_name(thesis_dict)
                theses.append(thesis_dict)
            except Exception as e:
                logger.warning(f"Failed to parse thesis: {e}")
                continue

        return sorted(theses, key=lambda x: x['entry_timestamp'], reverse=True)

    except Exception as e:
        logger.error(f"Failed to get active theses: {e}")
        return []


def get_current_market_regime() -> str:
    """
    Get the most recent market regime from available data.

    Priority:
    1. council_history -> entry_regime column (most recent council decision)
    2. "UNKNOWN" if no data

    """
    try:
        # Priority 1: Council history (most recent regime from actual decisions)
        council_path = _get_council_history_path()
        if os.path.exists(council_path):
            df = pd.read_csv(council_path, on_bad_lines='warn')
            if not df.empty and 'entry_regime' in df.columns:
                recent_regimes = df['entry_regime'].dropna()
                if not recent_regimes.empty:
                    return recent_regimes.iloc[-1]

        return "UNKNOWN"

    except Exception as e:
        logger.error(f"Error getting market regime: {e}")
        return "UNKNOWN"


def get_guardian_icon(guardian: str) -> str:
    """Returns an emoji icon for each guardian agent type."""
    icons = {
        'Agronomist': '🌱',
        'Logistics': '🚢',
        'VolatilityAnalyst': '📊',
        'Macro': '💹',
        'Sentiment': '🐦',
        'Master': '👑',
        'Fundamentalist': '📈'
    }
    return icons.get(guardian, '🤖')


def get_strategy_color(strategy_type: str) -> str:
    """Returns a color code for each strategy type."""
    colors = {
        'BULL_CALL_SPREAD': '#00CC96',  # Green
        'BEAR_PUT_SPREAD': '#EF553B',   # Red
        'IRON_CONDOR': '#636EFA',       # Blue
        'LONG_STRADDLE': '#AB63FA'      # Purple
    }
    return colors.get(strategy_type, '#FFFFFF')


# === CROSS-COMMODITY HELPERS (for portfolio home page) ===

def discover_active_commodities() -> list[str]:
    """Return tickers that have a data directory with state.json.

    Scans the data/ directory dynamically — no hardcoded ticker list.
    """
    active = []
    data_root = "data"
    if os.path.isdir(data_root):
        for entry in sorted(os.listdir(data_root)):
            # Commodity tickers are 2-4 uppercase letters
            if not entry.isalpha() or not entry.isupper() or len(entry) > 4:
                continue
            data_dir = os.path.join(data_root, entry)
            state_path = os.path.join(data_dir, "state.json")
            if os.path.isdir(data_dir) and os.path.exists(state_path):
                active.append(entry)
    return active if active else ["KC"]


def load_council_history_for_commodity(ticker: str) -> pd.DataFrame:
    """Load council_history.csv for a specific commodity, adding a 'commodity' column."""
    council_path = _resolve_data_path_for('council_history.csv', ticker)
    if not os.path.exists(council_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(council_path, on_bad_lines='warn')
        if not df.empty:
            if 'timestamp' in df.columns:
                df['timestamp'] = parse_ts_column(df['timestamp'])
            df['commodity'] = ticker
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_budget_status(ticker: str = None) -> dict:
    """Load current budget guard state for the dashboard.

    Reads budget_state.json for the given commodity. Returns an empty dict
    if the file doesn't exist (budget guard hasn't been initialized yet).
    """
    ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
    path = _resolve_data_path_for('budget_state.json', ticker)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_llm_daily_costs(ticker: str = None) -> pd.DataFrame:
    """Load historical daily LLM cost data.

    Reads llm_daily_costs.csv for the given commodity. Returns an empty
    DataFrame if the file doesn't exist yet (no daily reset has occurred).
    """
    ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
    path = _resolve_data_path_for('llm_daily_costs.csv', ticker)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame()


def get_system_heartbeat_for_commodity(ticker: str) -> dict:
    """Get orchestrator heartbeat for a specific commodity."""
    heartbeat = {
        'ticker': ticker,
        'orchestrator_status': 'OFFLINE',
        'orchestrator_last_pulse': None,
        'state_status': 'OFFLINE',
        'state_last_pulse': None,
    }
    log_path = f'logs/orchestrator_{ticker.lower()}.log'
    if os.path.exists(log_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        heartbeat['orchestrator_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['orchestrator_status'] = 'ONLINE' if minutes_since < 10 else 'STALE'

    state_path = os.path.join("data", ticker, "state.json")
    if os.path.exists(state_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(state_path))
        heartbeat['state_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['state_status'] = 'ONLINE' if minutes_since < 10 else 'STALE'

    return heartbeat


@st.cache_data(ttl=60)
def load_prompt_traces(ticker: str = None):
    """Load prompt trace data for dashboard display."""
    ticker = ticker or os.environ.get("COMMODITY_TICKER", "KC")
    try:
        from trading_bot.prompt_trace import get_prompt_trace_df
        df = get_prompt_trace_df(commodity=ticker)
        # Fill NaN for Streamlit compatibility
        numeric_cols = ['demo_count', 'tms_context_count', 'grounded_freshness_hours',
                        'prompt_tokens', 'completion_tokens', 'latency_ms']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        string_cols = ['prompt_source', 'model_provider', 'model_name',
                       'assigned_provider', 'assigned_model', 'persona_hash', 'dspy_version']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        return df
    except Exception as e:
        logger.error(f"Failed to load prompt traces: {e}")
        return pd.DataFrame()
