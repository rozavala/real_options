"""
Shared utilities for the Coffee Bot Real Options dashboard.
Contains data loading, caching, and decision grading logic.
"""

import streamlit as st
import pandas as pd
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
from trading_bot.decision_signals import get_decision_signals_df
from config_loader import load_config
from trading_bot.utils import configure_market_data_type
from trading_bot.timestamps import parse_ts_column

# === CONFIGURATION ===
# E4 FIX: Dynamic starting capital handled in get_starting_capital function
STATE_FILE_PATH = 'data/state.json'
ORCHESTRATOR_LOG_PATH = 'logs/orchestrator.log'
COUNCIL_HISTORY_PATH = 'data/council_history.csv'
DAILY_EQUITY_PATH = 'data/daily_equity.csv'


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
    return config


def get_commodity_profile(config: dict = None) -> dict:
    """
    Returns the commodity profile for the active symbol.
    Falls back to Coffee (KC) defaults if not configured.

    EXTENSIBILITY: Add new commodities to config.json when needed.
    """
    if config is None:
        config = get_config()

    symbol = config.get('symbol', 'KC')
    profiles = config.get('commodity_profile', {})

    # KC defaults — the only commodity we need right now
    default_profile = {
        'name': 'Coffee C Arabica',
        'price_unit': 'cents/lb',
        'stop_parse_range': [80, 800],
        'typical_price_range': [100, 600]
    }

    return profiles.get(symbol, default_profile)


@st.cache_data(ttl=60)
def load_trade_data():
    """Loads and caches the consolidated trade ledger."""
    try:
        df = get_trade_ledger_df()
        if not df.empty:
            df['timestamp'] = parse_ts_column(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to load trade_ledger.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_council_history():
    """
    Loads council_history.csv for decision analysis.
    Also loads any archived legacy files and concatenates them.
    """
    try:
        dataframes = []
        data_dir = os.path.dirname(COUNCIL_HISTORY_PATH)

        # Load main file
        if os.path.exists(COUNCIL_HISTORY_PATH):
            df = pd.read_csv(COUNCIL_HISTORY_PATH)
            if not df.empty:
                dataframes.append(df)

        # Load any legacy/archived files (safety net)
        if os.path.exists(data_dir):
            legacy_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.startswith('council_history_legacy') and f.endswith('.csv')
            ]
            for legacy_file in legacy_files:
                try:
                    legacy_df = pd.read_csv(legacy_file)
                    if not legacy_df.empty:
                        dataframes.append(legacy_df)
                except Exception as e:
                    logger.warning(f"Could not load legacy file {legacy_file}: {e}")

        if not dataframes:
            return pd.DataFrame()

        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Normalize timestamp
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = parse_ts_column(combined_df['timestamp'])

        # Remove duplicates (same timestamp + contract)
        if 'timestamp' in combined_df.columns and 'contract' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'contract'], keep='last')

        return combined_df.sort_values('timestamp', ascending=False).reset_index(drop=True)

    except Exception as e:
        st.error(f"Failed to load council_history.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_equity_data():
    """Loads daily_equity.csv for equity curve visualization."""
    try:
        if os.path.exists(DAILY_EQUITY_PATH):
            df = pd.read_csv(DAILY_EQUITY_PATH)
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
            if any(char.isdigit() for char in filename):
                continue
            name = filename.split('.')[0].capitalize()
            logs_content[name] = tail_file(log_path, n_lines=50)

        return logs_content
    except Exception as e:
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
    if os.path.exists(ORCHESTRATOR_LOG_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(ORCHESTRATOR_LOG_PATH))
        heartbeat['orchestrator_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['orchestrator_status'] = 'ONLINE' if minutes_since < heartbeat['alert_threshold_minutes'] else 'STALE'

    # Check State file
    if os.path.exists(STATE_FILE_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(STATE_FILE_PATH))
        heartbeat['state_last_pulse'] = mtime
        minutes_since = (datetime.now() - mtime).total_seconds() / 60
        heartbeat['state_status'] = 'ONLINE' if minutes_since < heartbeat['alert_threshold_minutes'] else 'STALE'

    return heartbeat


# === TASK SCHEDULE TRACKER ===

ACTIVE_SCHEDULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'active_schedule.json'
)
TASK_COMPLETIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'task_completions.json'
)

# Human-readable labels for task names (commodity-agnostic)
TASK_LABELS = {
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
    if not os.path.exists(ACTIVE_SCHEDULE_PATH):
        return result

    try:
        with open(ACTIVE_SCHEDULE_PATH, 'r') as f:
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
            name = task_entry['name']
            label = TASK_LABELS.get(name, name)
            inactive_tasks.append({
                'time_et': time_str,
                'name': name,
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
        if os.path.exists(TASK_COMPLETIONS_PATH):
            with open(TASK_COMPLETIONS_PATH, 'r') as f:
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
        name = task_entry['name']

        # Parse scheduled time to minutes since midnight
        parts = time_str.split(':')
        task_minutes = int(parts[0]) * 60 + int(parts[1])

        # Determine status
        completed_at = completions.get(name)
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
            'label': TASK_LABELS.get(name, name),
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
        from trading_bot.state_manager import StateManager
        state = StateManager._load_raw_sync()

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
    import json
    from datetime import datetime, timezone

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
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, 'r') as f:
                state = json.load(f)

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


# === LIVE DATA FUNCTIONS (IB Connection) ===

@st.cache_data(ttl=60)
def fetch_live_dashboard_data(_config):
    """
    Consolidated fetcher for Account Summary, Daily P&L, and Active Futures Data.
    Uses underscore prefix for config to prevent Streamlit from hashing it.
    """
    ib = IB()
    data = {
        "NetLiquidation": 0.0,
        "MaintMarginReq": 0.0,
        "DailyPnL": 0.0,
        "DailyPnLPct": 0.0,
        "KC_DailyChange": 0.0,
        "KC_Price": 0.0,
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

    except Exception as e:
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

        # Account Summary
        for av in ib.accountSummary():
            result['account_summary'][av.tag] = av.value
            if av.tag == 'NetLiquidation':
                result['net_liquidation'] = float(av.value)
            elif av.tag == 'UnrealizedPnL':
                result['unrealized_pnl'] = float(av.value)
            elif av.tag == 'RealizedPnL':
                result['realized_pnl'] = float(av.value)
            elif av.tag == 'MaintMarginReq':
                result['maint_margin'] = float(av.value)

        # Daily P&L
        accounts = ib.managedAccounts()
        if accounts:
            ib.reqPnL(accounts[0])
            ib.sleep(1)
            pnl_data = ib.pnl()
            if pnl_data:
                raw_pnl = pnl_data[0].dailyPnL
                # NaN is truthy in Python, so `or 0.0` doesn't catch it.
                # Must use explicit math.isnan() check.
                import math
                result['daily_pnl'] = 0.0 if (raw_pnl is None or math.isnan(raw_pnl)) else raw_pnl

        # Positions & Portfolio
        result['open_positions'] = ib.positions()
        result['portfolio_items'] = ib.portfolio()
        result['pending_orders'] = ib.openOrders()

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
    except Exception as e:
        return pd.DataFrame()


# === DECISION GRADING FUNCTIONS (Key for The Scorecard) ===

def grade_decision_quality(council_df: pd.DataFrame, lookback_days: int = 5) -> pd.DataFrame:
    """
    Categorizes every AI decision as WIN, LOSS, or PENDING.
    Also ensures 'pnl' column is populated for visualization.

    CRITICAL: Volatility trades use volatility_outcome field as source of truth.
    Thresholds calibrated to IV regime (~35%): Straddle=1.8%, Condor=1.5%
    """
    if council_df.empty:
        return pd.DataFrame()

    # Work on a copy to avoid modifying original
    graded_df = council_df.copy()

    outcomes = []
    pnl_values = []

    for idx, row in graded_df.iterrows():
        outcome = 'PENDING'
        pnl = float(row.get('pnl_realized', 0.0) or 0.0)  # Ensure numeric

        decision = row.get('master_decision', 'NEUTRAL')
        prediction_type = row.get('prediction_type', 'DIRECTIONAL')
        strategy_type = row.get('strategy_type', '')

        # ================================================================
        # VOLATILITY TRADES: Grade by volatility_outcome (Source of Truth)
        # ================================================================
        if prediction_type == 'VOLATILITY':
            vol_outcome = row.get('volatility_outcome')

            if vol_outcome == 'BIG_MOVE':
                outcome = 'WIN' if strategy_type == 'LONG_STRADDLE' else 'LOSS'
            elif vol_outcome == 'STAYED_FLAT':
                outcome = 'WIN' if strategy_type == 'IRON_CONDOR' else 'LOSS'
            # else: remains PENDING

        # ================================================================
        # DIRECTIONAL TRADES: Grade by P&L or trend direction
        # ================================================================
        elif prediction_type == 'DIRECTIONAL' or prediction_type is None:
            # Skip neutral directional decisions (no position taken)
            if decision == 'NEUTRAL':
                outcomes.append('PENDING')
                pnl_values.append(pnl)
                continue

            # Primary: Use P&L if available
            if pd.notna(row.get('pnl_realized')) and row.get('pnl_realized') != 0:
                pnl = float(row['pnl_realized'])
                if pnl > 0:
                    outcome = 'WIN'
                elif pnl < 0:
                    outcome = 'LOSS'

            # Secondary: Use actual_trend_direction
            elif pd.notna(row.get('actual_trend_direction')):
                actual = row['actual_trend_direction']
                if decision == 'BULLISH' and actual == 'UP':
                    outcome = 'WIN'
                elif decision == 'BULLISH' and actual == 'DOWN':
                    outcome = 'LOSS'
                elif decision == 'BEARISH' and actual == 'DOWN':
                    outcome = 'WIN'
                elif decision == 'BEARISH' and actual == 'UP':
                    outcome = 'LOSS'

        outcomes.append(outcome)
        pnl_values.append(pnl)

    # Add columns to dataframe
    graded_df['outcome'] = outcomes
    graded_df['pnl'] = pnl_values

    # Filter out rows that shouldn't be displayed (NEUTRAL directional with no position)
    # Keep VOLATILITY trades even if master_decision is NEUTRAL
    graded_df = graded_df[
        (graded_df['prediction_type'] == 'VOLATILITY') |
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

    # Work on a copy to avoid SettingWithCopy warnings and preserve original df
    df = council_df.copy()

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

    except Exception as e:
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
        if os.path.exists(COUNCIL_HISTORY_PATH):
            df = pd.read_csv(COUNCIL_HISTORY_PATH)
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
