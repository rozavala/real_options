"""
Shared utilities for the Coffee Bot Mission Control dashboard.
Contains data loading, caching, and decision grading logic.
"""

import streamlit as st
import pandas as pd
import os
import glob
import random
from datetime import datetime, timedelta
import yfinance as yf
import sys
import asyncio
import warnings

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
from trading_bot.model_signals import get_model_signals_df
from config_loader import load_config
from trading_bot.utils import configure_market_data_type

# === CONFIGURATION ===
DEFAULT_STARTING_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "250000"))
STATE_FILE_PATH = 'data/state.json'
ORCHESTRATOR_LOG_PATH = 'logs/orchestrator.log'
COUNCIL_HISTORY_PATH = 'data/council_history.csv'
DAILY_EQUITY_PATH = 'data/daily_equity.csv'


# === DATA LOADING FUNCTIONS ===

@st.cache_data
def get_config():
    """Loads and caches the application configuration."""
    config = load_config()
    if config is None:
        st.error("Fatal: Could not load config.json.")
        return {}
    return config


@st.cache_data(ttl=60)
def load_trade_data():
    """Loads and caches the consolidated trade ledger."""
    try:
        df = get_trade_ledger_df()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to load trade_ledger.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_council_history():
    """Loads council_history.csv for decision analysis."""
    try:
        if os.path.exists(COUNCIL_HISTORY_PATH):
            df = pd.read_csv(COUNCIL_HISTORY_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load council_history.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_equity_data():
    """Loads daily_equity.csv for equity curve visualization."""
    try:
        if os.path.exists(DAILY_EQUITY_PATH):
            df = pd.read_csv(DAILY_EQUITY_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load daily_equity.csv: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=15)
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
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                logs_content[name] = lines[-50:]
            except Exception as e:
                logs_content[name] = [f"Error reading file: {e}"]
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


def get_sentinel_status():
    """
    Parses state.json to determine which Sentinels are active.
    Returns a dict of sentinel names and their status.
    """
    import json
    sentinels = {
        'PriceSentinel': 'Unknown',
        'WeatherSentinel': 'Unknown',
        'NewsSentinel': 'Unknown',
        'LogisticsSentinel': 'Unknown'
    }

    try:
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, 'r') as f:
                state = json.load(f)
                # Parse sentinel status from state if available
                sentinel_state = state.get('sentinels', {})
                for name in sentinels:
                    if name in sentinel_state:
                        sentinels[name] = sentinel_state[name].get('status', 'Active')
                    else:
                        sentinels[name] = 'Active'  # Default assumption
    except Exception:
        pass

    return sentinels


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

    return data


@st.cache_data(ttl=300)
def fetch_todays_benchmark_data():
    """Fetches today's performance for SPY and KC=F from Yahoo Finance."""
    try:
        tickers = ['SPY', 'KC=F']
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
            except:
                changes[ticker] = 0.0
        return changes
    except:
        return {}


@st.cache_data(ttl=3600)
def fetch_benchmark_data(start_date, end_date):
    """Fetches S&P 500 (SPY) and Coffee Futures (KC=F) from Yahoo Finance for a date range."""
    try:
        tickers = ['SPY', 'KC=F']
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

    Logic:
    - For decisions with reconciled data (exit_price exists), use actual P&L.
    - For recent decisions without exit data, mark as PENDING.

    Args:
        council_df: DataFrame from council_history.csv
        lookback_days: Number of days to wait before grading (default 5)

    Returns:
        DataFrame with columns: timestamp, master_decision, master_confidence, outcome, pnl
    """
    if council_df.empty:
        return pd.DataFrame()

    grades = []
    now = datetime.now()

    # Thresholds for volatility grading
    STRADDLE_MOVE_THRESHOLD = 0.03  # 3% move = WIN for straddle
    CONDOR_FLAT_THRESHOLD = 0.02    # <2% move = WIN for condor

    for idx, row in council_df.iterrows():
        decision = row.get('master_decision', 'NEUTRAL')
        confidence = row.get('master_confidence', 0.5)
        timestamp = row.get('timestamp')

        prediction_type = row.get('prediction_type', 'DIRECTIONAL')
        strategy_type = row.get('strategy_type', '')

        # Skip neutral decisions (unless it's a volatility play)
        if decision == 'NEUTRAL' and prediction_type != 'VOLATILITY':
            continue

        grade_entry = {
            'timestamp': timestamp,
            'contract': row.get('contract', 'Unknown'),
            'master_decision': decision,
            'master_confidence': confidence,
            'entry_price': row.get('entry_price'),
            'exit_price': row.get('exit_price'),
            'pnl_realized': row.get('pnl_realized'),
            'actual_trend': row.get('actual_trend_direction'),
            'prediction_type': prediction_type,
            'strategy_type': strategy_type,
            'volatility_outcome': row.get('volatility_outcome'),
            'outcome': 'PENDING'
        }

        # Check if we have reconciled data
        if pd.notna(row.get('pnl_realized')):
            pnl = row['pnl_realized']
            grade_entry['outcome'] = 'WIN' if pnl > 0 else 'LOSS'
            grade_entry['pnl'] = pnl

        # Fallback grading using price movement if P&L not reconciled but prices available
        elif pd.notna(row.get('entry_price')) and pd.notna(row.get('exit_price')):
             try:
                entry = float(row['entry_price'])
                exit_p = float(row['exit_price'])
                pct_change = (exit_p - entry) / entry
                abs_change = abs(pct_change)

                if prediction_type == 'VOLATILITY':
                    if strategy_type == 'LONG_STRADDLE':
                        if abs_change >= STRADDLE_MOVE_THRESHOLD:
                            grade_entry['outcome'] = 'WIN'
                            grade_entry['volatility_outcome'] = 'BIG_MOVE'
                        else:
                            grade_entry['outcome'] = 'LOSS'
                            grade_entry['volatility_outcome'] = 'STAYED_FLAT'
                    elif strategy_type == 'IRON_CONDOR':
                        if abs_change <= CONDOR_FLAT_THRESHOLD:
                            grade_entry['outcome'] = 'WIN'
                            grade_entry['volatility_outcome'] = 'STAYED_FLAT'
                        else:
                            grade_entry['outcome'] = 'LOSS'
                            grade_entry['volatility_outcome'] = 'BIG_MOVE'
                else:
                    # Directional logic
                    actual_trend = 'NEUTRAL'
                    if pct_change > 0: actual_trend = 'UP'
                    elif pct_change < 0: actual_trend = 'DOWN'

                    if decision == 'BULLISH' and actual_trend == 'UP':
                        grade_entry['outcome'] = 'WIN'
                    elif decision == 'BEARISH' and actual_trend == 'DOWN':
                        grade_entry['outcome'] = 'WIN'
                    elif actual_trend in ['UP', 'DOWN']:
                        grade_entry['outcome'] = 'LOSS'
             except:
                 pass

        elif pd.notna(row.get('actual_trend_direction')):
            # Legacy fallback using string trend
            actual = row['actual_trend_direction']
            if decision == 'BULLISH' and actual == 'UP':
                grade_entry['outcome'] = 'WIN'
            elif decision == 'BEARISH' and actual == 'DOWN':
                grade_entry['outcome'] = 'WIN'
            elif actual in ['UP', 'DOWN']:
                grade_entry['outcome'] = 'LOSS'

        grades.append(grade_entry)

    return pd.DataFrame(grades)


def calculate_confusion_matrix(graded_df: pd.DataFrame) -> dict:
    """
    Calculates the confusion matrix for AI decisions.

    Returns:
        dict with keys: true_positive, false_positive, true_negative, false_negative,
                       precision, recall, accuracy
    """
    if graded_df.empty:
        return {
            'true_positive': 0, 'false_positive': 0,
            'true_negative': 0, 'false_negative': 0,
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
            'total': 0
        }

    # Filter to only graded decisions (not PENDING)
    graded = graded_df[graded_df['outcome'].isin(['WIN', 'LOSS'])]

    if graded.empty:
        return {
            'true_positive': 0, 'false_positive': 0,
            'true_negative': 0, 'false_negative': 0,
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0,
            'total': 0
        }

    # Calculate matrix values
    # True Positive: BULLISH decision -> WIN
    # False Positive: BULLISH decision -> LOSS
    # True Negative: BEARISH decision -> WIN
    # False Negative: BEARISH decision -> LOSS

    tp = len(graded[(graded['master_decision'] == 'BULLISH') & (graded['outcome'] == 'WIN')])
    fp = len(graded[(graded['master_decision'] == 'BULLISH') & (graded['outcome'] == 'LOSS')])
    tn = len(graded[(graded['master_decision'] == 'BEARISH') & (graded['outcome'] == 'WIN')])
    fn = len(graded[(graded['master_decision'] == 'BEARISH') & (graded['outcome'] == 'LOSS')])

    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        'true_positive': tp,
        'false_positive': fp,
        'true_negative': tn,
        'false_negative': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'total': total
    }


def calculate_agent_scores(council_df: pd.DataFrame, live_price: float = None) -> dict:
    """
    Calculates accuracy scores for each sub-agent based on actual outcomes.

    Args:
        council_df: DataFrame from council_history.csv
        live_price: Current market price for grading recent decisions

    Returns:
        dict with agent names as keys and dicts of {correct, total, accuracy} as values
    """
    agents = [
        'meteorologist_sentiment',
        'macro_sentiment',
        'geopolitical_sentiment',
        'fundamentalist_sentiment',
        'sentiment_sentiment',
        'technical_sentiment',
        'volatility_sentiment',
        'master_decision',
        'ml_signal'
    ]

    scores = {agent: {'correct': 0, 'total': 0, 'accuracy': 0.0} for agent in agents}

    if council_df.empty:
        return scores

    for idx, row in council_df.iterrows():
        # Determine actual outcome
        actual = None
        if pd.notna(row.get('actual_trend_direction')):
            actual = row['actual_trend_direction']
        elif live_price and pd.notna(row.get('entry_price')):
            entry = row['entry_price']
            if live_price > entry * 1.005:
                actual = 'UP'
            elif live_price < entry * 0.995:
                actual = 'DOWN'

        if not actual:
            continue

        # Score each agent
        for agent in agents:
            sentiment = row.get(agent, None)
            if not sentiment or sentiment == 'NEUTRAL':
                continue

            scores[agent]['total'] += 1

            # Map sentiment to expected direction
            if agent == 'master_decision':
                expected_up = sentiment == 'BULLISH'
            elif agent == 'ml_signal':
                expected_up = sentiment in ['BULLISH', 'Bullish', 'LONG', 'Long', 'long']
            else:
                expected_up = sentiment in ['BULLISH', 'Bullish', 'bullish']

            if (expected_up and actual == 'UP') or (not expected_up and actual == 'DOWN'):
                scores[agent]['correct'] += 1

    # Calculate accuracies
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
