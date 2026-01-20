"""A collection of utility functions for the trading bot.

This module provides various helper functions used across the trading bot,
including mathematical calculations for option pricing (Black-Scholes),
helpers for normalizing and parsing contract data from Interactive Brokers,
market hour checks, and trade logging utilities.
"""

import holidays
import asyncio
import csv
import logging
import os
import shutil
from datetime import datetime, time
import pytz
from ib_insync import *
import numpy as np
from scipy.stats import norm

from trading_bot.logging_config import setup_logging

# --- Logging Setup ---
setup_logging()

# Global lock for writing to the trade ledger to prevent race conditions
TRADE_LEDGER_LOCK = asyncio.Lock()

def configure_market_data_type(ib: IB):
    """
    Configures the market data type.

    UPDATED: Always requests LIVE (Type 1) data to ensure OrderManager
    liquidity checks pass. This assumes no parallel IB sessions are running.

    If you need to run parallel sessions, set FORCE_DELAYED_DATA=1 in .env
    """
    force_delayed = os.getenv("FORCE_DELAYED_DATA", "0") == "1"

    if force_delayed:
        logging.getLogger(__name__).info("🛠️ FORCE_DELAYED_DATA enabled: Using Delayed Market Data (Type 3)")
        ib.reqMarketDataType(3)
    else:
        logging.getLogger(__name__).info("🛠️ Configuring Market Data: Requesting LIVE (Type 1) Data.")
        ib.reqMarketDataType(1)

def _get_combo_description(trade: Trade) -> str:
    """Creates a human-readable description for a combo/bag trade."""
    if not isinstance(trade.contract, Bag) or not trade.contract.comboLegs:
        return trade.contract.localSymbol

    # Attempt to derive the underlying from the first leg's symbol
    try:
        first_leg = trade.contract.comboLegs[0]
        # This is an assumption, but for single-underlying spreads it holds
        underlying_symbol = util.stockContract(first_leg.conId).symbol
        return f"{underlying_symbol} Combo"
    except Exception:
        return f"Bag_{trade.order.permId}"


def log_order_event(trade: Trade, status: str, message: str = ""):
    """Logs the status change of an order to the `order_events.csv` file.

    This provides a detailed audit trail of every stage an order goes through,
    from submission to cancellation or fill.

    Args:
        trade (Trade): The `ib_insync.Trade` object.
        status (str): The new status of the order (e.g., 'Submitted', 'Filled').
        message (str, optional): Any additional message, like an error reason.
    """
    # Use absolute path to ensure ledger is in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'order_events.csv')
    file_exists = os.path.isfile(ledger_path)

    fieldnames = [
        'timestamp', 'orderId', 'permId', 'clientId', 'local_symbol',
        'action', 'quantity', 'lmtPrice', 'status', 'message'
    ]

    symbol = _get_combo_description(trade)

    try:
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'orderId': trade.order.orderId,
                'permId': trade.order.permId,
                'clientId': trade.order.clientId,
                'local_symbol': symbol,
                'action': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'lmtPrice': trade.order.lmtPrice,
                'status': status,
                'message': message
            })
    except Exception as e:
        logging.error(f"Error writing to order event log: {e}")


def price_option_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict | None:
    """Calculates the theoretical price and Greeks of an option.

    This function uses the Black-Scholes model to compute the price, delta,
    gamma, vega, and theta for a given European option.

    Args:
        S (float): The current price of the underlying asset.
        K (float): The strike price of the option.
        T (float): The time to expiration in years.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        option_type (str): The type of the option, 'C' for Call or 'P' for Put.

    Returns:
        A dictionary containing the calculated price and Greeks ('delta',
        'gamma', 'vega', 'theta'). Returns None if inputs are invalid (e.g.,
        time to expiration or sigma is zero or negative).
    """
    if T <= 0 or sigma <= 0:
        logging.warning(f"Invalid input for B-S model: T={T}, sigma={sigma}. Cannot price option.")
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = 0.0
    if option_type.upper() == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type.upper() == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
    else:
        logging.warning(f"Invalid option type for B-S model: {option_type}")
        return None

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type.upper() == 'C' else -d2)) / 365

    results = {"price": round(price, 4), "delta": round(delta, 4), "gamma": round(gamma, 4), "vega": round(vega, 4), "theta": round(theta, 4)}
    logging.info(f"Theoretical price calculated for {option_type} @ {K}: {results}")
    return results

async def get_position_details(ib: IB, position: Position) -> dict:
    """Determines the strategy type and key strikes from a position object.

    For a given `Position` object from `ib_insync`, this function identifies
    whether it is a single-leg option or part of a recognized combo strategy
    (e.g., Bull Call Spread, Iron Condor) by resolving its combo legs.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        position (Position): The position object to analyze.

    Returns:
        A dictionary with 'type' (e.g., 'SINGLE_LEG', 'BULL_CALL_SPREAD')
        and 'key_strikes' (a list of the primary strike prices). Returns
        'UNKNOWN' if the strategy cannot be identified.
    """
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}

    if isinstance(contract, FuturesOption):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(contract.strike)
        return details

    if not isinstance(contract, Bag):
        return details

    leg_contracts = []
    if not contract.comboLegs:
        return details

    # To get the right and strike, we need the full contract for each leg
    for leg in contract.comboLegs:
        try:
            leg_details = await ib.reqContractDetailsAsync(Contract(conId=leg.conId))
            if leg_details:
                leg_contracts.append(leg_details[0].contract)
        except Exception as e:
            logging.error(f"Could not request contract details for leg conId {leg.conId}: {e}")
            return details

    if len(leg_contracts) != len(contract.comboLegs):
        logging.warning(f"Could not resolve all legs for Bag contract {contract.conId}")
        return details

    # Now we can determine the strategy type
    actions = ''.join(sorted([leg.action[0] for leg in contract.comboLegs]))
    rights = ''.join(sorted([c.right for c in leg_contracts]))
    strikes = sorted([c.strike for c in leg_contracts])

    if len(leg_contracts) == 2:
        details['key_strikes'] = strikes
        if rights == 'CC' and actions == 'BS':
            details['type'] = 'BULL_CALL_SPREAD'
        elif rights == 'PP' and actions == 'BS':
            details['type'] = 'BEAR_PUT_SPREAD'
        elif rights == 'CP' and actions == 'BB':
            details['type'] = 'LONG_STRADDLE'
    elif len(leg_contracts) == 4:
        if rights == 'CCPP' and actions == 'BBSS':
            details['type'] = 'IRON_CONDOR'
            details['key_strikes'] = [strikes[1], strikes[2]]

    return details

def get_expiration_details(chain: dict, future_exp: str) -> dict | None:
    """Selects the best option expiration date for a given futures contract.

    The logic prefers the latest possible option expiration that occurs on or
    before the futures contract's expiration month. If none exist, it falls
    back to the nearest available expiration after today.

    Args:
        chain (dict): The option chain data from `build_option_chain`.
        future_exp (str): The expiration month of the future ('YYYYMM').

    Returns:
        A dictionary with 'exp_date', 'days_to_exp', and the list of 'strikes'
        for the chosen expiration. Returns None if no suitable expiration found.
    """
    valid_exp = [exp for exp in sorted(chain['expirations']) if exp[:6] <= future_exp]
    if not valid_exp:
        logging.warning(f"No option expiration on or before future {future_exp}. Using nearest available after today.")
        valid_exp = [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
        if not valid_exp:
            logging.error(f"No suitable option expirations found for future {future_exp}.")
            return None
        chosen_exp = valid_exp[0]
    else:
        chosen_exp = valid_exp[-1]

    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now(pytz.utc).date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}

async def _generate_position_id_from_trade(ib: IB, trade: Trade) -> str:
    """
    Generates a stable, canonical position identifier from a Trade object.
    It prioritizes the UUID stored in `orderRef`. If not present, it
    falls back to creating a sorted string of leg symbols for combos or the
    local symbol for single legs.
    """
    # Prioritize the unique orderRef if it exists and is a valid UUID string.
    if hasattr(trade.order, 'orderRef') and trade.order.orderRef and len(trade.order.orderRef) > 20:
        return trade.order.orderRef

    if isinstance(trade.contract, Bag) and trade.contract.comboLegs:
        leg_symbols = []
        leg_contracts = [Contract(conId=leg.conId) for leg in trade.contract.comboLegs]
        qualified_legs = await ib.qualifyContractsAsync(*leg_contracts)

        if qualified_legs:
            leg_symbols = sorted([c.localSymbol for c in qualified_legs])
            return "-".join(leg_symbols)
        else:
            return f"combo_{trade.order.permId}" # Fallback
    else:
        return trade.contract.localSymbol # For single-leg trades


async def log_trade_to_ledger(ib: IB, trade: Trade, reason: str = "Strategy Execution", specific_fill: Fill = None, combo_id: int = None, position_id: str = None):
    """Logs the details of a filled trade to the `trade_ledger.csv` file.

    For combo trades, this function logs each leg as a separate entry in the
    CSV, linked by a common `position_id` and `combo_id`.

    Args:
        ib (IB): The connected ib_insync instance.
        trade (Trade): The filled `ib_insync.Trade` object.
        reason (str): A string describing why the trade was executed.
        specific_fill (Fill, optional): If provided, log only this fill.
        combo_id (int, optional): The permanent ID of the parent combo order.
        position_id (str, optional): The stable identifier for the position.
    """
    # Use absolute path to ensure ledger is in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')

    fieldnames = [
        'timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity',
        'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
    ]

    rows_to_write = []
    final_combo_id = combo_id if combo_id is not None else trade.order.permId

    # Generate the position ID if it wasn't provided.
    final_position_id = position_id
    if not final_position_id:
        final_position_id = await _generate_position_id_from_trade(ib, trade)


    fills_to_log = [specific_fill] if specific_fill else trade.fills
    if not fills_to_log:
        logging.warning(f"Trade {trade.order.orderId} has no fills to log.")
        return

    for fill in fills_to_log:
        if not fill: continue

        # --- FIX 1: Do not log BAG contracts to ledger ---
        # IBKR Flex Queries only report the individual legs, not the combo container.
        if isinstance(fill.contract, Bag):
            logging.info(f"Skipping ledger entry for BAG contract {fill.contract.localSymbol} (legs will be logged individually).")
            continue

        contract = fill.contract

        # --- FIX 2: Force Qualification for Missing Symbols ---
        # If localSymbol is missing (common in single-leg closes), fetch it.
        if not contract.localSymbol or not hasattr(contract, 'right'):
            try:
                details = await ib.qualifyContractsAsync(contract)
                if details:
                    contract = details[0]
            except Exception as e:
                logging.warning(f"Could not qualify contract {contract.conId}: {e}")

        # Fallback to cache or simple check if qualification failed or wasn't needed
        if not hasattr(contract, 'right') or not hasattr(contract, 'strike'):
            cached_contract = ib.contracts.get(contract.conId)
            if cached_contract:
                logging.info(f"Found detailed contract for conId {contract.conId} in cache.")
                contract = cached_contract
            else:
                logging.warning(f"Could not find detailed contract for {contract.conId}. Ledger may be incomplete.")

        execution = fill.execution
        try:
            multiplier = float(contract.multiplier) if contract.multiplier else 37500.0
        except (ValueError, TypeError):
            multiplier = 37500.0

        total_value = (execution.price * execution.shares * multiplier) / 100.0
        action = 'BUY' if execution.side == 'BOT' else 'SELL'
        if action == 'BUY':
            total_value *= -1

        row = {
            'timestamp': execution.time.strftime('%Y-%m-%d %H:%M:%S'),
            'position_id': final_position_id,
            'combo_id': final_combo_id,
            'local_symbol': contract.localSymbol,
            'action': action,
            'quantity': execution.shares,
            'avg_fill_price': execution.price,
            'strike': contract.strike if hasattr(contract, 'strike') else 'N/A',
            'right': contract.right if hasattr(contract, 'right') else 'N/A',
            'total_value_usd': total_value,
            'reason': reason
        }
        rows_to_write.append(row)

    async with TRADE_LEDGER_LOCK:
        try:
            # CRITICAL: Check for file existence and content *inside* the lock
            # to prevent the TOCTOU race condition.
            try:
                file_exists_and_has_content = os.path.getsize(ledger_path) > 0
            except OSError:
                file_exists_and_has_content = False

            with open(ledger_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists_and_has_content:
                    writer.writeheader()
                writer.writerows(rows_to_write)
            logging.info(f"Logged {len(rows_to_write)} leg(s) to ledger for position_id {final_position_id} ({reason})")
        except Exception as e:
            logging.error(f"Error writing to trade ledger: {e}")


def archive_trade_ledger():
    """Archives the `trade_ledger.csv` file by moving it to the `archive_ledger` directory
    with a timestamp appended to its name.
    """
    ledger_filename = 'trade_ledger.csv'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, ledger_filename)

    if not os.path.exists(ledger_path):
        logging.info(f"'{ledger_filename}' not found, no action taken.")
        return

    archive_dir = os.path.join(base_dir, 'archive_ledger')
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        logging.info(f"Created archive directory at: {archive_dir}")

    # Format the current date and time to append to the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_filename = f'trade_ledger_{timestamp}.csv'
    archive_path = os.path.join(archive_dir, archive_filename)

    try:
        shutil.move(ledger_path, archive_path)
        logging.info(f"Successfully archived '{ledger_filename}' to '{archive_path}'")
    except Exception as e:
        logging.error(f"Failed to archive '{ledger_filename}': {e}")

def log_council_decision(decision_data):
    """
    Appends a row to 'data/council_history.csv' with the FULL details of the decision.
    Handles schema migration by archiving old files if headers mismatch.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, "council_history.csv")

    # Schema: Capture Sentiment (for stats) AND Summary (for reading) + entry_price
    fieldnames = [
        "timestamp", "contract", "entry_price",
        "ml_signal", "ml_confidence",
        "meteorologist_sentiment", "meteorologist_summary",
        "macro_sentiment", "macro_summary",
        "geopolitical_sentiment", "geopolitical_summary",
        "fundamentalist_sentiment", "fundamentalist_summary",
        "sentiment_sentiment", "sentiment_summary",
        "technical_sentiment", "technical_summary",
        "volatility_sentiment", "volatility_summary",
        "master_decision", "master_confidence", "master_reasoning",

        "prediction_type",
        "volatility_level",
        "strategy_type",

        "compliance_approved",
        "exit_price", "exit_timestamp", "pnl_realized", "actual_trend_direction",

        "volatility_outcome"
    ]

    # Check for schema mismatch and migrate if necessary
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)

            if header != fieldnames:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                legacy_path = os.path.join(data_dir, f"council_history_legacy_{timestamp}.csv")
                shutil.move(file_path, legacy_path)
                logging.info(f"Schema mismatch detected. Archived old history to {legacy_path}")
        except Exception as e:
            logging.error(f"Error checking CSV schema: {e}")

    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            # Filter data to match fieldnames and ensure timestamp is string
            row = {k: decision_data.get(k, '') for k in fieldnames}
            if isinstance(row.get('timestamp'), datetime):
                row['timestamp'] = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')

            writer.writerow(row)
            logging.info(f"Logged Council decision for {decision_data.get('contract')}")

    except Exception as e:
        logging.error(f"Failed to log council decision: {e}")

def is_market_open() -> bool:
    """Check if ICE coffee futures market is currently open.

    Based on ICE Coffee (KC) Trading Hours:
    - Electronic: Sun 6:00 PM - Fri 5:00 PM ET
    - Core Validation/Trading Hours: 03:30 AM - 02:00 PM ET

    This function checks:
    1. Weekends (Saturday/Sunday) -> CLOSED
    2. US Market Holidays (NYSE calendar) -> CLOSED
    3. Time of day (03:30 - 14:00 ET) -> OPEN if within window

    Returns:
        bool: True if market is currently open, False otherwise.
    """
    utc = pytz.UTC
    ny_tz = pytz.timezone('America/New_York')

    # Get current time in both UTC and NY
    now_utc = datetime.now(utc)
    now_ny = now_utc.astimezone(ny_tz)

    # 1. Check Weekend (Saturday=5, Sunday=6)
    if now_ny.weekday() >= 5:
        return False

    # 2. Check US Market Holidays (ICE follows NYSE calendar)
    # The 'market="NYSE"' parameter gives us the NYSE holiday calendar
    # which ICE coffee futures follow for closures
    us_holidays = holidays.US(years=now_ny.year, observed=True)
    # Also check financial market specific holidays
    try:
        # holidays >= 0.40 supports market parameter
        nyse_holidays = holidays.financial_holidays('NYSE', years=now_ny.year)
        if now_ny.date() in nyse_holidays:
            return False
    except (AttributeError, TypeError):
        # Fallback for older holidays versions
        pass

    if now_ny.date() in us_holidays:
        return False

    # 3. Time Check (03:30 AM - 02:00 PM ET)
    # Define thresholds in NY timezone
    market_open_ny = now_ny.replace(hour=3, minute=30, second=0, microsecond=0)
    market_close_ny = now_ny.replace(hour=14, minute=0, second=0, microsecond=0)

    # Convert to UTC for comparison
    market_open_utc = market_open_ny.astimezone(utc)
    market_close_utc = market_close_ny.astimezone(utc)

    return market_open_utc <= now_utc <= market_close_utc


def is_trading_day() -> bool:
    """Check if today is a trading day (weekday + not a holiday).

    Use this for 24/7 sentinels that should still run on trading days
    but outside of core market hours (e.g., overnight weather monitoring).

    Returns:
        bool: True if today is a trading day, False on weekends/holidays.
    """
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)

    # Check weekend
    if now_ny.weekday() >= 5:
        return False

    # Check holidays
    us_holidays = holidays.US(years=now_ny.year, observed=True)
    if now_ny.date() in us_holidays:
        return False

    return True
