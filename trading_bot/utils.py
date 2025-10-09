"""A collection of utility functions for the trading bot.

This module provides various helper functions used across the trading bot,
including mathematical calculations for option pricing (Black-Scholes),
helpers for normalizing and parsing contract data from Interactive Brokers,
market hour checks, and trade logging utilities.
"""

import csv
import logging
import os
from datetime import datetime
import pytz
from ib_insync import *
import numpy as np
from scipy.stats import norm

from logging_config import setup_logging

# --- Logging Setup ---
setup_logging()


def normalize_strike(strike: float) -> float:
    """Normalizes the strike price if it appears to be magnified by 100.

    Some data feeds from Interactive Brokers may return option strike prices
    as integers (e.g., 350.0 for a 3.5 strike). This function corrects such
    values by dividing them by 100 if they exceed a certain threshold.

    Args:
        strike (float): The strike price to normalize.

    Returns:
        The normalized strike price.
    """
    if strike > 100:
        return strike / 100.0
    return strike


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

def is_market_open(contract_details, exchange_timezone_str: str) -> bool:
    """Checks if the market for a given contract is currently open.

    It parses the 'liquidHours' string from the contract details to determine
    if the current time falls within any of the active trading sessions.

    Args:
        contract_details: The contract details object from ib_insync, which
            must contain the `liquidHours` attribute.
        exchange_timezone_str (str): The IANA timezone string for the
            contract's exchange (e.g., 'America/New_York').

    Returns:
        True if the market is open, False otherwise.
    """
    if not contract_details or not contract_details.liquidHours:
        return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str:
            continue
        try:
            start_str, end_str = session_str.split('-')
            session_start_dt = tz.localize(datetime.strptime(start_str, '%Y%m%d:%H%M'))
            if ':' in end_str:
                session_end_dt = tz.localize(datetime.strptime(end_str, '%Y%m%d:%H%M'))
            else:
                session_end_dt = session_start_dt.replace(
                    hour=int(end_str[:2]), minute=int(end_str[2:])
                )

            if session_start_dt <= now_tz < session_end_dt:
                logging.info("Market is open.")
                return True
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not parse liquid hours segment '{session_str}': {e}")
            continue
    logging.info("Market is closed.")
    return False

def calculate_wait_until_market_open(contract_details, exchange_timezone_str: str) -> float:
    """Calculates the time in seconds until the next market open.

    Parses the 'liquidHours' to find the start time of the next trading
    session and calculates the number of seconds to wait from now.

    Args:
        contract_details: The contract details object from ib_insync.
        exchange_timezone_str (str): The IANA timezone for the exchange.

    Returns:
        The number of seconds to wait until the next session opens. Returns
        3600 seconds (1 hour) as a default if the next open cannot be determined.
    """
    if not contract_details or not contract_details.liquidHours: return 3600
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    next_open_dt = None
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str: continue
        try:
            session_start_dt = tz.localize(datetime.strptime(session_str.split('-')[0], '%Y%m%d:%H%M'))
            if session_start_dt > now_tz and (next_open_dt is None or session_start_dt < next_open_dt):
                next_open_dt = session_start_dt
        except (ValueError, IndexError): continue
    if next_open_dt:
        wait_seconds = (next_open_dt - now_tz).total_seconds() + 60
        logging.info(f"Next market open: {next_open_dt:%Y-%m-%d %H:%M:%S}. Waiting {wait_seconds / 3600:.2f} hours.")
        return wait_seconds
    return 3600

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

    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}

def log_trade_to_ledger(trade: Trade, reason: str = "Strategy Execution"):
    """Logs the details of a filled trade to the `trade_ledger.csv` file.

    For combo trades, this function logs each leg as a separate entry in the
    CSV, linked by a common `combo_id` (the order's permId). This allows for
    detailed P&L analysis of multi-leg strategies.

    Args:
        trade (Trade): The filled `ib_insync.Trade` object.
        reason (str): A string describing why the trade was executed (e.g.,
            'Strategy Execution', 'Position Misaligned', 'Stop-Loss').
    """
    if trade.orderStatus.status != OrderStatus.Filled:
        return

    ledger_path = 'trade_ledger.csv'
    file_exists = os.path.isfile(ledger_path)

    fieldnames = [
        'timestamp', 'combo_id', 'local_symbol', 'action', 'quantity',
        'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
    ]

    rows_to_write = []
    combo_id = trade.order.permId

    if not trade.fills:
        logging.warning(f"Trade {trade.order.orderId} is Filled but has no fills to log.")
        return

    for fill in trade.fills:
        contract = fill.contract
        execution = fill.execution
        try:
            multiplier = float(contract.multiplier) if contract.multiplier else 37500.0
        except (ValueError, TypeError):
            multiplier = 37500.0

        total_value = execution.price * execution.shares * multiplier
        action = 'BUY' if execution.side == 'BOT' else 'SELL'

        row = {
            'timestamp': execution.time.strftime('%Y-%m-%d %H:%M:%S'),
            'combo_id': combo_id,
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

    try:
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)
        logging.info(f"Logged {len(rows_to_write)} leg(s) to ledger for combo_id {combo_id} ({reason})")
    except Exception as e:
        logging.error(f"Error writing to trade ledger: {e}")
