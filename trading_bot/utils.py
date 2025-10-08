import csv
import logging
import os
from datetime import datetime
import pytz
from ib_insync import *
import numpy as np
from scipy.stats import norm


def normalize_strike(strike: float) -> float:
    """Normalizes the strike price if it's magnified by 100."""
    if strike > 100:
        return strike / 100.0
    return strike


def price_option_black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> dict | None:
    """Calculates the theoretical price and Greeks of an option using the Black-Scholes model."""
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
    """Determines the strategy type from a position object by resolving combo legs."""
    contract = position.contract
    details = {'type': 'UNKNOWN', 'key_strikes': []}

    if isinstance(contract, FuturesOption):
        details['type'] = 'SINGLE_LEG'
        details['key_strikes'].append(normalize_strike(contract.strike))
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
    strikes = sorted([normalize_strike(c.strike) for c in leg_contracts])

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

def is_market_open(contract_details, exchange_timezone_str: str):
    if not contract_details or not contract_details.liquidHours:
        return False
    tz = pytz.timezone(exchange_timezone_str)
    now_tz = datetime.now(tz)
    for session_str in contract_details.liquidHours.split(';'):
        if 'CLOSED' in session_str:
            continue
        try:
            start_str, end_str = session_str.split('-')

            # Create timezone-aware start and end times for the session
            session_start_dt = tz.localize(datetime.strptime(start_str, '%Y%m%d:%H%M'))

            # Handle cases where end_str might not have a date (e.g., '1600')
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
    # Find expirations that are on or before the future's expiration month.
    valid_exp = [exp for exp in sorted(chain['expirations']) if exp[:6] <= future_exp]

    if not valid_exp:
        logging.warning(f"No option expiration on or before future {future_exp}. Using nearest available after today.")
        # If none found, find the first available expiration after today.
        valid_exp = [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
        if not valid_exp:
            logging.error(f"No suitable option expirations found for future {future_exp}.")
            return None
        # Take the earliest one available.
        chosen_exp = valid_exp[0]
    else:
        # Of the valid expirations, take the latest one.
        chosen_exp = valid_exp[-1]

    days = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
    return {'exp_date': chosen_exp, 'days_to_exp': days, 'strikes': chain['strikes_by_expiration'][chosen_exp]}

def log_trade_to_ledger(trade: Trade, reason: str = "Strategy Execution"):
    if trade.orderStatus.status != OrderStatus.Filled: return
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action, 'quantity': trade.orderStatus.filled, 'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * (100 if isinstance(trade.contract, Bag) else 37500),
        'order_id': trade.order.orderId, 'reason': reason
    }
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv'), 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not os.path.isfile(f.name): writer.writeheader()
            writer.writerow(row)
        logging.info(f"Logged trade to ledger ({reason}): {row['action']} {row['quantity']} @ {row['avg_fill_price']}")
    except Exception as e:
        logging.error(f"Error writing to trade ledger: {e}")