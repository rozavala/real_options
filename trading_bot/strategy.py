import logging

from ib_insync import *

from trading_bot.ib_interface import place_combo_order
from trading_bot.utils import get_expiration_details


async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    strategy_params, tuning = config['strategy'], config.get('strategy_tuning', {})
    spread_width = tuning.get('vertical_spread_strikes_apart', 2)
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None

    if signal['direction'] == 'BULLISH':
        if atm_idx + spread_width >= len(strikes): logging.warning("Not enough strikes for Bull Call Spread."); return None
        legs_def = [('C', 'BUY', strikes[atm_idx]), ('C', 'SELL', strikes[atm_idx + spread_width])]
    else: # BEARISH
        if atm_idx - spread_width < 0: logging.warning("Not enough strikes for Bear Put Spread."); return None
        legs_def = [('P', 'BUY', strikes[atm_idx]), ('P', 'SELL', strikes[atm_idx - spread_width])]

    return await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, underlying_price)


async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    logging.info(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None
    strikes = exp_details['strikes']
    try:
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying_price))
    except (ValueError, IndexError):
        logging.error("Could not find ATM strike in the chain."); return None

    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH': # Long Straddle
        legs_def, order_action = [('C', 'BUY', strikes[atm_idx]), ('P', 'BUY', strikes[atm_idx])], 'BUY'
    elif signal['level'] == 'LOW': # Iron Condor
        short_dist, wing_width = tuning.get('iron_condor_short_strikes_from_atm', 2), tuning.get('iron_condor_wing_strikes_apart', 2)
        if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)):
            logging.warning("Not enough strikes for Iron Condor."); return None
        s = { 'lp': strikes[atm_idx - short_dist - wing_width], 'sp': strikes[atm_idx - short_dist],
              'sc': strikes[atm_idx + short_dist], 'lc': strikes[atm_idx + short_dist + wing_width] }
        legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'

    return await place_combo_order(ib, config, order_action, legs_def, exp_details, chain, underlying_price)