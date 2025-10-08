"""Implements the specific option trading strategies for the bot.

This module contains the logic for constructing different types of option
spreads based on the trading signals generated from the API predictions.
It includes functions for both directional (bullish/bearish) and
volatility-based strategies.
"""

import logging
import numpy as np
from ib_insync import *

from logging_config import setup_logging
from trading_bot.ib_interface import place_combo_order
from trading_bot.utils import get_expiration_details

# --- Logging Setup ---
setup_logging()


def find_closest_strike(target_strike: float, available_strikes: list[float]) -> float | None:
    """Finds the strike in a list that is closest to a target value.

    Args:
        target_strike (float): The target strike price.
        available_strikes (list[float]): A list of available strike prices.

    Returns:
        The strike from the list that is numerically closest to the target.
        Returns None if the list of available strikes is empty.
    """
    if not available_strikes:
        return None
    return min(available_strikes, key=lambda s: abs(s - target_strike))


async def execute_directional_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    """Constructs and executes a directional option spread (Bull Call or Bear Put).

    Based on the signal's direction ('BULLISH' or 'BEARISH'), this function
    builds a two-leg vertical spread.
    - For a 'BULLISH' signal, it creates a Bull Call Spread by buying an
      at-the-money (ATM) call and selling an out-of-the-money (OTM) call.
    - For a 'BEARISH' signal, it creates a Bear Put Spread by buying an
      ATM put and selling an OTM put.

    The width of the spread is determined by the `spread_width_usd` parameter
    in the configuration.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        signal (dict): The trading signal, containing the 'direction'.
        chain (dict): The option chain for the underlying future.
        underlying_price (float): The current price of the underlying future.
        future_contract (Contract): The underlying future contract object.

    Returns:
        The `Trade` object for the placed combo order, or None if the
        strategy cannot be constructed (e.g., no suitable strikes found).
    """
    logging.info(f"--- Executing {signal['direction']} Spread for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    spread_width_usd = tuning.get('spread_width_usd', 0.05)

    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details:
        return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes)
    if atm_strike is None:
        logging.error("Could not find ATM strike in the chain.")
        return None

    logging.info(f"Identified ATM strike: {atm_strike} for underlying price {underlying_price}")

    legs_def = []
    if signal['direction'] == 'BULLISH':
        # Bull Call Spread: Buy ATM call, Sell OTM call
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike + spread_width_usd
        short_leg_strike = find_closest_strike(target_short_leg_strike, [s for s in strikes if s > long_leg_strike])

        if short_leg_strike is None:
            logging.warning(f"Could not find a suitable short leg for Bull Call Spread with ATM strike {long_leg_strike}.")
            return None

        legs_def = [('C', 'BUY', long_leg_strike), ('C', 'SELL', short_leg_strike)]
        logging.info(f"Defined Bull Call Spread legs: BUY {long_leg_strike}C, SELL {short_leg_strike}C")

    else:  # BEARISH
        # Bear Put Spread: Buy ATM put, Sell OTM put
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike - spread_width_usd
        short_leg_strike = find_closest_strike(target_short_leg_strike, [s for s in strikes if s < long_leg_strike])

        if short_leg_strike is None:
            logging.warning(f"Could not find a suitable short leg for Bear Put Spread with ATM strike {long_leg_strike}.")
            return None

        legs_def = [('P', 'BUY', long_leg_strike), ('P', 'SELL', short_leg_strike)]
        logging.info(f"Defined Bear Put Spread legs: BUY {long_leg_strike}P, SELL {short_leg_strike}P")

    return await place_combo_order(ib, config, 'BUY', legs_def, exp_details, chain, underlying_price)


async def execute_volatility_strategy(ib: IB, config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> Trade | None:
    """Constructs and executes a volatility-based option strategy.

    This function is a placeholder for volatility strategies. Based on the
    signal, it could execute trades like a Long Straddle or an Iron Condor.
    - 'HIGH' volatility signal: Builds a Long Straddle (Buy ATM Call + Buy ATM Put).
    - 'LOW' volatility signal: Builds an Iron Condor.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        signal (dict): The trading signal, containing the 'level' ('HIGH'/'LOW').
        chain (dict): The option chain for the underlying future.
        underlying_price (float): The current price of the underlying future.
        future_contract (Contract): The underlying future contract object.

    Returns:
        The `Trade` object for the placed combo order, or None if the
        strategy cannot be constructed.
    """
    logging.info(f"--- Executing {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details:
        return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes)
    if atm_strike is None:
        logging.error("Could not find ATM strike in the chain.")
        return None

    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH':  # Long Straddle
        legs_def, order_action = [('C', 'BUY', atm_strike), ('P', 'BUY', atm_strike)], 'BUY'
    elif signal['level'] == 'LOW':  # Iron Condor
        short_dist = tuning.get('iron_condor_short_strikes_from_atm', 2)
        wing_width = tuning.get('iron_condor_wing_strikes_apart', 2)

        try:
            atm_idx = strikes.index(atm_strike)
            if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)):
                logging.warning("Not enough strikes available for Iron Condor.")
                return None

            s = {
                'lp': strikes[atm_idx - short_dist - wing_width],
                'sp': strikes[atm_idx - short_dist],
                'sc': strikes[atm_idx + short_dist],
                'lc': strikes[atm_idx + short_dist + wing_width]
            }
            legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'
        except ValueError:
            logging.error(f"ATM strike {atm_strike} not found in index. Cannot place Iron Condor.")
            return None

    return await place_combo_order(ib, config, order_action, legs_def, exp_details, chain, underlying_price)