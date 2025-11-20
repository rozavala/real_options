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


def define_directional_strategy(config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> dict | None:
    """Constructs the definition for a directional option spread.

    Based on the signal's direction ('BULLISH' or 'BEARISH'), this function
    defines a two-leg vertical spread. It does not place an order.

    Args:
        config (dict): The application configuration dictionary.
        signal (dict): The trading signal, containing the 'direction'.
        chain (dict): The option chain for the underlying future.
        underlying_price (float): The current price of the underlying future.
        future_contract (Contract): The underlying future contract object.

    Returns:
        A dictionary defining the strategy, or None if it cannot be constructed.
    """
    logging.info(f"--- Defining {signal['direction']} Spread for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    spread_width_pct = tuning.get('spread_width_percentage', 0.15)
    spread_width_points = underlying_price * spread_width_pct

    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details:
        return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes)
    if atm_strike is None:
        logging.error("Could not find ATM strike in the chain.")
        return None

    logging.info(f"Identified ATM strike: {atm_strike} for underlying price {underlying_price}")

    legs_def, order_action = [], ''
    if signal['direction'] == 'BULLISH':
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike + spread_width_points
        short_leg_strike = find_closest_strike(target_short_leg_strike, [s for s in strikes if s > long_leg_strike])
        if short_leg_strike is None: return None
        legs_def, order_action = [('C', 'BUY', long_leg_strike), ('C', 'SELL', short_leg_strike)], 'BUY'
        logging.info(f"Defined Bull Call Spread legs: BUY {long_leg_strike}C, SELL {short_leg_strike}C")

    else:  # BEARISH
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike - spread_width_points
        short_leg_strike = find_closest_strike(target_short_leg_strike, [s for s in strikes if s < long_leg_strike])
        if short_leg_strike is None: return None
        legs_def, order_action = [('P', 'BUY', long_leg_strike), ('P', 'SELL', short_leg_strike)], 'BUY'
        logging.info(f"Defined Bear Put Spread legs: BUY {long_leg_strike}P, SELL {short_leg_strike}P")

    return {
        "action": order_action,
        "legs_def": legs_def,
        "exp_details": exp_details,
        "chain": chain,
        "underlying_price": underlying_price,
        "future_contract": future_contract
    }


def define_volatility_strategy(config: dict, signal: dict, chain: dict, underlying_price: float, future_contract: Contract) -> dict | None:
    """Constructs the definition for a volatility-based option strategy.

    This function defines the parameters for a volatility trade but does not
    place an order.

    Args:
        config (dict): The application configuration dictionary.
        signal (dict): The trading signal ('HIGH'/'LOW' volatility).
        chain (dict): The option chain for the underlying future.
        underlying_price (float): The current price of the underlying future.
        future_contract (Contract): The underlying future contract object.

    Returns:
        A dictionary defining the strategy, or None if it cannot be constructed.
    """
    logging.info(f"--- Defining {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes)
    if atm_strike is None: return None

    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH':  # Long Straddle
        legs_def, order_action = [('C', 'BUY', atm_strike), ('P', 'BUY', atm_strike)], 'BUY'
    elif signal['level'] == 'LOW':  # Iron Condor
        short_dist = tuning.get('iron_condor_short_strikes_from_atm', 2)
        wing_width = tuning.get('iron_condor_wing_strikes_apart', 2)
        try:
            atm_idx = strikes.index(atm_strike)
            if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)): return None
            s = {'lp': strikes[atm_idx - short_dist - wing_width], 'sp': strikes[atm_idx - short_dist], 'sc': strikes[atm_idx + short_dist], 'lc': strikes[atm_idx + short_dist + wing_width]}
            legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'
        except ValueError: return None

    return {
        "action": order_action,
        "legs_def": legs_def,
        "exp_details": exp_details,
        "chain": chain,
        "underlying_price": underlying_price,
        "future_contract": future_contract
    }