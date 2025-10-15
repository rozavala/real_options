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
from trading_bot.utils import get_expiration_details, price_option_black_scholes

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


def check_min_profit_threshold(strategy_def: dict, config: dict) -> bool:
    """
    Checks if a strategy's theoretical profit meets a minimum threshold after
    accounting for estimated transaction costs (spread and slippage).

    Args:
        strategy_def (dict): The dictionary defining the strategy, including legs.
        config (dict): The application's configuration dictionary.

    Returns:
        True if the strategy is profitable enough, False otherwise.
    """
    tuning = config.get('strategy_tuning', {})
    min_profit_threshold = tuning.get('minimum_profit_threshold', 0.0)
    risk_free_rate = tuning.get('risk_free_rate', 0.02)
    default_vol = tuning.get('default_volatility', 0.20)
    slippage_pct = tuning.get('slippage_spread_percentage', 0.5)

    total_theoretical_value = 0
    total_transaction_cost = 0

    time_to_exp_years = strategy_def['exp_details']['days_to_exp'] / 365.0

    for right, action, strike in strategy_def['legs_def']:
        option_key = f"{strike}:{right}"
        option_details = strategy_def['chain'].get(option_key)

        if not option_details or not option_details.ticker:
            logging.warning(f"No ticker for {option_key}, cannot check profit.")
            return False

        ticker = option_details.ticker
        iv = ticker.impliedVolatility or default_vol

        bs_price_result = price_option_black_scholes(
            S=strategy_def['underlying_price'],
            K=strike,
            T=time_to_exp_years,
            r=risk_free_rate,
            sigma=iv,
            option_type=right
        )

        if not bs_price_result:
            logging.warning(f"Black-Scholes calculation failed for {option_key}.")
            return False

        theoretical_price = bs_price_result['price']

        # Correctly calculate theoretical value for debit and credit spreads
        if action == 'BUY':
            total_theoretical_value -= theoretical_price
        else:  # SELL
            total_theoretical_value += theoretical_price

        # Calculate transaction costs for this leg
        bid_ask_spread = ticker.ask - ticker.bid if ticker.ask and ticker.bid else 0
        slippage = bid_ask_spread * slippage_pct
        total_transaction_cost += (bid_ask_spread / 2) + slippage

    # For a credit spread, we receive a credit. For a debit spread, we pay a debit.
    # The net profit is the credit received minus costs, or the debit paid plus costs.
    if strategy_def['action'] == 'SELL': # Credit Spread
        net_profit = total_theoretical_value - total_transaction_cost
    else: # Debit Spread
        net_profit = abs(total_theoretical_value) - total_transaction_cost

    logging.info(
        f"Profit Check: BS Value={total_theoretical_value:.4f}, "
        f"Costs={total_transaction_cost:.4f}, Net={net_profit:.4f}"
    )

    if net_profit < min_profit_threshold:
        logging.warning(
            f"Strategy failed profit check. Net Profit: {net_profit:.4f}, "
            f"Min Threshold: {min_profit_threshold}"
        )
        return False

    return True


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
    spread_width_points = tuning.get('spread_width_points', 15.0)

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

    strategy_def = {
        "action": order_action,
        "legs_def": legs_def,
        "exp_details": exp_details,
        "chain": chain,
        "underlying_price": underlying_price,
        "future_contract": future_contract
    }

    # Perform the minimum profit check before finalizing the strategy
    if not check_min_profit_threshold(strategy_def, config):
        return None

    return strategy_def


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

    strategy_def = {
        "action": order_action,
        "legs_def": legs_def,
        "exp_details": exp_details,
        "chain": chain,
        "underlying_price": underlying_price,
        "future_contract": future_contract
    }

    # Perform the minimum profit check before finalizing the strategy
    if not check_min_profit_threshold(strategy_def, config):
        return None

    return strategy_def