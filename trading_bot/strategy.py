"""Implements the specific option trading strategies for the bot.

This module contains the logic for constructing different types of option
spreads based on the trading signals generated from the API predictions.
It includes functions for both directional (bullish/bearish) and
volatility-based strategies.
"""

import logging
import math
import numpy as np
from ib_insync import *

from trading_bot.logging_config import setup_logging
from trading_bot.utils import get_expiration_details


def find_closest_strike(
    target: float,
    strikes: list,
    chain: dict = None,
    min_open_interest: int = 10,
    right: str = None
) -> float | None:
    """
    Find closest strike to target with optional liquidity filtering.

    v3.1: Added open interest check to avoid illiquid strikes.

    Args:
        target: Target strike price
        strikes: List of available strikes
        chain: Option chain dict (optional, for OI lookup)
        min_open_interest: Minimum OI threshold (default 10)
        right: 'C' or 'P' for OI lookup (required if chain provided)

    Returns:
        Closest liquid strike, or closest strike if no chain provided
    """
    if not strikes:
        return None

    # Sort by distance from target
    sorted_strikes = sorted(strikes, key=lambda x: abs(x - target))

    # If no chain provided, use pure distance (backward compatible)
    if chain is None:
        return sorted_strikes[0]

    # Check liquidity for each strike in order of distance
    for strike in sorted_strikes:
        oi = _get_strike_open_interest(chain, strike, right)

        if oi >= min_open_interest:
            if strike != sorted_strikes[0]:
                logging.info(
                    f"Skipped illiquid strike {sorted_strikes[0]} (OI={_get_strike_open_interest(chain, sorted_strikes[0], right)}), "
                    f"using {strike} (OI={oi})"
                )
            return strike
        else:
            logging.debug(f"Strike {strike} has low OI ({oi}), checking next...")

    # Fallback: use closest even if illiquid (with warning)
    logging.warning(
        f"No strikes near {target} meet min OI threshold ({min_open_interest}). "
        f"Using closest strike {sorted_strikes[0]} anyway."
    )
    return sorted_strikes[0]


def _get_strike_open_interest(chain: dict, strike: float, right: str) -> int:
    """
    Get open interest for a specific strike from the chain.

    Args:
        chain: Option chain dict with 'calls' and 'puts' keys
        strike: Strike price to look up
        right: 'C' for calls, 'P' for puts

    Returns:
        Open interest (0 if not found)
    """
    try:
        options_list = chain.get('calls' if right == 'C' else 'puts', [])

        for opt in options_list:
            # Handle both dict and object formats
            opt_strike = opt.get('strike') if isinstance(opt, dict) else getattr(opt, 'strike', None)
            if opt_strike and abs(opt_strike - strike) < 0.01:
                oi = opt.get('openInterest') if isinstance(opt, dict) else getattr(opt, 'openInterest', 0)
                return int(oi) if oi else 0

        return 0
    except Exception as e:
        logging.debug(f"Could not get OI for {strike}{right}: {e}")
        return 0


def find_strike_by_delta(
    chain: dict,
    target_delta: float,
    right: str,
    underlying_price: float
) -> float | None:
    """
    Find strike closest to target delta.

    v3.1: Delta-based strike selection for consistent risk profiles.

    Args:
        chain: Option chain with Greeks
        target_delta: Absolute delta value (e.g., 0.16 for 16-delta)
        right: 'C' for calls, 'P' for puts
        underlying_price: Current underlying price for fallback

    Returns:
        Strike price closest to target delta
    """
    options_list = chain.get('calls' if right == 'C' else 'puts', [])

    if not options_list:
        logging.warning(f"No {right} options in chain. Using ATM as fallback.")
        return underlying_price

    best_strike = None
    best_delta_diff = float('inf')

    for opt in options_list:
        # Extract delta (handle both dict and object formats)
        if isinstance(opt, dict):
            delta = abs(opt.get('delta', 0))
            strike = opt.get('strike', 0)
        else:
            delta = abs(getattr(opt, 'delta', 0) or 0)
            strike = getattr(opt, 'strike', 0)

        if delta == 0 or strike == 0:
            continue

        delta_diff = abs(delta - target_delta)
        if delta_diff < best_delta_diff:
            best_delta_diff = delta_diff
            best_strike = strike

    if best_strike is None:
        logging.warning(f"Could not find {target_delta:.0%} delta {right}. Using distance-based fallback.")
        # Fallback to index-based
        return None

    logging.info(f"Found {target_delta:.0%} delta {right} at strike {best_strike}")
    return best_strike


def _chain_has_greeks(chain: dict) -> bool:
    """Check if chain has delta values."""
    for opt in chain.get('calls', [])[:5]:
        delta = opt.get('delta') if isinstance(opt, dict) else getattr(opt, 'delta', None)
        if delta is not None and delta != 0:
            return True
    return False


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
    # === CRITICAL: NaN Price Guard ===
    if underlying_price is None or math.isnan(underlying_price) or underlying_price <= 0:
        logging.error(
            f"ABORT: Invalid underlying_price={underlying_price} for {future_contract.localSymbol}. "
            f"Cannot calculate strikes. Market may be closed or data unavailable."
        )
        return None

    logging.info(f"--- Defining {signal['direction']} Spread for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    spread_width_pct = tuning.get('spread_width_percentage', 0.15)
    spread_width_points = underlying_price * spread_width_pct

    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details:
        return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes, chain=chain, right='C' if signal['direction'] == 'BULLISH' else 'P')
    if atm_strike is None:
        logging.error("Could not find ATM strike in the chain.")
        return None

    logging.info(f"Identified ATM strike: {atm_strike} for underlying price {underlying_price}")

    legs_def, order_action = [], ''
    if signal['direction'] == 'BULLISH':
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike + spread_width_points
        short_leg_strike = find_closest_strike(
            target_short_leg_strike,
            [s for s in strikes if s > long_leg_strike],
            chain=chain,
            right='C'
        )
        if short_leg_strike is None:
            logging.warning(f"Strategy definition failed: Could not find suitable short strike near {target_short_leg_strike} for {future_contract.localSymbol}")
            return None
        legs_def, order_action = [('C', 'BUY', long_leg_strike), ('C', 'SELL', short_leg_strike)], 'BUY'
        logging.info(f"Defined Bull Call Spread legs: BUY {long_leg_strike}C, SELL {short_leg_strike}C")

    else:  # BEARISH
        long_leg_strike = atm_strike
        target_short_leg_strike = long_leg_strike - spread_width_points
        short_leg_strike = find_closest_strike(
            target_short_leg_strike,
            [s for s in strikes if s < long_leg_strike],
            chain=chain,
            right='P'
        )
        if short_leg_strike is None:
            logging.warning(f"Strategy definition failed: Could not find suitable short strike near {target_short_leg_strike} for {future_contract.localSymbol}")
            return None
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


def validate_iron_condor_risk(
    max_loss_per_position: float,
    account_equity: float,
    max_risk_pct: float = 0.02  # 2% of equity max
) -> bool:
    """
    Validates that Iron Condor max loss is within acceptable bounds.
    The wings ARE the catastrophe protection - we just need proper sizing.
    """
    max_acceptable_loss = account_equity * max_risk_pct

    if max_loss_per_position > max_acceptable_loss:
        logging.warning(
            f"Iron Condor max loss ${max_loss_per_position:.2f} exceeds "
            f"{max_risk_pct:.0%} of equity (${max_acceptable_loss:.2f}). Reducing size or rejecting."
        )
        return False

    return True


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
    # === CRITICAL: NaN Price Guard ===
    if underlying_price is None or math.isnan(underlying_price) or underlying_price <= 0:
        logging.error(
            f"ABORT: Invalid underlying_price={underlying_price} for {future_contract.localSymbol}. "
            f"Cannot calculate strikes. Market may be closed or data unavailable."
        )
        return None

    logging.info(f"--- Defining {signal['level']} Volatility Strategy for {future_contract.localSymbol} ---")
    tuning = config.get('strategy_tuning', {})
    exp_details = get_expiration_details(chain, future_contract.lastTradeDateOrContractMonth)
    if not exp_details: return None

    strikes = exp_details['strikes']
    atm_strike = find_closest_strike(underlying_price, strikes, chain=chain, right='C') # ATM usually has C and P
    if atm_strike is None:
        logging.warning(f"Strategy definition failed: Could not find ATM strike for {future_contract.localSymbol} near {underlying_price}")
        return None

    legs_def, order_action = [], ''
    if signal['level'] == 'HIGH':  # Long Straddle
        legs_def, order_action = [('C', 'BUY', atm_strike), ('P', 'BUY', atm_strike)], 'BUY'

    elif signal['level'] == 'LOW':  # Iron Condor
        # v3.1: Use delta-based strikes if available
        use_delta = tuning.get('iron_condor_use_delta', True)
        sell_delta = tuning.get('iron_condor_sell_delta', 0.16)  # 16-delta shorts
        buy_delta = tuning.get('iron_condor_buy_delta', 0.05)    # 5-delta longs

        if use_delta and _chain_has_greeks(chain):
            # Find strikes by delta
            short_put = find_strike_by_delta(chain, sell_delta, 'P', underlying_price)
            long_put = find_strike_by_delta(chain, buy_delta, 'P', underlying_price)
            short_call = find_strike_by_delta(chain, sell_delta, 'C', underlying_price)
            long_call = find_strike_by_delta(chain, buy_delta, 'C', underlying_price)

            if all([short_put, long_put, short_call, long_call]):
                # Validate wing order
                if long_put < short_put < short_call < long_call:
                    legs_def = [
                        ('P', 'BUY', long_put),
                        ('P', 'SELL', short_put),
                        ('C', 'SELL', short_call),
                        ('C', 'BUY', long_call)
                    ]
                    order_action = 'SELL'

                    logging.info(
                        f"Delta-based Iron Condor: {long_put}P/{short_put}P/{short_call}C/{long_call}C "
                        f"(~{sell_delta:.0%} delta shorts, ~{buy_delta:.0%} delta wings)"
                    )

                    return {
                        "action": order_action,
                        "legs_def": legs_def,
                        "exp_details": exp_details,
                        "chain": chain,
                        "underlying_price": underlying_price,
                        "future_contract": future_contract
                    }

        # Fallback to index-based (existing logic)
        logging.info("Using index-based Iron Condor (no Greeks available or delta logic failed)")
        short_dist = int(tuning.get('iron_condor_short_strikes_from_atm', 2))
        wing_width = int(tuning.get('iron_condor_wing_strikes_apart', 2))

        # Sanity check
        if short_dist < 1 or wing_width < 1:
            logging.error(f"Invalid Iron Condor params: short_dist={short_dist}, wing_width={wing_width}. Must be >= 1.")
            return None

        try:
            atm_idx = strikes.index(atm_strike)
            if not (atm_idx - short_dist - wing_width >= 0 and atm_idx + short_dist + wing_width < len(strikes)):
                logging.warning(f"Strategy definition failed: Iron Condor wings out of bounds for {future_contract.localSymbol}")
                return None
            s = {'lp': strikes[atm_idx - short_dist - wing_width], 'sp': strikes[atm_idx - short_dist], 'sc': strikes[atm_idx + short_dist], 'lc': strikes[atm_idx + short_dist + wing_width]}
            legs_def, order_action = [('P', 'BUY', s['lp']), ('P', 'SELL', s['sp']), ('C', 'SELL', s['sc']), ('C', 'BUY', s['lc'])], 'SELL'
        except ValueError:
            logging.warning(f"Strategy definition failed: ValueError in Iron Condor setup for {future_contract.localSymbol}")
            return None

    return {
        "action": order_action,
        "legs_def": legs_def,
        "exp_details": exp_details,
        "chain": chain,
        "underlying_price": underlying_price,
        "future_contract": future_contract
    }
