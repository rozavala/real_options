"""Functions for interacting with the Interactive Brokers (IB) TWS or Gateway.

This module abstracts the `ib_insync` library calls for operations specific
to this trading bot, such as fetching contracts, building option chains,
placing complex combo orders, and managing the order lifecycle.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta

from ib_insync import *
from ib_insync import Order # Explicit import to fix NameError in type hints

from trading_bot.logging_config import setup_logging
from trading_bot.utils import (
    price_option_black_scholes, log_trade_to_ledger, round_to_tick,
    get_tick_size, get_contract_multiplier
)


async def get_option_market_data(ib: IB, contract: Contract, underlying_future: Contract) -> dict | None:
    """
    Fetches live market data for a single option contract, including bid, ask,
    and implied volatility.
    """
    logging.info(f"Fetching market data for option: {contract.localSymbol}")
    # Generic tick list 106 provides model-based option greeks (modelOptionImpliedVol)
    # FIX: Remove invalid generic ticks 104 and 24. Use only 106 (Implied Vol).
    ticker = ib.reqMktData(contract, '106', False, False)
    try:
        await asyncio.sleep(2)  # Allow time for data to arrive

        # Extract data from the ticker
        bid = ticker.bid if not util.isNan(ticker.bid) else None
        ask = ticker.ask if not util.isNan(ticker.ask) else None

        # Priority for IV: Model Option IV > Model Greeks IV
        iv = None
        # 1. Try User-specified 'modelOptionImpliedVol' (Generic 106)
        if hasattr(ticker, 'modelOptionImpliedVol') and not util.isNan(ticker.modelOptionImpliedVol):
            iv = ticker.modelOptionImpliedVol
            logging.info(f"Using IBKR Model Option IV: {iv:.2%}")
        # 2. Try standard ib_insync 'modelGreeks.impliedVol' (Generic 106 standard mapping)
        elif ticker.modelGreeks and not util.isNan(ticker.modelGreeks.impliedVol):
            iv = ticker.modelGreeks.impliedVol
            logging.info(f"Using IBKR Model Greeks IV: {iv:.2%}")

    finally:
        ib.cancelMktData(contract) # Clean up the market data subscription

    # Price data is required
    if bid is None or ask is None:
        logging.error(f"Insufficient price data for {contract.localSymbol}. Bid: {bid}, Ask: {ask}")
        return None

    # IV is optional - proceed without it if unavailable
    if iv is None:
        from config import get_active_profile
        # We need config to get profile, but get_option_market_data signature doesn't include it.
        # Fallback: assume Coffee default if config not accessible, or use hardcoded safe value.
        # Ideally, we should pass config.
        # Since we can't easily change signature everywhere without breaking things,
        # we will use a safe default but log it.
        # M1 FIX: Actually we should look up from profile if possible.
        # Loading config here might be slow.
        # Let's assume the caller will handle or we use a safe default.
        # But wait, WS5.5 says "REPLACE WITH ... profile = get_active_profile(config)".
        # We don't have config here.
        # Strategy: Use a default, but note that the caller (create_combo_order_object)
        # DOES have config. We should pass IV/RiskFreeRate from there?
        # No, this function fetches market data.

        # NOTE: For now, hardcoding 35% as a safe fallback is acceptable if config isn't passed.
        # BUT to follow instructions, let's load config locally (cached).
        try:
            from config_loader import load_config
            from config import get_active_profile
            cfg = load_config()
            profile = get_active_profile(cfg)
            iv = profile.fallback_iv
            rfr = profile.risk_free_rate
            logging.warning(f"IV data missing for {contract.localSymbol}, using profile fallback IV: {iv:.0%}")
        except Exception:
            iv = 0.35
            rfr = 0.04
            logging.warning(f"IV data missing for {contract.localSymbol}, using hardcoded fallback IV: {iv:.0%}")
    else:
        # Load risk free rate
        try:
            from config_loader import load_config
            from config import get_active_profile
            cfg = load_config()
            profile = get_active_profile(cfg)
            rfr = profile.risk_free_rate
        except Exception:
            rfr = 0.04

    return {
        'bid': bid,
        'ask': ask,
        'implied_volatility': iv,
        'risk_free_rate': rfr  # M3 FIX: Use profile rate
    }

async def get_underlying_iv_metrics(ib: IB, future_contract: Contract) -> dict:
    """
    Fetches IV metrics from IBKR for the underlying.
    Returns dict with iv_rank, iv_percentile, current_iv (approximate from ATM option).
    """
    try:
        # Get ATM option for IV proxy
        chains = await asyncio.wait_for(ib.reqSecDefOptParamsAsync(
            future_contract.symbol,
            future_contract.exchange,
            future_contract.secType,
            future_contract.conId
        ), timeout=10)

        if not chains:
            return {'iv_rank': 'N/A', 'iv_percentile': 'N/A', 'current_iv': 'N/A'}

        # Get ticker with model greeks (generic tick 106)
        ticker = ib.reqMktData(future_contract, '106', False, False)
        try:
            await asyncio.sleep(2)

            iv_data = {
                'iv_rank': 'N/A',
                'iv_percentile': 'N/A',
                'current_iv': 'N/A'
            }

            # IBKR provides impliedVolatility on the underlying ticker for index options
            # For futures, we approximate from near-term ATM option
            if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks:
                if not util.isNan(ticker.modelGreeks.impliedVol):
                    iv_data['current_iv'] = f"{ticker.modelGreeks.impliedVol:.1%}"

            return iv_data
        finally:
            ib.cancelMktData(future_contract)

    except Exception as e:
        logging.warning(f"Failed to fetch IV metrics: {e}")
        return {'iv_rank': 'N/A', 'iv_percentile': 'N/A', 'current_iv': 'N/A'}

async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """
    Fetches the next N active futures contracts for a given symbol.
    Excludes contracts expiring within min_dte days (profile-defined).
    """
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        # M6 FIX: Use profile for min_dte
        from config_loader import load_config
        from config import get_active_profile
        cfg = load_config()
        profile = get_active_profile(cfg)
        min_dte = profile.min_dte

        cds = await asyncio.wait_for(
            ib.reqContractDetailsAsync(Future(symbol, exchange=exchange)), timeout=15
        )
        now = datetime.now()
        filtered_contracts = []

        for cd in cds:
            contract = cd.contract
            # Check basic future expiration (must be in future)
            if contract.lastTradeDateOrContractMonth > now.strftime('%Y%m'):
                # Apply min_dte rule
                # Parse expiration date. Typically YYYYMMDD for specific contracts.
                try:
                    exp_str = contract.lastTradeDateOrContractMonth
                    # Handle YYYYMMDD format
                    if len(exp_str) == 8:
                        exp_date = datetime.strptime(exp_str, '%Y%m%d')
                    # Handle YYYYMM format
                    elif len(exp_str) == 6:
                        exp_date = datetime.strptime(exp_str + "01", '%Y%m%d')
                    else:
                        continue # Invalid format

                    if exp_date >= (now + timedelta(days=min_dte)):
                        filtered_contracts.append(contract)
                    else:
                        logging.info(f"Skipping contract {contract.localSymbol} (Exp: {exp_str}): Expires < {min_dte} days.")
                except ValueError:
                    logging.warning(f"Could not parse expiration for {contract.localSymbol}: {contract.lastTradeDateOrContractMonth}")
                    continue

        active = sorted(filtered_contracts, key=lambda c: c.lastTradeDateOrContractMonth)
        logging.info(f"Found {len(active)} active tradeable contracts (>={min_dte}d exp). Returning the first {count}.")
        return active[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}"); return []


async def build_option_chain(ib: IB, future_contract: Contract) -> dict | None:
    """Fetches the full option chain for a given futures contract."""
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        chains = await asyncio.wait_for(
            ib.reqSecDefOptParamsAsync(future_contract.symbol, future_contract.exchange, 'FUT', future_contract.conId),
            timeout=10
        )
        if not chains:
            logging.warning(f"No option chains found for future {future_contract.localSymbol} (conId: {future_contract.conId})")
            return None
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])
        return {
            'exchange': chain.exchange,
            'tradingClass': chain.tradingClass,
            'expirations': sorted(chain.expirations),
            'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}
        }
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}"); return None


async def create_combo_order_object(ib: IB, config: dict, strategy_def: dict) -> tuple[Contract, Order] | None:
    """
    Prices a combo strategy and creates qualified Contract and Order objects without placing them.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        strategy_def (dict): A dictionary containing the strategy parameters.

    Returns:
        A tuple of (Contract, Order) or None if creation fails.
    """
    logging.info("--- Pricing individual legs and creating order object ---")
    action = strategy_def['action']
    legs_def = strategy_def['legs_def']
    exp_details = strategy_def['exp_details']
    chain = strategy_def['chain']
    underlying_price = strategy_def['underlying_price']

    # Get profile-driven specs
    tick_size = get_tick_size(config)
    # IBKR expects multiplier as string (e.g. "37500")
    contract_multiplier = str(get_contract_multiplier(config))

    # 1. Create all leg contract objects first
    leg_contracts = []
    for right, _, strike in legs_def:
        contract = FuturesOption(
            symbol=config['symbol'],
            lastTradeDateOrContractMonth=exp_details['exp_date'],
            strike=strike,
            right=right,
            exchange=chain['exchange'],
            # FIX: Use the dynamically fetched tradingClass instead of a hardcoded value.
            # The tradingClass for Coffee (KC) options is 'OKC', not 'OK'.
            tradingClass=chain['tradingClass'],
            multiplier=contract_multiplier
        )
        leg_contracts.append(contract)

    # 2. Probe: verify first leg (ATM) exists before qualifying all.
    #    reqSecDefOptParams returns the union of strikes across ALL expirations,
    #    so far-dated expirations may not have the selected strikes listed yet.
    #    A single-leg probe catches this early (1 IB round-trip instead of N).
    probe = leg_contracts[0]
    try:
        await asyncio.wait_for(ib.qualifyContractsAsync(probe), timeout=8)
    except Exception as e:
        logging.warning(f"Probe qualification failed for expiry {exp_details['exp_date']}: {e}")
        return None

    if probe.conId == 0:
        logging.warning(
            f"Strike {probe.strike}{probe.right} not available for expiry "
            f"{exp_details['exp_date']} (class={chain['tradingClass']}). "
            f"Far-dated expirations may have limited strike coverage. Skipping."
        )
        return None

    # 3. Qualify remaining legs
    remaining = leg_contracts[1:]
    if remaining:
        try:
            qualified_remaining = await asyncio.wait_for(
                ib.qualifyContractsAsync(*remaining), timeout=12
            )
        except Exception as e:
            logging.warning(f"Remaining leg qualification failed for expiry {exp_details['exp_date']}: {e}")
            return None

        if len(qualified_remaining) != len(remaining):
            logging.warning(
                f"Qualification returned {len(qualified_remaining)}/{len(remaining)} "
                f"remaining legs for expiry {exp_details['exp_date']}. Skipping."
            )
            return None

        for leg in qualified_remaining:
            if leg.conId == 0:
                logging.warning(
                    f"Strike not available: {leg.right} @ {leg.strike} "
                    f"(expiry={leg.lastTradeDateOrContractMonth}, class={leg.tradingClass})"
                )
                return None

    qualified_legs = leg_contracts  # All modified in-place by qualifyContractsAsync

    # 4. Price each leg theoretically and get live market spread
    net_theoretical_price = 0.0
    combo_bid_price = 0.0
    combo_ask_price = 0.0
    for i, q_leg in enumerate(qualified_legs):
        leg_action = legs_def[i][1]  # 'BUY' or 'SELL'

        market_data = await get_option_market_data(ib, q_leg, strategy_def['future_contract'])
        if not market_data:
            logging.error(f"Failed to get market data for {q_leg.localSymbol}. Aborting order.")
            return None

        # Calculate theoretical price using Black-Scholes
        pricing_result = price_option_black_scholes(
            S=underlying_price,
            K=q_leg.strike,
            T=exp_details['days_to_exp'] / 365,
            r=market_data['risk_free_rate'],
            sigma=market_data['implied_volatility'],
            option_type=q_leg.right
        )
        if not pricing_result:
            logging.error(f"Failed to price leg {leg_action} {q_leg.localSymbol}. Aborting."); return None

        price = pricing_result['price']
        net_theoretical_price += price if leg_action == 'BUY' else -price
        logging.info(f"  -> Leg Theoretical Price ({leg_action}): {q_leg.localSymbol} @ {price:.2f}")

        # Aggregate combo bid/ask from live market data
        leg_bid = market_data['bid']
        leg_ask = market_data['ask']
        if leg_action == 'BUY':
            combo_bid_price += leg_bid
            combo_ask_price += leg_ask
        else:  # 'SELL'
            combo_bid_price -= leg_ask
            combo_ask_price -= leg_bid

    # 5. Add Liquidity Filter and Calculate Limit Price (Ceiling/Floor) and Initial Price (Start)
    tuning_params = config.get('strategy_tuning', {})
    max_spread_pct = tuning_params.get('max_liquidity_spread_percentage', 0.25)
    fixed_slippage = tuning_params.get('fixed_slippage_cents', 0.5)
    # NEW: Configurable ceiling aggression (0.0 = mid, 1.0 = full ask/bid)
    ceiling_aggression = tuning_params.get('ceiling_aggression_factor', 0.75)

    market_spread = combo_ask_price - combo_bid_price

    # Liquidity Filter: Check if the market spread is too wide relative to the theoretical price
    if net_theoretical_price > 0 and (market_spread / net_theoretical_price) > max_spread_pct:
        logging.warning(
            f"LIQUIDITY FILTER FAILED: Market spread ({market_spread:.2f}) is "
            f"{(market_spread / net_theoretical_price):.1%} of theoretical price ({net_theoretical_price:.2f}), "
            f"which exceeds the max of {max_spread_pct:.1%}. Aborting order."
        )
        return None

    tick_size = get_tick_size(config)  # Commodity-agnostic tick size
    market_mid = (combo_bid_price + combo_ask_price) / 2

    # ================================================================
    # CEILING/FLOOR LOGIC (Amendment A — Flight Director Approved)
    #
    # BUY: ceiling = max(theoretical, market_aggressive), capped at ask
    #   → "Take the HIGHER cap so we CAN reach a fillable price"
    #   → Safety: never exceed the actual ask (negative edge)
    #
    # SELL: floor = min(theoretical, market_aggressive), floored at bid
    #   → "Take the LOWER floor so we CAN reach a fillable price"
    #   → Safety: never go below the actual bid (negative edge)
    # ================================================================

    if action == 'BUY':
        # Theoretical ceiling with slippage
        theoretical_ceiling = round_to_tick(
            net_theoretical_price + fixed_slippage, tick_size, 'BUY'
        )

        # Market-aware ceiling: interpolate between mid and ask based on aggression
        # aggression=0.0 → ceiling at mid (conservative, often unfillable)
        # aggression=0.75 → ceiling at 75% between mid and ask (recommended)
        # aggression=1.0 → ceiling at ask (most aggressive)
        market_aggressive_ceiling = round_to_tick(
            market_mid + (combo_ask_price - market_mid) * ceiling_aggression,
            tick_size, 'BUY'
        )

        # USE MAX: When market diverges from theoretical, trust the market
        # This ensures we can actually reach a fillable price
        ceiling_price = max(theoretical_ceiling, market_aggressive_ceiling)

        # SAFETY CAP: Never exceed the actual ask (paying above ask = negative edge)
        ceiling_price = min(ceiling_price, round_to_tick(combo_ask_price, tick_size, 'BUY'))

        # Start 1 tick above bid (passive entry)
        initial_price = round_to_tick(combo_bid_price + tick_size, tick_size, 'BUY')
        initial_price = min(initial_price, ceiling_price)

        logging.info(
            f"BUY Cap Calc: Theoretical={theoretical_ceiling:.2f}, "
            f"MarketAggressive={market_aggressive_ceiling:.2f} (aggression={ceiling_aggression}), "
            f"Mid={market_mid:.2f}, Ask={combo_ask_price:.2f}, Final Cap={ceiling_price:.2f}"
        )

    else:  # SELL
        # Theoretical floor with slippage
        theoretical_floor = round_to_tick(
            net_theoretical_price - fixed_slippage, tick_size, 'SELL'
        )

        # Market-aware floor: interpolate between mid and bid based on aggression
        market_aggressive_floor = round_to_tick(
            market_mid - (market_mid - combo_bid_price) * ceiling_aggression,
            tick_size, 'SELL'
        )

        # USE MIN: Take the LOWER floor so we CAN reach a fillable price
        floor_price = min(theoretical_floor, market_aggressive_floor)

        # SAFETY FLOOR: Never go below the actual bid (selling below bid = negative edge)
        floor_price = max(floor_price, round_to_tick(combo_bid_price, tick_size, 'SELL'))

        # Start 1 tick below ask (passive entry)
        initial_price = round_to_tick(combo_ask_price - tick_size, tick_size, 'SELL')
        initial_price = max(initial_price, floor_price)

        logging.info(
            f"SELL Floor Calc: Theoretical={theoretical_floor:.2f}, "
            f"MarketAggressive={market_aggressive_floor:.2f} (aggression={ceiling_aggression}), "
            f"Mid={market_mid:.2f}, Bid={combo_bid_price:.2f}, Final Floor={floor_price:.2f}"
        )

    logging.info(f"Net Theoretical: {net_theoretical_price:.2f}, Market Spread: {market_spread:.2f}")
    logging.info(
        f"Adaptive Strategy: Start @ {initial_price:.2f}, "
        f"Cap/Floor @ {ceiling_price if action == 'BUY' else floor_price:.2f}"
    )

    # 5b. Riskless combo guard: skip zero-debit spreads that IB will reject
    if action == 'BUY' and ceiling_price <= 0:
        logging.warning(
            f"RISKLESS COMBO FILTER: BUY spread has zero/negative ceiling price "
            f"({ceiling_price:.2f}). IB would reject as riskless. Skipping."
        )
        return None
    if action == 'SELL' and floor_price <= 0:
        logging.warning(
            f"RISKLESS COMBO FILTER: SELL spread has zero/negative floor price "
            f"({floor_price:.2f}). IB would reject as riskless. Skipping."
        )
        return None

    # 6. Build the Bag contract using qualified leg conIds
    combo = Bag(symbol=config['symbol'], exchange=chain['exchange'], currency='USD')
    for i, q_leg in enumerate(qualified_legs):
        leg_action = legs_def[i][1]
        combo.comboLegs.append(ComboLeg(conId=q_leg.conId, ratio=1, action=leg_action, exchange=chain['exchange']))

    # The Bag contract itself does not need to be qualified if the legs are.

    order_type = config.get('strategy_tuning', {}).get('order_type', 'LMT').upper()

    # Determine Quantity (Use override from strategy_def if available, else config)
    quantity = strategy_def.get('quantity', config['strategy']['quantity'])

    if order_type == 'MKT':
        order = MarketOrder(action, quantity, tif="DAY")
        logging.info(f"Creating Market Order for {action} {quantity}.")
    else: # Default to Limit Order
        # We set the initial price as the limit price
        order = LimitOrder(action, quantity, initial_price, tif="DAY")
        # Store the ceiling/floor in the order object for the manager to use
        if action == 'BUY':
            order.adaptive_limit_price = ceiling_price
        else:
            order.adaptive_limit_price = floor_price
        logging.info(f"Creating Limit Order for {action} {quantity} @ {initial_price:.2f} (Adaptive Cap: {order.adaptive_limit_price:.2f}).")

    logging.info("Using Custom Adaptive Logic (IBKR Algo Disabled).")

    # Assign a unique reference ID to the parent order.
    # IB will propagate this ID to all execution reports for the individual legs.
    order.orderRef = str(uuid.uuid4())
    logging.info(f"Assigned OrderRef: {order.orderRef}")

    return (combo, order)


def place_order(ib: IB, contract: Contract, order: Order) -> Trade | None:
    """
    Places a pre-constructed order, ensuring it has a unique `orderRef`.

    If the order does not already have an `orderRef`, this function assigns a
    new UUID to it. This ensures that all orders, including those created for
    risk management or position closing, have a unique identifier that can be
    tracked.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        contract (Contract): The qualified contract to be traded.
        order (Order): The order object to be placed.

    Returns:
        The `ib_insync.Trade` object for the placed order, or None if trading is OFF.
    """
    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logging.info(
            f"[OFF] WOULD PLACE {order.action} {order.totalQuantity} "
            f"{contract.localSymbol} @ {getattr(order, 'lmtPrice', 'MKT')}"
        )
        return None

    if not order.orderRef:
        order.orderRef = str(uuid.uuid4())
        logging.info(f"Assigned new unique OrderRef for tracking: {order.orderRef}")

    logging.info(f"Placing {order.action} order for {contract.localSymbol}...")
    trade = ib.placeOrder(contract, order)
    logging.info(f"Successfully placed order ID {trade.order.orderId} for {contract.localSymbol}.")
    return trade


async def place_directional_spread_with_protection(
    ib: IB,
    combo_contract: Contract,
    combo_order: Order,
    underlying_contract: Contract,
    entry_price: float,
    stop_distance_pct: float = 0.03,
    is_bullish_strategy: bool = True
) -> tuple[Trade, Trade | None]:
    """
    Places a directional spread with an exchange-native stop order on the underlying.

    The stop order acts as a "circuit breaker" if price gaps through our thesis.
    It hedges delta exposure, buying time to close the options position.

    Args:
        combo_contract: The BAG contract for the spread
        combo_order: The order for the spread
        underlying_contract: The underlying future (e.g., KCH6)
        entry_price: Current underlying price at entry
        stop_distance_pct: Distance for stop trigger (default 3%)
        is_bullish_strategy: True for Bull Spreads (Protects Downside), False for Bear (Protects Upside)

    Returns:
        Tuple of (spread_trade, stop_trade)
    """
    if not combo_order.orderRef:
        combo_order.orderRef = str(uuid.uuid4())

    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logging.info(
            f"[OFF] WOULD PLACE protected spread {combo_order.action} "
            f"{combo_contract.localSymbol} @ {getattr(combo_order, 'lmtPrice', 'MKT')} "
            f"+ catastrophe stop on {underlying_contract.localSymbol}"
        )
        return (None, None)

    # 1. Place the spread order
    logging.info(f"Placing protected spread order for {combo_contract.localSymbol}...")
    spread_trade = ib.placeOrder(combo_contract, combo_order)

    # 2. Determine stop parameters
    if is_bullish_strategy:
        # Bull spread (Long Delta): Protect against Drop
        stop_price = entry_price * (1 - stop_distance_pct)
        stop_action = 'SELL'  # Sell future to hedge
    else:
        # Bear spread (Short Delta): Protect against Rise
        stop_price = entry_price * (1 + stop_distance_pct)
        stop_action = 'BUY'  # Buy future to hedge

    # 3. Create stop order on underlying
    # Round to valid tick increment to avoid IB Warning 110.
    # For stops, round conservatively (triggers sooner = safer):
    #   SELL stop → round UP (ceil), BUY stop → round DOWN (floor)
    stop_price = round_to_tick(stop_price, action=stop_action)

    # NOTE: This creates a DELTA MISMATCH by design.
    stop_order = StopOrder(
        action=stop_action,
        totalQuantity=1,  # Single future hedges ~100 delta (approx for 1 spread?)
        # Ideally this should match spread delta, but spec says "Single future".
        # Spread is usually size 1. Future size 1.
        stopPrice=stop_price,
        tif='GTC',  # Good-til-cancelled
        outsideRth=True  # Active outside regular hours
    )
    stop_order.orderRef = f"CATASTROPHE_{spread_trade.order.orderRef}"

    # 4. Place the stop order
    stop_trade = ib.placeOrder(underlying_contract, stop_order)

    logging.info(
        f"Catastrophe protection placed: {stop_action} {underlying_contract.localSymbol} "
        f"@ {stop_price:.2f} (stop) for spread order {spread_trade.order.orderId}"
    )

    return spread_trade, stop_trade


async def close_spread_with_protection_cleanup(
    ib: IB,
    spread_trade: Trade, # Pass the trade or finding the stop by ref?
    # The spec used spread_position, stop_order_ref.
    # We will use stop_order_ref approach.
    stop_order_ref: str
):
    """Cancels the associated catastrophe stop when a spread is closed."""
    if not stop_order_ref:
        return

    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logging.info(f"[OFF] WOULD CANCEL catastrophe stop: {stop_order_ref}")
        return

    # Find and cancel the orphaned stop order
    try:
        open_orders = await asyncio.wait_for(ib.reqAllOpenOrdersAsync(), timeout=8)
    except asyncio.TimeoutError:
        logging.warning(f"reqAllOpenOrdersAsync timed out (8s) when cleaning up stop {stop_order_ref}")
        return
    for trade in open_orders:
        if trade.order.orderRef == stop_order_ref:
            ib.cancelOrder(trade.order)
            logging.info(f"Cancelled orphaned catastrophe stop: {stop_order_ref}")
            break
