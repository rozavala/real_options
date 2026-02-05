"""Manages the entire lifecycle of trading orders.

This module is responsible for generating, queuing, placing, and canceling
orders, as well as closing open positions at the end of the trading day. It
acts as the central hub for all order-related activities, decoupling the
strategy definition from the execution timing.
"""
import asyncio
import logging
import random
import time
import traceback
from collections import defaultdict
from ib_insync import *
from datetime import timezone

from config_loader import load_config
from notifications import send_pushover_notification

from trading_bot.ib_interface import (
    get_active_futures, build_option_chain, create_combo_order_object, place_order,
    place_directional_spread_with_protection, close_spread_with_protection_cleanup
)
from trading_bot.tms import TransactiveMemory
from trading_bot.signal_generator import generate_signals
from trading_bot.strategy import define_directional_strategy, define_volatility_strategy, validate_iron_condor_risk
from trading_bot.utils import (
    log_trade_to_ledger, log_order_event, configure_market_data_type,
    is_market_open, round_to_tick, COFFEE_OPTIONS_TICK_SIZE,
    get_tick_size, get_contract_multiplier, get_dollar_multiplier,
    hours_until_weekly_close
)
from trading_bot.calendars import is_trading_day, get_exchange_calendar
from trading_bot.compliance import ComplianceGuardian, calculate_spread_max_risk
from trading_bot.connection_pool import IBConnectionPool
from trading_bot.position_sizer import DynamicPositionSizer
from trading_bot.drawdown_circuit_breaker import DrawdownGuard
from trading_bot.order_queue import OrderQueueManager

# --- Module-level storage for orders ---
# Structure: [(contract, order, decision_data), ...]
ORDER_QUEUE = OrderQueueManager()

# Module-level lock for capital tracking
_CAPITAL_LOCK = asyncio.Lock()

from pathlib import Path
import json

CAPITAL_STATE_FILE = Path("data/capital_state.json")

def _load_committed_capital() -> float:
    """Load persisted committed capital or 0.0 if none."""
    if CAPITAL_STATE_FILE.exists():
        try:
            with open(CAPITAL_STATE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('committed_capital', 0.0)
        except Exception as e:
            logger.warning(f"Could not load capital state: {e}")
    return 0.0

def _save_committed_capital(amount: float):
    """Persist committed capital to disk."""
    CAPITAL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAPITAL_STATE_FILE, 'w') as f:
        json.dump({
            'committed_capital': amount,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }, f)

async def _reconcile_working_orders(ib: IB, config: dict) -> float:
    """
    Query IBKR for working orders and calculate committed capital.

    L1 FIX: Prevents double-leverage on restart by checking actual
    working orders rather than trusting in-memory state.

    v4.1 AMENDMENT: Filters by clientId/orderRef to count only THIS
    bot instance's orders, preventing cross-bot capital stealing.
    """
    try:
        open_orders = await ib.reqAllOpenOrdersAsync()
        if not open_orders:
            return 0.0

        # v4.1: FILTER BY CLIENT ID OR ORDER REF
        my_client_id = ib.client.clientId
        my_order_ref_prefix = config.get('execution', {}).get('order_ref_prefix', 'MISSION_CTRL')

        total_committed = 0.0
        skipped_foreign = 0

        for trade in open_orders:
            # Check if order belongs to this instance
            is_mine = (
                (trade.order.clientId == my_client_id) or
                (trade.order.orderRef and
                 trade.order.orderRef.startswith(my_order_ref_prefix))
            )

            if not is_mine:
                skipped_foreign += 1
                continue

            if trade.orderStatus.status in ('PreSubmitted', 'Submitted'):
                # Estimate risk (conservative: use margin requirement)
                # For spreads, this is approximate but safe
                # v5.1 FIX: Profile-driven estimate instead of hardcoded $500
                from config import get_active_profile
                profile = get_active_profile(config)
                fallback_risk_per_contract = (
                    profile.contract.tick_value * profile.contract.contract_size * 0.02
                )
                estimated_risk = abs(trade.order.totalQuantity) * fallback_risk_per_contract
                total_committed += estimated_risk
                logger.info(
                    f"Working order {trade.order.orderId}: ~${estimated_risk:.0f} committed "
                    f"(Fallback: {profile.ticker} @ ${fallback_risk_per_contract:.0f}/contract)"
                )

        if skipped_foreign > 0:
            logger.info(
                f"L1: Filtered {skipped_foreign} foreign orders "
                f"(client_id != {my_client_id}, ref != {my_order_ref_prefix}*)"
            )

        return total_committed
    except Exception as e:
        logger.warning(f"Working order query failed: {e}. Assuming no committed capital.")
        return 0.0

# Module-level set for TMS thesis deduplication
_recorded_thesis_positions = set()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrderManager")


def _get_order_display_name(contract, strategy_def: dict = None) -> str:
    """Generate a readable display name for orders, especially BAG/combo contracts."""
    from ib_insync import Bag

    # Derive ticker from contract (commodity-agnostic)
    ticker = contract.symbol or "UNK"

    # If localSymbol exists and is meaningful, use it
    if contract.localSymbol and contract.localSymbol.strip():
        return contract.localSymbol

    # For BAG contracts, build from strategy definition or legs
    if isinstance(contract, Bag):
        if strategy_def and 'legs_def' in strategy_def:
            legs = strategy_def['legs_def']
            # Format: KC-C352.5+P352.5 (for straddle) or KC-4LEG-IC (for Iron Condor)
            if len(legs) == 2:
                # Straddle or vertical spread
                leg_str = "+".join([f"{l[0]}{l[2]}" for l in legs])
                return f"{ticker}-{leg_str}"
            elif len(legs) == 4:
                return f"{ticker}-IRON_CONDOR"
            else:
                return f"{ticker}-{len(legs)}LEG"

        if contract.comboLegs:
            return f"{ticker}-{len(contract.comboLegs)}LEG-BAG"

    # Fallback
    return ticker


def _describe_bag(contract) -> str:
    """Generate a readable symbol for BAG contracts."""
    from ib_insync import Bag
    ticker = contract.symbol or "UNK"
    if isinstance(contract, Bag) and contract.comboLegs:
        return f"{ticker}-{len(contract.comboLegs)}LEG-BAG"
    return ticker


async def generate_and_execute_orders(config: dict, connection_purpose: str = "orchestrator_orders", shutdown_check=None, trigger_type=None):
    """
    Generates, queues, and immediately places orders.
    This ensures that order placement isn't skipped if generation takes
    longer than the gap between scheduled times.
    """
    # === EARLY EXIT: Skip entirely on non-trading days ===
    if not is_market_open():
        logger.info("Market is closed (weekend/holiday). Skipping order generation cycle.")
        return

    # === HOLDING-TIME GATE: Skip if insufficient time before forced close ===
    min_holding_hours = config.get('risk_management', {}).get(
        'min_holding_hours', 6.0
    )
    remaining_hours = hours_until_weekly_close(config)

    if remaining_hours < min_holding_hours:
        logger.info(
            f"Holding-time gate: Only {remaining_hours:.1f}h until weekly close "
            f"(minimum: {min_holding_hours}h). Skipping order generation."
        )
        send_pushover_notification(
            config.get('notifications', {}),
            "📅 Order Gen Skipped",
            f"Weekly close in {remaining_hours:.1f}h — below {min_holding_hours}h minimum. "
            f"No new positions today."
        )
        return

    logger.info(">>> Starting combined task: Generate and Execute Orders <<<")
    await generate_and_queue_orders(config, connection_purpose=connection_purpose, shutdown_check=shutdown_check, trigger_type=trigger_type)

    # Only proceed to placement if orders were actually queued
    if not ORDER_QUEUE.is_empty():
        await place_queued_orders(config, connection_purpose=connection_purpose)
    else:
        logger.info("No orders were queued. Skipping placement step.")
    logger.info(">>> Combined task 'Generate and Execute Orders' complete <<<")


async def generate_and_queue_orders(config: dict, connection_purpose: str = "orchestrator_orders", shutdown_check=None, trigger_type=None):
    """
    Generates trading strategies based on market data and API predictions,
    and queues them for later execution.
    """
    logger.info("--- Starting order generation and queuing process ---")
    await ORDER_QUEUE.clear()  # Clear queue at the start of each day

    try:
        # Use Connection Pool
        try:
            ib = await IBConnectionPool.get_connection(connection_purpose, config)
            configure_market_data_type(ib)
            logger.info(f"Connected to IB for signal generation (purpose: {connection_purpose}).")

            logger.info("Step 1: Generating structured signals via Council...")
            signals = await generate_signals(ib, config, shutdown_check=shutdown_check, trigger_type=trigger_type)
            logger.info(f"Generated {len(signals)} signals: {signals}")

            futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)

            # ==========================================================================
            # FLIGHT DIRECTOR OPTIMIZATION: Consolidated Pre-Signal Filtering
            # Implements FIX-002 (Deferred Month) and FIX-006 (Volume) with parallel I/O
            # ==========================================================================
            from datetime import datetime

            if not futures:
                logger.warning("No active futures found.")
                return

            # --- FILTER 1: Deferred Month (FIX-002) ---
            # M6/M7 FIX: Use profile for max_dte
            from config import get_active_profile
            profile = get_active_profile(config)
            max_days = profile.max_dte  # Was hardcoded 180 or config strategy

            date_filtered = []

            for f in futures:
                try:
                    # Handle both YYYYMM and YYYYMMDD formats
                    d_str = f.lastTradeDateOrContractMonth
                    fmt = '%Y%m' if len(d_str) == 6 else '%Y%m%d'
                    exp_date = datetime.strptime(d_str, fmt)
                    days_out = (exp_date - datetime.now()).days

                    if days_out <= max_days:
                        date_filtered.append(f)
                    else:
                        logger.info(
                            f"Skipping {f.localSymbol}: Too deferred "
                            f"({days_out} days > {max_days} max)"
                        )
                except Exception as e:
                    logger.warning(f"Date parse error for {f.localSymbol}: {e}. Keeping contract (fail-safe).")
                    date_filtered.append(f)  # Fail-safe: keep on parse error

            if not date_filtered:
                logger.warning("No futures remaining after Deferred Month Filter.")
                return

            logger.info(f"Date filter passed: {len(date_filtered)}/{len(futures)} contracts within {max_days} days.")

            # --- FILTER 2: Parallel Volume Check (FIX-006 Optimized) ---
            min_vol = config.get('strategy', {}).get('min_underlying_volume', 20)

            # Suppress expected Error 300 during snapshot cleanup
            def _suppress_300(reqId, errorCode, errorString, contract):
                if errorCode == 300:
                    logger.debug(f"Suppressed expected Error 300 for reqId {reqId} (snapshot cleanup)")

            ib.errorEvent += _suppress_300

            # Request market data for ALL candidates simultaneously (non-blocking)
            tickers = [ib.reqMktData(f, '', True, False) for f in date_filtered]

            logger.info(f"Checking liquidity for {len(tickers)} contracts (parallel)...")
            await asyncio.sleep(2.0)  # Single wait for all feeds to populate

            volume_filtered = []
            for f, t in zip(date_filtered, tickers):
                # Safely extract volume (handle NaN, None, missing)
                vol = 0
                try:
                    if t.volume is not None and not util.isNan(t.volume):
                        vol = int(t.volume)
                except (TypeError, ValueError):
                    vol = 0

                if vol >= min_vol:
                    volume_filtered.append(f)
                    logger.debug(f"{f.localSymbol}: Volume {vol} >= {min_vol} (PASS)")
                else:
                    logger.info(
                        f"Skipping {f.localSymbol}: Insufficient volume "
                        f"({vol} < {min_vol} minimum)"
                    )

                # CRITICAL: Cancel subscription to prevent feed accumulation
                try:
                    ib.cancelMktData(f)
                except Exception:
                    pass  # Error 300 is expected if subscription already expired

            # === v5.2 FIX: Explicit ticker cleanup after liquidity check ===
            # IBKR Error 300 occurs when reqId mappings from liquidity checks
            # collide with subsequent snapshot requests. Wait for cancellations.
            await asyncio.sleep(0.5)  # Let IB API process cancellation acknowledgements
            logger.info("Liquidity check cleanup complete.")

            ib.errorEvent -= _suppress_300

            futures = volume_filtered

            if not futures:
                logger.warning("No futures remaining after Volume Filter.")
                return

            logger.info(f"Volume filter passed: {len(futures)} liquid candidates proceeding to signal generation.")
            active_futures = futures
            # ==========================================================================
            # END CONSOLIDATED FILTERING
            # ==========================================================================

            # Initialize Sizer
            sizer = DynamicPositionSizer(config)

            # Fetch Account Value
            account_summary = await ib.accountSummaryAsync()
            net_liq_tag = next((v for v in account_summary if v.tag == 'NetLiquidation' and v.currency == 'USD'), None)
            account_value = float(net_liq_tag.value) if net_liq_tag else 0.0
            logger.info(f"Account Value for Sizing: ${account_value:,.2f}")

            # L1 FIX: Reconcile with IBKR before assuming zero committed
            persisted_capital = _load_committed_capital()
            working_capital = await _reconcile_working_orders(ib, config)  # v4.1: now takes config

            # Use whichever is higher (conservative)
            committed_capital = max(persisted_capital, working_capital)

            if committed_capital > 0:
                logger.info(f"L1 RECONCILED: Starting with ${committed_capital:,.2f} already committed")

            # === FLIGHT DIRECTOR WARNING ===
            # DO NOT PARALLELIZE THIS LOOP WITH asyncio.gather()
            # The committed_capital tracking requires sequential execution.
            # If parallelization is ever needed, use _CAPITAL_LOCK.

            for signal in signals:
                # === v7.0: Explicit NO TRADE short-circuit ===
                # Skip NEUTRAL directional signals (no trade warranted)
                # Skip NEUTRAL non-volatility signals (the "Cash is a Position" path)
                sig_direction = signal.get('direction', 'NEUTRAL')
                sig_pred_type = signal.get('prediction_type', 'DIRECTIONAL')

                if sig_direction == 'NEUTRAL' and sig_pred_type != 'VOLATILITY':
                    logger.info(
                        f"NO TRADE: {signal.get('contract_month', 'N/A')} — "
                        f"Direction={sig_direction}, Type={sig_pred_type}. "
                        f"Reason: {signal.get('reason', 'No reason provided')[:100]}"
                    )
                    continue

                # Also skip explicit NEUTRAL (legacy path)
                if sig_direction == 'NEUTRAL':
                    continue

                future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future:
                    logger.warning(f"No active future for signal month {signal.get('contract_month')}."); continue

                logger.info(f"Requesting snapshot market price for {future.localSymbol}...")

                price = float('nan')
                try:
                    # Retry loop for snapshot data
                    for attempt in range(3):
                        tickers = await ib.reqTickersAsync(future)
                        if not tickers:
                            logger.warning(f"No ticker returned for {future.localSymbol} (Attempt {attempt+1}/3)")
                            await asyncio.sleep(2)
                            continue

                        ticker = tickers[0]

                        # Priority A: Bid/Ask Midpoint
                        if ticker.bid > 0 and ticker.ask > 0:
                            price = (ticker.bid + ticker.ask) / 2
                            logger.info(f"Using bid/ask midpoint for {future.localSymbol}: {price}")
                            break

                        # Priority B: Last Price (with recency check if possible, but taking valid last for now)
                        if not util.isNan(ticker.last) and ticker.last > 0:
                             price = ticker.last
                             logger.info(f"Using last price for {future.localSymbol}: {price}")
                             break

                        # Priority C: Close Price (Fallback)
                        if not util.isNan(ticker.close) and ticker.close > 0:
                            price = ticker.close
                            logger.info(f"Using close price for {future.localSymbol}: {price}")
                            break

                        logger.warning(f"Ticker data incomplete for {future.localSymbol} (Attempt {attempt+1}/3): {ticker}")
                        await asyncio.sleep(2)
                    else:
                        logger.error(f"Failed to get valid price for {future.localSymbol} after retries.")

                except Exception as e:
                    logger.error(f"Error fetching snapshot price for {future.localSymbol}: {e}")

                if util.isNan(price):
                    logger.warning(f"Skipping strategy for {future.localSymbol} due to missing price.")
                    continue

                chain = await build_option_chain(ib, future)
                if not chain:
                    logger.warning(f"Skipping {future.localSymbol}: Failed to build option chain.")
                    continue

                define_func = define_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else define_volatility_strategy
                strategy_def = define_func(config, signal, chain, price, future)

                if strategy_def:
                    async with _CAPITAL_LOCK:
                        # Calculate available capital after commitments
                        available_capital = account_value - committed_capital

                        # Size the position based on remaining capital
                        vol_sent = signal.get('volatility_sentiment', 'NEUTRAL')
                        qty = await sizer.calculate_size(ib, signal, vol_sent, available_capital)

                        if qty <= 0:
                            logger.warning(f"Position sizer returned qty=0 for {signal.get('contract_month')}. Skipping.")
                            continue

                        strategy_def['quantity'] = qty

                        # --- Iron Condor Risk Validation ---
                        if signal['prediction_type'] == 'VOLATILITY' and signal.get('level') == 'LOW': # Iron Condor
                            # Calculate Max Risk (Conservative: Width of Wings * Multiplier * Qty)
                            # legs_def: [(right, action, strike), ...]
                            # Puts: Buy lp, Sell sp. Calls: Sell sc, Buy lc.
                            legs = strategy_def.get('legs_def', [])
                            if len(legs) == 4:
                                # Assume sorted or find by action/right
                                puts = sorted([l for l in legs if l[0] == 'P'], key=lambda x: x[2]) # [lp, sp]
                                calls = sorted([l for l in legs if l[0] == 'C'], key=lambda x: x[2]) # [sc, lc]

                                width = 0
                                if len(puts) == 2: width = max(width, abs(puts[1][2] - puts[0][2]))
                                if len(calls) == 2: width = max(width, abs(calls[1][2] - calls[0][2]))

                                # MECE FIX: Use profile-driven multipliers (handles KC/CC logic)
                                # multiplier = float(future.multiplier) if future.multiplier else 37500.0
                                dollar_multiplier = get_dollar_multiplier(config)
                                max_loss = width * dollar_multiplier * qty

                                logger.info(f"Iron Condor risk calc: width={width}, dollar_mult={dollar_multiplier}, qty={qty}, max_loss=${max_loss:,.2f}")

                                if not validate_iron_condor_risk(max_loss, account_value, config.get('catastrophe_protection', {}).get('iron_condor_max_risk_pct_of_equity', 0.02)):
                                    logger.warning(f"Skipping Iron Condor for {future.localSymbol} due to excessive risk.")
                                    continue

                        order_objects = await create_combo_order_object(ib, config, strategy_def)
                        if order_objects:
                            contract, order = order_objects

                            # Calculate and track committed capital
                            estimated_risk = await calculate_spread_max_risk(ib, contract, order, config)

                            async with _CAPITAL_LOCK:
                                committed_capital += estimated_risk
                                _save_committed_capital(committed_capital)

                            # Issue #4: Early abort if capital exhausted
                            if committed_capital > account_value:
                                logger.warning(f"Capital exhausted (${committed_capital:,.2f} committed vs ${account_value:,.2f} available). Stopping order generation.")
                                break

                            logger.info(
                                f"Capital tracking: Committed ${committed_capital:,.2f} | "
                                f"Remaining ${account_value - committed_capital:,.2f} | "
                                f"This order: ${estimated_risk:,.2f}"
                            )

                            # Store signal data with the order
                            signal['strategy_def'] = strategy_def
                            await ORDER_QUEUE.add((contract, order, signal))
                            logger.info(f"Successfully queued order for {future.localSymbol}.")
        except Exception as e:
            logger.error(f"Error in order generation block: {e}")
            raise e

        # NOTE: We do NOT disconnect here, as the pool manages connections.
        logger.info(f"--- Order generation complete. {len(ORDER_QUEUE)} orders queued. ---")

        # --- Create a detailed notification message ---

        # 1. Summarize the API signals
        signal_summary_parts = []
        if signals:
            for signal in signals:
                future_month = signal.get("contract_month", "N/A")
                confidence_val = signal.get("confidence", 0.0)
                conf_str = f"{confidence_val:.0%}" if isinstance(confidence_val, (int, float)) else str(confidence_val)

                if signal.get('prediction_type') == 'DIRECTIONAL':
                    direction = signal.get("direction", "N/A")
                    signal_summary_parts.append(
                        f"  {future_month}: {direction} ({conf_str})"
                    )
                elif signal.get('prediction_type') == 'VOLATILITY':
                    vol_level = signal.get("level", "N/A")
                    signal_summary_parts.append(
                        f"  {future_month}: {vol_level} VOL ({conf_str})"
                    )

        signal_summary = "<b>Signals:</b>\n" + "\n".join(signal_summary_parts) if signal_summary_parts else "No signals generated."

        # 2. Summarize the queued orders
        order_summary_parts = []
        if not ORDER_QUEUE.is_empty():
            # Access queue internals safely for reporting (no modification)
            # Note: We access _queue directly here for read-only reporting which is acceptable
            # as this runs in the same event loop and we just finished modification.
            current_queue = ORDER_QUEUE._queue
            for contract, order, _ in current_queue:
                # Determine the strategy name from the contract symbol if possible
                strategy_name = "Strategy" # Default
                if "PUT" in contract.localSymbol and "CALL" in contract.localSymbol:
                     strategy_name = "IRON_CONDOR" if "IRON" in contract.localSymbol else "STRADDLE/STRANGLE" # Placeholder
                elif "PUT" in contract.localSymbol:
                    strategy_name = "BEAR_PUT_SPREAD"
                elif "CALL" in contract.localSymbol:
                    strategy_name = "BULL_CALL_SPREAD"

                price_info = f"LMT @ ${order.lmtPrice:.2f}" if order.orderType == "LMT" else "MKT"
                order_summary_parts.append(
                    f"  - <b>{strategy_name}</b> for {contract.localSymbol}: {order.action} {order.totalQuantity} @ {price_info}"
                )
            order_summary = f"\n<b>{len(ORDER_QUEUE)} strategies generated and queued:</b>\n" + "\n".join(order_summary_parts)
        else:
            order_summary = "\nNo new orders were generated or queued based on these signals."

        # 3. Combine and send
        message = signal_summary + order_summary
        send_pushover_notification(config.get('notifications', {}), "Trading Orders Queued", message)

    except Exception as e:
        msg = f"A critical error occurred during order generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Generation CRITICAL", msg)


async def _get_market_data_for_leg(ib: IB, leg: ComboLeg) -> str:
    """
    Fetches comprehensive market data for a single contract leg with a timeout.
    Returns a formatted string for logging and notifications.
    """
    try:
        contract = await ib.reqContractDetailsAsync(Contract(conId=leg.conId))
        if not contract:
            logger.warning(f"Could not resolve contract for leg conId {leg.conId}")
            return f"Leg conId {leg.conId}: N/A"

        ticker = ib.reqMktData(contract[0].contract, '', False, False)
        try:
            await asyncio.sleep(2) # Give it a moment to populate

            # Wait for the ticker to update with a valid price, with a timeout
            start_time = time.time()
            while util.isNan(ticker.bid) and util.isNan(ticker.ask) and util.isNan(ticker.last):
                await asyncio.sleep(0.1)
                if (time.time() - start_time) > 15: # 15-second timeout
                    logger.error(f"Timeout waiting for market data for {contract[0].contract.localSymbol}.")
                    # ib.cancelMktData(ticker.contract) # Handled in finally
                    return f"Leg {contract[0].contract.localSymbol}: Timeout"

            # Fetch all data points
            leg_symbol = contract[0].contract.localSymbol
            leg_bid = ticker.bid if not util.isNan(ticker.bid) else "N/A"
            leg_ask = ticker.ask if not util.isNan(ticker.ask) else "N/A"
            leg_vol = ticker.volume if not util.isNan(ticker.volume) else "N/A"
            leg_last = ticker.last if not util.isNan(ticker.last) else "N/A"
        finally:
                try:
                    ib.cancelMktData(ticker.contract)
                except Exception:
                    pass
        
        # Return a formatted string
        return f"Leg {leg_symbol}: {leg_bid}x{leg_ask}, V:{leg_vol}, L:{leg_last}"

    except Exception as e:
        logger.error(f"Error fetching market data for leg {leg.conId}: {e}")
        return f"Leg conId {leg.conId}: Error"


async def _handle_and_log_fill(ib: IB, trade: Trade, fill: Fill, combo_id: int, position_uuid: str, decision_data: dict = None, config: dict = None):
    """
    Handles the logic for processing a trade fill, ensuring the contract
    details are complete before logging.
    """
    try:
        if isinstance(fill.contract, Bag):
            logger.info(f"Skipping ledger log for summary combo fill with conId: {fill.contract.conId}")
            return

        detailed_contract = Contract(conId=fill.contract.conId)
        await ib.qualifyContractsAsync(detailed_contract)

        corrected_fill = Fill(
            contract=detailed_contract,
            execution=fill.execution,
            commissionReport=fill.commissionReport,
            time=fill.time
        )

        if isinstance(trade.contract, Bag):
            for leg in trade.contract.comboLegs:
                if leg.conId == fill.contract.conId:
                    logger.info(f"Matched fill conId {fill.contract.conId} to leg in {trade.contract.localSymbol}.")
                    break
            else:
                logger.warning(f"Could not match fill conId {fill.contract.conId} to any leg in {trade.contract.localSymbol}.")

        # This now calls the async version of the function, passing the unique position ID
        await log_trade_to_ledger(ib, trade, "Strategy Execution", specific_fill=corrected_fill, combo_id=combo_id, position_id=position_uuid)
        logger.info(f"Successfully logged fill for {detailed_contract.localSymbol} (Order ID: {trade.order.orderId}, Position ID: {position_uuid})")

        # --- NEW: Record Trade Thesis to TMS (DEDUPLICATED) ---
        if decision_data and position_uuid:
            if position_uuid not in _recorded_thesis_positions:
                logger.info(f"Recording thesis for trade {position_uuid}...")
                _recorded_thesis_positions.add(position_uuid)

                # We determine strategy type from contract or decision?
                # Decision has 'prediction_type' and 'direction', not strategy name directly usually, unless added.
                # But we can infer.
                strategy_type = 'UNKNOWN'
                if isinstance(trade.contract, Bag):
                    if 'IRON' in trade.contract.localSymbol: strategy_type = 'IRON_CONDOR'
                    elif trade.contract.localSymbol.startswith('STRDL'): strategy_type = 'LONG_STRADDLE'
                    # Or use legs to infer
                    # Simple inference:
                    if decision_data.get('prediction_type') == 'DIRECTIONAL':
                        strategy_type = 'BULL_CALL_SPREAD' if decision_data.get('direction') == 'BULLISH' else 'BEAR_PUT_SPREAD'
                    elif decision_data.get('prediction_type') == 'VOLATILITY':
                        strategy_type = 'LONG_STRADDLE' if decision_data.get('level') == 'HIGH' else 'IRON_CONDOR'

                try:
                    # === FIX: Fetch Underlying Price for Thesis ===
                    underlying_price = None
                    contract_month = decision_data.get('contract_month')

                    if contract_month:
                        try:
                            # Reconstruct Future contract
                            future_contract = Future(
                                symbol=config.get('symbol', 'KC'),
                                lastTradeDateOrContractMonth=contract_month,
                                exchange=config.get('exchange', 'NYBOT')
                            )
                            # Define a quick fetch with timeout
                            async def _quick_fetch(c):
                                ticker = ib.reqMktData(c, '', True, False)
                                start_t = time.time()
                                while util.isNan(ticker.last) and util.isNan(ticker.close):
                                    if time.time() - start_t > 2.0: # 2s Timeout
                                        break
                                    await asyncio.sleep(0.1)
                                price = ticker.last if not util.isNan(ticker.last) else ticker.close
                                ib.cancelMktData(c)
                                return price if price and price > 0 else None

                            # Wrap in wait_for to be absolutely sure
                            underlying_price = await asyncio.wait_for(_quick_fetch(future_contract), timeout=2.5)

                        except Exception as ex:
                            logger.warning(f"Failed to fetch underlying price for thesis: {ex}")
                            underlying_price = decision_data.get('price') # Fallback to signal price

                    await record_entry_thesis_for_trade(
                        position_id=position_uuid,
                        strategy_type=strategy_type,
                        decision=decision_data,
                        entry_price=fill.execution.avgPrice, # Use fill price
                        config=config or {},
                        underlying_price=underlying_price,
                        contract_month=contract_month
                    )
                    logger.info(f"TMS: Successfully recorded entry thesis for trade {position_uuid}")
                except Exception as e:
                    logger.error(f"TMS: Failed to record thesis for {position_uuid}: {e}")
            else:
                logger.debug(f"TMS: Thesis already recorded for {position_uuid}, skipping duplicate.")

    except Exception as e:
        logger.error(f"Error processing and logging fill for order {trade.order.orderId}: {e}\n{traceback.format_exc()}")


async def check_liquidity_conditions(ib: IB, contract, order_size: int) -> tuple[bool, str]:
    """Check if liquidity conditions are safe for execution."""
    try:
        # --- PATCH: Bypass Depth Check for Combos (BAG) ---
        # IBKR returns 0 depth for synthetic combos even when legs have liquidity.
        # We rely on IBKR's Smart Routing to handle combo execution.
        if contract.secType == 'BAG':
            return True, "Liquidity check skipped for BAG (Combo) - relying on Smart Routing"
        # --- END PATCH ---

        # Request order book
        ticker = ib.reqMktDepth(contract, numRows=5)
        await asyncio.sleep(1)  # Allow data to populate

        # Calculate metrics
        bid_depth = sum(row.size for row in ticker.domBids if row.price > 0)
        ask_depth = sum(row.size for row in ticker.domAsks if row.price > 0)
        total_depth = bid_depth + ask_depth

        spread = 0.0
        if ticker.domBids and ticker.domAsks:
             spread = (ticker.domAsks[0].price - ticker.domBids[0].price)
        else:
             spread = float('inf')

        ib.cancelMktDepth(contract)

        # Safety checks
        min_depth = order_size * 3  # Want 3x our order size in book
        max_spread_pct = 0.5  # Max 0.5% spread

        if total_depth < min_depth:
            return False, f"Insufficient depth: {total_depth} (need {min_depth})"

        mid_price = 0.0
        if ticker.domAsks and ticker.domBids:
            mid_price = (ticker.domAsks[0].price + ticker.domBids[0].price) / 2

        if mid_price > 0:
            spread_pct = (spread / mid_price) * 100
            if spread_pct > max_spread_pct:
                return False, f"Wide spread: {spread_pct:.2f}% (max {max_spread_pct}%)"

        return True, f"Depth: {total_depth}, Spread: {spread_pct:.2f}%" if mid_price > 0 else "Depth OK, Spread Unknown"

    except Exception as e:
        logger.warning(f"Liquidity check failed (FAIL CLOSED): {e}")
        # FAIL CLOSED: Block the trade when we can't verify liquidity
        return False, f"Liquidity Check Error: {e}"


async def place_queued_orders(config: dict, orders_list: list = None, connection_purpose: str = "orchestrator_orders"):
    """
    Connects to IB, places orders, and monitors them.
    If 'orders_list' is provided, it processes those orders instead of the global ORDER_QUEUE.
    """
    # === D3 FIX: Explicit sequential processing ===
    # Use pop_all to atomically get and clear, preventing any race
    if orders_list is None:
        target_queue = await ORDER_QUEUE.pop_all()
    else:
        target_queue = orders_list

    logger.info(f"--- Placing and monitoring {len(target_queue)} orders ---")
    if not target_queue:
        logger.info("Order queue is empty. Nothing to place.")
        return

    logger.info(f"Processing {len(target_queue)} orders SEQUENTIALLY")

    # --- Initialize Compliance Guardian ---
    compliance = ComplianceGuardian(config)

    # Initialize Drawdown Guard
    drawdown_guard = DrawdownGuard(config)

    # Use Connection Pool
    ib = await IBConnectionPool.get_connection(connection_purpose, config)
    configure_market_data_type(ib)

    live_orders = {} # Dictionary to track order status by orderId

    # --- Event Handlers ---
    def on_order_status(trade: Trade):
        """Handles order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        if order_id in live_orders and live_orders[order_id]['status'] != status:
            live_orders[order_id]['status'] = status
            logger.info(f"Order {order_id} status update: {status}")
            log_order_event(trade, status)

    def on_exec_details(trade: Trade, fill: Fill):
        """
        Handles trade execution details for each leg, logs the trade, and
        tracks the overall fill status of the combo order.
        """
        order_id = trade.order.orderId
        if order_id in live_orders:
            leg_con_id = fill.contract.conId
            order_details = live_orders[order_id]

            if leg_con_id in order_details['filled_legs']:
                logger.info(f"Duplicate fill event for leg {leg_con_id} in order {order_id}. Ignoring.")
                return

            parent_trade = order_details['trade']
            combo_perm_id = parent_trade.order.permId
            # The unique position ID is the orderRef from the PARENT order object.
            position_uuid = parent_trade.order.orderRef
            decision_data = order_details.get('decision_data')

            logger.info(f"Fill received for leg {leg_con_id} in order {order_id}. Processing with Position ID {position_uuid}.")
            asyncio.create_task(_handle_and_log_fill(ib, trade, fill, combo_perm_id, position_uuid, decision_data, config))

            order_details['filled_legs'].add(leg_con_id)
            order_details['fill_prices'][leg_con_id] = fill.execution.avgPrice

            # --- Check if all legs of the combo are now filled ---
            if len(order_details['filled_legs']) == order_details['total_legs']:
                logger.info(f"All {order_details['total_legs']} legs of combo order {order_id} are now filled.")
                order_details['is_filled'] = True # Mark the entire order as filled

    def on_error(reqId: int, errorCode: int, errorString: str, contract):
        """
        Handle IB API errors, specifically Error 201 (Invalid Price).
        When a modification is rejected, revert to the last known good price.
        """
        if errorCode == 201 and reqId in live_orders:
            details = live_orders[reqId]
            details['modification_error_count'] += 1

            trade = details['trade']
            last_good = details['last_known_good_price']

            logger.warning(
                f"⚠️ Error 201 for Order {reqId}: Modification rejected. "
                f"Reverting internal tracking from {trade.order.lmtPrice:.2f} to {last_good:.2f}. "
                f"Error count: {details['modification_error_count']}/{details['max_modification_errors']}"
            )

            # Sync our view with IB's reverted state
            trade.order.lmtPrice = last_good

            if details['modification_error_count'] >= details['max_modification_errors']:
                logger.error(
                    f"🚫 Order {reqId} ({details['display_name']}): Max modification errors reached. "
                    f"Disabling adaptive walking - order will sit at {last_good:.2f} until fill or timeout."
                )
                details['adaptive_ceiling_price'] = last_good

    try:
        # Note: Connection already established above via Pool
        logger.info("Using pooled IB connection for order placement.")

        # --- NEW: Sort by Expiration (Nearest First) ---
        target_queue.sort(key=lambda x: x[0].lastTradeDateOrContractMonth if hasattr(x[0], 'lastTradeDateOrContractMonth') else '99999999')
        logger.info("Sorted orders by expiration proximity.")

        # --- NEW: Margin & Funds Check ---
        account_summary = await ib.accountSummaryAsync()
        avail_funds_tag = next((v for v in account_summary if v.tag == 'AvailableFunds' and v.currency == 'USD'), None)
        net_liq_tag = next((v for v in account_summary if v.tag == 'NetLiquidation' and v.currency == 'USD'), None)

        current_equity = 100000.0 # Default fallback
        if net_liq_tag:
            current_equity = float(net_liq_tag.value)

        if not avail_funds_tag:
            logger.warning("Could not fetch AvailableFunds. Proceeding without pre-check (Risky).")
            current_available_funds = float('inf')
        else:
            current_available_funds = float(avail_funds_tag.value)
            logger.info(f"Current Available Funds: ${current_available_funds:,.2f} | Equity: ${current_equity:,.2f}")

        # Fetch Current Positions for Compliance
        positions = await ib.reqPositionsAsync()
        current_position_count = sum(1 for p in positions if p.position != 0)

        # Filter Queue based on Margin Impact AND Compliance Review
        orders_to_place = []
        orders_checked_count = 0

        # Process one at a time with explicit ordering
        for idx, (contract, order, decision_data) in enumerate(target_queue):
            logger.info(f"Processing order {idx + 1}/{len(target_queue)}: {contract.localSymbol}")

            # === FIX-004: Capture display name for consistent logging ===
            strategy_def = decision_data.get('strategy_def') if isinstance(decision_data, dict) else None
            display_name = _get_order_display_name(contract, strategy_def)

            orders_checked_count += 1

            # --- Margin Safety Update: Force Refresh Every 3 Trades ---
            if orders_checked_count % 3 == 0:
                try:
                    logger.info("Refreshing account summary for margin safety...")
                    account_summary = await ib.accountSummaryAsync()
                    avail_funds_tag = next((v for v in account_summary if v.tag == 'AvailableFunds' and v.currency == 'USD'), None)
                    if avail_funds_tag:
                        current_available_funds = float(avail_funds_tag.value)
                        logger.info(f"Refreshed Available Funds: ${current_available_funds:,.2f}")
                except Exception as e:
                    logger.error(f"Failed to refresh margin info: {e}")

            try:
                # 0a. Drawdown Gate
                if drawdown_guard.enabled:
                    # Update status before each order group or rely on initial check?
                    # Safer to check once per batch or if we have a loop
                    # But we are inside a loop.
                    # Let's rely on status update from earlier?
                    # No, update_pnl needs IB. We have IB.
                    # Doing it for every order might be slow (AccountSummary).
                    # Maybe check once before the loop?
                    # Yes, but let's check strict status here.
                    if not drawdown_guard.is_entry_allowed():
                        logger.warning(f"DRAWDOWN GATE BLOCKED {display_name}: Circuit Breaker Active")
                        continue

                # 0b. Liquidity Gate (Fail Closed)
                liq_ok, liq_msg = await check_liquidity_conditions(ib, contract, order.totalQuantity)
                if not liq_ok:
                    logger.warning(f"LIQUIDITY GATE BLOCKED {contract.localSymbol}: {liq_msg}. Skipping order.")
                    continue

                # 1. Fetch Market Context (Spread & Trend)

                # Fetch spread
                ticker = ib.reqMktData(contract, '', True, False)
                await asyncio.sleep(2)

                spread_width = 0.0
                if ticker.bid > 0 and ticker.ask > 0:
                    spread_width = ticker.ask - ticker.bid
                else:
                    spread_width = 0.0 # Unknown

                # Fetch Trend (Fix Trend Blindness)
                trend_pct = 0.0
                try:
                    # Identify underlying contract
                    target_contract = None
                    if isinstance(contract, Bag):
                        # Extract first leg
                        if contract.comboLegs:
                            first_leg_conid = contract.comboLegs[0].conId
                            details = await ib.reqContractDetailsAsync(Contract(conId=first_leg_conid))
                            if details:
                                target_contract = details[0].contract
                    else:
                        target_contract = contract

                    if target_contract:
                        # Fetch 2 days of daily bars
                        bars = await ib.reqHistoricalDataAsync(
                            target_contract,
                            endDateTime='',
                            durationStr='2 D',
                            barSizeSetting='1 day',
                            whatToShow='TRADES',
                            useRTH=True
                        )
                        if bars and len(bars) >= 2:
                            close_curr = bars[-1].close
                            close_prev = bars[-2].close
                            if close_prev > 0:
                                trend_pct = (close_curr - close_prev) / close_prev
                except Exception as e:
                    logger.error(f"Failed to calculate trend for {contract.localSymbol}: {e}")

                # v5.1 FIX: Determine cycle type from decision_data context
                # Emergency orders pass through via orders_list parameter;
                # global queue orders are always from scheduled cycles
                _cycle_type = 'EMERGENCY' if orders_list is not None else 'SCHEDULED'

                order_context = {
                    'symbol': _describe_bag(contract) if isinstance(contract, Bag) else contract.localSymbol,
                    'bid_ask_spread': spread_width * 100 if contract.secType == 'BAG' else spread_width,
                    'total_position_count': current_position_count,
                    'market_trend_pct': trend_pct,
                    'ib': ib,
                    'contract': contract,
                    'order_object': order,  # NEW: Pass full order for risk calculation
                    'order_quantity': order.totalQuantity,
                    'account_equity': current_equity,
                    'cycle_type': _cycle_type,  # v5.1: Required for Fix 6 volume retry logic
                }

                # 2. Compliance Review
                # Pass 'price' (limit price) to order_context if possible for Notional calculation
                order_context['price'] = order.lmtPrice if order.orderType == 'LMT' else ticker.last if ticker.last else 0.0

                approved, reason = await compliance.review_order(order_context)
                if not approved:
                    logger.warning(f"Order for {display_name} blocked by Compliance: {reason}")
                    continue

                # 3. Margin Check
                what_if = await ib.whatIfOrderAsync(contract, order)
                margin_impact = float(what_if.initMarginChange)

                if margin_impact < current_available_funds:
                    orders_to_place.append((contract, order, decision_data))
                    current_available_funds -= margin_impact
                    current_position_count += 1 # Increment for next check
                    logger.info(f"Approved {display_name}: Margin ${margin_impact:,.2f} | Remaining: ${current_available_funds:,.2f}")
                else:
                    logger.warning(f"Skipping {display_name}: Margin ${margin_impact:,.2f} exceeds Available Funds.")

            except Exception as e:
                logger.error(f"Pre-trade check failed for {display_name}: {e}. Skipping safely.")

        if not orders_to_place:
            logger.warning("No orders fit within available funds or compliance limits. Aborting placement.")
            return

        # Attach event handlers
        ib.orderStatusEvent += on_order_status
        ib.execDetailsEvent += on_exec_details
        ib.errorEvent += on_error  # Error 201 handler

        placed_trades_summary = []
        tuning = config.get('strategy_tuning', {})
        adaptive_interval = tuning.get('adaptive_step_interval_seconds', 15)
        adaptive_pct = tuning.get('adaptive_step_percentage', 0.05)
        
        # Commodity-agnostic tick size (replaces hardcoded COFFEE_OPTIONS_TICK_SIZE)
        TICK_SIZE = get_tick_size(config)

        # --- PLACEMENT LOOP (ENHANCED LOGGING & NOTIFICATIONS) ---
        for contract, order, decision_data in orders_to_place:
            
            # --- 1. Get BAG (Spread) Market Data with a robust polling loop ---
            bag_ticker = ib.reqMktData(contract, '', False, False)
            try:
                await asyncio.sleep(2) # Give it a moment to populate
                start_time = time.time()
                while util.isNan(bag_ticker.bid) and util.isNan(bag_ticker.ask) and util.isNan(bag_ticker.last):
                    await asyncio.sleep(0.1)
                    if (time.time() - start_time) > 15: # 15-second timeout
                        logger.error(f"Timeout waiting for market data for BAG {contract.localSymbol}.")
                        break

                bag_bid = bag_ticker.bid if not util.isNan(bag_ticker.bid) else "N/A"
                bag_ask = bag_ticker.ask if not util.isNan(bag_ticker.ask) else "N/A"
                bag_vol = bag_ticker.volume if not util.isNan(bag_ticker.volume) else "N/A"
                bag_last = bag_ticker.last if not util.isNan(bag_ticker.last) else "N/A"
                bag_last_time = bag_ticker.time.strftime('%H:%M:%S') if bag_ticker.time else "N/A"
            finally:
                try:
                    ib.cancelMktData(contract) # Cleanup
                except Exception:
                    pass
            
            bag_state_str = f"BAG: {bag_bid}x{bag_ask}, V:{bag_vol}, L:{bag_last}@{bag_last_time}"

            # --- 2. Get LEG Market Data ---
            leg_state_strings = await asyncio.gather(
                *[_get_market_data_for_leg(ib, leg) for leg in contract.comboLegs]
            )
            leg_state_for_log = ", ".join(leg_state_strings)

            # --- 3. Create Enhanced Log and Place Order ---
            price_info_log = f"Limit: {order.lmtPrice:.2f}" if order.orderType == "LMT" else "Market Order"
            display_name = _get_order_display_name(contract, strategy_def=decision_data.get('strategy_def'))
            market_state_message = (
                f"Placing Order for {display_name}. {price_info_log}. "
                f"{bag_state_str}. "
                f"LEGs: {leg_state_for_log}"
            )
            logger.info(market_state_message)

            # --- 4. Place the actual order ---
            # Check for Catastrophe Protection Requirement
            enable_protection = config.get('catastrophe_protection', {}).get('enable_directional_stops', False)
            is_directional = decision_data.get('prediction_type') == 'DIRECTIONAL'

            if enable_protection and is_directional and isinstance(contract, Bag):
                # We need the underlying contract. Ideally passed or inferred.
                # In define_directional_strategy, we have future_contract.
                # But here we only have 'contract' (the BAG).
                # We need to re-fetch or infer the underlying future.
                # For safety, let's extract the symbol and expiry from the BAG or first leg.
                # Or better, pass 'future_contract' in decision_data? No.
                # We can construct it.

                # Fetch first leg details
                try:
                    underlying_future = None

                    # --- Primary: Extract from decision_data pipeline (BUG-6 Fix) ---
                    if isinstance(decision_data, dict):
                        strategy_def = decision_data.get('strategy_def')
                        # Amendment C: Explicit None guard
                        if strategy_def and isinstance(strategy_def, dict):
                            future_contract = strategy_def.get('future_contract')
                            if future_contract is not None:
                                underlying_future = future_contract
                                logger.info(f"Catastrophe stop: Resolved underlying future from strategy_def: {underlying_future}")

                    # --- Fallback: Resolve from first leg's underConId ---
                    if underlying_future is None:
                        leg1_conid = contract.comboLegs[0].conId
                        leg_details = await ib.reqContractDetailsAsync(Contract(conId=leg1_conid))

                        if leg_details:
                            # BUG-6 Fix: Use underConId, NOT option expiry
                            # Option expiry != Future expiry for FOPs
                            under_conid = leg_details[0].underConId
                            if under_conid:
                                underlying_future = Contract(conId=under_conid)
                                await ib.qualifyContractsAsync(underlying_future)
                                logger.info(f"Catastrophe stop: Resolved underlying future from underConId: {underlying_future}")
                            else:
                                # Last ditch: try constructing from symbol if underConId missing (risky but better than nothing?)
                                # Actually, without underConId or strategy_def, we can't reliably guess the future contract month
                                # because option expiry is different. Failsafe to standard order.
                                pass

                    if underlying_future is None:
                        raise ValueError("Could not resolve underlying future for catastrophe protection")

                    # Entry Price for Stop Calculation
                    # Use 'price' from decision_data (snapshot price at signal generation)
                    # or 'market_trend_pct' logic?
                    # decision_data['price'] is reliable enough.
                    entry_price_ref = decision_data.get('price')
                    if not entry_price_ref:
                         # Fallback to current ticker last
                         entry_price_ref = ticker.last

                    is_bullish = decision_data.get('direction') == 'BULLISH'

                    trade, stop_trade = await place_directional_spread_with_protection(
                        ib=ib,
                        combo_contract=contract,
                        combo_order=order,
                        underlying_contract=underlying_future,
                        entry_price=entry_price_ref,
                        stop_distance_pct=config.get('catastrophe_protection', {}).get('stop_distance_pct', 0.03),
                        is_bullish_strategy=is_bullish
                    )
                    log_order_event(trade, trade.orderStatus.status, f"{market_state_message} [Protected]")
                except Exception as e:
                    logger.error(f"Failed to place protected order, falling back to standard: {e}")
                    trade = place_order(ib, contract, order)
                    log_order_event(trade, trade.orderStatus.status, market_state_message)
            else:
                trade = place_order(ib, contract, order)
                log_order_event(trade, trade.orderStatus.status, market_state_message)
            
            live_orders[trade.order.orderId] = {
                'trade': trade,
                'status': trade.orderStatus.status,
                'display_name': display_name,  # FIX-004: Store for timeout logging
                'is_filled': False,
                'total_legs': len(contract.comboLegs) if isinstance(contract, Bag) else 1,
                'filled_legs': set(),
                'fill_prices': {}, # Stores fill prices by conId
                'last_update_time': time.time(),
                'adaptive_ceiling_price': getattr(order, 'adaptive_limit_price', None), # Store ceiling/floor
                'decision_data': decision_data,
                # === FIX: Error tracking fields ===
                'last_known_good_price': order.lmtPrice,
                'modification_error_count': 0,
                'max_modification_errors': 3,
            }

            # --- 5. Build Notification String (ENHANCED FOR ALL DATA) ---
            price_info_notify = f"LMT: {order.lmtPrice:.2f}" if order.orderType == "LMT" else "MKT"
            leg_state_for_notify = "<br>    ".join(leg_state_strings) # HTML newline
            readable_symbol = _describe_bag(contract) if hasattr(contract, 'comboLegs') else contract.localSymbol
            summary_line = (
                f"  - {readable_symbol}: {trade.order.action} {trade.order.totalQuantity} @ {price_info_notify}"
            )
            placed_trades_summary.append(summary_line)
            await asyncio.sleep(0.5) # Throttle order placement

        if placed_trades_summary:
            message = f"<b>{len(live_orders)} DAY orders submitted, now monitoring:</b>\n" + "\n".join(placed_trades_summary)
            send_pushover_notification(config.get('notifications', {}), "Orders Placed & Monitored", message)

        # --- Monitoring Loop ---
        start_time = time.time()
        monitoring_duration = 1800  # 30 minutes
        PRICE_TOLERANCE = 0.001  # $0.001 tolerance for floating point comparison

        logger.info(f"Monitoring orders for up to {monitoring_duration / 60} minutes...")
        while time.time() - start_time < monitoring_duration:
            if not live_orders: break

            active_orders_count = 0
            for order_id, details in list(live_orders.items()):
                trade = details['trade']
                # Use the live status from the trade object to ensure accuracy
                status = trade.orderStatus.status

                # --- FIX: Strict Status Check ---
                # Stop processing if the order is already done or dead
                if status in OrderStatus.DoneStates or status in ['Cancelled', 'Inactive', 'ApiCancelled']:
                    continue

                if status in OrderStatus.DoneStates:
                    continue

                active_orders_count += 1

                # --- Adaptive "Walking" Logic ---
                if trade.order.orderType == 'LMT' and details['adaptive_ceiling_price'] is not None:
                    # === L2 FIX: Skip walking for partially filled orders ===
                    # Modifying a partially filled order can reset queue priority
                    # and cause tracking desync. Let it fill naturally.
                    if trade.orderStatus.status == 'PartiallyFilled':
                        filled_qty = trade.orderStatus.filled
                        total_qty = trade.order.totalQuantity
                        if not details.get('partial_fill_logged', False):
                            logger.info(
                                f"Order {order_id}: PartiallyFilled ({filled_qty}/{total_qty}). "
                                f"Skipping price walk to preserve queue priority."
                            )
                            details['partial_fill_logged'] = True
                        continue  # Skip to next iteration without walking

                    # Check if enough time has passed for an update
                    if (time.time() - details['last_update_time']) >= adaptive_interval:
                        current_limit = trade.order.lmtPrice
                        ceiling = details['adaptive_ceiling_price']
                        action = trade.order.action

                        new_price = None

                        # --- DYNAMIC STEP CALCULATION (BUG-2 Fix) ---
                        # Use gap-based stepping to ensure we walk even in narrow bands
                        remaining_gap = abs(ceiling - current_limit)
                        target_steps = config.get('strategy_tuning', {}).get('adaptive_target_steps', 6)
                        gap_based_step = remaining_gap / max(target_steps, 1) if remaining_gap > 0 else 0

                        pct_based_step = abs(current_limit) * adaptive_pct

                        # Use the smaller of gap/pct to ensure granularity, but respect min tick
                        step_amount = max(min(gap_based_step, pct_based_step), TICK_SIZE)

                        # Skip if max errors reached
                        if details['modification_error_count'] >= details['max_modification_errors']:
                            continue

                        if action == 'BUY':
                            if current_limit < ceiling:
                                proposed_price = current_limit + step_amount
                                new_price = round_to_tick(proposed_price, TICK_SIZE, 'BUY')
                                new_price = min(new_price, ceiling)

                                if new_price <= current_limit and current_limit < ceiling:
                                    new_price = min(
                                        round_to_tick(current_limit + TICK_SIZE, TICK_SIZE, 'BUY'),
                                        ceiling
                                    )

                        else: # SELL
                            if current_limit > ceiling: # Here 'ceiling' is actually floor
                                proposed_price = current_limit - step_amount
                                new_price = round_to_tick(proposed_price, TICK_SIZE, 'SELL')
                                new_price = max(new_price, ceiling)

                                if new_price >= current_limit and current_limit > ceiling:
                                    new_price = max(
                                        round_to_tick(current_limit - TICK_SIZE, TICK_SIZE, 'SELL'),
                                        ceiling
                                    )

                        # FIX-001: Floating Point Comparison Bug
                        if new_price is not None:
                            price_actually_changed = abs(new_price - current_limit) > PRICE_TOLERANCE

                            if price_actually_changed:
                                stored_display_name = details.get('display_name', trade.contract.localSymbol or 'UNKNOWN')
                                logger.info(f"Adaptive Update for Order {order_id} ({stored_display_name}): {current_limit} -> {new_price} (Cap/Floor: {ceiling})")
                                trade.order.lmtPrice = new_price
                                ib.placeOrder(trade.contract, trade.order)
                                details['last_update_time'] = time.time()
                                details['last_known_good_price'] = new_price  # Track for rollback
                                details['modification_error_count'] = 0  # Reset on attempt
                                log_order_event(trade, "Updated", f"Adaptive Walk: Price updated to {new_price}")
                            else:
                                # Price at or very near cap/floor - log once then stop checking
                                if not details.get('cap_reached_logged', False):
                                    stored_display_name = details.get('display_name', trade.contract.localSymbol or 'UNKNOWN')
                                    logger.info(f"Order {order_id} ({stored_display_name}): Reached cap/floor at {ceiling:.2f}. Waiting for fill or timeout.")
                                    details['cap_reached_logged'] = True
                                    details['cap_reached_time'] = time.time()

                    # FIX-003: Periodic status report for orders at cap (every 5 minutes)
                    if details.get('cap_reached_logged', False):
                        time_at_cap = time.time() - details.get('cap_reached_time', time.time())
                        if time_at_cap > 0 and int(time_at_cap) % 300 == 0:  # Every 5 minutes
                            logger.info(
                                f"Order {order_id} ({details.get('display_name', 'UNKNOWN')}): "
                                f"Still at cap {ceiling:.2f} for {int(time_at_cap/60)} minutes. "
                                f"Current market: {trade.orderStatus.status}"
                            )

            if active_orders_count == 0:
                logger.info("All orders have reached a terminal state. Concluding monitoring.")
                break

            await asyncio.sleep(1) # Check every second, but update logic respects adaptive_interval
        else:
            logger.warning("Monitoring loop timed out. Some orders may not be in a terminal state.")

        # --- CANCELLATION AND FINAL REPORTING (ENHANCED LOGGING & NOTIFICATIONS) ---
        filled_orders = []
        cancelled_orders = []
        for order_id, details in live_orders.items():
            trade = details['trade']
            # Check the new 'is_filled' flag which confirms all legs are filled
            if details.get('is_filled', False):
                # --- Build Enhanced Fill Notification ---
                price_info = f"LMT: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "MKT"

                # Use the official avgFillPrice from the parent trade status, which is correct for both combos and single legs
                avg_fill_price = trade.orderStatus.avgFillPrice
                readable_symbol = _describe_bag(trade.contract) if hasattr(trade.contract, 'comboLegs') else trade.contract.localSymbol
                fill_line = (
                    f"  - ✅ {readable_symbol}: Filled @ ${avg_fill_price:.2f}"
                )
                filled_orders.append(fill_line)
            
            elif details['status'] not in OrderStatus.DoneStates:
                
                # --- 1. Get FINAL BAG (Spread) Market Data with a robust polling loop ---
                final_bag_ticker = ib.reqMktData(trade.contract, '', False, False)
                try:
                    await asyncio.sleep(2) # Give it a moment to populate
                    start_time = time.time()
                    while util.isNan(final_bag_ticker.bid) and util.isNan(final_bag_ticker.ask) and util.isNan(final_bag_ticker.last):
                        await asyncio.sleep(0.1)
                        if (time.time() - start_time) > 15: # 15-second timeout
                            logger.error(f"Timeout waiting for final market data for BAG {trade.contract.localSymbol}.")
                            break

                    final_bag_bid = final_bag_ticker.bid if not util.isNan(final_bag_ticker.bid) else "N/A"
                    final_bag_ask = final_bag_ticker.ask if not util.isNan(final_bag_ticker.ask) else "N/A"
                    final_bag_vol = final_bag_ticker.volume if not util.isNan(final_bag_ticker.volume) else "N/A"
                    final_bag_last = final_bag_ticker.last if not util.isNan(final_bag_ticker.last) else "N/A"
                    final_bag_last_time = final_bag_ticker.time.strftime('%H:%M:%S') if final_bag_ticker.time else "N/A"
                finally:
                    try:
                        ib.cancelMktData(trade.contract)
                    except Exception:
                        pass
                
                final_bag_state = (
                    f"BAG: {final_bag_bid}x{final_bag_ask}, V:{final_bag_vol}, L:{final_bag_last}@{final_bag_last_time}"
                )

                # --- 2. Get FINAL LEG Market Data ---
                final_leg_state_strings = await asyncio.gather(
                    *[_get_market_data_for_leg(ib, leg) for leg in trade.contract.comboLegs]
                )
                final_leg_state_for_log = ", ".join(final_leg_state_strings)

                # --- 3. Create Enhanced Log and Cancel Order ---
                price_info_log = f"Original Limit: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "Original Order: MKT"
                stored_display_name = details.get('display_name', 'UNKNOWN')
                log_message = (
                    f"Order {trade.order.orderId} ({stored_display_name}) TIMED OUT. "
                    f"{price_info_log}. "
                    f"Final {final_bag_state}. "
                    f"Final LEGs: {final_leg_state_for_log}"
                )
                logger.warning(log_message)
                log_order_event(trade, "TimedOut", log_message)

                ib.cancelOrder(trade.order)

                # --- 4. Update Notification String (ENHANCED FOR ALL DATA) ---
                price_info_notify = f"LMT: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "MKT"
                final_leg_state_for_notify = "<br>    ".join(final_leg_state_strings) # HTML newline
                readable_symbol = _describe_bag(trade.contract) if hasattr(trade.contract, 'comboLegs') else trade.contract.localSymbol
                cancel_line = (
                    f"  - {readable_symbol}: {price_info_notify} — Unfilled, cancelled"
                )
                cancelled_orders.append(cancel_line)
                await asyncio.sleep(0.2)

        # Build and send the final summary notification
        message_parts = []
        if filled_orders:
            message_parts.append(f"<b>✅ {len(filled_orders)} orders were successfully filled:</b>")
            message_parts.extend(filled_orders)
        if cancelled_orders:
            message_parts.append(f"\n<b>❌ {len(cancelled_orders)} unfilled orders were cancelled:</b>")
            message_parts.extend(cancelled_orders)

        if not message_parts:
            summary_message = "Order monitoring complete. No orders were filled or needed cancellation."
        else:
            summary_message = "\n".join(message_parts)

        send_pushover_notification(config.get('notifications', {}), "Order Monitoring Complete", summary_message)

        # Queue is already cleared via pop_all() at the start

        logger.info("--- Finished monitoring and cleanup. ---")

    except Exception as e:
        msg = f"A critical error occurred during order placement/monitoring: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Placement CRITICAL", msg)
    finally:
        # Clear thesis tracking for next run (FIX-005)
        _recorded_thesis_positions.clear()

        if ib.isConnected():
            ib.orderStatusEvent -= on_order_status
            ib.execDetailsEvent -= on_exec_details
            ib.errorEvent -= on_error
            # Do NOT disconnect pooled connection
            logger.info("Order placement complete. Releasing event handlers.")

import pandas as pd
from datetime import datetime, date, timedelta
import os

def get_trade_ledger_df():
    """
    Reads and consolidates the main and archived trade ledgers for analysis.
    This function is now robust to historical ledgers that may be missing
    the 'position_id' column, using 'combo_id' as a fallback.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive_ledger')

    dataframes = []

    def load_and_prepare_ledger(file_path):
        """Loads a single ledger file and ensures it has a position_id."""
        df = pd.read_csv(file_path)
        if 'position_id' not in df.columns:
            logger.warning(f"File '{file_path}' is missing 'position_id'. Using 'combo_id' as fallback.")
            if 'combo_id' in df.columns:
                df['position_id'] = df['combo_id']
            else:
                logger.error(f"Cannot create fallback 'position_id' as 'combo_id' is also missing in '{file_path}'.")
                # Return an empty dataframe with the expected columns if a critical column is missing
                return pd.DataFrame(columns=['timestamp', 'position_id']) # Ensure it has minimal columns
        return df

    if os.path.exists(ledger_path):
        dataframes.append(load_and_prepare_ledger(ledger_path))

    if os.path.exists(archive_dir):
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        for file in archive_files:
            dataframes.append(load_and_prepare_ledger(file))

    if not dataframes:
        return pd.DataFrame(columns=['timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity', 'reason'])

    full_ledger = pd.concat(dataframes, ignore_index=True)

    # Ensure timestamp column exists and is in datetime format
    if 'timestamp' in full_ledger.columns:
        full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp'])
        return full_ledger.sort_values(by='timestamp').reset_index(drop=True)
    else:
        logger.error("Consolidated ledger is missing the 'timestamp' column.")
        return pd.DataFrame(columns=['timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity', 'reason'])


async def close_stale_positions(config: dict, connection_purpose: str = "orchestrator_orders"):
    """
    Closes any open positions that have been held for more than 'max_holding_days' (Default: 2) trading days.
    This function now uses the live portfolio from IB as the source of truth,
    reconstructs position ages using a FIFO stack from the ledger, and
    groups closing orders by Position ID to ensure spreads are closed as combos.
    """
    max_holding_days = config.get('risk_management', {}).get('max_holding_days', 2)
    logger.info(f"--- Initiating position closing based on {max_holding_days}-day holding period ---")

    # --- Weekly Close Logic ---
    today_dt = datetime.now(timezone.utc)
    today_date = today_dt.date()
    weekday = today_date.weekday()  # 0=Mon, 4=Fri

    is_weekly_close = False
    weekly_close_reason = ""

    # Guard: Do not run on weekends
    if weekday >= 5:  # Saturday(5) or Sunday(6)
        logger.warning("Today is a weekend. Skipping position closing check.")
        return

    from trading_bot.calendars import get_exchange_calendar
    cal = get_exchange_calendar(config.get('exchange', 'ICE'))

    # Check if today is Friday
    if weekday == 4:
        is_weekly_close = True
        weekly_close_reason = "Friday Weekly Close"

    # Check if today is Thursday and tomorrow is a holiday
    elif weekday == 3:
        tomorrow = today_date + timedelta(days=1)
        holidays = cal.holidays(start=tomorrow, end=tomorrow)
        if not holidays.empty:
            is_weekly_close = True
            weekly_close_reason = "Holiday Tomorrow (Weekly Close)"

    if is_weekly_close:
        logger.info(f"--- Weekly Close Triggered: {weekly_close_reason}. Closing ALL positions. ---")

    closed_position_details = []
    failed_closes = []
    kept_positions_details = []
    orphaned_positions = []

    try:
        # --- 1. Connect to IB and get the ground truth of current positions ---

        # --- RETRY LOGIC (Issue 11 Fix) ---
        MAX_CONNECT_RETRIES = 3
        RETRY_DELAYS = [5.0, 10.0, 20.0]  # Exponential backoff

        ib = None
        for attempt in range(MAX_CONNECT_RETRIES):
            try:
                ib = await IBConnectionPool.get_connection(connection_purpose, config)
                break  # Success
            except (TimeoutError, ConnectionError, Exception) as e:
                if attempt < MAX_CONNECT_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"close_stale_positions connection attempt {attempt + 1}/{MAX_CONNECT_RETRIES} "
                        f"failed: {e}. Retrying in {delay}s..."
                    )
                    # Force reset the connection before retry
                    await IBConnectionPool._force_reset_connection(connection_purpose)
                    await asyncio.sleep(delay)
                else:
                    logger.critical(
                        f"close_stale_positions FAILED after {MAX_CONNECT_RETRIES} attempts. "
                        f"Positions may remain open!"
                    )
                    send_pushover_notification(
                        config.get('notifications', {}),
                        "🚨 CRITICAL: Position Close Failed",
                        f"close_stale_positions could not connect after {MAX_CONNECT_RETRIES} attempts. "
                        f"Manual intervention required to close positions before market close."
                    )
                    return  # Cannot proceed without connection

        if ib is None:
            return

        configure_market_data_type(ib)

        logger.info(f"Connected to IB for closing positions (purpose: {connection_purpose}).")

        live_positions = await ib.reqPositionsAsync()
        if not live_positions:
            logger.info("No open positions found in the IB account. Nothing to close.")
            return

        logger.info(f"Found {len(live_positions)} live position legs in the IB account.")

        non_zero_positions = [p for p in live_positions if p.position != 0]
        if len(non_zero_positions) != len(live_positions):
            logger.info(
                f"  Filtered to {len(non_zero_positions)} positions with non-zero quantity "
                f"({len(live_positions) - len(non_zero_positions)} already closed/zero-qty excluded)"
            )
        if not non_zero_positions:
            logger.info("All returned positions have zero quantity — nothing to close.")
            return

        # --- 2. Load trade ledger to get historical data (open dates) ---
        trade_ledger = get_trade_ledger_df()
        if trade_ledger.empty:
            logger.warning("Trade ledger is empty. Cannot determine position ages. Skipping closure.")
            return

        # Prepare Ledger: Sort by Timestamp DESCENDING (Newest First) for reconstruction
        trade_ledger.sort_values('timestamp', ascending=False, inplace=True)

        # --- 3. Calculate trading day logic ---
        # M4 FIX: Use Exchange Holiday Calendar
        cal = get_exchange_calendar(config.get('exchange', 'ICE'))
        holidays = cal.holidays(start=date(date.today().year - 1, 1, 1), end=date.today()).to_pydatetime().tolist()
        today = datetime.now(timezone.utc).date()
        custom_bday = pd.offsets.CustomBusinessDay(holidays=holidays)

        # Dictionary to store legs identified for closing, grouped by Position ID
        # Structure: { position_id: [ {contract_details}, ... ] }
        positions_to_close = defaultdict(list)

        # --- 4. FIFO Reconstruction and Age Verification ---
        # Map unique local symbols to live positions
        # Note: If multiple accounts, this assumes unique local_symbol per account or aggregated.
        live_pos_map = {p.contract.localSymbol: p for p in live_positions if p.position != 0}

        for symbol, pos in live_pos_map.items():
            live_qty = abs(pos.position)
            live_action_sign = 1 if pos.position > 0 else -1 # 1 for Long, -1 for Short

            # Filter ledger for this symbol and matching direction
            # If Live is Long (+), we look for 'BUY' trades in ledger.
            # If Live is Short (-), we look for 'SELL' trades in ledger.
            target_ledger_action = 'BUY' if live_action_sign > 0 else 'SELL'

            symbol_ledger = trade_ledger[
                (trade_ledger['local_symbol'] == symbol) &
                (trade_ledger['action'] == target_ledger_action)
            ].copy()

            if symbol_ledger.empty:
                if is_weekly_close:
                    # WEEKLY CLOSE OVERRIDE: Close orphaned positions too
                    # Safety > accuracy — we can't leave unknown positions open over the weekend
                    close_action = 'SELL' if live_action_sign > 0 else 'BUY'
                    positions_to_close[f"ORPHAN_{symbol}"].append({
                        'conId': pos.contract.conId,
                        'symbol': symbol,
                        'exchange': pos.contract.exchange or 'SMART',
                        'action': close_action,
                        'quantity': live_qty,
                        'multiplier': pos.contract.multiplier,
                        'currency': pos.contract.currency
                    })
                    logger.warning(
                        f"WEEKLY CLOSE OVERRIDE: Closing orphaned position {symbol} "
                        f"(Qty: {pos.position}) — no ledger match found"
                    )
                else:
                    orphaned_positions.append(f"{symbol} (Qty: {pos.position})")
                continue

            # Reconstruct the stack (Newest -> Oldest)
            accumulated_qty = 0

            for _, trade in symbol_ledger.iterrows():
                if accumulated_qty >= live_qty:
                    break # We have accounted for all live shares

                trade_qty = abs(trade['quantity'])
                qty_needed = live_qty - accumulated_qty
                qty_to_attribute = min(trade_qty, qty_needed)

                accumulated_qty += qty_to_attribute

                # Check Age
                trade_date = trade['timestamp']
                trading_days = pd.date_range(start=trade_date.date(), end=today, freq=custom_bday)
                age_in_trading_days = len(trading_days)

                logger.info(f"Reconstruction: {symbol} | Trade Date: {trade_date.date()} | Age: {age_in_trading_days} | Qty: {qty_to_attribute}")

                if is_weekly_close or age_in_trading_days >= max_holding_days:
                    # This specific lot is expired or forced closed. Mark for closure.
                    # We need to INVERT the action to close.
                    close_action = 'SELL' if target_ledger_action == 'BUY' else 'BUY'

                    positions_to_close[trade['position_id']].append({
                        'conId': pos.contract.conId,
                        'symbol': symbol,
                        'exchange': pos.contract.exchange,
                        'action': close_action,
                        'quantity': qty_to_attribute,
                        'multiplier': pos.contract.multiplier,
                        'currency': pos.contract.currency
                    })
                else:
                    kept_positions_details.append(f"{trade['position_id']} ({symbol}) - Age: {age_in_trading_days} days")

        # --- 5. Execution Phase (Grouping by Position ID) ---
        for pos_id, legs in positions_to_close.items():
            logger.info(f"Processing closure for Position ID {pos_id} with {len(legs)} legs.")

            try:
                # Deduplicate legs if necessary (though our logic shouldn't produce duplicates for same symbol/pos_id combo usually)
                # But we might have multiple 'lots' for the same symbol if they are split in ledger?
                # Actually, if we have 2 expired lots for same symbol/pos_id, we should sum them up.
                combined_legs = defaultdict(lambda: {'quantity': 0, 'details': None})
                for leg in legs:
                    key = leg['conId']
                    combined_legs[key]['quantity'] += leg['quantity']
                    combined_legs[key]['details'] = leg

                final_legs_list = [v['details'] for v in combined_legs.values()]
                # Update quantities in the list
                for leg in final_legs_list:
                    leg['quantity'] = combined_legs[leg['conId']]['quantity']

                if len(final_legs_list) > 1:
                    # Create COMBO (Bag) Order
                    contract = Contract()
                    contract.symbol = config.get('symbol', 'KC')
                    contract.secType = "BAG"
                    contract.currency = final_legs_list[0]['currency']
                    contract.exchange = final_legs_list[0]['exchange'] or config.get('exchange', 'SMART')

                    combo_legs = []
                    for leg in final_legs_list:
                        combo_leg = ComboLeg()
                        combo_leg.conId = leg['conId']
                        combo_leg.ratio = int(leg['quantity']) # Assuming integer ratios for now
                        combo_leg.action = leg['action']
                        combo_leg.exchange = leg['exchange'] or config.get('exchange', 'SMART')
                        combo_legs.append(combo_leg)

                    contract.comboLegs = combo_legs

                    # For BAG orders, the totalQuantity is usually 1 (multiplied by ratio)
                    # Use the GCD of quantities if possible, but for simple closes, usually ratio is quantity and bag size is 1.
                    # Or ratio is 1 and bag size is quantity?
                    # If we have 5 spreads, we usually set ratio 1, size 5.
                    # Here we might have different quantities for different legs? (Unbalanced close).
                    # If quantities differ (e.g. 2 calls, 1 put), we can't do a simple 1:1 spread.
                    # We will assume balanced spread or use ratio=qty, size=1.

                    # Logic: Find GCD of quantities?
                    # Simplification: Set Ratio = Quantity, Bag Order Size = 1.
                    # This works for "Closing what we have".

                    order_size = 1

                    logger.info(f"Constructed BAG order for Pos ID {pos_id}: {[l.action + ' ' + str(l.ratio) + ' x ' + str(l.conId) for l in combo_legs]}")

                else:
                    # Single Leg Order
                    leg = final_legs_list[0]

                    # CRITICAL FIX: Re-qualify by conId only to get correct strike format
                    minimal = Contract(conId=leg['conId'])
                    qualified = await ib.qualifyContractsAsync(minimal)
                    if qualified and qualified[0].conId != 0:
                        contract = qualified[0]
                    else:
                        logger.error(f"Could not re-qualify conId {leg['conId']} for close order")
                        # Fallback to manual construction (risky)
                        contract = Contract()
                        contract.conId = leg['conId']
                        contract.symbol = config.get('symbol', 'KC')
                        contract.secType = "FOP"

                    contract.exchange = leg['exchange'] or config.get('exchange', 'SMART')

                    order_size = leg['quantity']
                    logger.info(f"Constructed SINGLE order for Pos ID {pos_id}: {leg['action']} {order_size} {leg['symbol']}")

                # Place Order
                # For Bag, action is usually BUY or SELL. If we set leg actions explicitly, Bag action determines the side of the trade?
                # "For a Combo order, the action is determined by the actions of the legs."
                # Actually, IB Bag orders have an action. "BUY" a spread usually means Buy Leg 1, Sell Leg 2.
                # Here we set explicit actions in ComboLegs. The Bag order action is effectively "BUY" (to execute the legs as defined) or we match.
                # Standard practice: Set Bag Action to "BUY" and define leg actions as absolute.

                bag_action = 'BUY'
                if len(final_legs_list) == 1:
                    bag_action = final_legs_list[0]['action']

                # === IMPROVED: Leg-Based Price Calculation for Combo ===
                # IB's native BAG pricing is unreliable - calculate from legs
                lmt_price = 0.0
                MINIMUM_TICK = get_tick_size(config)  # Profile-driven tick size

                if contract.secType == "BAG" and contract.comboLegs:
                    # Calculate combo price from individual leg prices
                    logger.info(f"Calculating combo price from {len(contract.comboLegs)} legs...")
                    leg_prices_valid = True
                    calculated_combo_price = 0.0

                    for combo_leg in contract.comboLegs:
                        try:
                            # Fetch individual leg contract
                            leg_contract = Contract(conId=combo_leg.conId)
                            await ib.qualifyContractsAsync(leg_contract)

                            leg_ticker = ib.reqMktData(leg_contract, '', True, False)
                            await asyncio.sleep(1.5)  # Slightly longer wait for leg data

                            # Get leg price (prefer mid, fallback to last)
                            leg_bid = leg_ticker.bid if not util.isNan(leg_ticker.bid) else 0
                            leg_ask = leg_ticker.ask if not util.isNan(leg_ticker.ask) else 0
                            leg_last = leg_ticker.last if not util.isNan(leg_ticker.last) else 0

                            ib.cancelMktData(leg_contract)

                            if leg_bid > 0 and leg_ask > 0:
                                leg_mid = (leg_bid + leg_ask) / 2
                            elif leg_last > 0:
                                leg_mid = leg_last
                            else:
                                logger.warning(f"Leg {combo_leg.conId} ({leg_contract.localSymbol}): No valid price data")
                                leg_prices_valid = False
                                break

                            # Aggregate: BUY legs add to cost, SELL legs subtract
                            if combo_leg.action == 'BUY':
                                calculated_combo_price += leg_mid
                            else:  # SELL
                                calculated_combo_price -= leg_mid

                            logger.debug(f"Leg {leg_contract.localSymbol}: {combo_leg.action} @ {leg_mid:.2f}")

                        except Exception as e:
                            logger.warning(f"Failed to price leg {combo_leg.conId}: {e}")
                            leg_prices_valid = False
                            break

                    if leg_prices_valid:
                        # Add slippage buffer for execution (2% of combo value, minimum 1 tick)
                        slippage_buffer = max(abs(calculated_combo_price) * 0.02, MINIMUM_TICK)

                        # === UNIVERSAL AGGRESSION LOGIC ===
                        # BUY: Always ADD buffer (Higher mathematical value = Aggressive)
                        # SELL: Always SUBTRACT buffer (Lower mathematical value = Aggressive)
                        if bag_action == 'BUY':
                            lmt_price = calculated_combo_price + slippage_buffer
                        else:
                            lmt_price = calculated_combo_price - slippage_buffer

                        # Round to tick
                        lmt_price = round(lmt_price / MINIMUM_TICK) * MINIMUM_TICK
                        logger.info(f"Calculated combo limit price: {lmt_price:.2f} (raw: {calculated_combo_price:.2f}, slippage: {slippage_buffer:.2f})")
                else:
                    # Single leg - use standard price fetch
                    ticker = ib.reqMktData(contract, '', True, False)
                    try:
                        await asyncio.sleep(2)
                        if bag_action == 'BUY':
                            lmt_price = ticker.ask if not util.isNan(ticker.ask) and ticker.ask > 0 else ticker.last
                        else:
                            lmt_price = ticker.bid if not util.isNan(ticker.bid) and ticker.bid > 0 else ticker.last

                        # Apply minimum tick for single legs (always positive)
                        if lmt_price and not util.isNan(lmt_price):
                            lmt_price = max(lmt_price, MINIMUM_TICK)
                    finally:
                        ib.cancelMktData(contract)

                # === PRICE VALIDATION (Handles both Credit and Debit) ===
                is_invalid_price = False

                if util.isNan(lmt_price):
                    is_invalid_price = True
                    logger.warning(f"Limit price is NaN for {contract.symbol}")
                elif lmt_price == 0.0:
                    is_invalid_price = True
                    logger.warning(f"Limit price is zero for {contract.symbol}")
                elif abs(lmt_price) < MINIMUM_TICK:
                    # Price magnitude too small (whether credit or debit)
                    is_invalid_price = True
                    logger.warning(f"Limit price magnitude {abs(lmt_price):.4f} below minimum tick {MINIMUM_TICK}")

                if is_invalid_price:
                    logger.warning(f"Invalid limit price for {contract.symbol}. Using Market Order.")
                    order = MarketOrder(bag_action, order_size)
                else:
                    price_type = 'CREDIT' if lmt_price < 0 else 'DEBIT'
                    logger.info(f"Placing LIMIT order for {contract.symbol}. Price: {lmt_price:.2f} ({price_type})")
                    order = LimitOrder(bag_action, order_size, round(lmt_price, 2))

                order.outsideRth = True

                trade = place_order(ib, contract, order)
                logger.info(f"Placed close order {trade.order.orderId} for Pos ID {pos_id}. Waiting for fill...")

                # === CUSTOM ADAPTIVE PRICE WALKING FOR CLOSING ORDERS ===
                # IB Algo does NOT support BAG orders - must use custom logic

                fill_detected = False
                INITIAL_TIMEOUT_SECONDS = 45
                PRICE_WALK_INTERVAL = 5  # Walk price every 5 seconds
                MAX_WALKS = 6  # Maximum price adjustments
                WALK_INCREMENT_PCT = 0.01  # 1% per walk

                is_limit_order = isinstance(order, LimitOrder)
                initial_price = order.lmtPrice if is_limit_order else 0.0

                # === UNIVERSAL CEILING/FLOOR CALCULATION ===
                # BUY: Ceiling is HIGHER (more aggressive), Floor is LOWER (passive)
                # SELL: Floor is LOWER (more aggressive), Ceiling is HIGHER (passive)
                price_range = abs(initial_price) * 0.10 if initial_price != 0 else MINIMUM_TICK * 10

                if bag_action == 'BUY':
                    ceiling_price = initial_price + price_range  # Max we'll pay (aggressive limit)
                    floor_price = initial_price - price_range    # Not used for BUY
                else:
                    floor_price = initial_price - price_range    # Min we'll accept (aggressive limit)
                    ceiling_price = initial_price + price_range  # Not used for SELL

                walk_count = 0
                elapsed = 0

                while elapsed < INITIAL_TIMEOUT_SECONDS:
                    await asyncio.sleep(1)
                    elapsed += 1

                    # Check for fill
                    if trade.orderStatus.status == OrderStatus.Filled:
                        fill_detected = True
                        break
                    elif trade.orderStatus.status in [OrderStatus.Cancelled, OrderStatus.Inactive]:
                        logger.warning(f"Order {trade.order.orderId} cancelled/inactive before fill")
                        break

                    # === UNIVERSAL PRICE WALKING LOGIC ===
                    # === L2 FIX: Skip walking for partially filled close orders ===
                    if trade.orderStatus.status == 'PartiallyFilled':
                        filled = trade.orderStatus.filled
                        logger.info(f"Close order {trade.order.orderId}: PartiallyFilled ({filled}). Preserving queue position.")
                        continue

                    if is_limit_order and elapsed % PRICE_WALK_INTERVAL == 0 and walk_count < MAX_WALKS:
                        current_price = trade.order.lmtPrice

                        # Calculate walk amount (1% of absolute price, minimum 1 tick)
                        walk_amount = max(abs(current_price) * WALK_INCREMENT_PCT, MINIMUM_TICK) if current_price != 0 else MINIMUM_TICK

                        # BUY: Walk UP (add) towards ceiling
                        # SELL: Walk DOWN (subtract) towards floor
                        if bag_action == 'BUY':
                            new_price = current_price + walk_amount
                            new_price = min(new_price, ceiling_price)  # Don't exceed ceiling
                        else:
                            new_price = current_price - walk_amount
                            new_price = max(new_price, floor_price)    # Don't go below floor

                        # Round to tick
                        new_price = round(new_price / MINIMUM_TICK) * MINIMUM_TICK

                        if new_price != current_price:
                            walk_count += 1
                            price_type = 'CREDIT' if new_price < 0 else 'DEBIT'
                            logger.info(f"Price walk #{walk_count}: {current_price:.2f} -> {new_price:.2f} ({price_type})")

                            trade.order.lmtPrice = new_price
                            ib.placeOrder(trade.contract, trade.order)  # Modify existing order

                # If still not filled after adaptive walking, convert to Market
                if not fill_detected and trade.orderStatus.status == OrderStatus.Submitted:
                    logger.warning(f"Order {trade.order.orderId} not filled after {elapsed}s and {walk_count} walks. Converting to Market.")

                    # Cancel the limit order
                    ib.cancelOrder(trade.order)
                    await asyncio.sleep(2)

                    # Place market order
                    market_order = MarketOrder(trade.order.action, trade.order.totalQuantity)
                    market_order.outsideRth = True
                    market_trade = place_order(ib, trade.contract, market_order)

                    # Brief wait for market fill
                    for _ in range(15):
                        await asyncio.sleep(1)
                        if market_trade.orderStatus.status == OrderStatus.Filled:
                            trade = market_trade
                            fill_detected = True
                            logger.info(f"Market order filled at {market_trade.orderStatus.avgFillPrice:.2f}")
                            break

                    if not fill_detected:
                        logger.error(f"CRITICAL: Market order {market_trade.order.orderId} also failed to fill!")

                if fill_detected and trade.orderStatus.status == OrderStatus.Filled:
                    # Log to ledger.
                    # IMPORTANT: For Bag orders, we want to log the LEGS.
                    # log_trade_to_ledger handles extraction of fills if we pass the trade.
                    # But trade.fills for Bag might be empty if we didn't receive execDetails?
                    # IB returns separate execDetails for legs.
                    # We need to make sure we capture them.
                    # Since we are not running a full event loop listener here (just polling status),
                    # trade.fills might NOT be populated if 'execDetails' event didn't fire in the background.
                    # ib.sleep() or asyncio.sleep() in ib_insync allows background processing?
                    # Yes, asyncio.sleep() allows loop to run. ib_insync updates trade.fills automatically via events.

                    # We need to wait a bit more for fills to populate?
                    await asyncio.sleep(1)

                    await log_trade_to_ledger(ib, trade, "Stale Position Close", position_id=pos_id)

                    # Calculate PnL for report
                    # Sum realizedPNL from all fills
                    pnl = sum(f.commissionReport.realizedPNL for f in trade.fills if f.commissionReport)
                    price_str = f"{trade.orderStatus.avgFillPrice}"

                    desc = "Combo" if len(final_legs_list) > 1 else final_legs_list[0]['symbol']
                    closed_position_details.append({
                        "symbol": desc,
                        "action": "CLOSE",
                        "quantity": order_size, # Might be 1 for bag
                        "price": trade.orderStatus.avgFillPrice,
                        "pnl": pnl
                    })
                    logger.info(f"Successfully closed {desc} (Pos ID: {pos_id}). P&L: {pnl}")

                    # === CRITICAL: Invalidate thesis in TMS ===
                    try:
                        tms = TransactiveMemory()
                        close_reason = weekly_close_reason if is_weekly_close else f"Stale position (>{max_holding_days} days)"
                        tms.invalidate_thesis(pos_id, close_reason)
                        logger.info(f"TMS: Invalidated thesis for {pos_id}: {close_reason}")
                    except Exception as thesis_err:
                        logger.warning(f"TMS: Failed to invalidate thesis for {pos_id}: {thesis_err}")

                    # --- CLEANUP: Cancel Orphaned Catastrophe Stops ---
                    # The stop order has orderRef = CATASTROPHE_{pos_id}
                    # (since pos_id IS the original orderRef)
                    await close_spread_with_protection_cleanup(ib, None, f"CATASTROPHE_{pos_id}")

                else:
                    logger.warning(f"Order {trade.order.orderId} for Pos ID {pos_id} timed out or failed. Status: {trade.orderStatus.status}")
                    failed_closes.append(f"Pos ID {pos_id}: Status {trade.orderStatus.status}")
                    if trade.orderStatus.status not in OrderStatus.DoneStates:
                        ib.cancelOrder(trade.order)

                await asyncio.sleep(1) # Throttle

            except Exception as ex:
                logger.error(f"Error closing position {pos_id}: {ex}")
                failed_closes.append(f"Pos ID {pos_id}: Error {ex}")

        # --- 6. POST-CLOSE VERIFICATION (Weekly Close Only) ---
        if is_weekly_close and (closed_position_details or positions_to_close):
            logger.info("--- Post-Close Verification: Re-checking IB positions ---")
            await asyncio.sleep(5)  # Allow IB to settle

            try:
                verify_positions = await ib.reqPositionsAsync()
                remaining = [p for p in (verify_positions or []) if p.position != 0]

                if remaining:
                    remaining_symbols = [p.contract.localSymbol for p in remaining]
                    logger.critical(
                        f"⚠️ POST-CLOSE VERIFICATION FAILED: "
                        f"{len(remaining)} positions still open: {remaining_symbols}"
                    )

                    # RETRY: Attempt individual market orders for each remaining leg
                    for pos in remaining:
                        try:
                            close_action = 'SELL' if pos.position > 0 else 'BUY'
                            qty = abs(pos.position)
                            order = MarketOrder(close_action, qty)

                            # Re-qualify to get correct strike format
                            minimal = Contract(conId=pos.contract.conId)
                            requalified = await ib.qualifyContractsAsync(minimal)
                            close_contract = requalified[0] if requalified and requalified[0].conId != 0 else pos.contract

                            trade = ib.placeOrder(close_contract, order)
                            logger.warning(
                                f"RETRY: Sent {close_action} {qty} {pos.contract.localSymbol}"
                            )
                            await asyncio.sleep(2)
                        except Exception as retry_e:
                            logger.critical(
                                f"RETRY FAILED for {pos.contract.localSymbol}: {retry_e}"
                            )
                            failed_closes.append(
                                f"{pos.contract.localSymbol} (RETRY FAILED: {retry_e})"
                            )

                    # Re-verify after retries
                    await asyncio.sleep(5)
                    final_check = await ib.reqPositionsAsync()
                    final_remaining = [p for p in (final_check or []) if p.position != 0]

                    if final_remaining:
                        final_symbols = [p.contract.localSymbol for p in final_remaining]
                        alert_msg = (
                            f"🚨 CRITICAL: {len(final_remaining)} positions STILL OPEN "
                            f"after weekly close + retry!\n"
                            f"Symbols: {', '.join(final_symbols)}\n"
                            f"MANUAL INTERVENTION REQUIRED"
                        )
                        logger.critical(alert_msg)
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "🚨 WEEKLY CLOSE FAILED",
                            alert_msg
                        )
                    else:
                        logger.info("✅ Post-close retry successful — all positions now flat.")

                        # === CRITICAL FIX: Sweep-invalidate orphaned theses ===
                        # The retry path closes positions via individual market orders
                        # but has no position_id context to call invalidate_thesis().
                        # Now that we've confirmed IB is flat, sweep ALL active theses
                        # and invalidate any that were part of this close batch.
                        try:
                            sweep_tms = TransactiveMemory()
                            if sweep_tms.collection:
                                active_results = sweep_tms.collection.get(
                                    where={"active": "true"},
                                    include=['metadatas']
                                )
                                swept_count = 0
                                for meta in active_results.get('metadatas', []):
                                    tid = meta.get('trade_id')
                                    if tid and tid in positions_to_close:
                                        sweep_tms.invalidate_thesis(
                                            tid,
                                            f"Post-close sweep: {weekly_close_reason}"
                                        )
                                        swept_count += 1
                                        logger.info(f"TMS sweep: Invalidated orphaned thesis {tid}")

                                if swept_count > 0:
                                    logger.warning(
                                        f"TMS sweep: Invalidated {swept_count} orphaned theses "
                                        f"after post-close retry"
                                    )
                        except Exception as sweep_err:
                            logger.warning(f"TMS sweep failed (non-fatal): {sweep_err}")
                else:
                    logger.info("✅ Post-close verification passed — all positions flat.")

                    # Sweep-invalidate any theses that were marked for close
                    # but whose invalidation may have silently failed
                    try:
                        sweep_tms = TransactiveMemory()
                        if sweep_tms.collection:
                            active_results = sweep_tms.collection.get(
                                where={"active": "true"},
                                include=['metadatas']
                            )
                            swept_count = 0
                            for meta in active_results.get('metadatas', []):
                                tid = meta.get('trade_id')
                                if tid and tid in positions_to_close:
                                    sweep_tms.invalidate_thesis(
                                        tid,
                                        f"Post-close sweep: {weekly_close_reason if is_weekly_close else 'Stale position close'}"
                                    )
                                    swept_count += 1
                            if swept_count > 0:
                                logger.warning(
                                    f"TMS sweep: Fixed {swept_count} thesis(es) that "
                                    f"survived initial invalidation"
                                )
                    except Exception as sweep_err:
                        logger.warning(f"TMS sweep failed (non-fatal): {sweep_err}")

            except Exception as verify_e:
                logger.error(f"Post-close verification failed: {verify_e}")

        # --- 7. Build and send a comprehensive notification ---
        message_parts = []
        total_pnl = 0
        if closed_position_details:
            message_parts.append(f"<b>✅ Successfully closed {len(closed_position_details)} positions (Stale > {max_holding_days}d):</b>")
            for detail in closed_position_details:
                pnl_str = f"${detail['pnl']:,.2f}"
                pnl_color = "green" if detail['pnl'] >= 0 else "red"
                message_parts.append(
                    f"  - <b>{detail['symbol']}</b>: {detail['action']} {abs(detail['quantity'])} "
                    f"@ {detail['price']}. P&L: <font color='{pnl_color}'>{pnl_str}</font>"
                )
                total_pnl += detail.get('pnl', 0)
            message_parts.append(f"<b>Total Realized P&L: ${total_pnl:,.2f}</b>")

        if failed_closes:
            message_parts.append(f"\n<b>❌ Failed to close {len(failed_closes)} positions:</b>")
            message_parts.extend([f"  - {reason}" for reason in failed_closes])

        if kept_positions_details:
            unique_kept = sorted(list(set(kept_positions_details)))
            message_parts.append(f"\n<b> Positions held ({len(unique_kept)} unique legs/combos):</b>")
            message_parts.extend([f"  - {reason}" for reason in unique_kept])

        if orphaned_positions:
             message_parts.append(f"\n<b>️ Orphaned Positions Found:</b>")
             message_parts.extend([f"  - {pos}" for pos in orphaned_positions])

        if not message_parts:
            if is_weekly_close:
                message = "Weekly Close triggered, but no open positions were found to close."
            else:
                message = f"No positions were eligible for closing today based on the {max_holding_days}-day rule."
        else:
            message = "\n".join(message_parts)

        if is_weekly_close:
            notification_title = f"Weekly Market Close Report: P&L ${total_pnl:,.2f}"
        else:
            notification_title = f"Stale Position Close Report: P&L ${total_pnl:,.2f}"

        send_pushover_notification(config.get('notifications', {}), notification_title, message)

    except Exception as e:
        msg = f"A critical error occurred while closing positions: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Position Closing CRITICAL", msg)
    finally:
        # Do not disconnect pool
        pass

async def record_entry_thesis_for_trade(
    position_id: str,
    strategy_type: str,
    decision: dict,
    entry_price: float,
    config: dict,
    underlying_price: float = None,  # NEW: Underlying future price
    contract_month: str = None       # NEW: For contract reconstruction
):
    """Records the entry thesis for a newly opened position.

    Args:
        position_id: Unique identifier for the position
        strategy_type: Type of strategy (IRON_CONDOR, LONG_STRADDLE, etc.)
        decision: Council decision dict
        entry_price: Spread credit/debit (for options strategies)
        config: Application config
        underlying_price: Current underlying future price (CRITICAL for Iron Condor)
        contract_month: Contract month for underlying (e.g., '202603')
    """
    tms = TransactiveMemory()

    # Determine guardian agent based on the primary driver
    reason = decision.get('reason', '')
    guardian = _determine_guardian_from_reason(reason)

    # Build invalidation triggers
    invalidation_triggers = _build_invalidation_triggers(strategy_type, decision)

    # === CRITICAL FIX: Store underlying price for Iron Condor validation ===
    # For volatility strategies, we need the underlying price for breach calculations
    effective_entry_price = underlying_price if underlying_price else entry_price

    thesis_data = {
        'strategy_type': strategy_type,
        'guardian_agent': guardian,
        'primary_rationale': reason,
        'invalidation_triggers': invalidation_triggers,
        'supporting_data': {
            'entry_price': effective_entry_price,  # Underlying price for breach calc
            'spread_credit': entry_price,           # Original spread value
            'underlying_symbol': config.get('symbol', 'KC'),
            'contract_month': contract_month,
            'entry_regime': decision.get('regime', 'UNKNOWN'),
            'volatility_sentiment': decision.get('volatility_sentiment', 'NEUTRAL'),
            'confidence': decision.get('confidence', 0.5),
            'polymarket_slug': decision.get('polymarket_slug', ''),
            'polymarket_title': decision.get('polymarket_title', '')
        },
        'entry_timestamp': datetime.now(timezone.utc).isoformat(),
        'entry_regime': decision.get('regime', 'UNKNOWN')
    }

    tms.record_trade_thesis(position_id, thesis_data)
    logger.info(f"Recorded thesis for {position_id}: underlying_price=${effective_entry_price:.2f}, spread=${entry_price:.2f}")


def _determine_guardian_from_reason(reason: str) -> str:
    """Maps trade reasoning to the responsible specialist agent."""
    reason_lower = reason.lower()

    if any(kw in reason_lower for kw in ['frost', 'weather', 'drought', 'rain', 'temperature']):
        return 'Agronomist'
    elif any(kw in reason_lower for kw in ['port', 'shipping', 'logistics', 'strike', 'container']):
        return 'Logistics'
    elif any(kw in reason_lower for kw in ['volatility', 'iv', 'vix', 'range', 'premium']):
        return 'VolatilityAnalyst'
    elif any(kw in reason_lower for kw in ['macro', 'fed', 'dollar', 'brl', 'currency']):
        return 'Macro'
    elif any(kw in reason_lower for kw in ['sentiment', 'twitter', 'social']):
        return 'Sentiment'
    else:
        return 'Master'


def _build_invalidation_triggers(strategy_type: str, decision: dict) -> list:
    """Builds strategy-specific invalidation triggers."""
    triggers = []
    reason = decision.get('reason', '').lower()

    # Strategy-specific defaults
    if strategy_type == 'IRON_CONDOR':
        triggers.extend([
            'price_move_exceeds_2_percent',
            'iv_rank_above_70',
            'regime_shift_to_high_volatility'
        ])
    elif strategy_type == 'LONG_STRADDLE':
        triggers.extend([
            'catalyst_passed_without_move',
            'theta_burn_exceeds_hurdle',
            'volatility_crush'
        ])
    elif strategy_type in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
        # Narrative-specific triggers
        if 'frost' in reason:
            triggers.append('rain')
            triggers.append('warm front')
        if 'strike' in reason or 'port' in reason:
            triggers.append('strike resolution')
            triggers.append('port reopening')
        if 'supply' in reason:
            triggers.append('supply normalization')

    return triggers


async def cancel_all_open_orders(config: dict, connection_purpose: str = "orchestrator_orders"):
    """
    Fetches and cancels all open (non-filled) orders.
    """
    logger.info("--- Canceling any remaining open DAY orders ---")
    try:
        ib = await IBConnectionPool.get_connection(connection_purpose, config)
        configure_market_data_type(ib)

        logger.info(f"Connected to IB for canceling open orders (purpose: {connection_purpose}).")

        # Use ib.reqAllOpenOrdersAsync() to get all open orders from the broker
        open_trades = await ib.reqAllOpenOrdersAsync()
        if not open_trades:
            logger.info("No open orders found to cancel.")
            return

        logger.info(f"Found {len(open_trades)} open orders to cancel.")
        for trade in list(open_trades): # Iterate over a copy
            ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order ID {trade.order.orderId}.")
            await asyncio.sleep(0.2)

        logger.info(f"--- Finished canceling {len(open_trades)} orders ---")
        # Create a detailed notification message
        if open_trades:
            summary_items = [f"  - {trade.contract.localSymbol} ({trade.order.action} {trade.order.totalQuantity}) ID: {trade.order.orderId}" for trade in open_trades]
            message = f"<b>Canceled {len(open_trades)} unfilled DAY orders:</b>\n" + "\n".join(summary_items)
        else:
            message = "No open orders were found to cancel."

        send_pushover_notification(config.get('notifications', {}), "Open Orders Canceled", message)

    except Exception as e:
        msg = f"A critical error occurred while canceling orders: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Cancellation CRITICAL", msg)
    finally:
        # Do not disconnect pool
        pass
