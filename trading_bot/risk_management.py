"""Manages position risk, including alignment checks and P&L monitoring.

This module contains the core logic for three key risk management functions:
1.  **Position Alignment**: Pre-trade check to close misaligned positions.
2.  **Intraday P&L Monitoring**: A long-running task to monitor open positions
    against stop-loss and take-profit thresholds.
3.  **Order Fill Monitoring**: A long-running task to watch for and log the
    fills of open 'DAY' orders.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd

from ib_insync import *
from ib_insync import util

from trading_bot.logging_config import setup_logging
from trading_bot.utils import get_position_details, log_trade_to_ledger, get_dollar_multiplier
from trading_bot.order_manager import get_trade_ledger_df
from trading_bot.ib_interface import place_order, get_active_futures
from notifications import send_pushover_notification
from trading_bot.tms import TransactiveMemory


async def check_iron_condor_theses(ib: IB, config: dict):
    """
    Checks all active Iron Condor theses for price breach invalidation.

    This runs every 15 minutes during market hours to catch intraday breaches.
    Sends immediate Pushover alert if 2% threshold exceeded.
    """
    try:
        tms = TransactiveMemory()

        # Get all active theses from VolatilityAnalyst (Iron Condors)
        active_theses = tms.get_active_theses_by_guardian('VolatilityAnalyst')

        if not active_theses:
            return

        # Get current underlying price once (efficiency)
        futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=1)
        if not futures:
            logging.warning("IC thesis check: No active futures found")
            return

        underlying = futures[0]
        ticker = ib.reqMktData(underlying, '', True, False)
        await asyncio.sleep(1)
        current_price = ticker.last if not util.isNan(ticker.last) else ticker.close
        ib.cancelMktData(underlying)

        if not current_price or current_price <= 0:
            logging.warning(f"IC thesis check: Invalid underlying price {current_price}")
            return

        # Check each Iron Condor thesis
        for thesis in active_theses:
            if thesis.get('strategy_type') != 'IRON_CONDOR':
                continue

            entry_price = thesis.get('supporting_data', {}).get('entry_price', 0)

            # SANITY CHECK: Detect legacy theses where entry_price is a spread premium
            price_floor = config.get('validation', {}).get('underlying_price_floor', 100.0)
            if 0 < entry_price < price_floor:
                logging.warning(
                    f"IC thesis check: entry_price ${entry_price:.2f} < floor "
                    f"${price_floor:.2f}. Skipping — likely spread premium."
                )
                continue  # Skip this thesis entirely

            if entry_price <= 0:
                logging.warning(f"IC thesis has invalid entry_price: {entry_price}")
                continue

            move_pct = abs((current_price - entry_price) / entry_price) * 100

            # Tiered alerting
            if move_pct > 2.0:
                logging.critical(f"🚨 IRON CONDOR BREACH: {move_pct:.2f}% move exceeds 2% threshold!")
                send_pushover_notification(
                    config.get('notifications', {}),
                    "🚨 IRON CONDOR BREACH - CLOSE IMMEDIATELY",
                    f"Underlying has moved {move_pct:.1f}% from entry.\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"Current: ${current_price:.2f}\n"
                    f"Threshold: 2%\n\n"
                    f"MANUAL INTERVENTION REQUIRED"
                )
            elif move_pct > 1.5:
                logging.warning(f"⚠️ Iron Condor approaching breach: {move_pct:.2f}% (threshold: 2%)")
                # Optional: Send warning notification at 1.5%

    except Exception as e:
        logging.error(f"IC thesis check failed: {e}", exc_info=True)


async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract) -> bool:
    """Checks for and closes any positions misaligned with the current signal."""
    target_future_conId = future_contract.conId
    logging.info(f"--- Managing Positions for Future Contract: {future_contract.localSymbol} (conId: {target_future_conId}) ---")

    all_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=15)
    positions_for_this_future = []

    for p in all_positions:
        if p.contract.symbol != config['symbol'] or p.position == 0: continue
        try:
            details_list = await asyncio.wait_for(
                ib.reqContractDetailsAsync(Contract(conId=p.contract.conId)), timeout=5
            )
            if details_list and hasattr(details_list[0], 'underConId') and details_list[0].underConId == target_future_conId:
                positions_for_this_future.append(p)
        except Exception as e:
            logging.warning(f"Could not resolve underlying for position {p.contract.localSymbol}: {e}")

    if not positions_for_this_future:
        logging.info(f"No existing positions found for future {future_contract.localSymbol}.")
        return True

    target_strategy = ''
    if signal['prediction_type'] == 'DIRECTIONAL':
        target_strategy = 'BULL_CALL_SPREAD' if signal['direction'] == 'BULLISH' else 'BEAR_PUT_SPREAD'
    elif signal['prediction_type'] == 'VOLATILITY':
        target_strategy = 'LONG_STRADDLE' if signal['level'] == 'HIGH' else 'IRON_CONDOR'

    trades_to_close, position_is_aligned = [], False
    for pos in positions_for_this_future:
        pos_details = await get_position_details(ib, pos)
        current_strategy = pos_details['type']
        logging.info(f"Found open position for {future_contract.localSymbol}: {pos.position} of {current_strategy}")

        if current_strategy == 'SINGLE_LEG' or current_strategy != target_strategy:
            logging.info(f"Position MISALIGNED. Current: {current_strategy}, Target: {target_strategy}. Closing.")
            trades_to_close.append(pos)
        else:
            logging.info("Position is ALIGNED. Holding.")
            position_is_aligned = True

    if trades_to_close:
        for pos_to_close in trades_to_close:
            try:
                contract_to_close = Contract(conId=pos_to_close.contract.conId)
                await asyncio.wait_for(ib.qualifyContractsAsync(contract_to_close), timeout=5)
                # Normalize cents-based strikes: if strike is in cents (from ledger),
                # convert to dollars for IBKR. Use profile unit instead of magic number.
                from trading_bot.utils import CENTS_INDICATORS
                try:
                    from config.commodity_profiles import get_active_profile
                    _prof = get_active_profile(config)
                    _cents = any(ind in _prof.contract.unit.lower() for ind in CENTS_INDICATORS)
                    if _cents and contract_to_close.strike > 100:
                        contract_to_close.strike /= 100.0
                except Exception:
                    if contract_to_close.strike > 100:
                        contract_to_close.strike /= 100.0
                order = MarketOrder('BUY' if pos_to_close.position < 0 else 'SELL', abs(pos_to_close.position))
                trade = ib.placeOrder(contract_to_close, order)
                # Wait for fill with 60s timeout to prevent infinite blocking
                close_deadline = asyncio.get_event_loop().time() + 60
                while not trade.isDone():
                    if asyncio.get_event_loop().time() > close_deadline:
                        logging.error(f"Position close timed out (60s) for conId {pos_to_close.contract.conId}. Order may still be working.")
                        break
                    await ib.sleepAsync(0.1)
                if trade.orderStatus.status == OrderStatus.Filled:
                    log_trade_to_ledger(ib, trade, "Position Misaligned", combo_id=trade.order.permId)
            except Exception as e:
                logging.exception(f"Failed to close position for conId {pos_to_close.contract.conId}: {e}")
        return True

    return not position_is_aligned


async def get_unrealized_pnl(ib: IB, contract: Contract) -> float:
    """
    Retrieves the unrealized P&L for a specific contract.

    This function first attempts to find the P&L from the portfolio data,
    which is generally faster and more reliable. If the contract is not found
    in the portfolio or if the P&L is NaN, it falls back to the polling
    mechanism using `ib.pnlSingle()`.

    Args:
        ib: The connected IB client instance.
        contract: The contract for which to retrieve the P&L.

    Returns:
        The unrealized P&L as a float, or float('nan') if it cannot be determined.
    """
    # 1. Prioritize portfolio data
    portfolio_item = next((p for p in ib.portfolio() if p.contract.conId == contract.conId), None)
    if portfolio_item and not util.isNan(portfolio_item.unrealizedPNL):
        return portfolio_item.unrealizedPNL

    # 2. Fallback to polling pnlSingle
    pnl = ib.pnlSingle(contract.conId)
    start_time = time.time()
    while not pnl or util.isNan(pnl.unrealizedPnL):
        if time.time() - start_time > 15:  # 15-second timeout
            logging.error(f"Timeout waiting for valid PnL for {contract.localSymbol} via pnlSingle.")
            return float('nan')
        await asyncio.sleep(0.5)
        pnl = ib.pnlSingle(contract.conId)

    return pnl.unrealizedPnL if pnl else float('nan')


async def _group_positions_by_underlying(ib: IB, all_positions: list, closed_ids: set) -> dict:
    """Groups positions by their underlying contract ID."""
    positions_by_underlying = {}
    for p in all_positions:
        if p.position == 0 or p.contract.conId in closed_ids:
            continue
        # Only process Options / FuturesOptions
        if not isinstance(p.contract, (FuturesOption, Option)):
            continue

        try:
            details = await asyncio.wait_for(ib.reqContractDetailsAsync(p.contract), timeout=5)
            if details and details[0].underConId:
                under_con_id = details[0].underConId
                if under_con_id not in positions_by_underlying:
                    positions_by_underlying[under_con_id] = []
                positions_by_underlying[under_con_id].append(p)
        except Exception as e:
            logging.warning(f"Could not get details for conId {p.contract.conId}: {e}")

    return positions_by_underlying


async def _calculate_combo_risk_metrics(ib: IB, config: dict, combo_legs: list) -> dict:
    """Calculates P&L and risk metrics for a combo position."""
    # A. Calculate Live P&L and Entry Cost
    total_unrealized_pnl = 0.0
    total_entry_cost = 0.0
    strikes = []

    for leg_pos in combo_legs:
        pnl = await get_unrealized_pnl(ib, leg_pos.contract)
        if util.isNan(pnl):
            return None  # Cannot calculate metrics without PnL

        total_unrealized_pnl += pnl
        total_entry_cost += leg_pos.position * leg_pos.avgCost
        strikes.append(leg_pos.contract.strike)

    if util.isNan(total_unrealized_pnl):
        return None

    # B. Determine Max Profit / Max Loss Potentials
    spread_width = max(strikes) - min(strikes) if strikes else 0
    max_profit_potential = 0.0
    max_loss_potential = 0.0

    dollar_mult = get_dollar_multiplier(config)
    FALLBACK_PROFIT_MULTIPLIER = 2.0

    if total_entry_cost > 0:
        # DEBIT SPREAD
        max_loss_potential = total_entry_cost
        max_profit_potential = (spread_width * abs(combo_legs[0].position) * dollar_mult) - total_entry_cost
        if max_profit_potential <= 0:
            max_profit_potential = total_entry_cost * FALLBACK_PROFIT_MULTIPLIER
    else:
        # CREDIT SPREAD
        max_profit_potential = abs(total_entry_cost)

        # Determine Max Loss based on Strategy Type
        relevant_width = spread_width

        if len(combo_legs) == 4:
            # Iron Condor Check
            sorted_legs = sorted(combo_legs, key=lambda leg: leg.contract.strike)
            s_pos = [leg.position for leg in sorted_legs]
            s_strike = [leg.contract.strike for leg in sorted_legs]

            if (s_pos[0] > 0 and s_pos[-1] > 0 and s_pos[1] < 0 and s_pos[2] < 0):
                wing1 = s_strike[1] - s_strike[0]
                wing2 = s_strike[3] - s_strike[2]
                relevant_width = max(wing1, wing2)
                logging.info(f"Identified Iron Condor. Using Wing Width: {relevant_width}")

        quantity_mult = max(abs(leg.position) for leg in combo_legs)
        max_loss_potential = (relevant_width * quantity_mult * dollar_mult) - max_profit_potential

        if max_loss_potential <= 0:
            max_loss_potential = float('inf')

    # C. Calculate "Capture" and "Risk" Metrics
    capture_pct = 0.0
    risk_pct = 0.0

    if max_profit_potential > 0:
        capture_pct = total_unrealized_pnl / max_profit_potential

    if max_loss_potential == float('inf'):
        risk_pct = 0.0
    elif max_loss_potential > 0:
        risk_pct = total_unrealized_pnl / max_loss_potential

    combo_desc = f"{len(combo_legs)} Legs on {combo_legs[0].contract.localSymbol}"
    logging.info(f"Combo {combo_desc}: PnL=${total_unrealized_pnl:.2f} | Capture: {capture_pct:.1%} | Risk Used: {risk_pct:.1%}")

    return {
        'pnl': total_unrealized_pnl,
        'max_profit': max_profit_potential,
        'max_loss': max_loss_potential,
        'capture_pct': capture_pct,
        'risk_pct': risk_pct,
        'combo_desc': combo_desc
    }


def _determine_risk_action(metrics: dict, config: dict, position_open_dates: dict, grace_period_seconds: int, combo_legs: list) -> str:
    """Determines if a risk action is needed (Stop Loss / Take Profit)."""
    max_risk_loss_pct = config.get('risk_management', {}).get('stop_loss_max_risk_pct', 0.50)
    target_capture_pct = config.get('risk_management', {}).get('take_profit_capture_pct', 0.80)

    risk_pct = metrics['risk_pct']
    capture_pct = metrics['capture_pct']
    max_profit_potential = metrics['max_profit']
    max_loss_potential = metrics['max_loss']
    combo_desc = metrics['combo_desc']

    reason = None

    # STOP LOSS CHECK: Have we lost more than X% of our Max Risk?
    if max_loss_potential > 0 and risk_pct <= -abs(max_risk_loss_pct):
        reason = f"Stop-Loss (Hit {abs(risk_pct):.1%} of Max Risk)"

    # TAKE PROFIT CHECK: Have we captured X% of potential profit?
    elif max_profit_potential > 0 and capture_pct >= abs(target_capture_pct):
        reason = f"Take-Profit (Captured {capture_pct:.1%} of Max Profit)"

    if not reason:
        return None

    # E. Grace Period Logic (Check trade ledger for age)
    utc = pytz.utc
    now_utc = datetime.now(utc)

    position_age_seconds = float('inf')
    # This is a fallback identifier. It's not perfect but works for now.
    temp_position_id = "-".join(sorted([p.contract.localSymbol for p in combo_legs]))
    open_date = position_open_dates.get(temp_position_id)
    if open_date:
        position_age_seconds = (now_utc - open_date).total_seconds()

    if position_age_seconds < grace_period_seconds:
        logging.info(f"'{reason}' trigger for {combo_desc} IGNORED. Position age ({position_age_seconds:.0f}s) is within the grace period ({grace_period_seconds}s).")
        return None

    return reason


async def _execute_risk_closure(ib: IB, config: dict, combo_legs: list, reason: str, pnl: float, closed_ids: set, account: str):
    """Executes the risk closure (order placement and notification)."""
    combo_desc = f"{len(combo_legs)} Legs on {combo_legs[0].contract.localSymbol}"
    logging.info(f"{reason.upper()} TRIGGERED for {combo_desc}")
    initial_message = (
        f"<b>{reason}</b>\n"
        f"PnL: ${pnl:.2f}\n"
        f"Closing position..."
    )
    send_pushover_notification(config.get('notifications', {}), "Risk Alert", initial_message)

    # Close as a single atomic BAG (Combo) order
    try:
        # 1. Define the BAG Contract
        contract = Contract()
        contract.symbol = config.get('symbol', 'KC')
        contract.secType = "BAG"
        contract.currency = "USD"
        contract.exchange = config.get('exchange', 'NYBOT')

        combo_legs_list = []
        for leg_pos in combo_legs:
            # Closing action: Inverse of current position
            # If Long (pos > 0), we SELL. If Short (pos < 0), we BUY.
            action = 'SELL' if leg_pos.position > 0 else 'BUY'

            # === M5 FIX: Use actual position size as ratio ===
            # For closing, ratio should match what we hold
            leg_qty = abs(int(leg_pos.position))

            combo_legs_list.append(ComboLeg(
                conId=leg_pos.contract.conId,
                ratio=leg_qty,  # v3.1: Use actual position size
                action=action,
                exchange=config.get('exchange', 'NYBOT')
            ))

        contract.comboLegs = combo_legs_list

        # 2. Place the Order
        # We "BUY" the BAG because we defined the legs with their specific closing actions (BUY/SELL).
        # v3.1: Order quantity is 1 when ratios carry the size
        qty = 1

        order = MarketOrder("BUY", qty)
        order.outsideRth = True

        # === v3.1: Calculate GCD for display ===
        from math import gcd
        from functools import reduce
        ratios = [abs(int(p.position)) for p in combo_legs]
        common_divisor = reduce(gcd, ratios) if ratios else 1
        readable_qty = ratios[0] // common_divisor if common_divisor and ratios else 0

        logging.info(
            f"Placing BAG Market Order to close {combo_desc} "
            f"(Ratios: {[r//common_divisor for r in ratios]}, Qty: {readable_qty})"
        )
        place_order(ib, contract, order)

        # Mark as closed AFTER order placement succeeds — if place_order
        # throws, legs stay visible to future risk checks instead of becoming zombies
        for leg_pos in combo_legs:
            closed_ids.add(leg_pos.contract.conId)

    except Exception as e:
        logging.error(f"Failed to close combo {combo_desc} as BAG: {e}")

    # Clean up PnL subs
    for leg_pos in combo_legs:
        ib.cancelPnLSingle(account, '', leg_pos.contract.conId)


async def _check_risk_once(ib: IB, config: dict, closed_ids: set, stop_loss_pct: float, take_profit_pct: float):
    """
    Performs P&L risk check using 'Percent of Capture' (Profit) and 'Percent of Max Risk' (Loss).
    """
    if not ib.isConnected():
        return

    # Load new config parameters
    risk_config = config.get('risk_management', {})
    grace_period_seconds = risk_config.get('monitoring_grace_period_seconds', 0)

    # --- Grace Period Implementation ---
    trade_ledger = get_trade_ledger_df()
    utc = pytz.utc

    # Handle empty ledger
    if trade_ledger.empty:
        position_open_dates = {}
    else:
        open_positions_from_ledger = trade_ledger.groupby('position_id').filter(lambda x: x['quantity'].sum() != 0)
        position_open_dates = open_positions_from_ledger.groupby('position_id')['timestamp'].min().dt.tz_localize(utc).to_dict()

    # --- 1. Group Positions by Underlying ---
    all_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=15)
    if not all_positions:
        return

    positions_by_underlying = await _group_positions_by_underlying(ib, all_positions, closed_ids)

    logging.info(f"--- Risk Monitor: Checking {len(positions_by_underlying)} combo position(s) ---")

    # --- 2. Start PnL Subs ---
    account = ib.managedAccounts()[0]
    active_pnl_con_ids = {p.conId for p in ib.pnlSingle()}
    new_subs = False
    for _, legs in positions_by_underlying.items():
        for leg in legs:
            if leg.contract.conId not in active_pnl_con_ids:
                ib.reqPnLSingle(account, '', leg.contract.conId)
                new_subs = True
    if new_subs:
        await asyncio.sleep(2)

    # --- 3. Analyze Each Combo ---
    for under_con_id, combo_legs in positions_by_underlying.items():
        metrics = await _calculate_combo_risk_metrics(ib, config, combo_legs)
        if not metrics:
            continue

        reason = _determine_risk_action(metrics, config, position_open_dates, grace_period_seconds, combo_legs)

        if reason:
            await _execute_risk_closure(ib, config, combo_legs, reason, metrics['pnl'], closed_ids, account)


# --- Order Fill Monitoring ---

# A set to keep track of order IDs that have already been logged as filled.
# This prevents duplicate entries in the trade ledger.
_filled_order_ids = set()


async def _on_order_status(ib: IB, trade: Trade):
    """Event handler for order status updates."""
    if trade.orderStatus.status == OrderStatus.Filled and trade.order.orderId not in _filled_order_ids:
        try:
            # Skip logging for the parent Bag contract, only log the legs.
            if isinstance(trade.contract, Bag):
                logging.info(f"Skipping summary fill event for Bag order {trade.order.orderId}.")
                return

            # Log the trade immediately upon fill confirmation.
            reason = "Risk Management Closure" if trade.order.outsideRth else "Daily Strategy Fill"
            await log_trade_to_ledger(ib, trade, reason, combo_id=trade.order.permId)
            _filled_order_ids.add(trade.order.orderId)
            fill_msg = (
                f"FILLED: {trade.order.action} {trade.orderStatus.filled} "
                f"{trade.contract.localSymbol} @ {trade.orderStatus.avgFillPrice:.2f}"
            )
            logging.info(fill_msg)
            # Note: Notifications are sent from the `_on_fill` handler to include execution details.
        except Exception as e:
            logging.error(f"Error processing fill for order ID {trade.order.orderId}: {e}")
    elif trade.orderStatus.status in [OrderStatus.Cancelled, OrderStatus.Inactive]:
        # If an order is cancelled or becomes inactive, we can remove it from tracking.
        _filled_order_ids.discard(trade.order.orderId)


def _on_fill(trade: Trade, fill: Fill, config: dict):
    """Event handler for new fills (executions)."""
    # This handler is primarily for sending notifications with fill details.
    # The actual logging to the trade ledger is done in `_on_order_status`
    # to ensure it happens only once per order.

    # --- Check if this fill is from a risk management closure ---
    # We use the order's `outsideRth` flag as a proxy. When we place risk
    # management orders, we can set this to True to identify them here and
    # suppress the individual notification, as a consolidated one will be sent.
    if trade.order.outsideRth:
        logging.info(f"Skipping individual fill notification for risk management closure of order ID {fill.execution.orderId}.")
        return

    try:
        # Create a detailed notification message
        message = (
            f"<b>✅ Order Executed</b>\n"
            f"<b>{fill.execution.side} {fill.execution.shares}</b> {trade.contract.localSymbol}\n"
            f"Avg Price: ${fill.execution.price:.2f}\n"
            f"Order ID: {fill.execution.orderId}"
        )
        logging.info(f"Sending execution notification for Order ID {fill.execution.orderId}")
        send_pushover_notification(config.get('notifications', {}), "Order Execution", message)
    except Exception as e:
        logging.error(f"Error processing execution notification for order ID {trade.order.orderId}: {e}")


async def monitor_positions_for_risk(ib: IB, config: dict):
    """
    Continuously monitors for risk triggers (P&L) and order fills, handling
    real-time events.
    """
    risk_params = config.get('risk_management', {})
    # New parameters
    target_capture_pct = risk_params.get('take_profit_capture_pct', 0.80)
    max_risk_loss_pct = risk_params.get('stop_loss_max_risk_pct', 0.50)
    interval = risk_params.get('check_interval_seconds', 60)

    stop_str = f"{max_risk_loss_pct:.0%} (Max Risk)" if max_risk_loss_pct else "N/A"
    profit_str = f"{target_capture_pct:.0%} (Capture)" if target_capture_pct else "N/A"
    logging.info(f"Starting intraday P&L monitor. Stop: {stop_str}, Profit: {profit_str}, Interval: {interval}s")

    # --- Event Handler Setup ---
    # Create a lambda to pass the config to the fill handler.
    # This ensures the handler has access to notification settings.
    # We define it here so we can also remove it accurately in the finally block.
    def order_status_handler(trade: Trade):
        asyncio.create_task(_on_order_status(ib, trade))

    def fill_handler(t, f):
        _on_fill(t, f, config)

    # Clear filled order tracking from previous session to prevent unbounded growth
    # and avoid stale ID collisions after IB Gateway restarts
    _filled_order_ids.clear()

    # Register the event handlers before starting the tasks.
    ib.orderStatusEvent += order_status_handler
    ib.execDetailsEvent += fill_handler
    logging.info("Order status and execution event handlers registered.")

    # Request all open orders to ensure we are tracking everything upon startup.
    await asyncio.wait_for(ib.reqAllOpenOrdersAsync(), timeout=15)

    closed_ids = set()
    ic_check_counter = 0
    IC_CHECK_INTERVAL = 15  # Check IC theses every 15 minutes (assuming 60s interval)

    try:
        while True:
            try:
                logging.debug("P&L monitor cycle starting...")
                # Always run check if we have valid thresholds, but also the loop logic depends on them being present?
                if target_capture_pct or max_risk_loss_pct:
                    # Arguments are ignored inside _check_risk_once but required by signature
                    await _check_risk_once(ib, config, closed_ids, 0.0, 0.0)

                # === NEW: Iron Condor Thesis Validation ===
                ic_check_counter += 1
                if ic_check_counter >= IC_CHECK_INTERVAL:
                    logging.info("Running Iron Condor thesis validation check...")
                    await check_iron_condor_theses(ib, config)
                    ic_check_counter = 0

                logging.debug(f"P&L monitor cycle complete. Waiting {interval} seconds.")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logging.info("P&L monitoring task was cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in P&L monitor loop: {e}")
                await asyncio.sleep(60)
    finally:
        logging.info("Shutting down monitor. Unregistering event handlers and cancelling PnL subscriptions.")
        # Unregister event handlers to prevent memory leaks
        if ib.isConnected():
            if order_status_handler in ib.orderStatusEvent:
                ib.orderStatusEvent -= order_status_handler
            if fill_handler in ib.execDetailsEvent:
                ib.execDetailsEvent -= fill_handler

            account = ib.managedAccounts()[0]
            for pnl in ib.pnlSubscriptions():
                ib.cancelPnLSingle(account, pnl.modelCode, pnl.conId)

