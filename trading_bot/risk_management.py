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
import traceback

from ib_insync import *
from ib_insync import util

from logging_config import setup_logging
from trading_bot.utils import get_position_details, log_trade_to_ledger
from trading_bot.ib_interface import place_order
from notifications import send_pushover_notification

# --- Logging Setup ---
setup_logging()


async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract) -> bool:
    """Checks for and closes any positions misaligned with the current signal."""
    target_future_conId = future_contract.conId
    logging.info(f"--- Managing Positions for Future Contract: {future_contract.localSymbol} (conId: {target_future_conId}) ---")

    all_positions = await ib.reqPositionsAsync()
    positions_for_this_future = []

    for p in all_positions:
        if p.contract.symbol != config['symbol'] or p.position == 0: continue
        try:
            details_list = await ib.reqContractDetailsAsync(Contract(conId=p.contract.conId))
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
                await ib.qualifyContractsAsync(contract_to_close)
                if contract_to_close.strike > 100: contract_to_close.strike /= 100.0
                order = MarketOrder('BUY' if pos_to_close.position < 0 else 'SELL', abs(pos_to_close.position))
                trade = ib.placeOrder(contract_to_close, order)
                while not trade.isDone():
                    await ib.sleepAsync(0.1)
                if trade.orderStatus.status == OrderStatus.Filled:
                    log_trade_to_ledger(ib, trade, "Position Misaligned", combo_id=trade.order.permId)
            except Exception as e:
                logging.error(f"Failed to close position for conId {pos_to_close.contract.conId}: {e}\n{traceback.format_exc()}")
        return True

    return not position_is_aligned


async def _check_risk_once(ib: IB, config: dict, closed_ids: set, stop_loss_pct: float, take_profit_pct: float):
    """Performs a single iteration of the P&L risk check for all open positions."""
    if not ib.isConnected(): return

    all_positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.conId not in closed_ids]
    if not all_positions: return

    positions_by_underlying = {}
    for p in all_positions:
        if not isinstance(p.contract, (FuturesOption, Option)): continue
        try:
            details = await ib.reqContractDetailsAsync(p.contract)
            if details and details[0].underConId:
                under_con_id = details[0].underConId
                if under_con_id not in positions_by_underlying: positions_by_underlying[under_con_id] = []
                positions_by_underlying[under_con_id].append(p)
        except Exception as e:
            logging.warning(f"Could not get details for conId {p.contract.conId}: {e}")

    account = ib.managedAccounts()[0]
    logging.info(f"--- Risk Monitor: Checking {len(positions_by_underlying)} combo position(s) ---")

    # Start PnL subscriptions for any positions that are not yet being tracked
    active_pnl_con_ids = {p.conId for p in ib.pnlSingle()}
    new_subscriptions_made = False
    for under_con_id, combo_legs in positions_by_underlying.items():
        for leg_pos in combo_legs:
            if leg_pos.contract.conId not in active_pnl_con_ids:
                logging.info(f"Starting new PnL subscription for {leg_pos.contract.localSymbol}")
                ib.reqPnLSingle(account, '', leg_pos.contract.conId)
                new_subscriptions_made = True

    # If new subscriptions were started, wait for data to arrive.
    if new_subscriptions_made:
        await asyncio.sleep(2)

    # Create a map for efficient lookup of PnL data
    pnl_map = {p.conId: p for p in ib.pnlSingle()}

    for under_con_id, combo_legs in positions_by_underlying.items():
        total_unrealized_pnl, total_entry_cost = 0, 0
        for leg_pos in combo_legs:
            pnl = pnl_map.get(leg_pos.contract.conId)

            if not pnl or util.isNan(pnl.unrealizedPnL):
                logging.warning(f"Could not get PnL for leg {leg_pos.contract.localSymbol}. Skipping combo risk check."); total_unrealized_pnl = float('nan'); break

            total_unrealized_pnl += pnl.unrealizedPnL
            # multiplier = float(leg_pos.contract.multiplier) if leg_pos.contract.multiplier else 37500.0
            # Multiplier is already included in the cost so no needed again
            total_entry_cost += leg_pos.position * leg_pos.avgCost

        if util.isNan(total_unrealized_pnl) or total_entry_cost == 0: continue

        pnl_percentage = total_unrealized_pnl / abs(total_entry_cost)
        combo_symbols = " | ".join([p.contract.localSymbol for p in combo_legs])
        logging.info(f"Combo {combo_symbols}: Entry Cost=${total_entry_cost:.2f}, PnL=${total_unrealized_pnl:.2f} ({pnl_percentage:+.2%})")

        reason = None
        if stop_loss_pct and pnl_percentage <= -abs(stop_loss_pct): reason = "Stop-Loss"
        elif take_profit_pct and pnl_percentage >= abs(take_profit_pct): reason = "Take-Profit"

        if reason:
            logging.info(f"{reason.upper()} TRIGGERED for combo {combo_symbols} at {pnl_percentage:+.2%}")

            initial_message = (
                f"<b>{reason} Triggered for {combo_symbols}</b>\n"
                f"P&L: {pnl_percentage:+.2%} (${total_unrealized_pnl:+.2f}) vs "
                f"Entry Cost: ${total_entry_cost:.2f}\n"
                f"Closing position..."
            )
            send_pushover_notification(config.get('notifications', {}), f"Risk Alert: {reason}", initial_message)

            closed_trades_details = []
            for leg_pos_to_close in combo_legs:
                try:
                    leg_pos_to_close.contract.exchange = config.get('exchange', 'NYBOT')
                    order = MarketOrder('BUY' if leg_pos_to_close.position < 0 else 'SELL', abs(leg_pos_to_close.position))

                    # Set a flag to identify this as a risk management order
                    order.outsideRth = True

                    trade = place_order(ib, leg_pos_to_close.contract, order)

                    # --- Wait for the fill and capture details ---
                    start_time = time.time()
                    while not trade.isDone():
                        await asyncio.sleep(0.2)
                        if time.time() - start_time > 30: # 30-second timeout
                            logging.error(f"Timeout waiting for fill on risk closure for {leg_pos_to_close.contract.localSymbol}")
                            break

                    if trade.orderStatus.status == OrderStatus.Filled and trade.fills:
                        fill = trade.fills[0]
                        closed_trades_details.append(
                            f"  - Closed <b>{leg_pos_to_close.contract.localSymbol}</b> "
                            f"@{fill.execution.avgPrice:.2f}, "
                            f"P&L: ${fill.commissionReport.realizedPNL:,.2f}"
                        )
                        # Log the closure to the ledger
                        log_trade_to_ledger(ib, trade, reason, combo_id=trade.order.permId)
                    else:
                        closed_trades_details.append(f"  - Failed to close {leg_pos_to_close.contract.localSymbol}. Status: {trade.orderStatus.status}")

                    closed_ids.add(leg_pos_to_close.contract.conId)

                except Exception as e:
                    logging.error(f"Failed to close leg {leg_pos_to_close.contract.localSymbol}: {e}")
                    closed_trades_details.append(f"  - Error closing {leg_pos_to_close.contract.localSymbol}")

            # --- Send the consolidated final notification ---
            if closed_trades_details:
                final_message = (
                    f"<b>Closure Report for {combo_symbols}:</b>\n" +
                    "\n".join(closed_trades_details)
                )
                send_pushover_notification(config.get('notifications', {}), f"Position Closed: {reason}", final_message)

            for leg_pos in combo_legs: ib.cancelPnLSingle(account, '', leg_pos.contract.conId)


# --- Order Fill Monitoring ---

# A set to keep track of order IDs that have already been logged as filled.
# This prevents duplicate entries in the trade ledger.
_filled_order_ids = set()


def _on_order_status(ib: IB, trade: Trade):
    """Event handler for order status updates."""
    if trade.orderStatus.status == OrderStatus.Filled and trade.order.orderId not in _filled_order_ids:
        try:
            # Log the trade immediately upon fill confirmation.
            log_trade_to_ledger(ib, trade, "Daily Strategy Fill", combo_id=trade.order.permId)
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
    stop_loss_pct = risk_params.get('stop_loss_pct')
    take_profit_pct = risk_params.get('take_profit_pct')
    interval = risk_params.get('check_interval_seconds', 60)

    stop_str = f"{stop_loss_pct:.0%}" if stop_loss_pct else "N/A"
    profit_str = f"{take_profit_pct:.0%}" if take_profit_pct else "N/A"
    logging.info(f"Starting intraday P&L monitor. Stop: {stop_str}, Profit: {profit_str}, Interval: {interval}s")

    # --- Event Handler Setup ---
    # Create a lambda to pass the config to the fill handler.
    # This ensures the handler has access to notification settings.
    # We define it here so we can also remove it accurately in the finally block.
    order_status_handler = lambda t: _on_order_status(ib, t)
    fill_handler = lambda t, f: _on_fill(t, f, config)

    # Register the event handlers before starting the tasks.
    ib.orderStatusEvent += order_status_handler
    ib.execDetailsEvent += fill_handler
    logging.info("Order status and execution event handlers registered.")

    # Request all open orders to ensure we are tracking everything upon startup.
    await ib.reqAllOpenOrdersAsync()

    closed_ids = set()
    try:
        while True:
            try:
                logging.info("P&L monitor cycle starting...")
                if stop_loss_pct or take_profit_pct:
                    await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)

                logging.info(f"P&L monitor cycle complete. Waiting {interval} seconds.")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logging.info("P&L monitoring task was cancelled."); break
            except Exception as e:
                logging.error(f"Error in P&L monitor loop: {e}"); await asyncio.sleep(60)
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

