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
import traceback

from ib_insync import *
from ib_insync import util

from logging_config import setup_logging
from trading_bot.utils import get_position_details, log_trade_to_ledger
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
                    log_trade_to_ledger(trade, "Position Misaligned")
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

    for under_con_id, combo_legs in positions_by_underlying.items():
        total_unrealized_pnl, total_entry_cost = 0, 0
        for leg_pos in combo_legs:
            pnl = ib.reqPnLSingle(account, '', leg_pos.contract.conId)
            await ib.sleepAsync(0.1)
            if util.isNan(pnl.unrealizedPnL):
                logging.warning(f"Could not get PnL for leg {leg_pos.contract.localSymbol}. Skipping combo risk check."); total_unrealized_pnl = float('nan'); break
            total_unrealized_pnl += pnl.unrealizedPnL
            multiplier = float(leg_pos.contract.multiplier) if leg_pos.contract.multiplier else 37500.0
            total_entry_cost += abs(leg_pos.position * leg_pos.avgCost * multiplier)

        if util.isNan(total_unrealized_pnl) or total_entry_cost == 0: continue

        pnl_percentage = total_unrealized_pnl / total_entry_cost
        combo_symbols = " | ".join([p.contract.localSymbol for p in combo_legs])
        logging.info(f"Combo {combo_symbols}: Entry Cost=${total_entry_cost:.2f}, PnL=${total_unrealized_pnl:.2f} ({pnl_percentage:+.2%})")

        reason = None
        if stop_loss_pct and pnl_percentage <= -abs(stop_loss_pct): reason = "Stop-Loss"
        elif take_profit_pct and pnl_percentage >= abs(take_profit_pct): reason = "Take-Profit"

        if reason:
            logging.info(f"{reason.upper()} TRIGGERED for combo {combo_symbols} at {pnl_percentage:+.2%}")
            send_pushover_notification(config, reason, f"Combo {combo_symbols} closed at {pnl_percentage:+.2%}")
            for leg_pos_to_close in combo_legs:
                try:
                    order = MarketOrder('BUY' if leg_pos_to_close.position < 0 else 'SELL', abs(leg_pos_to_close.position))
                    trade = ib.placeOrder(leg_pos_to_close.contract, order)
                    while not trade.isDone():
                        await ib.sleepAsync(0.1)
                    if trade.orderStatus.status == OrderStatus.Filled:
                        log_trade_to_ledger(trade, f"Combo {reason}")
                        closed_ids.add(leg_pos_to_close.contract.conId)
                except Exception as e:
                    logging.error(f"Failed to close leg {leg_pos_to_close.contract.localSymbol}: {e}")
            for leg_pos in combo_legs: ib.cancelPnLSingle(account, '', leg_pos.contract.conId)


async def _check_fills_once(ib: IB, config: dict, filled_order_ids: set):
    """Checks for newly filled orders and logs them."""
    logging.info("--- Order Fill Monitor: Checking for new fills ---")
    newly_filled_count = 0
    for trade in ib.trades():
        if trade.orderStatus.status == OrderStatus.Filled and trade.order.orderId not in filled_order_ids:
            try:
                log_trade_to_ledger(trade, "Daily Strategy Fill")
                filled_order_ids.add(trade.order.orderId)
                newly_filled_count += 1
                fill_msg = f"FILLED: {trade.order.action} {trade.orderStatus.filled} {trade.contract.localSymbol} @ {trade.orderStatus.avgFillPrice:.2f}"
                logging.info(fill_msg)
                send_pushover_notification(config.get('notifications', {}), "Order Filled", fill_msg)
            except Exception as e:
                logging.error(f"Error processing fill for order ID {trade.order.orderId}: {e}")
    if newly_filled_count > 0:
        logging.info(f"Successfully processed and logged {newly_filled_count} new fills.")


async def monitor_positions_for_risk(ib: IB, config: dict):
    """Continuously monitors for risk triggers and order fills."""
    risk_params = config.get('risk_management', {})
    stop_loss_pct = risk_params.get('stop_loss_pct')
    take_profit_pct = risk_params.get('take_profit_pct')
    interval = risk_params.get('check_interval_seconds', 60)

    # Correctly format the strings for logging to avoid ValueError
    stop_str = f"{stop_loss_pct:.0%}" if stop_loss_pct else "N/A"
    profit_str = f"{take_profit_pct:.0%}" if take_profit_pct else "N/A"
    logging.info(f"Starting intraday monitor. Stop: {stop_str}, Profit: {profit_str}, Interval: {interval}s")

    closed_ids, filled_order_ids = set(), set()
    try:
        while True:
            try:
                logging.info("Monitor cycle starting...")
                if stop_loss_pct or take_profit_pct:
                    await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)
                await _check_fills_once(ib, config, filled_order_ids)
                logging.info(f"Monitor cycle complete. Waiting {interval} seconds.")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logging.info("Monitoring task was cancelled."); break
            except Exception as e:
                logging.error(f"Error in monitor loop: {e}"); await asyncio.sleep(60)
    finally:
        logging.info("Monitor loop is shutting down. Cancelling all PnL subscriptions.")
        if ib.isConnected():
            account = ib.managedAccounts()[0]
            for pnl in ib.pnlSubscriptions():
                ib.cancelPnLSingle(account, pnl.modelCode, pnl.conId)

