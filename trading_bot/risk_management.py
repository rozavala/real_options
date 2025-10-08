"""Manages position risk, including alignment checks and P&L monitoring.

This module contains the core logic for two key risk management functions:
1.  **Position Alignment**: Before entering a new trade, it checks existing
    positions for the same underlying future. If a position is found that
    does not align with the new trading signal (e.g., a bullish position
    exists when the new signal is bearish), it closes the misaligned position.
2.  **Intraday P&L Monitoring**: It provides a long-running task that can
    monitor the unrealized P&L of open combo positions and close them if they
    hit pre-defined stop-loss or take-profit percentage thresholds.
"""

import asyncio
import logging
import traceback

from ib_insync import *
from ib_insync import util

from logging_config import setup_logging
from trading_bot.ib_interface import wait_for_fill
from trading_bot.utils import get_position_details
from notifications import send_pushover_notification

# --- Logging Setup ---
setup_logging()


async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract) -> bool:
    """Checks for and closes any positions misaligned with the current signal.

    This function is a pre-trade check. It scans all open positions to find
    any options associated with the target `future_contract`. It then compares
    the strategy type of the existing position (e.g., 'BULL_CALL_SPREAD')
    with the strategy implied by the new `signal`. If they do not match, the
    existing position is closed. It also closes any orphaned single-leg positions.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        signal (dict): The new trading signal to be executed.
        underlying_price (float): The current price of the underlying future.
        future_contract (Contract): The future contract for which a trade is
            being considered.

    Returns:
        `True` if the bot should proceed with opening a new trade (i.e., no
        aligned position already exists). `False` if an aligned position is
        already in place and no new trade should be opened.
    """
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

        # A single leg is always considered misaligned as we only trade combos.
        if current_strategy == 'SINGLE_LEG':
            logging.info(f"Position is an orphaned SINGLE_LEG. Closing.")
            trades_to_close.append(pos)
        elif current_strategy != target_strategy:
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
                await wait_for_fill(ib, trade, config, reason="Position Misaligned")
            except Exception as e:
                logging.error(f"Failed to close position for conId {pos_to_close.contract.conId}: {e}\n{traceback.format_exc()}")
        return True

    if position_is_aligned:
        logging.info(f"An aligned position for {future_contract.localSymbol} already exists.")
        return False
    return True


async def _check_risk_once(ib: IB, config: dict, closed_ids: set, stop_loss_pct: float, take_profit_pct: float):
    """Performs a single iteration of the risk check for all open positions.

    This helper function groups all open option positions by their underlying
    future contract. It then calculates the aggregated P&L for each combo
    position and compares it against the stop-loss and take-profit
    percentage thresholds. If a threshold is breached, it closes all legs
    of the combo position.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
        closed_ids (set): A set of conIds for positions that have already been
            closed in this session, to avoid attempting to close them again.
        stop_loss_pct (float): The P&L percentage at which to trigger a stop-loss.
        take_profit_pct (float): The P&L percentage at which to trigger a take-profit.
    """
    if not ib.isConnected():
        return

    all_positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.conId not in closed_ids]
    if not all_positions:
        return

    # Group positions by their underlying contract ID
    positions_by_underlying = {}
    for p in all_positions:
        if not isinstance(p.contract, (FuturesOption, Option)):
            continue
        try:
            details = await ib.reqContractDetailsAsync(p.contract)
            if details and details[0].underConId:
                under_con_id = details[0].underConId
                if under_con_id not in positions_by_underlying:
                    positions_by_underlying[under_con_id] = []
                positions_by_underlying[under_con_id].append(p)
        except Exception as e:
            logging.warning(f"Could not get details for conId {p.contract.conId}: {e}")

    account = ib.managedAccounts()[0]
    logging.info(f"--- Risk Monitor: Checking {len(positions_by_underlying)} combo position(s) ---")

    for under_con_id, combo_legs in positions_by_underlying.items():
        total_unrealized_pnl = 0
        total_entry_cost = 0

        for leg_pos in combo_legs:
            pnl = ib.reqPnLSingle(account, '', leg_pos.contract.conId)
            await ib.sleepAsync(0.1) # Brief pause to allow PnL data to arrive
            if util.isNan(pnl.unrealizedPnL):
                logging.warning(f"Could not get PnL for leg {leg_pos.contract.localSymbol}. Skipping combo risk check.")
                total_unrealized_pnl = float('nan')
                break

            total_unrealized_pnl += pnl.unrealizedPnL

            try:
                multiplier = float(leg_pos.contract.multiplier) if leg_pos.contract.multiplier else 37500.0
            except (ValueError, TypeError):
                multiplier = 37500.0

            total_entry_cost += abs(leg_pos.position * leg_pos.avgCost * multiplier)

        if util.isNan(total_unrealized_pnl) or total_entry_cost == 0:
            continue

        pnl_percentage = total_unrealized_pnl / total_entry_cost
        combo_symbols = " | ".join([p.contract.localSymbol for p in combo_legs])
        logging.info(f"Combo {combo_symbols}: Entry Cost=${total_entry_cost:.2f}, "
                     f"Unrealized PnL=${total_unrealized_pnl:.2f} ({pnl_percentage:+.2%})")

        reason = None
        if stop_loss_pct and pnl_percentage <= -abs(stop_loss_pct):
            reason = "Stop-Loss"
        elif take_profit_pct and pnl_percentage >= abs(take_profit_pct):
            reason = "Take-Profit"

        if reason:
            logging.info(f"{reason.upper()} TRIGGERED for combo {combo_symbols} at {pnl_percentage:+.2%}")
            send_pushover_notification(config, reason, f"Combo {combo_symbols} closed at {pnl_percentage:+.2%}")

            for leg_pos_to_close in combo_legs:
                try:
                    order = MarketOrder('BUY' if leg_pos_to_close.position < 0 else 'SELL', abs(leg_pos_to_close.position))
                    trade = ib.placeOrder(leg_pos_to_close.contract, order)
                    await wait_for_fill(ib, trade, config, reason=f"Combo {reason}")
                    closed_ids.add(leg_pos_to_close.contract.conId)
                except Exception as e:
                    logging.error(f"Failed to close leg {leg_pos_to_close.contract.localSymbol} for combo {under_con_id}: {e}")
            # After closing a combo, cancel the PnL subscriptions for its legs
            for leg_pos in combo_legs:
                ib.cancelPnLSingle(account, '', leg_pos.contract.conId)


async def monitor_positions_for_risk(ib: IB, config: dict):
    """Continuously monitors open positions for stop-loss/take-profit triggers.

    This function runs an infinite loop that periodically calls `_check_risk_once`
    to evaluate the P&L of all open positions. The check interval and risk
    thresholds (stop-loss/take-profit percentages) are read from the config.

    Note:
        This function is designed to be run as a long-running, standalone
        background task (see `position_monitor.py`).

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        config (dict): The application configuration dictionary.
    """
    risk_params = config.get('risk_management', {})
    stop_loss_pct = risk_params.get('stop_loss_pct')
    take_profit_pct = risk_params.get('take_profit_pct')
    interval = risk_params.get('check_interval_seconds', 300)

    if not stop_loss_pct and not take_profit_pct:
        logging.info("Risk management disabled: No stop-loss or take-profit percentages configured.")
        return

    logging.info(f"Starting intraday risk monitor. Stop: {stop_loss_pct:.0%}, Profit: {take_profit_pct:.0%}, Check Interval: {interval}s")
    closed_ids = set()
    try:
        while True:
            try:
                logging.info("Risk monitor performing check...")
                await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)
                logging.info(f"Risk monitor check complete. Waiting {interval} seconds for next check.")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logging.info("Risk monitoring task was cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in risk monitor loop: {e}")
                await asyncio.sleep(60) # Wait longer after an error
    finally:
        logging.info("Risk monitor loop is shutting down. Cancelling all PnL subscriptions.")
        # Ensure all PnL subscriptions are cancelled on exit
        account = ib.managedAccounts()[0]
        for pnl in ib.pnlSubscriptions():
            ib.cancelPnLSingle(account, pnl.modelCode, pnl.conId)