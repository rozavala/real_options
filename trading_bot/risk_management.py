import asyncio
import logging
import traceback

from ib_insync import *
from ib_insync import util

from trading_bot.ib_interface import wait_for_fill
from trading_bot.utils import get_position_details
from notifications import send_pushover_notification


async def manage_existing_positions(ib: IB, config: dict, signal: dict, underlying_price: float, future_contract: Contract) -> bool:
    """Checks for existing positions for a given future and closes them if misaligned."""
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
    """
    Helper function to perform a single iteration of the risk check.
    This version groups positions by underlying to manage combo risk correctly.
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


async def monitor_positions_for_risk(ib: IB, config: dict):
    risk_params = config.get('risk_management', {})
    stop_loss_pct = risk_params.get('stop_loss_pct')
    take_profit_pct = risk_params.get('take_profit_pct')
    interval = risk_params.get('check_interval_seconds', 300)

    if not stop_loss_pct and not take_profit_pct:
        return
    logging.info(f"Starting intraday risk monitor. Stop: {stop_loss_pct:.0%}, Profit: {take_profit_pct:.0%}")
    closed_ids = set()
    while True:
        try:
            await _check_risk_once(ib, config, closed_ids, stop_loss_pct, take_profit_pct)
            await asyncio.sleep(interval)
        except Exception as e:
            logging.error(f"Error in risk monitor loop: {e}")