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

        if current_strategy != target_strategy:
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


async def _check_risk_once(ib: IB, config: dict, closed_ids: set, stop_loss: float, take_profit: float):
    """Helper function to perform a single iteration of the risk check."""
    if not ib.isConnected():
        return
    positions = [p for p in await ib.reqPositionsAsync() if p.position != 0 and p.contract.conId not in closed_ids]
    if not positions:
        return

    account = ib.managedAccounts()[0]
    logging.info("--- Risk Monitor: Checking open positions ---")
    for p in positions:
        pnl = ib.reqPnLSingle(account, '', p.contract.conId)

        if util.isNan(pnl.unrealizedPnL):
            logging.warning(f"Could not get PnL for {p.contract.localSymbol}. Skipping risk check.")
            continue

        pnl_per_contract = pnl.unrealizedPnL / abs(p.position)
        logging.info(f"Position {p.contract.localSymbol}: Unrealized PnL/Contract = ${pnl_per_contract:.2f}")
        reason = "Stop-Loss" if stop_loss and pnl_per_contract < -abs(stop_loss) else "Take-Profit" if take_profit and pnl_per_contract > abs(take_profit) else None
        if reason:
            logging.info(f"{reason.upper()} TRIGGERED for {p.contract.localSymbol}. PnL/contract: ${pnl_per_contract:.2f}")
            send_pushover_notification(config, reason, f"{p.contract.localSymbol} closed. PnL/contract: ${pnl_per_contract:.2f}")

            # Create a new, clean contract object just for closing the trade
            contract_to_close = Contract(conId=p.contract.conId)
            await ib.qualifyContractsAsync(contract_to_close)

            # As a final safeguard, manually correct the strike price if it's still incorrect after qualifying
            if hasattr(contract_to_close, 'strike') and contract_to_close.strike > 100:
                contract_to_close.strike /= 100.0

            order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
            trade = ib.placeOrder(contract_to_close, order)
            await wait_for_fill(ib, trade, config, reason=reason)
            closed_ids.add(p.contract.conId)


async def monitor_positions_for_risk(ib: IB, config: dict):
    risk_params = config.get('risk_management', {})
    stop_loss, take_profit, interval = risk_params.get('stop_loss_per_contract_usd'), risk_params.get('take_profit_per_contract_usd'), risk_params.get('check_interval_seconds', 300)
    if not stop_loss and not take_profit:
        return
    logging.info(f"Starting intraday risk monitor. Stop: ${stop_loss}, Profit: ${take_profit}")
    closed_ids = set()
    while True:
        try:
            await _check_risk_once(ib, config, closed_ids, stop_loss, take_profit)
            await asyncio.sleep(interval)
        except Exception as e:
            logging.error(f"Error in risk monitor loop: {e}")