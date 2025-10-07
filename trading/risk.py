import asyncio
import logging
from ib_insync import *

from notifications import send_pushover_notification
from trading.execution import wait_for_fill

# --- Risk Management ---

def is_trade_sane(signal: dict, config: dict) -> bool:
    """
    Performs pre-trade sanity checks, such as ensuring the signal's
    confidence is above a minimum threshold.
    """
    min_confidence = config.get('risk_management', {}).get('min_confidence_threshold', 0.0)

    # Get confidence, defaulting to 0.0 if it's missing or explicitly None
    confidence = signal.get('confidence') or 0.0

    if confidence < min_confidence:
        original_confidence = signal.get('confidence', 'Not Provided')
        logging.warning(f"Sanity Check FAILED: Signal confidence '{original_confidence}' is below threshold {min_confidence}.")
        return False

    logging.info("Sanity Check PASSED: Signal confidence is within acceptable limits.")
    return True


async def monitor_positions_for_risk(ib: IB, config: dict):
    """
    A background task that continuously monitors open positions for stop-loss
    or take-profit triggers based on a percentage of the entry cost.
    """
    risk_params = config.get('risk_management', {})
    stop_loss_pct = risk_params.get('stop_loss_percentage')
    take_profit_pct = risk_params.get('take_profit_percentage')
    check_interval = risk_params.get('check_interval_seconds', 300)

    if not stop_loss_pct and not take_profit_pct:
        logging.info("Risk monitor disabled: No stop-loss or take-profit percentage configured.")
        return

    logging.info(f"Starting intraday risk monitor. Stop: {stop_loss_pct}%, Profit: {take_profit_pct}%. Interval: {check_interval}s.")

    # This set will keep track of positions we've already sent a closing order for
    closed_order_ids = set()

    while True:
        try:
            await asyncio.sleep(check_interval)
            if not ib.isConnected():
                logging.warning("Risk monitor paused: IBKR not connected.")
                continue

            # We need both portfolio positions and account PnL
            positions = await ib.reqPositionsAsync()
            account = ib.managedAccounts()[0]

            # Filter for non-zero positions we haven't already acted on
            active_positions = [p for p in positions if p.position != 0 and p.contract.conId not in closed_order_ids]
            if not active_positions:
                continue

            logging.info("--- Risk Monitor: Checking open positions ---")

            for p in active_positions:
                # To get PnL for a single position, we must subscribe and wait for the update.
                pnl_subscription = ib.reqPnLSingle(account, '', p.contract.conId)
                # Wait a moment for the PnL data to arrive. This is a simple polling approach.
                await asyncio.sleep(1)

                pnl_data = pnl_subscription.pnl

                # It's crucial to cancel the subscription to avoid receiving continuous updates.
                ib.cancelPnLSingle(pnl_subscription)

                if util.isNan(pnl_data.unrealizedPnL):
                    logging.warning(f"Could not get PnL for {p.contract.localSymbol}. Skipping risk check for this position.")
                    continue

                # The cost basis is the total cost of the position
                cost_basis_per_contract = p.avgCost
                pnl_per_contract = pnl_data.unrealizedPnL / abs(p.position)

                # Calculate the percentage change
                if cost_basis_per_contract == 0:
                    logging.warning(f"Cost basis for {p.contract.localSymbol} is zero. Cannot calculate percentage PnL.")
                    continue

                pnl_percentage = (pnl_per_contract / cost_basis_per_contract) * 100
                logging.info(f"Position {p.contract.localSymbol}: Cost/Contract=${cost_basis_per_contract:.2f}, PnL/Contract=${pnl_per_contract:.2f} ({pnl_percentage:.2f}%)")

                # Check for trigger conditions
                reason = None
                if stop_loss_pct and pnl_percentage <= -abs(stop_loss_pct):
                    reason = "Stop-Loss"
                elif take_profit_pct and pnl_percentage >= abs(take_profit_pct):
                    reason = "Take-Profit"

                if reason:
                    logging.warning(f"{reason.upper()} TRIGGERED for {p.contract.localSymbol} at {pnl_percentage:.2f}%.")
                    send_pushover_notification(config, f"Risk Alert: {reason} Triggered", f"{p.contract.localSymbol} closed. PnL: {pnl_percentage:.2f}%")

                    # Close the position
                    contract_to_close = p.contract
                    order = MarketOrder('BUY' if p.position < 0 else 'SELL', abs(p.position))
                    trade = ib.placeOrder(contract_to_close, order)

                    # Add conId to the set to prevent duplicate closing orders
                    closed_order_ids.add(p.contract.conId)

                    # Wait for the order to fill
                    asyncio.create_task(wait_for_fill(ib, trade, config, reason=reason))
                    break # Move to the next check cycle after closing a position

        except (asyncio.CancelledError, ConnectionError):
            logging.info("Risk monitor stopping due to connection loss or cancellation.")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the risk monitor loop: {e}", exc_info=True)
            # Continue running unless it's a critical error
            await asyncio.sleep(60) # Wait a minute before retrying after an error