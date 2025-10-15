"""Standalone script for monitoring positions and executing trades.

This script serves as the primary 'listener' process. It runs as a long-lived
process, managed by the main orchestrator. It has two main responsibilities:
1.  Continuously monitor open positions against risk thresholds (P&L).
2.  Poll the order queue for new trading signals, then take full responsibility
    for executing them. This includes fetching market data, defining the
    strategy, pricing the legs, creating the order, and placing it, all using
    a single, persistent connection to the broker to avoid race conditions.
"""

import asyncio
import json
import logging
import os
import random
import signal
import time
import traceback
from ib_insync import *

from config_loader import load_config
from logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.risk_management import monitor_positions_for_risk
from trading_bot.utils import log_trade_to_ledger, log_order_event
from trading_bot.ib_interface import (
    get_active_futures, build_option_chain, create_combo_order_object, place_order
)
from trading_bot.strategy import define_directional_strategy, define_volatility_strategy

# --- Constants ---
ORDER_QUEUE_DIR = "order_queue"

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Global State ---
_filled_order_ids = set()


def _order_status_handler(trade: Trade):
    """Callback for order status updates."""
    logger.info(f"Order {trade.order.orderId} status: {trade.orderStatus.status}")
    log_order_event(trade, trade.orderStatus.status, trade.orderStatus.whyHeld)

    if trade.orderStatus.status == OrderStatus.Filled:
        if trade.order.orderId not in _filled_order_ids:
            try:
                log_trade_to_ledger(trade, "Daily Strategy Fill")
                _filled_order_ids.add(trade.order.orderId)
                logger.info(f"Logged fill for order {trade.order.orderId} to trade ledger.")
            except Exception as e:
                logger.error(f"Error logging trade for order {trade.order.orderId}: {e}")
    elif trade.orderStatus.status in [OrderStatus.Cancelled, OrderStatus.Inactive]:
        _filled_order_ids.discard(trade.order.orderId)


def _exec_details_handler(trade: Trade, fill: Fill, config: dict):
    """Callback for new fills (executions)."""
    try:
        message = (
            f"<b>✅ Order Executed</b>\n"
            f"<b>{fill.execution.side} {fill.execution.shares}</b> of {trade.contract.localSymbol}\n"
            f"Avg Price: ${fill.execution.avgPrice:.2f}\n"
            f"Commission: ${fill.commissionReport.commission:.2f} {fill.commissionReport.currency}\n"
            f"Order ID: {fill.execution.orderId}"
        )
        logger.info(f"Sending execution notification for Order ID {fill.execution.orderId}")
        send_pushover_notification(config.get('notifications', {}), "Order Execution", message)
    except Exception as e:
        logger.error(f"Error processing execution notification for order ID {trade.order.orderId}: {e}")


async def process_signal_and_place_order(ib: IB, config: dict, signal: dict):
    """
    Takes a raw signal, orchestrates the entire order creation and placement
    process, and places the trade.
    """
    logger.info(f"--- Processing Signal for contract month: {signal.get('contract_month')} ---")
    try:
        # Step 1: Find the corresponding active future for the signal
        active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
        future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
        if not future:
            logger.warning(f"No active future found for signal month {signal.get('contract_month')}. Aborting.")
            return

        # Step 2: Get the underlying market price
        logger.info(f"Requesting market price for {future.localSymbol}...")
        ticker = ib.reqMktData(future, '', False, False)
        price = float('nan')
        try:
            start_time = time.time()
            while util.isNan(ticker.marketPrice()):
                await asyncio.sleep(0.1)
                if (time.time() - start_time) > 30:
                    logger.error(f"Timeout waiting for market price for {future.localSymbol}.")
                    break
            else:
                market_price = ticker.marketPrice()
                if not util.isNan(market_price):
                    price = market_price
                    logger.info(f"Successfully received market price for {future.localSymbol}: {price}")
                else:
                    logger.error(f"Received invalid NaN price for {future.localSymbol}. Ticker: {ticker}")
        finally:
            ib.cancelMktData(future)

        if util.isNan(price):
            logger.error("Could not obtain a valid market price. Aborting order placement.")
            return

        # Step 3: Build option chain and define the strategy
        chain = await build_option_chain(ib, future)
        if not chain:
            logger.error(f"Could not build option chain for {future.localSymbol}. Aborting.")
            return

        define_func = define_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else define_volatility_strategy
        strategy_def = define_func(config, signal, chain, price, future)
        if not strategy_def:
            logger.warning(f"Strategy definition failed for {future.localSymbol}. No order placed.")
            return

        # Step 4: Create the combo order object (pricing and qualification)
        order_objects = await create_combo_order_object(ib, config, strategy_def)
        if not order_objects:
            logger.error(f"Failed to create combo order object for {future.localSymbol}. Aborting.")
            return

        # Step 5: Place the final order
        contract, order = order_objects
        place_order(ib, contract, order)
        logger.info(f"--- Successfully Placed Order for {future.localSymbol} ---")

    except Exception as e:
        logger.error(f"An unexpected error occurred during signal processing for {signal.get('contract_month')}: {e}")
        logger.error(traceback.format_exc())


async def poll_order_queue_and_place(ib: IB, config: dict):
    """Continuously polls the signal queue and processes new signals."""
    logger.info(f"Starting signal queue poller for directory: '{ORDER_QUEUE_DIR}'")
    os.makedirs(ORDER_QUEUE_DIR, exist_ok=True)
    while True:
        try:
            for filename in os.listdir(ORDER_QUEUE_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(ORDER_QUEUE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            signal_data = json.load(f)

                        logger.info(f"Processing signal from file: {filename}")
                        # The core logic is now encapsulated in this function
                        await process_signal_and_place_order(ib, config, signal_data)

                        os.remove(filepath)
                        logger.info(f"Successfully processed and removed signal file: {filename}")

                    except Exception as e:
                        logger.error(f"Error processing signal file {filename}: {e}")
                        os.rename(filepath, filepath + ".failed")

            await asyncio.sleep(config.get("strategy_tuning", {}).get("queue_poll_interval", 10))
        except asyncio.CancelledError:
            logger.info("Signal queue poller task cancelled."); break
        except Exception as e:
            logger.error(f"Error in signal queue poller loop: {e}"); await asyncio.sleep(30)


async def main():
    """Main entry point for the position monitoring and execution script."""
    config = load_config()
    if not config:
        logger.critical("Position monitor cannot start without a valid configuration.")
        return

    ib = IB()

    async def shutdown(sig: signal.Signals):
        logger.info(f"Received shutdown signal: {sig.name}. Disconnecting from IB...")
        if ib.isConnected():
            ib.disconnect()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete.")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    tasks = []
    try:
        conn_settings = config.get('connection', {})
        client_id = conn_settings.get('clientId', 55)
        logger.info(f"Connecting to {conn_settings.get('host', '127.0.0.1')}:{conn_settings.get('port', 7497)} with client ID {client_id} for central monitoring...")
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=client_id,
            timeout=30
        )

        exec_handler_with_config = lambda t, f: _exec_details_handler(t, f, config)
        ib.orderStatusEvent += _order_status_handler
        ib.execDetailsEvent += exec_handler_with_config
        logger.info("Central event handlers for fills and status updates are now active.")

        send_pushover_notification(
            config.get('notifications', {}),
            "Central Monitor Online",
            "The central monitoring and execution service is now online."
        )

        pnl_monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
        tasks.append(pnl_monitor_task)

        queue_poller_task = asyncio.create_task(poll_order_queue_and_place(ib, config))
        tasks.append(queue_poller_task)

        logger.info("P&L monitoring and Signal Queue polling tasks are now running concurrently.")

        # Signal to the orchestrator that the monitor is fully initialized and ready.
        print("MONITOR_READY", flush=True)

        await asyncio.gather(*tasks)

    except ConnectionRefusedError:
        logger.critical("Connection to TWS/Gateway was refused. Is it running?")
    except Exception as e:
        error_msg = f"A critical error occurred in the central monitor: {e}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        send_pushover_notification(config.get('notifications', {}), "Monitor CRITICAL ERROR", error_msg)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if ib.isConnected():
            ib.disconnect()
        logger.info("Central monitor has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Position monitor stopped by user.")