"""Standalone script for monitoring open trading positions for risk.

This script serves as the main entry point for the intraday risk monitoring
process. It runs as a separate, long-lived process, managed by the main
orchestrator. Its sole responsibility is to connect to Interactive Brokers
and run the `monitor_positions_for_risk` function, which continuously checks
open positions against configured stop-loss and take-profit thresholds.
"""

import asyncio
import json
import logging
import os
import random
import signal
import traceback
from ib_insync import *

from config_loader import load_config
from logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.risk_management import monitor_positions_for_risk
from trading_bot.utils import log_trade_to_ledger, log_order_event
from trading_bot.ib_interface import place_order

# --- Constants ---
ORDER_QUEUE_DIR = "order_queue"

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Global State ---
# A set to keep track of order IDs that have already been logged as filled.
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
        # Create a detailed notification message
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


async def poll_order_queue_and_place(ib: IB, config: dict):
    """Continuously polls the order queue directory and places new orders."""
    logger.info(f"Starting order queue poller for directory: '{ORDER_QUEUE_DIR}'")
    while True:
        try:
            for filename in os.listdir(ORDER_QUEUE_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(ORDER_QUEUE_DIR, filename)
                    try:
                        with open(filepath, 'r') as f:
                            order_data = json.load(f)

                        # Reconstruct the contract and order objects
                        contract = Contract(**order_data['contract'])
                        order = Order(**order_data['order'])

                        logger.info(f"Placing order from file: {filename}")
                        place_order(ib, contract, order)

                        # Remove the file after processing
                        os.remove(filepath)
                        logger.info(f"Successfully processed and removed order file: {filename}")

                    except Exception as e:
                        logger.error(f"Error processing order file {filename}: {e}")
                        # Move problematic file to avoid reprocessing loop
                        os.rename(filepath, filepath + ".failed")

            await asyncio.sleep(config.get("strategy_tuning", {}).get("queue_poll_interval", 10))
        except asyncio.CancelledError:
            logger.info("Order queue poller task cancelled."); break
        except Exception as e:
            logger.error(f"Error in order queue poller loop: {e}"); await asyncio.sleep(30)


async def main():
    """Main entry point for the position monitoring script.

    This function performs the following steps:
    1. Loads the application configuration.
    2. Establishes a connection to the Interactive Brokers Gateway/TWS using a
       dedicated client ID.
    3. Sets up signal handlers for graceful shutdown (SIGINT, SIGTERM).
    4. Starts and awaits the `monitor_positions_for_risk` task, which contains
       the main risk monitoring loop.
    5. Ensures disconnection and task cancellation on shutdown or error.
    """
    config = load_config()
    if not config:
        logger.critical("Position monitor cannot start without a valid configuration.")
        return

    ib = IB()

    async def shutdown(sig: signal.Signals):
        """Handles graceful shutdown on receiving a signal.

        This function logs the shutdown signal, cancels all running asyncio
        tasks, and disconnects from the IB Gateway.

        Args:
            sig (signal.Signals): The signal that triggered the shutdown.
        """
        logger.info(f"Received shutdown signal: {sig.name}. Disconnecting from IB...")
        if ib.isConnected():
            ib.disconnect()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete.")

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    tasks = []
    try:
        conn_settings = config.get('connection', {})
        host = conn_settings.get('host', '127.0.0.1')
        port = conn_settings.get('port', 7497)
        client_id = conn_settings.get('clientId', 55) + random.randint(1, 100)

        logger.info(f"Connecting to {host}:{port} with client ID {client_id} for central monitoring...")
        await ib.connectAsync(host, port, clientId=client_id, timeout=30)

        # --- Set up event handlers ---
        # Create a lambda to pass the config to the fill handler
        exec_handler_with_config = lambda t, f: _exec_details_handler(t, f, config)
        ib.orderStatusEvent += _order_status_handler
        ib.execDetailsEvent += exec_handler_with_config
        logger.info("Central event handlers for fills and status updates are now active.")

        send_pushover_notification(
            config.get('notifications', {}),
            "Central Monitor Online",
            "The central monitoring and execution service is now online."
        )

        # --- Start concurrent tasks ---
        pnl_monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
        tasks.append(pnl_monitor_task)

        queue_poller_task = asyncio.create_task(poll_order_queue_and_place(ib, config))
        tasks.append(queue_poller_task)

        logger.info("P&L monitoring and Order Queue polling tasks are now running concurrently.")
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
