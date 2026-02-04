"""Standalone script for monitoring open trading positions for risk.

This script serves as the main entry point for the intraday risk monitoring
process. It runs as a separate, long-lived process, managed by the main
orchestrator. Its sole responsibility is to connect to Interactive Brokers
and run the `monitor_positions_for_risk` function, which continuously checks
open positions against configured stop-loss and take-profit thresholds.
"""

import asyncio
import logging
import random
import signal
import traceback

from ib_insync import IB

from config_loader import load_config
from trading_bot.logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.risk_management import monitor_positions_for_risk
from trading_bot.utils import configure_market_data_type

# --- Logging Setup ---
setup_logging(log_file=None)
logger = logging.getLogger(__name__)


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
            # === NEW: Give Gateway time to cleanup ===
            await asyncio.sleep(3.0)
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Shutdown complete.")

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    monitor_task = None
    try:
        conn_settings = config.get('connection', {})
        host = conn_settings.get('host', '127.0.0.1')
        port = conn_settings.get('port', 7497)
        # Use a client ID offset from the main one to avoid conflicts
        # Using 300+ range to avoid collision with IBConnectionPool (100-200)
        client_id = 300 + random.randint(0, 99)

        logger.info(f"Connecting to {host}:{port} with client ID {client_id} for monitoring...")
        await ib.connectAsync(host, port, clientId=client_id)

        # --- FIX: Configure Market Data Type based on Environment ---
        configure_market_data_type(ib)
        # ------------------------------------------------------------

        send_pushover_notification(
            config.get('notifications', {}),
            "Position Monitor Started",
            "The position monitoring service has started and is now watching open positions."
        )

        # Start the main risk monitoring loop
        monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
        await monitor_task

    except ConnectionRefusedError:
        logger.critical("Connection to TWS/Gateway was refused. Is it running?")
    except Exception as e:
        error_msg = f"A critical error occurred in the position monitor: {e}\n{traceback.format_exc()}"
        logger.critical(error_msg)
        send_pushover_notification(config.get('notifications', {}), "Monitor CRITICAL ERROR", error_msg)
    finally:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
        if ib.isConnected():
            ib.disconnect()
            # === NEW: Give Gateway time to cleanup ===
            await asyncio.sleep(3.0)
        logger.info("Position monitor has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Position monitor stopped by user.")
