import asyncio
import logging
import random
import signal
import traceback

from ib_insync import IB

from config_loader import load_config
from logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.risk_management import monitor_positions_for_risk

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger(__name__)


async def main():
    """
    Main entry point for the position monitoring script.
    Connects to IB and starts the risk monitoring loop.
    """
    config = load_config()
    if not config:
        logger.critical("Position monitor cannot start without a valid configuration.")
        return

    ib = IB()

    async def shutdown(sig):
        """Graceful shutdown handler."""
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

    monitor_task = None
    try:
        conn_settings = config.get('connection', {})
        host = conn_settings.get('host', '127.0.0.1')
        port = conn_settings.get('port', 7497)
        client_id = conn_settings.get('clientId', 10) + 1  # Use a different client ID

        logger.info(f"Connecting to {host}:{port} with client ID {client_id} for monitoring...")
        await ib.connectAsync(host, port, clientId=client_id)
        send_pushover_notification(
            config.get('notifications', {}),
            "Position Monitor Started",
            "The position monitoring service has started and is now watching open positions."
        )

        # Start the monitoring loop
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
        logger.info("Position monitor has shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Position monitor stopped by user.")