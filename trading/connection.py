import asyncio
import logging
import random
from ib_insync import IB
from notifications import send_pushover_notification

# --- Connection Management ---

async def connect_to_ibkr(ib: IB, config: dict):
    """
    Establishes and manages the connection to IBKR TWS/Gateway.
    Returns True if connection is successful, False otherwise.
    """
    if ib.isConnected():
        logging.info("IBKR connection is already active.")
        return True

    conn_settings = config.get('connection', {})
    host = conn_settings.get('host', '127.0.0.1')
    port = conn_settings.get('port', 7497)
    client_id = conn_settings.get('clientId', random.randint(1, 1000))

    try:
        logging.info(f"Connecting to IBKR at {host}:{port} with ClientID {client_id}...")
        await ib.connectAsync(host, port, clientId=client_id)
        ib.RequestTimeout = 10  # Set a default timeout for requests
        logging.info("Successfully connected to IBKR.")
        send_pushover_notification(config, "Bot Status: CONNECTED", "Trading script has successfully connected to IBKR.")
        return True
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
        error_msg = f"Connection to IBKR failed: {e}. Please ensure TWS/Gateway is running and API settings are correct."
        logging.error(error_msg)
        send_pushover_notification(config, "Bot Status: CONNECTION FAILED", error_msg)
        return False

def disconnect_from_ibkr(ib: IB):
    """Disconnects from IBKR if a connection is active."""
    if ib.isConnected():
        logging.info("Disconnecting from IBKR.")
        ib.disconnect()

async def send_heartbeat(ib: IB):
    """
    A background task that periodically checks the connection status
    to ensure it remains alive.
    """
    logging.info("Heartbeat task started.")
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            if ib.isConnected():
                server_time = await ib.reqCurrentTimeAsync()
                if server_time:
                    logging.info(f"Heartbeat: Connection is alive. Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    logging.warning("Heartbeat: Connection check returned no time. May be unstable.")
            else:
                logging.warning("Heartbeat: Connection is lost. Attempting to reconnect on the next cycle.")
                # The main loop will handle the actual reconnection logic.
                break
        except Exception as e:
            logging.error(f"Error in heartbeat loop: {e}")
            break # Exit if there's an unrecoverable error