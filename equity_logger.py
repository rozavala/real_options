import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
import pandas as pd
from ib_insync import IB

# Setup logging
logger = logging.getLogger("EquityLogger")

async def log_equity_snapshot(config: dict):
    """
    Connects to IB, fetches NetLiquidation, and logs it to data/daily_equity.csv.
    """
    logger.info("--- Starting Equity Snapshot Logging ---")

    file_path = os.path.join("data", "daily_equity.csv")

    # 1. Connect to IB
    ib = IB()
    try:
        host = config['connection']['host']
        port = config['connection']['port']
        # Use a random client ID to avoid conflicts
        client_id = random.randint(2000, 9999)

        logger.info(f"Connecting to IB at {host}:{port} with clientId {client_id}...")
        await ib.connectAsync(host, port, clientId=client_id)

        # 2. Fetch Net Liquidation
        # accountSummary returns a list of AccountValue objects
        # Note: accountSummary() is synchronous, but reqAccountSummaryAsync is async.
        # ib_insync's accountSummary() returns cached values if subscribed, or fetches if not?
        # Actually accountSummary() is a blocking call in sync mode. In async mode, we should use:
        # await ib.reqAccountSummaryAsync() might not be available or returns subscription.
        # correct way in async:
        # await ib.connectAsync(...)
        # values = await ib.accountSummaryAsync() (if available) or:
        # In newer ib_insync, accountSummary() fetches if not subscribed.
        # However, to be safe, we can run it in executor or check docs.
        # Standard usage:
        # summary = await ib.accountSummaryAsync()
        # But wait, looking at ib_insync source/docs, accountSummary() is often synchronous blocking call?
        # No, let's use the one that definitely works.
        # We can use ib.accountSummary() but inside a loop it might block.
        # But since we are connected via connectAsync, we should use non-blocking calls.
        # Actually, let's stick to the method that retrieves data.

        # Use the async method to avoid blocking the event loop or causing runtime errors
        summary = await ib.accountSummaryAsync()

        net_liq = 0.0
        for item in summary:
            if item.tag == 'NetLiquidation':
                try:
                    net_liq = float(item.value)
                    logger.info(f"Fetched NetLiquidation: ${net_liq:,.2f}")
                    break
                except ValueError:
                    logger.warning(f"Could not parse NetLiquidation value: {item.value}")

        if net_liq == 0.0:
            logger.warning("NetLiquidation was 0.0 or not found. Skipping log.")
            ib.disconnect()
            return

        # 3. Prepare Data
        # We use today's date (normalized to midnight for daily series)
        today = datetime.now().date()
        new_row = {'timestamp': pd.Timestamp(today), 'total_value_usd': net_liq}

        # 4. Load or Create CSV
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.info(f"{file_path} not found. Creating new file with backfill.")
            # Backfill logic: Assume $50k start if file is new
            start_date = pd.Timestamp(today - timedelta(days=1))
            df = pd.DataFrame([
                {'timestamp': start_date, 'total_value_usd': 50000.0}
            ])

        # 5. Append and Save
        df = df[df['timestamp'].dt.date != today]

        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

        df.sort_values('timestamp', inplace=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Updated {file_path} with value ${net_liq:,.2f}")

    except Exception as e:
        logger.error(f"Failed to log equity snapshot: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB.")

if __name__ == "__main__":
    # Standalone execution
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from config_loader import load_config
    from logging_config import setup_logging

    setup_logging()
    cfg = load_config()
    if cfg:
        asyncio.run(log_equity_snapshot(cfg))
    else:
        print("Could not load config.")
