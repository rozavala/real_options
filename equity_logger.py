import asyncio
import logging
import os
import random
import sys
from datetime import datetime, timedelta, time, timezone
import pandas as pd
from ib_insync import IB
import httpx

# Use absolute imports if running as a script within the package
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from config_loader import load_config
    from trading_bot.logging_config import setup_logging
    from trading_bot.utils import configure_market_data_type
    # We can't easily import from reconcile_trades if it's a script in the same dir
    # unless we treat the dir as a package or add it to path.
    # We already added '.' to path.
    from reconcile_trades import fetch_flex_query_report, parse_flex_csv_to_df
else:
    from config_loader import load_config
    from reconcile_trades import fetch_flex_query_report, parse_flex_csv_to_df
    from trading_bot.utils import configure_market_data_type

# Setup logging
logger = logging.getLogger("EquityLogger")

async def sync_equity_from_flex(config: dict):
    """
    Fetches the last 365 days of Net Asset Value from IBKR Flex Query (ID 1341111)
    and updates `data/{ticker}/daily_equity.csv`.

    The Flex Query returns 'ReportDate' and 'Total'.
    This function normalizes the date to closing time (17:00 NY Time) and saves
    the file to ensure the local equity history matches the broker's official record.

    Only runs for the primary commodity (KC) since NetLiquidation is account-wide.
    Non-primary engines skip to avoid duplicate Flex queries and identical data.
    """
    # Guard: only the primary engine syncs account-wide equity
    is_primary = config.get('commodity', {}).get('is_primary', True)
    if not is_primary:
        logger.info("Equity sync skipped (non-primary engine).")
        return

    logger.info("--- Starting Equity Synchronization from Flex Query ---")

    # Get the ID from the config (loaded from .env), or fallback to os.getenv just in case
    query_id = config.get('flex_query', {}).get('equity_query_id') or os.getenv('FLEX_EQUITY_ID')
    
    if not query_id:
        logger.error("Missing FLEX_EQUITY_ID in configuration. Cannot sync equity.")
        return

    try:
        token = config['flex_query']['token']
    except KeyError:
        logger.error("Config missing 'flex_query' token. Cannot sync equity.")
        return

    # 1. Fetch Report
    csv_data = await fetch_flex_query_report(token, query_id)
    if not csv_data:
        logger.error("Failed to fetch Equity Flex Query report.")
        return

    # 2. Parse CSV from Flex Query
    try:
        # Use fallback for StringIO
        import io
        flex_df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        logger.error(f"Error reading CSV data: {e}")
        return

    if flex_df.empty:
        logger.warning("Equity Flex Query report is empty.")
        return

    # 3. Process Flex Query Data
    if 'ReportDate' not in flex_df.columns or 'Total' not in flex_df.columns:
        logger.error(f"Equity report missing expected columns. Found: {flex_df.columns.tolist()}")
        return

    try:
        # Normalize Date from 'YYYYMMDD' (e.g. 20250602)
        flex_df['ReportDate'] = pd.to_datetime(flex_df['ReportDate'].astype(str), format='%Y%m%d')

        # Set time to 17:00:00 to represent "closing" value consistent with snapshot
        flex_df['timestamp'] = flex_df['ReportDate'] + pd.Timedelta(hours=17)

        # Normalize Total
        flex_df['total_value_usd'] = pd.to_numeric(flex_df['Total'], errors='coerce')

        # Select and clean
        flex_df = flex_df[['timestamp', 'total_value_usd']].dropna()

    except Exception as e:
        logger.error(f"Error processing Equity Flex Query data: {e}", exc_info=True)
        return

    # 4. Merge with Local Data (Preserving extra local data)
    data_dir = config.get('data_dir', 'data')
    file_path = os.path.join(data_dir, "daily_equity.csv")

    final_df = flex_df.copy() # Start with Flex data as base (Source of Truth)

    if os.path.exists(file_path):
        try:
            local_df = pd.read_csv(file_path)
            if not local_df.empty and 'timestamp' in local_df.columns:
                local_df['timestamp'] = pd.to_datetime(local_df['timestamp'])

                # Find timestamps in local that are NOT in Flex
                # We use timestamp as the key.
                # Since we standardized Flex timestamps to 17:00, ensure local ones match or handled correctly.
                # If local has data for a day not in Flex, we keep it.

                # Identify rows in local_df where timestamp is NOT in flex_df['timestamp']
                unique_local = local_df[~local_df['timestamp'].isin(flex_df['timestamp'])]

                if not unique_local.empty:
                    logger.info(f"Preserving {len(unique_local)} local equity records not present in Flex Query.")
                    final_df = pd.concat([final_df, unique_local], ignore_index=True)
        except Exception as e:
            logger.warning(f"Could not read existing local equity file for merging: {e}")

    # 5. Save to CSV
    try:
        # Sort by timestamp
        final_df = final_df.sort_values('timestamp')

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        final_df.to_csv(file_path, index=False)
        logger.info(f"Successfully synced {len(final_df)} equity records to {file_path}.")
    except Exception as e:
         logger.error(f"Failed to write to {file_path}: {e}")


async def log_equity_snapshot(config: dict):
    """
    Connects to IB, fetches NetLiquidation, and logs it to data/{ticker}/daily_equity.csv.

    Only runs for the primary commodity (KC) since NetLiquidation is account-wide.
    Non-primary engines skip to avoid duplicate IB connections and identical data.
    """
    # Guard: only the primary engine logs account-wide equity
    is_primary = config.get('commodity', {}).get('is_primary', True)
    if not is_primary:
        logger.info("Equity snapshot skipped (non-primary engine).")
        return

    logger.info("--- Starting Equity Snapshot Logging ---")

    data_dir = config.get('data_dir', 'data')
    file_path = os.path.join(data_dir, "daily_equity.csv")
    os.makedirs(data_dir, exist_ok=True)

    # 1. Connect to IB
    ib = IB()
    try:
        host = config['connection']['host']
        port = config['connection']['port']
        # Use a random client ID to avoid conflicts
        client_id = random.randint(2000, 9999)

        logger.info(f"Connecting to IB at {host}:{port} with clientId {client_id}...")
        await ib.connectAsync(host, port, clientId=client_id)

        # --- FIX: Configure Market Data Type based on Environment ---
        configure_market_data_type(ib)
        # ------------------------------------------------------------

        # 2. Fetch Net Liquidation
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
        # Use today's date at 17:00 to match the sync format (closing time)
        today = datetime.now(timezone.utc).date()
        timestamp = datetime.combine(today, time(17, 0, 0))

        new_row = {'timestamp': timestamp, 'total_value_usd': net_liq}

        # 4. Load or Create CSV
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.info(f"{file_path} not found. Creating new file with backfill.")
            start_date = timestamp - timedelta(days=1)
            df = pd.DataFrame([
                {'timestamp': start_date, 'total_value_usd': float(os.getenv('INITIAL_CAPITAL', '50000.0'))}
            ])

        # 5. Append and Save
        # Remove existing entry for today if any (to update it)
        # Note: We compare dates only
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
            # === NEW: Give Gateway time to cleanup ===
            await asyncio.sleep(3.0)
            logger.info("Disconnected from IB.")

if __name__ == "__main__":
    # Standalone execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync', action='store_true', help='Sync equity from Flex Query')
    parser.add_argument('--commodity', type=str,
                        default=os.environ.get("COMMODITY_TICKER", "KC"),
                        help="Commodity ticker (e.g. KC, CC)")
    args = parser.parse_args()

    ticker = args.commodity.upper()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', ticker)
    os.makedirs(data_dir, exist_ok=True)

    setup_logging(log_file="logs/equity_logger.log")
    cfg = load_config()

    if cfg:
        cfg['data_dir'] = data_dir
        cfg.setdefault('commodity', {})['ticker'] = ticker
        if args.sync:
            asyncio.run(sync_equity_from_flex(cfg))
        else:
            asyncio.run(log_equity_snapshot(cfg))
    else:
        logger.error("Could not load config.")
        sys.exit(1)
