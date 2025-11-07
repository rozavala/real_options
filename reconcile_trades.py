"""
A standalone script to reconcile trades from Interactive Brokers with the local trade ledger.

This tool fetches trade executions from the last 24 hours from IB, compares them
against the local `trade_ledger.csv` and its archives, and outputs any missing
trades to a `missing_trades.csv` file.

Note: The Interactive Brokers API via `reqExecutionsAsync` only provides fills
from the last 24 hours. This script is therefore designed for daily reconciliation.
"""

import asyncio
import csv
import logging
import os
from datetime import datetime

import pandas as pd
from ib_insync import IB, Fill

from config_loader import load_config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeReconciler")


def write_missing_trades_to_csv(missing_trades: list[Fill]):
    """
    Writes the list of missing trade fills to a `missing_trades.csv` file
    in the project's root directory. The format matches the `trade_ledger.csv`.
    """
    if not missing_trades:
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'missing_trades.csv')

    fieldnames = [
        'timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity',
        'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
    ]

    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for fill in missing_trades:
                contract = fill.contract
                execution = fill.execution

                # --- Calculate Total Value ---
                try:
                    multiplier = float(contract.multiplier) if contract.multiplier else 37500.0
                except (ValueError, TypeError):
                    multiplier = 37500.0  # Default for coffee futures

                total_value = (execution.price * execution.shares * multiplier) / 100.0
                action = 'BUY' if execution.side == 'BOT' else 'SELL'
                if action == 'BUY':
                    total_value *= -1

                # --- Assemble Row ---
                row = {
                    'timestamp': execution.time.strftime('%Y-%m-%d %H:%M:%S'),
                    # Position ID would need to be manually determined or inferred.
                    # For reconciliation, we can leave it blank or use the permId.
                    'position_id': f"permId_{execution.permId}",
                    'combo_id': execution.permId,
                    'local_symbol': contract.localSymbol,
                    'action': action,
                    'quantity': execution.shares,
                    'avg_fill_price': execution.price,
                    'strike': contract.strike if hasattr(contract, 'strike') else 'N/A',
                    'right': contract.right if hasattr(contract, 'right') else 'N/A',
                    'total_value_usd': f"{total_value:.2f}",
                    'reason': 'RECONCILIATION_MISSING'
                }
                writer.writerow(row)

        logger.info(f"Successfully wrote {len(missing_trades)} missing trade(s) to '{output_path}'.")

    except IOError as e:
        logger.error(f"Error writing to missing trades CSV file: {e}")


def reconcile_trades(ib_trades: list[Fill], local_trades_df: pd.DataFrame) -> list[Fill]:
    """
    Compares the list of IB trades with the local trade ledger and returns
    the IB trades that are missing from the ledger.

    A unique key is generated for each trade to handle the comparison.
    """
    if local_trades_df.empty:
        return ib_trades

    # --- Generate a set of unique keys for local trades for efficient lookup ---
    local_trade_keys = set()
    for _, row in local_trades_df.iterrows():
        if 'combo_id' in row and 'local_symbol' in row and 'timestamp' in row:
            # The timestamp is already a timezone-aware pandas timestamp object in UTC
            key = (
                str(row['combo_id']),
                row['local_symbol'],
                row['timestamp'].round('S')  # Round to the nearest second
            )
            local_trade_keys.add(key)

    # --- Find IB trades that are not in the local ledger ---
    missing_trades = []
    for fill in ib_trades:
        # Generate the same unique key for the IB trade.
        # The fill.time is already a timezone-aware datetime object in UTC.
        ib_key = (
            str(fill.execution.permId),
            fill.contract.localSymbol,
            fill.time.replace(microsecond=0)  # Round to the nearest second
        )

        if ib_key not in local_trade_keys:
            missing_trades.append(fill)

    logger.info(f"Reconciliation complete. Found {len(missing_trades)} missing trade(s).")
    return missing_trades


def get_trade_ledger_df() -> pd.DataFrame:
    """
    Reads and consolidates the main and archived trade ledgers into a single
    DataFrame for analysis.
    """
    # Use an absolute path relative to this script's location to find the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive_ledger')

    dataframes = []

    # --- Load Main Ledger ---
    if os.path.exists(ledger_path):
        try:
            dataframes.append(pd.read_csv(ledger_path))
            logger.info(f"Loaded {len(dataframes[-1])} trades from the main ledger.")
        except pd.errors.EmptyDataError:
            logger.warning("The main trade ledger is empty.")
        except Exception as e:
            logger.error(f"Error reading main trade ledger: {e}")

    # --- Load Archived Ledgers ---
    if os.path.exists(archive_dir):
        archive_files = [
            os.path.join(archive_dir, f)
            for f in os.listdir(archive_dir)
            if f.startswith('trade_ledger_') and f.endswith('.csv')
        ]
        logger.info(f"Found {len(archive_files)} archived ledger files.")
        for file in archive_files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error reading archived ledger '{file}': {e}")

    if not dataframes:
        logger.warning("No trade ledger data found.")
        return pd.DataFrame()

    # --- Consolidate and Process ---
    full_ledger = pd.concat(dataframes, ignore_index=True)
    if 'timestamp' in full_ledger.columns:
        # Convert timestamp to datetime objects, assuming they are in UTC
        full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp']).dt.tz_localize('UTC')
    else:
        logger.error("Consolidated ledger is missing the 'timestamp' column.")
        return pd.DataFrame()

    logger.info(f"Consolidated a total of {len(full_ledger)} trades from all ledgers.")
    return full_ledger


async def get_ib_trades(config: dict) -> list[Fill]:
    """
    Connects to Interactive Brokers and fetches all trade executions (fills)
    from the last 24 hours.
    """
    ib = IB()
    conn_settings = config.get('connection', {})
    # Use a unique client ID for the reconciliation script to avoid conflicts
    client_id = conn_settings.get('client_id_reconciliation', 999)

    try:
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=client_id,
            timeout=15
        )
        logger.info("Successfully connected to IB.")

        # reqExecutionsAsync() only returns fills from the last 24 hours.
        fills = await ib.reqExecutionsAsync()
        logger.info(f"Fetched {len(fills)} total fills from IB's 24-hour history.")
        return fills

    except asyncio.TimeoutError:
        logger.error("Connection to IB timed out.", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Failed to get live trade data from IB: {e}", exc_info=True)
        return []
    finally:
        if ib.isConnected():
            ib.disconnect()
            logger.info("Disconnected from IB.")


async def main():
    """
    Main function to orchestrate the trade reconciliation process.
    """
    logger.info("Starting trade reconciliation for the last 24 hours.")

    # --- 1. Load Configuration ---
    config = load_config()
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        return

    # --- 2. Fetch Trades from IB (Last 24 hours) ---
    ib_trades = await get_ib_trades(config)
    if not ib_trades:
        logger.warning("No trades were fetched from IB for the past 24 hours. Exiting.")
        return

    # --- 3. Load Local Trade Ledger ---
    local_trades_df = get_trade_ledger_df()
    if local_trades_df.empty:
        logger.warning("Local trade ledger is empty. All fetched IB trades will be considered missing.")

    # --- 4. Reconcile Trades ---
    missing_trades = reconcile_trades(ib_trades, local_trades_df)

    if not missing_trades:
        logger.info("No missing trades found. The local ledger is up to date.")
        return

    # --- 5. Output Missing Trades ---
    write_missing_trades_to_csv(missing_trades)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Trade reconciler script manually terminated.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
