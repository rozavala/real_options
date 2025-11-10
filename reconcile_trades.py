"""
A standalone script to reconcile trades from Interactive Brokers with the local trade ledger.

This tool fetches trade executions from an IBKR Flex Query, compares them
against the local `trade_ledger.csv` and its archives, and outputs any missing
trades to a `missing_trades.csv` file.
"""

import asyncio
import csv
import io
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx  # Added for HTTP requests
import pandas as pd
# Removed ib_insync imports

from config_loader import load_config

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeReconciler")


def write_missing_trades_to_csv(missing_trades_df: pd.DataFrame):
    """
    Writes the DataFrame of missing trades to a `missing_trades.csv` file.
    The format matches the `trade_ledger.csv`.
    """
    if missing_trades_df.empty:
        return

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'missing_trades.csv')

    fieldnames = [
        'timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity',
        'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
    ]

    try:
        # Create the final DataFrame with the 'reason' column and ensure field order
        final_df = missing_trades_df.copy()
        final_df['reason'] = 'RECONCILIATION_MISSING'
        
        # Format timestamp to string for CSV
        final_df['timestamp'] = final_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Reorder columns to match the ledger
        final_df = final_df.reindex(columns=fieldnames)
        
        final_df.to_csv(
            output_path, 
            index=False, 
            header=True, 
            float_format='%.2f'
        )
        
        logger.info(f"Successfully wrote {len(final_df)} missing trade(s) to '{output_path}'.")

    except IOError as e:
        logger.error(f"Error writing to missing trades CSV file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in write_missing_trades_to_csv: {e}")


def reconcile_trades(ib_trades_df: pd.DataFrame, local_trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares the IB trades DataFrame with the local trade ledger DataFrame
    and returns a DataFrame of IB trades that are missing from the ledger.
    """
    if local_trades_df.empty:
        logger.warning("Local ledger is empty. All IB trades will be marked as missing.")
        return ib_trades_df

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
    
    logger.info(f"Loaded {len(local_trade_keys)} unique trade keys from local ledger.")

    # --- Find IB trades that are not in the local ledger ---
    missing_rows = []
    for _, row in ib_trades_df.iterrows():
        # Generate the same unique key for the IB trade.
        ib_key = (
            str(row['combo_id']),
            row['local_symbol'],
            row['timestamp'].round('S')  # Round to the nearest second
        )

        if ib_key not in local_trade_keys:
            missing_rows.append(row)

    logger.info(f"Reconciliation complete. Found {len(missing_rows)} missing trade(s).")
    return pd.DataFrame(missing_rows)


def get_trade_ledger_df() -> pd.DataFrame:
    """
    Reads and consolidates the main and archived trade ledgers into a single
    DataFrame for analysis.
    (This function is unchanged from your original script)
    """
    # Use an absolute path relative to this script's location to find the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive_ledger')

    dataframes = []

    # --- Load Main Ledger ---
    if os.path.exists(ledger_path):
        try:
            df = pd.read_csv(ledger_path)
            if not df.empty:
                dataframes.append(df)
                logger.info(f"Loaded {len(dataframes[-1])} trades from the main ledger.")
            else:
                logger.warning("The main trade ledger is empty.")
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
                if not df.empty:
                    dataframes.append(df)
            except pd.errors.EmptyDataError:
                logger.warning(f"Archived ledger '{file}' is empty.")
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


async def fetch_flex_query_report(config: dict) -> str | None:
    """
    Connects to the IBKR Flex Web Service to request and download a trade report.
    """
    try:
        token = config['flex_query']['token']
        query_id = config['flex_query']['query_id']
    except KeyError:
        logger.critical("Config file is missing 'flex_query' section with 'token' and 'query_id'.")
        return None

    base_url = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService."
    request_url = f"{base_url}SendRequest?t={token}&q={query_id}&v=3"
    
    async with httpx.AsyncClient() as client:
        try:
            # --- 1. Send the initial request for the report ---
            logger.info(f"Requesting Flex Query report (Query ID: {query_id})...")
            resp = await client.get(request_url, timeout=30.0)
            resp.raise_for_status()

            # --- 2. Parse the XML response to get the Reference Code ---
            root = ET.fromstring(resp.content)
            status = root.find('Status').text
            if status != 'Success':
                error_code = root.find('ErrorCode').text
                error_msg = root.find('ErrorMessage').text
                logger.error(f"IB Flex Query request failed. Code: {error_code}, Msg: {error_msg}")
                return None
            
            ref_code = root.find('ReferenceCode').text
            logger.info(f"Successfully requested report. Reference Code: {ref_code}")

            # --- 3. Poll the server to get the statement ---
            get_url = f"{base_url}GetStatement?t={token}&q={ref_code}&v=3"
            for i in range(10):  # Poll up to 10 times
                await asyncio.sleep(10)  # Wait 10 seconds between polls
                logger.info(f"Polling for report (Attempt {i+1}/10)...")
                
                resp = await client.get(get_url, timeout=30.0)
                
                # The report is ready when the response is not an XML error message
                # A successful CSV report will not start with '<?xml'
                if not resp.text.strip().startswith('<?xml'):
                    logger.info("Successfully downloaded Flex Query report.")
                    return resp.text
                
                # If it IS an XML, it's probably an error or "still processing"
                root = ET.fromstring(resp.content)
                status = root.find('Status').text
                if status != 'Success':
                    error_code = root.find('ErrorCode').text
                    if error_code == '1018': # "Report is not ready"
                         logger.info("Report not yet ready, will poll again.")
                    else:
                        error_msg = root.find('ErrorMessage').text
                        logger.error(f"IB Flex Query retrieval failed. Code: {error_code}, Msg: {error_msg}")
                        return None

            logger.error("Failed to retrieve report after 10 attempts.")
            return None

        except httpx.RequestError as e:
            logger.error(f"HTTP error occurred while fetching Flex Query: {e}", exc_info=True)
            return None
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML response from IB: {e}", exc_info=True)
            return None


def parse_flex_csv_to_df(csv_data: str) -> pd.DataFrame:
    """
    Parses the raw CSV string from the Flex Query into a clean DataFrame
    that matches the script's required format.
    """
    # Find the start of the 'Executions' data block
    lines = csv_data.splitlines()
    data_start_index = -1
    header = []

    for i, line in enumerate(lines):
        if line.startswith("Executions,Data"):
            header = lines[i+1].split(',')
            data_start_index = i + 2
            break

    if data_start_index == -1:
        logger.error("Could not find 'Executions,Data' section in the Flex Query report.")
        return pd.DataFrame()

    # Clean the header (e.g., "Executions,Symbol" -> "Symbol")
    header[0] = header[0].split(',')[1]

    # Find the end of the data block
    data_end_index = len(lines)
    for i, line in enumerate(lines[data_start_index:], start=data_start_index):
        if not line.startswith("Executions,Data"):
            data_end_index = i
            break
            
    # Extract the relevant data lines and parse
    data_rows = []
    reader = csv.reader(lines[data_start_index:data_end_index])
    for row in reader:
        # Clean the first element (e.g., "Executions,Data" -> "Data")
        row[0] = row[0].split(',')[1]
        data_rows.append(row)

    if not data_rows:
        logger.warning("Executions section was found but contained no data.")
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=header)
    logger.info(f"Parsed {len(df)} executions from the Flex Query report.")

    # --- Data Cleaning and Type Conversion ---
    try:
        df['Trade Price'] = pd.to_numeric(df['Trade Price'])
        df['Quantity'] = pd.to_numeric(df['Quantity'])
        # Use user's default multiplier if missing
        df['Multiplier'] = pd.to_numeric(df['Multiplier'].replace('', '37500.0')).fillna(37500.0)
        df['Strike'] = pd.to_numeric(df['Strike'].replace('', '0')).fillna(0)
        # IB timestamps are typically EST/EDT. tz_localize(None).tz_localize('UTC')
        # is a robust way to convert them to UTC, assuming they are naive.
        # A safer way is to know the report's TZ. Assuming 'America/New_York'
        df['Trade Date/Time'] = pd.to_datetime(df['Trade Date/Time']).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    except Exception as e:
        logger.error(f"Error during data type conversion: {e}", exc_info=True)
        return pd.DataFrame()

    # --- Map to script's internal column names ---
    df_out = pd.DataFrame()
    df_out['timestamp'] = df['Trade Date/Time']
    df_out['position_id'] = 'transId_' + df['Transaction ID']
    df_out['combo_id'] = df['Transaction ID']
    df_out['local_symbol'] = df['Symbol']
    df_out['action'] = df['Buy/Sell'].apply(lambda x: 'BUY' if x == 'B' else 'SELL')
    df_out['quantity'] = df['Quantity']
    df_out['avg_fill_price'] = df['Trade Price']
    df_out['strike'] = df['Strike'].replace(0, 'N/A').astype(str)
    df_out['right'] = df['Put/Call'].replace('', 'N/A')

    # --- Calculate Value (using your original logic) ---
    # Defaulting to 37500.0 multiplier was in your original script
    multiplier = df['Multiplier']
    
    total_value = (df['Trade Price'] * df['Quantity'] * multiplier) / 100.0
    df_out['total_value_usd'] = total_value.where(df_out['action'] == 'SELL', -total_value)

    return df_out


async def main():
    """
    Main function to orchestrate the trade reconciliation process.
    """
    logger.info("Starting trade reconciliation using Flex Query.")

    # --- 1. Load Configuration ---
    config = load_config()
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        return
    if 'flex_query' not in config:
        logger.critical("Configuration is missing the 'flex_query' section. Exiting.")
        return

    # --- 2. Fetch Trades from IB (Flex Query) ---
    csv_data = await fetch_flex_query_report(config)
    if not csv_data:
        logger.warning("No trade data was fetched from IB Flex Query. Exiting.")
        return
        
    # --- 3. Parse Flex Query Report ---
    ib_trades_df = parse_flex_csv_to_df(csv_data)
    if ib_trades_df.empty:
        logger.warning("Failed to parse trades from the Flex Query report. Exiting.")
        return

    # --- 4. Load Local Trade Ledger ---
    local_trades_df = get_trade_ledger_df()
    if local_trades_df.empty:
        logger.warning("Local trade ledger is empty. All fetched IB trades will be considered missing.")

    # --- 5. Reconcile Trades ---
    missing_trades_df = reconcile_trades(ib_trades_df, local_trades_df)

    if missing_trades_df.empty:
        logger.info("No missing trades found. The local ledger is up to date.")
        return

    # --- 6. Output Missing Trades ---
    write_missing_trades_to_csv(missing_trades_df)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Trade reconciler script manually terminated.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
