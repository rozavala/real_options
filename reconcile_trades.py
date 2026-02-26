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
from notifications import send_pushover_notification

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TradeReconciler")


def parse_position_flex_csv_to_df(csv_data: str) -> pd.DataFrame:
    """
    Parses the raw CSV string from the Active Position Flex Query (ID 1341164 in dev and  ID 1370923 in prod)
    into a DataFrame. Expected columns: Symbol, Quantity, CostBasisPrice, FifoPnlUnrealized, OpenDateTime.
    """
    # Define the standard empty return with columns to prevent Merge crashes
    empty_df = pd.DataFrame(columns=['Symbol', 'Quantity'])

    if not csv_data or not csv_data.strip():
        logger.warning("Received empty CSV data from Position Flex Query.")
        return empty_df

    try:
        df = pd.read_csv(io.StringIO(csv_data))
        logger.info(f"Parsed {len(df)} positions from the Flex Query report.")
    except pd.errors.EmptyDataError:
        logger.warning("Position Flex Query report was empty.")
        return empty_df
    except Exception as e:
        logger.error(f"Failed to read Position CSV data: {e}", exc_info=True)
        return empty_df

    if df.empty:
        return empty_df

    # Normalize columns if needed
    required_cols = ['Symbol', 'Quantity']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column '{col}' in Position Flex Query. Available: {df.columns.tolist()}")
            return empty_df

    # Convert Quantity to numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

    return df


def get_local_active_positions(ledger: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculates the system's view of active positions by aggregating the trade ledger.
    Returns a DataFrame with index 'Symbol' and column 'Quantity'.
    """
    if ledger is None:
        ledger = get_trade_ledger_df()

    if ledger.empty:
        return pd.DataFrame(columns=['Symbol', 'Quantity'])

    # Map 'action' to sign: BUY -> +1, SELL -> -1.
    # NOTE: The trade ledger 'quantity' is absolute.
    # Standard convention: Long is +, Short is -.
    # Opening a Long: BUY (+). Closing a Long: SELL (-).
    # Opening a Short: SELL (-). Closing a Short: BUY (+).
    # This implies BUY adds to position (algebraically) and SELL subtracts?
    # Wait.
    # If I Buy 1 contract, my position is +1.
    # If I Sell 1 contract (to close), my position is 0.
    # If I Sell 1 contract (to open short), my position is -1.
    # If I Buy 1 contract (to close short), my position is 0.
    # So: BUY is always +Quantity, SELL is always -Quantity.

    ledger['signed_quantity'] = ledger.apply(
        lambda row: row['quantity'] if row['action'] == 'BUY' else -row['quantity'], axis=1
    )

    # Group by Symbol and sum
    # Note: 'local_symbol' in ledger corresponds to 'Symbol' in Flex Query
    positions = ledger.groupby('local_symbol')['signed_quantity'].sum().reset_index()
    positions.rename(columns={'local_symbol': 'Symbol', 'signed_quantity': 'Quantity'}, inplace=True)

    # Filter out closed positions (Quantity == 0)
    positions = positions[positions['Quantity'] != 0]

    return positions


async def reconcile_active_positions(config: dict):
    """
    Fetches the active positions Flex Query and compares it with the local ledger.
    Sends a notification if discrepancies are found.
    """
    logger.info("Starting Active Position Reconciliation...")

    query_id = config.get('flex_query', {}).get('active_positions_query_id')
    token = config.get('flex_query', {}).get('token')

    if not query_id or not token:
        logger.warning("Missing 'active_positions_query_id' or 'token' in config. Skipping position check.")
        return

    # 1. Fetch IB Active Positions
    csv_data = await fetch_flex_query_report(token, query_id)
    if not csv_data:
        logger.warning("Failed to fetch Active Position report.")
        return

    ib_positions = parse_position_flex_csv_to_df(csv_data)

    # 1b. Filter to active commodity symbol
    # IBKR uses different symbol prefixes for futures vs options:
    # KC futures = "KC*", KC options = "KO*"; CC futures = "CC*", CC options = "DC*"
    ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
    _IBKR_SYMBOL_PREFIXES = {"KC": ("KC", "KO"), "CC": ("CC", "DC"), "SB": ("SB", "SO"), "NG": ("NG", "ON")}
    prefixes = _IBKR_SYMBOL_PREFIXES.get(ticker, (ticker,))
    if not ib_positions.empty:
        pre_count = len(ib_positions)
        ib_positions = ib_positions[ib_positions['Symbol'].apply(lambda s: any(s.startswith(p) for p in prefixes))]
        if len(ib_positions) < pre_count:
            logger.info(f"Filtered IB positions by commodity {ticker} (prefixes {prefixes}): {pre_count} -> {len(ib_positions)}")

    # 2. Get Local Active Positions
    # Load full ledger first to check for recent trades
    full_ledger = get_trade_ledger_df(config.get('data_dir'))
    local_positions = get_local_active_positions(full_ledger)

    # 3. Exclude symbols traded recently (last 24 hours)
    # Positions with recent activity might not be reflected in the Flex Query yet.
    skipped_symbols = set()
    if not full_ledger.empty and 'timestamp' in full_ledger.columns:
        # Assuming timestamps are timezone-aware (UTC) as per get_trade_ledger_df
        now_utc = pd.Timestamp.now(tz='UTC')
        cutoff_time = now_utc - pd.Timedelta(hours=24)

        recent_trades = full_ledger[full_ledger['timestamp'] >= cutoff_time]
        skipped_symbols = set(recent_trades['local_symbol'].unique())

        if skipped_symbols:
            logger.info(f"Skipping reconciliation for recently traded symbols: {skipped_symbols}")

    # 4. Compare
    # Merge on Symbol
    # We use outer join to catch symbols present in one but not the other
    merged = pd.merge(ib_positions, local_positions, on='Symbol', how='outer', suffixes=('_ib', '_local'))

    # Filter out skipped symbols
    merged = merged[~merged['Symbol'].isin(skipped_symbols)]

    merged['Quantity_ib'] = merged['Quantity_ib'].fillna(0)
    merged['Quantity_local'] = merged['Quantity_local'].fillna(0)

    merged['Diff'] = merged['Quantity_ib'] - merged['Quantity_local']

    # Filter for discrepancies (Diff != 0), handling potential float inaccuracies
    discrepancies = merged[merged['Diff'].abs() > 0.0001]

    if not discrepancies.empty:
        logger.warning(f"Found {len(discrepancies)} position discrepancies.")

        message_lines = ["Active Position Discrepancies Found:"]
        for _, row in discrepancies.iterrows():
            sym = row['Symbol']
            ib_qty = row['Quantity_ib']
            loc_qty = row['Quantity_local']
            message_lines.append(f"- {sym}: IB={ib_qty}, Local={loc_qty}")

        message = "\n".join(message_lines)

        # Send Notification
        send_pushover_notification(
            config.get('notifications', {}),
            f"Position Reconciliation Alert [{ticker}]",
            message
        )
    else:
        logger.info("Active Position Reconciliation complete. No discrepancies found.")


def write_missing_trades_to_csv(missing_trades_df: pd.DataFrame, data_dir: str = None):
    """
    Writes the DataFrame of missing trades to a `missing_trades.csv` file
    inside the `archive_ledger` directory.
    The format matches the `trade_ledger.csv`.
    """
    if missing_trades_df.empty:
        return

    if data_dir:
        archive_dir = os.path.join(data_dir, 'archive_ledger')
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        archive_dir = os.path.join(base_dir, 'archive_ledger')
    output_path = os.path.join(archive_dir, 'trade_ledger_missing_trades.csv')

    # Create the archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)

    fieldnames = [
        'timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity',
        'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
    ]

    try:
        # Create the final DataFrame with the 'reason' column and ensure field order
        final_df = missing_trades_df.copy()
        final_df['reason'] = 'RECONCILIATION_MISSING'
        
        # Sort by timestamp before writing
        final_df.sort_values(by='timestamp', inplace=True)

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


def write_superfluous_trades_to_csv(superfluous_trades_df: pd.DataFrame, data_dir: str = None):
    """
    Writes the DataFrame of superfluous local trades to a CSV file
    inside the `archive_ledger` directory.
    """
    if superfluous_trades_df.empty:
        return

    if data_dir:
        archive_dir = os.path.join(data_dir, 'archive_ledger')
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        archive_dir = os.path.join(base_dir, 'archive_ledger')
    output_path = os.path.join(archive_dir, 'superfluous_local_trades.csv')

    # Create the archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    try:
        # Format timestamp back to string for CSV consistency
        final_df = superfluous_trades_df.copy()
        if 'timestamp' in final_df.columns:
            # Sort by timestamp before writing
            final_df.sort_values(by='timestamp', inplace=True)
            final_df['timestamp'] = pd.to_datetime(final_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        final_df.to_csv(
            output_path,
            index=False,
            header=True,
            float_format='%.2f'
        )

        logger.info(f"Successfully wrote {len(final_df)} superfluous local trade(s) to '{output_path}'.")

    except IOError as e:
        logger.error(f"Error writing to superfluous trades CSV file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in write_superfluous_trades_to_csv: {e}")


def reconcile_trades(ib_trades_df: pd.DataFrame, local_trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares IB trades with the local ledger to identify discrepancies.

    This function performs a two-way comparison:
    1. It finds trades that exist in the IB report but are missing from the local ledger.
    2. It finds trades that exist in the local ledger but are not in the IB report.

    This enhanced version ignores `combo_id` and `position_id` and instead
    matches trades based on a combination of symbol, quantity, action, and
    a 2-second timestamp tolerance for robustness.

    Returns:
        A tuple containing two DataFrames:
        - missing_from_local_df: Trades in IB report, but not in local ledger.
        - superfluous_in_local_df: Trades in local ledger, but not in IB report.
    """
    if ib_trades_df.empty:
        logger.warning("IB trades report is empty. All local trades will be marked as superfluous.")
        return pd.DataFrame(), local_trades_df

    if local_trades_df.empty:
        logger.warning("Local ledger is empty. All IB trades will be marked as missing.")
        return ib_trades_df, pd.DataFrame()

    # Convert quantities to a consistent numeric type
    local_trades_df['quantity'] = pd.to_numeric(local_trades_df['quantity'], errors='coerce')
    ib_trades_df['quantity'] = pd.to_numeric(ib_trades_df['quantity'], errors='coerce')

    # Drop rows where quantity could not be parsed
    local_trades_df.dropna(subset=['quantity'], inplace=True)
    ib_trades_df.dropna(subset=['quantity'], inplace=True)

    # Use a copy to track which local trades have been matched
    local_trades_unmatched = local_trades_df.copy()
    missing_from_local = []

    for ib_index, ib_trade in ib_trades_df.iterrows():
        is_matched = False
        # Find potential matches based on symbol, action, and quantity
        potential_matches = local_trades_unmatched[
            (local_trades_unmatched['local_symbol'] == ib_trade['local_symbol']) &
            (local_trades_unmatched['action'] == ib_trade['action']) &
            (local_trades_unmatched['quantity'] == ib_trade['quantity'])
        ]

        if not potential_matches.empty:
            # Check for a timestamp match within the tolerance
            time_diff = (potential_matches['timestamp'] - ib_trade['timestamp']).abs()

            # Find the index of the first match within the 2-second window
            match_indices = potential_matches[time_diff <= pd.Timedelta(seconds=2)].index

            if len(match_indices) > 0:
                # If a match is found, remove it from the unmatched pool
                # to prevent it from being matched again.
                local_trades_unmatched.drop(match_indices[0], inplace=True)
                is_matched = True

        if not is_matched:
            missing_from_local.append(ib_trade)

    logger.info(f"Reconciliation complete. "
                f"Found {len(missing_from_local)} trade(s) missing from local ledger. "
                f"Found {len(local_trades_unmatched)} superfluous trade(s) in local ledger.")

    return pd.DataFrame(missing_from_local), local_trades_unmatched


def get_trade_ledger_df(data_dir: str = None) -> pd.DataFrame:
    """
    Reads and consolidates the main and archived trade ledgers into a single
    DataFrame for analysis.
    """
    if data_dir:
        ledger_path = os.path.join(data_dir, 'trade_ledger.csv')
        archive_dir = os.path.join(data_dir, 'archive_ledger')
    else:
        # Legacy: define paths relative to the script's location (project root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
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
        return pd.DataFrame(columns=['timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity', 'reason'])

    # --- Consolidate and Process ---
    full_ledger = pd.concat(dataframes, ignore_index=True)
    if 'timestamp' in full_ledger.columns:
        # Convert timestamp to datetime objects, assuming they are in UTC
        full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp']).dt.tz_localize('UTC')
    else:
        logger.error("Consolidated ledger is missing the 'timestamp' column.")
        return pd.DataFrame(columns=['timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity', 'reason'])

    logger.info(f"Consolidated a total of {len(full_ledger)} trades from all ledgers.")
    return full_ledger


async def fetch_flex_query_report(
    token: str,
    query_id: str,
    max_send_retries: int = 3,
    send_retry_delay: int = 30
) -> str | None:
    """
    Fetch Flex Query report with retry logic for both phases.

    v5.1 FIX: SendRequest phase now retries on transient errors (1004, 1018, 1019).
    These codes indicate IBKR maintenance or temporary unavailability.
    """
    # NOTE: httpx is already imported at module level — do NOT add a local import

    TRANSIENT_ERROR_CODES = {'1004', '1018', '1019'}
    
    # Define base_url for Phase 2 usage
    base_url = "https://gdcdyn.interactivebrokers.com/Universal/servlet/FlexStatementService."

    request_url = (
        f"{base_url}SendRequest?t={token}&q={query_id}&v=3"
    )

    reference_code = None

    async with httpx.AsyncClient() as client:

        # === PHASE 1: SendRequest (WITH RETRY) ===
        for attempt in range(1, max_send_retries + 1):
            try:
                resp = await client.get(request_url, timeout=30.0)
                root = ET.fromstring(resp.content)
                status = root.find('Status').text

                if status == 'Success':
                    reference_code = root.find('ReferenceCode').text
                    logger.info(f"Flex Query SendRequest succeeded (attempt {attempt})")
                    break

                error_code = root.find('ErrorCode').text
                error_msg = root.find('ErrorMessage').text

                if error_code in TRANSIENT_ERROR_CODES and attempt < max_send_retries:
                    wait_time = send_retry_delay * attempt
                    logger.warning(
                        f"Flex Query transient error (attempt {attempt}/{max_send_retries}): "
                        f"Code {error_code} — {error_msg}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"Flex Query SendRequest failed (attempt {attempt}): "
                        f"Code {error_code} — {error_msg}"
                    )
                    return None

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < max_send_retries:
                    wait_time = send_retry_delay * attempt
                    logger.warning(
                        f"Flex Query network error (attempt {attempt}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Flex Query SendRequest failed after {max_send_retries} attempts: {e}")
                    return None

        if not reference_code:
            logger.error("Flex Query SendRequest: no reference code obtained")
            return None

        # Mapping for legacy variable name in Phase 2
        ref_code = reference_code

        # === PHASE 2: GetStatement (EXISTING LOGIC — DO NOT MODIFY) ===
        # Keep the existing polling loop with 10 attempts exactly as-is.
        get_url = f"{base_url}GetStatement?t={token}&q={ref_code}&v=3"

        try:
            for i in range(10):  # Poll up to 10 times
                await asyncio.sleep(10)  # Wait 10 seconds between polls
                logger.info(f"Polling for report (Attempt {i+1}/10)...")
                
                resp = await client.get(get_url, timeout=30.0)
                
                # The report is ready when the response is not an XML error message
                if not resp.text.strip().startswith('<?xml'):
                    logger.info("Successfully downloaded Flex Query report.")
                    return resp.text
                
                # If it IS an XML, it's a "still processing" or error message
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


def parse_flex_csv_to_df(csv_data: str, config: dict = None) -> pd.DataFrame:
    """
    Parses the raw CSV string from the Flex Query into a clean DataFrame
    that matches the script's required format.

    This new version reads the CSV data directly, as the debug file shows
    a clean CSV without extra section headers.
    """
    if not csv_data or not csv_data.strip():
        logger.warning("Received empty CSV data from Flex Query.")
        return pd.DataFrame()

    try:
        # Use io.StringIO to treat the CSV string as a file
        df = pd.read_csv(io.StringIO(csv_data))
        logger.info(f"Parsed {len(df)} executions from the Flex Query report.")
        
    except pd.errors.EmptyDataError:
        logger.warning("Flex Query report was empty.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to read CSV data into pandas: {e}", exc_info=True)
        return pd.DataFrame()

    if df.empty:
        logger.warning("Flex Query report contained no data.")
        return pd.DataFrame()

    # --- Column Name Normalization ---
    # Standardize column names from different reports
    column_mappings = {
        'Price': 'TradePrice',
        'Date/Time': 'DateTime',
        'TradeID': 'TransactionID',
        'IBOrderID': 'SharedOrderID',
        'OrderID': 'SharedOrderID',
        'OrderReference': 'OrderReference'
    }
    df.rename(columns=column_mappings, inplace=True)
        
    # --- Commodity profile info for multiplier/cents detection ---
    from trading_bot.utils import get_active_ticker, CENTS_INDICATORS
    from config.commodity_profiles import get_commodity_profile
    _cfg = config or {}
    _ticker = get_active_ticker(_cfg)
    try:
        _profile = get_commodity_profile(_ticker)
        _default_multiplier = float(_profile.contract.contract_size)
        _is_cents_based = any(ind in _profile.contract.unit.lower() for ind in CENTS_INDICATORS)
    except Exception:
        _default_multiplier = 37500.0
        _is_cents_based = True  # Safe fallback (KC)

    # --- Data Cleaning and Type Conversion ---
    try:
        # Use the exact column names from your debug_report.csv
        df['TradePrice'] = pd.to_numeric(df['TradePrice'])
        df['Quantity'] = pd.to_numeric(df['Quantity'])
        df['Multiplier'] = pd.to_numeric(df['Multiplier'].replace('', str(_default_multiplier))).fillna(_default_multiplier)

        # Parse Strike: Convert empty to 0
        df['Strike'] = pd.to_numeric(df['Strike'].replace('', '0')).fillna(0)

        # Normalize Strike for cents-based contracts (e.g., KC cents/lb)
        # Ledger uses cents (e.g. 550.0 for 5.5), while IB report uses dollars (5.5).
        # We multiply by 100 to match the ledger convention.
        if _is_cents_based:
            df['Strike'] = df['Strike'] * 100

        # Parse the specific 'DateTime' format from your report: '20251106;122010'
        df['parsed_datetime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d;%H%M%S')
        
        # Convert from IBKR's timezone (assumed 'America/New_York') to UTC
        df['timestamp_utc'] = df['parsed_datetime'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')

    except KeyError as e:
        logger.error(f"Missing expected column in Flex Query report: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error during data type conversion: {e}", exc_info=True)
        return pd.DataFrame()

    # --- Map to script's internal column names ---
    df_out = pd.DataFrame()
    df_out['timestamp'] = df['timestamp_utc']

    # --- Generate Combo-Aware position_id ---
    # The 'Symbol' field from the report is already a good leg description.
    df['leg_description'] = df['Symbol'].astype(str)

    # --- Grouping Logic ---
    # Create a preliminary grouping key for fallbacks.
    df['prelim_group'] = df['Symbol'].str.split(' ').str[0] + '_' + df['DateTime']

    # For each preliminary group, create a canonical (sorted) key of all its legs.
    # This ensures that even if two different combos on the same underlying execute
    # at the exact same second, they get different grouping IDs.
    canonical_key = df.groupby('prelim_group')['leg_description'].transform(
        lambda legs: '-'.join(sorted(legs))
    )
    # The final fallback key is a combination of the preliminary group and the canonical key.
    fallback_id = df['prelim_group'] + '_' + canonical_key

    # Prioritize OrderReference, but use the robust fallback_id if it's missing.
    if 'OrderReference' in df.columns:
        df['grouping_id'] = df['OrderReference'].fillna(fallback_id)
    else:
        df['grouping_id'] = fallback_id

    # Use the grouping_id directly as position_id if it's available (which contains the UUID)
    # The previous logic of joining leg descriptions was overwriting the valid UUID.
    # We only use the fallback logic if OrderReference (and thus grouping_id) was missing/invalid.
    df_out['position_id'] = df['grouping_id']

    df_out['combo_id'] = df['TransactionID'].astype(str) # Keep transID for combo_id
    df_out['local_symbol'] = df['Symbol']
    df_out['action'] = df['Buy/Sell'] # Use the column directly
    df_out['quantity'] = df['Quantity'].abs() # Use absolute quantity for ledger
    df_out['avg_fill_price'] = df['TradePrice']
    df_out['strike'] = df['Strike'].replace(0, 'N/A').astype(str)
    df_out['right'] = df['Put/Call'].replace('', 'N/A')

    # --- Calculate Value (using your original script's logic) ---
    # A 'BUY' is a negative value (cash outflow)
    multiplier = df['Multiplier']
    # Normalize TradePrice to CENTS for cents-based contracts to match local ledger format.
    # Flex Query returns dollars (e.g. 0.41) while ledger uses cents (41.0).
    if _is_cents_based:
        df_out['avg_fill_price'] = df['TradePrice'] * 100
    else:
        df_out['avg_fill_price'] = df['TradePrice']

    # Calculate Value
    # If we use the normalized price (cents), we divide by 100.
    # Value = Price(Cents) * Quantity * Multiplier / 100
    # Or simply: Original Price(Dollars) * Quantity * Multiplier
    # We use the original price for calculation to be safe, but store the normalized price.
    total_value = (df['TradePrice'] * df_out['quantity'] * multiplier)
    
    # Apply negative sign for 'BUY' actions
    df_out['total_value_usd'] = total_value.where(df_out['action'] == 'SELL', -total_value)

    logger.info(f"Successfully processed {len(df_out)} trades into the reconciliation DataFrame.")
    return df_out


async def main(lookback_days: int = None, config: dict = None):
    """
    Main function to orchestrate the trade reconciliation process.
    Fetches reports from multiple Flex Queries, consolidates them, and then
    reconciles them against the local ledger.
    Returns dataframes of discrepancies.
    """
    logger.info("Starting trade reconciliation using Flex Queries.")

    # --- 1. Load Configuration ---
    if config is None:
        config = load_config()
    # Inject data_dir for commodity isolation if not already set
    if 'data_dir' not in config:
        ticker = os.environ.get("COMMODITY_TICKER", "KC").upper()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config['data_dir'] = os.path.join(base_dir, 'data', ticker)
    try:
        token = config['flex_query']['token']
        query_ids = config['flex_query']['query_ids']
    except KeyError:
        logger.critical("Config file is missing 'flex_query' section with 'token' and 'query_ids'.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 2. Fetch and Parse Trades from all configured IB Flex Queries ---
    all_ib_trades = []
    for query_id in query_ids:
        logger.info(f"Fetching report for Query ID: {query_id}")
        csv_data = await fetch_flex_query_report(token, query_id)
        if csv_data:
            ib_trades_df = parse_flex_csv_to_df(csv_data, config=config)
            if not ib_trades_df.empty:
                all_ib_trades.append(ib_trades_df)
        else:
            logger.warning(f"No data returned for Query ID: {query_id}")

    if not all_ib_trades:
        logger.warning("No trade data was fetched from any IB Flex Query. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    # --- 3. Consolidate and Deduplicate reports ---
    ib_trades_df = pd.concat(all_ib_trades, ignore_index=True)

    # Deduplicate based on a composite key of trade properties
    cols_to_check = ['timestamp', 'local_symbol', 'action', 'quantity', 'avg_fill_price', 'combo_id']
    ib_trades_df.drop_duplicates(subset=cols_to_check, keep='first', inplace=True)

    logger.info(f"Consolidated to {len(ib_trades_df)} unique trades from all reports.")

    # --- 3b. Filter to active commodity symbol ---
    # The IBKR Flex Query returns ALL account trades. Filter to only trades
    # matching the active commodity. IBKR uses different prefixes for futures
    # vs options: KC futures = "KC*", KC options = "KO*"; CC/"DC*"; SB/"SO*"
    ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
    _IBKR_SYMBOL_PREFIXES = {"KC": ("KC", "KO"), "CC": ("CC", "DC"), "SB": ("SB", "SO"), "NG": ("NG", "ON")}
    prefixes = _IBKR_SYMBOL_PREFIXES.get(ticker, (ticker,))
    pre_filter_count = len(ib_trades_df)
    ib_trades_df = ib_trades_df[ib_trades_df['local_symbol'].apply(lambda s: any(s.startswith(p) for p in prefixes))]
    if len(ib_trades_df) < pre_filter_count:
        logger.info(f"Filtered IB trades by commodity {ticker} (prefixes {prefixes}): {pre_filter_count} -> {len(ib_trades_df)}")

    # --- 4. Load Local Trade Ledger ---
    local_trades_df = get_trade_ledger_df(config.get('data_dir'))
    if local_trades_df.empty:
        logger.warning("Local trade ledger is empty. All fetched IB trades will be considered missing.")

    # --- 5. Filter trades to the last 33 days for comparison ---
    # v3.1: Configurable lookback with environment override
    if lookback_days is None:
        lookback_days = int(os.getenv('RECONCILIATION_LOOKBACK_DAYS', '90'))

    cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)

    # Ensure timestamp column is timezone-aware for comparison
    ib_trades_df['timestamp'] = pd.to_datetime(ib_trades_df['timestamp']).dt.tz_convert('UTC')

    initial_ib_count = len(ib_trades_df)
    ib_trades_df = ib_trades_df[ib_trades_df['timestamp'] >= cutoff_date]
    logger.info(f"Filtered IB trades from {initial_ib_count} to {len(ib_trades_df)} records within the last {lookback_days} days.")

    if not local_trades_df.empty:
        initial_local_count = len(local_trades_df)
        local_trades_df = local_trades_df[local_trades_df['timestamp'] >= cutoff_date]
        logger.info(f"Filtered local ledger from {initial_local_count} to {len(local_trades_df)} records within the last {lookback_days} days.")

    # --- 6. Reconcile Trades ---
    missing_trades_df, superfluous_trades_df = reconcile_trades(ib_trades_df, local_trades_df)

    if missing_trades_df.empty and superfluous_trades_df.empty:
        logger.info("No discrepancies found. The local ledger is perfectly in sync with the IB report.")
    else:
        # --- 7. Output Discrepancy Reports ---
        logger.info("Discrepancies found. Writing to output files.")
        write_missing_trades_to_csv(missing_trades_df, config.get('data_dir'))
        write_superfluous_trades_to_csv(superfluous_trades_df, config.get('data_dir'))

    # --- 6. Return the dataframes for the orchestrator ---
    return missing_trades_df, superfluous_trades_df


async def full_reconciliation(config: dict) -> int:
    """
    Run full historical reconciliation (monthly job).
    J3 FIX: Catches orphaned trades beyond daily window.
    """
    logger.info("Running FULL trade reconciliation (no date limit)")

    # Run main with 365 days lookback
    missing, superfluous = await main(lookback_days=365)

    count = len(missing) + len(superfluous)
    if count > 0:
        logger.warning(f"Full reconciliation found {count} discrepancies ({len(missing)} missing, {len(superfluous)} superfluous)")
    else:
        logger.info("Full reconciliation complete: all trades matched")

    return count


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Trade reconciler script manually terminated.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
