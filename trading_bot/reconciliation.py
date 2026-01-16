"""
Module for reconciling historical Council decisions with actual market outcomes.
"""
import asyncio
import csv
import logging
import os
import random
from datetime import datetime, timedelta
import pytz
import pandas as pd
from ib_insync import IB, Contract, util
from trading_bot.brier_scoring import get_brier_tracker

logger = logging.getLogger(__name__)

async def reconcile_council_history(config: dict, ib: IB = None):
    """
    Reconciles the Council History CSV by backfilling missing exit prices and outcomes.

    Logic:
    1. Loads 'data/council_history.csv'.
    2. Identifies rows where 'exit_price' is missing and enough time has passed (approx 27h).
    3. Connects to IB (if not provided) to fetch historical prices for those contracts.
    4. Calculates realized P&L (theoretical) and actual trend direction.
    5. Updates the CSV.
    """
    logger.info("--- Starting Council History Reconciliation ---")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'council_history.csv')

    if not os.path.exists(file_path):
        logger.warning("No council_history.csv found. Skipping reconciliation.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read council_history.csv: {e}")
        return

    if df.empty:
        logger.info("Council history is empty.")
        return

    # --- SELF-MANAGED IB CONNECTION ---
    managed_connection = False
    if ib is None:
        ib = IB()
        managed_connection = True
        try:
             host = config['connection']['host']
             port = config['connection']['port']
             # Use random clientId to avoid conflicts
             client_id = random.randint(5000, 9999)
             await ib.connectAsync(host, port, clientId=client_id)
        except Exception as e:
             logger.error(f"Failed to connect to IB for reconciliation: {e}")
             return

    try:
        # Proceed with reconciliation using 'ib'
        await _process_reconciliation(ib, df, config, file_path)
    except Exception as e:
        logger.error(f"Error during reconciliation process: {e}")
    finally:
        if managed_connection and ib.isConnected():
            ib.disconnect()


async def _process_reconciliation(ib: IB, df: pd.DataFrame, config: dict, file_path: str):
    """Internal helper to process the DataFrame rows using the active IB connection."""

    # Ensure columns exist (if migrated recently, they should be there, but good to be safe)
    required_cols = ['exit_price', 'exit_timestamp', 'pnl_realized', 'actual_trend_direction']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Identify candidates for reconciliation
    # Criteria:
    # 1. exit_price is NaN or empty
    # 2. timestamp is older than X hours (e.g., 20h to be safe for next-day close)
    # The user mentioned "around 27h generally". We check if current time > entry time + 20h to catch the next closing.
    # Actually, we want the price at the *holding period end*.
    # If the trade is older than 27h, we can definitely reconcile.

    # Convert timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f"Error parsing timestamps: {e}")
        return

    now = datetime.now()
    updates_made = False

    # Filter for rows that need processing
    # We iterate manually to handle async calls and potential failures gracefully per row
    for index, row in df.iterrows():
        # Check if already reconciled
        if pd.notna(row['exit_price']) and row['exit_price'] != '':
            continue

        entry_time = row['timestamp']
        # If entry is naive, assume local/system time which matches 'now'.
        # Safest is to treat everything as naive or aware consistently.
        # df['timestamp'] usually comes from datetime.now() stringified.

        # Check age. If it's less than ~20 hours old, it might still be open.
        # We only want to grade "finished" tests.
        if (now - entry_time).total_seconds() < 20 * 3600:
            continue

        contract_str = row['contract']  # e.g., "KC H4 (202403)" or just "KC H4"
        entry_price = float(row['entry_price']) if pd.notna(row['entry_price']) else 0.0
        decision = row['master_decision']

        if not contract_str or entry_price == 0:
            continue

        logger.info(f"Reconciling trade from {entry_time}: {contract_str} ({decision})")

        # 1. Parse Contract
        # Expected format: "Symbol (YYYYMM)" or similar from signal_generator.py
        # signal_generator.py: f"{contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]})"
        try:
            # Regex or split to get symbol and date
            # Example: "KCH5 (202503)"
            if '(' in contract_str and ')' in contract_str:
                parts = contract_str.split('(')
                symbol = parts[0].strip()
                last_trade_date = parts[1].replace(')', '').strip()
            else:
                # Fallback or skip
                logger.warning(f"Could not parse contract string: {contract_str}")
                continue

            contract = Contract()
            contract.symbol = config.get('symbol', 'KC') # 'KC'
            contract.secType = 'FUT'
            contract.exchange = config.get('exchange', 'NYBOT') # 'NYBOT'
            contract.currency = 'USD'
            contract.lastTradeDateOrContractMonth = last_trade_date
            contract.includeExpired = True

            # Qualify to be sure
            details = await ib.reqContractDetailsAsync(contract)
            if not details:
                logger.error(f"Contract not found: {contract_str}")
                continue

            qualified_contract = details[0].contract
            qualified_contract.includeExpired = True

            # 2. Determine Exit Time
            # Logic: Entry Time + ~27 hours (Next Day Close)
            # We want the closing price of the day AFTER entry.
            # If entry is Monday 14:00, exit is Tuesday 17:00.
            # We can request Historical Data around that target time.

            target_exit_time = entry_time + timedelta(hours=27)

            # If target exit time is in the future, we can't reconcile yet (should have been caught by age check)
            if target_exit_time > now:
                continue

            # 3. Fetch Historical Data
            # We ask for TRADES for a short window around target_exit_time
            # Or simpler: Ask for daily bars covering the entry and next few days.
            end_str = (target_exit_time + timedelta(days=2)).strftime('%Y%m%d %H:%M:%S')

            bars = await ib.reqHistoricalDataAsync(
                qualified_contract,
                endDateTime=end_str,
                durationStr='1 M', # 1 Month to be safe and cover the range
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )

            if not bars:
                logger.warning(f"No historical data found for {contract_str}")
                continue

            # Find the bar that corresponds to the "Exit Day"
            # The exit day is the day of target_exit_time.
            target_date = target_exit_time.date()

            matched_bar = None
            for bar in bars:
                # bar.date is usually datetime.date for daily bars
                if bar.date == target_date:
                    matched_bar = bar
                    break

            # If exact match failed (maybe holiday?), try next available bar?
            # Or just take the bar immediately after entry_time?
            if not matched_bar:
                 # Find first bar after entry_time
                 entry_date = entry_time.date()
                 post_entry_bars = [b for b in bars if b.date > entry_date]
                 if post_entry_bars:
                     matched_bar = post_entry_bars[0] # The next trading day

            if not matched_bar:
                logger.warning(f"Could not find a valid exit bar for {contract_str} after {entry_time}")
                continue

            exit_price = matched_bar.close

            # 4. Calculate Results
            pnl = 0.0
            trend = "NEUTRAL"

            if decision == 'BULLISH':
                pnl = exit_price - entry_price
            elif decision == 'BEARISH':
                pnl = entry_price - exit_price

            if exit_price > entry_price:
                trend = 'BULLISH'
            elif exit_price < entry_price:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'

            # Update DataFrame
            df.at[index, 'exit_price'] = exit_price
            df.at[index, 'exit_timestamp'] = target_exit_time.strftime('%Y-%m-%d %H:%M:%S') # Approx
            df.at[index, 'pnl_realized'] = round(pnl, 4)
            df.at[index, 'actual_trend_direction'] = trend

            updates_made = True
            logger.info(f"Reconciled {contract_str}: Entry {entry_price} -> Exit {exit_price} ({trend}) | P&L: {pnl}")

            # Update Brier Score
            try:
                # Actual trend is BULLISH or BEARISH or NEUTRAL
                # Master Decision is BULLISH or BEARISH or NEUTRAL
                # We record the Master's prediction
                tracker = get_brier_tracker()

                # Record Master Strategist prediction
                # Note: trend logic above: if exit > entry => BULLISH.
                # 'actual_trend_direction' holds this.
                tracker.record_prediction(
                    agent='master_decision', # Using key from score card
                    predicted=decision,
                    actual=trend,
                    timestamp=target_exit_time
                )
                logger.info(f"Recorded Brier score for master: Predicted {decision} vs Actual {trend}")

            except Exception as brier_e:
                logger.error(f"Failed to record Brier score: {brier_e}")

            # Respect rate limits slightly
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            continue

    if updates_made:
        try:
            # Use columns order from utils.py to keep it clean
            df.to_csv(file_path, index=False)
            logger.info("Successfully updated council_history.csv with reconciliation results.")
        except Exception as e:
            logger.error(f"Failed to save updated CSV: {e}")
    else:
        logger.info("No rows required updates.")
