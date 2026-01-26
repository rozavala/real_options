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
            # === NEW: Give Gateway time to cleanup ===
            await asyncio.sleep(3.0)


async def _process_reconciliation(ib: IB, df: pd.DataFrame, config: dict, file_path: str):
    """Internal helper to process the DataFrame rows using the active IB connection."""

    # Ensure columns exist (if migrated recently, they should be there, but good to be safe)
    required_cols = ['exit_price', 'exit_timestamp', 'pnl_realized', 'actual_trend_direction', 'volatility_outcome']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Identify candidates for reconciliation
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        logger.error(f"Error parsing timestamps: {e}")
        return

    now = datetime.now()
    updates_made = False

    # Filter for rows that need processing
    for index, row in df.iterrows():
        # --- FORCE RECALCULATION CHECK ---
        # NEW: If it's a VOLATILITY trade with 0 P&L, process it anyway to fix the bug
        is_broken_volatility = (
            row.get('prediction_type') == 'VOLATILITY' and
            pd.notna(row.get('exit_price')) and
            row.get('exit_price') != '' and
            float(row.get('pnl_realized', 0) or 0) == 0.0
        )

        # Skip only if reconciled AND not a broken volatility row
        if pd.notna(row['exit_price']) and row['exit_price'] != '' and not is_broken_volatility:
            continue

        prediction_type = row.get('prediction_type', 'DIRECTIONAL')
        strategy_type = row.get('strategy_type', '')

        entry_time = row['timestamp']

        # Check age. If it's less than ~20 hours old, it might still be open.
        # We only want to grade "finished" tests.
        if (now - entry_time).total_seconds() < 20 * 3600 and not is_broken_volatility:
            continue

        contract_str = row['contract']  # e.g., "KC H4 (202403)" or just "KC H4"
        entry_price = float(row['entry_price']) if pd.notna(row['entry_price']) else 0.0
        decision = row['master_decision']

        if not contract_str or entry_price == 0:
            continue

        logger.info(f"Reconciling trade from {entry_time}: {contract_str} ({decision})")

        # Determine Target Exit Time (Default)
        target_exit_time = entry_time + timedelta(hours=27)

        try:
            exit_price = 0.0

            # 1. GET EXIT PRICE
            if is_broken_volatility:
                # Use existing data
                exit_price = float(row['exit_price'])
                # Try to use existing exit timestamp if valid
                if pd.notna(row.get('exit_timestamp')):
                    try:
                        target_exit_time = pd.to_datetime(row['exit_timestamp'])
                    except:
                        pass
                logger.info(f"Force-recalculating broken volatility row: {contract_str}")

            else:
                # FULL RECONCILIATION PROCESS

                # Parse Contract
                if '(' in contract_str and ')' in contract_str:
                    parts = contract_str.split('(')
                    symbol = parts[0].strip()
                    last_trade_date = parts[1].replace(')', '').strip()
                else:
                    logger.warning(f"Could not parse contract string: {contract_str}")
                    continue

                contract = Contract()
                contract.symbol = config.get('symbol', 'KC')
                contract.secType = 'FUT'
                contract.exchange = config.get('exchange', 'NYBOT')
                contract.currency = 'USD'
                contract.lastTradeDateOrContractMonth = last_trade_date
                contract.includeExpired = True

                # Qualify
                details = await ib.reqContractDetailsAsync(contract)
                if not details:
                    logger.error(f"Contract not found: {contract_str}")
                    continue
                qualified_contract = details[0].contract
                qualified_contract.includeExpired = True

                # If target exit time is in the future, we can't reconcile yet
                if target_exit_time > now:
                    continue

                # Fetch Historical Data
                end_str = (target_exit_time + timedelta(days=2)).strftime('%Y%m%d %H:%M:%S')
                bars = await ib.reqHistoricalDataAsync(
                    qualified_contract,
                    endDateTime=end_str,
                    durationStr='1 M',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )

                if not bars:
                    logger.warning(f"No historical data found for {contract_str}")
                    continue

                # Find bar
                target_date = target_exit_time.date()
                matched_bar = None
                for bar in bars:
                    if bar.date == target_date:
                        matched_bar = bar
                        break
                if not matched_bar:
                     entry_date = entry_time.date()
                     post_entry_bars = [b for b in bars if b.date > entry_date]
                     if post_entry_bars:
                         matched_bar = post_entry_bars[0]

                if not matched_bar:
                    logger.warning(f"Could not find a valid exit bar for {contract_str}")
                    continue

                exit_price = matched_bar.close


            # ================================================================
            # VOLATILITY-AWARE P&L CALCULATION (NET BASIS)
            # ================================================================

            # 1. Calculate percentage move
            pct_change = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
            abs_pct_change = abs(pct_change)

            # 2. Determine Trend Direction
            if exit_price > entry_price:
                trend = 'BULLISH'
            elif exit_price < entry_price:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'

            # 3. Strategy-Specific P&L (NET BASIS)
            # Thresholds calibrated to IV regime (~31-35%)
            # Daily Vol ≈ 35% / 16 = 2.2%, Breakeven ≈ 0.8 × 2.2% = 1.76% → 1.8%
            STRADDLE_THRESHOLD = 0.018  # 1.8%
            CONDOR_THRESHOLD = 0.015    # 1.5%

            pnl = 0.0
            vol_outcome = None

            if prediction_type == 'VOLATILITY':

                if strategy_type == 'LONG_STRADDLE':
                    if abs_pct_change >= STRADDLE_THRESHOLD:
                        # WIN: Profit is the move BEYOND the breakeven (premium cost)
                        # Net profit = (actual_move - breakeven) × entry_price
                        net_move = abs_pct_change - STRADDLE_THRESHOLD
                        pnl = net_move * entry_price
                        vol_outcome = 'BIG_MOVE'
                        logger.info(f"STRADDLE WIN: {abs_pct_change:.2%} move exceeds {STRADDLE_THRESHOLD:.2%} threshold. Net P&L: {pnl:.2f}")
                    else:
                        # LOSS: Lost the premium (approximately the threshold value)
                        # Max loss is capped at premium paid
                        pnl = -1 * (STRADDLE_THRESHOLD * entry_price)
                        vol_outcome = 'STAYED_FLAT'
                        logger.info(f"STRADDLE LOSS: {abs_pct_change:.2%} move below {STRADDLE_THRESHOLD:.2%} threshold. P&L: {pnl:.2f}")

                elif strategy_type == 'IRON_CONDOR':
                    if abs_pct_change <= CONDOR_THRESHOLD:
                        # WIN: Keep the premium (approximately 1% of notional)
                        pnl = entry_price * 0.01
                        vol_outcome = 'STAYED_FLAT'
                        logger.info(f"CONDOR WIN: {abs_pct_change:.2%} move within {CONDOR_THRESHOLD:.2%} threshold. P&L: {pnl:.2f}")
                    else:
                        # LOSS: Move exceeded wings, loss proportional to excess
                        net_move = abs_pct_change - CONDOR_THRESHOLD
                        pnl = -1 * (net_move * entry_price)
                        vol_outcome = 'BIG_MOVE'
                        logger.info(f"CONDOR LOSS: {abs_pct_change:.2%} move exceeds {CONDOR_THRESHOLD:.2%} threshold. P&L: {pnl:.2f}")

                # Commit volatility outcome
                df.at[index, 'volatility_outcome'] = vol_outcome

            else:
                # Standard Directional Logic (unchanged)
                if decision == 'BULLISH':
                    pnl = exit_price - entry_price
                elif decision == 'BEARISH':
                    pnl = entry_price - exit_price
                # NEUTRAL directional = no position, pnl stays 0

            # 4. Commit Results
            df.at[index, 'exit_price'] = exit_price
            df.at[index, 'exit_timestamp'] = target_exit_time.strftime('%Y-%m-%d %H:%M:%S')
            df.at[index, 'pnl_realized'] = round(pnl, 4)
            df.at[index, 'actual_trend_direction'] = trend

            updates_made = True
            logger.info(
                f"Reconciled {contract_str}: Entry {entry_price:.2f} -> Exit {exit_price:.2f} "
                f"({abs_pct_change:.2%} move) | Strategy: {strategy_type or 'DIRECTIONAL'} | "
                f"Outcome: {vol_outcome or trend} | P&L: {pnl:.4f}"
            )

            # Update Brier Score
            try:
                tracker = get_brier_tracker()

                if prediction_type == 'VOLATILITY':
                    # Record Volatility Prediction
                    tracker.record_volatility_prediction(
                        strategy_type=strategy_type,
                        predicted_vol_level='HIGH' if strategy_type == 'LONG_STRADDLE' else 'LOW', # implied
                        actual_outcome=vol_outcome or 'UNKNOWN',
                        timestamp=target_exit_time
                    )

                else:
                    # Standard directional recording
                    tracker.record_prediction(
                        agent='master_decision',
                        predicted=decision,
                        actual=trend,
                        timestamp=target_exit_time
                    )
                    logger.info(f"Recorded Brier for master: Predicted {decision} vs Actual {trend}")

                    # Also record ML Model if available
                    ml_signal_direction = row.get('ml_signal', 'NEUTRAL')
                    if ml_signal_direction and ml_signal_direction != 'NEUTRAL':
                        ml_normalized = ml_signal_direction.upper()
                        if ml_normalized == 'LONG':
                            ml_normalized = 'BULLISH'
                        elif ml_normalized == 'SHORT':
                            ml_normalized = 'BEARISH'

                        tracker.record_prediction(
                            agent='ml_model',
                            predicted=ml_normalized,
                            actual=trend,
                            timestamp=target_exit_time
                        )

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

            # === RESOLVE INDIVIDUAL AGENT PREDICTIONS ===
            # This closes the feedback loop by scoring each agent's prediction
            try:
                from trading_bot.brier_scoring import resolve_pending_predictions
                resolved_count = resolve_pending_predictions(file_path)
                if resolved_count > 0:
                    logger.info(f"Feedback Loop: Resolved {resolved_count} individual agent predictions")
            except Exception as resolve_e:
                logger.error(f"Failed to resolve agent predictions: {resolve_e}")

        except Exception as e:
            logger.error(f"Failed to save updated CSV: {e}")
    else:
        logger.info("No rows required updates.")
