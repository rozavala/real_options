"""
Module for reconciling historical Council decisions with actual market outcomes.
"""
import asyncio
import csv
import logging
import os
import random
from datetime import datetime, timedelta, timezone
import pytz
import pandas as pd
from ib_insync import IB, Contract, util
from trading_bot.brier_scoring import get_brier_tracker
from trading_bot.timestamps import parse_ts_column, parse_ts_single

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


def _calculate_actual_exit_time(entry_time: datetime, config: dict) -> datetime:
    """
    Calculate the actual exit time for a council decision, respecting
    the weekly close policy (close all positions before weekends/holidays).

    This is commodity-agnostic — it uses the exchange calendar rather than
    hardcoded dates. The function answers: "Given this entry time, when would
    the position actually have been closed?"

    Rules (matching close_stale_positions logic):
    1. Default exit: entry + 27 hours (next session close)
    2. Friday entries: exit at CLOSE_TIME on same Friday (weekly close)
    3. Thursday entries when Friday is holiday: exit at CLOSE_TIME Thursday
    4. Any entry where default exit lands on non-trading day:
       exit at CLOSE_TIME on the last trading day before the weekend/holiday

    Args:
        entry_time: Timezone-aware datetime of the council decision
        config: System config (unused now, available for future exchange config)

    Returns:
        Timezone-aware datetime of the actual exit
    """
    import pytz
    from pandas.tseries.holiday import USFederalHolidayCalendar
    import holidays as holidays_lib

    ny_tz = pytz.timezone('America/New_York')

    # Position close time from schedule (11:00 ET = primary close_stale_positions)
    CLOSE_HOUR = 11
    CLOSE_MINUTE = 0

    # Convert entry to NY time for calendar logic
    if entry_time.tzinfo is None:
        entry_time = pytz.UTC.localize(entry_time)
    entry_ny = entry_time.astimezone(ny_tz)

    # Default: entry + 27 hours
    default_exit = entry_time + timedelta(hours=27)
    default_exit_ny = default_exit.astimezone(ny_tz)

    # Build holiday set for the relevant year(s)
    years_to_check = {entry_ny.year, default_exit_ny.year}
    us_holidays = set()
    for year in years_to_check:
        us_holidays.update(holidays_lib.US(years=year, observed=True).keys())
        try:
            nyse = holidays_lib.financial_holidays('NYSE', years=year)
            us_holidays.update(nyse.keys())
        except (AttributeError, TypeError):
            pass

    def is_trading_day(d):
        """Check if a date is a trading day (weekday + not holiday)."""
        return d.weekday() < 5 and d not in us_holidays

    def last_trading_day_close(from_date):
        """Roll backward from from_date to find the last trading day's close."""
        check_date = from_date
        while not is_trading_day(check_date):
            check_date -= timedelta(days=1)
        return ny_tz.localize(
            datetime.combine(check_date, datetime.min.time()).replace(
                hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
            )
        ).astimezone(pytz.UTC)

    # --- RULE: Check if entry day itself triggers weekly close ---
    entry_date = entry_ny.date()
    entry_weekday = entry_date.weekday()  # 0=Mon, 4=Fri

    # Friday entry → weekly close same day
    if entry_weekday == 4:
        friday_close_ny = entry_ny.replace(
            hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
        )
        friday_close_utc = friday_close_ny.astimezone(pytz.UTC)
        # Only use Friday close if entry is BEFORE close time
        if entry_time < friday_close_utc:
            return friday_close_utc

    # Thursday entry when Friday is a holiday → weekly close same day
    if entry_weekday == 3:
        friday_date = entry_date + timedelta(days=1)
        if friday_date in us_holidays or friday_date.weekday() >= 5:
            thursday_close_ny = entry_ny.replace(
                hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
            )
            thursday_close_utc = thursday_close_ny.astimezone(pytz.UTC)
            if entry_time < thursday_close_utc:
                return thursday_close_utc

    # --- RULE: Check if default exit (entry+27h) lands on non-trading period ---
    default_exit_date = default_exit_ny.date()

    if not is_trading_day(default_exit_date):
        # Roll backward to the last trading day's close
        return last_trading_day_close(default_exit_date)

    # --- RULE: Check if default exit is after close time on a weekly-close day ---
    default_exit_weekday = default_exit_date.weekday()

    if default_exit_weekday == 4:  # Exits on a Friday
        friday_close_ny = default_exit_ny.replace(
            hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
        )
        friday_close_utc = friday_close_ny.astimezone(pytz.UTC)
        # Cap at Friday close (won't be held over the weekend)
        if default_exit > friday_close_utc:
            return friday_close_utc

    # Default case: use entry + 27h (standard weekday exit)
    return default_exit


async def _process_reconciliation(ib: IB, df: pd.DataFrame, config: dict, file_path: str):
    """Internal helper to process the DataFrame rows using the active IB connection."""
    # Ensure pandas is available (module-level import)
    assert pd is not None, "pandas not imported at module level"

    # Ensure columns exist (if migrated recently, they should be there, but good to be safe)
    required_cols = ['exit_price', 'exit_timestamp', 'pnl_realized', 'actual_trend_direction', 'volatility_outcome']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Identify candidates for reconciliation
    try:
        df['timestamp'] = parse_ts_column(df['timestamp'])
    except Exception as e:
        logger.error(f"Error parsing timestamps: {e}")
        return

    now = datetime.now(timezone.utc)
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

        # Determine Target Exit Time (respects weekly close policy)
        target_exit_time = _calculate_actual_exit_time(entry_time, config)

        # Log if exit time was adjusted from default
        default_exit = entry_time + timedelta(hours=27)
        if abs((target_exit_time - default_exit).total_seconds()) > 3600:
            logger.info(
                f"Exit time adjusted for {contract_str}: "
                f"default {default_exit.strftime('%a %H:%M UTC')} → "
                f"actual {target_exit_time.strftime('%a %H:%M UTC')} "
                f"(weekly close policy)"
            )

        try:
            exit_price = 0.0

            # 1. GET EXIT PRICE
            if is_broken_volatility:
                # Use existing data
                exit_price = float(row['exit_price'])
                # Try to use existing exit timestamp if valid
                if pd.notna(row.get('exit_timestamp')):
                    parsed_exit = parse_ts_single(str(row['exit_timestamp']))
                    if parsed_exit:
                        target_exit_time = parsed_exit
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
                # Use IB's preferred UTC format to suppress Warning 2174
                end_str = (target_exit_time + timedelta(days=2)).strftime('%Y%m%d-%H:%M:%S') + ' UTC'
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

                # ================================================================
                # BAR MATCHING — BACKWARD-ROLLING (Weekly Close Aware)
                # ================================================================
                # Since positions are closed before weekends/holidays,
                # target_exit_time is always on a trading day. But we still
                # need robust fallback for edge cases and data gaps.
                #
                # Strategy (commodity-agnostic — adapts to any trading calendar):
                # 1. Exact target date (ideal — should always work now)
                # 2. Last available bar ON or BEFORE target (backward roll)
                # 3. First bar AFTER entry date (minimum viable)
                # 4. Skip if no valid bar (data not yet available)
                # ================================================================
                target_date = target_exit_time.date()
                entry_date = entry_time.date()
                matched_bar = None

                # Sort bars by date for reliable matching
                sorted_bars = sorted(bars, key=lambda b: b.date)

                if not sorted_bars:
                    logger.warning(f"No historical bars returned for {contract_str}")
                    continue

                # Strategy 1: Exact target date (should work for all properly
                # calculated exit times since _calculate_actual_exit_time
                # always returns a trading day)
                for bar in sorted_bars:
                    if bar.date == target_date:
                        matched_bar = bar
                        break

                # Strategy 2: Last available bar on or before target date
                # (backward roll — correct for weekly close policy)
                if not matched_bar:
                    on_or_before = [b for b in sorted_bars if b.date <= target_date]
                    if on_or_before:
                        matched_bar = on_or_before[-1]  # last (most recent) bar
                        logger.info(
                            f"Bar date adjusted for {contract_str}: "
                            f"target {target_date} → actual {matched_bar.date} "
                            f"(backward roll to last trading session)"
                        )

                # Strategy 3: First bar after entry (minimum viable — at least
                # captures some directional movement)
                if not matched_bar:
                    post_entry = [b for b in sorted_bars if b.date > entry_date]
                    if post_entry:
                        matched_bar = post_entry[0]
                        logger.info(
                            f"Using first post-entry bar for {contract_str}: "
                            f"{matched_bar.date} (fallback)"
                        )

                # Strategy 4: Data not yet available
                if not matched_bar:
                    last_bar_date = sorted_bars[-1].date if sorted_bars else None
                    if last_bar_date and last_bar_date < target_date:
                        logger.info(
                            f"Exit bar for {contract_str} not yet available "
                            f"(target: {target_date}, latest bar: {last_bar_date}). "
                            f"Will retry next reconciliation run."
                        )
                    else:
                        logger.warning(
                            f"Could not find valid exit bar for {contract_str} "
                            f"despite bars through {last_bar_date}"
                        )
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
            # PHASE 1: Legacy CSV resolution (backward compat)
            try:
                from trading_bot.brier_scoring import resolve_pending_predictions
                resolved_count = resolve_pending_predictions(file_path)
                if resolved_count > 0:
                    logger.info(f"Feedback Loop (Legacy): Resolved {resolved_count} agent predictions")
            except Exception as resolve_e:
                logger.error(f"Failed to resolve legacy agent predictions: {resolve_e}")

            # PHASE 2: Enhanced probabilistic resolution
            try:
                from trading_bot.brier_bridge import resolve_agent_prediction, reset_enhanced_tracker
                # Removed inner import pandas as pd to fix scoping issue

                # Read the just-resolved structured predictions to find what was resolved
                structured_file = "data/agent_accuracy_structured.csv"
                if os.path.exists(structured_file):
                    struct_df = pd.read_csv(structured_file)

                    # Find recently resolved predictions (not PENDING, not ORPHANED)
                    resolved_mask = (
                        (struct_df['actual'] != 'PENDING') &
                        (struct_df['actual'] != 'ORPHANED')
                    )

                    resolved_df = struct_df[resolved_mask]

                    enhanced_resolved = 0
                    for _, row in resolved_df.iterrows():
                        cycle_id = str(row.get('cycle_id', ''))
                        agent = str(row.get('agent', ''))
                        actual = str(row.get('actual', ''))

                        if not agent or not actual or actual in ('PENDING', 'ORPHANED'):
                            continue

                        brier = resolve_agent_prediction(
                            agent=agent,
                            actual_outcome=actual,
                            cycle_id=cycle_id,
                        )
                        if brier is not None:
                            enhanced_resolved += 1

                    if enhanced_resolved > 0:
                        logger.info(f"Feedback Loop (Enhanced): Scored {enhanced_resolved} predictions with Brier scores")
                        reset_enhanced_tracker()  # Reset singleton so voting picks up new scores

            except Exception as enhanced_e:
                # Enhanced resolution failure MUST NOT block reconciliation
                logger.warning(f"Enhanced Brier resolution failed (non-critical): {enhanced_e}")

        except Exception as e:
            logger.error(f"Failed to save updated CSV: {e}")
    else:
        logger.info("No rows required updates.")
