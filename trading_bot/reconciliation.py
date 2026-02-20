"""
Module for reconciling historical Council decisions with actual market outcomes.
"""
import asyncio
import csv
import logging
import os
import random
from datetime import datetime, timedelta, timezone, date
import pytz
import pandas as pd
from ib_insync import IB, Contract, util
from trading_bot.brier_scoring import get_brier_tracker
from trading_bot.timestamps import parse_ts_column, parse_ts_single, format_ib_datetime
from trading_bot.trade_journal import TradeJournal
from trading_bot.tms import TransactiveMemory
from trading_bot.heterogeneous_router import get_router

logger = logging.getLogger(__name__)

def store_reflexion_lesson(agent_name: str, contract: str,
                            predicted: str, actual: str, reasoning: str):
    """Store a lesson when an agent's prediction was wrong."""
    try:
        # TransactiveMemory imported at module level
        tms = TransactiveMemory()

        lesson = (
            f"On {contract}, predicted {predicted} but actual was {actual}. "
            f"Original reasoning: {reasoning[:200]}. "
            f"Lesson: This prediction was incorrect — review the reasoning for bias."
        )

        tms.encode(agent_name, lesson, {
            "agent": agent_name,
            "contract": contract,
            "predicted": predicted,
            "actual": actual,
            "type": "reflexion_lesson",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"Stored reflexion lesson for {agent_name} on {contract}")

    except Exception as e:
        logger.warning(f"Failed to store reflexion lesson: {e}")

async def reconcile_council_history(config: dict, ib: IB = None):
    """
    Reconciles the Council History CSV by backfilling missing exit prices and outcomes.

    Logic:
    1. Loads 'data/{ticker}/council_history.csv'.
    2. Identifies rows where 'exit_price' is missing and enough time has passed (approx 27h).
    3. Connects to IB (if not provided) to fetch historical prices for those contracts.
    4. Calculates realized P&L (theoretical) and actual trend direction.
    5. Updates the CSV.
    """
    logger.info("--- Starting Council History Reconciliation ---")

    data_dir = config.get('data_dir')
    if data_dir:
        file_path = os.path.join(data_dir, 'council_history.csv')
    else:
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
             try:
                 ib.disconnect()
             except Exception:
                 pass
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
    Calculate when a position entered at entry_time would actually be closed.

    v5.4 REWRITE: Deterministic calculation that mirrors close_stale_positions behavior.

    Algorithm:
    1. Get the effective close time from config (accounts for schedule offset)
    2. If entry is before close time on a weekly-close day (Friday, or Thursday
       before a holiday), exit is same-day at close time
    3. Otherwise, walk forward to the next trading day and exit at close time
    4. If that next trading day is a weekly-close day, cap at its close time

    This replaces the old entry+27h heuristic which didn't match actual execution.

    Commodity-agnostic: uses exchange calendar and config-driven close time.
    """
    import pytz
    from trading_bot.calendars import get_exchange_calendar
    from trading_bot.utils import get_effective_close_time

    ny_tz = pytz.timezone('America/New_York')
    CLOSE_HOUR, CLOSE_MINUTE = get_effective_close_time(config)

    # Convert entry to NY time
    if entry_time.tzinfo is None:
        entry_time = pytz.UTC.localize(entry_time)
    entry_ny = entry_time.astimezone(ny_tz)

    # Build exchange holiday set
    cal = get_exchange_calendar(config.get('exchange', 'ICE'))
    entry_year = entry_ny.year
    exchange_holidays = set()
    for year in {entry_year, entry_year + 1}:
        try:
            hols = cal.holidays(
                start=date(year, 1, 1),
                end=date(year, 12, 31)
            )
            exchange_holidays.update(d.date() for d in hols)
        except Exception:
            pass

    def is_trading_day(d):
        """Check if a date is a trading day (weekday + not exchange holiday)."""
        return d.weekday() < 5 and d not in exchange_holidays

    def make_close_time(d):
        """Create a UTC datetime for close time on a given date."""
        close_ny = ny_tz.localize(
            datetime.combine(d, datetime.min.time()).replace(
                hour=CLOSE_HOUR, minute=CLOSE_MINUTE, second=0, microsecond=0
            )
        )
        return close_ny.astimezone(pytz.UTC)

    def is_weekly_close_day(d):
        """Check if this date triggers a weekly close (Friday, or Thu before holiday Friday)."""
        if d.weekday() == 4:  # Friday
            return True
        if d.weekday() == 3:  # Thursday
            friday = d + timedelta(days=1)
            if friday in exchange_holidays or friday.weekday() >= 5:
                return True
        return False

    entry_date = entry_ny.date()

    # RULE 1: If entry is on a weekly-close day AND before close time → exit same day
    if is_weekly_close_day(entry_date):
        same_day_close = make_close_time(entry_date)
        if entry_time < same_day_close:
            return same_day_close

    # RULE 2: Walk forward to next trading day
    next_day = entry_date + timedelta(days=1)
    safety_limit = 10  # Prevent infinite loop (max gap: ~4 days for holiday weekends)
    for _ in range(safety_limit):
        if is_trading_day(next_day):
            break
        next_day += timedelta(days=1)

    # RULE 3: Exit at close time on the next trading day
    # If that day is a weekly-close day, the close time already accounts for it
    return make_close_time(next_day)


async def _process_reconciliation(ib: IB, df: pd.DataFrame, config: dict, file_path: str):
    """Internal helper to process the DataFrame rows using the active IB connection."""
    # Ensure pandas is available (module-level import)
    assert pd is not None, "pandas not imported at module level"

    # Ensure columns exist (if migrated recently, they should be there, but good to be safe)
    required_cols = ['exit_price', 'exit_timestamp', 'pnl_realized', 'actual_trend_direction', 'volatility_outcome']
    str_cols = {'exit_timestamp', 'actual_trend_direction', 'volatility_outcome'}
    for col in required_cols:
        if col not in df.columns:
            # Use object dtype for string columns to avoid FutureWarning on mixed-type assignment
            df[col] = pd.Series([None] * len(df), dtype='object' if col in str_cols else 'float64')

    # Identify candidates for reconciliation
    try:
        df['timestamp'] = parse_ts_column(df['timestamp'])
    except Exception as e:
        logger.error(f"Error parsing timestamps: {e}")
        return

    now = datetime.now(timezone.utc)
    updates_made = False

    # Initialize components for journaling
    tms = TransactiveMemory()
    router = None
    try:
        router = get_router(config)
    except Exception as e:
        logger.warning(f"Router initialization failed, journal will use TMS only: {e}")

    journal = TradeJournal(config, tms=tms, router=router)

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
        if pd.notna(row.get('exit_price')) and row.get('exit_price') != '' and not is_broken_volatility:
            continue

        prediction_type = row.get('prediction_type', 'DIRECTIONAL')
        strategy_type = row.get('strategy_type', '')

        entry_time = row.get('timestamp')

        # Check age. If it's less than ~20 hours old, it might still be open.
        # We only want to grade "finished" tests.
        if (now - entry_time).total_seconds() < 20 * 3600 and not is_broken_volatility:
            continue

        contract_str = row.get('contract', '')  # e.g., "KC H4 (202403)" or just "KC H4"
        entry_price = float(row.get('entry_price', 0)) if pd.notna(row.get('entry_price')) else 0.0
        decision = row.get('master_decision', '')
        missing_entry_price = (entry_price == 0)

        if not contract_str:
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
                exit_price = float(row.get('exit_price', 0))
                # Try to use existing exit timestamp if valid
                if pd.notna(row.get('exit_timestamp')):
                    parsed_exit = parse_ts_single(str(row.get('exit_timestamp', '')))
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
                end_str = format_ib_datetime(target_exit_time + timedelta(days=2))
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

                # If entry_price was missing, derive it from entry-day bar
                if missing_entry_price:
                    entry_date_obj = entry_time.date()
                    entry_bar = next((b for b in sorted_bars if b.date == entry_date_obj), None)
                    if not entry_bar:
                        # Fallback: last bar on or before entry date
                        on_or_before = [b for b in sorted_bars if b.date <= entry_date_obj]
                        entry_bar = on_or_before[-1] if on_or_before else None
                    if entry_bar:
                        entry_price = entry_bar.close
                        df.at[index, 'entry_price'] = entry_price
                        logger.info(f"Backfilled entry_price for {contract_str}: {entry_price:.2f} (from {entry_bar.date} bar)")
                    else:
                        logger.warning(f"Cannot derive entry_price for {contract_str} — no entry-day bar")
                        continue


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

            # --- Reflexion: Store lessons for incorrect agents ---
            if trend in ['BULLISH', 'BEARISH']: # Only learn from directional outcomes
                agent_cols = [c for c in df.columns if c.endswith('_sentiment')]
                for col in agent_cols:
                    agent_name = col.replace('_sentiment', '')
                    sentiment = row.get(col, 'NEUTRAL')
                    summary_col = f"{agent_name}_summary"
                    reasoning = row.get(summary_col, 'No summary')

                    if sentiment in ['BULLISH', 'BEARISH'] and sentiment != trend:
                        store_reflexion_lesson(
                            agent_name=agent_name,
                            contract=contract_str,
                            predicted=sentiment,
                            actual=trend,
                            reasoning=str(reasoning)
                        )

            updates_made = True
            logger.info(
                f"Reconciled {contract_str}: Entry {entry_price:.2f} -> Exit {exit_price:.2f} "
                f"({abs_pct_change:.2%} move) | Strategy: {strategy_type or 'DIRECTIONAL'} | "
                f"Outcome: {vol_outcome or trend} | P&L: {pnl:.4f}"
            )

            # --- Trade Journal ---
            try:
                # Construct entry decision object from row
                entry_decision = {
                    'reasoning': row.get('master_reasoning', ''),
                    'direction': row.get('master_decision', ''),
                    'confidence': float(row.get('master_confidence', 0.0) or 0.0),
                    'strategy_type': row.get('strategy_type', ''),
                    'trigger_type': row.get('trigger_type', ''),
                }

                # Construct exit data
                exit_data = {
                    'exit_price': exit_price,
                    'exit_time': target_exit_time.isoformat(),
                    'actual_trend': trend,
                    'volatility_outcome': vol_outcome
                }

                await journal.generate_post_mortem(
                    position_id=str(row.get('cycle_id', f"unknown_{index}")),
                    entry_decision=entry_decision,
                    exit_data=exit_data,
                    pnl=pnl,
                    contract=contract_str
                )
            except Exception as e:
                logger.warning(f"Failed to generate trade journal entry: {e}")

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

                    # ML model archived in v4.0 — skip recording
                    # (Legacy data preserved in CSV for historical analysis)

            except Exception as brier_e:
                logger.error(f"Failed to record Brier score: {brier_e}")

            # Respect rate limits slightly
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error processing row {index}: {e}")
            continue

    if updates_made:
        try:
            # Use columns order from utils.py to keep it clean (atomic write)
            temp_path = file_path + ".tmp"
            df.to_csv(temp_path, index=False)
            os.replace(temp_path, file_path)
            logger.info("Successfully updated council_history.csv with reconciliation results.")

            # NOTE: Structured CSV prediction resolution (agent_accuracy_structured.csv)
            # is handled by run_brier_reconciliation → resolve_with_cycle_aware_match(),
            # which runs as a separate scheduled task after reconciliation completes.
            # Enhanced Brier JSON resolution is handled by the tracker's council_history
            # backfill (Pass 3) during that same task.

        except Exception as e:
            logger.error(f"Failed to save updated CSV: {e}")
    else:
        logger.info("No rows required updates.")
