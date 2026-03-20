#!/usr/bin/env python3
"""
Backfill execution_funnel.csv from existing council_history.csv,
decision_signals.csv, trade_ledger archives, and order_events.csv.

Produces endpoint events only (COUNCIL_DECISION, STRATEGY_SELECTION,
COMPLIANCE_AUDIT, ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED, POSITION_CLOSED).
Middle stages (CONVICTION_GATE, DRAWDOWN_GATE, etc.) are unrecoverable from
historical data and will fill in via real-time instrumentation going forward.

Usage:
    python scripts/backfill_execution_funnel.py [--commodity KC] [--all] [--data-dir /path/to/data]
"""

import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.execution_funnel import SCHEMA_COLUMNS, FunnelStage
from trading_bot.exit_reasons import classify_exit_reason


def _safe_float(val):
    """Convert to float or return None."""
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def backfill_commodity(ticker: str, data_root: str) -> int:
    """Backfill funnel events for a single commodity. Returns row count."""
    data_dir = os.path.join(data_root, ticker)
    output_path = os.path.join(data_dir, 'execution_funnel.csv')

    if not os.path.isdir(data_dir):
        print(f"  [{ticker}] Data directory not found: {data_dir}. Skipping.")
        return 0

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing = pd.read_csv(output_path)
        backfill_count = (existing.get('source', pd.Series()) == 'BACKFILL').sum()
        if backfill_count > 0:
            print(f"  [{ticker}] Already has {backfill_count} backfill rows. Skipping.")
            print(f"  [{ticker}] To re-run, delete {output_path} first.")
            return 0

    rows = []

    # --- 1. Council History → COUNCIL_DECISION + COMPLIANCE_AUDIT ---
    ch_path = os.path.join(data_dir, 'council_history.csv')
    if os.path.exists(ch_path):
        print(f"  [{ticker}] Reading council_history.csv...")
        ch = pd.read_csv(ch_path, on_bad_lines='warn')
        for _, r in ch.iterrows():
            cycle_id = r.get('cycle_id', f"{ticker}-backfill-{_}")
            contract = r.get('contract', 'unknown')
            direction = r.get('master_decision', 'NEUTRAL')
            confidence = _safe_float(r.get('master_confidence', 0.0))
            regime = r.get('entry_regime', 'UNKNOWN')
            ts = r.get('timestamp', '')
            ws = _safe_float(r.get('weighted_score', 0.0))
            thesis = r.get('thesis_strength', 'N/A')
            conv_mult = _safe_float(r.get('conviction_multiplier', 1.0))

            # COUNCIL_DECISION
            rows.append({
                'timestamp': ts,
                'cycle_id': cycle_id,
                'contract': contract,
                'stage': FunnelStage.COUNCIL_DECISION.value,
                'outcome': 'PASS' if direction not in ('NEUTRAL', 'N/A', None, '') else 'INFO',
                'detail': f"direction={direction}, confidence={confidence}, thesis={thesis}, weighted_score={ws}, conviction_mult={conv_mult}",
                'price_snapshot': _safe_float(r.get('entry_price')),
                'regime': regime,
                'source': 'BACKFILL',
            })

            # COMPLIANCE_AUDIT
            compliance_val = r.get('compliance_approved')
            if pd.notna(compliance_val):
                approved = str(compliance_val).lower() in ('true', '1', 'yes')
                compliance_reason = r.get('compliance_reason', 'N/A')
                rows.append({
                    'timestamp': ts,
                    'cycle_id': cycle_id,
                    'contract': contract,
                    'stage': FunnelStage.COMPLIANCE_AUDIT.value,
                    'outcome': 'PASS' if approved else 'BLOCK',
                    'detail': 'Approved' if approved else str(compliance_reason)[:200],
                    'price_snapshot': _safe_float(r.get('entry_price')),
                    'regime': regime,
                    'source': 'BACKFILL',
                })

            # STRATEGY_SELECTION (from council_history strategy_type)
            # EMERGENCY excluded: it's a trigger_type stored in strategy_type by legacy
            # code from early 2026-02; these rows never resulted in real strategy selection.
            strategy = r.get('strategy_type', 'NONE')
            pred_type = r.get('prediction_type', 'DIRECTIONAL')
            _NON_STRATEGIES = ('NONE', '', 'N/A', 'EMERGENCY')
            _is_real_strategy = pd.notna(strategy) and strategy not in _NON_STRATEGIES
            if _is_real_strategy:
                rows.append({
                    'timestamp': ts,
                    'cycle_id': cycle_id,
                    'contract': contract,
                    'stage': FunnelStage.STRATEGY_SELECTION.value,
                    'outcome': 'PASS',
                    'detail': f"prediction_type={pred_type}, strategy={strategy}",
                    'price_snapshot': _safe_float(r.get('entry_price')),
                    'regime': regime,
                    'source': 'BACKFILL',
                })

            # PNL_RECONCILED (from exit data in council_history)
            # Only create for rows with a real executed strategy — NEUTRAL+no-strategy
            # rows (master_decision=NEUTRAL, strategy_type=NONE/EMERGENCY) are non-trades
            # with pnl_realized=0.0 which must not count as "resolved" in the dashboard.
            pnl = _safe_float(r.get('pnl_realized'))
            if pnl is not None and _is_real_strategy:
                rows.append({
                    'timestamp': r.get('exit_timestamp', ts),
                    'cycle_id': cycle_id,
                    'contract': contract,
                    'stage': FunnelStage.PNL_RECONCILED.value,
                    'outcome': 'PASS' if pnl > 0 else 'FAIL',
                    'detail': f"pnl={pnl:.2f}, trend={r.get('actual_trend_direction', 'N/A')}",
                    'price_snapshot': _safe_float(r.get('exit_price')),
                    'regime': regime,
                    'source': 'BACKFILL',
                })

        print(f"  [{ticker}] Processed {len(ch)} council history rows.")

    # --- 2. Order Events → ORDER_PLACED, ORDER_FILLED, ORDER_CANCELLED ---
    oe_path = os.path.join(data_dir, 'order_events.csv')
    if os.path.exists(oe_path):
        print(f"  [{ticker}] Reading order_events.csv...")
        oe = pd.read_csv(oe_path, on_bad_lines='warn')
        _seen_placed = set()     # Dedup: only first Submitted per order_id → ORDER_PLACED
        _seen_cancelled = set()  # Dedup: only first Cancelled per order_id → ORDER_CANCELLED
        for _, r in oe.iterrows():
            ts = r.get('timestamp', '')
            status = str(r.get('status', '')).lower()
            order_id = r.get('orderId', 'unknown')
            local_sym = r.get('local_symbol', 'unknown')
            action = r.get('action', '')
            qty = r.get('quantity', 0)
            lmt_price = _safe_float(r.get('lmtPrice'))
            msg = str(r.get('message', ''))[:200]

            if 'submitted' in status or 'presubmitted' in status:
                if order_id in _seen_placed:
                    continue  # IB re-emits Submitted on price modifications — skip dupes
                _seen_placed.add(order_id)
                rows.append({
                    'timestamp': ts,
                    'cycle_id': f"ORDER-{order_id}",
                    'contract': local_sym,
                    'stage': FunnelStage.ORDER_PLACED.value,
                    'outcome': 'PASS',
                    'detail': f"action={action}, qty={qty}, order_id={order_id}",
                    'initial_limit': lmt_price,
                    'source': 'BACKFILL',
                })
            elif 'filled' in status:
                rows.append({
                    'timestamp': ts,
                    'cycle_id': f"ORDER-{order_id}",
                    'contract': local_sym,
                    'stage': FunnelStage.ORDER_FILLED.value,
                    'outcome': 'PASS',
                    'detail': f"action={action}, qty={qty}, order_id={order_id}",
                    'fill_price': lmt_price,
                    'source': 'BACKFILL',
                })
            elif 'cancelled' in status or 'timedout' in status:
                if order_id in _seen_cancelled:
                    continue  # IB re-emits Cancelled repeatedly near timeout — skip dupes
                _seen_cancelled.add(order_id)
                rows.append({
                    'timestamp': ts,
                    'cycle_id': f"ORDER-{order_id}",
                    'contract': local_sym,
                    'stage': FunnelStage.ORDER_CANCELLED.value,
                    'outcome': 'BLOCK',
                    'detail': f"action={action}, reason={msg}",
                    'walk_away_price': lmt_price,
                    'source': 'BACKFILL',
                })
            elif 'updated' in status and 'adaptive' in msg.lower():
                rows.append({
                    'timestamp': ts,
                    'cycle_id': f"ORDER-{order_id}",
                    'contract': local_sym,
                    'stage': FunnelStage.PRICE_WALK_STEP.value,
                    'outcome': 'INFO',
                    'detail': msg,
                    'source': 'BACKFILL',
                })

        print(f"  [{ticker}] Processed {len(oe)} order events.")

    # --- 3. Trade Ledger (latest archive) → ORDER_FILLED / POSITION_CLOSED ---
    ledger_dir = os.path.join(data_dir, 'archive_ledger')
    if os.path.isdir(ledger_dir):
        # Find the latest _v2 or regular ledger
        v2_files = sorted(glob.glob(os.path.join(ledger_dir, '*_v2.csv')))
        regular_files = sorted(glob.glob(os.path.join(ledger_dir, '*.csv')))
        ledger_file = v2_files[-1] if v2_files else (regular_files[-1] if regular_files else None)

        if ledger_file:
            print(f"  [{ticker}] Reading trade ledger: {os.path.basename(ledger_file)}...")
            tl = pd.read_csv(ledger_file, on_bad_lines='warn')
            for _, r in tl.iterrows():
                ts = r.get('timestamp', '')
                position_id = r.get('position_id', 'unknown')
                local_sym = r.get('local_symbol', 'unknown')
                action = r.get('action', '')
                qty = r.get('quantity', 0)
                fill_price = _safe_float(r.get('avg_fill_price'))
                reason = str(r.get('reason', 'Strategy Execution'))

                # Determine if this is an entry or exit based on reason
                exit_reason = classify_exit_reason(reason)

                if reason == 'Strategy Execution':
                    # Entry fill
                    rows.append({
                        'timestamp': ts,
                        'cycle_id': position_id,
                        'contract': local_sym,
                        'stage': FunnelStage.ORDER_FILLED.value,
                        'outcome': 'PASS',
                        'detail': f"action={action}, qty={qty}, reason={reason}",
                        'fill_price': fill_price,
                        'source': 'BACKFILL',
                    })
                else:
                    # Exit / close
                    rows.append({
                        'timestamp': ts,
                        'cycle_id': position_id,
                        'contract': local_sym,
                        'stage': FunnelStage.POSITION_CLOSED.value,
                        'outcome': 'PASS',
                        'detail': f"action={action}, qty={qty}, exit_reason={exit_reason.value}",
                        'fill_price': fill_price,
                        'source': 'BACKFILL',
                    })

            print(f"  [{ticker}] Processed {len(tl)} trade ledger entries.")

    # --- Write ---
    if rows:
        df = pd.DataFrame(rows)
        # Ensure all schema columns exist
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[SCHEMA_COLUMNS]

        # Sort by timestamp
        df['_ts_sort'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df = df.sort_values('_ts_sort', na_position='last').drop(columns=['_ts_sort'])

        df.to_csv(output_path, index=False)
        print(f"  [{ticker}] Wrote {len(df)} backfill events to {output_path}")
        return len(df)
    else:
        print(f"  [{ticker}] No data found to backfill.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Backfill execution funnel from existing CSVs")
    parser.add_argument('--commodity', type=str, help='Single commodity ticker (e.g., KC)')
    parser.add_argument('--all', action='store_true', help='Backfill all commodities')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Root data directory (default: <project>/data)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_dir or os.path.join(project_root, 'data')

    if args.all:
        tickers = [d for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))
                    and d not in ('__pycache__', 'tms', 'archive')]
    elif args.commodity:
        tickers = [args.commodity]
    else:
        tickers = ['KC', 'CC', 'NG']

    print(f"Backfilling execution funnel from: {data_root}")
    print(f"Commodities: {', '.join(sorted(tickers))}\n")

    total = 0
    for ticker in sorted(tickers):
        print(f"--- {ticker} ---")
        total += backfill_commodity(ticker, data_root)
        print()

    print(f"Backfill complete: {total} total events across {len(tickers)} commodities.")


if __name__ == '__main__':
    main()
