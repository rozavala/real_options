#!/usr/bin/env python3
"""
Calibrate liquidity filter thresholds from observed spread distributions.

Reads [liquidity_metric: ...] log lines and computes per-commodity statistics
for both tick-based and percentage-based spread metrics.

Usage:
    python scripts/calibrate_liquidity_thresholds.py --log-dir logs/ --percentile 95
    python scripts/calibrate_liquidity_thresholds.py --log-dir logs/ --fail-only
    python scripts/calibrate_liquidity_thresholds.py --log-dir logs/ --output json
"""
import argparse
import re
import os
import json as json_mod
import statistics
from collections import defaultdict
from datetime import datetime, timedelta


def parse_liquidity_metrics(log_dir: str, lookback_days: int = 14,
                            fail_only: bool = False) -> dict:
    """Extract spread metrics from PASS and FAIL log lines.

    Supports two log formats:
      - v2.2 enriched: [liquidity_metric: contract=X, ..., hour_utc=H]
      - v1 legacy: [liquidity_metric: contract=X, spread_pct=P, hour_utc=H]
    """
    # v2.2 enriched pattern — uses alternation groups for inf-capable fields
    pattern_v2 = re.compile(
        r'\[liquidity_metric: contract=(\w+), expiry=(\S+), '
        r'spread_pct=([0-9.]+|inf), spread_ticks=([0-9.]+|inf), '
        r'abs_spread=([0-9.]+), theo=(-?[0-9.]+), '
        r'allowed=([0-9.]+), binding=(\w+), '
        r'tick_allow=([0-9.]+), pct_allow=([0-9.]+), '
        r'hour_utc=(\d+)\]'
    )
    # v1 fallback pattern
    pattern_v1 = re.compile(
        r'\[liquidity_metric: contract=(\w+), expiry=\S+, '
        r'spread_pct=([0-9.]+), hour_utc=(\d+)\]'
    )

    cutoff = datetime.now(tz=None) - timedelta(days=lookback_days)
    metrics = defaultdict(list)

    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith('.log'):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                if 'liquidity_metric' not in line:
                    continue

                # Filter by status
                is_fail = 'FILTER FAILED' in line
                if fail_only and not is_fail:
                    continue

                # Extract timestamp
                try:
                    ts_str = line[:19]
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                    if ts < cutoff:
                        continue
                except (ValueError, IndexError):
                    continue

                # Try v2.2 first, fall back to v1
                m = pattern_v2.search(line)
                if m:
                    entry = {
                        'contract': m.group(1),
                        'expiry': m.group(2),
                        'spread_pct': float(m.group(3)) if m.group(3) != 'inf' else float('inf'),
                        'spread_ticks': float(m.group(4)) if m.group(4) != 'inf' else float('inf'),
                        'abs_spread': float(m.group(5)),
                        'theo': float(m.group(6)),
                        'allowed': float(m.group(7)),
                        'binding': m.group(8),
                        'tick_allow': float(m.group(9)),
                        'pct_allow': float(m.group(10)),
                        'hour_utc': int(m.group(11)),
                        'status': 'FAIL' if is_fail else 'PASS',
                        'format': 'v2'
                    }
                else:
                    m = pattern_v1.search(line)
                    if m:
                        entry = {
                            'contract': m.group(1),
                            'spread_pct': float(m.group(2)),
                            'hour_utc': int(m.group(3)),
                            'status': 'FAIL' if is_fail else 'PASS',
                            'format': 'v1'
                        }
                    else:
                        continue

                metrics[entry['contract']].append(entry)

    return metrics


def compute_thresholds(metrics: dict, percentile: float = 95.0) -> dict:
    """Compute suggested thresholds at the given percentile.

    Analyzes BOTH absolute spreads (for tick calibration) and
    percentage ratios (for pct calibration) to avoid moneyness skew.
    """
    suggestions = {}
    for contract, observations in sorted(metrics.items()):
        n = len(observations)

        # Separate v2 (enriched) from v1 (legacy) entries
        v2_entries = [e for e in observations if e.get('format') == 'v2']
        has_v2 = len(v2_entries) > 0

        # --- Percentage ratio analysis (always available) ---
        finite_ratios = sorted([e['spread_pct'] for e in observations
                                if e['spread_pct'] != float('inf')])
        if not finite_ratios:
            finite_ratios = [0.0]

        p_idx = min(int(len(finite_ratios) * percentile / 100), len(finite_ratios) - 1)
        ratio_at_p = finite_ratios[p_idx]

        result = {
            'observations': n,
            'pass_count': sum(1 for e in observations if e['status'] == 'PASS'),
            'fail_count': sum(1 for e in observations if e['status'] == 'FAIL'),
            'ratio_median': round(statistics.median(finite_ratios), 3),
            f'ratio_p{int(percentile)}': round(ratio_at_p, 3),
            'suggested_max_spread_pct': round(ratio_at_p * 1.20, 2),
        }

        # --- Absolute tick analysis (v2 only) ---
        if has_v2:
            finite_ticks = sorted([e['spread_ticks'] for e in v2_entries
                                   if e['spread_ticks'] != float('inf')])
            all_abs = sorted([e['abs_spread'] for e in v2_entries])

            if not finite_ticks:
                finite_ticks = [0.0]

            t_n = len(finite_ticks)
            t_idx = min(int(t_n * percentile / 100), t_n - 1)
            a_idx = min(int(len(all_abs) * percentile / 100), len(all_abs) - 1)

            ticks_at_p = finite_ticks[t_idx]
            abs_at_p = all_abs[a_idx]

            # Binding constraint distribution
            tick_bound = sum(1 for e in v2_entries if e.get('binding') == 'TICK_FLOOR')
            pct_bound = sum(1 for e in v2_entries if e.get('binding') == 'PCT_CEILING')

            result.update({
                'v2_observations': len(v2_entries),
                'tick_median': round(statistics.median(finite_ticks), 1),
                f'tick_p{int(percentile)}': round(ticks_at_p, 1),
                'suggested_max_spread_ticks': int(ticks_at_p * 1.20) + 1,
                'abs_spread_median': round(statistics.median(all_abs), 4),
                f'abs_spread_p{int(percentile)}': round(abs_at_p, 4),
                'binding_tick_floor_pct': round(tick_bound / len(v2_entries) * 100, 1) if v2_entries else 0,
                'binding_pct_ceiling_pct': round(pct_bound / len(v2_entries) * 100, 1) if v2_entries else 0,
            })

            # Credit spread analysis
            credit_entries = [e for e in v2_entries if e.get('theo', 0) < 0]
            if credit_entries:
                result['credit_spread_count'] = len(credit_entries)
                result['credit_spread_pass_count'] = sum(1 for e in credit_entries if e['status'] == 'PASS')

        # Hour distribution (worst hours for failures)
        hour_fail_counts = defaultdict(int)
        for e in observations:
            if e['status'] == 'FAIL':
                hour_fail_counts[e['hour_utc']] += 1

        result['worst_hours_utc'] = sorted(
            hour_fail_counts.keys(),
            key=lambda h: hour_fail_counts[h],
            reverse=True
        )[:3]

        suggestions[contract] = result

    return suggestions


def main():
    parser = argparse.ArgumentParser(description='Calibrate liquidity thresholds')
    parser.add_argument('--log-dir', default='logs/', help='Directory containing log files')
    parser.add_argument('--lookback-days', type=int, default=14, help='Days of history')
    parser.add_argument('--percentile', type=float, default=95.0, help='Target percentile')
    parser.add_argument('--fail-only', action='store_true',
                        help='Analyze only FAIL entries (use before PASS logging is deployed)')
    parser.add_argument('--output', choices=['text', 'json'], default='text',
                        help='Output format (json produces ready-to-paste config snippets)')
    args = parser.parse_args()

    metrics = parse_liquidity_metrics(args.log_dir, args.lookback_days, args.fail_only)

    if not metrics:
        print("No liquidity_metric entries found in logs.")
        if not args.fail_only:
            print("Try --fail-only to analyze existing FAIL logs before PASS logging is deployed.")
        return

    suggestions = compute_thresholds(metrics, args.percentile)

    if args.output == 'json':
        config_snippets = {}
        for contract, data in suggestions.items():
            snippet = {
                'max_liquidity_spread_percentage': data['suggested_max_spread_pct'],
            }
            if 'suggested_max_spread_ticks' in data:
                snippet['max_liquidity_spread_ticks'] = data['suggested_max_spread_ticks']
            config_snippets[contract] = snippet

        print(json_mod.dumps({
            'calibration_report': {
                'generated': datetime.now(tz=None).isoformat() + 'Z',
                'lookback_days': args.lookback_days,
                'percentile': args.percentile,
                'fail_only': args.fail_only,
            },
            'suggested_overrides': config_snippets,
            'details': suggestions
        }, indent=2))
        return

    # Text output
    print(f"\n{'='*70}")
    print(f"Liquidity Threshold Calibration Report (Hybrid Tick/Percentage)")
    print(f"Lookback: {args.lookback_days} days | Percentile: {args.percentile}%"
          f"{' | FAIL-only' if args.fail_only else ''}")
    print(f"{'='*70}\n")

    for contract, data in suggestions.items():
        print(f"  {contract}:")
        print(f"    Observations: {data['observations']} "
              f"({data['pass_count']} pass, {data['fail_count']} fail)")
        print(f"    Ratio (spread/theo):  median={data['ratio_median']}, "
              f"P{int(args.percentile)}={data[f'ratio_p{int(args.percentile)}']}")
        print(f"    -> Suggested max_spread_pct: {data['suggested_max_spread_pct']}")

        if 'v2_observations' in data:
            p_key = f'tick_p{int(args.percentile)}'
            print(f"    Ticks (abs spread/tick): median={data['tick_median']}, "
                  f"P{int(args.percentile)}={data[p_key]}")
            print(f"    -> Suggested max_spread_ticks: {data['suggested_max_spread_ticks']}")
            print(f"    Binding: tick_floor={data['binding_tick_floor_pct']:.0f}%, "
                  f"pct_ceiling={data['binding_pct_ceiling_pct']:.0f}%")
            if 'credit_spread_count' in data:
                print(f"    Credit spreads: {data['credit_spread_count']} "
                      f"({data['credit_spread_pass_count']} pass)")
        else:
            print(f"    (No v2 enriched data -- tick analysis unavailable)")

        if data['worst_hours_utc']:
            print(f"    Worst hours (UTC): {data['worst_hours_utc']}")
        print()

    print("To apply, update config.json commodity_overrides -> strategy_tuning")
    print("or CommodityProfile defaults in config/commodity_profiles.py\n")


if __name__ == '__main__':
    main()
