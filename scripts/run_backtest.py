#!/usr/bin/env python3
"""
KC Coffee Futures Backtest Runner

Validates the backtest engine with real KC=F price data using a simple
SMA crossover signal. Not meant to be profitable — just proves the
engine works end-to-end and establishes a performance baseline.

Usage:
    python scripts/run_backtest.py
"""

import sys
import os
import json
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd

from backtesting.simple_backtest import (
    SimpleBacktester,
    BacktestConfig,
    SignalDirection,
    StrategyType,
)


def sma_crossover_signal(row: pd.Series, historical: pd.DataFrame) -> dict:
    """
    SMA(5)/SMA(20) crossover signal.

    - SMA(5) > SMA(20) → BULLISH → BULL_PUT_SPREAD
    - SMA(5) < SMA(20) → BEARISH → BEAR_CALL_SPREAD
    - Confidence scaled by distance between SMAs (0.0–1.0)
    """
    if len(historical) < 20:
        return None

    close = historical["close"]
    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()

    current_sma5 = sma5.iloc[-1]
    current_sma20 = sma20.iloc[-1]

    if pd.isna(current_sma5) or pd.isna(current_sma20) or current_sma20 == 0:
        return None

    # Confidence: distance between SMAs as percentage of price, capped at 1.0
    distance_pct = abs(current_sma5 - current_sma20) / current_sma20
    confidence = min(distance_pct * 20, 1.0)  # 5% distance → confidence 1.0

    if current_sma5 > current_sma20:
        return {
            "direction": SignalDirection.BULLISH,
            "strategy": StrategyType.BULL_PUT_SPREAD,
            "confidence": confidence,
        }
    elif current_sma5 < current_sma20:
        return {
            "direction": SignalDirection.BEARISH,
            "strategy": StrategyType.BEAR_CALL_SPREAD,
            "confidence": confidence,
        }

    return None


def main():
    ticker = "KC=F"
    period = "2y"

    print(f"=== KC Coffee Futures Backtest ===")
    print(f"Fetching {period} of {ticker} data...")

    # Fetch price data
    data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        print(f"ERROR: No data returned for {ticker}. Check ticker/network.")
        sys.exit(1)

    # yfinance returns MultiIndex columns for single ticker; flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Normalize column names to lowercase
    data.columns = [c.lower() for c in data.columns]

    # Ensure 'date' column from index
    data = data.reset_index()
    data = data.rename(columns={data.columns[0]: "date"})

    start_date = data["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = data["date"].iloc[-1].strftime("%Y-%m-%d")

    print(f"Period: {start_date} to {end_date}")
    print(f"Data points: {len(data)}")
    print()

    # Configure backtest
    # KC coffee: 37,500 lbs/contract, price in cents/lb → multiplier = 375
    config = BacktestConfig(
        initial_capital=50000.0,
        max_position_pct=0.10,
        max_hold_days=2,
        commission_per_contract=2.50,
        contract_multiplier=375.0,
    )

    # Run backtest
    backtester = SimpleBacktester(config)
    result = backtester.run(data, sma_crossover_signal)

    # Print results
    metrics = result.metrics
    print("--- Results ---")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
    print(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"Avg P&L/Trade: ${metrics.get('avg_pnl', 0):,.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print()

    # Print last 10 trades
    if result.trades:
        print(f"--- Trade Log (last 10 of {len(result.trades)}) ---")
        for trade in result.trades[-10:]:
            entry = trade.entry_date.strftime("%Y-%m-%d") if hasattr(trade.entry_date, "strftime") else str(trade.entry_date)[:10]
            exit_d = trade.exit_date.strftime("%Y-%m-%d") if trade.exit_date and hasattr(trade.exit_date, "strftime") else "OPEN"
            sign = "+" if trade.pnl >= 0 else ""
            print(
                f"  {entry}  {trade.direction.value:<8}  {trade.strategy.value:<18}  "
                f"Entry: {trade.entry_price:>7.2f}  Exit: {trade.exit_price or 0:>7.2f}  "
                f"P&L: {sign}${trade.pnl:,.2f}"
            )
        print()

    # Save results to JSON
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"backtest_{timestamp}.json")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "period": period,
        "signal": "sma_crossover_5_20",
        "data_range": {"start": start_date, "end": end_date, "points": len(data)},
        "config": {
            "initial_capital": config.initial_capital,
            "max_position_pct": config.max_position_pct,
            "max_hold_days": config.max_hold_days,
            "commission_per_contract": config.commission_per_contract,
            "contract_multiplier": config.contract_multiplier,
        },
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "trades": [
            {
                "entry_date": str(t.entry_date)[:10],
                "exit_date": str(t.exit_date)[:10] if t.exit_date else None,
                "direction": t.direction.value,
                "strategy": t.strategy.value,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "contracts": t.contracts,
                "pnl": round(t.pnl, 2),
                "outcome": t.outcome,
            }
            for t in result.trades
        ],
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
