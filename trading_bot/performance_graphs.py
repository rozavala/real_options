import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_performance_charts(trade_df: pd.DataFrame, signals_df: pd.DataFrame) -> list[str]:
    """
    Generates a series of life-to-date performance charts.
    - Chart 1: Cumulative P&L (Line Chart)
    - Chart 2: Daily P&L (Bar Chart)
    - Chart 3: P&L by Model Signal (Bar Chart)

    Args:
        trade_df (pd.DataFrame): DataFrame with all trade data.
        signals_df (pd.DataFrame): DataFrame with all model signal data.

    Returns:
        A list of file paths for the generated charts.
    """
    if trade_df.empty:
        return []

    output_paths = []

    # --- Data Prep ---
    trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])

    # --- Dynamic Granularity ---
    unique_days = trade_df['timestamp'].dt.normalize().nunique()
    if unique_days == 1:
        # Intraday (hourly) view
        time_unit = 'H'
        title_suffix = "Intraday"
        date_format = '%H:%M'
        bar_width = 0.03
    else:
        # Life-to-date (daily) view
        time_unit = 'D'
        title_suffix = "Life-to-Date"
        date_format = '%Y-%m-%d'
        bar_width = 0.8

    pnl_by_time = trade_df.resample(time_unit, on='timestamp')['total_value_usd'].sum()
    cumulative_pnl = pnl_by_time.cumsum()

    # --- Chart 1: Cumulative P&L ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl.index, cumulative_pnl, marker='o', linestyle='-', color='b', markersize=4)
    plt.title(f"Cumulative P&L ({title_suffix})")
    plt.ylabel("Equity Curve (USD)")
    plt.xlabel("Time" if time_unit == 'H' else "Date")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.gcf().autofmt_xdate()
    path1 = os.path.abspath('cumulative_pnl_ltd.png')
    plt.savefig(path1, dpi=150)
    plt.close()
    output_paths.append(path1)

    # --- Chart 2: Daily P&L ---
    plt.figure(figsize=(12, 6))
    colors = ['g' if x >= 0 else 'r' for x in pnl_by_time]
    plt.bar(pnl_by_time.index, pnl_by_time, color=colors, width=bar_width)
    plt.title(f"P&L by { 'Hour' if time_unit == 'H' else 'Day'} ({title_suffix})")
    plt.ylabel("Net P&L (USD)")
    plt.xlabel("Time" if time_unit == 'H' else "Date")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.gcf().autofmt_xdate()
    path2 = os.path.abspath('daily_pnl_ltd.png')
    plt.savefig(path2, dpi=150)
    plt.close()
    output_paths.append(path2)

    # --- Chart 3: P&L by Model Signal (Life-to-Date) ---
    if not signals_df.empty:
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        merged_df = pd.merge_asof(trade_df.sort_values('timestamp'), signals_df.sort_values('timestamp'), on='timestamp', direction='backward')
        pnl_by_signal = merged_df.groupby('signal')['total_value_usd'].sum()

        # Defensive check: Only plot if there's data
        if not pnl_by_signal.empty:
            plt.figure(figsize=(10, 6))
            colors = ['g' if x >= 0 else 'r' for x in pnl_by_signal]
            pnl_by_signal.plot(kind='bar', color=colors)
            plt.title("Total P&L by Model Signal (Life-to-Date)")
            plt.ylabel("Total Net P&L (USD)")
            plt.xlabel("Model Signal")
            plt.xticks(rotation=0)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            path3 = os.path.abspath('pnl_by_signal_ltd.png')
            plt.savefig(path3, dpi=150)
            plt.close()
            output_paths.append(path3)

    return output_paths
