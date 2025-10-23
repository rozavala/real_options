import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def generate_performance_chart(df: pd.DataFrame, output_path: str = 'daily_performance.png') -> str | None:
    """
    Generates a Month-to-Date (MTD) performance chart and saves it to a PNG file.
    The top subplot shows a cumulative P&L line chart (equity curve).
    The bottom subplot shows a daily P&L bar chart.

    Args:
        df (pd.DataFrame): DataFrame containing the complete trade ledger data.
        output_path (str): The path to save the output PNG file.

    Returns:
        The absolute path to the saved chart PNG file, or None if the DataFrame is empty.
    """
    if df.empty:
        return None

    # --- Data Preparation ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Determine the analysis month from the most recent trade
    latest_date = df['timestamp'].max()
    start_of_month = latest_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Filter for Month-to-Date (MTD) data
    mtd_df = df[df['timestamp'] >= start_of_month].copy()
    if mtd_df.empty:
        return None # No trades this month

    # Aggregate daily P&L for the bar chart
    daily_pnl = mtd_df.resample('D', on='timestamp')['signed_value_usd'].sum()

    # Calculate cumulative P&L for the line chart (equity curve)
    mtd_df_sorted = mtd_df.sort_values('timestamp')
    mtd_df_sorted['cumulative_pnl'] = mtd_df_sorted['signed_value_usd'].cumsum()

    # --- Chart Creation ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Cumulative P&L (MTD) Line Chart ---
    ax1.plot(mtd_df_sorted['timestamp'], mtd_df_sorted['cumulative_pnl'],
             marker='o', linestyle='-', color='b', markersize=4)
    ax1.set_title(f"Cumulative P&L (MTD - {start_of_month.strftime('%B %Y')})")
    ax1.set_ylabel('Equity Curve (USD)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Daily P&L (MTD) Bar Chart ---
    colors = ['g' if x >= 0 else 'r' for x in daily_pnl]
    ax2.bar(daily_pnl.index, daily_pnl, color=colors, width=0.6)
    ax2.set_title(f"Daily P&L (MTD - {start_of_month.strftime('%B %Y')})")
    ax2.set_ylabel('Net P&L (USD)')
    ax2.set_xlabel('Date')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Formatting ---
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.tight_layout()

    # --- Save to PNG ---
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return os.path.abspath(output_path)
