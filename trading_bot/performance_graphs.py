import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_performance_charts(
    trade_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    equity_df: pd.DataFrame = None,
    starting_capital: float = 250000.0
) -> list[str]:
    """
    Generates a series of life-to-date performance charts.
    - Chart 1: Cumulative P&L (Line Chart) - Uses Net Liquidation if available, else Cash Flow.
    - Chart 2: Daily P&L (Bar Chart) - Uses Daily Change in Net Liquidation if available, else Cash Flow.
    - Chart 3: P&L by Model Signal (Bar Chart) - Uses Trade Ledger (Realized P&L).

    Args:
        trade_df (pd.DataFrame): DataFrame with all trade data (ledger).
        signals_df (pd.DataFrame): DataFrame with all model signal data.
        equity_df (pd.DataFrame, optional): DataFrame with 'timestamp' and 'total_value_usd' (NetLiq).
        starting_capital (float): The starting capital for calculating return from NetLiq.

    Returns:
        A list of file paths for the generated charts.
    """
    if trade_df.empty and (equity_df is None or equity_df.empty):
        return []

    output_paths = []

    # --- Data Prep ---
    trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])

    # --- Dynamic Granularity (Determine from trade_df or equity_df) ---
    # Default to Daily
    time_unit = 'D'
    title_suffix = "Life-to-Date"
    date_format = '%Y-%m-%d'
    bar_width = 0.8

    # Check granularity based on available data
    if not trade_df.empty:
        unique_days = trade_df['timestamp'].dt.normalize().nunique()
        if unique_days == 1:
            time_unit = 'H'
            title_suffix = "Intraday"
            date_format = '%H:%M'
            bar_width = 0.03

    # --- Prepare Series for Charts 1 & 2 ---
    if equity_df is not None and not equity_df.empty:
        # Use Equity Data (Net Liquidation Value)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])

        # Ensure sorted by date
        equity_df = equity_df.sort_values('timestamp')

        # Chart 1: Cumulative P&L (NetLiq - Start)
        cumulative_pnl = equity_df.set_index('timestamp')['total_value_usd'] - starting_capital
        chart1_title = f"Total Equity P&L ({title_suffix})"

        # Chart 2: Daily P&L (Change in NetLiq)
        # Resample to ensure we have daily bars even if logging is sparse, though daily_equity should be daily.
        # Use 'last' for daily close value.
        daily_net_liq = equity_df.set_index('timestamp').resample('D')['total_value_usd'].last().dropna()
        pnl_by_time = daily_net_liq.diff()
        # For the very first day, the P&L is NetLiq - Start (if it's the first day of trading)
        # OR just 0/NaN if we treat it as change.
        # If we want the first bar to represent P&L since start:
        if not pnl_by_time.empty and pd.isna(pnl_by_time.iloc[0]):
             pnl_by_time.iloc[0] = daily_net_liq.iloc[0] - starting_capital

        chart2_title = f"Daily Change in Equity ({title_suffix})"

    else:
        # Fallback to Trade Ledger (Cash Flow)
        pnl_by_time = trade_df.resample(time_unit, on='timestamp')['total_value_usd'].sum()
        cumulative_pnl = pnl_by_time.cumsum()
        chart1_title = f"Cumulative Cash Flow ({title_suffix})"
        chart2_title = f"Daily Cash Flow ({title_suffix})"


    # --- Chart 1: Cumulative P&L ---
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl.index, cumulative_pnl, marker='o', linestyle='-', color='b', markersize=4)
    plt.title(chart1_title)
    plt.ylabel("P&L (USD)")
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
    if not pnl_by_time.empty:
        colors = ['g' if x >= 0 else 'r' for x in pnl_by_time]
        plt.bar(pnl_by_time.index, pnl_by_time, color=colors, width=bar_width)
    plt.title(chart2_title)
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
    # This always uses the Trade Ledger because signals match to specific trades
    if not signals_df.empty and not trade_df.empty:
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        merged_df = pd.merge_asof(trade_df.sort_values('timestamp'), signals_df.sort_values('timestamp'), on='timestamp', direction='backward')
        pnl_by_signal = merged_df.groupby('signal')['total_value_usd'].sum()

        # Defensive check: Only plot if there's data
        if not pnl_by_signal.empty:
            plt.figure(figsize=(10, 6))
            colors = ['g' if x >= 0 else 'r' for x in pnl_by_signal]
            pnl_by_signal.plot(kind='bar', color=colors)
            plt.title("Total P&L by Model Signal (Realized)")
            plt.ylabel("Total Net P&L (USD)")
            plt.xlabel("Model Signal")
            plt.xticks(rotation=0)
            plt.grid(True, axis='y', linestyle='--', alpha=0.6)
            path3 = os.path.abspath('pnl_by_signal_ltd.png')
            plt.savefig(path3, dpi=150)
            plt.close()
            output_paths.append(path3)

    return output_paths
