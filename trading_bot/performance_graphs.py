import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter


def generate_performance_chart(df: pd.DataFrame, output_path: str = 'daily_performance.png') -> str | None:
    """
    Generates a performance chart from a DataFrame and saves it to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the trade ledger data.
                          Must include 'timestamp', 'total_value_usd', and 'action' columns.
        output_path (str): The path to save the output PNG file.

    Returns:
        The absolute path to the saved chart image, or None if the DataFrame is empty.
    """
    if df.empty:
        return None

    # Set SELL as positive cashflows and BUY as negative cashflows
    df['net_value'] = df.apply(
        lambda row: row['total_value_usd'] if row['action'] == 'SELL' else -row['total_value_usd'],
        axis=1
    )

    # Calculates the cumulative net value
    df['cumulative_net_value'] = df['net_value'].cumsum()

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fig.suptitle('Trading Performance', fontsize=16)

    # Plot the cashflows on the top subplot
    ax1.bar(df['timestamp'], df['net_value'], color=['green' if x > 0 else 'red' for x in df['net_value']], label='Daily P&L')
    ax1.set_title('Daily Profit & Loss')
    ax1.set_ylabel('P&L (USD)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # Plot the cumulative net value on the bottom subplot
    ax2.plot(df['timestamp'], df['cumulative_net_value'], color='#1E90FF', marker='o', linestyle='-', label='Cumulative P&L')
    ax2.set_title('Cumulative Performance')
    ax2.set_xlabel('Date & Time')
    ax2.set_ylabel('Total P&L (USD)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    # Improve date formatting
    fig.autofmt_xdate()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

    return os.path.abspath(output_path)

if __name__ == '__main__':
    # For standalone testing of the chart generation
    chart_path = generate_performance_chart()
    if chart_path:
        print(f"Performance chart saved to: {chart_path}")