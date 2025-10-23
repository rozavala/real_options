import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_performance_chart(df: pd.DataFrame, output_path: str = 'daily_performance.png') -> str | None:
    """
    Generates a performance chart using Matplotlib and saves it to a PNG file.

    Args:
        df (pd.DataFrame): DataFrame containing the trade ledger data.
        output_path (str): The path to save the output PNG file.

    Returns:
        The absolute path to the saved chart PNG file, or None if the DataFrame is empty.
    """
    if df.empty:
        return None

    df['net_value'] = df['signed_value_usd']
    df['cumulative_net_value'] = df['net_value'].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [1, 1]})

    # Daily P&L Bar Chart
    colors = ['g' if x > 0 else 'r' for x in df['net_value']]
    ax1.bar(df['timestamp'], df['net_value'], color=colors)
    ax1.set_title('Daily Profit & Loss')
    ax1.set_ylabel('P&L (USD)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Cumulative P&L Line Chart
    ax2.plot(df['timestamp'], df['cumulative_net_value'], marker='o', linestyle='-', color='b')
    ax2.set_title('Cumulative Performance')
    ax2.set_xlabel('Date & Time')
    ax2.set_ylabel('Total P&L (USD)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Format x-axis dates for better readability
    plt.gcf().autofmt_xdate()
    formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
    plt.gca().xaxis.set_major_formatter(formatter)

    fig.suptitle('Trading Performance', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save to PNG
    plt.savefig(output_path)
    plt.close(fig)

    return os.path.abspath(output_path)
