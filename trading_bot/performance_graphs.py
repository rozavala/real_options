import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_performance_chart(df: pd.DataFrame, output_path: str = 'daily_performance.png') -> str | None:
    """
    Generates an interactive performance chart from a DataFrame and saves it to a PNG file.

    Args:
        df (pd.DataFrame): DataFrame containing the trade ledger data.
                          Must include 'timestamp', 'total_value_usd', and 'action' columns.
        output_path (str): The path to save the output PNG file.

    Returns:
        The absolute path to the saved chart PNG file, or None if the DataFrame is empty.
    """
    if df.empty:
        return None

    # The 'signed_value_usd' column represents cashflow (SELL positive, BUY negative)
    df['net_value'] = df['signed_value_usd']

    # Calculates the cumulative net value
    df['cumulative_net_value'] = df['net_value'].cumsum()

    # Create a figure with two subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Daily Profit & Loss', 'Cumulative Performance')
    )

    # Daily P&L Bar Chart
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['net_value'],
        marker_color=['green' if x > 0 else 'red' for x in df['net_value']],
        name='Daily P&L'
    ), row=1, col=1)

    # Cumulative P&L Line Chart
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_net_value'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#1E90FF')
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title_text='Trading Performance',
        xaxis_title='Date & Time',
        yaxis_title='P&L (USD)',
        yaxis2_title='Total P&L (USD)',
        showlegend=False,
        height=600
    )

    # Save to PNG
    fig.write_image(output_path)

    return os.path.abspath(output_path)