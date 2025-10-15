import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_performance_chart(df: pd.DataFrame, output_path: str = 'daily_performance.html') -> str | None:
    """
    Generates an interactive performance chart from a DataFrame and saves it to an HTML file.

    Args:
        df (pd.DataFrame): DataFrame containing the trade ledger data.
                          Must include 'timestamp', 'total_value_usd', and 'action' columns.
        output_path (str): The path to save the output HTML file.

    Returns:
        The absolute path to the saved chart HTML file, or None if the DataFrame is empty.
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

    # Save to HTML
    fig.write_html(output_path)

    return os.path.abspath(output_path)

if __name__ == '__main__':
    # For standalone testing of the chart generation
    # Create a dummy dataframe for testing
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 12:00', '2023-01-02 10:00', '2023-01-02 12:00']),
        'total_value_usd': [100, 150, -50, 200],
        'action': ['SELL', 'SELL', 'BUY', 'SELL']
    }
    df = pd.DataFrame(data)

    chart_path = generate_performance_chart(df)
    if chart_path:
        print(f"Performance chart saved to: {chart_path}")