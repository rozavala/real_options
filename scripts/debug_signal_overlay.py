#!/usr/bin/env python3
"""Diagnostic script for Signal Overlay candlestick rendering issue."""
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=== Step 1: Fetch raw data ===")
df = yf.download('KC=F', period='5d', interval='1d', progress=False, auto_adjust=True)
print(f"Raw columns: {df.columns.tolist()}")
print(f"MultiIndex? {isinstance(df.columns, pd.MultiIndex)}")
print(f"Raw dtypes:\n{df.dtypes}")
print(f"Raw data:\n{df}\n")

print("=== Step 2: Flatten MultiIndex ===")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
print(f"Flattened columns: {df.columns.tolist()}")
print(f"Column types: {[type(c).__name__ for c in df.columns]}")
print(f"Flattened dtypes:\n{df.dtypes}\n")

print("=== Step 3: Check OHLCV values ===")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns:
        vals = df[col]
        print(f"  {col}: dtype={vals.dtype}, NaN={vals.isna().sum()}, values={vals.tolist()}")
    else:
        print(f"  {col}: MISSING!")

print(f"\n=== Step 4: Timezone handling ===")
print(f"Index tz: {df.index.tz}")
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
else:
    df.index = df.index.tz_convert('UTC')
df.index = df.index.tz_convert('America/New_York')
print(f"After conversion: {df.index.tz}")

print(f"\n=== Step 5: Create num_index ===")
df['num_index'] = range(len(df))
print(f"num_index: {df['num_index'].tolist()}")

print(f"\n=== Step 6: Test Candlestick trace ===")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                    specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

fig.add_trace(go.Candlestick(
    x=df['num_index'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="KC Test",
), row=1, col=1)

fig.add_trace(go.Bar(
    x=df['num_index'],
    y=df['Volume'],
    name='Volume',
), row=2, col=1, secondary_y=True)

# Check trace data
for i, trace in enumerate(fig.data):
    print(f"Trace {i}: type={trace.type}, name={trace.name}")
    if hasattr(trace, 'open') and trace.open is not None:
        print(f"  open={list(trace.open)[:5]}")
        print(f"  high={list(trace.high)[:5]}")
        print(f"  low={list(trace.low)[:5]}")
        print(f"  close={list(trace.close)[:5]}")
        print(f"  x={list(trace.x)[:5]}")
    elif hasattr(trace, 'y') and trace.y is not None:
        print(f"  y={list(trace.y)[:5]}")
        print(f"  x={list(trace.x)[:5]}")

# Save to HTML to test rendering
fig.update_layout(xaxis=dict(rangeslider=dict(visible=False, thickness=0)), template="plotly_dark", height=600)
fig.write_html("/tmp/candlestick_test.html")
print(f"\n=== Step 7: Saved test chart to /tmp/candlestick_test.html ===")
print("Open it in a browser or run: python3 -m http.server 9999 --directory /tmp")
print("Then visit http://<droplet-ip>:9999/candlestick_test.html")
