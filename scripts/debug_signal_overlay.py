#!/usr/bin/env python3
"""Diagnostic: test 4 different candlestick approaches to find what renders."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Hardcoded data - no yfinance dependency
dates = ['2026-02-05', '2026-02-06', '2026-02-09']
opens = [310.15, 312.05, 295.20]
highs = [313.00, 315.00, 296.95]
lows = [301.60, 293.65, 285.50]
closes = [308.40, 296.55, 293.10]
volumes = [24237, 29513, 0]

# TEST 1: Simple candlestick, date x-axis, NO subplots
fig1 = go.Figure(data=[go.Candlestick(
    x=dates, open=opens, high=highs, low=lows, close=closes, name="Test 1: Dates"
)])
fig1.update_layout(title="TEST 1: Date x-axis, no subplots", template="plotly_dark",
                   xaxis_rangeslider_visible=False, height=400)
fig1.write_html("/tmp/test1_dates_simple.html")
print("Saved test1_dates_simple.html")

# TEST 2: Numerical x-axis, NO subplots
fig2 = go.Figure(data=[go.Candlestick(
    x=[0, 1, 2], open=opens, high=highs, low=lows, close=closes, name="Test 2: Nums"
)])
fig2.update_layout(title="TEST 2: Numerical x-axis, no subplots", template="plotly_dark",
                   xaxis_rangeslider_visible=False, height=400)
fig2.write_html("/tmp/test2_nums_simple.html")
print("Saved test2_nums_simple.html")

# TEST 3: Date x-axis WITH subplots (matching Signal Overlay structure)
fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                     specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
fig3.add_trace(go.Candlestick(
    x=dates, open=opens, high=highs, low=lows, close=closes, name="Price"
), row=1, col=1)
fig3.add_trace(go.Bar(x=dates, y=volumes, name='Volume'), row=2, col=1, secondary_y=True)
fig3.update_layout(title="TEST 3: Dates + subplots", template="plotly_dark",
                   xaxis=dict(rangeslider=dict(visible=False, thickness=0)), height=600)
fig3.write_html("/tmp/test3_dates_subplots.html")
print("Saved test3_dates_subplots.html")

# TEST 4: Numerical x-axis WITH subplots (what Signal Overlay actually does)
fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                     specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
fig4.add_trace(go.Candlestick(
    x=[0, 1, 2], open=opens, high=highs, low=lows, close=closes, name="Price"
), row=1, col=1)
fig4.add_trace(go.Bar(x=[0, 1, 2], y=volumes, name='Volume'), row=2, col=1, secondary_y=True)
fig4.update_layout(title="TEST 4: Nums + subplots", template="plotly_dark",
                   xaxis=dict(rangeslider=dict(visible=False, thickness=0)), height=600)
fig4.write_html("/tmp/test4_nums_subplots.html")
print("Saved test4_nums_subplots.html")

# TEST 5: OHLC trace instead of Candlestick (alternative rendering)
fig5 = go.Figure(data=[go.Ohlc(
    x=dates, open=opens, high=highs, low=lows, close=closes, name="Test 5: OHLC"
)])
fig5.update_layout(title="TEST 5: OHLC trace (not candlestick)", template="plotly_dark",
                   xaxis_rangeslider_visible=False, height=400)
fig5.write_html("/tmp/test5_ohlc.html")
print("Saved test5_ohlc.html")

print("\nAll tests saved. Check each in browser:")
print("  http://<ip>:9999/test1_dates_simple.html")
print("  http://<ip>:9999/test2_nums_simple.html")
print("  http://<ip>:9999/test3_dates_subplots.html")
print("  http://<ip>:9999/test4_nums_subplots.html")
print("  http://<ip>:9999/test5_ohlc.html")
