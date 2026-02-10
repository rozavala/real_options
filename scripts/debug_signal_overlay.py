#!/usr/bin/env python3
"""Test fixes for candlestick + subplots rendering."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

dates = ['2026-02-05', '2026-02-06', '2026-02-09']
opens = [310.15, 312.05, 295.20]
highs = [313.00, 315.00, 296.95]
lows = [301.60, 293.65, 285.50]
closes = [308.40, 296.55, 293.10]
volumes = [24237, 29513, 0]

# FIX A: Disable rangeslider on BOTH xaxis and xaxis2
fig_a = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                      specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
fig_a.add_trace(go.Candlestick(x=dates, open=opens, high=highs, low=lows, close=closes), row=1, col=1)
fig_a.add_trace(go.Bar(x=dates, y=volumes), row=2, col=1, secondary_y=True)
fig_a.update_layout(
    template="plotly_dark", height=600,
    xaxis=dict(rangeslider=dict(visible=False, thickness=0)),
    xaxis2=dict(rangeslider=dict(visible=False, thickness=0)),
)
fig_a.write_html("/tmp/fix_a_both_rangesliders.html")
print("Saved fix_a_both_rangesliders.html")

# FIX B: Separate figures (no subplots at all)
fig_b1 = go.Figure(data=[go.Candlestick(x=dates, open=opens, high=highs, low=lows, close=closes)])
fig_b1.update_layout(template="plotly_dark", height=450, xaxis_rangeslider_visible=False, title="Price")
fig_b1.write_html("/tmp/fix_b_price_only.html")
print("Saved fix_b_price_only.html")

# FIX C: Use go.Scatter (line chart) instead of Candlestick in subplots
fig_c = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                      specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
fig_c.add_trace(go.Scatter(x=dates, y=closes, mode='lines', name='Close', line=dict(color='#00CC96')), row=1, col=1)
fig_c.add_trace(go.Bar(x=dates, y=volumes, name='Volume'), row=2, col=1, secondary_y=True)
fig_c.update_layout(template="plotly_dark", height=600, title="FIX C: Line chart in subplots")
fig_c.write_html("/tmp/fix_c_line_subplots.html")
print("Saved fix_c_line_subplots.html")

# FIX D: Candlestick in subplots with explicit y-range and no shared xaxes
fig_d = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.75, 0.25])
fig_d.add_trace(go.Candlestick(x=dates, open=opens, high=highs, low=lows, close=closes), row=1, col=1)
fig_d.add_trace(go.Bar(x=dates, y=volumes), row=2, col=1)
fig_d.update_layout(
    template="plotly_dark", height=600,
    xaxis=dict(rangeslider=dict(visible=False)),
    yaxis=dict(range=[280, 320]),
)
fig_d.write_html("/tmp/fix_d_no_shared_x.html")
print("Saved fix_d_no_shared_x.html")

print("\nCheck each:")
print("  http://<ip>:9999/fix_a_both_rangesliders.html")
print("  http://<ip>:9999/fix_b_price_only.html")
print("  http://<ip>:9999/fix_c_line_subplots.html")
print("  http://<ip>:9999/fix_d_no_shared_x.html")
