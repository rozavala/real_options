"""
Page 8: LLM Monitor

Purpose: Visibility into LLM API costs, budget utilization, provider health,
and latency trends. Makes the cost infrastructure (budget_guard, router_metrics,
api_costs) visible on the dashboard.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide", page_title="LLM Monitor | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

from dashboard_utils import (
    _resolve_data_path_for,
    load_budget_status,
    load_llm_daily_costs,
    get_config,
)

config = get_config()

st.title("LLM Monitor")
st.caption("API costs, budget utilization, provider health, and latency")


# =====================================================================
# SECTION 1: Today's Budget
# =====================================================================
st.subheader("Today's Budget")

budget = load_budget_status(ticker)

if budget:
    daily_spend = budget.get('daily_spend', 0.0)
    daily_budget = budget.get('daily_budget', 15.0)
    remaining = max(0.0, daily_budget - daily_spend)
    pct_used = (daily_spend / daily_budget * 100) if daily_budget > 0 else 0
    request_count = budget.get('request_count', 0)
    sentinel_only = budget.get('budget_hit', False)

    # Throttle level label
    remaining_pct = remaining / daily_budget if daily_budget > 0 else 1.0
    if remaining_pct > 0.50:
        throttle_label = "All Clear"
        throttle_color = "green"
    elif remaining_pct > 0.25:
        throttle_label = "BACKGROUND blocked"
        throttle_color = "orange"
    elif remaining_pct > 0.10:
        throttle_label = "LOW+ blocked"
        throttle_color = "red"
    elif sentinel_only:
        throttle_label = "BUDGET HIT"
        throttle_color = "red"
    else:
        throttle_label = "CRITICAL only"
        throttle_color = "red"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Spend", f"${daily_spend:.2f}")
    col2.metric("Remaining", f"${remaining:.2f}")
    col3.metric("Requests", str(request_count))
    col4.metric("Throttle", throttle_label)

    # Progress bar
    if pct_used < 50:
        bar_color = "green"
    elif pct_used < 75:
        bar_color = "orange"
    else:
        bar_color = "red"

    # Streamlit progress bar (0.0 to 1.0)
    st.progress(
        min(pct_used / 100, 1.0),
        text=f"${daily_spend:.2f} / ${daily_budget:.2f} ({pct_used:.1f}%)"
    )

    if sentinel_only:
        st.error("Budget exhausted — sentinel-only mode active. LLM reasoning disabled until midnight UTC reset.")
else:
    st.info("Budget guard not initialized. Data appears after the orchestrator starts.")


# =====================================================================
# SECTION 1b: X/Twitter API Usage
# =====================================================================
x_api_calls = budget.get('x_api_calls', 0) if budget else 0
if x_api_calls > 0:
    st.markdown("---")
    st.subheader("X (Twitter) API Usage")
    col1, col2 = st.columns(2)
    col1.metric("API Calls Today", str(x_api_calls))
    col2.caption(
        "X API calls are counted but not included in the LLM budget. "
        "These use the X Bearer Token (separate billing)."
    )


# =====================================================================
# SECTION 2: Cost by Agent Role
# =====================================================================
st.markdown("---")
st.subheader("Cost by Agent Role (Today)")

cost_by_source = budget.get('cost_by_source', {}) if budget else {}

if cost_by_source:
    import plotly.express as px

    # Build DataFrame sorted by cost descending
    source_rows = [
        {'Role': source, 'Cost ($)': round(cost, 4)}
        for source, cost in cost_by_source.items()
    ]
    source_df = pd.DataFrame(source_rows).sort_values('Cost ($)', ascending=True)

    fig = px.bar(
        source_df,
        x='Cost ($)',
        y='Role',
        orientation='h',
        height=max(300, len(source_df) * 28 + 80),
    )
    fig.update_layout(
        margin=dict(t=10, b=30),
        yaxis_title=None,
        xaxis_title='Cost (USD)',
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No per-source cost data yet. Costs are tracked as API calls are made.")


# =====================================================================
# SECTION 3: Provider Health
# =====================================================================
st.markdown("---")
st.subheader("Provider Health")

metrics_path = _resolve_data_path_for('router_metrics.json', ticker)

try:
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            router_data = json.load(f)
    else:
        router_data = None
except Exception:
    router_data = None

if router_data and router_data.get('requests'):
    # Aggregate by provider
    provider_stats = {}
    for key, counts in router_data.get('requests', {}).items():
        _role, provider = key.split(':') if ':' in key else ('unknown', key)
        if provider not in provider_stats:
            provider_stats[provider] = {'success': 0, 'failure': 0, 'latencies': []}
        provider_stats[provider]['success'] += counts.get('success', 0)
        provider_stats[provider]['failure'] += counts.get('failure', 0)

    # Collect latencies by provider
    for key, lat_list in router_data.get('latencies', {}).items():
        _role, provider = key.split(':') if ':' in key else ('unknown', key)
        if provider in provider_stats:
            provider_stats[provider]['latencies'].extend(
                [entry['ms'] for entry in lat_list if isinstance(entry, dict)]
            )

    # Provider summary table
    table_rows = []
    for provider, stats in sorted(provider_stats.items()):
        total = stats['success'] + stats['failure']
        success_rate = stats['success'] / total if total > 0 else 1.0
        lats = stats['latencies']

        avg_lat = sum(lats) / len(lats) if lats else 0
        p95_lat = sorted(lats)[int(len(lats) * 0.95)] if len(lats) >= 5 else (max(lats) if lats else 0)

        table_rows.append({
            'Provider': provider,
            'Requests': total,
            'Success Rate': f"{success_rate:.1%}",
            'Avg Latency (ms)': f"{avg_lat:.0f}" if lats else "N/A",
            'P95 Latency (ms)': f"{p95_lat:.0f}" if lats else "N/A",
        })

    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    # Latency box plot
    all_latencies = []
    for provider, stats in provider_stats.items():
        for ms in stats['latencies']:
            all_latencies.append({'Provider': provider, 'Latency (ms)': ms})

    if all_latencies:
        import plotly.express as px

        lat_df = pd.DataFrame(all_latencies)
        fig_lat = px.box(
            lat_df,
            x='Provider',
            y='Latency (ms)',
            height=350,
        )
        fig_lat.update_layout(margin=dict(t=10, b=30))
        st.plotly_chart(fig_lat, use_container_width=True)

    # Fallback chains
    fallbacks = router_data.get('fallbacks', {})
    if fallbacks:
        with st.expander("Fallback Chains"):
            fb_rows = []
            for role, chains in fallbacks.items():
                for chain, count in chains.items():
                    fb_rows.append({'Role': role, 'Chain': chain, 'Count': count})
            fb_df = pd.DataFrame(fb_rows).sort_values('Count', ascending=False)
            st.dataframe(fb_df, hide_index=True, use_container_width=True)
else:
    st.info("No router metrics data yet. Data appears after the first LLM API call.")


# =====================================================================
# SECTION 4: Daily Cost Trend (last 30 days)
# =====================================================================
st.markdown("---")
st.subheader("Daily Cost Trend")

costs_df = load_llm_daily_costs(ticker)

if not costs_df.empty and 'date' in costs_df.columns and 'total_usd' in costs_df.columns:
    import plotly.graph_objects as go

    # Last 30 days
    costs_df = costs_df.sort_values('date').tail(30)

    fig_trend = go.Figure()

    fig_trend.add_trace(go.Scatter(
        x=costs_df['date'],
        y=costs_df['total_usd'],
        mode='lines+markers',
        name='Daily Cost',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=6),
    ))

    # Budget limit reference line
    cost_config = config.get('cost_management', {})
    budget_limit = cost_config.get('daily_budget_usd', 15.0)
    fig_trend.add_hline(
        y=budget_limit,
        line_dash="dash",
        line_color="red",
        line_width=1,
        annotation_text=f"Budget: ${budget_limit:.0f}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    fig_trend.update_layout(
        yaxis_title='Cost (USD)',
        xaxis_title='Date',
        height=350,
        margin=dict(t=20, b=40),
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Request count trend (secondary)
    if 'request_count' in costs_df.columns:
        with st.expander("Request Volume"):
            st.bar_chart(costs_df.set_index('date')['request_count'])
else:
    st.info(
        "No daily cost history yet. History accumulates after the first midnight UTC reset "
        "following budget guard initialization."
    )


# =====================================================================
# SECTION 5: Model Pricing Reference
# =====================================================================
st.markdown("---")
st.subheader("Model Pricing Reference")

try:
    cost_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'api_costs.json'
    )
    with open(cost_file, 'r') as f:
        api_costs = json.load(f)

    costs = api_costs.get('costs_per_1k_tokens', {})
    last_updated = api_costs.get('last_updated', 'Unknown')

    pricing_rows = []
    for model, pricing in sorted(costs.items()):
        if model == 'default':
            continue
        if isinstance(pricing, dict):
            pricing_rows.append({
                'Model': model,
                'Input ($/1K tokens)': f"${pricing.get('input', 0):.5f}",
                'Output ($/1K tokens)': f"${pricing.get('output', 0):.5f}",
            })

    if pricing_rows:
        st.dataframe(pd.DataFrame(pricing_rows), hide_index=True, use_container_width=True)
        st.caption(f"Pricing last updated: {last_updated}")

except Exception as e:
    st.warning(f"Could not load api_costs.json: {e}")


st.markdown("---")
st.caption("LLM Monitor | Real Options")
