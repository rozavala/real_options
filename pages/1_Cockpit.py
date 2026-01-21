"""
Page 1: The Cockpit (Situational Awareness)

Purpose: "Morning coffee" screen - Is the system running? Is capital safe? Any emergencies?
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    get_config,
    get_system_heartbeat,
    get_sentinel_status,
    get_ib_connection_health,
    fetch_live_dashboard_data,
    fetch_todays_benchmark_data,
    load_council_history,
    grade_decision_quality,
    calculate_rolling_win_rate,
    get_status_color,
    get_active_theses,
    get_current_market_regime,
    get_guardian_icon,
    get_strategy_color
)

st.set_page_config(layout="wide", page_title="Cockpit | Coffee Bot")

st.title("🦅 The Cockpit")
st.caption("Situational Awareness - System health, capital safety, and emergency controls")

# --- Global Time Settings ---
st.markdown("### 🕒 **All times displayed in UTC**")

# --- Load Data ---
config = get_config()
heartbeat = get_system_heartbeat()

# --- Market Clock Widget ---
import pytz
from datetime import timezone

utc_now = datetime.now(timezone.utc)
ny_tz = pytz.timezone('America/New_York')
ny_now = utc_now.astimezone(ny_tz)

# Determine Market Status (using same logic as Utils but for display)
market_open_ny = ny_now.replace(hour=3, minute=30, second=0, microsecond=0)
market_close_ny = ny_now.replace(hour=14, minute=0, second=0, microsecond=0)
is_open = market_open_ny <= ny_now <= market_close_ny and ny_now.weekday() < 5

status_color = "🟢" if is_open else "🔴"
status_text = "OPEN" if is_open else "CLOSED"

clock_cols = st.columns(3)
with clock_cols[0]:
    st.metric("UTC Time", utc_now.strftime("%H:%M:%S"))
with clock_cols[1]:
    st.metric("New York Time (Market)", ny_now.strftime("%H:%M:%S"))
with clock_cols[2]:
    st.metric("Market Status", f"{status_color} {status_text}")

st.markdown("---")

# === SECTION 1: System Heartbeat Monitor ===
st.subheader("💓 System Heartbeat")

hb_cols = st.columns(4)

with hb_cols[0]:
    orch_status = heartbeat['orchestrator_status']
    orch_color = "🟢" if orch_status == "ONLINE" else "🔴" if orch_status == "OFFLINE" else "🟡"
    st.metric(
        "Orchestrator",
        f"{orch_color} {orch_status}",
        help="Green if log updated within 10 minutes"
    )
    if heartbeat['orchestrator_last_pulse']:
        st.caption(f"Last pulse: {heartbeat['orchestrator_last_pulse'].strftime('%H:%M:%S')}")

with hb_cols[1]:
    state_status = heartbeat['state_status']
    state_color = "🟢" if state_status == "ONLINE" else "🔴" if state_status == "OFFLINE" else "🟡"
    st.metric(
        "State Manager",
        f"{state_color} {state_status}",
        help="Green if state.json updated within 10 minutes"
    )

with hb_cols[2]:
    # Sentinel Status Summary
    sentinels = get_sentinel_status()
    active_count = sum(1 for s in sentinels.values() if s == 'Active')
    st.metric("Sentinels Active", f"{active_count}/{len(sentinels)}")

with hb_cols[3]:
    ib_health = get_ib_connection_health()
    ib_status = "ONLINE" if ib_health.get("sentinel_ib") == "CONNECTED" else "OFFLINE"
    ib_color = "🟢" if ib_status == "ONLINE" else "🔴"
    st.metric("IB Gateway", f"{ib_color} {ib_status}")

    if ib_health.get("reconnect_backoff", 0) > 0:
        st.caption(f"⏳ Backoff: {ib_health['reconnect_backoff']}s")

# Sentinel Details Expander
with st.expander("🔍 Sentinel Details"):
    sentinel_cols = st.columns(4)
    for idx, (name, status) in enumerate(get_sentinel_status().items()):
        with sentinel_cols[idx % 4]:
            icon = "🟢" if status == "Active" else "💤"
            st.write(f"{icon} **{name}**: {status}")

    # === NEW: X Sentiment Sensor Status ===
    st.markdown("---")
    st.markdown("**📡 X Sentiment Sensor**")

    try:
        from trading_bot.state_manager import StateManager
        sensor_state = StateManager.load_state(namespace="sensors", max_age=3600)
        x_status = sensor_state.get("x_sentiment", {})

        if isinstance(x_status, dict):
            status = x_status.get("sentiment_sensor_status", "UNKNOWN")
            failures = x_status.get("consecutive_failures", 0)
            degraded_until = x_status.get("degraded_until")

            status_icon = "🟢" if status == "ONLINE" else "🔴" if status == "OFFLINE" else "🟡"
            st.write(f"{status_icon} Status: **{status}**")

            if failures > 0:
                st.write(f"⚠️ Consecutive failures: {failures}")

            if degraded_until:
                st.write(f"⏰ Degraded until: {degraded_until}")
        else:
            st.write("⚪ Status: Not initialized")
    except Exception as e:
        st.write(f"❓ Unable to fetch X sensor status: {e}")

st.markdown("---")

# === SECTION 2: Financial HUD ===
st.subheader("💰 Financial HUD")

if config:
    live_data = fetch_live_dashboard_data(config)
    benchmarks = fetch_todays_benchmark_data()

    fin_cols = st.columns(5)

    with fin_cols[0]:
        st.metric(
            "Net Liquidation",
            f"${live_data['NetLiquidation']:,.0f}",
            help="Total account value"
        )

    with fin_cols[1]:
        daily_pnl = live_data['DailyPnL']
        daily_pct = live_data['DailyPnLPct']

        # Handle NaN/None values from IB when market is closed or no data
        import math
        is_invalid = (
            daily_pnl is None or
            (isinstance(daily_pnl, float) and math.isnan(daily_pnl))
        )

        if is_invalid:
            st.metric(
                "Daily P&L",
                "$0.00",
                "N/A",
                help="No P&L data available - market may be closed or no positions"
            )
        else:
            st.metric(
                "Daily P&L",
                f"${daily_pnl:,.2f}",
                f"{daily_pct:+.2f}%"
            )

    with fin_cols[2]:
        margin_util = 0.0
        if live_data['NetLiquidation'] > 0:
            margin_util = (live_data['MaintMarginReq'] / live_data['NetLiquidation']) * 100
        st.metric(
            "Margin Utilization",
            f"{margin_util:.1f}%",
            delta_color="inverse" if margin_util > 50 else "normal"
        )

    with fin_cols[3]:
        st.metric("S&P 500", f"{benchmarks.get('SPY', 0):+.2f}%")

    with fin_cols[4]:
        st.metric("Coffee", f"{benchmarks.get('KC=F', 0):+.2f}%")

    # Rolling Win Rate Sparkline
    st.markdown("---")
    st.subheader("📊 Rolling Win Rate (Last 20 Decisions)")

    council_df = load_council_history()
    if not council_df.empty:
        graded = grade_decision_quality(council_df)
        rolling = calculate_rolling_win_rate(graded, window=20)

        if not rolling.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling['timestamp'],
                y=rolling['rolling_win_rate'],
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#00CC96', width=2)
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="Win Rate %",
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough graded decisions for sparkline.")
    else:
        st.info("No council history available.")

else:
    st.error("Configuration not loaded. Cannot fetch live data.")

st.markdown("---")

# === SECTION: Active Position Theses ===
st.subheader("📋 Active Position Theses")
st.caption("Real-time view of open positions and their trading rationales")

# Regime Status Banner
current_regime = get_current_market_regime(config)
regime_colors = {
    'HIGH_VOLATILITY': ('🔴', '#EF553B'),
    'RANGE_BOUND': ('🟢', '#00CC96'),
    'TRENDING': ('🟡', '#FECB52'),
    'UNKNOWN': ('⚪', '#AAAAAA')
}
regime_icon, regime_color = regime_colors.get(current_regime, ('⚪', '#AAAAAA'))

st.markdown(
    f"""
    <div style="background-color: {regime_color}22; padding: 10px; border-radius: 5px; border-left: 4px solid {regime_color};">
        <strong>{regime_icon} Current Market Regime:</strong> {current_regime}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# Active Theses Table
active_theses = get_active_theses()

if active_theses:
    # Summary metrics
    thesis_cols = st.columns(4)

    with thesis_cols[0]:
        st.metric("Active Positions", len(active_theses))

    with thesis_cols[1]:
        # Count by strategy
        strategy_counts = {}
        for t in active_theses:
            s = t['strategy_type']
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        dominant = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'None'
        st.metric("Dominant Strategy", dominant.replace('_', ' ').title())

    with thesis_cols[2]:
        # Average age
        avg_age = sum(t['age_hours'] for t in active_theses) / len(active_theses)
        st.metric("Avg Position Age", f"{avg_age:.1f}h")

    with thesis_cols[3]:
        # Average confidence
        avg_conf = sum(t['confidence'] for t in active_theses) / len(active_theses)
        st.metric("Avg Entry Confidence", f"{avg_conf:.0%}")

    st.markdown("")

    # Detailed thesis cards
    for thesis in active_theses:
        strategy_color = get_strategy_color(thesis['strategy_type'])
        guardian_icon = get_guardian_icon(thesis['guardian_agent'])

        with st.expander(
            f"{guardian_icon} **{thesis['position_id']}** | {thesis['strategy_type'].replace('_', ' ')} | Age: {thesis['age_hours']:.1f}h",
            expanded=False
        ):
            detail_cols = st.columns([2, 1, 1])

            with detail_cols[0]:
                st.markdown(f"**Rationale:** {thesis['primary_rationale']}")
                st.markdown(f"**Guardian Agent:** {guardian_icon} {thesis['guardian_agent']}")
                st.markdown(f"**Entry Regime:** {thesis['entry_regime']}")

            with detail_cols[1]:
                st.metric("Entry Price", f"${thesis['entry_price']:.2f}")
                st.metric("Confidence", f"{thesis['confidence']:.0%}")

            with detail_cols[2]:
                st.markdown("**Kill Triggers:**")
                for trigger in thesis['invalidation_triggers'][:3]:  # Show top 3
                    st.markdown(f"- `{trigger}`")
                if len(thesis['invalidation_triggers']) > 3:
                    st.caption(f"+{len(thesis['invalidation_triggers']) - 3} more")

else:
    st.info("No active position theses. The system has no open positions or theses haven't been recorded yet.")

# Regime-specific warnings
if active_theses and current_regime != 'UNKNOWN':
    # Check for regime mismatches
    at_risk = [t for t in active_theses if t['entry_regime'] != current_regime]

    if at_risk:
        st.warning(
            f"⚠️ **Regime Mismatch Alert:** {len(at_risk)} position(s) were entered in a different regime. "
            f"Morning audit will evaluate these for potential closure."
        )

st.markdown("---")

# === SECTION 3: Emergency Controls ===
st.subheader("🚨 Emergency Controls")

ctrl_cols = st.columns(3)

with ctrl_cols[0]:
    st.warning("⚠️ Global Cooldown")
    # TODO: Read TriggerDeduplicator state from state.json
    st.caption("No active cooldown")

with ctrl_cols[1]:
    if st.button("🛑 EMERGENCY HALT", type="primary", use_container_width=True):
        if config:
            with st.spinner("Cancelling all open orders..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders

                    # Create a new loop if needed or run in existing
                    try:
                        asyncio.run(cancel_all_open_orders(config))
                    except RuntimeError:
                        # If loop is already running (e.g. streamlit quirk)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(cancel_all_open_orders(config))

                    st.success("All open orders cancelled.")
                except Exception as e:
                    st.error(f"Failed to cancel orders: {e}")
        else:
            st.error("Config not loaded")

with ctrl_cols[2]:
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# === SECTION: Router Health Metrics ===
st.subheader("🔀 Router Health")

try:
    from trading_bot.heterogeneous_router import get_router

    if config:
        router = get_router(config)
        metrics = router.get_metrics_summary()

        router_cols = st.columns(4)

        with router_cols[0]:
            st.metric(
                "Total Requests",
                metrics.get('total_requests', 0)
            )

        with router_cols[1]:
            success_rate = metrics.get('overall_success_rate', 1.0) * 100
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                delta_color="normal" if success_rate > 95 else "inverse"
            )

        with router_cols[2]:
            st.metric(
                "Fallback Count",
                metrics.get('fallback_count', 0),
                delta_color="inverse"  # Lower is better
            )

        with router_cols[3]:
            st.metric(
                "Since Reset",
                metrics.get('last_reset', 'N/A')[:10] if metrics.get('last_reset') else 'N/A'
            )

        # Provider breakdown
        with st.expander("📊 Provider Breakdown"):
            provider_data = []
            for provider, counts in metrics.get('by_provider', {}).items():
                provider_data.append({
                    'Provider': provider,
                    'Success': counts.get('success', 0),
                    'Failure': counts.get('failure', 0),
                    'Success Rate': f"{counts.get('success_rate', 1.0)*100:.1f}%"
                })

            if provider_data:
                st.dataframe(pd.DataFrame(provider_data), width="stretch")
            else:
                st.info("No provider data yet")

        # Top fallback chains
        if metrics.get('top_fallback_chains'):
            with st.expander("⚠️ Top Fallback Chains"):
                for chain in metrics['top_fallback_chains'][:5]:
                    st.write(f"**{chain['role']}**: {chain['chain']} ({chain['count']} times)")

        # Recent Fallback Errors (New)
        router_instance = get_router(config)
        # Access internal metrics directly if get_metrics_summary doesn't expose it yet
        # But get_router_metrics() is singleton.
        from trading_bot.router_metrics import get_router_metrics
        rm = get_router_metrics()

        if 'recent_errors' in rm._metrics and rm._metrics['recent_errors']:
            with st.expander("⚠️ Recent Fallback Events"):
                recent_errors = rm._metrics['recent_errors']
                # Sort newest first
                recent_errors = sorted(recent_errors, key=lambda x: x['timestamp'], reverse=True)

                # Convert to DF
                err_df = pd.DataFrame(recent_errors)
                if not err_df.empty:
                    st.dataframe(
                        err_df[['timestamp', 'role', 'primary', 'fallback', 'error']],
                        width="stretch"
                    )

except ImportError:
    st.info("Router metrics not available")
except Exception as e:
    st.warning(f"Could not load router metrics: {e}")
