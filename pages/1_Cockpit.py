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
import json
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import (
    get_config,
    get_system_heartbeat,
    get_sentinel_status,
    get_ib_connection_health,
    fetch_all_live_data,
    fetch_todays_benchmark_data,
    load_council_history,
    grade_decision_quality,
    calculate_rolling_win_rate,
    get_status_color,
    get_active_theses,
    get_current_market_regime,
    get_guardian_icon,
    get_strategy_color,
    get_commodity_profile
)

def load_deduplicator_metrics() -> dict:
    """Load trigger deduplication metrics."""
    try:
        with open('data/deduplicator_state.json', 'r') as f:
            data = json.load(f)
            metrics = data.get('metrics', {})
            total = metrics.get('total_triggers', 0)
            processed = metrics.get('processed', 0)

            return {
                'total_triggers': total,
                'processed': processed,
                'filtered': total - processed,
                'efficiency': processed / total if total > 0 else 1.0,
            }
    except Exception:
        return {'total_triggers': 0, 'processed': 0, 'efficiency': 1.0}


def _parse_price_from_text(text: str, entry_price: float, min_price: float, max_price: float) -> float | None:
    """Helper to extract price from trigger text."""
    text_upper = text.upper()

    # Look for price keywords
    if any(kw in text_upper for kw in ['STOP', 'CLOSE', 'EXIT', 'PRICE <', 'PRICE >', 'BELOW', 'ABOVE', 'BREACH']):
        match = re.search(r'\$?(\d+(?:\.\d{1,2})?)', text)
        if match:
            price = float(match.group(1))
            if min_price <= price <= max_price:
                return price

    # Check for percentage-based stops
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if pct_match and entry_price and entry_price > 0:
        pct = float(pct_match.group(1)) / 100
        calculated_stop = entry_price * (1 - pct)
        if min_price <= calculated_stop <= max_price:
            return calculated_stop

    return None


def extract_stop_price_from_triggers(
    triggers: any,
    entry_price: float,
    config: dict = None
) -> float | None:
    """
    Extracts stop/invalidation price from various trigger formats.

    Handles:
    - Dict: {'stop_loss_price': 320.0}
    - List: ['Close if price < 320', 'Monitor frost']
    - String: 'Stop at 5% loss'

    Uses config-driven price bounds to filter false positives.
    """
    if triggers is None:
        return None

    # Get commodity-specific bounds (defaults to KC)
    profile = get_commodity_profile(config)
    min_price, max_price = profile.get('stop_parse_range', [80, 800])

    # === Dict format ===
    if isinstance(triggers, dict):
        for key in ['stop_loss_price', 'stop_price', 'invalidation_price', 'exit_price']:
            if key in triggers:
                price = float(triggers[key])
                if min_price <= price <= max_price:
                    return price
        return None

    # === List format ===
    if isinstance(triggers, list):
        for trigger in triggers:
            if isinstance(trigger, str):
                price = _parse_price_from_text(trigger, entry_price, min_price, max_price)
                if price is not None:
                    return price
        return None

    # === String format ===
    if isinstance(triggers, str):
        return _parse_price_from_text(triggers, entry_price, min_price, max_price)

    return None


def render_thesis_card_enhanced(thesis: dict, live_data: dict, config: dict = None):
    """Renders thesis card with live P&L and distance to invalidation."""
    position_id = thesis.get('position_id', 'UNKNOWN')
    strategy = thesis.get('strategy_type', 'DIRECTIONAL')
    entry_price = thesis.get('entry_price', 0) or thesis.get('supporting_data', {}).get('entry_price', 0)

    # Find matching position in portfolio
    unrealized_pnl = None
    for item in live_data.get('portfolio_items', []):
        if hasattr(item, 'contract') and position_id in str(item.contract.localSymbol):
            unrealized_pnl = getattr(item, 'unrealizedPNL', None)
            break

    # Extract stop price from triggers
    triggers = thesis.get('invalidation_triggers', [])
    stop_price = extract_stop_price_from_triggers(triggers, entry_price, config)

    # Calculate distance to stop
    distance_pct = None
    if stop_price and entry_price and entry_price != 0:
        distance_pct = abs(entry_price - stop_price) / entry_price

    # Render
    with st.container():
        st.markdown(f"### 📋 {position_id}")
        st.caption(f"Strategy: **{strategy}** | Guardian: {thesis.get('guardian_agent', 'Master')}")

        cols = st.columns(4)

        with cols[0]:
            st.metric("Entry", f"${entry_price:.2f}" if entry_price else "N/A")

        with cols[1]:
            if unrealized_pnl is not None:
                st.metric("Unrealized P&L", f"${unrealized_pnl:+,.2f}")
            else:
                st.metric("Unrealized P&L", "N/A")

        with cols[2]:
            st.metric("Stop Price", f"${stop_price:.2f}" if stop_price else "N/A")

        with cols[3]:
            if distance_pct is not None:
                if distance_pct < 0.05:
                    st.error(f"⚠️ {distance_pct:.1%} to stop")
                elif distance_pct < 0.10:
                    st.warning(f"🔶 {distance_pct:.1%} to stop")
                else:
                    st.success(f"✅ {distance_pct:.1%} to stop")
            else:
                st.info("No stop defined")

        with st.expander("📜 Invalidation Triggers"):
            if isinstance(triggers, list):
                for t in triggers:
                    st.write(f"• {t}")
            else:
                st.write(str(triggers) if triggers else "None defined")


def render_portfolio_risk_summary(live_data: dict):
    """Portfolio risk using margin/P&L proxies (not live Greeks)."""
    st.subheader("📊 Portfolio Risk")

    net_liq = live_data.get('net_liquidation', 0)
    margin = live_data.get('maint_margin', 0)
    daily_pnl = live_data.get('daily_pnl', 0)

    cols = st.columns(4)

    with cols[0]:
        st.metric("Net Liquidation", f"${net_liq:,.0f}")

    with cols[1]:
        if net_liq > 0:
            margin_util = (margin / net_liq) * 100
            st.metric("Margin Util", f"{margin_util:.1f}%")
        else:
            st.metric("Margin Util", "N/A")

    with cols[2]:
        st.metric("Daily P&L", f"${daily_pnl:+,.0f}")

    with cols[3]:
        pos_count = len([p for p in live_data.get('open_positions', []) if p.position != 0])
        st.metric("Open Positions", pos_count)


def render_prediction_markets():
    """
    Prediction Market Radar Panel.

    CRITICAL: This provides visibility into Dynamic Discovery auto-selection.
    Without this, you have no idea what markets the sentinel is actually tracking.
    """
    st.subheader("🔮 Prediction Markets (Macro Radar)")

    try:
        from trading_bot.state_manager import StateManager

        # Load the sentinel's state cache
        state = StateManager.load_state(namespace="prediction_market_state")

        if not state:
            st.info("No prediction market data active. Sentinel may not have run yet.")
            return

        # Handle varying number of topics (1-6)
        num_topics = len(state)
        cols_per_row = min(3, num_topics)

        # Create responsive columns
        cols = st.columns(cols_per_row)

        for i, (topic, data) in enumerate(state.items()):
            col_idx = i % cols_per_row

            with cols[col_idx]:
                # Extract state data
                title = data.get('title', topic)[:50]  # Truncate long titles
                price = data.get('price', 0)
                slug = data.get('slug', 'N/A')
                hwm = data.get('severity_hwm', 0)
                tag = data.get('tag', topic[:10])
                timestamp = data.get('timestamp', 'Unknown')

                # Color-code based on alert state
                # "off" = red (alerting), "normal" = green (stable)
                delta_color = "off" if hwm > 0 else "normal"

                # Display metric card
                st.metric(
                    label=f"{tag}: {price*100:.0f}%",
                    value=f"{price:.2f}",
                    delta=f"HWM: {hwm}" if hwm > 0 else "Stable",
                    delta_color=delta_color
                )

                # Market title (what the sentinel actually picked!)
                st.caption(f"*{title}*")

                # Link to Polymarket for verification
                if slug and slug != 'N/A':
                    st.markdown(f"[View Market ↗](https://polymarket.com/event/{slug})")

                # Show last update time
                if timestamp and timestamp != 'Unknown':
                    try:
                        from datetime import datetime
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        age_min = (datetime.now(ts.tzinfo) - ts).total_seconds() / 60
                        if age_min < 60:
                            st.caption(f"Updated {age_min:.0f}m ago")
                        else:
                            st.caption(f"Updated {age_min/60:.1f}h ago")
                    except:
                        pass

                st.divider()

        # Add refresh hint
        st.caption("💡 Data refreshes every 5 minutes via sentinel polling")

    except FileNotFoundError:
        st.info("Prediction market state file not found. Sentinel may not have run yet.")
    except Exception as e:
        st.error(f"Radar Error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.set_page_config(layout="wide", page_title="Cockpit | Mission Control")

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

# Deduplicator Metrics
dedup_metrics = load_deduplicator_metrics()
if dedup_metrics.get('total_triggers', 0) > 0:
    st.markdown("---")
    st.caption("🛡️ Trigger Deduplicator")
    d_cols = st.columns(4)
    with d_cols[0]:
        st.metric("Total Triggers", dedup_metrics['total_triggers'])
    with d_cols[1]:
        st.metric("Processed", dedup_metrics['processed'])
    with d_cols[2]:
        st.metric("Filtered", dedup_metrics['filtered'])
    with d_cols[3]:
        st.metric("Efficiency", f"{dedup_metrics['efficiency']:.1%}")

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
        import json
        import re

        # Try fresh data first (max_age=3600 = 1 hour)
        sensor_state = StateManager.load_state(namespace="sensors", max_age=3600)
        x_status = sensor_state.get("x_sentiment", {})

        is_stale = False
        stale_minutes = 0

        # CRITICAL FIX: Handle STALE string return from StateManager
        # When data is older than max_age, StateManager returns a string like:
        # "STALE (900min old) - {'sentiment_sensor_status': 'ONLINE', ...}"
        if isinstance(x_status, str) and x_status.startswith("STALE"):
            is_stale = True
            # Extract minutes from "STALE (XXXmin old)"
            match = re.search(r'STALE \((\d+)min old\)', x_status)
            if match:
                stale_minutes = int(match.group(1))

            # Extract the JSON payload after the " - "
            json_match = re.search(r' - ({.*})', x_status)
            if json_match:
                try:
                    # Python dict uses single quotes, JSON needs double quotes
                    x_status = json.loads(json_match.group(1).replace("'", '"'))
                except json.JSONDecodeError:
                    # Fallback: Load raw state directly
                    raw_state = StateManager._load_raw_sync()
                    x_entry = raw_state.get("sensors", {}).get("x_sentiment", {})
                    x_status = x_entry.get("data", {}) if isinstance(x_entry, dict) else {}

        # Now x_status should be a dict (or empty)
        if isinstance(x_status, dict) and x_status:
            status = x_status.get("sentiment_sensor_status", "UNKNOWN")
            failures = x_status.get("consecutive_failures", 0)
            degraded_until = x_status.get("degraded_until")

            # Determine icon based on status AND staleness
            if is_stale:
                status_icon = "🟡"
                status_display = f"{status} (data {stale_minutes}m old)"
            else:
                status_icon = "🟢" if status == "ONLINE" else "🔴" if status == "OFFLINE" else "🟡"
                status_display = status

            st.write(f"{status_icon} Status: **{status_display}**")

            if failures > 0:
                st.write(f"⚠️ Consecutive failures: {failures}")

            if degraded_until:
                st.write(f"⏰ Degraded until: {degraded_until}")
        else:
            # Truly not initialized - no data ever saved
            st.write("⚪ Status: Not initialized")
            st.caption("X Sentiment Sentinel has not run yet")

    except Exception as e:
        st.write(f"❓ Unable to fetch X sensor status: {e}")

st.markdown("---")

# === SECTION 2: Financial HUD ===
if config:
    live_data = fetch_all_live_data(config)
    benchmarks = fetch_todays_benchmark_data()

    # Render Portfolio Risk
    render_portfolio_risk_summary(live_data)

    # Render Benchmarks
    st.caption("Market Benchmarks")
    bench_cols = st.columns(6)
    with bench_cols[0]:
        st.metric("S&P 500", f"{benchmarks.get('SPY', 0):+.2f}%")
    with bench_cols[1]:
        # Dynamic commodity from config
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        name = config.get('commodity', {}).get('name', 'Coffee')
        yf_ticker = f"{ticker}=F"
        st.metric(name, f"{benchmarks.get(yf_ticker, 0):+.2f}%")

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

# === SECTION: Prediction Markets (Macro Radar) ===
with st.expander("🔮 Prediction Markets (Macro Radar)", expanded=True):
    render_prediction_markets()

st.markdown("---")

# === SECTION: Active Position Theses ===
st.subheader("📋 Active Position Theses")
st.caption("Real-time view of open positions and their trading rationales")

# Regime Status Banner
current_regime = get_current_market_regime()
regime_display = {
    'HIGH_VOLATILITY': ('🔴', 'High Volatility'),
    'RANGE_BOUND': ('🟡', 'Range Bound'),
    'TRENDING_UP': ('🟢', 'Trending Up'),
    'TRENDING_DOWN': ('🔻', 'Trending Down'),
    'UNKNOWN': ('⚪', 'Unknown')
}
icon, label = regime_display.get(current_regime, ('⚪', current_regime))
st.write(f"{icon} Current Market Regime: **{label}**")

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
        render_thesis_card_enhanced(thesis, live_data, config)

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
    if st.button("🛑 EMERGENCY HALT", type="primary", width="stretch"):
        if config:
            with st.spinner("Cancelling all open orders..."):
                try:
                    from trading_bot.order_manager import cancel_all_open_orders

                    # Create a new loop if needed or run in existing
                    try:
                        asyncio.run(cancel_all_open_orders(config, connection_purpose="dashboard_orders"))
                    except RuntimeError:
                        # If loop is already running (e.g. streamlit quirk)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(cancel_all_open_orders(config, connection_purpose="dashboard_orders"))

                    st.success("All open orders cancelled.")
                except Exception as e:
                    st.error(f"Failed to cancel orders: {e}")
        else:
            st.error("Config not loaded")

with ctrl_cols[2]:
    if st.button("🔄 Refresh All Data", width="stretch"):
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
