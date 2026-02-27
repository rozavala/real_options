"""
Page 5: Utilities

Purpose: System maintenance and operational tools for Real Options.
"""

import streamlit as st
import subprocess
import os
import sys
import socket
import json
import asyncio
import traceback
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import get_config, _resolve_data_path
from trading_bot.weighted_voting import TriggerType

# Load config at module level
config = get_config()

st.set_page_config(layout="wide", page_title="Utilities | Real Options")

from _commodity_selector import selected_commodity
ticker = selected_commodity()

st.title("🔧 Utilities")
st.caption("System maintenance and operational tools")

# Helper to detect environment
def get_current_environment():
    """Detect current environment from hostname or config."""
    hostname = socket.gethostname()
    if 'prod' in hostname.lower():
        return 'prod'
    if os.getenv("TRADING_BOT_ENV") == "prod" or os.getenv("COFFEE_BOT_ENV") == "prod":
        return "prod"
    return 'dev'

current_env = get_current_environment()
st.info(f"📍 Current Environment: **{current_env}**")
st.markdown("---")

# ==============================================================================
# SECTION 1: Log Collection
# ==============================================================================
st.subheader("📥 Log Collection")
st.markdown("""
Collect and archive logs to the centralized logs branch for analysis and debugging.
This captures orchestrator logs, dashboard logs, state files, and trading data.
""")

confirm_collect = st.checkbox("I confirm I want to collect logs", key="confirm_collect")
if st.button(
    "🚀 Collect Logs",
    type="primary",
    disabled=not confirm_collect,
    help="Triggers the log collection script to archive system logs, state files, and trading data for analysis."
):
    with st.spinner(f"Collecting {current_env} logs..."):
        try:
            env = os.environ.copy()
            env["LOG_ENV_NAME"] = current_env

            result = subprocess.run(
                ["bash", "scripts/collect_logs.sh"],
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            if result.returncode == 0:
                st.success(f"✅ Successfully collected {current_env} logs!")
                with st.expander("View Output"):
                    st.code(result.stdout)
            else:
                st.error(f"❌ Log collection failed")
                with st.expander("View Error"):
                    st.code(result.stderr or result.stdout)

        except subprocess.TimeoutExpired:
            st.error("⏱️ Log collection timed out (120s)")
        except FileNotFoundError:
            st.error("❌ Script not found: scripts/collect_logs.sh")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# ==============================================================================
# SECTION 2: Log Analysis
# ==============================================================================
st.subheader("📊 Log Analysis")
st.markdown("Quick access to log analysis utilities for the current environment.")

# Row 1: Status and Analyze
row1_cols = st.columns(2)

with row1_cols[0]:
    if st.button("📋 View Status", width='stretch', help="Show current orchestrator status and recent activity."):
        with st.spinner("Fetching status..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "status"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

with row1_cols[1]:
    if st.button(f"🔍 Analyze {current_env.upper()}", width='stretch', help=f"Run log analysis for the {current_env} environment."):
        with st.spinner(f"Analyzing {current_env} environment..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "analyze", current_env],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

# Row 2: Errors with configurable hours
st.markdown("##### 🚨 Error Analysis")
error_cols = st.columns([2, 1])

with error_cols[0]:
    hours_option = st.selectbox(
        "Time Range",
        options=[6, 12, 24, 48, 72],
        index=2,  # Default to 24 hours
        format_func=lambda x: f"Last {x} hours",
        key="error_hours"
    )

with error_cols[1]:
    st.markdown("&nbsp;")  # Spacer
    if st.button(f"🚨 Show Errors", width='stretch', help=f"Filter logs for errors in the last {hours_option} hours."):
        with st.spinner(f"Finding errors from last {hours_option} hours..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "errors", current_env, str(hours_option)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                output = result.stdout or result.stderr
                if "No errors found" in output or not output.strip():
                    st.success(f"✅ No errors found in the last {hours_option} hours!")
                else:
                    st.code(output)
            except Exception as e:
                st.error(f"Error: {e}")

# Row 3: Additional utilities
st.markdown("##### 📈 Performance & Health")
perf_cols = st.columns(2)

with perf_cols[0]:
    if st.button(f"📈 Trading Performance", width='stretch', help="Show trading performance metrics from recent logs."):
        with st.spinner("Loading performance data..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "performance", current_env],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

with perf_cols[1]:
    if st.button(f"🏥 System Health", width='stretch', help="Check system health: memory, disk, processes, and service status."):
        with st.spinner("Checking system health..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "health", current_env],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")

# ==============================================================================
# SECTION 3: Manual Trading Operations
# ==============================================================================
st.subheader("🎯 Manual Trading Operations")
st.markdown("""
Manually trigger trading operations outside of the normal schedule.
**Use with caution** - these bypass normal market hours validation.
""")

manual_cols = st.columns(2)

with manual_cols[0]:
    st.warning("⚠️ **Generate & Execute Orders**")
    st.caption("Runs full order generation cycle: data pull → Council deliberation → signals → order placement")

    # Safety Interlock
    confirm_exec = st.checkbox("I confirm I want to execute live trades", key="confirm_exec_orders")
    confirm_text = st.text_input("Type 'EXECUTE' to confirm:", key="confirm_text_orders")

    is_authorized = confirm_exec and confirm_text == "EXECUTE"

    if st.button(
        "🚀 Force Generate & Execute Orders",
        type="primary",
        disabled=not is_authorized,
        help="Bypasses the normal schedule to run a full analysis and trade cycle immediately (Data Pull → Council → Signals → Orders). Requires 'EXECUTE' confirmation."
    ):
        if not config:
            st.error("❌ Config not loaded")
        else:
            with st.spinner("Running order generation and execution..."):
                try:
                    import sys
                    import os
                    # Ensure path is correct for imports
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from trading_bot.order_manager import generate_and_execute_orders

                    # === FLIGHT DIRECTOR MANDATE: Connection Cleanup ===
                    async def run_with_cleanup():
                        """Run order generation with guaranteed connection cleanup."""
                        try:
                            await generate_and_execute_orders(config, connection_purpose="dashboard_orders", trigger_type=TriggerType.MANUAL)
                        finally:
                            # Ensure pool connections are released before loop closes
                            try:
                                from trading_bot.connection_pool import IBConnectionPool
                                logger = logging.getLogger("Utilities")
                                await IBConnectionPool.release_all()
                                logger.info("Connection pool released successfully")
                            except Exception as e:
                                logging.getLogger("Utilities").error(f"Error releasing connection pool: {e}")

                    # Run the async function
                    try:
                        asyncio.run(run_with_cleanup())

                        # === POST-EXECUTION DIAGNOSTIC ===
                        try:
                            from trading_bot.order_manager import ORDER_QUEUE
                            queued_count = len(ORDER_QUEUE)

                            if queued_count == 0:
                                st.warning(
                                    "⚠️ Order generation completed but **0 orders were queued**. "
                                    "This typically means the Hard Freshness Gate or another safety "
                                    "guard blocked all contracts. Check the orchestrator log for details."
                                )

                                # Show quick diagnostic
                                from trading_bot.state_manager import StateManager
                                reports = StateManager.load_state_with_metadata()
                                stale_agents = [
                                    name for name, meta in reports.items()
                                    if isinstance(meta, dict) and meta.get('age_hours', 0) > 24
                                ]
                                if stale_agents:
                                    st.error(
                                        f"🔍 **Likely cause:** {len(stale_agents)} agents have stale data (>24h): "
                                        f"{', '.join(stale_agents)}"
                                    )
                            else:
                                st.success(f"✅ Order generation completed! {queued_count} orders queued.")

                        except Exception as diag_e:
                            st.success("✅ Order generation and execution completed!")
                            # Diagnostic failure should not prevent success message

                        st.info("💡 Check Cockpit page for new positions and Council page for decision details")

                    except RuntimeError:
                        # If loop is already running
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            loop.run_until_complete(run_with_cleanup())
                            st.success("✅ Order generation and execution completed!")
                            st.info("💡 Check Cockpit page for new positions and Council page for decision details")
                        finally:
                            loop.close()

                except Exception as e:
                    st.error(f"❌ Order generation failed: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

with manual_cols[1]:
    st.warning("⚠️ **Cancel All Open Orders**")
    st.caption("Immediately cancels all unfilled DAY orders in IB")

    confirm_cancel_all = st.checkbox("I confirm I want to CANCEL all open orders", key="confirm_cancel_all")
    if st.button("🛑 Cancel All Open Orders", disabled=not confirm_cancel_all):
        if not config:
            st.error("❌ Config not loaded")
        else:
            with st.spinner("Cancelling all open orders..."):
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from trading_bot.order_manager import cancel_all_open_orders

                    try:
                        asyncio.run(cancel_all_open_orders(config, connection_purpose="dashboard_orders"))
                        st.success("✅ All open orders cancelled!")
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(cancel_all_open_orders(config, connection_purpose="dashboard_orders"))
                        st.success("✅ All open orders cancelled!")

                except Exception as e:
                    st.error(f"❌ Failed to cancel orders: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

manual_cols2 = st.columns(2)

with manual_cols2[0]:
    st.warning("⚠️ **Close Stale Positions**")
    st.caption("Closes positions held longer than max_holding_days")

    confirm_close_stale = st.checkbox("I confirm I want to CLOSE stale positions", key="confirm_close_stale")
    if st.button("🔄 Force Close Stale Positions", disabled=not confirm_close_stale):
        if not config:
            st.error("❌ Config not loaded")
        else:
            with st.spinner("Closing stale positions..."):
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from trading_bot.order_manager import close_stale_positions

                    try:
                        asyncio.run(close_stale_positions(config, connection_purpose="dashboard_close"))
                        st.success("✅ Stale position closure completed!")
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(close_stale_positions(config, connection_purpose="dashboard_close"))
                        st.success("✅ Stale position closure completed!")

                except Exception as e:
                    st.error(f"❌ Failed to close positions: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

with manual_cols2[1]:
    st.info("ℹ️ **Sync Equity Data**")
    st.caption("Forces fresh equity sync from IB Flex Query")

    confirm_sync = st.checkbox("I confirm I want to force equity sync", key="confirm_sync")
    if st.button(
        "💰 Force Equity Sync",
        disabled=not confirm_sync,
        help="Manually triggers a fresh equity data pull from Interactive Brokers Flex Query reports."
    ):
        if not config:
            st.error("❌ Config not loaded")
        else:
            with st.spinner("Syncing equity data from IBKR..."):
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from equity_logger import sync_equity_from_flex

                    try:
                        asyncio.run(sync_equity_from_flex(config))
                        st.success("✅ Equity data synced successfully!")
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(sync_equity_from_flex(config))
                        st.success("✅ Equity data synced successfully!")

                except Exception as e:
                    st.error(f"❌ Equity sync failed: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

st.markdown("---")

# ==============================================================================
# SECTION 4: System Diagnostics
# ==============================================================================
st.subheader("🔍 System Diagnostics")

diag_cols = st.columns(3)

with diag_cols[0]:
    st.info("**IB Connection Test**")
    if st.button("🔌 Test IB Gateway Connection", help="Open a test connection to IB Gateway and verify connectivity."):
        if not config:
            st.error("❌ Config not loaded")
        else:
            with st.spinner("Testing IB connection..."):
                try:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from trading_bot.connection_pool import IBConnectionPool

                    async def test_connection():
                        try:
                            ib = await IBConnectionPool.get_connection("test_utilities", config)
                            if ib and ib.isConnected():
                                return True, "Connection successful"
                            return False, "Connection failed"
                        except Exception as e:
                            return False, str(e)
                        finally:
                            # FLIGHT DIRECTOR FIX: Guaranteed cleanup
                            # This prevents the "CLOSE-WAIT" zombie accumulation
                            await IBConnectionPool.release_connection("test_utilities")

                    try:
                        success, message = asyncio.run(test_connection())
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success, message = loop.run_until_complete(test_connection())

                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")

with diag_cols[1]:
    st.info("**Test Notifications**")
    if st.button("📱 Send Test Notification", help="Send a test push notification via Pushover to verify alerting works."):
        if not config:
            st.error("❌ Config not loaded")
        else:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from notifications import send_pushover_notification

                send_pushover_notification(
                    config.get('notifications', {}),
                    "Test Notification",
                    f"This is a test notification from Real Options Utilities at {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
                )
                st.success("✅ Test notification sent!")
            except Exception as e:
                st.error(f"❌ Failed to send notification: {str(e)}")

with diag_cols[2]:
    st.info("**Market Status**")
    if st.button("🕐 Check Market Status", help="Show current time in UTC/NY, trading day status, and market open/close."):
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from trading_bot.utils import is_market_open, is_trading_day
            import pytz

            utc_now = datetime.now(timezone.utc)
            ny_tz = pytz.timezone('America/New_York')
            ny_now = utc_now.astimezone(ny_tz)

            market_open = is_market_open()
            trading_day = is_trading_day()

            st.write(f"**UTC Time:** {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**NY Time:** {ny_now.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Is Trading Day:** {'✅ Yes' if trading_day else '❌ No'}")
            st.write(f"**Market Status:** {'🟢 OPEN' if market_open else '🔴 CLOSED'}")

            if not trading_day:
                st.warning("Today is a weekend or holiday")

        except Exception as e:
            st.error(f"❌ Failed to check market status: {str(e)}")

st.markdown("---")

# ==============================================================================
# SECTION 5: System Health Digest
# ==============================================================================
st.subheader("📋 System Health Digest")
st.markdown("""
Generate a comprehensive health assessment that synthesizes ~15 data files per commodity
into a single JSON report. Covers feedback loops, agent calibration, sentinel efficiency,
risk rails, decision traces, portfolio trends, and a composite health score.
**Read-only** — no IB connections, no LLM calls.
""")

digest_cols = st.columns([2, 1])

with digest_cols[0]:
    if st.button("📋 Generate Health Digest", type="primary", width='stretch',
                 help="Generates a comprehensive system health report from current data files (~5s). Safe to run anytime."):
        with st.spinner("Generating System Health Digest..."):
            try:
                from trading_bot.system_digest import generate_system_digest
                digest = generate_system_digest(config)

                if digest:
                    # Health score banner
                    health = digest.get('system_health_score', {})
                    overall = health.get('overall')
                    if overall is not None:
                        if overall >= 0.75:
                            st.success(f"System Health: **{overall:.2f}/1.00** — Healthy")
                        elif overall >= 0.50:
                            st.warning(f"System Health: **{overall:.2f}/1.00** — Degraded")
                        else:
                            st.error(f"System Health: **{overall:.2f}/1.00** — Critical")

                    # Health score components
                    components = health.get('components', {})
                    if components:
                        comp_cols = st.columns(4)
                        labels = {
                            'feedback_health': ('Feedback', '🔄'),
                            'prediction_accuracy': ('Prediction', '🎯'),
                            'execution_quality': ('Execution', '⚡'),
                            'sentinel_efficiency': ('Sentinels', '📡'),
                        }
                        for i, (key, (label, icon)) in enumerate(labels.items()):
                            with comp_cols[i]:
                                val = components.get(key)
                                st.metric(f"{icon} {label}", f"{val:.2f}" if val is not None else "N/A")

                    # Executive summary
                    st.info(f"**Summary:** {digest.get('executive_summary', 'N/A')}")

                    # Improvement opportunities
                    opportunities = digest.get('improvement_opportunities', [])
                    if opportunities:
                        high = [o for o in opportunities if o['priority'] == 'HIGH']
                        medium = [o for o in opportunities if o['priority'] == 'MEDIUM']
                        if high:
                            for o in high:
                                st.error(f"**HIGH** [{o['component']}] {o['observation']}")
                        if medium:
                            for o in medium:
                                st.warning(f"**MEDIUM** [{o['component']}] {o['observation']}")

                    # Per-commodity details
                    for t, block in digest.get('commodities', {}).items():
                        if block.get('status') == 'no_data_directory':
                            continue
                        with st.expander(f"**{t}** — Details"):
                            cog = block.get('cognitive_layer', {})
                            regime = block.get('regime_context', {}).get('regime', 'UNKNOWN')
                            st.write(f"**Decisions today:** {cog.get('decisions_today', 0)} | **Regime:** {regime}")

                            fb = block.get('feedback_loop', {})
                            st.write(f"**Feedback loop:** {fb.get('status', 'N/A')} (resolution rate: {fb.get('resolution_rate', 'N/A')})")

                            freshness = block.get('data_freshness', {})
                            st.write(f"**Data freshness:** {freshness.get('status', 'N/A')} ({freshness.get('stale_count', 0)} stale)")

                            traces = block.get('decision_traces', [])
                            if traces:
                                st.write(f"**Last {len(traces)} decisions:**")
                                for trace in traces:
                                    direction = trace.get('direction', '?')
                                    conf = trace.get('confidence')
                                    strategy = trace.get('strategy', '')
                                    conf_str = f" ({conf:.0%})" if conf else ""
                                    st.caption(f"  {trace.get('timestamp', '')[:16]} — {direction}{conf_str} → {strategy}")

                    # Full JSON in expander
                    with st.expander("📄 Full Digest JSON"):
                        st.json(digest)

                    st.caption(f"Digest ID: `{digest.get('digest_id', 'N/A')}` | Schema v{digest.get('schema_version', '?')}")
                else:
                    st.error("❌ Digest generation returned None — check orchestrator logs")

            except Exception as e:
                st.error(f"❌ Digest generation failed: {str(e)}")
                with st.expander("View Error Details"):
                    st.code(traceback.format_exc())

with digest_cols[1]:
    # Show last digest if available
    st.markdown("**Last Digest**")
    try:
        digest_path = _resolve_data_path("system_health_digest.json")
        if os.path.exists(digest_path):
            mtime = os.path.getmtime(digest_path)
            st.caption(f"🕒 {datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
            with open(digest_path, 'r') as f:
                last_digest = json.load(f)
            score = last_digest.get('system_health_score', {}).get('overall')
            if score is not None:
                st.metric("Health Score", f"{score:.2f}")
            opps = last_digest.get('improvement_opportunities', [])
            high_count = sum(1 for o in opps if o.get('priority') == 'HIGH')
            st.metric("Issues", f"{high_count} HIGH / {len(opps)} total")
        else:
            st.caption("No previous digest found")
    except Exception:
        st.caption("Could not load last digest")

st.markdown("---")

# ==============================================================================
# SECTION 6: State Management (renumbered from 5)
# ==============================================================================
st.subheader("📁 State Management")

state_cols = st.columns(2)

with state_cols[0]:
    st.info("**View System State**")
    if st.button("👁️ Show state.json Contents", help="Display the current orchestrator state file (sentinels, triggers, flags)."):
        try:
            state_path = _resolve_data_path("state.json")
            if os.path.exists(state_path):
                mtime = os.path.getmtime(state_path)
                st.caption(f"🕒 Last updated: {datetime.fromtimestamp(mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                    st.json(state_data)
            else:
                st.warning("⚠️ state.json file not found")
        except Exception as e:
            st.error(f"❌ Failed to load state.json: {str(e)}")

with state_cols[1]:
    st.warning("**Clear State File**")
    st.caption("⚠️ This will reset all state data (sentinels, triggers, etc.)")

    # Use a form with confirmation checkbox to prevent accidental clicks
    with st.form("clear_state_form"):
        confirm_clear = st.checkbox("I understand this will clear all state data")
        submit_clear = st.form_submit_button("🗑️ Clear State File")

        if submit_clear:
            if not confirm_clear:
                st.error("❌ Please confirm by checking the box")
            else:
                try:
                    state_path = _resolve_data_path("state.json")
                    if os.path.exists(state_path):
                        # Create backup first
                        backup_path = f"{state_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        import shutil
                        shutil.copy2(state_path, backup_path)

                        # Clear state file
                        with open(state_path, 'w') as f:
                            json.dump({}, f, indent=2)

                        st.success(f"✅ State file cleared! Backup saved to: {os.path.basename(backup_path)}")
                    else:
                        st.warning("⚠️ No state.json file to clear")
                except Exception as e:
                    st.error(f"❌ Failed to clear state: {str(e)}")

st.markdown("---")

# ==============================================================================
# SECTION 7: System Validation
# ==============================================================================
st.subheader("🔍 System Validation")
st.markdown("""
Run comprehensive preflight checks to verify all system components are operational.
This validates the entire architecture from sentinels to council to order execution.
""")

validation_cols = st.columns([2, 1])

with validation_cols[0]:
    confirm_val = st.checkbox("I confirm I want to run system validation", key="confirm_val")
    run_validation = st.button("🚀 Run System Validation", type="primary", width='stretch', disabled=not confirm_val, help="Run preflight checks on all system components (~30s in quick mode, ~2min full).")

with validation_cols[1]:
    json_output = st.checkbox("JSON Output", value=False)
    quick_mode = st.checkbox("Quick Mode (Skip slow tests)", value=True)

if run_validation:
    with st.spinner("Running comprehensive system validation..."):
        try:
            cmd = [sys.executable, "verify_system_readiness.py"]

            if json_output:
                cmd.append("--json")
            else:
                cmd.append("--verbose")

            if quick_mode:
                cmd.append("--quick")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            # Parse exit code
            if result.returncode == 0:
                st.success("✅ All systems operational!")
            elif result.returncode == 2:
                st.warning("⚠️ System operational with warnings")
            else:
                st.error("❌ Critical issues detected")

            # Display output
            if json_output:
                try:
                    # Robust JSON parsing (ignore preceding warnings)
                    output_str = result.stdout.strip()
                    json_start = output_str.find("{")
                    json_end = output_str.rfind("}") + 1

                    if json_start >= 0 and json_end > json_start:
                        data = json.loads(output_str[json_start:json_end])
                    else:
                        data = json.loads(output_str) # Fallback attempt

                    # Summary metrics
                    summary = data.get('summary', {})
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        total = summary.get('total', 0)
                        passed = summary.get('passed', 0)
                        health = (passed / total * 100) if total > 0 else 0
                        st.metric("Health", f"{health:.1f}%")
                    with metric_cols[1]:
                        st.metric("Passed", summary.get('passed', 0), delta=None)
                    with metric_cols[2]:
                        st.metric("Warnings", summary.get('warnings', 0), delta=None, delta_color="off")
                    with metric_cols[3]:
                        st.metric("Failures", summary.get('failed', 0), delta=None, delta_color="inverse")

                    # Results table
                    if data.get('checks'):
                        import pandas as pd
                        df = pd.DataFrame(data['checks'])

                        # Color-code status
                        def color_status(val):
                            colors = {
                                'PASS': 'background-color: #d4edda; color: #155724',
                                'WARN': 'background-color: #fff3cd; color: #856404',
                                'FAIL': 'background-color: #f8d7da; color: #721c24',
                                'SKIP': 'background-color: #e2e3e5; color: #383d41',
                                'INFO': 'background-color: #cce5ff; color: #004085',
                            }
                            return colors.get(val, '')

                        # Select relevant columns
                        display_df = df[['status', 'name', 'message', 'details', 'duration_ms']]

                        styled_df = display_df.style.map(color_status, subset=['status'])
                        st.dataframe(styled_df, width='stretch', hide_index=True)

                except json.JSONDecodeError:
                    st.warning("Could not parse JSON output. Raw output below:")
                    st.code(result.stdout)
            else:
                # Plain text output
                st.code(result.stdout or result.stderr, language=None)

        except subprocess.TimeoutExpired:
            st.error("⏱️ Validation timed out (120s)")
        except FileNotFoundError:
            st.error("❌ verify_system_readiness.py not found")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# ==============================================================================
# SECTION 8: Data Reconciliation
# ==============================================================================
st.subheader("🔄 Data Reconciliation")
st.markdown("""
Manually trigger reconciliation processes to verify data integrity across all systems.
These processes normally run automatically in the orchestrator but can be triggered
on-demand for debugging or immediate verification.
""")

# --- Last-run timestamps for each reconciliation process ---
import re as _re

_RECON_STATE_PATH = _resolve_data_path("reconciliation_runs.json")

def _load_recon_timestamps() -> dict:
    """Load last-run timestamps for reconciliation processes."""
    try:
        if os.path.exists(_RECON_STATE_PATH):
            with open(_RECON_STATE_PATH, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_recon_timestamp(process_name: str):
    """Record the current time as the last run for a reconciliation process."""
    try:
        data = _load_recon_timestamps()
        data[process_name] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(_RECON_STATE_PATH), exist_ok=True)
        with open(_RECON_STATE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _format_last_run(process_name: str) -> str:
    """Return a human-readable 'last run' string."""
    ts_data = _load_recon_timestamps()
    ts_str = ts_data.get(process_name)
    if not ts_str:
        return "Never run"
    try:
        ts = datetime.fromisoformat(ts_str)
        delta = datetime.now(timezone.utc) - ts
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        elif hours < 24:
            return f"{int(hours)}h ago"
        else:
            return f"{int(hours / 24)}d ago"
    except Exception:
        return ts_str

def _parse_recon_summary(raw_output: str) -> str:
    """Parse reconciliation subprocess output into a concise summary."""
    lines = raw_output.strip().splitlines() if raw_output else []
    if not lines:
        return "Completed (no output)"

    summaries = []
    # Look for common patterns in reconciliation output
    for line in lines:
        line_lower = line.lower()
        # Match "Updated N rows", "Resolved N predictions", "N discrepancies", etc.
        if any(kw in line_lower for kw in ['updated', 'resolved', 'backfilled', 'synced', 'processed', 'found', 'graded', 'wrote']):
            summaries.append(line.strip())
        elif _re.search(r'\d+\s+(row|record|prediction|trade|position|entr)', line_lower):
            summaries.append(line.strip())

    if summaries:
        return " | ".join(summaries[:3])  # Show up to 3 summary lines
    # Fallback: last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:120]
    return "Completed"

# Show last-run overview
recon_names = {
    'council_history': 'Council History',
    'trade_ledger': 'Trade Ledger',
    'positions': 'Active Positions',
    'brier': 'Brier Scores',
}
last_run_cols = st.columns(len(recon_names))
for i, (key, label) in enumerate(recon_names.items()):
    with last_run_cols[i]:
        st.caption(f"**{label}**")
        st.text(_format_last_run(key))

confirm_recon = st.checkbox("Unlock reconciliation tools", key="confirm_recon")

# Create a 2x2 grid for reconciliation buttons
recon_row1 = st.columns(2)
recon_row2 = st.columns(2)

# --- Council History Reconciliation ---
with recon_row1[0]:
    st.markdown("**📊 Council History**")
    st.caption("Backfill exit prices and P&L for closed positions")

    if st.button("🔄 Reconcile Council History", width='stretch', key="recon_council", help="Backfill exit prices and P&L from IB historical data (~2-3 min).", disabled=not confirm_recon):
        with st.spinner("Reconciling council history with market outcomes..."):
            try:
                result = subprocess.run(
                    [sys.executable, "backfill_council_history.py"],
                    capture_output=True,
                    text=True,
                    timeout=180,  # Council reconciliation can take time with IB calls
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                raw_output = (result.stdout + result.stderr).strip()
                if result.returncode == 0:
                    _save_recon_timestamp('council_history')
                    summary = _parse_recon_summary(raw_output)
                    st.success(f"Council history reconciled: {summary}")
                    st.cache_data.clear()
                    with st.expander("View Full Output"):
                        st.code(raw_output or "No output")
                else:
                    st.error("❌ Council history reconciliation failed")
                    with st.expander("View Error"):
                        st.code(raw_output or "No output")

            except subprocess.TimeoutExpired:
                st.error("Reconciliation timed out (180s)")
            except FileNotFoundError:
                st.error("backfill_council_history.py not found")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Trade Ledger Reconciliation ---
with recon_row1[1]:
    st.markdown("**📝 Trade Ledger**")
    st.caption("Compare local ledger with IB Flex Query reports")

    if st.button("🔄 Reconcile Trade Ledger", width='stretch', key="recon_trades", help="Compare local ledger with IB Flex Query reports (~1-2 min).", disabled=not confirm_recon):
        with st.spinner("Reconciling trade ledger with IB reports..."):
            try:
                result = subprocess.run(
                    [sys.executable, "reconcile_trades.py"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                raw_output = (result.stdout + result.stderr).strip()
                if result.returncode == 0:
                    _save_recon_timestamp('trade_ledger')
                    if "No discrepancies found" in raw_output:
                        st.success("Trade ledger is perfectly in sync!")
                    else:
                        summary = _parse_recon_summary(raw_output)
                        st.warning(f"Discrepancies found: {summary}")

                    with st.expander("View Full Output"):
                        st.code(raw_output or "No output")
                else:
                    st.error("❌ Trade reconciliation failed")
                    with st.expander("View Error"):
                        st.code(raw_output or "No output")

            except subprocess.TimeoutExpired:
                st.error("Reconciliation timed out (120s)")
            except FileNotFoundError:
                st.error("reconcile_trades.py not found")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Active Positions Reconciliation ---
with recon_row2[0]:
    st.markdown("**📍 Active Positions**")
    st.caption("Verify current positions against IB")

    if st.button("🔄 Reconcile Positions", width='stretch', key="recon_positions", help="Verify current IB positions match local calculations (~30s).", disabled=not confirm_recon):
        with st.spinner("Reconciling active positions..."):
            try:
                # Create a temporary script to run just the position reconciliation
                result = subprocess.run(
                    [sys.executable, "-c", """
import asyncio
import sys
import os
sys.path.append(os.path.abspath('.'))
from config_loader import load_config
from reconcile_trades import reconcile_active_positions

async def main():
    config = load_config()
    await reconcile_active_positions(config)

asyncio.run(main())
"""],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                raw_output = (result.stdout + result.stderr).strip()
                if result.returncode == 0:
                    _save_recon_timestamp('positions')
                    if "No discrepancies found" in raw_output:
                        st.success("Positions are in sync!")
                    else:
                        summary = _parse_recon_summary(raw_output)
                        st.warning(f"Position discrepancies: {summary}")

                    with st.expander("View Full Output"):
                        st.code(raw_output or "No output")
                else:
                    st.error("❌ Position reconciliation failed")
                    with st.expander("View Error"):
                        st.code(raw_output or "No output")

            except subprocess.TimeoutExpired:
                st.error("Reconciliation timed out (60s)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Equity Sync (reference to Section 3) ---
with recon_row2[1]:
    st.markdown("**💰 Equity History**")
    st.caption("Use **Force Equity Sync** in Manual Trading Operations above")

recon_row3 = st.columns(2)

# --- Brier Score Reconciliation ---
with recon_row3[0]:
    st.markdown("**🎯 Brier Score Reconciliation**")
    st.caption("Grade pending agent predictions against market outcomes")

    # Show pending count
    try:
        import pandas as pd
        structured_path = _resolve_data_path("agent_accuracy_structured.csv")
        if os.path.exists(structured_path):
            structured_df = pd.read_csv(structured_path)
            pending_count = (structured_df['actual'] == 'PENDING').sum() if 'actual' in structured_df.columns else 0
            if pending_count > 0:
                st.warning(f"**{pending_count}** predictions pending resolution")
            else:
                st.success("All predictions resolved")
    except Exception:
        pass

    if st.button("🔄 Reconcile Brier Scores", width='stretch', key="recon_brier", help="Grade pending predictions against market outcomes (~3-5 min).", disabled=not confirm_recon):
        with st.spinner("Running Brier reconciliation (council history + prediction grading)..."):
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/manual_brier_reconciliation.py"],
                    capture_output=True,
                    text=True,
                    timeout=300,  # Council history reconciliation needs IB historical data
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                raw_output = (result.stdout + result.stderr).strip()
                if result.returncode == 0:
                    _save_recon_timestamp('brier')
                    summary = _parse_recon_summary(raw_output)
                    st.success(f"Brier reconciliation complete: {summary}")
                    st.cache_data.clear()
                    with st.expander("View Full Output"):
                        st.code(raw_output or "No output")
                else:
                    st.error("❌ Brier reconciliation failed")
                    with st.expander("View Error"):
                        st.code(raw_output or "No output")

            except subprocess.TimeoutExpired:
                st.error("Reconciliation timed out (300s)")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")

# Info box with reconciliation details
with st.expander("ℹ️ About Reconciliation Processes"):
    st.markdown("""
    ### Reconciliation Types

    **Council History**: Backfills missing exit prices and P&L for positions that have closed.
    Only processes trades older than 27 hours to allow time for settlement.

    **Trade Ledger**: Compares the last 33 days of trades from IB Flex Queries with the local
    ledger to identify missing or superfluous entries. Writes discrepancy reports to
    `archive_ledger/` directory.

    **Active Positions**: Validates that current positions match between IB and local calculations.
    Excludes symbols traded in the last 24 hours to avoid timing issues.

    **Equity History**: Syncs the official Net Asset Value history (last 365 days) from IBKR
    to ensure `daily_equity.csv` matches broker records. Uses 17:00 NY time as the daily close.

    **Brier Scores**: Three-step process: (1) backfills `actual_trend_direction` in council history
    via IB historical prices, (2) resolves pending predictions in `agent_accuracy_structured.csv`
    by matching to reconciled council decisions, (3) syncs resolutions to `enhanced_brier.json`.
    Grades predictions once their calculated exit time has passed (same-day on Fridays).

    ### When to Use

    - **After system recovery**: Verify data integrity after downtime
    - **Before important decisions**: Ensure all data is current
    - **During debugging**: Identify data discrepancies manually
    - **Weekly verification**: Regular health checks of data quality

    ### Automatic Execution

    These reconciliation processes run automatically:
    - **Council History**: During end-of-day reconciliation (weekdays at 17:15 ET)
    - **Trade Ledger**: During end-of-day reconciliation (weekdays at 17:15 ET)
    - **Active Positions**: During end-of-day reconciliation (weekdays at 17:15 ET)
    - **Equity History**: During end-of-day reconciliation (weekdays at 17:15 ET)
    """)

st.markdown("---")

# ==============================================================================
# SECTION 9: Cache Management
# ==============================================================================
st.subheader("🗑️ Cache Management")
st.markdown("Clear cached data to force fresh data loads from sources.")

cache_cols = st.columns(2)

with cache_cols[0]:
    confirm_clear_cache = st.checkbox("I confirm I want to clear all cached data", key="confirm_clear_cache")
    if st.button(
        "🔄 Clear All Caches",
        disabled=not confirm_clear_cache,
        help="Clears all application cache. This will force fresh data re-fetching on next page visit, which may take a few seconds."
    ):
        st.cache_data.clear()
        st.success("✅ All caches cleared!")
        st.rerun()

with cache_cols[1]:
    st.info("💡 Clearing caches forces fresh data loads on next page visit.")

st.markdown("---")

# ==============================================================================
# SECTION 10: System Info
# ==============================================================================
st.subheader("ℹ️ System Information")

info_cols = st.columns(3)

with info_cols[0]:
    st.metric("Python Version", sys.version.split()[0], help="The version of the Python interpreter running this application.")

with info_cols[1]:
    import streamlit
    st.metric("Streamlit Version", streamlit.__version__, help="The version of the Streamlit framework used to build this dashboard.")

with info_cols[2]:
    st.metric("Current Time (UTC)", datetime.now(timezone.utc).strftime("%H:%M:%S"), help="Current system time in UTC. All bot schedules and log timestamps use UTC for consistency.")

# Display recent log files
st.markdown("### 📄 Recent Log Files")
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

if os.path.exists(logs_dir):
    log_files = []
    for f in os.listdir(logs_dir):
        filepath = os.path.join(logs_dir, f)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            log_files.append({
                "File": f,
                "Size": f"{stat.st_size / 1024:.1f} KB",
                "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            })

    if log_files:
        import pandas as pd
        st.dataframe(pd.DataFrame(log_files), hide_index=True)
    else:
        st.info("No log files found.")
else:
    st.warning("Logs directory not found.")

st.markdown("---")

# ==============================================================================
# SECTION 11: Sentinel Statistics
# ==============================================================================
st.subheader("🛡️ Sentinel Statistics")

try:
    from trading_bot.sentinel_stats import SENTINEL_STATS

    stats = SENTINEL_STATS.get_dashboard_stats()

    if stats:
        # Dynamically create columns (limit to 4 per row for layout sanity)
        num_stats = len(stats)
        rows = (num_stats + 3) // 4

        for r in range(rows):
            cols = st.columns(4)
            batch = list(stats.items())[r*4 : (r+1)*4]

            for idx, (name, data) in enumerate(batch):
                with cols[idx]:
                    st.metric(
                        label=name.replace('Sentinel', '').strip(),
                        value=f"{data['alerts_today']} today",
                        delta=f"{data['conversion_rate']:.0%} → trades",
                        help=f"**{name}** stats for today. 'Conversion rate' shows the percentage of alerts that were validated by the Council and resulted in a trade decision."
                    )
    else:
        st.info("No sentinel alerts recorded yet.")
except Exception as e:
    st.warning(f"Could not load sentinel stats: {e}")
