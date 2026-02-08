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
from dashboard_utils import get_config
from trading_bot.weighted_voting import TriggerType

# Load config at module level
config = get_config()

st.set_page_config(layout="wide", page_title="Utilities | Real Options")

st.title("🔧 Utilities")
st.caption("System maintenance and operational tools")

# Helper to detect environment
def get_current_environment():
    """Detect current environment from hostname or config."""
    hostname = socket.gethostname()
    if 'prod' in hostname.lower() or hostname == 'coffee-bot-prod':
        return 'prod'
    if os.getenv("COFFEE_BOT_ENV") == "prod":
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

if st.button("🚀 Collect Logs", type="primary"):
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
    if st.button("📋 View Status", width='stretch'):
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
    if st.button(f"🔍 Analyze {current_env.upper()}", width='stretch'):
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
    if st.button(f"🚨 Show Errors", width='stretch'):
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
    if st.button(f"📈 Trading Performance", width='stretch'):
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
    if st.button(f"🏥 System Health", width='stretch'):
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

    if st.button("🚀 Force Generate & Execute Orders", type="primary", disabled=not is_authorized):
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
                                print(f"Error releasing connection pool: {e}")

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

    if st.button("💰 Force Equity Sync"):
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
    if st.button("🔌 Test IB Gateway Connection"):
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
    if st.button("📱 Send Test Notification"):
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
    if st.button("🕐 Check Market Status"):
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
# SECTION 5: State Management
# ==============================================================================
st.subheader("📁 State Management")

state_cols = st.columns(2)

with state_cols[0]:
    st.info("**View System State**")
    if st.button("👁️ Show state.json Contents"):
        try:
            state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "state.json")
            if os.path.exists(state_path):
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
                    state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "state.json")
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
# SECTION 6: System Validation
# ==============================================================================
st.subheader("🔍 System Validation")
st.markdown("""
Run comprehensive preflight checks to verify all system components are operational.
This validates the entire architecture from sentinels to council to order execution.
""")

validation_cols = st.columns([2, 1])

with validation_cols[0]:
    run_validation = st.button("🚀 Run System Validation", type="primary", width='stretch')

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

                        styled_df = display_df.style.applymap(color_status, subset=['status'])
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
# SECTION 7: Data Reconciliation
# ==============================================================================
st.subheader("🔄 Data Reconciliation")
st.markdown("""
Manually trigger reconciliation processes to verify data integrity across all systems.
These processes normally run automatically in the orchestrator but can be triggered
on-demand for debugging or immediate verification.
""")

# Create a 2x2 grid for reconciliation buttons
recon_row1 = st.columns(2)
recon_row2 = st.columns(2)

# --- Council History Reconciliation ---
with recon_row1[0]:
    st.markdown("**📊 Council History**")
    st.caption("Backfill exit prices and P&L for closed positions")

    if st.button("🔄 Reconcile Council History", width='stretch', key="recon_council"):
        with st.spinner("Reconciling council history with market outcomes..."):
            try:
                result = subprocess.run(
                    [sys.executable, "backfill_council_history.py"],
                    capture_output=True,
                    text=True,
                    timeout=180,  # Council reconciliation can take time with IB calls
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                if result.returncode == 0:
                    st.success("✅ Council history reconciliation complete!")
                    # Clear cache to show updated data
                    st.cache_data.clear()
                    with st.expander("View Details"):
                        st.code(result.stdout)
                else:
                    st.error("❌ Council history reconciliation failed")
                    with st.expander("View Error"):
                        st.code(result.stderr or result.stdout)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Reconciliation timed out (180s)")
            except FileNotFoundError:
                st.error("❌ backfill_council_history.py not found")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# --- Trade Ledger Reconciliation ---
with recon_row1[1]:
    st.markdown("**📝 Trade Ledger**")
    st.caption("Compare local ledger with IB Flex Query reports")

    if st.button("🔄 Reconcile Trade Ledger", width='stretch', key="recon_trades"):
        with st.spinner("Reconciling trade ledger with IB reports..."):
            try:
                result = subprocess.run(
                    [sys.executable, "reconcile_trades.py"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                if result.returncode == 0:
                    # Check if discrepancies were found
                    output = result.stdout
                    if "No discrepancies found" in output:
                        st.success("✅ Trade ledger is perfectly in sync!")
                    else:
                        st.warning("⚠️ Discrepancies found - check archive_ledger/ directory")

                    with st.expander("View Details"):
                        st.code(output)
                else:
                    st.error("❌ Trade reconciliation failed")
                    with st.expander("View Error"):
                        st.code(result.stderr or result.stdout)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Reconciliation timed out (120s)")
            except FileNotFoundError:
                st.error("❌ reconcile_trades.py not found")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# --- Active Positions Reconciliation ---
with recon_row2[0]:
    st.markdown("**📍 Active Positions**")
    st.caption("Verify current positions against IB")

    if st.button("🔄 Reconcile Positions", width='stretch', key="recon_positions"):
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

                if result.returncode == 0:
                    output = result.stdout
                    if "No discrepancies found" in output:
                        st.success("✅ Positions are in sync!")
                    else:
                        st.warning("⚠️ Position discrepancies detected")

                    with st.expander("View Details"):
                        st.code(output)
                else:
                    st.error("❌ Position reconciliation failed")
                    with st.expander("View Error"):
                        st.code(result.stderr or result.stdout)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Reconciliation timed out (60s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# --- Equity Sync (Legacy/Subprocess) ---
with recon_row2[1]:
    st.markdown("**💰 Equity History (Subprocess)**")
    st.caption("Sync equity data from IBKR Flex Query (Legacy)")

    if st.button("🔄 Sync Equity Data", width='stretch', key="recon_equity"):
        with st.spinner("Syncing equity data from Flex Query..."):
            try:
                result = subprocess.run(
                    [sys.executable, "equity_logger.py", "--sync"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                if result.returncode == 0:
                    st.success("✅ Equity data synced successfully!")
                    st.cache_data.clear()
                    with st.expander("View Output"):
                        st.code(result.stdout)
                else:
                    st.error("❌ Sync failed")
                    with st.expander("View Error"):
                        st.code(result.stderr or result.stdout)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Sync timed out (60s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

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
# SECTION 8: Cache Management
# ==============================================================================
st.subheader("🗑️ Cache Management")
st.markdown("Clear cached data to force fresh data loads from sources.")

cache_cols = st.columns(2)

with cache_cols[0]:
    if st.button("🔄 Clear All Caches"):
        st.cache_data.clear()
        st.success("✅ All caches cleared!")
        st.rerun()

with cache_cols[1]:
    st.info("💡 Clearing caches forces fresh data loads on next page visit.")

st.markdown("---")

# ==============================================================================
# SECTION 9: System Info
# ==============================================================================
st.subheader("ℹ️ System Information")

info_cols = st.columns(3)

with info_cols[0]:
    st.metric("Python Version", sys.version.split()[0])

with info_cols[1]:
    import streamlit
    st.metric("Streamlit Version", streamlit.__version__)

with info_cols[2]:
    st.metric("Current Time (UTC)", datetime.now(timezone.utc).strftime("%H:%M:%S"))

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
# SECTION 10: Sentinel Statistics
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
                        delta=f"{data['conversion_rate']:.0%} → trades"
                    )
    else:
        st.info("No sentinel alerts recorded yet.")
except Exception as e:
    st.warning(f"Could not load sentinel stats: {e}")
