"""
Page 5: Utilities

Purpose: System maintenance and operational tools for Mission Control.
"""

import streamlit as st
import subprocess
import os
import sys
import socket
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import get_config

st.set_page_config(layout="wide", page_title="Utilities | Coffee Bot")

st.title("🔧 Utilities")
st.caption("System maintenance and operational tools")

# Helper to detect environment
def get_current_environment():
    """Detect current environment from hostname or config."""
    hostname = socket.gethostname()
    # Production droplet typically has 'prod' in hostname or specific IP
    # Adjust this logic based on your actual naming convention
    if 'prod' in hostname.lower() or hostname == 'coffee-bot-prod':
        return 'prod'
    # Fallback to env var if set
    if os.getenv("COFFEE_BOT_ENV") == "prod":
        return "prod"
    return 'dev'

current_env = get_current_environment()

st.markdown("---")

# === SECTION 1: Log Collection ===
st.subheader("📥 Log Collection")
st.markdown("""
Collect and archive logs to the centralized logs branch for analysis and debugging.
This captures orchestrator logs, dashboard logs, state files, and trading data.
""")

col1, col2 = st.columns(2)

with col1:
    st.info(f"📍 Current Environment: **{current_env}**")

with col2:
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

# === SECTION 2: Equity Sync ===
st.subheader("💰 Equity Data Sync")
st.markdown("""
Synchronize equity data from IBKR Flex Query. This updates `data/daily_equity.csv`
with the official Net Asset Value history from your broker.
""")

if st.button("🔄 Sync Equity from IBKR"):
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
                # Clear cache so dashboard shows updated data
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

# === SECTION 3: Log Analysis Quick Actions ===
st.subheader("📊 Log Analysis")
st.markdown("Quick access to log analysis utilities for the current environment.")

# Row 1: Status and Analyze
row1_cols = st.columns(2)

with row1_cols[0]:
    if st.button("📋 View Status", use_container_width=True):
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
    if st.button(f"🔍 Analyze {current_env.upper()}", use_container_width=True):
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
    if st.button(f"🚨 Show Errors", use_container_width=True):
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
    if st.button(f"📈 Trading Performance", use_container_width=True):
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
    if st.button(f"🏥 System Health", use_container_width=True):
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

# === SECTION 4: Cache Management ===
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

# === SECTION 5: System Info ===
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
