"""
Page 5: Utilities

Purpose: System maintenance and operational tools for Mission Control.
"""

import streamlit as st
import subprocess
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboard_utils import get_config

st.set_page_config(layout="wide", page_title="Utilities | Coffee Bot")

st.title("🔧 Utilities")
st.caption("System maintenance and operational tools")

st.markdown("---")

# === SECTION 1: Log Collection ===
st.subheader("📥 Log Collection")
st.markdown("""
Collect and archive logs to the centralized logs branch for analysis and debugging.
This captures orchestrator logs, dashboard logs, state files, and trading data.
""")

col1, col2 = st.columns(2)

with col1:
    env_option = st.selectbox(
        "Environment",
        options=["dev", "prod"],
        help="Select which environment to collect logs from"
    )

with col2:
    st.markdown("&nbsp;")  # Spacer for alignment
    if st.button("🚀 Collect Logs", type="primary"):
        with st.spinner(f"Collecting {env_option} logs..."):
            try:
                # Set environment variable and run script
                env = os.environ.copy()
                env["LOG_ENV_NAME"] = env_option

                result = subprocess.run(
                    ["bash", "scripts/collect_logs.sh"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

                if result.returncode == 0:
                    st.success(f"✅ Successfully collected {env_option} logs!")
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
st.markdown("Quick access to log analysis utilities.")

analysis_cols = st.columns(3)

with analysis_cols[0]:
    if st.button("📋 View Status"):
        with st.spinner("Fetching status..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "status"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

with analysis_cols[1]:
    if st.button("🔍 Analyze Dev"):
        with st.spinner("Analyzing dev environment..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "analyze", "dev"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                st.code(result.stdout or result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

with analysis_cols[2]:
    if st.button("🔍 Analyze Prod"):
        with st.spinner("Analyzing prod environment..."):
            try:
                result = subprocess.run(
                    ["bash", "scripts/log_analysis.sh", "analyze", "prod"],
                    capture_output=True,
                    text=True,
                    timeout=30,
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
    st.metric("Current Time (UTC)", datetime.utcnow().strftime("%H:%M:%S"))

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
