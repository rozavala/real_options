"""Shared commodity selector for dashboard pages 1-7.

Renders a sidebar selectbox for commodity switching. On change:
1. Updates st.session_state['commodity_ticker']
2. Updates os.environ['COMMODITY_TICKER']
3. Re-initializes module-level data directories
4. Clears Streamlit data cache
5. Triggers st.rerun()
"""

import streamlit as st
import os


_COMMODITY_LABELS = {
    "KC": "\u2615 KC (Coffee)",
    "CC": "\U0001f36b CC (Cocoa)",
    "NG": "\U0001f525 NG (Natural Gas)",
    "SB": "\U0001f36c SB (Sugar)",
}


def _reinit_data_dirs(ticker: str):
    """Re-point module-level data directories for the selected commodity."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', ticker)

    from trading_bot.decision_signals import set_data_dir as set_signals_dir
    set_signals_dir(data_dir)

    from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
    set_brier_bridge_dir(data_dir)


def selected_commodity() -> str:
    """Render commodity selector in sidebar and return the active ticker.

    Call this at the top of each page (after st.set_page_config).
    """
    # Ensure session state is initialized
    if 'commodity_ticker' not in st.session_state:
        st.session_state['commodity_ticker'] = os.environ.get('COMMODITY_TICKER', 'KC')

    current = st.session_state['commodity_ticker']

    # Sync env var from session state (handles cross-page navigation)
    if os.environ.get('COMMODITY_TICKER') != current:
        os.environ['COMMODITY_TICKER'] = current
        _reinit_data_dirs(current)

    # Discover which commodities have data directories
    from dashboard_utils import discover_active_commodities
    available = discover_active_commodities()

    selected = st.sidebar.selectbox(
        "Commodity",
        available,
        index=available.index(current) if current in available else 0,
        format_func=lambda x: _COMMODITY_LABELS.get(x, x),
    )

    if selected != current:
        st.session_state['commodity_ticker'] = selected
        os.environ['COMMODITY_TICKER'] = selected
        _reinit_data_dirs(selected)
        st.cache_data.clear()
        st.rerun()

    return selected
