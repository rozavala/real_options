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

from config.commodity_profiles import CommodityType


# Emoji by commodity type — no per-ticker maintenance needed
_TYPE_EMOJI = {
    CommodityType.SOFT: "\u2615",      # coffee cup (generic soft)
    CommodityType.ENERGY: "\U0001f525", # fire
    CommodityType.METAL: "\U0001fab6",  # rock
    CommodityType.GRAIN: "\U0001f33e",  # rice/grain
}

# Optional per-ticker emoji overrides (only when the type default isn't ideal)
_TICKER_EMOJI = {
    "CC": "\U0001f36b",  # chocolate bar
    "SB": "\U0001f36c",  # candy
}


def _commodity_label(ticker: str) -> str:
    """Build a display label like '☕ KC (Coffee Arabica)' from CommodityProfile."""
    try:
        from config.commodity_profiles import get_commodity_profile
        profile = get_commodity_profile(ticker)
        emoji = _TICKER_EMOJI.get(ticker, _TYPE_EMOJI.get(profile.commodity_type, "\U0001f4ca"))
        return f"{emoji} {ticker} ({profile.name})"
    except Exception:
        return ticker


def _reinit_data_dirs(ticker: str):
    """Re-point module-level data directories for the selected commodity.

    Must cover every module with a set_data_dir() that the dashboard reads.
    Mirrors the orchestrator's init sequence (orchestrator.py L4926-4938).
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', ticker)

    from trading_bot.decision_signals import set_data_dir as set_signals_dir
    set_signals_dir(data_dir)

    from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
    set_brier_bridge_dir(data_dir)

    from trading_bot.state_manager import StateManager
    StateManager.set_data_dir(data_dir)

    from trading_bot.sentinel_stats import set_data_dir as set_stats_dir
    set_stats_dir(data_dir)

    from trading_bot.utils import set_data_dir as set_utils_dir
    set_utils_dir(data_dir)

    from trading_bot.brier_scoring import set_data_dir as set_brier_scoring_dir
    set_brier_scoring_dir(data_dir)

    from trading_bot.prompt_trace import set_data_dir as set_prompt_trace_dir
    set_prompt_trace_dir(data_dir)

    from trading_bot.router_metrics import set_data_dir as set_router_metrics_dir
    set_router_metrics_dir(data_dir)

    from trading_bot.task_tracker import set_data_dir as set_tracker_dir
    set_tracker_dir(data_dir)


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
        format_func=_commodity_label,
    )

    if selected != current:
        st.session_state['commodity_ticker'] = selected
        os.environ['COMMODITY_TICKER'] = selected
        _reinit_data_dirs(selected)
        st.cache_data.clear()
        st.rerun()

    return selected
