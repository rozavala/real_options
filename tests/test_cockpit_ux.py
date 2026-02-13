
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit before any imports that use it
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st
# Mock st.cache_data decorator
class MockCacheData:
    def __call__(self, func=None, ttl=None):
        return (lambda f: f) if func is None else func
    def clear(self):
        pass
mock_st.cache_data = MockCacheData()

# Mock st.columns to return a list of mocks
def mock_columns(spec):
    if isinstance(spec, int):
        count = spec
    else:
        count = len(spec)
    return [MagicMock() for _ in range(count)]
mock_st.columns = MagicMock(side_effect=mock_columns)

# Mock st.expander context manager
mock_st.expander = MagicMock()
mock_st.expander.return_value.__enter__ = MagicMock()
mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)
# Mock st.container context manager
mock_st.container = MagicMock()
mock_st.container.return_value.__enter__ = MagicMock()
mock_st.container.return_value.__exit__ = MagicMock(return_value=None)
# Mock st.popover context manager (if used)
if not hasattr(mock_st, 'popover'):
    mock_st.popover = MagicMock()
    mock_st.popover.return_value.__enter__ = MagicMock()
    mock_st.popover.return_value.__exit__ = MagicMock(return_value=None)

# Mock other dependencies
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.express'] = MagicMock()
sys.modules['ib_insync'] = MagicMock()

# Mock dashboard_utils functions to avoid side effects
mock_du = MagicMock()
sys.modules['dashboard_utils'] = mock_du
from dashboard_utils import get_config, get_commodity_profile  # noqa: E402, F401

# Configure mock return values for top-level calls
mock_du.get_system_heartbeat.return_value = {
    'orchestrator_status': 'ONLINE', 'orchestrator_last_pulse': None,
    'state_status': 'ONLINE', 'state_last_pulse': None
}
mock_du.get_sentinel_status.return_value = {}
mock_du.get_ib_connection_health.return_value = {
    'sentinel_ib': 'CONNECTED', 'micro_ib': 'CONNECTED',
    'last_successful_connection': None, 'reconnect_backoff': 0
}
mock_du.load_deduplicator_metrics.return_value = {}
mock_du.load_task_schedule_status.return_value = {'available': False}
mock_du.fetch_all_live_data.return_value = {
    'net_liquidation': 50000.0,
    'maint_margin': 10000.0,
    'daily_pnl': 0.0,
    'open_positions': [],
    'portfolio_items': []
}
mock_du.get_active_theses.return_value = []
mock_du.fetch_todays_benchmark_data.return_value = {}
mock_du.get_commodity_profile.return_value = {
    'stop_parse_range': [80, 800],
    'typical_price_range': [100, 600]
}

# Helper to load the cockpit module
def load_cockpit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("Cockpit", "pages/1_Cockpit.py")
    cockpit = importlib.util.module_from_spec(spec)
    sys.modules["Cockpit"] = cockpit

    # Patch env vars to avoid warnings or side effects
    with patch.dict(os.environ, {"TRADING_MODE": "LIVE", "INITIAL_CAPITAL": "50000"}):
        spec.loader.exec_module(cockpit)

    return cockpit

class TestCockpitUX:
    def setup_method(self):
        self.cockpit = load_cockpit_module()
        # Reset mocks after module load triggers initial calls
        mock_st.reset_mock()

    def test_render_thesis_card_enhanced_pnl_coloring(self):
        """Test P&L coloring in thesis card."""
        # Setup mock data
        thesis = {
            'position_id': 'KOK6_C3275',
            'strategy_type': 'BULL_CALL_SPREAD',
            'entry_price': 10.0,
            'entry_timestamp': datetime(2024, 1, 1),
            'invalidation_triggers': [],
            'display_name': 'BCS KOK6'
        }

        # Mock portfolio item
        mock_item = MagicMock()
        # Ensure localSymbol is a string
        contract_mock = MagicMock()
        contract_mock.localSymbol = 'KOK6_C3275'
        mock_item.contract = contract_mock

        mock_item.unrealizedPNL = 500.0
        mock_item.marketValue = 1500.0

        live_data = {
            'portfolio_items': [mock_item]
        }

        # Call function
        self.cockpit.render_thesis_card_enhanced(thesis, live_data)

        # Assertions
        # Verify st.metric calls
        # We expect one for Entry, one for Unrealized P&L, one for Stop Price, one for Distance

        # Find the call for "Unrealized P&L"
        # args[0] is label, args[1] is value
        pnl_calls = [
            call for call in mock_st.metric.call_args_list
            if call.args and call.args[0] == "Unrealized P&L"
        ]

        if len(pnl_calls) != 1:
            print(f"DEBUG: All metric calls: {mock_st.metric.call_args_list}")

        assert len(pnl_calls) == 1, "Should have exactly one metric call for Unrealized P&L"

        args, kwargs = pnl_calls[0]
        value = args[1]

        # NEW BEHAVIOR CHECK:
        # Expect: st.metric("Unrealized P&L", "$+500.00", "+500.00", help=...)

        assert value == "$+500.00"

        # Delta should be present as 3rd arg or kwargs
        if len(args) > 2:
            delta = args[2]
        else:
            delta = kwargs.get('delta')

        assert delta == "+500.00", f"Expected delta '+500.00', got {delta}"

        # Check help text for Market Value
        help_text = kwargs.get('help', '')
        assert "Market Value: $1,500.00" in help_text, "Help text missing Market Value"

    def test_render_portfolio_risk_summary_pnl_coloring(self):
        """Test Daily P&L coloring in risk summary."""
        live_data = {
            'net_liquidation': 50000.0,
            'maint_margin': 10000.0,
            'daily_pnl': -250.0,
            'open_positions': []
        }

        self.cockpit.render_portfolio_risk_summary(live_data)

        pnl_calls = [
            call for call in mock_st.metric.call_args_list
            if call.args and call.args[0] == "Daily P&L"
        ]

        assert len(pnl_calls) == 1
        args, kwargs = pnl_calls[0]

        # New behavior: value includes sign, AND delta matches it
        assert args[1] == "$-250" # formatted with +,.0f

        # Verify delta IS set
        if len(args) > 2:
            delta = args[2]
        else:
            delta = kwargs.get('delta')

        assert delta == "-250", f"Expected delta '-250', got {delta}"

if __name__ == "__main__":
    pytest.main([__file__])
