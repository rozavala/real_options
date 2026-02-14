
import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit before any imports that use it
# We need to use patch.dict on sys.modules to safely mock modules only for this test
# without polluting the global namespace for other tests.

@pytest.fixture
def mock_streamlit():
    """Safely mocks streamlit and other dependencies for the duration of the test."""
    with patch.dict(sys.modules):
        mock_st = MagicMock()
        sys.modules['streamlit'] = mock_st

        # Mock st.cache_data decorator
        class MockCacheData:
            def __call__(self, func=None, ttl=None):
                return (lambda f: f) if func is None else func
            def clear(self):
                pass
        mock_st.cache_data = MockCacheData()

        # Mock UI components
        def mock_columns(spec):
            count = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(count)]
        mock_st.columns = MagicMock(side_effect=mock_columns)

        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)

        mock_st.container.return_value.__enter__ = MagicMock()
        mock_st.container.return_value.__exit__ = MagicMock(return_value=None)

        if not hasattr(mock_st, 'popover'):
            mock_st.popover = MagicMock()
            mock_st.popover.return_value.__enter__ = MagicMock()
            mock_st.popover.return_value.__exit__ = MagicMock(return_value=None)

        # Mock dependencies
        sys.modules['plotly'] = MagicMock()
        sys.modules['plotly.graph_objects'] = MagicMock()
        sys.modules['plotly.express'] = MagicMock()
        sys.modules['ib_insync'] = MagicMock()

        # Mock dashboard_utils
        mock_du = MagicMock()
        sys.modules['dashboard_utils'] = mock_du

        # Configure default mock returns
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

        yield mock_st

def load_cockpit_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("Cockpit", "pages/1_Cockpit.py")
    cockpit = importlib.util.module_from_spec(spec)

    # Patch env vars to avoid warnings or side effects during module load
    with patch.dict(os.environ, {"TRADING_MODE": "LIVE", "INITIAL_CAPITAL": "50000"}):
        spec.loader.exec_module(cockpit)

    return cockpit

class TestCockpitUX:

    def test_render_thesis_card_enhanced_pnl_coloring(self, mock_streamlit):
        """Test P&L coloring in thesis card."""
        # Load module inside the test context where mocks are active
        cockpit = load_cockpit_module()

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
        contract_mock = MagicMock()
        contract_mock.localSymbol = 'KOK6_C3275'
        mock_item.contract = contract_mock
        mock_item.unrealizedPNL = 500.0
        mock_item.marketValue = 1500.0

        live_data = {
            'portfolio_items': [mock_item]
        }

        # Call function
        cockpit.render_thesis_card_enhanced(thesis, live_data)

        # Assertions
        # We expect a call to st.markdown for the label "**Unrealized P&L**"
        # And another for the colored value

        # Find calls that contain the formatted P&L string
        markdown_calls = [
            call for call in mock_streamlit.markdown.call_args_list
            if call.args and "$+500.00" in call.args[0]
        ]

        assert len(markdown_calls) == 1, "Should have exactly one markdown call for Unrealized P&L value"

        args, kwargs = markdown_calls[0]
        content = args[0]

        # Check for color coding (green for positive)
        # :green[$+500.00]
        assert ":green" in content, f"Markdown content should apply green color: {content}"

        # Check help tooltip
        help_text = kwargs.get('help', '')
        assert "Market Value: $1,500.00" in help_text, "Help text missing Market Value"

    def test_render_portfolio_risk_summary_pnl_coloring(self, mock_streamlit):
        """Test Daily P&L coloring in risk summary using delta."""
        cockpit = load_cockpit_module()

        live_data = {
            'net_liquidation': 50000.0,
            'maint_margin': 10000.0,
            'daily_pnl': -250.0,
            'open_positions': []
        }

        cockpit.render_portfolio_risk_summary(live_data)

        pnl_calls = [
            call for call in mock_streamlit.metric.call_args_list
            if call.args and call.args[0] == "Daily P&L"
        ]

        # If dashboard runs some code on import that calls this, we might have multiple calls.
        # But here we just called it once.
        # The failure log shows: [call('Daily P&L', '$+0', '+0'), call('Daily P&L', '$-250', '-250')]
        # This implies load_cockpit_module() might be running the script body which calls render_portfolio_risk_summary
        # with default (0) values, and then we call it again.
        # We should check the LAST call.

        assert len(pnl_calls) >= 1
        args, kwargs = pnl_calls[-1]

        # Check value
        assert args[1] == "$-250"

        # Check delta matches value (since it's a daily change metric)
        if len(args) > 2:
            delta = args[2]
        else:
            delta = kwargs.get('delta')

        assert delta == "-250", f"Expected delta '-250', got {delta}"

if __name__ == "__main__":
    pytest.main([__file__])
