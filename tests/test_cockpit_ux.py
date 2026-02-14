
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCockpitUX:

    @pytest.fixture
    def mock_streamlit(self):
        """Safely mocks streamlit metrics for isolated testing."""
        with patch.dict(sys.modules):
            mock_st = MagicMock()
            sys.modules['streamlit'] = mock_st
            yield mock_st

    def test_pnl_coloring_logic(self, mock_streamlit):
        """
        Verify that _render_colored_pnl helper applies correct markdown color syntax.
        Importing logic from script dynamically to avoid fragility of full module import.
        """
        # Dynamically import ONLY the helper function we need to test
        # We read the file content and exec just the function definition
        with open("pages/1_Cockpit.py", "r") as f:
            content = f.read()

        # Extract the helper function code block
        import ast
        tree = ast.parse(content)
        func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_render_colored_pnl")

        # Compile and exec the function definition in a local scope
        code = compile(ast.Module(body=[func_node], type_ignores=[]), filename="<string>", mode="exec")
        local_scope = {'st': mock_streamlit} # Inject mocked streamlit
        exec(code, local_scope)
        _render_colored_pnl = local_scope["_render_colored_pnl"]

        # Test Positive P&L (Green)
        _render_colored_pnl("Test P&L", 500.0)
        markdown_calls = mock_streamlit.markdown.call_args_list
        assert any(":green" in call.args[0] for call in markdown_calls if call.args), "Positive P&L should be green"
        assert any("$+500.00" in call.args[0] for call in markdown_calls if call.args), "Should format with + sign"

        # Test Negative P&L (Red)
        mock_streamlit.reset_mock()
        _render_colored_pnl("Test P&L", -250.0)
        markdown_calls = mock_streamlit.markdown.call_args_list
        assert any(":red" in call.args[0] for call in markdown_calls if call.args), "Negative P&L should be red"
        assert any("$-250.00" in call.args[0] for call in markdown_calls if call.args), "Should format with - sign"

        # Test Zero P&L (Green - usually non-negative is standard, or check logic)
        # Logic is: color = "green" if value >= 0 else "red"
        mock_streamlit.reset_mock()
        _render_colored_pnl("Test P&L", 0.0)
        markdown_calls = mock_streamlit.markdown.call_args_list
        assert any(":green" in call.args[0] for call in markdown_calls if call.args), "Zero P&L should be green"

if __name__ == "__main__":
    pytest.main([__file__])
