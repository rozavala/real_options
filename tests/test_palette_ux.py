import ast
import os
import unittest

class TestPaletteUX(unittest.TestCase):
    def test_portfolio_risk_tooltips(self):
        """
        Verify that render_portfolio_risk_summary in pages/1_Cockpit.py
        has help tooltips for 'Open Positions' and 'Daily P&L' metrics.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '1_Cockpit.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        render_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'render_portfolio_risk_summary':
                render_func = node
                break

        self.assertIsNotNone(render_func, "render_portfolio_risk_summary function not found")

        # We need to find ALL occurrences and ensure they have help
        open_pos_calls = []
        daily_pnl_calls = []

        for node in ast.walk(render_func):
            if isinstance(node, ast.Call):
                # Check for st.metric calls
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'metric':
                    # Check first argument (label)
                    label = None
                    if node.args and isinstance(node.args[0], ast.Constant):
                        label = node.args[0].value
                    elif node.args and isinstance(node.args[0], ast.Str): # Python < 3.8 support
                        label = node.args[0].s

                    if label == "Open Positions":
                        has_help = any(k.arg == 'help' for k in node.keywords)
                        open_pos_calls.append(has_help)
                    elif label == "Daily P&L":
                        has_help = any(k.arg == 'help' for k in node.keywords)
                        daily_pnl_calls.append(has_help)

        self.assertTrue(len(open_pos_calls) > 0, "No 'Open Positions' metric found")
        self.assertTrue(all(open_pos_calls), "All 'Open Positions' metrics must have a help tooltip")

        self.assertTrue(len(daily_pnl_calls) > 0, "No 'Daily P&L' metric found")
        self.assertTrue(all(daily_pnl_calls), "All 'Daily P&L' metrics must have a help tooltip")

if __name__ == '__main__':
    unittest.main()
