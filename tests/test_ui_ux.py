import ast
import os
import unittest

class TestCockpitUX(unittest.TestCase):
    def test_sentinel_row_progressive_enhancement(self):
        """
        Verify that _render_sentinel_row in pages/1_Cockpit.py uses
        progressive enhancement (st.popover) for error details.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '1_Cockpit.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        # Find the _render_sentinel_row function
        render_func = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_render_sentinel_row':
                render_func = node
                break

        self.assertIsNotNone(render_func, "_render_sentinel_row function not found")

        # Look for the error handling block
        # We expect:
        # if error:
        #     if hasattr(st, "popover"): ...

        found_progressive_enhancement = False

        for node in ast.walk(render_func):
            if isinstance(node, ast.If):
                # Check if it's the "if error:" block (simplified check)
                # In AST, we check if the test is a Name 'error'
                if isinstance(node.test, ast.Name) and node.test.id == 'error':
                    # Check the first statement in the body
                    if node.body and isinstance(node.body[0], ast.If):
                        inner_if = node.body[0]
                        # Check "if hasattr(st, 'popover'):"
                        if (isinstance(inner_if.test, ast.Call) and
                            isinstance(inner_if.test.func, ast.Name) and
                            inner_if.test.func.id == 'hasattr'):

                            args = inner_if.test.args
                            if (len(args) == 2 and
                                isinstance(args[0], ast.Name) and args[0].id == 'st' and
                                isinstance(args[1], ast.Constant) and args[1].value == 'popover'):

                                found_progressive_enhancement = True

                                # Verify True branch uses popover
                                has_popover_call = False
                                for sub in ast.walk(inner_if.body[0]):
                                    if isinstance(sub, ast.Attribute) and sub.attr == 'popover':
                                        has_popover_call = True
                                self.assertTrue(has_popover_call, "True branch should use st.popover")

                                # Verify False branch uses expander
                                has_expander_call = False
                                if inner_if.orelse:
                                    for sub in ast.walk(inner_if.orelse[0]):
                                        if isinstance(sub, ast.Attribute) and sub.attr == 'expander':
                                            has_expander_call = True
                                self.assertTrue(has_expander_call, "False branch should use st.expander")

        self.assertTrue(found_progressive_enhancement,
                        "Did not find progressive enhancement pattern for error display")

    def test_market_clock_tooltips(self):
        """
        Verify that the Market Clock Widget in pages/1_Cockpit.py has tooltips
        displaying the date for "UTC Time" and "New York Time (Market)".
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '1_Cockpit.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_utc = False
        found_ny = False

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'metric':
                # Check arguments
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label == "UTC Time":
                    found_utc = True
                    # Check for help keyword argument
                    has_help = any(kw.arg == 'help' for kw in node.keywords)
                    self.assertTrue(has_help, "UTC Time metric is missing 'help' tooltip with date")

                if label == "New York Time (Market)":
                    found_ny = True
                    # Check for help keyword argument
                    has_help = any(kw.arg == 'help' for kw in node.keywords)
                    self.assertTrue(has_help, "New York Time metric is missing 'help' tooltip with date")

        self.assertTrue(found_utc, "Could not find 'UTC Time' metric call")
        self.assertTrue(found_ny, "Could not find 'New York Time (Market)' metric call")


class TestCouncilUX(unittest.TestCase):
    def test_council_metric_tooltips(self):
        """Verify that key metrics in pages/3_The_Council.py have help tooltips."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '3_The_Council.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        target_metrics = ["Dominant Agent", "Weighted Score", "Active Voters", "Trigger", "Confidence", "Compliance", "Consensus"]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'metric':
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_metrics:
                    found_metrics[label] = True
                    has_help = any(kw.arg == 'help' for kw in node.keywords)
                    self.assertTrue(has_help, f"Metric '{label}' is missing 'help' tooltip")

        for metric, found in found_metrics.items():
            self.assertTrue(found, f"Could not find metric '{metric}' in pages/3_The_Council.py")


class TestDashboardUX(unittest.TestCase):
    def test_dashboard_metric_tooltips(self):
        """Verify that key metrics in dashboard.py have help tooltips."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        target_metrics = ["Net Liquidation", "Daily P&L", "Portfolio VaR (95%)", "Trades", "Win Rate"]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'metric':
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_metrics:
                    found_metrics[label] = True
                    has_help = any(kw.arg == 'help' for kw in node.keywords)
                    self.assertTrue(has_help, f"Metric '{label}' is missing 'help' tooltip")

        for metric, found in found_metrics.items():
            self.assertTrue(found, f"Could not find metric '{metric}' in dashboard.py")

    def test_dashboard_activity_dataframe_config(self):
        """Verify that the Recent Activity dataframe in dashboard.py uses column_config."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_config = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'dataframe':
                # Check if this is the Recent Activity dataframe (it's inside Section 4)
                # We look for column_config keyword
                for kw in node.keywords:
                    if kw.arg == 'column_config':
                        found_config = True
                        break

        self.assertTrue(found_config, "Recent Activity dataframe is missing 'column_config'")


class TestPortfolioUX(unittest.TestCase):
    def test_portfolio_semantic_status(self):
        """Verify that pages/9_Portfolio.py uses semantic status containers."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '9_Portfolio.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_semantic = False
        semantic_funcs = {'success', 'warning', 'error', 'info'}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in semantic_funcs:
                    # Check if it's the status display (contains "Portfolio Status")
                    for arg in node.args:
                        if isinstance(arg, ast.JoinedStr):
                            for value in arg.values:
                                if isinstance(value, ast.Constant) and "Portfolio Status" in str(value.value):
                                    found_semantic = True
                        elif isinstance(arg, ast.Constant) and "Portfolio Status" in str(arg.value):
                            found_semantic = True

        self.assertTrue(found_semantic, "Portfolio status display does not use semantic containers (st.success/warning/etc.)")

    def test_portfolio_metric_tooltips(self):
        """Verify that metrics in pages/9_Portfolio.py have help tooltips."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '9_Portfolio.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        target_metrics = ["Net Liquidation", "Peak Equity", "Daily P&L", "Drawdown", "VaR (95%)", "VaR Limit", "Utilization"]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'metric':
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_metrics:
                    found_metrics[label] = True
                    has_help = any(kw.arg == 'help' for kw in node.keywords)
                    self.assertTrue(has_help, f"Metric '{label}' in Portfolio is missing 'help' tooltip")

        for metric, found in found_metrics.items():
            self.assertTrue(found, f"Could not find metric '{metric}' in pages/9_Portfolio.py")

class TestUtilitiesUX(unittest.TestCase):
    def test_utilities_safety_interlocks(self):
        """Verify that high-impact buttons in pages/5_Utilities.py have safety interlocks."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        target_buttons = ["🚀 Collect Logs", "💰 Force Equity Sync", "🚀 Run System Validation"]
        found_buttons = {b: False for b in target_buttons}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'button':
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_buttons:
                    found_buttons[label] = True
                    has_disabled = any(kw.arg == 'disabled' for kw in node.keywords)
                    self.assertTrue(has_disabled, f"Button '{label}' is missing safety interlock (disabled attribute)")

        for button, found in found_buttons.items():
            self.assertTrue(found, f"Could not find button '{button}' in pages/5_Utilities.py")


if __name__ == '__main__':
    unittest.main()
