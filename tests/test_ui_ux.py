import ast
import os
import unittest


class TestCockpitUX(unittest.TestCase):
    def test_sentinel_row_progressive_enhancement(self):
        """
        Verify that _render_sentinel_row in pages/1_Cockpit.py uses
        progressive enhancement (st.popover) for error details.
        """
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "1_Cockpit.py"
        )

        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        # Find the _render_sentinel_row function
        render_func = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_render_sentinel_row"
            ):
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
                if isinstance(node.test, ast.Name) and node.test.id == "error":
                    # Check the first statement in the body
                    if node.body and isinstance(node.body[0], ast.If):
                        inner_if = node.body[0]
                        # Check "if hasattr(st, 'popover'):"
                        if (
                            isinstance(inner_if.test, ast.Call)
                            and isinstance(inner_if.test.func, ast.Name)
                            and inner_if.test.func.id == "hasattr"
                        ):
                            args = inner_if.test.args
                            if (
                                len(args) == 2
                                and isinstance(args[0], ast.Name)
                                and args[0].id == "st"
                                and isinstance(args[1], ast.Constant)
                                and args[1].value == "popover"
                            ):
                                found_progressive_enhancement = True

                                # Verify True branch uses popover
                                has_popover_call = False
                                for sub in ast.walk(inner_if.body[0]):
                                    if (
                                        isinstance(sub, ast.Attribute)
                                        and sub.attr == "popover"
                                    ):
                                        has_popover_call = True
                                self.assertTrue(
                                    has_popover_call,
                                    "True branch should use st.popover",
                                )

                                # Verify False branch uses expander
                                has_expander_call = False
                                if inner_if.orelse:
                                    for sub in ast.walk(inner_if.orelse[0]):
                                        if (
                                            isinstance(sub, ast.Attribute)
                                            and sub.attr == "expander"
                                        ):
                                            has_expander_call = True
                                self.assertTrue(
                                    has_expander_call,
                                    "False branch should use st.expander",
                                )

        self.assertTrue(
            found_progressive_enhancement,
            "Did not find progressive enhancement pattern for error display",
        )

    def test_market_clock_tooltips(self):
        """
        Verify that the Market Clock Widget in pages/1_Cockpit.py has tooltips
        displaying the date for "UTC Time" and "New York Time (Market)".
        """
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "1_Cockpit.py"
        )

        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_utc = False
        found_ny = False

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                # Check arguments
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label and "UTC Time" in label:
                    found_utc = True
                    # Check for help keyword argument
                    has_help = any(kw.arg == "help" for kw in node.keywords)
                    self.assertTrue(has_help, f"{label} metric is missing 'help' tooltip with date")

                if label and "New York Time" in label:
                    found_ny = True
                    # Check for help keyword argument
                    has_help = any(kw.arg == "help" for kw in node.keywords)
                    self.assertTrue(has_help, f"{label} metric is missing 'help' tooltip with date")

        self.assertTrue(found_utc, "Could not find 'UTC Time' metric call")
        self.assertTrue(found_ny, "Could not find 'New York Time' metric call")

    def test_cockpit_metric_tooltips(self):
        """Verify that key metrics in pages/1_Cockpit.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "1_Cockpit.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "Unrealized P&L",
            "Margin Util",
            "VaR(95%)",
            "VaR(99%)",
            "Utilization",
            "Legs",
            "Since Reset",
            "📋 Total Tasks",
            "✅ Completed",
            "⚠️ Overdue",
            "⏭️ Skipped",
            "⏳ Upcoming",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Cockpit is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/1_Cockpit.py"
            )

    def test_cockpit_task_dataframe_config(self):
        """Verify that the Task Schedule dataframe in Cockpit uses column_config."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '1_Cockpit.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_config = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'dataframe':
                for kw in node.keywords:
                    if kw.arg == 'column_config':
                        # Check if it has "Task Description" which is unique to our change
                        config_str = ast.dump(kw.value)
                        if "Task Description" in config_str:
                            found_config = True
                            break
        self.assertTrue(found_config, "Task Schedule dataframe is missing 'column_config' with 'Task Description'")


class TestFunnelUX(unittest.TestCase):
    def test_funnel_metric_tooltips(self):
        """Verify that key metrics in pages/10_The_Funnel.py have help tooltips and emojis."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "10_The_Funnel.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "📡 Signal-to-Trade",
            "⛽ Fill Rate",
            "📉 Avg Slippage",
            "🛡️ Conviction Blocks",
            "👣 Avg Walk Steps",
            "🎯 Signal Win Rate",
            "💸 Alpha Left on Table",
            "✅ Skill",
            "🍀 Lucky",
            "📉 Market Risk",
            "❌ Bad Call",
            "📊 Mean",
            "🎯 Median",
            "🔝 P75",
            "🚀 Max",
            "✅ Favorable",
            "⚠️ Adverse",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_metrics:
                    found_metrics[label] = True
                    has_help = any(kw.arg == "help" for kw in node.keywords)
                    self.assertTrue(
                        has_help,
                        f"Metric '{label}' in The Funnel is missing 'help' tooltip",
                    )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/10_The_Funnel.py"
            )

    def test_funnel_dataframe_config(self):
        """Verify that dataframes in The Funnel use column_config."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "10_The_Funnel.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_configs = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dataframe"
            ):
                for kw in node.keywords:
                    if kw.arg == "column_config":
                        found_configs += 1
                        break

        # We updated 6 dataframes with column_config
        self.assertGreaterEqual(
            found_configs, 6, "Expected at least 6 dataframes with column_config in The Funnel"
        )

    def test_funnel_download_button_tooltip(self):
        """Verify that the download button in pages/10_The_Funnel.py has a help tooltip."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "10_The_Funnel.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_button = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "download_button":
                has_help = False
                for kw in node.keywords:
                    if kw.arg == "help":
                        has_help = True
                        break
                self.assertTrue(has_help, f"download_button missing tooltip help in {file_path}")
                found_button = True
        self.assertTrue(found_button, "No download_button found to check in The Funnel")


class TestDashboardUX(unittest.TestCase):
    def test_dashboard_metric_tooltips(self):
        """Verify that key metrics in dashboard.py have help tooltips."""
        file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.py")
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "Net Liquidation",
            "Daily P&L",
            "Portfolio VaR (95%)",
            "Trades",
            "Win Rate",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help, f"Metric '{label}' is missing 'help' tooltip"
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(found, f"Could not find metric '{metric}' in dashboard.py")

    def test_dashboard_activity_dataframe_config(self):
        """Verify that the Recent Activity dataframe in dashboard.py uses column_config."""
        file_path = os.path.join(os.path.dirname(__file__), "..", "dashboard.py")
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_config = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dataframe"
            ):
                # Check if this is the Recent Activity dataframe (it's inside Section 4)
                # We look for column_config keyword
                for kw in node.keywords:
                    if kw.arg == "column_config":
                        found_config = True
                        break

        self.assertTrue(
            found_config, "Recent Activity dataframe is missing 'column_config'"
        )


class TestPortfolioUX(unittest.TestCase):
    def test_portfolio_semantic_status(self):
        """Verify that pages/9_Portfolio.py uses semantic status containers."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "9_Portfolio.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_semantic = False
        semantic_funcs = {"success", "warning", "error", "info"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in semantic_funcs:
                    # Check if it's the status display (contains "Portfolio Status")
                    for arg in node.args:
                        if isinstance(arg, ast.JoinedStr):
                            for value in arg.values:
                                if isinstance(
                                    value, ast.Constant
                                ) and "Portfolio Status" in str(value.value):
                                    found_semantic = True
                        elif isinstance(
                            arg, ast.Constant
                        ) and "Portfolio Status" in str(arg.value):
                            found_semantic = True

        self.assertTrue(
            found_semantic,
            "Portfolio status display does not use semantic containers (st.success/warning/etc.)",
        )

    def test_portfolio_metric_tooltips(self):
        """Verify that metrics in pages/9_Portfolio.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "9_Portfolio.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "Net Liquidation",
            "Peak Equity",
            "Daily P&L",
            "Drawdown",
            "VaR (95%)",
            "VaR (99%)",
            "Utilization",
            "Legs",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Portfolio is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/9_Portfolio.py"
            )


class TestUtilitiesUX(unittest.TestCase):
    def test_utilities_checkbox_tooltips(self):
        """Verify that all checkboxes in pages/5_Utilities.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "5_Utilities.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "checkbox"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value
                elif isinstance(node.args[0], ast.JoinedStr):
                    # Handle f-strings if any
                    label = "f-string"

                if label:
                    has_help = any(kw.arg == "help" for kw in node.keywords)
                    self.assertTrue(
                        has_help,
                        f"Checkbox '{label}' in Utilities is missing 'help' tooltip",
                    )

    def test_utilities_safety_interlocks(self):
        """Verify that high-impact buttons in pages/5_Utilities.py have safety interlocks."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "5_Utilities.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_buttons = [
            "🚀 Collect Logs",
            "💰 Force Equity Sync",
            "🚀 Run System Validation",
        ]
        found_buttons = {b: False for b in target_buttons}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "button"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                if label in found_buttons:
                    found_buttons[label] = True
                    has_disabled = any(kw.arg == "disabled" for kw in node.keywords)
                    self.assertTrue(
                        has_disabled,
                        f"Button '{label}' is missing safety interlock (disabled attribute)",
                    )

        for button, found in found_buttons.items():
            self.assertTrue(
                found, f"Could not find button '{button}' in pages/5_Utilities.py"
            )


class TestCouncilUX(unittest.TestCase):
    def test_council_metric_tooltips(self):
        """Verify that key metrics in pages/3_The_Council.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "3_The_Council.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "👑 Dominant Agent",
            "⚖️ Weighted Score",
            "🗳️ Active Voters",
            "⚡ Trigger",
            "🎯 Confidence",
            "🛡️ Compliance",
            "🤝 Consensus",
            "📉 Market Move",
            "🚪 Exit Price",
            "💰 P&L",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Council is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/3_The_Council.py"
            )


class TestSignalOverlayUX(unittest.TestCase):
    def test_signal_overlay_metric_tooltips(self):
        """Verify that key metrics in pages/6_Signal_Overlay.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "6_Signal_Overlay.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = [
            "Period Change",
            "High",
            "Low",
            "Range",
            "Total Signals",
            "🟢 Bullish",
            "🔴 Bearish",
            "🟣 Volatility",
            "⚪ Neutral",
        ]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Signal Overlay is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/6_Signal_Overlay.py"
            )

    def test_signal_overlay_download_button_tooltip(self):
        """Verify that the download button in pages/6_Signal_Overlay.py has a help tooltip."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "6_Signal_Overlay.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        found_button = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "download_button":
                has_help = False
                for kw in node.keywords:
                    if kw.arg == "help":
                        has_help = True
                        break
                self.assertTrue(has_help, f"download_button missing tooltip help in {file_path}")
                found_button = True
        self.assertTrue(found_button, "No download_button found to check in Signal Overlay")


class TestBrierAnalysisUX(unittest.TestCase):
    def test_brier_analysis_metric_tooltips(self):
        """Verify that key metrics in pages/7_Brier_Analysis.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "7_Brier_Analysis.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        target_metrics = ["Total Predictions", "Resolved", "Pending"]
        found_metrics = {m: False for m in target_metrics}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use substring matching to handle emojis/icons
                for target in target_metrics:
                    if label and target in label:
                        found_metrics[target] = True
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Brier Analysis is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/7_Brier_Analysis.py"
            )


class TestScorecardUX(unittest.TestCase):
    def test_scorecard_metric_tooltips(self):
        """Verify that key metrics in pages/2_The_Scorecard.py have help tooltips."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "2_The_Scorecard.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        # Metric labels with their corresponding semantic emojis
        target_metrics = {
            "Precision": "🎯",
            "Recall": "🔍",
            "Accuracy": "✅",
            "Total Graded": "🔢",
            "Avg Winning Duration": "⏱️",
            "Avg Losing Duration": "⏱️",
            "Win Rate": "🎯",
            "Resolved Cycles": "📋",
            "Override Rate": "🔄",
            "Override Delta": "⚖️",
            "Confidence": "🎯",
            "Aligned Win Rate": "🤝",
            "Override Win Rate": "🚀",
            "Thesis Calibration": "💡",
            "NEUTRAL Rate": "➖",
            "NEUTRAL Calls": "➖",
            "NEUTRAL Accuracy": "🎯",
            "Directional Win Rate": "📈",
        }
        found_metrics = {m: False for m in target_metrics.keys()}

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "metric"
            ):
                if not node.args:
                    continue

                label = None
                if isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value

                # Use matching to handle emojis/icons
                for target, emoji in target_metrics.items():
                    # Exact match to avoid "Win Rate" matching "Aligned Win Rate"
                    if label == f"{emoji} {target}":
                        found_metrics[target] = True

                        # Verify help tooltip exists
                        has_help = any(kw.arg == "help" for kw in node.keywords)
                        self.assertTrue(
                            has_help,
                            f"Metric '{label}' in Scorecard is missing 'help' tooltip",
                        )

        for metric, found in found_metrics.items():
            self.assertTrue(
                found, f"Could not find metric '{metric}' in pages/2_The_Scorecard.py"
            )


class TestFinancialsUX(unittest.TestCase):
    def test_strategy_efficiency_dataframe_config(self):
        """Verify that the Strategy Efficiency dataframe in Trade Analytics uses column_config."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '4_Financials.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_config = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'dataframe':
                for kw in node.keywords:
                    if kw.arg == 'column_config':
                        # Check for Strategy Efficiency specific headers
                        config_str = ast.dump(kw.value)
                        if "🛡️ Strategy" in config_str and "💰 Total P&L" in config_str:
                            found_config = True
                            break
        self.assertTrue(found_config, "Strategy Efficiency dataframe is missing 'column_config' with professional headers")


class TestLLMMonitorUX(unittest.TestCase):
    def test_llm_monitor_dataframe_config(self):
        """Verify that the Provider Health dataframe in LLM Monitor uses column_config."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '8_LLM_Monitor.py')
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_config = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'dataframe':
                for kw in node.keywords:
                    if kw.arg == 'column_config':
                        # Check for ProgressColumn config
                        config_str = ast.dump(kw.value)
                        if "ProgressColumn" in config_str and "Success Rate" in config_str:
                            found_config = True
                            # Check max_value=100 (fixes the 1.0% vs 100% bug)
                            self.assertIn("100", config_str, "ProgressColumn should have max_value=100")
                            break
        self.assertTrue(found_config, "LLM Monitor Provider Health dataframe is missing 'column_config' with 'Success Rate' ProgressColumn")


class TestBuildPositionPnlMap(unittest.TestCase):
    """Tests for dashboard_utils.build_position_pnl_map()."""

    def _make_item(self, local_symbol, unrealized_pnl, position=1):
        """Create a mock portfolio item."""
        class MockContract:
            pass
        class MockItem:
            pass
        contract = MockContract()
        contract.localSymbol = local_symbol
        item = MockItem()
        item.contract = contract
        item.unrealizedPNL = unrealized_pnl
        item.position = position
        return item

    def test_single_leg_match(self):
        """Single leg maps correctly via ledger."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C3.5'],
            'position_id': ['pid-001'],
        })
        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5', 150.0)]}

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import build_position_pnl_map
            result = build_position_pnl_map(live_data, ticker='KC')

        self.assertEqual(result, {'pid-001': 150.0})

    def test_multi_leg_spread_aggregation(self):
        """Multiple legs with same position_id aggregate P&L."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C3.5', 'KCH6 C3.75'],
            'position_id': ['pid-001', 'pid-001'],
        })
        live_data = {'portfolio_items': [
            self._make_item('KCH6 C3.5', 100.0),
            self._make_item('KCH6 C3.75', -30.0),
        ]}

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import build_position_pnl_map
            result = build_position_pnl_map(live_data, ticker='KC')

        self.assertAlmostEqual(result['pid-001'], 70.0)

    def test_empty_portfolio(self):
        """Empty portfolio returns empty dict."""
        from dashboard_utils import build_position_pnl_map
        result = build_position_pnl_map({'portfolio_items': []}, ticker='KC')
        self.assertEqual(result, {})

    def test_empty_ledger(self):
        """Empty ledger returns empty dict."""
        import pandas as pd
        from unittest.mock import patch

        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5', 100.0)]}

        with patch('dashboard_utils.load_trade_data', return_value=pd.DataFrame()):
            from dashboard_utils import build_position_pnl_map
            result = build_position_pnl_map(live_data, ticker='KC')

        self.assertEqual(result, {})

    def test_missing_position_id_column(self):
        """Ledger without position_id column returns empty dict."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({'local_symbol': ['KCH6 C3.5']})
        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5', 100.0)]}

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import build_position_pnl_map
            result = build_position_pnl_map(live_data, ticker='KC')

        self.assertEqual(result, {})

    def test_no_match_excluded(self):
        """Portfolio items not in ledger are excluded from map."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C3.5'],
            'position_id': ['pid-001'],
        })
        live_data = {'portfolio_items': [self._make_item('KCH6 P3.0', 50.0)]}

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import build_position_pnl_map
            result = build_position_pnl_map(live_data, ticker='KC')

        self.assertEqual(result, {})


class TestFindUntrackedIbkrPositions(unittest.TestCase):
    """Tests for dashboard_utils.find_untracked_ibkr_positions()."""

    def _make_item(self, local_symbol, position=1, unrealized_pnl=0, market_value=0, avg_cost=0):
        class MockContract:
            pass
        class MockItem:
            pass
        contract = MockContract()
        contract.localSymbol = local_symbol
        item = MockItem()
        item.contract = contract
        item.position = position
        item.unrealizedPNL = unrealized_pnl
        item.marketValue = market_value
        item.averageCost = avg_cost
        return item

    def test_all_tracked(self):
        """No untracked items when all symbols map to active theses."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C3.5'],
            'position_id': ['pid-001'],
        })
        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5')]}
        theses = [{'position_id': 'pid-001'}]

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import find_untracked_ibkr_positions
            result = find_untracked_ibkr_positions(live_data, theses, ticker='KC')

        self.assertEqual(result, [])

    def test_untracked_detected(self):
        """Detects items not linked to any thesis."""
        import pandas as pd
        from unittest.mock import patch

        ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C3.5'],
            'position_id': ['pid-001'],
        })
        live_data = {'portfolio_items': [
            self._make_item('KCH6 C3.5'),
            self._make_item('KCH6 P3.0', position=2, unrealized_pnl=-50),
        ]}
        theses = [{'position_id': 'pid-001'}]

        with patch('dashboard_utils.load_trade_data', return_value=ledger):
            from dashboard_utils import find_untracked_ibkr_positions
            result = find_untracked_ibkr_positions(live_data, theses, ticker='KC')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['local_symbol'], 'KCH6 P3.0')

    def test_zero_qty_excluded(self):
        """Zero-quantity positions are not reported as untracked."""
        import pandas as pd
        from unittest.mock import patch

        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5', position=0)]}

        with patch('dashboard_utils.load_trade_data', return_value=pd.DataFrame()):
            from dashboard_utils import find_untracked_ibkr_positions
            result = find_untracked_ibkr_positions(live_data, [], ticker='KC')

        self.assertEqual(result, [])

    def test_empty_theses(self):
        """All non-zero positions reported when no theses exist."""
        import pandas as pd
        from unittest.mock import patch

        live_data = {'portfolio_items': [self._make_item('KCH6 C3.5', position=1)]}

        with patch('dashboard_utils.load_trade_data', return_value=pd.DataFrame()):
            from dashboard_utils import find_untracked_ibkr_positions
            result = find_untracked_ibkr_positions(live_data, [], ticker='KC')

        self.assertEqual(len(result), 1)

    def test_empty_portfolio(self):
        """Empty portfolio returns empty list."""
        from dashboard_utils import find_untracked_ibkr_positions
        result = find_untracked_ibkr_positions({'portfolio_items': []}, [], ticker='KC')
        self.assertEqual(result, [])


class TestRenderThesisCardSignature(unittest.TestCase):
    """AST tests: verify render_thesis_card_enhanced structure after refactor."""

    def _get_tree_and_func(self):
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "1_Cockpit.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "render_thesis_card_enhanced":
                return tree, node
        self.fail("render_thesis_card_enhanced not found")

    def test_pnl_map_param_present(self):
        """Function must accept a pnl_map parameter."""
        _, func = self._get_tree_and_func()
        param_names = [arg.arg for arg in func.args.args]
        self.assertIn("pnl_map", param_names)

    def test_no_stop_price_metric(self):
        """No st.metric call with 'Stop Price' label should exist."""
        _, func = self._get_tree_and_func()
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "metric":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "Stop Price":
                    self.fail("Found 'Stop Price' metric — should have been removed")

    def test_uses_three_columns(self):
        """st.columns should be called with 3, not 4."""
        _, func = self._get_tree_and_func()
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "columns":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 4:
                    self.fail("Found st.columns(4) — should be st.columns(3)")

    def test_extract_stop_price_removed(self):
        """extract_stop_price_from_triggers should not exist in the file."""
        file_path = os.path.join(
            os.path.dirname(__file__), "..", "pages", "1_Cockpit.py"
        )
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "extract_stop_price_from_triggers":
                self.fail("extract_stop_price_from_triggers still exists — should have been removed")


if __name__ == "__main__":
    unittest.main()
