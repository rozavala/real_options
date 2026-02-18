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

    def test_utilities_safety_interlocks(self):
        """
        Verify that high-impact buttons in pages/5_Utilities.py have safety interlocks.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')

        with open(file_path, 'r') as f:
            content = f.read()
            tree = ast.parse(content)

        button_configs = [
            {"label": "🚀 Collect Logs", "interlock": "confirm_logs"},
            {"label": "🚀 Run System Validation", "interlock": "confirm_validation"},
            {"label": "💰 Force Equity Sync", "interlock": "confirm_sync"}
        ]

        for config in button_configs:
            label = config["label"]
            interlock = config["interlock"]

            found_button = False
            found_interlock_check = False

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'button':
                        # Check label
                        has_label = False
                        if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == label:
                            has_label = True
                        for kw in node.keywords:
                            if kw.arg == 'label' and isinstance(kw.value, ast.Constant) and kw.value.value == label:
                                has_label = True

                        if has_label:
                            found_button = True
                            # Check for disabled=not interlock
                            has_disabled = False
                            for kw in node.keywords:
                                if kw.arg == 'disabled':
                                    if isinstance(kw.value, ast.UnaryOp) and isinstance(kw.value.op, ast.Not):
                                        if isinstance(kw.value.operand, ast.Name) and kw.value.operand.id == interlock:
                                            has_disabled = True
                            self.assertTrue(has_disabled, f"Button '{label}' missing 'disabled=not {interlock}'")

                    if node.func.attr == 'checkbox':
                        # Check for key=interlock
                        for kw in node.keywords:
                            if kw.arg == 'key' and isinstance(kw.value, ast.Constant) and kw.value.value == interlock:
                                found_interlock_check = True

            self.assertTrue(found_button, f"Button '{label}' not found")
            self.assertTrue(found_interlock_check, f"Interlock checkbox with key='{interlock}' not found")

    def test_utilities_visual_consistency(self):
        """
        Verify visual consistency and data freshness enhancements in pages/5_Utilities.py.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')

        with open(file_path, 'r') as f:
            content = f.read()

        # Simple string checks for these ones as they are more about specific lines
        self.assertIn('st.caption(f"🕒 Last modified: **{last_updated}**")', content)
        self.assertIn('st.dataframe(pd.DataFrame(log_files), hide_index=True, width="stretch")', content)

if __name__ == '__main__':
    unittest.main()
