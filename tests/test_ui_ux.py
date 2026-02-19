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

class TestUtilitiesUX(unittest.TestCase):
    def test_safety_interlocks_in_utilities(self):
        """
        Verify that high-impact buttons in pages/5_Utilities.py
        implement the Safety Interlock pattern (disabled until confirmed).
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        buttons_to_check = [
            "🚀 Collect Logs",
            "💰 Force Equity Sync",
            "🚀 Run System Validation",
            "🚀 Force Generate & Execute Orders",
            "🛑 Cancel All Open Orders",
            "🔄 Force Close Stale Positions"
        ]

        found_buttons = {label: False for label in buttons_to_check}

        for node in ast.walk(tree):
            # Look for st.button calls
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'button'):

                # Check if the first argument (label) is one we're interested in
                label = None
                if node.args and isinstance(node.args[0], ast.Constant):
                    label = node.args[0].value
                else:
                    # Check keywords if label is passed as keyword
                    for kw in node.keywords:
                        if kw.arg == 'label' and isinstance(kw.value, ast.Constant):
                            label = kw.value.value

                if label in buttons_to_check:
                    # Verify it has a 'disabled' keyword argument
                    has_disabled = any(kw.arg == 'disabled' for kw in node.keywords)
                    if has_disabled:
                        found_buttons[label] = True

        for label, found in found_buttons.items():
            self.assertTrue(found, f"Safety interlock (disabled attribute) missing for button: '{label}' in Utilities page")

if __name__ == '__main__':
    unittest.main()
