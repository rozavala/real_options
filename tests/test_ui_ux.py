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

    def test_refresh_toast_implementation(self):
        """
        Verify that the refresh button action includes a toast notification.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '1_Cockpit.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_toast = False
        for node in ast.walk(tree):
            # Check for st.toast call
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'toast':
                    found_toast = True
                    break

        self.assertTrue(found_toast, "st.toast usage not found in 1_Cockpit.py")

if __name__ == '__main__':
    unittest.main()
