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
    def test_log_collection_safety_interlock(self):
        """
        Verify that the "Collect Logs" button in pages/5_Utilities.py
        is protected by a safety interlock (st.checkbox).
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        # Scan for confirm_logs = st.checkbox(...) and st.button(..., disabled=not confirm_logs)
        found_checkbox = False
        found_protected_button = False

        for node in ast.walk(tree):
            # Find checkbox assignment: confirm_logs = st.checkbox(...)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'confirm_logs':
                        if (isinstance(node.value, ast.Call) and
                            isinstance(node.value.func, ast.Attribute) and
                            node.value.func.attr == 'checkbox'):
                            found_checkbox = True

            # Find button call: st.button("🚀 Collect Logs", ..., disabled=not confirm_logs)
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'button'):

                    # Check if first argument is "🚀 Collect Logs"
                    is_collect_logs = False
                    if node.args and isinstance(node.args[0], ast.Constant) and "Collect Logs" in str(node.args[0].value):
                        is_collect_logs = True

                    if is_collect_logs:
                        # Check keyword arguments for disabled=not confirm_logs
                        for kw in node.keywords:
                            if kw.arg == 'disabled':
                                if (isinstance(kw.value, ast.UnaryOp) and
                                    isinstance(kw.value.op, ast.Not) and
                                    isinstance(kw.value.operand, ast.Name) and
                                    kw.value.operand.id == 'confirm_logs'):
                                    found_protected_button = True

        self.assertTrue(found_checkbox, "Safety checkbox 'confirm_logs' not found in Utilities")
        self.assertTrue(found_protected_button, "Collect Logs button is not protected by 'confirm_logs' interlock")

if __name__ == '__main__':
    unittest.main()
