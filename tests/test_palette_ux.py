import ast
import os
import unittest

class TestUtilitiesUX(unittest.TestCase):
    def setUp(self):
        self.file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '5_Utilities.py')
        with open(self.file_path, 'r') as f:
            self.tree = ast.parse(f.read())

    def test_safety_interlock_log_collection(self):
        """Verify Safety Interlock for Log Collection."""
        self._verify_button_interlock("🚀 Collect Logs", "confirm_collect")

    def test_safety_interlock_equity_sync(self):
        """Verify Safety Interlock for Force Equity Sync."""
        self._verify_button_interlock("💰 Force Equity Sync", "confirm_equity_sync")

    def test_safety_interlock_system_validation(self):
        """Verify Safety Interlock for System Validation."""
        self._verify_button_interlock("🚀 Run System Validation", "confirm_validation")

    def test_safety_interlock_reconciliation(self):
        """Verify Safety Interlock for Reconciliation buttons."""
        recon_buttons = [
            "🔄 Reconcile Council History",
            "🔄 Reconcile Trade Ledger",
            "🔄 Reconcile Positions",
            "🔄 Sync Equity Data",
            "🔄 Reconcile Brier Scores"
        ]
        for label in recon_buttons:
            self._verify_button_interlock(label, "confirm_recon_all")

    def _verify_button_interlock(self, button_label, checkbox_var_name):
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                # Check for st.button
                if (isinstance(node.func, ast.Attribute) and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'st' and
                    node.func.attr == 'button'):

                    # Check if the first argument or 'label' keyword matches button_label
                    label_matches = False
                    if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == button_label:
                        label_matches = True
                    else:
                        for kw in node.keywords:
                            if kw.arg == 'label' and isinstance(kw.value, ast.Constant) and kw.value.value == button_label:
                                label_matches = True
                                break

                    if label_matches:
                        # Check for 'disabled' keyword
                        disabled_kw = next((kw for kw in node.keywords if kw.arg == 'disabled'), None)
                        self.assertIsNotNone(disabled_kw, f"Button '{button_label}' missing 'disabled' parameter")

                        # Check if disabled=not <checkbox_var_name>
                        # disabled=not confirm_collect
                        if (isinstance(disabled_kw.value, ast.UnaryOp) and
                            isinstance(disabled_kw.value.op, ast.Not) and
                            isinstance(disabled_kw.value.operand, ast.Name) and
                            disabled_kw.value.operand.id == checkbox_var_name):
                            found = True
                            break

        self.assertTrue(found, f"Safety interlock for '{button_label}' with variable '{checkbox_var_name}' not found")

if __name__ == '__main__':
    unittest.main()
