import ast
import os
import unittest

class TestCouncilUX(unittest.TestCase):
    def test_agent_summaries_color_coding(self):
        """
        Verify that agent summaries in pages/3_The_Council.py are color-coded
        using st.success, st.error, and st.info based on sentiment.
        """
        file_path = os.path.join(os.path.dirname(__file__), '..', 'pages', '3_The_Council.py')

        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())

        found_loop = False
        found_success = False
        found_error = False
        found_info = False

        for node in ast.walk(tree):
            # Look for the loop iterating over summary_cols.items()
            if isinstance(node, ast.For):
                if (isinstance(node.iter, ast.Call) and
                    isinstance(node.iter.func, ast.Attribute) and
                    node.iter.func.attr == 'items' and
                    isinstance(node.iter.func.value, ast.Name) and
                    node.iter.func.value.id == 'summary_cols'):

                    found_loop = True

                    # Search inside the loop body for st.success, st.error, st.info
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                            if sub.func.attr == 'success':
                                found_success = True
                            elif sub.func.attr == 'error':
                                found_error = True
                            elif sub.func.attr == 'info':
                                found_info = True

        self.assertTrue(found_loop, "Could not find the summary_cols loop in pages/3_The_Council.py")
        self.assertTrue(found_success, "Did not find st.success() call (for BULLISH sentiment)")
        self.assertTrue(found_error, "Did not find st.error() call (for BEARISH sentiment)")
        self.assertTrue(found_info, "Did not find st.info() call (for NEUTRAL sentiment)")

if __name__ == '__main__':
    unittest.main()
