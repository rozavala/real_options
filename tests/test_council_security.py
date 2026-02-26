
import unittest
import sys
import os
import ast

class TestCouncilUX(unittest.TestCase):
    def test_unsafe_allow_html_removed(self):
        """Verify unsafe_allow_html is not used in pages/3_The_Council.py"""
        file_path = "pages/3_The_Council.py"
        with open(file_path, "r") as f:
            content = f.read()

        # Check for the string directly
        self.assertNotIn("unsafe_allow_html=True", content, "Found unsafe_allow_html=True in source code")

    def test_safe_components_used(self):
        """Verify safe Streamlit components are used instead"""
        file_path = "pages/3_The_Council.py"
        with open(file_path, "r") as f:
            content = f.read()

        # Check for new safe implementations
        self.assertIn('st.markdown(f"**Trigger Type:** {html.escape(trigger_badge)}")', content)
        self.assertIn('st.caption("Conviction Pipeline"', content)
        self.assertIn('st.caption("Thesis Strength"', content)
        self.assertIn('st.caption(f"Thesis: {html.escape(thesis_strength)}")', content)
        self.assertIn('st.metric(label="Decision", value=decision)', content)

if __name__ == "__main__":
    unittest.main()
