# Palette's Journal

This journal records CRITICAL UX/accessibility learnings.

## 2025-02-23 - Progressive Enhancement for Streamlit
**Learning:** Streamlit versions vary widely in deployments. Features like `st.page_link` (v1.31+) are huge UX wins but can crash older apps.
**Action:** Always wrap newer UI components in `if hasattr(st, "feature"):` blocks to ensure backward compatibility while delivering modern UX where possible.

## 2025-02-23 - Tooltip verification in Streamlit
**Learning:** Verifying tooltips in Streamlit via Playwright requires targeting `[data-testid="stTooltipHoverTarget"]`. This element wraps the help icon inside the metric container.
**Action:** Use this locator pattern for future Streamlit tooltip tests.

## 2025-02-06 - Safety Interlock for High-Impact Actions
**Learning:** For high-impact or destructive UI actions (like emergency halts), a simple button is not enough. Adding a confirmation checkbox that must be checked to enable the button provides a strong safety interlock and prevents accidental execution.
**Action:** Implement confirmation checkboxes for destructive actions and use the `disabled` attribute of buttons to enforce the state.
