# Palette's Journal

This journal records CRITICAL UX/accessibility learnings.

## 2025-02-23 - Progressive Enhancement for Streamlit
**Learning:** Streamlit versions vary widely in deployments. Features like `st.page_link` (v1.31+) are huge UX wins but can crash older apps.
**Action:** Always wrap newer UI components in `if hasattr(st, "feature"):` blocks to ensure backward compatibility while delivering modern UX where possible.

## 2025-02-23 - Tooltip verification in Streamlit
**Learning:** Verifying tooltips in Streamlit via Playwright requires targeting `[data-testid="stTooltipHoverTarget"]`. This element wraps the help icon inside the metric container.
**Action:** Use this locator pattern for future Streamlit tooltip tests.
