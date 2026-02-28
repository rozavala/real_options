# Palette's Journal

This journal records CRITICAL UX/accessibility learnings.

## 2025-02-23 - Progressive Enhancement for Streamlit
**Learning:** Streamlit versions vary widely in deployments. Features like `st.page_link` (v1.31+) are huge UX wins but can crash older apps.
**Action:** Always wrap newer UI components in `if hasattr(st, "feature"):` blocks to ensure backward compatibility while delivering modern UX where possible.

## 2025-02-23 - Tooltip verification in Streamlit
**Learning:** Verifying tooltips in Streamlit via Playwright requires targeting `[data-testid="stTooltipHoverTarget"]`. This element wraps the help icon inside the metric container.
**Action:** Use this locator pattern for future Streamlit tooltip tests.

## 2026-02-07 - Safety Interlock Pattern for High-Impact Controls
**Learning:** For high-stakes actions (emergency halts, order cancellations) in trading dashboards, a single-click interface is dangerous. A "Safety Interlock" (a mandatory confirmation checkbox that enables the action button) significantly reduces the risk of accidental execution while maintaining accessibility.
**Action:** Implement this pattern for all destructive or high-impact manual controls. Use clear, imperative labels for the checkbox (e.g., "I confirm I want to...") and provide tooltips explaining the button's action.

## 2026-02-08 - Safety Interlock for State-Resetting Actions
**Learning:** Even non-destructive actions like "Clear All Caches" can be disruptive in a data-heavy dashboard by forcing expensive re-fetching. Applying the Safety Interlock pattern here prevents accidental UI "jank" and improves the perceived reliability of the system.
**Action:** Use the Safety Interlock pattern for any action that resets UI state or triggers significant data reloads. Add descriptive tooltips to clarify the impact of such actions.

## 2026-02-12 - Progressive Enhancement for Copy Functionality
**Learning:** `st.popover` (Streamlit v1.33+) provides a cleaner UX for secondary actions like "Copy ID" compared to `st.expander`.
**Action:** Use `if hasattr(st, "popover"):` to conditionally render the modern UI, falling back to `st.expander` to ensure accessibility and functionality across different deployments.

## 2026-02-22 - Contextual Clarity via Tooltips
**Learning:** In data-dense trading dashboards, metrics and high-impact buttons can be ambiguous to new or stressed users. Using the `help` parameter in `st.metric` and `st.button` provides essential "just-in-time" documentation without cluttering the visual interface.
**Action:** Always provide descriptive tooltips for system-level metrics (e.g., Python/Streamlit versions, UTC time) and high-stakes manual triggers (e.g., Force Generate) to reduce cognitive load and operational risk.

## 2026-02-24 - Semantic Containers for Scannability
**Learning:** Using `st.success`, `st.error`, and `st.info` as wrappers for text content (not just alerts) provides immediate, color-coded context that improves scannability for sentiment-heavy data.
**Action:** Apply this pattern when displaying categorical data with strong positive/negative connotations (like Bullish/Bearish sentiment) to reduce cognitive load.

## 2026-02-23 - Standardizing Contextual Clarity across Dashboard Pages
**Learning:** In multi-page trading dashboards, inconsistent tooltip usage and status indicators across different pages can create cognitive friction. Users expect a "unified language" for metrics (e.g., Net Liquidation, VaR) regardless of which page they are viewing.
**Action:** Standardize tooltip descriptions for identical metrics across all pages. Use semantic containers (`st.success`, `st.warning`, `st.error`) instead of manual markdown styling for system statuses to provide a consistent visual hierarchy and better scannability.

## 2026-02-24 - Human-Readable Temporal Context
**Learning:** In dashboards where decisions are historical (like trade logs or agent councils), absolute timestamps (e.g., "2025-02-24 10:00") are harder to process than relative durations (e.g., "2h ago"). Combining both—absolute for precision and relative for quick context—in selection dropdowns and page headers drastically improves the "time-to-insight".
**Action:** Use a centralized `_relative_time` utility to append human-readable durations to all high-impact timestamps in dropdowns and headers.
