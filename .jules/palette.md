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

## 2026-02-12 - Expectation Management for Long-Running Tasks
**Learning:** For background tasks that exceed 60 seconds (like reconciliation or log collection), adding estimated duration to the `help` tooltip reduces user anxiety and prevents premature page refreshes.
**Action:** Always include a "May take X minutes" note in the tooltip for any action that isn't instantaneous.
