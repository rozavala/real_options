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

## 2026-02-11 - Broadening the Safety Interlock Scope
**Learning:** Users often assume that "Utilities" or "Maintenance" pages are safe zones where buttons can be clicked freely. However, network-bound tasks like IBKR Flex Query reconciliation or log archiving can cause the UI to "hang" for minutes. Applying the Safety Interlock pattern to ALL long-running tasks (not just destructive ones) improves the perceived reliability and prevents accidental trigger of expensive/slow operations.
**Action:** Audit all maintenance and utility pages for buttons triggering subprocesses or network calls longer than 10 seconds. Apply mandatory confirmation checkboxes to these elements.
