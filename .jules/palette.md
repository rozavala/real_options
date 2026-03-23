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

## 2026-02-25 - Native Numeric Sorting in Dataframes
**Learning:** Streamlit's `st.dataframe` sorts columns alphabetically if they contain strings (like '$10.00' or '85%'). This breaks the usability of financial tables.
**Action:** Always convert formatted strings to numeric types before rendering in `st.dataframe`. Use `st.column_config.NumberColumn` for currency formatting and `st.column_config.ProgressColumn` for percentages to maintain visual polish while enabling native numeric sorting.

## 2026-02-24 - Semantic Containers for Scannability
**Learning:** Using `st.success`, `st.error`, and `st.info` as wrappers for text content (not just alerts) provides immediate, color-coded context that improves scannability for sentiment-heavy data.
**Action:** Apply this pattern when displaying categorical data with strong positive/negative connotations (like Bullish/Bearish sentiment) to reduce cognitive load.

## 2026-02-23 - Standardizing Contextual Clarity across Dashboard Pages
**Learning:** In multi-page trading dashboards, inconsistent tooltip usage and status indicators across different pages can create cognitive friction. Users expect a "unified language" for metrics (e.g., Net Liquidation, VaR) regardless of which page they are viewing.
**Action:** Standardize tooltip descriptions for identical metrics across all pages. Use semantic containers (`st.success`, `st.warning`, `st.error`) instead of manual markdown styling for system statuses to provide a consistent visual hierarchy and better scannability.

## 2026-02-24 - Human-Readable Temporal Context
**Learning:** In dashboards where decisions are historical (like trade logs or agent councils), absolute timestamps (e.g., "2025-02-24 10:00") are harder to process than relative durations (e.g., "2h ago"). Combining both—absolute for precision and relative for quick context—in selection dropdowns and page headers drastically improves the "time-to-insight".
**Action:** Use a centralized `_relative_time` utility to append human-readable durations to all high-impact timestamps in dropdowns and headers.

## 2026-02-27 - Semantic Engine Health Metrics
**Learning:** Refactoring technical status lists into `st.metric` cards provides a higher-density, more professional look. Using the `delta` parameter for relative pulse time ("Pulse: 2m ago") and the `help` parameter for secondary metadata (cycle counts, absolute timestamps) creates a clear information hierarchy.
**Action:** When displaying system or agent health, prioritize semantic metrics with relative temporal context in the delta field to improve immediate situational awareness.

## 2026-03-04 - Visual Context for Time and Status
**Learning:** In dashboards with multiple time zones (UTC vs local) and system heartbeats, adding semantic emojis (🕒, 🗽, 🚦) to labels and relative deltas ("Last: 2m ago") to status metrics drastically reduces the cognitive effort to verify system synchronicity and health.
**Action:** Prepend semantic emojis to Market Clock labels and use the `delta` parameter in `st.metric` for all system health heartbeats (IB Gateway, State Manager, etc.) to provide immediate situational awareness.

## 2026-03-04 - Semantic Highlighting for Task Failures
**Learning:** Using `delta_color="inverse"` for failure-prone counts (like Overdue Tasks) ensures that non-zero values are highlighted in red (Streamlit default for inverse delta), making operational issues stand out even when the primary metric is numeric.
**Action:** Apply the inverse delta pattern to metrics where "up" is "bad" to ensure consistent visual signaling of system friction.

## 2026-03-05 - Completing the Micro-UX Tooltip Coverage
**Learning:** During a codebase audit, several Streamlit metric components (`st.metric`) across utility and financial pages were found to be missing the `help` parameter. Consistency in providing native tooltips for metrics (like System Health, Failed Checks, and Realized P&L) is critical for a predictable user experience, as users learn to rely on hover states for context.
**Action:** Always ensure that `st.metric` calls include a descriptive `help` attribute, especially for derived or aggregated values where the calculation or significance might not be immediately obvious.

## 2026-03-05 - Completing Tooltip Coverage for High-Impact Actions
**Learning:** While the dashboard has a standard of using tooltips (`help` parameter) on high-stakes manual triggers (like "Force Generate & Execute Orders" and "Clear All Caches"), several critical buttons (e.g., "Cancel All Open Orders" and "Force Close Stale Positions") were found lacking them. Users need immediate, clear context for what destructive actions will do before clicking them, even if there is a safety interlock checkbox.
**Action:** Ensure exhaustive tooltip coverage across all Streamlit components that trigger state mutations or irreversible actions. The `help` string should explicitly describe the scope and consequence of the action (e.g., "Immediately cancels all unfilled DAY orders in Interactive Brokers.").

## 2026-03-09 - Standardizing Dashboard Scannability and Sorting
**Learning:** In multi-page dashboards, using inconsistent formatting for similar data types (e.g., LLM success rates vs Portfolio P&L) increases cognitive load. Prepending semantic emojis to subheaders and metrics creates a "visual anchor" that speeds up navigation. Furthermore, using formatted strings in `st.dataframe` breaks native sorting, which is critical for performance monitoring tables like LLM provider health.
**Action:** Standardize visual anchors (emojis) across all primary metric views. Always prioritize raw numeric data in `st.dataframe` combined with `st.column_config` to ensure both visual polish and functional sorting capabilities.

## 2026-03-10 - Visual Anchors in Data Tables
**Learning:** Data tables often feel "heavy" and purely technical. Prepending semantic emojis to `st.column_config` headers (e.g., 🕒 Time, 📜 Decision) provides visual anchors that make the table more approachable and scan-friendly, especially when mixed with other metric cards that use icons.
**Action:** Consistently use emojis in `column_config` titles for high-impact dashboards to maintain the visual language of the application.
## 2024-05-23 - Add Tooltips to `st.download_button`
**Learning:** Streamlit UI components like `st.download_button` can lack sufficient context, reducing accessibility and clarity. Adding the built-in `help` parameter provides native, descriptive tooltips that explain button functionality.
**Action:** Consistently enforce `help` tooltips on `st.download_button` components (and others like `st.metric` or `st.checkbox`) across the dashboard to improve micro-UX and accessibility. Use `ast` unit tests to statically verify their presence.

## 2024-05-24 - [Micro-UX: Tooltips for Sidebar Filters]
**Learning:** Adding descriptive `help` tooltips to input components like `st.selectbox` provides valuable context for users to understand the purpose of the filters without cluttering the UI, enhancing accessibility.
**Action:** Always include a `help` parameter with clear, concise descriptions for all `st.selectbox`, `st.checkbox`, and similar input components in Streamlit dashboards.
