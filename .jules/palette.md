## 2024-05-23 - Financial Data Visualization
**Learning:** Users rely on color cues (Green/Red) for financial data like P&L. `st.metric` in Streamlit requires the `delta` parameter to trigger these colors; passing a signed string to `value` is insufficient for visual scanning.
**Action:** Always use the `delta` parameter for P&L or performance metrics to ensure they are color-coded (Green for positive, Red for negative) and provide immediate visual feedback.
