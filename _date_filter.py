"""Reusable sidebar date range filter for dashboard pages."""

import streamlit as st
import pandas as pd
from datetime import timedelta


def date_range_filter(df, timestamp_col='timestamp', key_prefix=''):
    """Sidebar date range picker. Returns filtered DataFrame."""
    if df.empty or timestamp_col not in df.columns:
        return df
    ts = pd.to_datetime(df[timestamp_col], errors='coerce')
    valid = ts.dropna()
    if valid.empty:
        return df
    min_date = valid.min().date()
    max_date = valid.max().date()
    if min_date == max_date:
        return df
    default_start = max(min_date, max_date - timedelta(days=30))
    date_val = st.sidebar.date_input(
        "Date range", value=(default_start, max_date),
        min_value=min_date, max_value=max_date,
        key=f"{key_prefix}_daterange"
    )
    # date_input may return a single date if user hasn't finished selecting
    if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
        start, end = date_val
    else:
        return df
    mask = (ts.dt.date >= start) & (ts.dt.date <= end)
    return df[mask].copy()
