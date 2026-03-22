"""Reusable sidebar date range filter for dashboard pages."""

import streamlit as st
import pandas as pd
from datetime import timedelta


def date_range_picker(df, timestamp_col='timestamp', key='page', default_days=10):
    """Render ONE sidebar date range picker derived from a DataFrame's timestamps.

    Returns (start_date, end_date) or None if the df is empty / single-day.
    Call this once per page, then use apply_date_filter() on each DataFrame.
    """
    if df.empty or timestamp_col not in df.columns:
        return None
    ts = pd.to_datetime(df[timestamp_col], errors='coerce')
    valid = ts.dropna()
    if valid.empty:
        return None
    min_date = valid.min().date()
    max_date = valid.max().date()
    if min_date == max_date:
        return None
    default_start = max(min_date, max_date - timedelta(days=default_days))
    date_val = st.sidebar.date_input(
        "Date range", value=(default_start, max_date),
        min_value=min_date, max_value=max_date,
        key=f"{key}_daterange"
    )
    if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
        return date_val[0], date_val[1]
    return None


def apply_date_filter(df, start, end, timestamp_col='timestamp'):
    """Filter a DataFrame to rows where timestamp_col falls within [start, end]."""
    if df.empty or timestamp_col not in df.columns:
        return df
    ts = pd.to_datetime(df[timestamp_col], errors='coerce')
    mask = (ts.dt.date >= start) & (ts.dt.date <= end)
    return df[mask].copy()


def date_range_filter(df, timestamp_col='timestamp', key_prefix='page'):
    """Convenience: render picker + filter in one call. Use on single-df pages."""
    dates = date_range_picker(df, timestamp_col, key=key_prefix)
    if dates is None:
        return df
    return apply_date_filter(df, dates[0], dates[1], timestamp_col)
