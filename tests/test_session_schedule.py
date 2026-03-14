"""Tests for session-anchored schedule generation.

Covers: _build_session_schedule, get_trading_cutoff, parse_trading_hours,
signal distribution, pre/post-close task ordering, and is_market_open with config.
"""

import pytest
from datetime import datetime, time, timedelta, timezone
from unittest.mock import patch, MagicMock
import pytz

from orchestrator import (
    ScheduledTask,
    FUNCTION_REGISTRY,
    build_schedule,
    _build_session_schedule,
    get_trading_cutoff,
)
from config.commodity_profiles import parse_trading_hours


# === parse_trading_hours ===

def test_parse_trading_hours_kc():
    """KC trading hours parse correctly."""
    open_t, close_t = parse_trading_hours("04:15-13:30")
    assert open_t == time(4, 15)
    assert close_t == time(13, 30)


def test_parse_trading_hours_cc():
    """CC trading hours parse correctly."""
    open_t, close_t = parse_trading_hours("04:45-13:30")
    assert open_t == time(4, 45)
    assert close_t == time(13, 30)


def test_parse_trading_hours_midnight():
    """Edge case: hours starting at midnight."""
    open_t, close_t = parse_trading_hours("00:00-23:59")
    assert open_t == time(0, 0)
    assert close_t == time(23, 59)


def test_parse_trading_hours_single_digit():
    """Single-digit hours."""
    open_t, close_t = parse_trading_hours("9:30-16:00")
    assert open_t == time(9, 30)
    assert close_t == time(16, 0)


# === Helper: build a session config for KC ===

def _make_session_config(symbol='KC', signal_count=4, start_pct=0.05, end_pct=0.80,
                          cutoff_minutes=78):
    """Build a test config with session template."""
    return {
        'symbol': symbol,
        'schedule': {
            'mode': 'session',
            'daily_trading_cutoff_et': {'hour': 12, 'minute': 15},
            'session_template': {
                'signal_count': signal_count,
                'signal_start_pct': start_pct,
                'signal_end_pct': end_pct,
                'cutoff_before_close_minutes': cutoff_minutes,
                'pre_open_tasks': [
                    {'id': 'start_monitoring', 'offset_minutes': -45, 'function': 'start_monitoring', 'label': 'Start Position Monitoring'},
                    {'id': 'process_deferred_triggers', 'offset_minutes': -44, 'function': 'process_deferred_triggers', 'label': 'Process Deferred Triggers'},
                    {'id': 'cleanup_orphaned_theses', 'offset_minutes': -15, 'function': 'cleanup_orphaned_theses', 'label': 'Daily Thesis Cleanup'},
                ],
                'intra_session_tasks': [
                    {'id': 'audit_mid_session', 'session_pct': 0.50, 'function': 'run_position_audit_cycle', 'label': 'Audit: Mid-Session'},
                ],
                'pre_close_tasks': [
                    {'id': 'audit_pre_close', 'offset_minutes': -35, 'function': 'run_position_audit_cycle', 'label': 'Audit: Pre-Close'},
                    {'id': 'close_stale_primary', 'offset_minutes': -25, 'function': 'close_stale_positions', 'label': 'Close Stale: Primary'},
                    {'id': 'close_stale_fallback', 'offset_minutes': -15, 'function': 'close_stale_positions_fallback', 'label': 'Close Stale: Fallback'},
                    {'id': 'emergency_hard_close', 'offset_minutes': -5, 'function': 'emergency_hard_close', 'label': 'Emergency Hard Close'},
                ],
                'post_close_tasks': [
                    {'id': 'log_equity_snapshot', 'offset_minutes': 5, 'function': 'log_equity_snapshot', 'label': 'Log Equity Snapshot'},
                    {'id': 'reconcile_and_analyze', 'offset_minutes': 8, 'function': 'reconcile_and_analyze', 'label': 'Reconcile & Analyze'},
                    {'id': 'brier_reconciliation', 'offset_minutes': 11, 'function': 'run_brier_reconciliation', 'label': 'Brier Reconciliation'},
                    {'id': 'sentinel_effectiveness', 'offset_minutes': 14, 'function': 'sentinel_effectiveness_check', 'label': 'Sentinel Effectiveness Check'},
                    {'id': 'eod_shutdown', 'offset_minutes': 17, 'function': 'cancel_and_stop_monitoring', 'label': 'End-of-Day Shutdown'},
                ],
            },
        },
    }


# === Session schedule: KC ===

def test_session_schedule_kc():
    """KC session schedule has 4 signals within trading hours."""
    config = _make_session_config('KC')
    result = build_schedule(config)

    # Count signal tasks
    signals = [t for t in result if t.func_name == 'guarded_generate_orders']
    assert len(signals) == 4

    # KC opens at 04:15, closes at 13:30
    kc_open = time(4, 15)
    kc_close = time(13, 30)

    # All signals must be within trading hours
    for sig in signals:
        assert sig.time_et >= kc_open, f"Signal {sig.id} at {sig.time_et} is before market open {kc_open}"
        assert sig.time_et <= kc_close, f"Signal {sig.id} at {sig.time_et} is after market close {kc_close}"

    # Total tasks: 3 pre-open + 4 signals + 1 intra + 4 pre-close + 5 post-close = 17
    assert len(result) == 17

    # All IDs must be unique
    ids = [t.id for t in result]
    assert len(set(ids)) == len(ids)


def test_session_schedule_kc_pre_close_before_market():
    """All pre-close tasks are before KC market close (13:30)."""
    config = _make_session_config('KC')
    result = build_schedule(config)

    kc_close = time(13, 30)
    pre_close_ids = {'audit_pre_close', 'close_stale_primary', 'close_stale_fallback', 'emergency_hard_close'}

    for task in result:
        if task.id in pre_close_ids:
            assert task.time_et < kc_close, (
                f"Pre-close task {task.id} at {task.time_et} is not before close {kc_close}"
            )


def test_session_schedule_kc_post_close_after_market():
    """All post-close tasks are after KC market close (13:30)."""
    config = _make_session_config('KC')
    result = build_schedule(config)

    kc_close = time(13, 30)
    post_close_ids = {'log_equity_snapshot', 'reconcile_and_analyze', 'brier_reconciliation',
                      'sentinel_effectiveness', 'eod_shutdown'}

    for task in result:
        if task.id in post_close_ids:
            assert task.time_et > kc_close, (
                f"Post-close task {task.id} at {task.time_et} is not after close {kc_close}"
            )


# === Session schedule: CC (Cocoa) ===

def test_session_schedule_cc():
    """CC session schedule adapts to 04:45 open."""
    config = _make_session_config('CC')
    result = build_schedule(config)

    signals = [t for t in result if t.func_name == 'guarded_generate_orders']
    assert len(signals) == 4

    # CC opens at 04:45, closes at 13:30
    cc_open = time(4, 45)
    cc_close = time(13, 30)

    # All signals within CC trading hours
    for sig in signals:
        assert sig.time_et >= cc_open, f"Signal {sig.id} at {sig.time_et} before CC open {cc_open}"
        assert sig.time_et <= cc_close, f"Signal {sig.id} at {sig.time_et} after CC close {cc_close}"

    # First signal should be later than KC's first signal (CC opens 30 min later)
    kc_config = _make_session_config('KC')
    kc_result = build_schedule(kc_config)
    kc_signals = [t for t in kc_result if t.func_name == 'guarded_generate_orders']

    assert signals[0].time_et > kc_signals[0].time_et


# === Signal distribution ===

def test_session_schedule_signal_distribution_1():
    """Single signal at start_pct."""
    config = _make_session_config('KC', signal_count=1, start_pct=0.05, end_pct=0.80)
    result = build_schedule(config)
    signals = [t for t in result if t.func_name == 'guarded_generate_orders']
    assert len(signals) == 1
    assert signals[0].id == 'signal_open'


def test_session_schedule_signal_distribution_3():
    """3 signals evenly distributed."""
    config = _make_session_config('KC', signal_count=3, start_pct=0.05, end_pct=0.80)
    result = build_schedule(config)
    signals = [t for t in result if t.func_name == 'guarded_generate_orders']
    assert len(signals) == 3

    # Check IDs
    assert signals[0].id == 'signal_open'
    assert signals[1].id == 'signal_early'
    assert signals[2].id == 'signal_mid'


def test_session_schedule_signal_distribution_5():
    """5 signals use all named slots."""
    config = _make_session_config('KC', signal_count=5, start_pct=0.05, end_pct=0.95)
    result = build_schedule(config)
    signals = [t for t in result if t.func_name == 'guarded_generate_orders']
    assert len(signals) == 5

    expected_ids = ['signal_open', 'signal_early', 'signal_mid', 'signal_late', 'signal_5']
    actual_ids = [s.id for s in signals]
    assert actual_ids == expected_ids


def test_session_schedule_sorted_by_time():
    """All tasks sorted chronologically."""
    config = _make_session_config('KC')
    result = build_schedule(config)

    times = [(t.time_et.hour, t.time_et.minute) for t in result]
    assert times == sorted(times)


# === Pre-close tasks submit orders ===

def test_pre_close_tasks_before_market_close():
    """All order-submitting tasks have times before market close."""
    config = _make_session_config('KC')
    result = build_schedule(config)
    kc_close = time(13, 30)

    order_funcs = {'close_stale_positions', 'close_stale_positions_fallback',
                   'emergency_hard_close', 'run_position_audit_cycle'}

    for task in result:
        if task.func_name in order_funcs:
            assert task.time_et <= kc_close, (
                f"Order-submitting task {task.id} ({task.func_name}) at {task.time_et} "
                f"is after market close {kc_close}"
            )


# === Post-close tasks are read-only ===

def test_post_close_tasks_no_orders():
    """Post-close tasks use only read-only/admin functions."""
    config = _make_session_config('KC')
    result = build_schedule(config)
    kc_close = time(13, 30)

    order_funcs = {'guarded_generate_orders', 'close_stale_positions',
                   'close_stale_positions_fallback', 'emergency_hard_close',
                   'run_position_audit_cycle'}

    for task in result:
        if task.time_et > kc_close:
            assert task.func_name not in order_funcs, (
                f"Post-close task {task.id} uses order-submitting function {task.func_name}"
            )


# === get_trading_cutoff ===

def test_get_trading_cutoff_session_mode():
    """Session mode cutoff = close - cutoff_before_close_minutes."""
    config = _make_session_config('KC', cutoff_minutes=78)
    h, m = get_trading_cutoff(config)

    # KC closes at 13:30. 13:30 - 78min = 12:12 ET
    assert h == 12
    assert m == 12


def test_get_trading_cutoff_absolute_fallback():
    """Absolute mode uses daily_trading_cutoff_et from config."""
    config = {
        'schedule': {
            'mode': 'absolute',
            'daily_trading_cutoff_et': {'hour': 10, 'minute': 45},
        }
    }
    h, m = get_trading_cutoff(config)
    assert h == 10
    assert m == 45


def test_get_trading_cutoff_no_config():
    """Missing config falls back to default cutoff."""
    h, m = get_trading_cutoff({})
    assert h == 10
    assert m == 45


# === is_market_open with config ===

def test_is_market_open_with_config_weekday_within_hours():
    """is_market_open(config) returns True during KC trading hours on weekday."""
    from trading_bot.utils import is_market_open

    # Wednesday 10:00 AM ET — well within KC 04:15-13:30
    ny_tz = pytz.timezone('America/New_York')
    mock_time = ny_tz.localize(datetime(2026, 1, 14, 10, 0, 0))  # Wednesday

    config = {'symbol': 'KC'}

    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time.astimezone(pytz.UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = is_market_open(config)

    assert result is True


def test_is_market_open_with_config_after_close():
    """is_market_open(config) returns False after KC close."""
    from trading_bot.utils import is_market_open

    # Wednesday 15:00 ET — after KC 13:30 close
    ny_tz = pytz.timezone('America/New_York')
    mock_time = ny_tz.localize(datetime(2026, 1, 14, 15, 0, 0))

    config = {'symbol': 'KC'}

    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time.astimezone(pytz.UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = is_market_open(config)

    assert result is False


def test_is_market_open_without_config_backward_compat():
    """is_market_open() without config uses hardcoded fallback."""
    from trading_bot.utils import is_market_open

    # Wednesday 10:00 ET — within hardcoded 03:30-14:00
    ny_tz = pytz.timezone('America/New_York')
    mock_time = ny_tz.localize(datetime(2026, 1, 14, 10, 0, 0))

    with patch('trading_bot.utils.datetime') as mock_dt:
        mock_dt.now.return_value = mock_time.astimezone(pytz.UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        result = is_market_open()

    assert result is True


# === Duplicate ID detection ===

def test_session_schedule_duplicate_id_raises():
    """Duplicate IDs in session template raise ValueError."""
    config = _make_session_config('KC')
    # Add a duplicate ID in pre_close_tasks
    config['schedule']['session_template']['pre_close_tasks'].append(
        {'id': 'emergency_hard_close', 'offset_minutes': -3, 'function': 'emergency_hard_close', 'label': 'Dup'}
    )

    with pytest.raises(ValueError, match="Duplicate schedule task ID.*emergency_hard_close"):
        build_schedule(config)


# === Fallback to absolute mode ===

def test_build_schedule_absolute_mode_still_works():
    """Absolute mode with tasks array still works."""
    config = {
        'schedule': {
            'mode': 'absolute',
            'tasks': [
                {'id': 'sig1', 'time_et': '09:00', 'function': 'guarded_generate_orders', 'label': 'Sig 1'},
                {'id': 'sig2', 'time_et': '11:00', 'function': 'guarded_generate_orders', 'label': 'Sig 2'},
            ]
        }
    }
    result = build_schedule(config)
    assert len(result) == 2
    assert result[0].id == 'sig1'
    assert result[1].id == 'sig2'


# === commodity_filter in intra-session tasks ===

def test_commodity_filter_skips_non_matching():
    """Intra-session task with commodity_filter='NG' is skipped for KC engine."""
    config = _make_session_config('KC')
    config['schedule']['session_template']['intra_session_tasks'].append(
        {'id': 'eia_storage_signal', 'session_pct': 0.29, 'function': 'guarded_generate_orders',
         'label': 'Signal: EIA Storage Report', 'commodity_filter': 'NG'}
    )
    result = build_schedule(config)

    # The NG-only task should NOT appear in KC schedule
    ids = [t.id for t in result]
    assert 'eia_storage_signal' not in ids


def test_commodity_filter_includes_matching():
    """Intra-session task with commodity_filter='NG' IS included for NG engine."""
    config = _make_session_config('NG')
    config['schedule']['session_template']['intra_session_tasks'].append(
        {'id': 'eia_storage_signal', 'session_pct': 0.29, 'function': 'guarded_generate_orders',
         'label': 'Signal: EIA Storage Report', 'commodity_filter': 'NG'}
    )
    result = build_schedule(config)

    ids = [t.id for t in result]
    assert 'eia_storage_signal' in ids


def test_commodity_filter_absent_includes_all():
    """Intra-session task without commodity_filter is included for any engine."""
    config = _make_session_config('KC')
    # The default audit_mid_session has no commodity_filter
    result = build_schedule(config)

    ids = [t.id for t in result]
    assert 'audit_mid_session' in ids
