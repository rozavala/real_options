"""Tests for config-driven schedule infrastructure.

Covers: build_schedule, _build_default_schedule, apply_schedule_offset,
get_next_task (both ScheduledTask list and legacy dict), per-instance
completion tracking, and active_schedule.json format.
"""

import pytest
import json
import os
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import pytz

from orchestrator import (
    ScheduledTask,
    FUNCTION_REGISTRY,
    build_schedule,
    _build_default_schedule,
    apply_schedule_offset,
    get_next_task,
    schedule,
)


# === build_schedule ===

def test_build_schedule_from_config():
    """Config tasks array produces correct ScheduledTask list."""
    config = {
        'schedule': {
            'tasks': [
                {'id': 'signal_early', 'time_et': '09:00', 'function': 'guarded_generate_orders', 'label': 'Signal: Early'},
                {'id': 'signal_peak',  'time_et': '15:00', 'function': 'guarded_generate_orders', 'label': 'Signal: Peak'},
                {'id': 'audit_am',     'time_et': '13:30', 'function': 'run_position_audit_cycle', 'label': 'Audit: AM'},
            ]
        }
    }
    result = build_schedule(config)

    assert len(result) == 3
    # Should be sorted by time
    assert result[0].id == 'signal_early'
    assert result[0].time_et == time(9, 0)
    assert result[0].func_name == 'guarded_generate_orders'
    assert result[0].label == 'Signal: Early'
    assert callable(result[0].function)

    assert result[1].id == 'audit_am'
    assert result[1].time_et == time(13, 30)

    assert result[2].id == 'signal_peak'
    assert result[2].time_et == time(15, 0)


def test_build_schedule_fallback():
    """No 'tasks' key in config falls back to 19-task default."""
    config = {'schedule': {'dev_offset_minutes': 0}}
    result = build_schedule(config)
    assert len(result) == 19
    # Check a few known defaults
    ids = [t.id for t in result]
    assert 'signal_early' in ids
    assert 'signal_euro' in ids
    assert 'signal_us_open' in ids
    assert 'signal_peak' in ids
    assert 'signal_settlement' in ids
    assert 'audit_morning' in ids
    assert 'audit_post_close' in ids
    assert 'audit_pre_close' in ids


def test_build_schedule_empty_config():
    """Empty config falls back to defaults."""
    result = build_schedule({})
    assert len(result) == 19


def test_build_schedule_duplicate_id_raises():
    """Duplicate task IDs raise ValueError."""
    config = {
        'schedule': {
            'tasks': [
                {'id': 'my_task', 'time_et': '09:00', 'function': 'guarded_generate_orders', 'label': 'A'},
                {'id': 'my_task', 'time_et': '10:00', 'function': 'guarded_generate_orders', 'label': 'B'},
            ]
        }
    }
    with pytest.raises(ValueError, match="Duplicate schedule task ID.*my_task"):
        build_schedule(config)


def test_build_schedule_unknown_function_skipped():
    """Unknown function names are skipped with a warning."""
    config = {
        'schedule': {
            'tasks': [
                {'id': 'good', 'time_et': '09:00', 'function': 'guarded_generate_orders', 'label': 'Good'},
                {'id': 'bad',  'time_et': '10:00', 'function': 'nonexistent_function',    'label': 'Bad'},
            ]
        }
    }
    result = build_schedule(config)
    assert len(result) == 1
    assert result[0].id == 'good'


def test_build_schedule_session_mode():
    """Session mode builds schedule from commodity profile trading hours."""
    config = {
        'symbol': 'KC',
        'schedule': {
            'mode': 'session',
            'session_template': {
                'signal_pcts': [0.20, 0.62, 0.80],
                'cutoff_before_close_minutes': 78,
                'pre_open_tasks': [
                    {'id': 'start_monitoring', 'offset_minutes': -45, 'function': 'start_monitoring', 'label': 'Start'},
                ],
                'intra_session_tasks': [],
                'pre_close_tasks': [
                    {'id': 'emergency_hard_close', 'offset_minutes': -5, 'function': 'emergency_hard_close', 'label': 'EHC'},
                ],
                'post_close_tasks': [
                    {'id': 'eod_shutdown', 'offset_minutes': 17, 'function': 'cancel_and_stop_monitoring', 'label': 'EOD'},
                ],
            }
        }
    }
    result = build_schedule(config)
    assert len(result) == 6  # 1 pre-open + 3 signals + 1 pre-close + 1 post-close
    ids = [t.id for t in result]
    assert 'start_monitoring' in ids
    assert 'signal_open' in ids
    assert 'signal_mid' in ids
    assert 'emergency_hard_close' in ids
    assert 'eod_shutdown' in ids
    # All times should be within a reasonable day range
    for t in result:
        assert t.time_et.hour >= 3  # earliest is ~45 min before 04:15 open


# === _build_default_schedule ===

def test_default_schedule_structure():
    """Default schedule has 19 tasks with unique IDs."""
    defaults = _build_default_schedule()
    assert len(defaults) == 19
    ids = [t.id for t in defaults]
    assert len(set(ids)) == 19, "All task IDs must be unique"

    # All functions must be callable
    for task in defaults:
        assert callable(task.function)
        assert task.func_name == task.function.__name__
        assert task.label  # non-empty label


def test_default_schedule_matches_config():
    """Config.json schedule mode and structure are valid."""
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    schedule_cfg = config['schedule']
    mode = schedule_cfg.get('mode', 'absolute')

    if mode == 'session':
        # Session mode: validate session_template structure
        tmpl = schedule_cfg['session_template']
        if 'signal_pcts' in tmpl:
            assert len(tmpl['signal_pcts']) >= 1
            assert all(0 <= p <= 1.0 for p in tmpl['signal_pcts'])
        else:
            assert tmpl['signal_count'] >= 1
            assert 0 <= tmpl['signal_start_pct'] < tmpl['signal_end_pct'] <= 1.0
        assert tmpl['cutoff_before_close_minutes'] > 0

        # All task groups must have valid function names
        from orchestrator import FUNCTION_REGISTRY
        for group in ('pre_open_tasks', 'intra_session_tasks', 'pre_close_tasks', 'post_close_tasks'):
            for entry in tmpl.get(group, []):
                assert entry['function'] in FUNCTION_REGISTRY, (
                    f"Unknown function '{entry['function']}' in {group}"
                )
                assert entry['id']  # non-empty ID
    else:
        # Absolute mode: IDs must match defaults
        config_ids = [t['id'] for t in schedule_cfg['tasks']]
        default_ids = [t.id for t in _build_default_schedule()]
        assert config_ids == default_ids, "config.json task IDs must match built-in defaults"


# === Backward-compat module-level schedule dict ===

def test_module_schedule_dict_exists():
    """Module-level `schedule` dict exists for backward compat."""
    assert isinstance(schedule, dict)
    assert len(schedule) > 0
    # All values should be callables
    for t, func in schedule.items():
        assert isinstance(t, time)
        assert callable(func)


# === apply_schedule_offset ===

def test_apply_offset_preserves_ids():
    """Offset shifts times but preserves IDs and labels."""
    tasks = [
        ScheduledTask(id='signal_early', time_et=time(9, 0),
                      function=MagicMock(), func_name='guarded_generate_orders',
                      label='Signal: Early'),
        ScheduledTask(id='audit_am', time_et=time(13, 30),
                      function=MagicMock(), func_name='run_position_audit_cycle',
                      label='Audit: AM'),
    ]
    shifted = apply_schedule_offset(tasks, offset_minutes=-30)

    assert len(shifted) == 2
    assert shifted[0].id == 'signal_early'
    assert shifted[0].time_et == time(8, 30)
    assert shifted[0].label == 'Signal: Early'
    assert shifted[0].func_name == 'guarded_generate_orders'

    assert shifted[1].id == 'audit_am'
    assert shifted[1].time_et == time(13, 0)


def test_apply_offset_legacy_dict():
    """Offset on legacy dict returns a dict."""
    mock_fn = MagicMock(__name__='mock_fn')
    original = {time(9, 0): mock_fn}
    shifted = apply_schedule_offset(original, offset_minutes=30)
    assert isinstance(shifted, dict)
    assert time(9, 30) in shifted
    assert shifted[time(9, 30)] is mock_fn


# === get_next_task ===

def test_get_next_task_with_scheduled_tasks():
    """get_next_task with list[ScheduledTask] returns (datetime, ScheduledTask)."""
    mock_fn = MagicMock(__name__='mock_fn')
    tasks = [
        ScheduledTask(id='early', time_et=time(9, 0), function=mock_fn,
                      func_name='mock_fn', label='Early Task'),
        ScheduledTask(id='late', time_et=time(15, 0), function=mock_fn,
                      func_name='mock_fn', label='Late Task'),
    ]

    # Wednesday 08:00 NY → next should be 'early' at 09:00
    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC
    now_ny = ny_tz.localize(datetime(2026, 1, 14, 8, 0, 0))  # Wednesday
    now_utc = now_ny.astimezone(utc)

    next_run, next_task = get_next_task(now_utc, tasks)

    assert isinstance(next_task, ScheduledTask)
    assert next_task.id == 'early'
    assert next_task.label == 'Early Task'

    expected_ny = ny_tz.localize(datetime(2026, 1, 14, 9, 0, 0))
    assert next_run == expected_ny.astimezone(utc)


def test_get_next_task_legacy_dict():
    """get_next_task with dict returns (datetime, callable) for backward compat."""
    mock_task = MagicMock(__name__='mock_task')
    test_schedule = {time(9, 0): mock_task}

    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC
    now_ny = ny_tz.localize(datetime(2026, 1, 14, 8, 0, 0))  # Wednesday
    now_utc = now_ny.astimezone(utc)

    next_run, next_task = get_next_task(now_utc, test_schedule)

    # Legacy path returns the callable directly, not a ScheduledTask
    assert next_task is mock_task
    assert not isinstance(next_task, ScheduledTask)


def test_get_next_task_picks_correct_instance():
    """With multiple same-function tasks, returns the correct instance by time."""
    mock_fn = MagicMock(__name__='guarded_generate_orders')
    tasks = [
        ScheduledTask(id='signal_early', time_et=time(9, 0), function=mock_fn,
                      func_name='guarded_generate_orders', label='Early'),
        ScheduledTask(id='signal_peak', time_et=time(15, 0), function=mock_fn,
                      func_name='guarded_generate_orders', label='Peak'),
    ]

    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC

    # At 10:00 → should pick signal_peak (09:00 already passed)
    now_ny = ny_tz.localize(datetime(2026, 1, 14, 10, 0, 0))  # Wednesday, 10 AM
    now_utc = now_ny.astimezone(utc)

    _, next_task = get_next_task(now_utc, tasks)
    assert next_task.id == 'signal_peak'


# === Per-instance completion tracking ===

def test_per_instance_completion_tracking():
    """Different task_ids are tracked independently by task_tracker."""
    from trading_bot.task_tracker import record_task_completion, has_task_completed_today

    # Record two different signal completions
    record_task_completion('signal_early')
    record_task_completion('signal_euro')

    assert has_task_completed_today('signal_early')
    assert has_task_completed_today('signal_euro')
    assert not has_task_completed_today('signal_us_open')
    assert not has_task_completed_today('signal_peak')


# === active_schedule.json format ===

def test_active_schedule_json_format():
    """Verify the active_schedule.json structure includes id and label fields."""
    # Simulate what main() writes
    task_list = _build_default_schedule()

    schedule_data = {
        "generated_at": datetime.now().isoformat(),
        "env": "DEV",
        "offset_minutes": 0,
        "tasks": [
            {
                "id": task.id,
                "time_et": task.time_et.strftime('%H:%M'),
                "name": task.func_name,
                "label": task.label,
            }
            for task in task_list
        ]
    }

    assert len(schedule_data['tasks']) == 19

    # Every entry has id, time_et, name, and label
    for entry in schedule_data['tasks']:
        assert 'id' in entry
        assert 'time_et' in entry
        assert 'name' in entry
        assert 'label' in entry
        assert entry['id']  # non-empty
        assert entry['label']  # non-empty

    # Verify unique IDs
    ids = [e['id'] for e in schedule_data['tasks']]
    assert len(set(ids)) == 19

    # Verify signal tasks have distinct IDs but same function name
    signal_entries = [e for e in schedule_data['tasks'] if e['name'] == 'guarded_generate_orders']
    assert len(signal_entries) == 5
    signal_ids = [e['id'] for e in signal_entries]
    assert len(set(signal_ids)) == 5  # all unique


# === FUNCTION_REGISTRY ===

def test_function_registry_covers_all_defaults():
    """All functions used in default schedule exist in FUNCTION_REGISTRY."""
    defaults = _build_default_schedule()
    for task in defaults:
        assert task.func_name in FUNCTION_REGISTRY, f"{task.func_name} missing from FUNCTION_REGISTRY"
        assert FUNCTION_REGISTRY[task.func_name] is task.function
