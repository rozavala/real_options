"""Tests for trading_bot/prompt_trace.py"""

import csv
import pytest
import pandas as pd
from unittest.mock import patch

from trading_bot.prompt_trace import (
    hash_persona,
    PromptTraceRecord,
    PromptTraceCollector,
    get_prompt_trace_df,
    set_data_dir,
    SCHEMA_COLUMNS,
)


class TestHashPersona:
    def test_deterministic(self):
        """Same input always produces same hash."""
        assert hash_persona("test") == hash_persona("test")

    def test_length(self):
        """Hash is always 12 characters."""
        assert len(hash_persona("")) == 12
        assert len(hash_persona("x" * 10000)) == 12

    def test_empty_input(self):
        """Empty string produces a valid hex hash."""
        result = hash_persona("")
        assert len(result) == 12
        assert all(c in '0123456789abcdef' for c in result)

    def test_different_inputs(self):
        """Different inputs produce different hashes."""
        assert hash_persona("persona A") != hash_persona("persona B")


class TestPromptTraceRecord:
    def test_defaults(self):
        """All fields have sensible defaults."""
        rec = PromptTraceRecord()
        assert rec.phase == ""
        assert rec.prompt_source == "legacy"
        assert rec.demo_count == 0
        assert rec.reflexion_applied is False
        assert rec.latency_ms == 0.0

    def test_custom_values(self):
        rec = PromptTraceRecord(
            phase="research",
            agent="agronomist",
            prompt_source="dspy_optimized",
            demo_count=3,
        )
        assert rec.phase == "research"
        assert rec.agent == "agronomist"
        assert rec.demo_count == 3


class TestPromptTraceCollector:
    def test_record_and_flush(self, tmp_path):
        """Records are written to CSV on flush."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector(cycle_id="KC-abc123", commodity="KC", contract="KCN6")
            collector.record(PromptTraceRecord(phase="research", agent="agronomist"))
            count = collector.flush()

        assert count == 1
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]['phase'] == 'research'
        assert df.iloc[0]['agent'] == 'agronomist'
        assert df.iloc[0]['cycle_id'] == 'KC-abc123'
        assert df.iloc[0]['commodity'] == 'KC'

    def test_cycle_field_injection(self, tmp_path):
        """Collector injects cycle_id, commodity, contract if not set on record."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector(cycle_id="KC-xyz789", commodity="KC", contract="KCH6")
            rec = PromptTraceRecord(phase="debate", agent="permabear")
            collector.record(rec)
            collector.flush()

        df = pd.read_csv(csv_path)
        assert df.iloc[0]['cycle_id'] == 'KC-xyz789'
        assert df.iloc[0]['commodity'] == 'KC'
        assert df.iloc[0]['contract'] == 'KCH6'

    def test_empty_flush(self, tmp_path):
        """Flushing with no records returns 0 and doesn't create file."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector()
            count = collector.flush()

        assert count == 0
        assert not csv_path.exists()

    def test_multi_phase(self, tmp_path):
        """Multiple records from different phases are all written."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector(cycle_id="KC-multi", commodity="KC")
            collector.record(PromptTraceRecord(phase="research", agent="agronomist"))
            collector.record(PromptTraceRecord(phase="debate", agent="permabear"))
            collector.record(PromptTraceRecord(phase="decision", agent="master"))
            count = collector.flush()

        assert count == 3
        df = pd.read_csv(csv_path)
        assert list(df['phase']) == ['research', 'debate', 'decision']

    def test_schema_columns(self, tmp_path):
        """CSV has all 20 schema columns."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector(cycle_id="KC-schema")
            collector.record(PromptTraceRecord(phase="research", agent="test"))
            collector.flush()

        df = pd.read_csv(csv_path)
        assert len(SCHEMA_COLUMNS) == 20
        for col in SCHEMA_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_nan_defaults(self, tmp_path):
        """Numeric fields default to 0, not NaN."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector()
            collector.record(PromptTraceRecord(phase="test"))
            collector.flush()

        df = pd.read_csv(csv_path)
        assert df.iloc[0]['demo_count'] == 0
        assert df.iloc[0]['prompt_tokens'] == 0
        assert df.iloc[0]['latency_ms'] == 0.0

    def test_concurrent_collectors(self, tmp_path):
        """Two collectors writing to the same file don't corrupt it."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            c1 = PromptTraceCollector(cycle_id="KC-c1", commodity="KC")
            c2 = PromptTraceCollector(cycle_id="KC-c2", commodity="KC")
            c1.record(PromptTraceRecord(phase="research", agent="agronomist"))
            c2.record(PromptTraceRecord(phase="debate", agent="permabear"))
            c1.flush()
            c2.flush()

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert set(df['cycle_id']) == {'KC-c1', 'KC-c2'}

    def test_append_mode(self, tmp_path):
        """Subsequent flushes append to existing file."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            c1 = PromptTraceCollector(cycle_id="KC-first")
            c1.record(PromptTraceRecord(phase="research"))
            c1.flush()

            c2 = PromptTraceCollector(cycle_id="KC-second")
            c2.record(PromptTraceRecord(phase="debate"))
            c2.flush()

        df = pd.read_csv(csv_path)
        assert len(df) == 2

    def test_flush_clears_buffer(self, tmp_path):
        """After flush, buffer is empty."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector()
            collector.record(PromptTraceRecord(phase="research"))
            assert collector.flush() == 1
            assert collector.flush() == 0  # Second flush has nothing

    def test_record_preserves_explicit_fields(self, tmp_path):
        """Fields explicitly set on the record are not overwritten by collector defaults."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector(cycle_id="KC-default", commodity="KC")
            rec = PromptTraceRecord(phase="research", cycle_id="KC-explicit", commodity="CC")
            collector.record(rec)
            collector.flush()

        df = pd.read_csv(csv_path)
        assert df.iloc[0]['cycle_id'] == 'KC-explicit'
        assert df.iloc[0]['commodity'] == 'CC'


class TestGetPromptTraceDf:
    def test_missing_file(self, tmp_path):
        """Returns empty DataFrame when file doesn't exist."""
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(tmp_path / "nope.csv")):
            df = get_prompt_trace_df()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == SCHEMA_COLUMNS

    def test_commodity_filter(self, tmp_path):
        """Filters by commodity when specified."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector()
            collector.record(PromptTraceRecord(phase="research", commodity="KC"))
            collector.record(PromptTraceRecord(phase="research", commodity="CC"))
            collector.flush()

            df = get_prompt_trace_df(commodity="KC")

        assert len(df) == 1
        assert df.iloc[0]['commodity'] == 'KC'

    def test_forward_compat(self, tmp_path):
        """Old schema files (missing columns) get defaults added."""
        csv_path = tmp_path / "prompt_trace.csv"
        # Write a CSV with only a few columns
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'cycle_id', 'phase'])
            writer.writeheader()
            writer.writerow({'timestamp': '2026-02-22T00:00:00Z', 'cycle_id': 'KC-old', 'phase': 'research'})

        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            df = get_prompt_trace_df()

        assert 'model_provider' in df.columns
        assert 'latency_ms' in df.columns
        assert len(df) == 1

    def test_set_data_dir(self, tmp_path):
        """set_data_dir updates the file path."""
        import trading_bot.prompt_trace as pt
        original = pt.PROMPT_TRACE_PATH
        try:
            set_data_dir(str(tmp_path))
            assert pt.PROMPT_TRACE_PATH == str(tmp_path / "prompt_trace.csv")
        finally:
            pt.PROMPT_TRACE_PATH = original


class TestDashboardNaN:
    def test_numeric_fields_not_nan(self, tmp_path):
        """Numeric fields should not be NaN when loaded for dashboard display."""
        csv_path = tmp_path / "prompt_trace.csv"
        with patch('trading_bot.prompt_trace.PROMPT_TRACE_PATH', str(csv_path)):
            collector = PromptTraceCollector()
            collector.record(PromptTraceRecord(phase="research"))
            collector.flush()

            df = get_prompt_trace_df()

        numeric_cols = ['demo_count', 'tms_context_count', 'grounded_freshness_hours',
                        'prompt_tokens', 'completion_tokens', 'latency_ms']
        for col in numeric_cols:
            assert not df[col].isna().any(), f"{col} has NaN values"
