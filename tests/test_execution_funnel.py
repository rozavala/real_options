"""Tests for execution funnel logger, FunnelStage enum, and ExitReason enum."""
import os
import pytest
import pandas as pd

from trading_bot.execution_funnel import (
    log_funnel_event, FunnelStage, SCHEMA_COLUMNS, set_data_dir, get_funnel_df
)
from trading_bot.exit_reasons import ExitReason, classify_exit_reason


class TestFunnelStageEnum:
    def test_all_stages_are_strings(self):
        for stage in FunnelStage:
            assert isinstance(stage.value, str)

    def test_mece_no_duplicates(self):
        values = [s.value for s in FunnelStage]
        assert len(values) == len(set(values))

    def test_stage_ordering_completeness(self):
        """Verify all major pipeline stages are represented."""
        stage_names = {s.value for s in FunnelStage}
        required = {
            'COUNCIL_DECISION', 'CONVICTION_GATE', 'COMPLIANCE_AUDIT',
            'CONFIDENCE_THRESHOLD', 'ORDER_PLACED', 'ORDER_FILLED',
            'ORDER_CANCELLED', 'POSITION_OPENED', 'POSITION_CLOSED',
        }
        assert required.issubset(stage_names), f"Missing stages: {required - stage_names}"


class TestExitReasonEnum:
    def test_all_reasons_are_strings(self):
        for reason in ExitReason:
            assert isinstance(reason.value, str)

    def test_no_duplicate_values(self):
        values = [r.value for r in ExitReason]
        assert len(values) == len(set(values))

    def test_options_specific_exits_exist(self):
        assert ExitReason.EXPIRED_WORTHLESS
        assert ExitReason.EXPIRED_ITM
        assert ExitReason.ASSIGNED
        assert ExitReason.EXERCISED

    def test_emergency_exits_exist(self):
        assert ExitReason.EMERGENCY_HARD_CLOSE
        assert ExitReason.EMERGENCY_HARD_CLOSE_RETRY
        assert ExitReason.CATASTROPHE_FILL


class TestClassifyExitReason:
    def test_exact_matches(self):
        assert classify_exit_reason("Strategy Execution") == ExitReason.STRATEGY_EXECUTION
        assert classify_exit_reason("Stale Position Close") == ExitReason.STALE_CLOSE
        assert classify_exit_reason("Risk Management Closure") == ExitReason.RISK_MANAGEMENT

    def test_stop_loss_dynamic(self):
        result = classify_exit_reason("Stop-Loss (Hit 55.2% of Max Risk)")
        assert result == ExitReason.STOP_LOSS

    def test_take_profit_dynamic(self):
        result = classify_exit_reason("Take-Profit (Captured 82.1% of Max Profit)")
        assert result == ExitReason.TAKE_PROFIT

    def test_emergency_variants(self):
        assert classify_exit_reason("EMERGENCY_HARD_CLOSE") == ExitReason.EMERGENCY_HARD_CLOSE
        assert classify_exit_reason("EMERGENCY_HARD_CLOSE_RETRY") == ExitReason.EMERGENCY_HARD_CLOSE_RETRY
        assert classify_exit_reason("EMERGENCY_HARD_CLOSE_RETRY_PARTIAL") == ExitReason.EMERGENCY_HARD_CLOSE_RETRY_PARTIAL

    def test_weekly_close_variants(self):
        assert classify_exit_reason("Friday Weekly Close") == ExitReason.FRIDAY_WEEKLY_CLOSE
        assert classify_exit_reason("Holiday Tomorrow (Weekly Close)") == ExitReason.HOLIDAY_WEEKLY_CLOSE

    def test_contradict_variants(self):
        assert classify_exit_reason("CONTRADICT: position already closed") == ExitReason.CONTRADICT_CLOSED
        assert classify_exit_reason("CONTRADICT: closed on direction reversal") == ExitReason.CONTRADICT_REVERSAL

    def test_reconciliation(self):
        assert classify_exit_reason("RECONCILIATION_MISSING") == ExitReason.RECONCILIATION_MISSING
        assert classify_exit_reason("PHANTOM_RECONCILIATION") == ExitReason.PHANTOM_RECONCILIATION

    def test_spread_close(self):
        assert classify_exit_reason("Spread Close: abc12345") == ExitReason.SPREAD_CLOSE

    def test_empty_and_none(self):
        assert classify_exit_reason("") == ExitReason.STRATEGY_EXECUTION
        assert classify_exit_reason(None) == ExitReason.STRATEGY_EXECUTION

    def test_unknown_fallback(self):
        result = classify_exit_reason("some completely unknown reason xyz")
        assert result == ExitReason.RISK_MANAGEMENT


class TestLogFunnelEvent:
    def test_creates_csv_with_header(self, tmp_path):
        set_data_dir(str(tmp_path))
        result = log_funnel_event(
            cycle_id="KC-test1",
            contract="KCK6 (202605)",
            stage=FunnelStage.COUNCIL_DECISION,
            outcome="PASS",
            detail="test event",
            regime="TRENDING",
            source="BACKFILL",
        )
        assert result is True
        csv_path = os.path.join(str(tmp_path), 'execution_funnel.csv')
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert list(df.columns) == SCHEMA_COLUMNS
        assert len(df) == 1
        assert df.iloc[0]['stage'] == 'COUNCIL_DECISION'

    def test_appends_without_duplicate_headers(self, tmp_path):
        set_data_dir(str(tmp_path))
        log_funnel_event(cycle_id="t1", contract="c1", stage=FunnelStage.COUNCIL_DECISION, source="BACKFILL")
        log_funnel_event(cycle_id="t2", contract="c2", stage=FunnelStage.ORDER_FILLED, source="BACKFILL")
        csv_path = os.path.join(str(tmp_path), 'execution_funnel.csv')
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert df.iloc[0]['cycle_id'] == 't1'
        assert df.iloc[1]['cycle_id'] == 't2'

    def test_never_raises(self):
        # Even with impossible path, should return False not raise
        set_data_dir("/nonexistent/path/that/cannot/exist")
        result = log_funnel_event(
            cycle_id="test", contract="test",
            stage=FunnelStage.ORDER_PLACED,
            source="BACKFILL",
        )
        assert isinstance(result, bool)

    def test_detail_truncation(self, tmp_path):
        set_data_dir(str(tmp_path))
        long_detail = "x" * 1000
        log_funnel_event(
            cycle_id="t1", contract="c1",
            stage=FunnelStage.COUNCIL_DECISION,
            detail=long_detail,
            source="BACKFILL",
        )
        csv_path = os.path.join(str(tmp_path), 'execution_funnel.csv')
        df = pd.read_csv(csv_path)
        assert len(df.iloc[0]['detail']) <= 500

    def test_source_flag(self, tmp_path):
        set_data_dir(str(tmp_path))
        log_funnel_event(
            cycle_id="bf1", contract="c1",
            stage=FunnelStage.COUNCIL_DECISION, source="BACKFILL",
        )
        log_funnel_event(
            cycle_id="rt1", contract="c2",
            stage=FunnelStage.ORDER_PLACED, source="BACKFILL",
        )
        csv_path = os.path.join(str(tmp_path), 'execution_funnel.csv')
        df = pd.read_csv(csv_path)
        assert set(df['source']) == {'BACKFILL'}

    def test_numeric_fields_rounded(self, tmp_path):
        set_data_dir(str(tmp_path))
        log_funnel_event(
            cycle_id="t1", contract="c1",
            stage=FunnelStage.ORDER_FILLED,
            price_snapshot=123.456789,
            bid=1.23456789,
            ask=1.34567890,
            fill_price=1.28901234,
            source="BACKFILL",
        )
        csv_path = os.path.join(str(tmp_path), 'execution_funnel.csv')
        df = pd.read_csv(csv_path)
        assert df.iloc[0]['price_snapshot'] == 123.46
        assert df.iloc[0]['bid'] == 1.2346
        assert df.iloc[0]['fill_price'] == 1.289


class TestGetFunnelDf:
    def test_returns_empty_for_missing_file(self, tmp_path):
        df = get_funnel_df(ticker="NONEXISTENT")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == SCHEMA_COLUMNS

    def test_loads_existing_data(self, tmp_path):
        set_data_dir(str(tmp_path))
        log_funnel_event(
            cycle_id="t1", contract="c1",
            stage=FunnelStage.ORDER_PLACED,
            source="BACKFILL",
        )
        set_data_dir(str(tmp_path))
        df = get_funnel_df()
        assert len(df) == 1
        assert df.iloc[0]['stage'] == 'ORDER_PLACED'
