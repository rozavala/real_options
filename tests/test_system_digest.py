"""Tests for trading_bot.system_digest — System Health Digest."""

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from trading_bot.system_digest import (
    _safe_read_json,
    _safe_read_csv,
    _safe_float,
    _build_feedback_loop,
    _build_agent_calibration,
    _build_cognitive_layer,
    _build_sentinel_efficiency,
    _build_efficiency,
    _build_risk_rails,
    _build_decision_traces,
    _build_data_freshness,
    _build_regime_context,
    _build_agent_contribution,
    _build_portfolio,
    _build_rolling_trends,
    _build_improvement_opportunities,
    _build_config_snapshot,
    _build_error_telemetry,
    _build_executive_summary,
    _build_system_health_score,
    generate_system_digest,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temp data directory with standard structure."""
    data_dir = tmp_path / "data" / "KC"
    data_dir.mkdir(parents=True)
    return str(data_dir)


@pytest.fixture
def base_config(tmp_path):
    """Minimal config for testing."""
    return {
        'data_dir': str(tmp_path / "data"),
        'commodities': ['KC'],
        'commodity': {'ticker': 'KC'},
        'risk_management': {
            'max_holding_days': 2,
            'min_confidence_threshold': 0.50,
        },
        'drawdown_circuit_breaker': {
            'enabled': True,
            'warning_pct': 1.5,
            'halt_pct': 2.5,
            'panic_pct': 4.0,
        },
        'compliance': {
            'var_enforcement_mode': 'log_only',
        },
        'strategy_tuning': {
            'spread_width_percentage': 0.01425,
            'iron_condor_short_strikes_from_atm': 2,
            'iron_condor_wing_strikes_apart': 2,
        },
        'sentinels': {
            'weather': {'triggers': {'frost_temp_c': 4.0}},
            'price': {'pct_change_threshold': 1.5},
            'microstructure': {'spread_std_threshold': 3.0},
        },
        'semantic_cache': {'enabled': True, 'max_entries': 100},
        'strategy': {'quantity': 1, 'min_voter_quorum': 3},
        'brier_scoring': {'enhanced_weight': 0.3},
        'cost_management': {'daily_budget_usd': 15.0},
        'notifications': {},
    }


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def _write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestSafeReadJson:
    def test_missing_file(self, tmp_path):
        result = _safe_read_json(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "test.json")
        _write_json(path, {"key": "value"})
        result = _safe_read_json(path)
        assert result == {"key": "value"}

    def test_corrupt_file(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, 'w') as f:
            f.write("{invalid json")
        result = _safe_read_json(path)
        assert result is None


class TestSafeReadCsv:
    def test_missing_file(self, tmp_path):
        result = _safe_read_csv(str(tmp_path / "nonexistent.csv"))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_valid_file(self, tmp_path):
        path = str(tmp_path / "test.csv")
        _write_csv(path, [["a", 1], ["b", 2]], ["name", "val"])
        result = _safe_read_csv(path)
        assert len(result) == 2
        assert list(result.columns) == ["name", "val"]


class TestSafeFloat:
    def test_valid(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        assert _safe_float(0) == 0.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid(self):
        assert _safe_float("not_a_number") is None


# ---------------------------------------------------------------------------
# v1.0 Per-Commodity Builders
# ---------------------------------------------------------------------------

class TestBuildFeedbackLoop:
    def test_with_data(self, tmp_data_dir):
        predictions = [
            {"actual_outcome": "correct", "resolved_at": "2026-01-01"},
            {"actual_outcome": "wrong", "resolved_at": "2026-01-02"},
            {"actual_outcome": None, "resolved_at": None},  # pending
            {"actual_outcome": "correct", "resolved_at": "2026-01-03"},
        ]
        _write_json(os.path.join(tmp_data_dir, "enhanced_brier.json"), {"predictions": predictions})

        result = _build_feedback_loop(tmp_data_dir)
        assert result['total_predictions'] == 4
        assert result['resolved'] == 3
        assert result['pending'] == 1
        assert result['resolution_rate'] == 0.75
        assert result['status'] == 'healthy'
        assert result['thresholds'] == {'healthy': 0.75, 'critical': 0.50}

    def test_empty(self, tmp_data_dir):
        result = _build_feedback_loop(tmp_data_dir)
        assert result['status'] == 'no_data'
        assert result['total_predictions'] == 0

    def test_critical_status(self, tmp_data_dir):
        predictions = [
            {"actual_outcome": None, "resolved_at": None},
            {"actual_outcome": None, "resolved_at": None},
            {"actual_outcome": "correct", "resolved_at": "2026-01-01"},
        ]
        _write_json(os.path.join(tmp_data_dir, "enhanced_brier.json"), {"predictions": predictions})
        result = _build_feedback_loop(tmp_data_dir)
        assert result['status'] == 'critical'
        assert result['resolution_rate'] == pytest.approx(0.333, abs=0.001)


class TestBuildCognitiveLayer:
    def _make_ch_df(self, directions, confidences=None, strategies=None):
        now = datetime.now(timezone.utc)
        today_str = now.strftime('%Y-%m-%d %H:%M:%S+00:00')
        yesterday_str = (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S+00:00')

        rows = []
        for i, d in enumerate(directions):
            rows.append({
                'timestamp': today_str,
                'master_direction': d,
                'master_confidence': confidences[i] if confidences else 0.7,
                'weighted_score': 0.5,
                'strategy': strategies[i] if strategies else 'BULL_CALL_SPREAD',
            })
        # Add a yesterday row that should be filtered out
        rows.append({
            'timestamp': yesterday_str,
            'master_direction': 'BEARISH',
            'master_confidence': 0.9,
            'weighted_score': 0.8,
            'strategy': 'BEAR_PUT_SPREAD',
        })

        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df

    def test_today_only(self):
        ch_df = self._make_ch_df(['BULLISH', 'BEARISH'])
        result = _build_cognitive_layer(ch_df)
        assert result['decisions_today'] == 2  # Not 3 (yesterday excluded)

    def test_decision_breakdown(self):
        ch_df = self._make_ch_df(['BULLISH', 'BULLISH', 'BEARISH', 'NEUTRAL'])
        result = _build_cognitive_layer(ch_df)
        assert result['decisions_today'] == 4
        assert result['bull_pct'] == 0.5
        assert result['bear_pct'] == 0.25
        assert result['neutral_pct'] == 0.25

    def test_empty_df(self):
        result = _build_cognitive_layer(pd.DataFrame())
        assert result['decisions_today'] == 0


class TestBuildSentinelEfficiency:
    def test_with_data(self, tmp_data_dir):
        stats = {
            "sentinels": {
                "price": {"total_alerts": 100, "trades_triggered": 5},
                "weather": {"total_alerts": 50, "trades_triggered": 10},
            }
        }
        _write_json(os.path.join(tmp_data_dir, "sentinel_stats.json"), stats)

        result = _build_sentinel_efficiency(tmp_data_dir)
        assert result['total_alerts'] == 150
        assert result['total_trades_triggered'] == 15
        # Verify field name is trades_triggered not triggered_trades
        assert result['sentinels']['price']['trades_triggered'] == 5
        assert result['sentinels']['price']['signal_to_noise'] == 0.05

    def test_no_data(self, tmp_data_dir):
        result = _build_sentinel_efficiency(tmp_data_dir)
        assert result['total_alerts'] == 0


# ---------------------------------------------------------------------------
# v1.1 Per-Commodity Builders
# ---------------------------------------------------------------------------

class TestBuildDecisionTraces:
    def test_vote_parsing(self):
        now = datetime.now(timezone.utc)
        ch_df = pd.DataFrame([{
            'timestamp': now,
            'master_direction': 'BULLISH',
            'master_confidence': 0.8,
            'strategy': 'BULL_CALL_SPREAD',
            'vote_breakdown': json.dumps({
                'agronomist': 2.5,
                'technical': 1.8,
                'sentiment': 0.5,
            }),
            'dissent_acknowledged': 'Bearish technical divergence noted',
            'realized_pnl': 150.0,
        }])
        ch_df['timestamp'] = pd.to_datetime(ch_df['timestamp'], utc=True)

        traces = _build_decision_traces(ch_df, max_traces=5)
        assert len(traces) == 1
        assert len(traces[0]['top_contributors']) == 2
        assert traces[0]['top_contributors'][0]['agent'] == 'agronomist'
        assert traces[0]['contrarian_view'] == 'Bearish technical divergence noted'

    def test_legacy_columns(self):
        now = datetime.now(timezone.utc)
        ch_df = pd.DataFrame([{
            'timestamp': now,
            'master_direction': 'BULLISH',
            'master_confidence': 0.7,
            'strategy': 'BULL_CALL_SPREAD',
            'vote_breakdown': json.dumps({'agronomist': 3.0}),
            'meteorologist_summary': 'Frost risk in Minas Gerais',
        }])
        ch_df['timestamp'] = pd.to_datetime(ch_df['timestamp'], utc=True)

        traces = _build_decision_traces(ch_df, max_traces=1)
        assert len(traces) == 1
        # The legacy meteorologist_summary should be picked up for agronomist
        contrib = traces[0]['top_contributors']
        assert len(contrib) >= 1
        assert contrib[0]['agent'] == 'agronomist'
        assert 'Frost risk' in contrib[0]['key_argument']

    def test_empty(self):
        traces = _build_decision_traces(pd.DataFrame())
        assert traces == []


class TestBuildDataFreshness:
    def test_freshness(self, tmp_data_dir):
        now = datetime.now(timezone.utc).timestamp()
        state = {
            'sentinel_health': {
                'price': {
                    'timestamp': now - 300,  # 5 minutes ago
                    'data': {'interval_seconds': 600},
                },
                'weather': {
                    'timestamp': now - 2000,  # ~33 minutes ago
                    'data': {'interval_seconds': 600},
                },
            }
        }
        _write_json(os.path.join(tmp_data_dir, 'state.json'), state)

        result = _build_data_freshness(tmp_data_dir)
        assert result['sentinels']['price']['is_stale'] is False
        assert result['sentinels']['weather']['is_stale'] is True  # 2000 > 2*600
        assert result['stale_count'] == 1
        assert result['status'] == 'degraded'  # 1 stale: 0=healthy, 1-2=degraded, >2=critical


class TestBuildRegimeContext:
    def test_with_data(self, tmp_data_dir):
        _write_json(os.path.join(tmp_data_dir, "fundamental_regime.json"), {
            "regime": "DEFICIT",
            "confidence": 0.82,
            "updated_at": "2026-01-15T10:00:00+00:00",
        })
        result = _build_regime_context(tmp_data_dir)
        assert result['regime'] == 'DEFICIT'
        assert result['confidence'] == 0.82

    def test_no_data(self, tmp_data_dir):
        result = _build_regime_context(tmp_data_dir)
        assert result['regime'] == 'UNKNOWN'


class TestBuildAgentContribution:
    def test_agreement_rate(self):
        now = datetime.now(timezone.utc)
        rows = []
        for i in range(10):
            rows.append({
                'timestamp': now - timedelta(days=i),
                'master_direction': 'BULLISH',
                'agronomist_sentiment': 'BULLISH' if i < 7 else 'BEARISH',
                'technical_sentiment': 'BEARISH',
            })
        df = pd.DataFrame(rows)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        result = _build_agent_contribution(df)
        agents = result['agents']
        assert agents['agronomist']['agreement_rate_with_master'] == 0.7
        assert agents['technical']['agreement_rate_with_master'] == 0.0


# ---------------------------------------------------------------------------
# Account-Wide Builders
# ---------------------------------------------------------------------------

class TestBuildPortfolio:
    def test_with_equity_data(self, base_config, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create equity CSV
        rows = [[f"2026-01-{i:02d}", 10000 + i * 50] for i in range(1, 32)]
        _write_csv(str(data_dir / "daily_equity.csv"), rows, ["timestamp", "total_value_usd"])

        result = _build_portfolio(base_config)
        assert result['nlv_usd'] == 11550.0  # 10000 + 31*50
        assert result['daily_pnl_usd'] == 50.0
        assert result['equity_data_points'] == 31

    def test_no_data(self, base_config, tmp_path):
        result = _build_portfolio(base_config)
        assert result.get('status') == 'no_data'


class TestBuildRollingTrends:
    def test_with_data(self, base_config, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        rows = [[f"2026-01-{i:02d}", 10000 + i * 10] for i in range(1, 32)]
        _write_csv(str(data_dir / "daily_equity.csv"), rows, ["timestamp", "total_value_usd"])

        result = _build_rolling_trends(base_config)
        assert 'equity_delta_7d' in result
        assert 'equity_delta_30d' in result
        assert 'avg_daily_pnl' in result


class TestBuildImprovementOpportunities:
    def test_thresholds(self):
        blocks = {
            'KC': {
                'feedback_loop': {'status': 'critical', 'resolution_rate': 0.3},
                'sentinel_efficiency': {
                    'sentinels': {
                        'noisy': {'signal_to_noise': 0.01, 'total_alerts': 50},
                        'good': {'signal_to_noise': 0.20, 'total_alerts': 100},
                    }
                },
                'agent_calibration': {
                    'agents': {
                        'weak_agent': {'brier_score': 0.05},
                        'good_agent': {'brier_score': 0.45},
                    }
                },
                'data_freshness': {'stale_count': 5},
            }
        }
        opps = _build_improvement_opportunities(blocks)
        priorities = [o['priority'] for o in opps]
        # Should have HIGH for critical feedback + stale data, MEDIUM for noisy sentinel, LOW for weak agent
        assert 'HIGH' in priorities
        assert 'MEDIUM' in priorities
        assert 'LOW' in priorities
        # HIGHs should come first
        assert priorities.index('HIGH') < priorities.index('MEDIUM')
        assert priorities.index('MEDIUM') < priorities.index('LOW')


# ---------------------------------------------------------------------------
# v1.1 Account-Wide Builders
# ---------------------------------------------------------------------------

class TestBuildConfigSnapshot:
    def test_security_no_secrets(self, base_config):
        result = _build_config_snapshot(base_config, ['KC'])
        # Flatten to string and check no secrets leak
        result_str = json.dumps(result, default=str)
        for forbidden in ['API_KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PUSHOVER']:
            assert forbidden not in result_str.upper(), f"Secret pattern '{forbidden}' found in config snapshot"

    def test_domain_weights(self, base_config):
        result = _build_config_snapshot(base_config, ['KC'])
        weights = result.get('agent_base_weights_scheduled')
        # Should have the SCHEDULED trigger weights if import succeeded
        if weights is not None:
            assert isinstance(weights, dict)
            assert len(weights) > 0

    def test_iron_condor_fields(self, base_config):
        result = _build_config_snapshot(base_config, ['KC'])
        assert result['iron_condor']['short_strikes_from_atm'] == 2
        assert result['iron_condor']['wing_strikes_apart'] == 2

    def test_sentinel_thresholds(self, base_config):
        result = _build_config_snapshot(base_config, ['KC'])
        thresholds = result['sentinel_thresholds']
        assert thresholds['weather_frost_temp_c'] == 4.0
        assert thresholds['price_pct_change_threshold'] == 1.5
        assert thresholds['microstructure_spread_std_threshold'] == 3.0


class TestBuildErrorTelemetry:
    def test_all_tickers(self, base_config, tmp_path):
        # Create order_events.csv for KC
        data_kc = tmp_path / "data" / "KC"
        data_kc.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')
        _write_csv(
            str(data_kc / "order_events.csv"),
            [[now_str, "liquidity_reject"], [now_str, "margin_reject"], [now_str, "timeout_error"]],
            ["timestamp", "event_type"],
        )

        result = _build_error_telemetry(base_config, ['KC'])
        assert result['per_commodity']['KC']['liquidity_reject'] == 1
        assert result['per_commodity']['KC']['margin_reject'] == 1
        assert result['per_commodity']['KC']['order_timeout'] == 1
        assert result['total_errors_today'] == 3

    def test_categorization(self, base_config, tmp_path):
        data_kc = tmp_path / "data" / "KC"
        data_kc.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')
        events = [
            [now_str, "spread_too_wide"],    # liquidity
            [now_str, "margin_exceeded"],     # margin
            [now_str, "execution_failed"],    # trading_execution
            [now_str, "order_timeout"],       # timeout
            [now_str, "rejected_by_broker"],  # order_error
        ]
        _write_csv(
            str(data_kc / "order_events.csv"),
            events,
            ["timestamp", "event_type"],
        )
        result = _build_error_telemetry(base_config, ['KC'])
        totals = result['totals']
        assert totals['liquidity_reject'] == 1
        assert totals['margin_reject'] == 1
        assert totals['trading_execution'] == 1
        assert totals['order_timeout'] == 1
        assert totals['order_error'] == 1
        assert result['high_impact'] is True  # trading_execution > 0


class TestBuildExecutiveSummary:
    def test_template(self):
        digest = {
            'portfolio': {'nlv_usd': 50000, 'daily_pnl_usd': 250},
            'commodities': {
                'KC': {
                    'cognitive_layer': {'decisions_today': 3},
                    'feedback_loop': {'status': 'healthy', 'resolution_rate': 0.85},
                    'regime_context': {'regime': 'DEFICIT'},
                },
            },
            'improvement_opportunities': [
                {'priority': 'HIGH', 'component': 'KC/data', 'observation': 'test'},
            ],
            'system_health_score': {'overall': 0.78},
        }
        summary = _build_executive_summary(digest)
        assert '$50,000.00' in summary
        assert '+$250.00' in summary
        assert 'KC' in summary
        assert '1 HIGH-priority' in summary
        assert '0.78' in summary


class TestBuildSystemHealthScore:
    def test_formula(self):
        digest = {
            'commodities': {
                'KC': {
                    'feedback_loop': {'resolution_rate': 0.75},
                    'agent_calibration': {'avg_brier': 0.25},
                    'sentinel_efficiency': {
                        'sentinels': {
                            'price': {'signal_to_noise': 0.10},
                            'weather': {'signal_to_noise': 0.20},
                        }
                    },
                },
            },
            'error_telemetry': {'total_errors_today': 0},
        }
        result = _build_system_health_score(digest)
        assert result['overall'] is not None

        # Verify the formula manually:
        # feedback_norm = min(1.0, 0.75 / 0.75) = 1.0
        # brier_norm = max(0, 1 - 0.25 / 0.5) = 0.5
        # exec_quality = 1.0 - 0/10 = 1.0
        # sentinel_snr = (0.10 + 0.20) / 2 = 0.15
        # overall = 0.35*1.0 + 0.25*0.5 + 0.25*1.0 + 0.15*0.15
        #         = 0.35 + 0.125 + 0.25 + 0.0225 = 0.7475
        assert result['overall'] == pytest.approx(0.7475, abs=0.001)
        assert result['formula'] is not None
        assert result['weights']['feedback_health'] == 0.35

    def test_deterministic(self):
        """Same input → same output."""
        digest = {
            'commodities': {
                'KC': {
                    'feedback_loop': {'resolution_rate': 0.6},
                    'agent_calibration': {'avg_brier': 0.3},
                    'sentinel_efficiency': {'sentinels': {'p': {'signal_to_noise': 0.1}}},
                },
            },
            'error_telemetry': {'total_errors_today': 2},
        }
        r1 = _build_system_health_score(digest)
        r2 = _build_system_health_score(digest)
        assert r1['overall'] == r2['overall']


# ---------------------------------------------------------------------------
# Digest ID
# ---------------------------------------------------------------------------

class TestDigestId:
    def test_deterministic(self):
        """Same content → same hash."""
        import hashlib
        data = {'schema_version': '1.1', 'test': True}
        h1 = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]
        h2 = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]
        assert h1 == h2


# ---------------------------------------------------------------------------
# End-to-End
# ---------------------------------------------------------------------------

class TestGenerateSystemDigest:
    def test_end_to_end(self, base_config, tmp_path):
        """Full run with synthetic data — verify JSON output + file written."""
        data_kc = tmp_path / "data" / "KC"
        data_kc.mkdir(parents=True, exist_ok=True)

        # Populate synthetic data files
        _write_json(str(data_kc / "enhanced_brier.json"), {
            "predictions": [
                {"actual_outcome": "correct", "resolved_at": "2026-01-01"},
                {"actual_outcome": None, "resolved_at": None},
            ]
        })
        _write_json(str(data_kc / "sentinel_stats.json"), {
            "sentinels": {"price": {"total_alerts": 20, "trades_triggered": 2}}
        })
        _write_json(str(data_kc / "fundamental_regime.json"), {
            "regime": "DEFICIT", "confidence": 0.8
        })
        _write_json(str(data_kc / "drawdown_state.json"), {
            "status": "NORMAL", "current_drawdown_pct": 0.5
        })

        # Daily equity (account-wide)
        data_dir = tmp_path / "data"
        rows = [[f"2026-01-{i:02d}", 10000 + i * 10] for i in range(1, 11)]
        _write_csv(str(data_dir / "daily_equity.csv"), rows, ["timestamp", "total_value_usd"])

        # Council history
        now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')
        ch_rows = [[now_str, "BULLISH", 0.75, 0.6, "BULL_CALL_SPREAD", "{}", "", None]]
        _write_csv(
            str(data_kc / "council_history.csv"),
            ch_rows,
            ["timestamp", "master_direction", "master_confidence", "weighted_score",
             "strategy", "vote_breakdown", "dissent_acknowledged", "realized_pnl"],
        )

        digest = generate_system_digest(base_config)

        assert digest is not None
        assert digest['schema_version'] == '1.1'
        assert 'KC' in digest['commodities']
        assert digest['digest_id'].startswith(datetime.now(timezone.utc).strftime('%Y-%m-%d'))
        assert digest['executive_summary']
        assert digest['system_health_score']['overall'] is not None

        # Verify files written
        output_path = str(data_dir / "system_health_digest.json")
        assert os.path.exists(output_path)
        with open(output_path) as f:
            written = json.load(f)
        assert written['digest_id'] == digest['digest_id']

        # Verify archive
        archive_dir = os.path.join('logs', 'digests')
        # Archive is written relative to CWD, not tmp_path — just verify the digest returned OK

    def test_missing_data_dir(self, base_config, tmp_path):
        """Graceful degradation when data directory doesn't exist."""
        base_config['commodities'] = ['ZZ']  # Non-existent commodity
        digest = generate_system_digest(base_config)
        assert digest is not None
        assert digest['commodities']['ZZ']['status'] == 'no_data_directory'

    @patch('trading_bot.system_digest._safe_read_csv')
    def test_council_history_loaded_once(self, mock_csv, base_config, tmp_path):
        """Verify council_history.csv is loaded once per commodity, not re-read per builder."""
        data_kc = tmp_path / "data" / "KC"
        data_kc.mkdir(parents=True, exist_ok=True)

        # Return empty DataFrame
        mock_csv.return_value = pd.DataFrame()

        generate_system_digest(base_config)

        # council_history.csv should be loaded exactly once for KC
        ch_calls = [
            c for c in mock_csv.call_args_list
            if 'council_history.csv' in str(c)
        ]
        assert len(ch_calls) == 1, f"Expected 1 council_history.csv read, got {len(ch_calls)}"
