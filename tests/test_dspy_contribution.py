"""Tests for DSPy contribution-aware scoring and data pipeline."""

import csv
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from trading_bot.dspy_optimizer import (
    CouncilDataset,
    _compute_metric_score,
    _safe_float,
    evaluate_baseline,
    should_suggest_enable,
)


# ---------------------------------------------------------------------------
# _compute_metric_score
# ---------------------------------------------------------------------------


class TestComputeMetricScore:
    """Test the standalone metric function used by optimizer + eval."""

    def test_positive_contribution_matching_direction(self):
        """Positive contribution + matching direction → high score."""
        score = _compute_metric_score(
            pred_direction="BULLISH",
            pred_confidence=0.8,
            actual="BULLISH",
            contribution_score=0.6,
            example_direction="BULLISH",
        )
        assert score == pytest.approx(0.7 + 0.3 * 0.6)

    def test_positive_contribution_different_direction(self):
        """Positive contribution but prediction differs from example → 0.2."""
        score = _compute_metric_score(
            pred_direction="BEARISH",
            pred_confidence=0.8,
            actual="BULLISH",
            contribution_score=0.6,
            example_direction="BULLISH",
        )
        assert score == 0.2

    def test_negative_contribution_matching_direction(self):
        """Negative contribution + matching direction → 0.0 (reproducing bad call)."""
        score = _compute_metric_score(
            pred_direction="BULLISH",
            pred_confidence=0.8,
            actual="BEARISH",
            contribution_score=-0.5,
            example_direction="BULLISH",
        )
        assert score == 0.0

    def test_negative_contribution_different_direction(self):
        """Negative contribution + different direction → 0.5 (avoided mistake)."""
        score = _compute_metric_score(
            pred_direction="BEARISH",
            pred_confidence=0.8,
            actual="BEARISH",
            contribution_score=-0.5,
            example_direction="BULLISH",
        )
        assert score == 0.5

    def test_zero_contribution_neutral_on_noise(self):
        """Zero contribution + NEUTRAL on NEUTRAL outcome → good abstention."""
        score = _compute_metric_score(
            pred_direction="NEUTRAL",
            pred_confidence=0.7,
            actual="NEUTRAL",
            contribution_score=0.0,
            example_direction="NEUTRAL",
        )
        assert score == pytest.approx(0.4 + 0.2 * 0.7)

    def test_zero_contribution_neutral_on_real_move(self):
        """Zero contribution + NEUTRAL on directional outcome → missed signal."""
        score = _compute_metric_score(
            pred_direction="NEUTRAL",
            pred_confidence=0.7,
            actual="BULLISH",
            contribution_score=0.0,
            example_direction="NEUTRAL",
        )
        assert score == 0.15

    def test_zero_contribution_directional_prediction(self):
        """Zero contribution + directional prediction → 0.25."""
        score = _compute_metric_score(
            pred_direction="BULLISH",
            pred_confidence=0.8,
            actual="NEUTRAL",
            contribution_score=0.0,
            example_direction="NEUTRAL",
        )
        assert score == 0.25

    def test_no_contribution_neutral_fallback(self):
        """No contribution data + NEUTRAL → small credit."""
        score = _compute_metric_score(
            pred_direction="NEUTRAL",
            pred_confidence=0.5,
            actual="BULLISH",
            contribution_score=None,
        )
        assert score == 0.15

    def test_no_contribution_correct_direction(self):
        """No contribution data + correct direction → high score."""
        score = _compute_metric_score(
            pred_direction="BULLISH",
            pred_confidence=0.9,
            actual="BULLISH",
            contribution_score=None,
        )
        # direction_match=1.0, calibration=1-|0.9-1.0|=0.9
        assert score == pytest.approx(0.7 + 0.3 * 0.9)

    def test_no_contribution_wrong_direction(self):
        """No contribution data + wrong direction → low score."""
        score = _compute_metric_score(
            pred_direction="BEARISH",
            pred_confidence=0.9,
            actual="BULLISH",
            contribution_score=None,
        )
        # direction_match=0, calibration=1-|0.9-0.0|=0.1
        assert score == pytest.approx(0.0 + 0.3 * 0.1)

    def test_positive_contribution_clamped_at_1(self):
        """Large positive contribution clamped at 1.0 in abs()."""
        score = _compute_metric_score(
            pred_direction="BULLISH",
            pred_confidence=0.9,
            actual="BULLISH",
            contribution_score=2.5,
            example_direction="BULLISH",
        )
        assert score == pytest.approx(0.7 + 0.3 * 1.0)

    def test_always_neutral_is_suboptimal(self):
        """Always-NEUTRAL strategy scores lower than selective commitment.

        Given 40% NEUTRAL outcomes and 60% directional, an always-NEUTRAL
        agent should score worse than one that commits selectively.
        """
        # Always-NEUTRAL agent: 0.4 * 0.5 + 0.6 * 0.15 = 0.29
        always_neutral = (
            0.4 * _compute_metric_score("NEUTRAL", 0.5, "NEUTRAL", 0.0, "NEUTRAL")
            + 0.6 * _compute_metric_score("NEUTRAL", 0.5, "BULLISH", 0.0, "NEUTRAL")
        )

        # Selective agent: commits on directional, abstains on noise.
        # Gets directional right 50% of the time.
        selective = (
            0.4 * _compute_metric_score("NEUTRAL", 0.5, "NEUTRAL", 0.0, "NEUTRAL")
            + 0.3 * _compute_metric_score("BULLISH", 0.8, "BULLISH", 0.5, "BULLISH")
            + 0.3 * _compute_metric_score("BULLISH", 0.8, "BEARISH", -0.5, "BULLISH")
        )

        assert selective > always_neutral


# ---------------------------------------------------------------------------
# _build_contribution_lookup
# ---------------------------------------------------------------------------


class TestBuildContributionLookup:
    """Test the data pipeline from council_history to contribution scores."""

    def _make_dataset(self, tmp_path, council_rows, brier_preds):
        """Create a CouncilDataset with minimal fixtures."""
        data_dir = tmp_path / "KC"
        data_dir.mkdir()

        # Write enhanced_brier.json
        brier_path = data_dir / "enhanced_brier.json"
        brier_path.write_text(json.dumps({"predictions": brier_preds}))

        # Write council_history.csv
        if council_rows:
            csv_path = data_dir / "council_history.csv"
            fieldnames = list(council_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in council_rows:
                    writer.writerow(row)

        return CouncilDataset(str(data_dir))

    def test_basic_lookup(self, tmp_path):
        """Lookup is built from vote_breakdown + actual outcome."""
        council_rows = [{
            "cycle_id": "KC-001",
            "contract": "KCN6",
            "entry_price": "100.0",
            "exit_price": "110.0",
            "trigger_type": "scheduled",
            "thesis_strength": "HIGH",
            "master_decision": "BULLISH",
            "prediction_type": "DIRECTIONAL",
            "strategy_type": "",
            "volatility_outcome": "",
            "actual_trend_direction": "BULLISH",
            "vote_breakdown": json.dumps([
                {"agent": "macro", "direction": "BULLISH", "confidence": 0.8, "final_weight": 1.0},
                {"agent": "technical", "direction": "BEARISH", "confidence": 0.6, "final_weight": 1.0},
            ]),
        }]
        brier_preds = [
            {"cycle_id": "KC-001", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.8, "prob_neutral": 0.1, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
            {"cycle_id": "KC-001", "agent": "technical", "actual_outcome": "BULLISH",
             "prob_bullish": 0.2, "prob_neutral": 0.2, "prob_bearish": 0.6, "timestamp": "2026-01-01"},
        ]

        ds = self._make_dataset(tmp_path, council_rows, brier_preds)
        predictions = ds.load()

        # macro said BULLISH, outcome BULLISH → positive contribution
        macro_score = predictions["macro"][0].get("contribution_score")
        assert macro_score is not None
        assert macro_score > 0

        # technical said BEARISH, outcome BULLISH → negative contribution
        tech_score = predictions["technical"][0].get("contribution_score")
        assert tech_score is not None
        assert tech_score < 0

    def test_materiality_threshold_makes_neutral(self, tmp_path):
        """Small price moves are reclassified as NEUTRAL via materiality threshold."""
        council_rows = [{
            "cycle_id": "KC-002",
            "contract": "KCN6",
            "entry_price": "100.0",
            "exit_price": "100.3",  # 0.3% < KC threshold of 0.6%
            "trigger_type": "scheduled",
            "thesis_strength": "MODERATE",
            "master_decision": "BULLISH",
            "prediction_type": "DIRECTIONAL",
            "strategy_type": "",
            "volatility_outcome": "",
            "actual_trend_direction": "BULLISH",
            "vote_breakdown": json.dumps([
                {"agent": "macro", "direction": "BULLISH", "confidence": 0.7, "final_weight": 1.0},
            ]),
        }]
        brier_preds = [
            {"cycle_id": "KC-002", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.7, "prob_neutral": 0.2, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
        ]

        ds = self._make_dataset(tmp_path, council_rows, brier_preds)
        predictions = ds.load()

        # Actual outcome reclassified as NEUTRAL due to small move (0.3% < 0.6% threshold)
        # When actual_outcome is NEUTRAL, _compute_score PATH 1 returns 0.0
        # (market didn't move enough — nobody credited or blamed)
        score = predictions["macro"][0].get("contribution_score")
        assert score is not None
        assert score == 0.0

    def test_empty_vote_breakdown_skipped(self, tmp_path):
        """Rows with empty vote_breakdown produce no lookup entries."""
        council_rows = [{
            "cycle_id": "KC-003",
            "contract": "KCN6",
            "entry_price": "100.0",
            "exit_price": "110.0",
            "trigger_type": "scheduled",
            "thesis_strength": "HIGH",
            "master_decision": "BULLISH",
            "prediction_type": "DIRECTIONAL",
            "strategy_type": "",
            "volatility_outcome": "",
            "actual_trend_direction": "BULLISH",
            "vote_breakdown": "",
        }]
        brier_preds = [
            {"cycle_id": "KC-003", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.8, "prob_neutral": 0.1, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
        ]

        ds = self._make_dataset(tmp_path, council_rows, brier_preds)
        predictions = ds.load()

        # No contribution score because vote_breakdown was empty
        assert predictions["macro"][0].get("contribution_score") is None

    def test_unresolved_cycle_skipped(self, tmp_path):
        """Rows with NaN actual_trend_direction are skipped."""
        council_rows = [{
            "cycle_id": "KC-004",
            "contract": "KCN6",
            "entry_price": "100.0",
            "exit_price": "",
            "trigger_type": "scheduled",
            "thesis_strength": "HIGH",
            "master_decision": "BULLISH",
            "prediction_type": "DIRECTIONAL",
            "strategy_type": "",
            "volatility_outcome": "",
            "actual_trend_direction": "NaN",
            "vote_breakdown": json.dumps([
                {"agent": "macro", "direction": "BULLISH", "confidence": 0.8, "final_weight": 1.0},
            ]),
        }]
        brier_preds = [
            {"cycle_id": "KC-004", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.8, "prob_neutral": 0.1, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
        ]

        ds = self._make_dataset(tmp_path, council_rows, brier_preds)
        predictions = ds.load()

        assert predictions["macro"][0].get("contribution_score") is None

    def test_malformed_json_vote_breakdown_skipped(self, tmp_path):
        """Rows with invalid JSON vote_breakdown don't crash, just skip."""
        council_rows = [{
            "cycle_id": "KC-005",
            "contract": "KCN6",
            "entry_price": "100.0",
            "exit_price": "110.0",
            "trigger_type": "scheduled",
            "thesis_strength": "HIGH",
            "master_decision": "BULLISH",
            "prediction_type": "DIRECTIONAL",
            "strategy_type": "",
            "volatility_outcome": "",
            "actual_trend_direction": "BULLISH",
            "vote_breakdown": "not valid json{{{",
        }]
        brier_preds = [
            {"cycle_id": "KC-005", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.8, "prob_neutral": 0.1, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
        ]

        ds = self._make_dataset(tmp_path, council_rows, brier_preds)
        predictions = ds.load()  # Should not raise

        assert predictions["macro"][0].get("contribution_score") is None

    def test_no_council_history(self, tmp_path):
        """Dataset works fine without council_history.csv — no enrichment."""
        data_dir = tmp_path / "KC"
        data_dir.mkdir()
        brier_path = data_dir / "enhanced_brier.json"
        brier_path.write_text(json.dumps({"predictions": [
            {"cycle_id": "KC-001", "agent": "macro", "actual_outcome": "BULLISH",
             "prob_bullish": 0.8, "prob_neutral": 0.1, "prob_bearish": 0.1, "timestamp": "2026-01-01"},
        ]}))

        ds = CouncilDataset(str(data_dir))
        predictions = ds.load()

        assert predictions["macro"][0].get("contribution_score") is None
        assert predictions["macro"][0].get("market_context") is not None


# ---------------------------------------------------------------------------
# Stats and evaluation with contribution metrics
# ---------------------------------------------------------------------------


class TestStatsWithMetric:
    """Test that stats() and evaluate_baseline() include metric scores."""

    def _make_dataset(self, tmp_path, examples):
        """Create a dataset from pre-built prediction dicts."""
        data_dir = tmp_path / "KC"
        data_dir.mkdir()
        brier_preds = []
        for ex in examples:
            brier_preds.append({
                "cycle_id": ex.get("cycle_id", "KC-001"),
                "agent": ex["agent"],
                "actual_outcome": ex["actual"],
                "prob_bullish": ex.get("prob_bullish", 0.5),
                "prob_neutral": ex.get("prob_neutral", 0.3),
                "prob_bearish": ex.get("prob_bearish", 0.2),
                "timestamp": "2026-01-01",
            })
        brier_path = data_dir / "enhanced_brier.json"
        brier_path.write_text(json.dumps({"predictions": brier_preds}))
        return CouncilDataset(str(data_dir))

    def test_stats_includes_avg_metric_score(self, tmp_path):
        """stats() returns avg_metric_score per agent."""
        examples = [
            {"agent": "macro", "actual": "BULLISH", "prob_bullish": 0.8,
             "prob_neutral": 0.1, "prob_bearish": 0.1, "cycle_id": "KC-001"},
            {"agent": "macro", "actual": "BEARISH", "prob_bullish": 0.2,
             "prob_neutral": 0.1, "prob_bearish": 0.7, "cycle_id": "KC-002"},
        ]
        ds = self._make_dataset(tmp_path, examples)
        stats = ds.stats()
        assert "avg_metric_score" in stats["agents"]["macro"]
        assert isinstance(stats["agents"]["macro"]["avg_metric_score"], float)

    def test_stats_includes_contribution_coverage(self, tmp_path):
        """stats() reports what fraction of examples have contribution scores."""
        examples = [
            {"agent": "macro", "actual": "BULLISH", "prob_bullish": 0.8,
             "prob_neutral": 0.1, "prob_bearish": 0.1, "cycle_id": "KC-001"},
        ]
        ds = self._make_dataset(tmp_path, examples)
        stats = ds.stats()
        # Without council_history, no contribution scores → coverage = 0
        assert stats["agents"]["macro"]["contribution_coverage"] == 0.0

    def test_evaluate_baseline_includes_metric(self, tmp_path):
        """evaluate_baseline() propagates avg_metric_score from stats."""
        examples = [
            {"agent": "macro", "actual": "BULLISH", "prob_bullish": 0.8,
             "prob_neutral": 0.1, "prob_bearish": 0.1, "cycle_id": "KC-001"},
        ]
        ds = self._make_dataset(tmp_path, examples)
        stats = ds.stats()
        baseline = evaluate_baseline(stats)
        assert "avg_metric_score" in baseline["macro"]


# ---------------------------------------------------------------------------
# should_suggest_enable with metric
# ---------------------------------------------------------------------------


class TestShouldSuggestEnableMetric:
    """Test that should_suggest_enable uses metric scores when available."""

    def test_uses_metric_over_directional(self):
        """When avg_metric_score is available, uses it for comparison."""
        baseline = {
            "macro": {"directional_accuracy": 0.50, "avg_metric_score": 0.40},
            "technical": {"directional_accuracy": 0.50, "avg_metric_score": 0.40},
        }
        # Metric improved a lot, directional stayed same
        optimized = {
            "macro": {"directional_accuracy": 0.50, "avg_metric_score": 0.60},
            "technical": {"directional_accuracy": 0.50, "avg_metric_score": 0.55},
        }
        stats = {
            "agents": {
                "macro": {"total": 200, "class_balance": 0.45},
                "technical": {"total": 200, "class_balance": 0.45},
            }
        }
        suggest, explanation = should_suggest_enable(baseline, optimized, stats, min_for_suggest=100)
        # avg metric improvement is ~17.5%, should recommend
        assert suggest is True

    def test_falls_back_to_directional_without_metric(self):
        """Without avg_metric_score, uses directional_accuracy."""
        baseline = {
            "macro": {"directional_accuracy": 0.40},
        }
        optimized = {
            "macro": {"directional_accuracy": 0.55},
        }
        stats = {
            "agents": {
                "macro": {"total": 200, "class_balance": 0.45},
            }
        }
        suggest, explanation = should_suggest_enable(baseline, optimized, stats, min_for_suggest=100)
        assert suggest is True


# ---------------------------------------------------------------------------
# Abstention ceiling
# ---------------------------------------------------------------------------


class TestAbstentionCeiling:
    """Test that abstention ceiling config values are respected."""

    def test_ceiling_values_in_config(self):
        """Config has per-agent max_neutral_rate entries."""
        import json
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        rates = cfg.get("dspy", {}).get("max_neutral_rate", {})
        assert rates.get("default") == 0.60
        assert rates.get("technical") == 0.40
        assert rates.get("sentiment") == 0.40
        assert rates.get("volatility") == 0.40
        assert rates.get("macro") == 0.55
        assert rates.get("agronomist") == 0.70
        assert rates.get("geopolitical") == 0.75

    def test_episodic_agents_have_higher_ceiling(self):
        """Episodic domain agents (agronomist, inventory) get looser ceilings."""
        import json
        from pathlib import Path
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        with open(config_path) as f:
            cfg = json.load(f)
        rates = cfg.get("dspy", {}).get("max_neutral_rate", {})
        # Episodic agents should have higher ceilings than daily-data agents
        assert rates["agronomist"] > rates["technical"]
        assert rates["inventory"] > rates["sentiment"]
        assert rates["geopolitical"] > rates["macro"]
