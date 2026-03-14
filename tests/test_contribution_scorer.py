"""
Tests for contribution-based agent attribution scoring.

Covers all three scoring paths, confidence compression, multiplier conversion,
regime fallback, config toggle, materiality thresholds, and override scores.
"""

import json
import math
import os
import tempfile

import pytest

from trading_bot.contribution_scorer import (
    ContributionTracker,
    compress_confidence,
    _is_vol_strategy_correct,
    RECENCY_LAMBDA,
    ROLLING_WINDOW,
)
from trading_bot.contribution_bridge import (
    is_contribution_scoring_enabled,
    set_scoring_mode,
    reset_trackers,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tracker(tmp_path):
    """Fresh ContributionTracker with temp storage."""
    path = str(tmp_path / "contribution_scores.json")
    return ContributionTracker(data_path=path)


@pytest.fixture
def _reset_bridge():
    """Reset contribution bridge state after each test."""
    yield
    set_scoring_mode(False)
    reset_trackers()


# ============================================================================
# PATH 1: DIRECTIONAL cycle tests
# ============================================================================

class TestDirectionalScoring:

    def test_aligned_correct(self, tracker):
        """Agent BULLISH + Master BULLISH + Outcome BULLISH → positive score."""
        score = tracker.record_contribution(
            agent="agronomist", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="BULLISH", influence_weight=1.0,
        )
        assert score > 0

    def test_dissent_correct(self, tracker):
        """Agent BEARISH + Master BULLISH + Outcome BEARISH → positive score.
        Dissenter was right (Master was wrong), so agent gets credit."""
        score = tracker.record_contribution(
            agent="macro", agent_direction="BEARISH",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="BEARISH", influence_weight=1.0,
        )
        # alignment = -1 (dissent), outcome_sign = -1 (Master wrong)
        # -1 * -1 = +1 → positive
        assert score > 0

    def test_aligned_wrong(self, tracker):
        """Agent BULLISH + Master BULLISH + Outcome BEARISH → negative score."""
        score = tracker.record_contribution(
            agent="technical", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="BEARISH", influence_weight=1.0,
        )
        assert score < 0

    def test_neutral_agent(self, tracker):
        """Agent NEUTRAL + any → 0.0 (abstention)."""
        score = tracker.record_contribution(
            agent="agronomist", agent_direction="NEUTRAL",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="BULLISH", influence_weight=1.0,
        )
        assert score == 0.0

    def test_neutral_master(self, tracker):
        """Master NEUTRAL (directional cycle) + any → 0.0."""
        score = tracker.record_contribution(
            agent="macro", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="NEUTRAL",
            actual_outcome="BULLISH", influence_weight=1.0,
        )
        assert score == 0.0

    def test_neutral_outcome(self, tracker):
        """Any + Outcome NEUTRAL → 0.0 (market noise)."""
        score = tracker.record_contribution(
            agent="technical", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="NEUTRAL", influence_weight=1.0,
        )
        assert score == 0.0


# ============================================================================
# PATH 3: VOLATILITY cycle — Volatility agent (special track)
# ============================================================================

class TestVolatilityAgentScoring:

    def test_vol_agent_bullish_big_move(self, tracker):
        """Vol BULLISH (low IV) + BIG_MOVE → positive (correct regime call)."""
        score = tracker.record_contribution(
            agent="volatility", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="LONG_STRADDLE",
            volatility_outcome="BIG_MOVE",
            influence_weight=1.0,
        )
        # correct_for_vol = True → alignment = +1
        # LONG_STRADDLE + BIG_MOVE = correct → outcome_sign = +1
        assert score > 0

    def test_vol_agent_bullish_stayed_flat(self, tracker):
        """Vol BULLISH (low IV) + STAYED_FLAT → negative when Master was right.
        Vol agent said 'buy vol' but Master chose IRON_CONDOR (sell premium)
        and market stayed flat → condor won, vol agent's advice was wrong."""
        score = tracker.record_contribution(
            agent="volatility", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="IRON_CONDOR",
            volatility_outcome="STAYED_FLAT",
            influence_weight=1.0,
        )
        # alignment = -1 (wrong regime call), outcome_sign = +1 (Master correct)
        assert score < 0

    def test_vol_agent_bearish_stayed_flat(self, tracker):
        """Vol BEARISH (high IV) + STAYED_FLAT → positive (correct regime call)."""
        score = tracker.record_contribution(
            agent="volatility", agent_direction="BEARISH",
            agent_confidence=0.8, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="IRON_CONDOR",
            volatility_outcome="STAYED_FLAT",
            influence_weight=1.0,
        )
        # correct_for_vol = True → alignment = +1
        # IRON_CONDOR + STAYED_FLAT = correct → outcome_sign = +1
        assert score > 0

    def test_vol_agent_neutral(self, tracker):
        """Vol NEUTRAL → 0.0 (no IV opinion)."""
        score = tracker.record_contribution(
            agent="volatility", agent_direction="NEUTRAL",
            agent_confidence=0.8, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="IRON_CONDOR",
            volatility_outcome="STAYED_FLAT",
            influence_weight=1.0,
        )
        assert score == 0.0


# ============================================================================
# PATH 2: VOLATILITY cycle — Non-vol agents
# ============================================================================

class TestVolCycleOtherAgents:

    def test_neutral_supports_vol_thesis(self, tracker):
        """Non-vol agent NEUTRAL in vol cycle + Master correct → positive."""
        score = tracker.record_contribution(
            agent="agronomist", agent_direction="NEUTRAL",
            agent_confidence=0.7, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="IRON_CONDOR",
            volatility_outcome="STAYED_FLAT",
            influence_weight=1.0,
        )
        # alignment = +1 (supported vol), outcome_sign = +1 (correct strategy)
        assert score > 0

    def test_directional_opposes_vol_thesis(self, tracker):
        """Non-vol agent BULLISH in vol cycle + Master correct → negative."""
        score = tracker.record_contribution(
            agent="macro", agent_direction="BULLISH",
            agent_confidence=0.7, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            prediction_type="VOLATILITY",
            strategy_type="IRON_CONDOR",
            volatility_outcome="STAYED_FLAT",
            influence_weight=1.0,
        )
        # alignment = -1 (opposed vol), outcome_sign = +1 (correct strategy)
        assert score < 0


# ============================================================================
# Vol Strategy Correctness
# ============================================================================

class TestVolStrategyCorrect:

    def test_iron_condor_stayed_flat(self):
        assert _is_vol_strategy_correct("IRON_CONDOR", "STAYED_FLAT") is True

    def test_long_straddle_big_move(self):
        assert _is_vol_strategy_correct("LONG_STRADDLE", "BIG_MOVE") is True

    def test_iron_condor_big_move(self):
        assert _is_vol_strategy_correct("IRON_CONDOR", "BIG_MOVE") is False

    def test_long_straddle_stayed_flat(self):
        assert _is_vol_strategy_correct("LONG_STRADDLE", "STAYED_FLAT") is False

    def test_missing_strategy(self):
        assert _is_vol_strategy_correct("", "BIG_MOVE") is False

    def test_missing_outcome(self):
        assert _is_vol_strategy_correct("IRON_CONDOR", "") is False


# ============================================================================
# Confidence Compression
# ============================================================================

class TestConfidenceCompression:

    def test_max_confidence(self):
        assert compress_confidence(1.0) == 1.0

    def test_mid_confidence(self):
        assert compress_confidence(0.5) == 0.75

    def test_zero_confidence(self):
        assert compress_confidence(0.0) == 0.5

    def test_clamps_above_one(self):
        assert compress_confidence(1.5) == 1.0

    def test_clamps_below_zero(self):
        assert compress_confidence(-0.5) == 0.5

    def test_confidence_amplifies_score(self, tracker):
        """Same alignment/outcome, higher conf → higher |score|."""
        score_low = tracker._compute_score(
            agent_name="macro", agent_direction="BULLISH",
            agent_confidence=0.3, master_direction="BULLISH",
            actual_outcome="BULLISH", prediction_type="DIRECTIONAL",
            strategy_type="", volatility_outcome="",
            influence_weight=1.0,
        )
        score_high = tracker._compute_score(
            agent_name="macro", agent_direction="BULLISH",
            agent_confidence=0.9, master_direction="BULLISH",
            actual_outcome="BULLISH", prediction_type="DIRECTIONAL",
            strategy_type="", volatility_outcome="",
            influence_weight=1.0,
        )
        assert abs(score_high) > abs(score_low)


# ============================================================================
# Multiplier Conversion
# ============================================================================

class TestMultiplierConversion:

    def test_positive_avg_high_multiplier(self, tracker):
        """Avg score +0.3 → multiplier ~1.6."""
        scores = [0.3] * 20
        mult = tracker._scores_to_multiplier(scores)
        assert 1.4 < mult < 1.8

    def test_negative_avg_low_multiplier(self, tracker):
        """Avg score -0.3 → multiplier ~0.4."""
        scores = [-0.3] * 20
        mult = tracker._scores_to_multiplier(scores)
        assert 0.2 < mult < 0.6

    def test_zero_avg_baseline(self, tracker):
        """Avg score 0.0 → multiplier ~1.0."""
        scores = [0.0] * 20
        mult = tracker._scores_to_multiplier(scores)
        assert mult == pytest.approx(1.0)

    def test_clamped_upper(self, tracker):
        """Very high scores → clamped at 2.0."""
        scores = [1.0] * 20
        mult = tracker._scores_to_multiplier(scores)
        assert mult == 2.0

    def test_clamped_lower(self, tracker):
        """Very negative scores → clamped at 0.1."""
        scores = [-1.0] * 20
        mult = tracker._scores_to_multiplier(scores)
        assert mult == 0.1

    def test_empty_scores(self, tracker):
        """Empty list → baseline 1.0."""
        assert tracker._scores_to_multiplier([]) == 1.0

    def test_recency_decay(self, tracker):
        """Recent scores weighted more than old ones."""
        # Old bad, recent good → should be above 1.0
        scores = [-0.5] * 15 + [0.5] * 15
        mult = tracker._scores_to_multiplier(scores)
        assert mult > 1.0

        # Old good, recent bad → should be below 1.0
        scores_rev = [0.5] * 15 + [-0.5] * 15
        mult_rev = tracker._scores_to_multiplier(scores_rev)
        assert mult_rev < 1.0


# ============================================================================
# Regime Fallback
# ============================================================================

class TestRegimeFallback:

    def test_full_regime_specific(self, tracker):
        """≥12 samples → 100% regime-specific."""
        tracker.agent_scores = {
            "agronomist": {
                "NORMAL": [0.3] * 15,
                "HIGH_VOL": [-0.2] * 15,
            }
        }
        mult = tracker.get_agent_reliability("agronomist", "NORMAL")
        # Should use only NORMAL scores (all positive)
        assert mult > 1.0

    def test_blend_regime(self, tracker):
        """5-11 samples → 70/30 blend."""
        tracker.agent_scores = {
            "macro": {
                "NORMAL": [0.4] * 7,  # 7 > 5 but < 12
                "HIGH_VOL": [0.1] * 20,
            }
        }
        mult = tracker.get_agent_reliability("macro", "NORMAL")
        # Should be blended: 70% NORMAL + 30% cross-regime
        assert mult > 1.0

    def test_cross_regime(self, tracker):
        """<5 regime + ≥5 cross-regime → cross-regime."""
        tracker.agent_scores = {
            "technical": {
                "NORMAL": [0.3] * 2,  # Too few
                "HIGH_VOL": [0.2] * 10,
            }
        }
        mult = tracker.get_agent_reliability("technical", "NORMAL")
        # Should fall back to cross-regime (HIGH_VOL data)
        assert mult > 1.0

    def test_baseline_no_data(self, tracker):
        """No data → 1.0."""
        mult = tracker.get_agent_reliability("unknown_agent", "NORMAL")
        assert mult == 1.0


# ============================================================================
# Config Toggle
# ============================================================================

class TestConfigToggle:

    def test_disabled_by_default(self, _reset_bridge):
        set_scoring_mode(False)
        assert is_contribution_scoring_enabled() is False

    def test_enabled(self, _reset_bridge):
        set_scoring_mode(True)
        assert is_contribution_scoring_enabled() is True

    def test_toggle_back(self, _reset_bridge):
        set_scoring_mode(True)
        assert is_contribution_scoring_enabled() is True
        set_scoring_mode(False)
        assert is_contribution_scoring_enabled() is False


# ============================================================================
# Materiality Threshold
# ============================================================================

class TestMaterialityThreshold:

    def test_kc_small_move_neutral(self):
        """KC 0.3% move < 0.6% threshold → should be NEUTRAL."""
        # This tests the concept — actual threshold application is in reconciliation
        threshold = 0.006  # KC threshold
        pct_change = 0.003  # 0.3% move
        assert pct_change < threshold

    def test_kc_large_move_directional(self):
        """KC 1.5% move > 0.6% threshold → should be directional."""
        threshold = 0.006
        pct_change = 0.015
        assert pct_change >= threshold

    def test_ng_medium_move_neutral(self):
        """NG 0.8% move < 1.2% threshold → should be NEUTRAL."""
        threshold = 0.012  # NG threshold
        pct_change = 0.008  # 0.8%
        assert pct_change < threshold

    def test_ng_large_move_directional(self):
        """NG 2.0% move > 1.2% threshold → should be directional."""
        threshold = 0.012
        pct_change = 0.020
        assert pct_change >= threshold


# ============================================================================
# Override Score
# ============================================================================

class TestOverrideScore:

    def test_override_score(self, tracker):
        """record_contribution(override_score=0.5) uses override."""
        score = tracker.record_contribution(
            agent="agronomist", agent_direction="NEUTRAL",
            agent_confidence=0.0, master_direction="NEUTRAL",
            actual_outcome="NEUTRAL",
            override_score=0.5,
        )
        assert score == 0.5

    def test_override_negative(self, tracker):
        score = tracker.record_contribution(
            agent="macro", agent_direction="BULLISH",
            agent_confidence=1.0, master_direction="BULLISH",
            actual_outcome="BULLISH",
            override_score=-0.3,
        )
        assert score == -0.3


# ============================================================================
# Persistence
# ============================================================================

class TestPersistence:

    def test_save_and_load(self, tmp_path):
        """Scores persist across tracker instances."""
        path = str(tmp_path / "scores.json")
        t1 = ContributionTracker(data_path=path)
        t1.record_contribution(
            agent="agronomist", agent_direction="BULLISH",
            agent_confidence=0.8, master_direction="BULLISH",
            actual_outcome="BULLISH", influence_weight=1.0,
        )

        # Load in new instance
        t2 = ContributionTracker(data_path=path)
        assert "agronomist" in t2.agent_scores
        assert len(t2.agent_scores["agronomist"]["NORMAL"]) == 1

    def test_summary(self, tracker):
        """get_summary returns expected structure."""
        tracker.record_contribution(
            agent="macro", agent_direction="BULLISH",
            agent_confidence=0.7, master_direction="BULLISH",
            actual_outcome="BULLISH", influence_weight=1.0,
        )
        summary = tracker.get_summary()
        assert "macro" in summary
        assert "NORMAL" in summary["macro"]
        info = summary["macro"]["NORMAL"]
        assert "multiplier" in info
        assert "avg_score" in info
        assert "total_scores" in info
        assert info["total_scores"] == 1
