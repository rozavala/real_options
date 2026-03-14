import unittest
from datetime import datetime, timezone
import os
import shutil
import tempfile
import sys
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.enhanced_brier import (
    EnhancedBrierTracker,
    ProbabilisticPrediction,
    MarketRegime,
    normalize_regime,
    _REGIME_ALIASES
)

class TestEnhancedBrierMath(unittest.TestCase):

    def setUp(self):
        # Use a temporary directory for data files
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "enhanced_brier.json")
        self.tracker = EnhancedBrierTracker(data_path=self.data_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_brier_score_calculation(self):
        """Verify Brier score calculation for known cases."""
        # 1. Perfect Prediction (Bullish)
        # Prob: [1.0, 0.0, 0.0], Outcome: BULLISH ([1.0, 0.0, 0.0])
        # Brier = ((1-1)^2 + (0-0)^2 + (0-0)^2) / 3 = 0.0
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=1.0,
            prob_neutral=0.0,
            prob_bearish=0.0,
            cycle_id="test-1"
        )
        pred.actual_outcome = "BULLISH"
        score = pred.calc_brier_score()
        self.assertAlmostEqual(score, 0.0, places=4)

        # 2. Random Prediction (Uniform)
        # Prob: [1/3, 1/3, 1/3], Outcome: BULLISH ([1.0, 0.0, 0.0])
        # Brier = ((1/3-1)^2 + (1/3-0)^2 + (1/3-0)^2) / 3
        #       = ((-2/3)^2 + (1/3)^2 + (1/3)^2) / 3
        #       = (4/9 + 1/9 + 1/9) / 3 = (6/9) / 3 = 2/9 ≈ 0.2222
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=1/3,
            prob_neutral=1/3,
            prob_bearish=1/3,
            cycle_id="test-2"
        )
        pred.actual_outcome = "BULLISH"
        score = pred.calc_brier_score()
        self.assertAlmostEqual(score, 2/9, places=4)

        # 3. Completely Wrong Prediction
        # Prob: [0.0, 0.0, 1.0] (Bearish), Outcome: BULLISH ([1.0, 0.0, 0.0])
        # Brier = ((0-1)^2 + (0-0)^2 + (1-0)^2) / 3
        #       = (1 + 0 + 1) / 3 = 2/3 ≈ 0.6667
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=0.0,
            prob_neutral=0.0,
            prob_bearish=1.0,
            cycle_id="test-3"
        )
        pred.actual_outcome = "BULLISH"
        score = pred.calc_brier_score()
        self.assertAlmostEqual(score, 2/3, places=4)

    def test_probability_normalization(self):
        """Verify probabilities are normalized correctly."""
        # Sum > 1.0 (e.g., 0.5, 0.5, 0.5 -> sum=1.5)
        # Normalized: 0.5/1.5 = 1/3 each
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=0.5,
            prob_neutral=0.5,
            prob_bearish=0.5,
            cycle_id="test-norm-1"
        )
        self.assertAlmostEqual(pred.prob_bullish + pred.prob_neutral + pred.prob_bearish, 1.0, places=4)
        self.assertAlmostEqual(pred.prob_bullish, 1/3, places=4)

        # Sum < 1.0 (e.g., 0.1, 0.1, 0.1 -> sum=0.3)
        # Normalized: 0.1/0.3 = 1/3 each
        # Note: logic handles sum < 0.001 by forcing uniform, else normalizes
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=0.1,
            prob_neutral=0.1,
            prob_bearish=0.1,
            cycle_id="test-norm-2"
        )
        self.assertAlmostEqual(pred.prob_bullish + pred.prob_neutral + pred.prob_bearish, 1.0, places=4)
        self.assertAlmostEqual(pred.prob_bullish, 1/3, places=4)

        # Sum near zero (e.g., 0.0, 0.0, 0.0) -> Force uniform
        pred = ProbabilisticPrediction(
            timestamp=datetime.now(timezone.utc),
            agent="test_agent",
            prob_bullish=0.0,
            prob_neutral=0.0,
            prob_bearish=0.0,
            cycle_id="test-norm-3"
        )
        self.assertAlmostEqual(pred.prob_bullish, 1/3, places=4)

    def test_reliability_multiplier_calculation(self):
        """Verify reliability multiplier calculation logic."""
        # Inject mock scores directly
        agent = "test_agent"
        regime = MarketRegime.NORMAL.value

        # Helper to set average score
        def set_avg_score(score, count=30):
            if agent not in self.tracker.agent_scores:
                self.tracker.agent_scores[agent] = {}
            self.tracker.agent_scores[agent][regime] = [score] * count

        # 1. Perfect Score (0.0) -> Multiplier should be 2.0
        # Formula: 2.0 - (0.0 * 4.0) = 2.0
        set_avg_score(0.0)
        mult = self.tracker.get_agent_reliability(agent, regime)
        self.assertEqual(mult, 2.0)

        # 2. Baseline Score (0.25) -> Multiplier should be 1.0
        # Formula: 2.0 - (0.25 * 4.0) = 2.0 - 1.0 = 1.0
        set_avg_score(0.25)
        mult = self.tracker.get_agent_reliability(agent, regime)
        self.assertEqual(mult, 1.0)

        # 3. Random Score (~0.2222) -> Slightly > 1.0
        # Formula: 2.0 - (0.2222 * 4.0) = 2.0 - 0.8888 = 1.1112
        set_avg_score(2/9)
        mult = self.tracker.get_agent_reliability(agent, regime)
        self.assertAlmostEqual(mult, 1.1111, places=4)

        # 4. Poor Score (0.5) -> Multiplier should be 0.1 (clamped)
        # Formula: 2.0 - (0.5 * 4.0) = 0.0 -> Clamped to 0.1
        set_avg_score(0.5)
        mult = self.tracker.get_agent_reliability(agent, regime)
        self.assertEqual(mult, 0.1)

        # 5. Terrible Score (0.6667) -> Multiplier should be 0.1 (clamped)
        # Formula: 2.0 - (0.6667 * 4.0) = 2.0 - 2.6668 = -0.6668 -> Clamped to 0.1
        set_avg_score(2/3)
        mult = self.tracker.get_agent_reliability(agent, regime)
        self.assertEqual(mult, 0.1)

    def test_reliability_min_samples(self):
        """Verify minimum sample size requirement."""
        agent = "new_agent"
        regime = MarketRegime.NORMAL.value

        # 0 samples -> Baseline 1.0
        self.assertEqual(self.tracker.get_agent_reliability(agent, regime), 1.0)

        # 4 samples (perfect score) -> Still Baseline 1.0 (< 5 samples)
        if agent not in self.tracker.agent_scores:
            self.tracker.agent_scores[agent] = {}
        self.tracker.agent_scores[agent][regime] = [0.0] * 4

        self.assertEqual(self.tracker.get_agent_reliability(agent, regime), 1.0)

        # 5 samples (perfect score) -> 2.0 (>= 5 samples)
        self.tracker.agent_scores[agent][regime].append(0.0)
        self.assertEqual(self.tracker.get_agent_reliability(agent, regime), 2.0)

    def test_regime_normalization(self):
        """Verify regime string normalization."""
        # Ensure 'HIGH_VOLATILITY' maps to 'HIGH_VOL'
        self.assertEqual(normalize_regime("HIGH_VOLATILITY"), MarketRegime.HIGH_VOL)

        # Ensure 'low_vol' maps to 'RANGE_BOUND' (as per alias)
        self.assertEqual(normalize_regime("low_vol"), MarketRegime.RANGE_BOUND)

        # Ensure enum input returns enum
        self.assertEqual(normalize_regime(MarketRegime.TRENDING_UP), MarketRegime.TRENDING_UP)

        # Ensure unknown returns NORMAL
        self.assertEqual(normalize_regime("SUPER_WEIRD_MARKET"), MarketRegime.NORMAL)

    # === v8.0: 4-Path Fallback Tests ===

    def test_cross_regime_blend_when_specific_regime_sparse(self):
        """Path 2: Agent has <5 scores in requested regime but >=5 total across others."""
        agent = "blend_agent"
        # 2 scores in HIGH_VOL (too few for path 1)
        # 8 scores in NORMAL (enough for path 2 blend)
        self.tracker.agent_scores[agent] = {
            'HIGH_VOL': [0.1, 0.15],          # 2 scores, avg=0.125
            'NORMAL': [0.2] * 8,              # 8 scores, avg=0.2
        }
        self.tracker._legacy_accuracy_cache = {}

        result = self.tracker.get_agent_reliability(agent, 'HIGH_VOL')

        # Should NOT be 1.0 (baseline) — cross-regime blend kicks in
        self.assertNotEqual(result, 1.0)
        # HIGH_VOL mult: 2.0 - (0.125 * 4.0) = 1.5, weight 2
        # NORMAL mult: 2.0 - (0.2 * 4.0) = 1.2, weight 8
        # Blended: (1.5*2 + 1.2*8) / 10 = 1.26
        self.assertAlmostEqual(result, 1.26, places=2)

    def test_accuracy_floor_penalizes_bad_agent(self):
        """Path 3: Agent with <30% legacy accuracy and 0 Enhanced scores → 0.5."""
        agent = "bad_agent"
        self.tracker._legacy_accuracy_cache = {agent: 0.20}
        # No enhanced data at all
        self.tracker.agent_scores.pop(agent, None)

        result = self.tracker.get_agent_reliability(agent, 'NORMAL')
        self.assertEqual(result, 0.5)

    def test_accuracy_floor_skipped_when_accuracy_ok(self):
        """Path 3 skip: Agent with >=30% legacy accuracy and 0 Enhanced scores → baseline 1.0."""
        agent = "ok_agent"
        self.tracker._legacy_accuracy_cache = {agent: 0.55}
        self.tracker.agent_scores.pop(agent, None)

        result = self.tracker.get_agent_reliability(agent, 'NORMAL')
        self.assertEqual(result, 1.0)


    # === Volatility-Aware Resolution Tests ===

    def test_resolve_outcome_for_cycle_directional(self):
        """Directional cycles use actual_trend_direction unchanged."""
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        assert resolve_outcome_for_cycle("BULLISH", "DIRECTIONAL", "") == "BULLISH"
        assert resolve_outcome_for_cycle("BEARISH", "DIRECTIONAL", "") == "BEARISH"
        assert resolve_outcome_for_cycle("NEUTRAL", "DIRECTIONAL", "") == "NEUTRAL"

    def test_resolve_outcome_for_cycle_vol_stayed_flat(self):
        """VOLATILITY + STAYED_FLAT → always NEUTRAL.

        This is the most business-critical case: an iron condor that profited
        because the market stayed flat. Without this fix, agents who correctly
        predicted NEUTRAL get penalized because a tiny price uptick (e.g., +0.3%)
        resolves the cycle as BULLISH in actual_trend_direction.
        """
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        # Even if price ticked up (BULLISH), condor staying flat = NEUTRAL
        assert resolve_outcome_for_cycle("BULLISH", "VOLATILITY", "STAYED_FLAT") == "NEUTRAL"
        assert resolve_outcome_for_cycle("BEARISH", "VOLATILITY", "STAYED_FLAT") == "NEUTRAL"
        # Even NEUTRAL direction + STAYED_FLAT = NEUTRAL (no change, but consistent)
        assert resolve_outcome_for_cycle("NEUTRAL", "VOLATILITY", "STAYED_FLAT") == "NEUTRAL"

    def test_resolve_outcome_for_cycle_vol_big_move(self):
        """VOLATILITY + BIG_MOVE → use actual direction.

        When the market breaks out, agents who predicted the right direction
        should be rewarded. Agents who predicted NEUTRAL should be penalized.
        """
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        assert resolve_outcome_for_cycle("BULLISH", "VOLATILITY", "BIG_MOVE") == "BULLISH"
        assert resolve_outcome_for_cycle("BEARISH", "VOLATILITY", "BIG_MOVE") == "BEARISH"

    def test_resolve_outcome_for_cycle_missing_fields(self):
        """Missing/empty fields fall back to directional safely."""
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        assert resolve_outcome_for_cycle("BULLISH", "", "") == "BULLISH"
        assert resolve_outcome_for_cycle("BEARISH", "VOLATILITY", "") == "BEARISH"
        assert resolve_outcome_for_cycle("", "VOLATILITY", "BIG_MOVE") == ""

    def test_resolve_outcome_for_cycle_nan_handling(self):
        """Pandas NaN and None values don't cause incorrect matches."""
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        # float('nan') stringifies to "nan" → should not match any valid outcome
        assert resolve_outcome_for_cycle("BULLISH", "nan", "nan") == "BULLISH"
        # None values handled safely
        assert resolve_outcome_for_cycle("BEARISH", None, None) == "BEARISH"
        assert resolve_outcome_for_cycle(None, "VOLATILITY", "STAYED_FLAT") == "NEUTRAL"

    def test_resolve_outcome_unexpected_vol_outcome_warns(self):
        """Unexpected volatility_outcome values emit a warning and fall through."""
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        with self.assertLogs('trading_bot.enhanced_brier', level='WARNING') as cm:
            result = resolve_outcome_for_cycle("BULLISH", "VOLATILITY", "SOMETHING_WEIRD")
        # Should fall through to directional
        assert result == "BULLISH"
        # Should have logged a warning about unexpected value
        assert any("Unexpected volatility_outcome" in msg for msg in cm.output)

    def test_resolve_outcome_unresolvable_warns(self):
        """Completely unresolvable inputs emit a warning and return empty string."""
        from trading_bot.enhanced_brier import resolve_outcome_for_cycle
        with self.assertLogs('trading_bot.enhanced_brier', level='WARNING') as cm:
            result = resolve_outcome_for_cycle("GARBAGE", "VOLATILITY", "BIG_MOVE")
        assert result == ""
        assert any("Unresolvable outcome" in msg for msg in cm.output)


if __name__ == '__main__':
    unittest.main()
