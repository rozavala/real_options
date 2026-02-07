"""Tests for the Brier Bridge dual-write system."""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from trading_bot.brier_bridge import (
    record_agent_prediction,
    resolve_agent_prediction,
    get_agent_reliability,
    _confidence_to_probs,
    backfill_enhanced_from_csv,
)


class TestConfidenceToProbs(unittest.TestCase):
    def test_bullish_high_confidence(self):
        b, n, be = _confidence_to_probs('BULLISH', 0.8)
        self.assertAlmostEqual(b, 0.8)
        self.assertAlmostEqual(n + be, 0.2)
        self.assertAlmostEqual(b + n + be, 1.0)

    def test_bearish_high_confidence(self):
        b, n, be = _confidence_to_probs('BEARISH', 0.7)
        self.assertAlmostEqual(be, 0.7)
        self.assertAlmostEqual(b + n + be, 1.0)

    def test_neutral_high_confidence(self):
        b, n, be = _confidence_to_probs('NEUTRAL', 0.6)
        self.assertAlmostEqual(n, 0.6)
        self.assertAlmostEqual(b + n + be, 1.0)

    def test_probabilities_always_sum_to_one(self):
        for direction in ['BULLISH', 'BEARISH', 'NEUTRAL']:
            for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
                b, n, be = _confidence_to_probs(direction, conf)
                self.assertAlmostEqual(b + n + be, 1.0, places=6)


class TestGetAgentReliability(unittest.TestCase):
    @patch('trading_bot.brier_bridge._get_enhanced_tracker')
    def test_falls_back_to_legacy_when_enhanced_unavailable(self, mock_tracker):
        mock_tracker.return_value = None
        result = get_agent_reliability('agronomist')
        self.assertEqual(result, 1.0)  # Default baseline

    @patch('trading_bot.brier_bridge._get_enhanced_tracker')
    def test_returns_enhanced_when_available(self, mock_tracker):
        mock_enhanced = MagicMock()
        mock_enhanced.get_agent_reliability.return_value = 1.5
        mock_tracker.return_value = mock_enhanced

        # Pass config to enable 100% enhanced weight
        # config = {'brier_scoring': {'enhanced_weight': 1.0}}
        result = get_agent_reliability('agronomist', 'HIGH_VOL')
        self.assertEqual(result, 1.5)

    @patch('trading_bot.brier_bridge._get_enhanced_tracker')
    def test_regime_fallback_to_normal(self, mock_get_tracker):
        # Mock tracker
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        # Setup: HIGH_VOLATILITY returns 1.0 (no data), NORMAL returns 1.5
        def side_effect(agent, regime):
            if regime == "HIGH_VOL":
                return 1.0
            if regime == "NORMAL":
                return 1.5
            return 1.0

        mock_tracker.get_agent_reliability.side_effect = side_effect

        # Action: request for HIGH_VOLATILITY
        result = get_agent_reliability('agronomist', 'HIGH_VOLATILITY')

        # Assert: should return NORMAL value (1.5)
        self.assertEqual(result, 1.5)

        # Verify both calls were made
        expected_calls = [
            unittest.mock.call('agronomist', 'HIGH_VOL'),
            unittest.mock.call('agronomist', 'NORMAL')
        ]
        mock_tracker.get_agent_reliability.assert_has_calls(expected_calls)


class TestRecordAgentPrediction(unittest.TestCase):
    @patch('trading_bot.brier_bridge._get_enhanced_tracker')
    @patch('trading_bot.brier_scoring.get_brier_tracker')
    def test_dual_write_both_systems(self, mock_legacy, mock_enhanced_fn):
        mock_legacy_tracker = MagicMock()
        mock_legacy.return_value = mock_legacy_tracker

        mock_enhanced_tracker = MagicMock()
        mock_enhanced_fn.return_value = mock_enhanced_tracker

        record_agent_prediction(
            agent='agronomist',
            predicted_direction='BULLISH',
            predicted_confidence=0.75,
            cycle_id='test_cycle_001',
        )

        # Verify legacy was called
        mock_legacy_tracker.record_prediction_structured.assert_called_once()

        # Verify enhanced was called
        mock_enhanced_tracker.record_prediction.assert_called_once()


class TestBackfill(unittest.TestCase):
    @patch('trading_bot.brier_bridge._get_enhanced_tracker')
    def test_backfill_calls_tracker(self, mock_get_tracker):
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker
        mock_tracker.backfill_from_resolved_csv.return_value = 5

        result = backfill_enhanced_from_csv()

        self.assertEqual(result, 5)
        mock_tracker.backfill_from_resolved_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
