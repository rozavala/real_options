
import unittest
import pandas as pd
import sys
import os

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_utils import calculate_agent_scores

class TestDashboardUtils(unittest.TestCase):
    def test_calculate_agent_scores_ml_signal(self):
        # Create a mock dataframe
        data = {
            'actual_trend_direction': ['UP', 'DOWN', 'UP', 'DOWN'],
            'ml_signal': ['LONG', 'SHORT', 'NEUTRAL', 'LONG'],
            'master_decision': ['BULLISH', 'BEARISH', 'NEUTRAL', 'BULLISH'],
            'meteorologist_sentiment': ['BULLISH', 'BEARISH', 'BULLISH', 'BEARISH']
        }
        df = pd.DataFrame(data)

        # Calculate scores
        scores = calculate_agent_scores(df)

        # Verify ml_signal scores
        ml_scores = scores.get('ml_signal')
        self.assertIsNotNone(ml_scores)

        # Row 0: LONG vs UP -> Correct
        # Row 1: SHORT vs DOWN -> Correct
        # Row 2: NEUTRAL -> Skipped
        # Row 3: LONG vs DOWN -> Incorrect

        # Expected: 2 correct, 3 total (NEUTRAL is skipped)
        self.assertEqual(ml_scores['correct'], 2)
        self.assertEqual(ml_scores['total'], 3)
        self.assertAlmostEqual(ml_scores['accuracy'], 2/3)

        print("ML Signal Scores verified:", ml_scores)

if __name__ == '__main__':
    unittest.main()
