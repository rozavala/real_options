import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from trading_bot.position_sizer import DynamicPositionSizer

class TestDynamicPositionSizerRobustness(unittest.TestCase):
    def setUp(self):
        # Use quantity 100 to detect multiplier differences
        self.config = {
            'strategy': {'quantity': 100},
            'risk_management': {'max_heat_pct': 0.25}
        }
        self.sizer = DynamicPositionSizer(self.config)
        self.ib = MagicMock()
        # Mock reqPositionsAsync to return empty list by default
        self.ib.reqPositionsAsync = AsyncMock(return_value=[])

    def test_calculate_size_string_confidence(self):
        async def run():
            # confidence as string "0.8" should be parsed to 0.8
            # Multiplier: 0.5 + (0.8 * 0.5) = 0.9
            # Size: 100 * 0.9 = 90
            signal = {'confidence': "0.8", 'prediction_type': 'DIRECTIONAL'}

            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=100000.0
            )
            # Expecting 90
            self.assertEqual(size, 90, f"Expected size 90 for confidence 0.8, got {size}")

        asyncio.run(run())

    def test_calculate_size_none_confidence(self):
        async def run():
            # confidence None should default to 0.5
            # Multiplier: 0.5 + (0.5 * 0.5) = 0.75
            # Size: 100 * 0.75 = 75
            signal = {'confidence': None, 'prediction_type': 'DIRECTIONAL'}
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=100000.0
            )
            self.assertEqual(size, 75, f"Expected size 75 for None confidence (default 0.5), got {size}")
        asyncio.run(run())

    def test_calculate_size_band_confidence(self):
        async def run():
            # confidence 'HIGH' should be parsed to 0.80 (from confidence_utils)
            # Multiplier: 0.5 + (0.80 * 0.5) = 0.90
            # Size: 100 * 0.90 = 90
            signal = {'confidence': "HIGH", 'prediction_type': 'DIRECTIONAL'}

            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=100000.0
            )
            self.assertEqual(size, 90, f"Expected size 90 for HIGH confidence, got {size}")
        asyncio.run(run())

if __name__ == '__main__':
    unittest.main()
