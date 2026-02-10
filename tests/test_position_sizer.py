import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from trading_bot.position_sizer import DynamicPositionSizer

class TestDynamicPositionSizer(unittest.TestCase):
    def setUp(self):
        self.config = {
            'strategy': {'quantity': 1},
            'risk_management': {'max_heat_pct': 0.25}
        }
        self.sizer = DynamicPositionSizer(self.config)
        self.ib = MagicMock()
        # Mock reqPositionsAsync to return empty list by default
        self.ib.reqPositionsAsync = AsyncMock(return_value=[])

    def test_calculate_size_normal(self):
        async def run():
            signal = {'confidence': 1.0, 'prediction_type': 'DIRECTIONAL'}
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=100000.0
            )
            self.assertGreater(size, 0)
        asyncio.run(run())

    def test_calculate_size_negative_account_value(self):
        async def run():
            signal = {'confidence': 1.0, 'prediction_type': 'DIRECTIONAL'}
            # Expecting 0 size when account value is negative
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=-5000.0
            )
            self.assertEqual(size, 0, "Should return 0 size for negative account value")
        asyncio.run(run())

    def test_calculate_size_zero_account_value(self):
        async def run():
            signal = {'confidence': 1.0, 'prediction_type': 'DIRECTIONAL'}
            # Expecting 0 size when account value is zero
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=0.0
            )
            self.assertEqual(size, 0, "Should return 0 size for zero account value")
        asyncio.run(run())

if __name__ == '__main__':
    unittest.main()
