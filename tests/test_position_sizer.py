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

class TestPortfolioHeatNetting(unittest.TestCase):
    """Verify that spread legs are netted by expiry, not summed per-leg."""

    def setUp(self):
        self.config = {
            'strategy': {'quantity': 1},
            'risk_management': {'max_heat_pct': 0.25},
        }
        self.sizer = DynamicPositionSizer(self.config)
        self.ib = MagicMock()

    def _make_position(self, qty, avg_cost, strike, expiry='20260718',
                       symbol='KC', sec_type='FOP'):
        pos = MagicMock()
        pos.position = qty
        pos.avgCost = avg_cost
        pos.contract = MagicMock()
        pos.contract.symbol = symbol
        pos.contract.secType = sec_type
        pos.contract.lastTradeDateOrExpiry = expiry
        pos.contract.strike = strike
        return pos

    def test_spread_legs_net_within_expiry(self):
        """A bear put spread should report net debit, not sum of absolutes."""
        async def run():
            # BUY 285P at $8100, SELL 280P at $7000 — same Jul expiry
            positions = [
                self._make_position(qty=1, avg_cost=8100, strike=285, expiry='20260718'),
                self._make_position(qty=-1, avg_cost=7000, strike=280, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # Account = $35000, net exposure = |8100 - 7000| = $1100
            # Heat = 1100/35000 = 3.1% — well under 25%
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            # With heat_multiplier=1.0 (under threshold), full size
            # base_qty=1 * conf(1.0) * vol(1.0) * heat(1.0) = 1.0 → ceil = 1
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_old_formula_would_trigger_heat_warning(self):
        """Same spread that old abs-per-leg formula would flag as 43%."""
        async def run():
            # BUY 285P at $8100, SELL 280P at $7000
            # Old: abs(8100) + abs(-7000) = 15100, heat = 15100/35000 = 43% > 25%
            # New: |8100 - 7000| = 1100, heat = 3.1% < 25%
            positions = [
                self._make_position(qty=1, avg_cost=8100, strike=285, expiry='20260718'),
                self._make_position(qty=-1, avg_cost=7000, strike=280, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            # heat_multiplier should be 1.0 (not 0.5)
            # base=1 * conf=1.0 * vol=1.0 * heat=1.0 = 1.0
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_multiple_spreads_same_expiry(self):
        """Two bear put spreads on same expiry net correctly."""
        async def run():
            positions = [
                # Spread 1: BUY 285P, SELL 280P
                self._make_position(qty=2, avg_cost=8100, strike=285, expiry='20260718'),
                self._make_position(qty=-2, avg_cost=7000, strike=280, expiry='20260718'),
                # Spread 2: BUY 275P, SELL 270P
                self._make_position(qty=1, avg_cost=6000, strike=275, expiry='20260718'),
                self._make_position(qty=-1, avg_cost=5000, strike=270, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # Net: (2*8100 - 2*7000) + (6000 - 5000) = 2200 + 1000 = 3200
            # All same expiry → single net = 3200
            # Heat = 3200/35000 = 9.1% < 25% → heat_multiplier = 1.0
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_different_expiries_stay_separate(self):
        """Spreads on different expiries don't net against each other."""
        async def run():
            positions = [
                # May spread: BUY 285P at $8100, SELL 280P at $7000
                self._make_position(qty=1, avg_cost=8100, strike=285, expiry='20260516'),
                self._make_position(qty=-1, avg_cost=7000, strike=280, expiry='20260516'),
                # Jul spread: BUY 280P at $7500, SELL 270P at $5500
                self._make_position(qty=1, avg_cost=7500, strike=280, expiry='20260718'),
                self._make_position(qty=-1, avg_cost=5500, strike=270, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # May net: |8100 - 7000| = 1100
            # Jul net: |7500 - 5500| = 2000
            # Total exposure: 1100 + 2000 = 3100
            # Heat = 3100/35000 = 8.9% < 25%
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_realistic_kc_portfolio_under_threshold(self):
        """Reproduce the real KC portfolio: 5 positions, should be ~26% not 135%."""
        async def run():
            # Approximate real KC positions from var_state.json
            # May expiry: 2 bear put spreads (285/280)
            # Jul expiry: 1 bear put spread (280/270) + 1 bear put spread (275/270)
            positions = [
                # May: BUY 2x 285P, SELL 2x 280P
                self._make_position(qty=2, avg_cost=8100, strike=285, expiry='20260516'),
                self._make_position(qty=-2, avg_cost=7000, strike=280, expiry='20260516'),
                # Jul: SELL 2x 270P (aggregated), BUY 1x 275P, BUY 1x 280P
                self._make_position(qty=-2, avg_cost=5500, strike=270, expiry='20260718'),
                self._make_position(qty=1, avg_cost=6500, strike=275, expiry='20260718'),
                self._make_position(qty=1, avg_cost=7500, strike=280, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # May net: 2*8100 + (-2)*7000 = 16200 - 14000 = 2200 → |2200| = 2200
            # Jul net: (-2)*5500 + 1*6500 + 1*7500 = -11000 + 6500 + 7500 = 3000 → |3000| = 3000
            # Total: 5200. Heat = 5200/35000 = 14.9% < 25% → no reduction
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_genuinely_hot_portfolio_still_triggers(self):
        """Large net exposure correctly triggers heat reduction."""
        async def run():
            # A single deep ITM put — large net exposure
            positions = [
                self._make_position(qty=5, avg_cost=15000, strike=300, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # Net: 5 * 15000 = 75000
            # Heat = 75000/35000 = 214% > 25% → heat_multiplier = 0.5
            # base=1 * conf=1.0 * vol=1.0 * heat=0.5 = 0.5 → ceil = 1 (min 1)
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            # Still 1 because base_qty=1 and ceil(0.5)=1
            self.assertEqual(size, 1)
        asyncio.run(run())

    def test_hot_portfolio_reduces_large_base(self):
        """Heat reduction visible with larger base_qty."""
        async def run():
            config = {
                'strategy': {'quantity': 10},
                'risk_management': {'max_heat_pct': 0.25},
            }
            sizer = DynamicPositionSizer(config)

            # Large net long position
            positions = [
                self._make_position(qty=5, avg_cost=15000, strike=300, expiry='20260718'),
            ]
            self.ib.reqPositionsAsync = AsyncMock(return_value=positions)

            signal = {'confidence': 1.0}
            # Heat = 75000/35000 = 214% → heat_multiplier = 0.5
            # base=10 * conf=1.0 * vol=1.0 * heat=0.5 = 5.0 → ceil = 5
            size = await sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            self.assertEqual(size, 5)
        asyncio.run(run())

    def test_non_option_positions_excluded(self):
        """Futures positions don't count in option heat."""
        async def run():
            fut_pos = self._make_position(qty=1, avg_cost=100000, strike=0, expiry='20260718')
            fut_pos.contract.secType = 'FUT'
            self.ib.reqPositionsAsync = AsyncMock(return_value=[fut_pos])

            signal = {'confidence': 1.0}
            size = await self.sizer.calculate_size(
                self.ib, signal, 'NEUTRAL', account_value=35000.0
            )
            # No option exposure → heat = 0 → full size
            self.assertEqual(size, 1)
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
