import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Future, FuturesOption, Bag, ComboLeg, Trade, Order, OrderStatus, LimitOrder
from datetime import datetime

from trading_bot.ib_interface import (
    get_active_futures,
    build_option_chain,
    create_combo_order_object,
    place_order,
)


class TestIbInterface(unittest.TestCase):

    @patch('trading_bot.ib_interface.datetime')
    def test_get_active_futures(self, mock_datetime):
        async def run_test():
            # Mock current time to be mid-2025 so 202512 is in the future
            mock_datetime.now.return_value = datetime(2025, 6, 15)
            # Ensure strptime returns real datetime objects for comparison
            mock_datetime.strptime.side_effect = lambda *args, **kwargs: datetime.strptime(*args, **kwargs)

            ib = MagicMock()
            mock_cd1 = MagicMock(contract=Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512'))
            mock_cd2 = MagicMock(contract=Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='202603'))
            ib.reqContractDetailsAsync = AsyncMock(return_value=[mock_cd1, mock_cd2])
            futures = await get_active_futures(ib, 'KC', 'NYBOT')
            self.assertEqual(len(futures), 2)
            self.assertEqual(futures[0].lastTradeDateOrContractMonth, '202512')

        asyncio.run(run_test())

    def test_build_option_chain(self):
        async def run_test():
            ib = MagicMock()
            future_contract = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512', exchange='NYBOT')
            mock_chain = MagicMock(exchange='NYBOT', tradingClass='KCO', expirations=['20251120'], strikes=[340.0, 350.0])
            ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[mock_chain])

            chain = await build_option_chain(ib, future_contract)

            self.assertIsNotNone(chain)
            self.assertEqual(chain['exchange'], 'NYBOT')
            # Assert that the strikes are NOT normalized
            self.assertEqual(chain['strikes_by_expiration']['20251120'], [340.0, 350.0])

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_order_object_market_order(self, mock_get_market_data, mock_price_bs):
        """
        Tests that a MarketOrder is created when the config is set to 'MKT'.
        """
        async def run_test():
            # 1. Setup Mocks — side_effect modifies contracts in-place (like real IB)
            ib = AsyncMock()
            conid_map = {3.5: 101, 3.6: 102}
            async def qualify_side_effect(*contracts):
                for c in contracts:
                    c.conId = conid_map.get(c.strike, 0)
                return list(contracts)
            ib.qualifyContractsAsync = AsyncMock(side_effect=qualify_side_effect)
            mock_get_market_data.side_effect = [
                {'bid': 0.9, 'ask': 1.1, 'implied_volatility': 0.2, 'risk_free_rate': 0.05},
                {'bid': 0.4, 'ask': 0.6, 'implied_volatility': 0.18, 'risk_free_rate': 0.05}
            ]
            mock_price_bs.side_effect = [{'price': 1.0}, {'price': 0.5}]

            # 2. Setup Inputs - with order_type set to MKT
            config = {
                'symbol': 'KC',
                'strategy': {'quantity': 1},
                'strategy_tuning': {
                    'order_type': 'MKT',
                    'max_liquidity_spread_percentage': 0.9, # 90%
                }
            }
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'},
                "underlying_price": 100.0,
                "future_contract": Future(conId=1, symbol='KC')
            }

            # 3. Execute
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNotNone(result)
            _, market_order = result
            self.assertIsInstance(market_order, Order)
            self.assertEqual(market_order.orderType, "MKT")
            self.assertEqual(market_order.action, "BUY")
            self.assertEqual(market_order.totalQuantity, 1)

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_order_object(self, mock_get_market_data, mock_price_bs):
        """
        Tests that the combo order price is calculated correctly by combining
        the theoretical price with a dynamic, spread-based slippage.
        """
        async def run_test():
            # 1. Setup Mocks — side_effect modifies contracts in-place (like real IB)
            ib = AsyncMock()
            conid_map = {3.5: 101, 3.6: 102}
            async def qualify_side_effect(*contracts):
                for c in contracts:
                    c.conId = conid_map.get(c.strike, 0)
                return list(contracts)
            ib.qualifyContractsAsync = AsyncMock(side_effect=qualify_side_effect)

            # Mock for market data (bid, ask, IV)
            mock_get_market_data.side_effect = [
                {'bid': 0.9, 'ask': 1.1, 'implied_volatility': 0.2, 'risk_free_rate': 0.05},
                {'bid': 0.4, 'ask': 0.6, 'implied_volatility': 0.18, 'risk_free_rate': 0.05}
            ]

            # Mock for theoretical pricing
            mock_price_bs.side_effect = [{'price': 1.0}, {'price': 0.5}]

            # 2. Setup Inputs
            config = {
                'symbol': 'KC',
                'strategy': {'quantity': 1},
                'strategy_tuning': {
                    'max_liquidity_spread_percentage': 0.9, # 90%
                    'fixed_slippage_cents': 0.2
                }
            }
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'},
                "underlying_price": 100.0,
                "future_contract": Future(conId=1, symbol='KC')
            }

            # 3. Execute
            result = await create_combo_order_object(ib, config, strategy_def)

            # 4. Assertions
            self.assertIsNotNone(result)
            _, limit_order = result

            # Theoretical Price: 1.0 (buy) - 0.5 (sell) = 0.5
            # Fixed Slippage: 0.2
            # Ceiling Price (BUY) = Theoretical Price + Fixed Slippage = 0.5 + 0.2 = 0.7

            # Combo Bid (Synthetic) = Leg1_Bid (0.9) - Leg2_Ask (0.6) = 0.3
            # Initial Price (BUY) = Combo Bid + 0.05 = 0.3 + 0.05 = 0.35
            # Ensure Initial Price <= Ceiling Price (0.35 <= 0.7) -> True

            expected_price = 0.35
            self.assertAlmostEqual(limit_order.lmtPrice, expected_price, places=2)

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_probe_failure_returns_none(self, mock_get_market_data, mock_price_bs):
        """
        Tests that if the probe leg (ATM) fails to qualify (conId=0), the
        function returns None without attempting remaining legs.
        """
        async def run_test():
            ib = AsyncMock()

            # Probe leg (first) fails — far-dated expiry with no strikes listed
            async def qualify_side_effect(*contracts):
                for c in contracts:
                    c.conId = 0  # All fail
                return list(contracts)
            ib.qualifyContractsAsync = AsyncMock(side_effect=qualify_side_effect)

            config = {'symbol': 'KC', 'strategy': {'quantity': 1}}
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20261113', 'days_to_exp': 275},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KO'}, "underlying_price": 280.0,
            }

            result = await create_combo_order_object(ib, config, strategy_def)

            self.assertIsNone(result)
            # Probe was called once; remaining legs never attempted
            self.assertEqual(ib.qualifyContractsAsync.call_count, 1)
            mock_get_market_data.assert_not_called()
            mock_price_bs.assert_not_called()

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_create_combo_remaining_leg_failure_returns_none(self, mock_get_market_data, mock_price_bs):
        """
        Tests that if the probe succeeds but a remaining leg fails (conId=0),
        the function returns None.
        """
        async def run_test():
            ib = AsyncMock()

            # First leg qualifies, second doesn't
            async def qualify_side_effect(*contracts):
                for c in contracts:
                    if c.strike == 3.5:
                        c.conId = 101
                    else:
                        c.conId = 0  # Short leg not available
                return list(contracts)
            ib.qualifyContractsAsync = AsyncMock(side_effect=qualify_side_effect)

            config = {'symbol': 'KC', 'strategy': {'quantity': 1}}
            strategy_def = {
                "action": "BUY", "legs_def": [('C', 'BUY', 3.5), ('C', 'SELL', 3.6)],
                "exp_details": {'exp_date': '20251220', 'days_to_exp': 30},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'KCO'}, "underlying_price": 100.0,
            }

            result = await create_combo_order_object(ib, config, strategy_def)

            self.assertIsNone(result)
            # Both probe and remaining legs were attempted
            self.assertEqual(ib.qualifyContractsAsync.call_count, 2)
            mock_get_market_data.assert_not_called()
            mock_price_bs.assert_not_called()

        asyncio.run(run_test())

    def test_place_order(self):
        """Tests that place_order calls ib.placeOrder with the correct arguments."""
        ib = MagicMock()
        mock_contract = Bag(symbol='KC')
        mock_order = LimitOrder('BUY', 1, 1.23)

        place_order(ib, mock_contract, mock_order)
        ib.placeOrder.assert_called_once_with(mock_contract, mock_order)

    @patch('trading_bot.ib_interface.price_option_black_scholes')
    @patch('trading_bot.ib_interface.get_option_market_data')
    def test_inverted_spread_corrected_by_iv_consistency(self, mock_get_market_data, mock_price_bs):
        """
        Tests that a BUY spread with mixed IV sources (one FALLBACK, one IBKR)
        triggers IV consistency correction. The FALLBACK leg gets repriced with
        the IBKR IV, which should fix the inversion.

        Reproduces the CCK6 bug where IB rejected as 'riskless combination'.
        Before the IV consistency fix, this returned None (inverted spread).
        After the fix, the FALLBACK leg is repriced with IBKR IV, correcting the spread.
        """
        async def run_test():
            ib = AsyncMock()
            conid_map = {3100.0: 101, 3050.0: 102}
            async def qualify_side_effect(*contracts):
                for c in contracts:
                    c.conId = conid_map.get(c.strike, 0)
                return list(contracts)
            ib.qualifyContractsAsync = AsyncMock(side_effect=qualify_side_effect)

            # BUY leg IV=35% (fallback), SELL leg IV=57% (IBKR)
            mock_get_market_data.side_effect = [
                {'bid': 238.0, 'ask': 242.0, 'implied_volatility': 0.35, 'iv_source': 'FALLBACK', 'risk_free_rate': 0.05},
                {'bid': 223.0, 'ask': 227.0, 'implied_volatility': 0.57, 'iv_source': 'IBKR', 'risk_free_rate': 0.05}
            ]
            # Original: BUY=149.67 (fallback IV), SELL=225.36 (IBKR IV) → inverted
            # Repriced: BUY=240.50 (IBKR-derived IV) → net = 240.50 - 225.36 = +15.14 (corrected)
            mock_price_bs.side_effect = [
                {'price': 149.67},  # BUY leg initial (fallback IV)
                {'price': 225.36},  # SELL leg initial (IBKR IV)
                {'price': 240.50},  # BUY leg repriced (IBKR-derived IV)
            ]

            config = {
                'symbol': 'CC',
                'exchange': 'NYBOT',
                'strategy': {'quantity': 1},
                'strategy_tuning': {
                    'max_liquidity_spread_percentage': 0.99,
                    'fixed_slippage_cents': 0.5
                }
            }
            strategy_def = {
                "action": "BUY",
                "legs_def": [('P', 'BUY', 3100.0), ('P', 'SELL', 3050.0)],
                "exp_details": {'exp_date': '20260515', 'days_to_exp': 84},
                "chain": {'exchange': 'NYBOT', 'tradingClass': 'COK'},
                "underlying_price": 3100.5,
                "future_contract": Future(conId=1, symbol='CC')
            }

            result = await create_combo_order_object(ib, config, strategy_def)
            # IV consistency correction should fix the inversion — order should proceed
            self.assertIsNotNone(result)
            # Verify the repricing was called (3 BS calls: 2 original + 1 reprice)
            self.assertEqual(mock_price_bs.call_count, 3)

        asyncio.run(run_test())

    @patch('trading_bot.ib_interface.datetime')
    def test_get_active_futures_filters_45_days(self, mock_datetime):
        """
        Verifies that contracts expiring within 45 days are excluded.
        """
        async def run_test():
            # Mock current time to a fixed date
            # Today: Jan 1, 2024
            fixed_now = datetime(2024, 1, 1)
            mock_datetime.now.return_value = fixed_now
            mock_datetime.strptime.side_effect = lambda *args, **kwargs: datetime.strptime(*args, **kwargs)

            ib = MagicMock()

            # Setup Contracts
            # 1. Expiring in 10 days (Exclude) -> Jan 11, 2024
            c1 = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='20240111', localSymbol='KC_TooSoon')

            # 2. Expiring in 44 days (Exclude) -> Feb 14, 2024 (Jan 1 + 44d = Feb 14)
            c2 = Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='20240214', localSymbol='KC_JustTooSoon')

            # 3. Expiring in 46 days (Include) -> Feb 16, 2024 (Jan 1 + 46d = Feb 16)
            c3 = Future(conId=3, symbol='KC', lastTradeDateOrContractMonth='20240216', localSymbol='KC_JustRight')

            # 4. Expiring next year (Include)
            c4 = Future(conId=4, symbol='KC', lastTradeDateOrContractMonth='20250315', localSymbol='KC_NextYear')

            # 5. Invalid Date Format (Skip gracefully)
            c5 = Future(conId=5, symbol='KC', lastTradeDateOrContractMonth='INVALID', localSymbol='KC_BadDate')

            mock_details = [
                MagicMock(contract=c1),
                MagicMock(contract=c2),
                MagicMock(contract=c3),
                MagicMock(contract=c4),
                MagicMock(contract=c5)
            ]

            ib.reqContractDetailsAsync = AsyncMock(return_value=mock_details)

            # Execute
            futures = await get_active_futures(ib, 'KC', 'NYBOT', count=10)

            # Assertions
            # Should only contain c3 and c4
            self.assertEqual(len(futures), 2)
            self.assertEqual(futures[0].localSymbol, 'KC_JustRight')
            self.assertEqual(futures[1].localSymbol, 'KC_NextYear')

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()