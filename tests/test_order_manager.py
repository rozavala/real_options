import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ib_insync import Bag, ComboLeg, Contract, Order, Trade, Position
import pandas as pd
from datetime import datetime, timedelta
import os

from trading_bot.order_manager import generate_and_queue_orders, place_queued_orders, close_stale_positions, ORDER_QUEUE
from ib_insync import util

class TestOrderManager(unittest.IsolatedAsyncioTestCase):

    @patch('trading_bot.order_manager.generate_signals')
    @patch('trading_bot.order_manager.IBConnectionPool')
    async def test_generate_and_queue_orders_calls_signal_generator(self, mock_pool, mock_generate_signals):
        mock_ib_instance = AsyncMock()
        mock_pool.get_connection = AsyncMock(return_value=mock_ib_instance)
        mock_generate_signals.return_value = [] # Return empty list to stop early

        config = {'strategy': {'signal_threshold': 0.5}}
        await generate_and_queue_orders(config)

        mock_generate_signals.assert_called_once()
        mock_pool.get_connection.assert_called_once()

class TestPositionClosing(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Use a temporary ledger path for tests to avoid messing with real data
        self.ledger_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_ledger.csv')
        # Fix the date to a Monday to ensure deterministic age calculations
        # and avoid Weekly Close triggers.
        # Mon Oct 23 2023. 2 days ago = Sat Oct 21. Age = 1 (Mon). Kept.
        self.mock_now = datetime(2023, 10, 23, 17, 20) # Monday
        self.create_mock_ledger(self.mock_now)

    def tearDown(self):
        # Clean up the temporary ledger
        if os.path.exists(self.ledger_path):
            os.remove(self.ledger_path)

    def create_mock_ledger(self, base_date):
        ten_days_ago = base_date - timedelta(days=10)
        two_days_ago = base_date - timedelta(days=2)

        data = {
            'timestamp': [ten_days_ago.strftime('%Y-%m-%d %H:%M:%S'), two_days_ago.strftime('%Y-%m-%d %H:%M:%S'), ten_days_ago.strftime('%Y-%m-%d %H:%M:%S')],
            'position_id': ['OLD_POSITION', 'NEW_POSITION', 'COMBO_POSITION'],
            'combo_id': [123, 456, 789],
            'local_symbol': ['KC 26JUL24 250 C', 'KC 26JUL24 260 C', 'KC 26JUL24 300 P'], # Simple symbol for combo part
            'action': ['BUY', 'BUY', 'BUY'],
            'quantity': [1, 1, 1],
            'avg_fill_price': [1.0, 1.0, 1.0],
            'strike': [250, 260, 300],
            'right': ['C', 'C', 'P'],
            'total_value_usd': [-100, -100, -100],
            'reason': ['Strategy Execution', 'Strategy Execution', 'Strategy Execution']
        }
        # Add another leg for combo
        data['timestamp'].append(ten_days_ago.strftime('%Y-%m-%d %H:%M:%S'))
        data['position_id'].append('COMBO_POSITION')
        data['combo_id'].append(789)
        data['local_symbol'].append('KC 26JUL24 310 P')
        data['action'].append('SELL') # A spread
        data['quantity'].append(-1)
        data['avg_fill_price'].append(1.0)
        data['strike'].append(310)
        data['right'].append('P')
        data['total_value_usd'].append(100)
        data['reason'].append('Strategy Execution')

        # Add a position with RECONCILIATION_MISSING reason that should be closed
        data['timestamp'].append(ten_days_ago.strftime('%Y-%m-%d %H:%M:%S'))
        data['position_id'].append('RECON_POSITION')
        data['combo_id'].append(111)
        data['local_symbol'].append('KC 26JUL24 200 C')
        data['action'].append('BUY')
        data['quantity'].append(1)
        data['avg_fill_price'].append(1.0)
        data['strike'].append(200)
        data['right'].append('C')
        data['total_value_usd'].append(-100)
        data['reason'].append('RECONCILIATION_MISSING')

        df = pd.DataFrame(data)
        df.to_csv(self.ledger_path, index=False)

    # Use MagicMock for place_order because it is a synchronous function in the codebase
    @patch('trading_bot.order_manager.place_order', new_callable=MagicMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IBConnectionPool')
    async def test_closes_only_old_positions_and_correct_symbols(self, mock_pool, mock_log_trade, mock_place_order):
        mock_ib_instance = AsyncMock()
        mock_pool.get_connection = AsyncMock(return_value=mock_ib_instance)
        mock_ib_instance.connectAsync = AsyncMock()
        mock_ib_instance.reqPositionsAsync = AsyncMock()
        mock_ib_instance.isConnected = MagicMock(return_value=True)

        # Mock Market Data for Limit Order logic
        # reqMktData returns a Ticker
        mock_ticker = MagicMock()
        mock_ticker.bid = 5.0
        mock_ticker.ask = 5.5
        mock_ticker.last = 5.25
        mock_ticker.close = 5.25

        # Since reqMktData is synchronous in ib_insync (it returns the ticker object immediately, which updates later),
        # we just return the mock ticker.
        # We must explicitly set it as a MagicMock to avoid AsyncMock treating it as a coroutine
        mock_ib_instance.reqMktData = MagicMock(return_value=mock_ticker)
        mock_ib_instance.cancelMktData = MagicMock()

        # Mock Positions
        mock_positions = [
            Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 250 C', symbol='KC', conId=1, exchange='NYBOT', secType='FOP'), position=1, avgCost=1.0),
            Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 260 C', symbol='KC', conId=2, exchange='NYBOT', secType='FOP'), position=1, avgCost=1.0),
            Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 300 P', symbol='KC', conId=3, exchange='NYBOT', secType='FOP'), position=1, avgCost=1.0),
            Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 310 P', symbol='KC', conId=4, exchange='NYBOT', secType='FOP'), position=-1, avgCost=1.0),
            Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 200 C', symbol='KC', conId=5, exchange='NYBOT', secType='FOP'), position=1, avgCost=1.0),
        ]
        mock_ib_instance.reqPositionsAsync.return_value = mock_positions

        # Mock qualifyContractsAsync to return populated contracts based on conId
        populated_contracts = {
            1: Contract(localSymbol='KC 26JUL24 250 C', symbol='KC', conId=1, exchange='NYBOT', secType='FOP'),
            2: Contract(localSymbol='KC 26JUL24 260 C', symbol='KC', conId=2, exchange='NYBOT', secType='FOP'),
            3: Contract(localSymbol='KC 26JUL24 300 P', symbol='KC', conId=3, exchange='NYBOT', secType='FOP'),
            4: Contract(localSymbol='KC 26JUL24 310 P', symbol='KC', conId=4, exchange='NYBOT', secType='FOP'),
            5: Contract(localSymbol='KC 26JUL24 200 C', symbol='KC', conId=5, exchange='NYBOT', secType='FOP'),
        }

        async def mock_qualify(*contracts):
            result = []
            for c in contracts:
                if c.conId in populated_contracts:
                    result.append(populated_contracts[c.conId])
                else:
                    result.append(c)
            return result

        mock_ib_instance.qualifyContractsAsync = AsyncMock(side_effect=mock_qualify)

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = 'Filled'
        mock_fill = MagicMock()
        mock_fill.commissionReport.realizedPNL = 50.0
        mock_trade.fills = [mock_fill]
        mock_trade.orderStatus.avgFillPrice = 1.0

        mock_place_order.return_value = mock_trade

        config = {'symbol': 'KC', 'exchange': 'NYBOT'}

        # Patch datetime to avoid Weekly Close logic triggering
        with patch('trading_bot.order_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = self.mock_now
            mock_datetime.date.return_value = self.mock_now.date()

            await close_stale_positions(config)

        # We expect 3 calls to place_order: 1 for OLD_POSITION, 1 for COMBO_POSITION, 1 for RECON_POSITION
        self.assertEqual(mock_place_order.call_count, 3)

        single_order_calls = []
        bag_order_call = None

        for call in mock_place_order.call_args_list:
            args, kwargs = call
            contract = args[1]
            if contract.secType == 'BAG':
                bag_order_call = call
            else:
                single_order_calls.append(call)

        # VERIFY SINGLE ORDERS
        self.assertEqual(len(single_order_calls), 2)

        # Sort calls by conId to make assertions deterministic
        single_order_calls.sort(key=lambda call: call[0][1].conId)

        # 1. OLD_POSITION (conId 1)
        args, _ = single_order_calls[0]
        single_contract_1 = args[1]
        single_order_1 = args[2]

        self.assertEqual(single_contract_1.conId, 1)
        self.assertEqual(single_contract_1.symbol, 'KC')
        self.assertEqual(single_contract_1.secType, 'FOP')
        self.assertEqual(single_order_1.orderType, 'LMT')
        self.assertEqual(single_order_1.lmtPrice, 5.0)
        self.assertEqual(single_order_1.action, 'SELL')

        # 2. RECON_POSITION (conId 5)
        args, _ = single_order_calls[1]
        single_contract_2 = args[1]
        single_order_2 = args[2]

        self.assertEqual(single_contract_2.conId, 5)
        self.assertEqual(single_contract_2.symbol, 'KC')
        self.assertEqual(single_contract_2.secType, 'FOP')
        self.assertEqual(single_order_2.orderType, 'LMT')
        self.assertEqual(single_order_2.lmtPrice, 5.0)
        self.assertEqual(single_order_2.action, 'SELL')

        # VERIFY BAG ORDER
        self.assertIsNotNone(bag_order_call)
        args, _ = bag_order_call
        bag_contract = args[1]
        bag_order = args[2]

        self.assertEqual(bag_contract.symbol, 'KC')
        self.assertEqual(bag_contract.secType, 'BAG')

        # Verify Limit Order Logic
        # Combo Position: ... -> Close Action: BUY (bag_action defaults to BUY as per code, unless we changed it logic?)
        # Wait, the code sets bag_action based on final_legs_list[0]['action'] if size is 1?
        # Or if multiple legs, bag_action = 'BUY' and we rely on leg actions?
        # My code:
        # bag_action = 'BUY'
        # if len(final_legs_list) == 1:
        #    bag_action = final_legs_list[0]['action']

        # So for Bag it is BUY.
        # Limit price for BUY is ASK (5.5).
        # UPDATED: Code now calculates price from legs (5.25 - 5.25 = 0) + slippage (0.05).
        self.assertEqual(bag_order.orderType, 'LMT')
        self.assertEqual(bag_order.lmtPrice, 0.05)
        self.assertEqual(bag_order.action, 'BUY')

import pytest


class TestStaleCloseContractFormat:
    """Verify close_stale_positions uses pre-qualified contracts from reqPositionsAsync
    to avoid Error 478 strike format mismatch (e.g., strike=270.0 vs 2.7 for KC coffee)."""

    @pytest.fixture
    def ledger_path(self, tmp_path):
        """Create a temp ledger with a single old position."""
        path = tmp_path / "trade_ledger.csv"
        ten_days_ago = datetime(2023, 10, 13, 10, 0)
        data = {
            'timestamp': [ten_days_ago.strftime('%Y-%m-%d %H:%M:%S')],
            'position_id': ['TEST_POS'],
            'combo_id': [100],
            'local_symbol': ['KON6 P2.7'],
            'action': ['BUY'],
            'quantity': [1],
            'avg_fill_price': [0.50],
            'strike': [270],
            'right': ['P'],
            'total_value_usd': [-1875],
            'reason': ['Strategy Execution'],
        }
        pd.DataFrame(data).to_csv(path, index=False)
        return str(path)

    @pytest.mark.asyncio
    @patch('trading_bot.order_manager.place_order', new_callable=MagicMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IBConnectionPool')
    async def test_single_leg_uses_ib_contract_not_requalify(
        self, mock_pool, mock_log_trade, mock_place_order, ledger_path
    ):
        """Single-leg close must use pos.contract from reqPositionsAsync, not re-qualify.

        The IB contract from reqPositionsAsync has strike=2.7 (correct exchange format).
        Re-qualifying via qualifyContractsAsync would return strike=270.0 → Error 478.
        """
        mock_ib = AsyncMock()
        mock_pool.get_connection = AsyncMock(return_value=mock_ib)
        mock_ib.isConnected = MagicMock(return_value=True)

        # Contract from reqPositionsAsync has the correct exchange format (strike=2.7)
        ib_contract = Contract(
            localSymbol='KON6 P2.7', symbol='KC', conId=999,
            exchange='NYBOT', secType='FOP', strike=2.7, right='P',
            multiplier='37500', currency='USD'
        )
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[
            Position(account='test', contract=ib_contract, position=1, avgCost=1875.0)
        ])

        # If qualifyContractsAsync is called, it returns WRONG strike format
        bad_contract = Contract(conId=999, strike=270.0, secType='FOP', symbol='KC')
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[bad_contract])

        mock_ticker = MagicMock()
        mock_ticker.bid = 0.50
        mock_ticker.ask = 0.55
        mock_ticker.last = 0.52
        mock_ib.reqMktData = MagicMock(return_value=mock_ticker)
        mock_ib.cancelMktData = MagicMock()

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = 'Filled'
        mock_trade.orderStatus.avgFillPrice = 0.52
        mock_fill = MagicMock()
        mock_fill.commissionReport.realizedPNL = -50.0
        mock_trade.fills = [mock_fill]
        mock_trade.order.orderId = 1
        mock_place_order.return_value = mock_trade

        config = {'symbol': 'KC', 'exchange': 'NYBOT'}

        with patch('trading_bot.order_manager.get_trade_ledger_df') as mock_ledger, \
             patch('trading_bot.order_manager.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2023, 10, 23, 17, 20)  # Monday
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_ledger.return_value = pd.read_csv(ledger_path, parse_dates=['timestamp'])

            await close_stale_positions(config)

        # Verify place_order was called
        assert mock_place_order.call_count == 1

        # The contract passed to place_order must be the IB contract (strike=2.7),
        # NOT the re-qualified one (strike=270.0)
        call_args = mock_place_order.call_args[0]
        placed_contract = call_args[1]
        assert placed_contract.strike == 2.7, (
            f"Expected strike=2.7 from reqPositionsAsync, got {placed_contract.strike}. "
            "Error 478 would occur with strike=270.0"
        )
        # qualifyContractsAsync should NOT have been called for the close order
        mock_ib.qualifyContractsAsync.assert_not_called()

    @pytest.mark.asyncio
    @patch('trading_bot.order_manager.place_order', new_callable=MagicMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IBConnectionPool')
    async def test_missing_exchange_falls_back_to_config(
        self, mock_pool, mock_log_trade, mock_place_order, ledger_path
    ):
        """When pos.contract has no exchange, the config exchange should be used.

        reqPositionsAsync can return contracts without exchange field,
        causing Error 321 (Missing order exchange). The fix fills in from config.
        """
        mock_ib = AsyncMock()
        mock_pool.get_connection = AsyncMock(return_value=mock_ib)
        mock_ib.isConnected = MagicMock(return_value=True)

        # Contract from reqPositionsAsync WITHOUT exchange (reproduces Error 321)
        ib_contract = Contract(
            localSymbol='KON6 P2.7', symbol='KC', conId=999,
            exchange='', secType='FOP', strike=2.7, right='P',
            multiplier='37500', currency='USD'
        )
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[
            Position(account='test', contract=ib_contract, position=1, avgCost=1875.0)
        ])
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        mock_ticker = MagicMock()
        mock_ticker.bid = 0.50
        mock_ticker.ask = 0.55
        mock_ticker.last = 0.52
        mock_ib.reqMktData = MagicMock(return_value=mock_ticker)
        mock_ib.cancelMktData = MagicMock()

        mock_trade = MagicMock()
        mock_trade.orderStatus.status = 'Filled'
        mock_trade.orderStatus.avgFillPrice = 0.52
        mock_fill = MagicMock()
        mock_fill.commissionReport.realizedPNL = -50.0
        mock_trade.fills = [mock_fill]
        mock_trade.order.orderId = 1
        mock_place_order.return_value = mock_trade

        config = {'symbol': 'KC', 'exchange': 'NYBOT'}

        with patch('trading_bot.order_manager.get_trade_ledger_df') as mock_ledger, \
             patch('trading_bot.order_manager.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2023, 10, 23, 17, 20)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_ledger.return_value = pd.read_csv(ledger_path, parse_dates=['timestamp'])

            await close_stale_positions(config)

        assert mock_place_order.call_count == 1
        placed_contract = mock_place_order.call_args[0][1]
        assert placed_contract.exchange == 'NYBOT', (
            f"Expected exchange='NYBOT' from config fallback, got '{placed_contract.exchange}'. "
            "Error 321 would occur with empty exchange."
        )


class TestHoldingTimeGate:
    """Tests for the Friday-specific holding-time threshold in generate_and_execute_orders."""

    @pytest.mark.asyncio
    @patch('trading_bot.order_manager.is_market_open', return_value=True)
    @patch('trading_bot.order_manager.hours_until_weekly_close', return_value=3.0)
    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.generate_and_queue_orders', new_callable=AsyncMock)
    async def test_friday_uses_relaxed_threshold(
        self, mock_gen, mock_notify, mock_hours, mock_market
    ):
        """On weekly-close day with 3h remaining, friday threshold (2h) should allow."""
        from trading_bot.order_manager import generate_and_execute_orders
        config = {'risk_management': {
            'min_holding_hours': 6.0,
            'friday_min_holding_hours': 2.0,
        }, 'notifications': {}}
        await generate_and_execute_orders(config)
        # Should NOT be skipped — generate_and_queue_orders should be called
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    @patch('trading_bot.order_manager.is_market_open', return_value=True)
    @patch('trading_bot.order_manager.hours_until_weekly_close', return_value=3.0)
    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.generate_and_queue_orders', new_callable=AsyncMock)
    async def test_normal_day_uses_standard_threshold(
        self, mock_gen, mock_notify, mock_hours, mock_market
    ):
        """On normal day, hours_until_weekly_close returns inf, so standard threshold applies."""
        mock_hours.return_value = float('inf')
        from trading_bot.order_manager import generate_and_execute_orders
        config = {'risk_management': {
            'min_holding_hours': 6.0,
            'friday_min_holding_hours': 2.0,
        }, 'notifications': {}}
        await generate_and_execute_orders(config)
        # inf > 6.0 so should proceed
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    @patch('trading_bot.order_manager.is_market_open', return_value=True)
    @patch('trading_bot.order_manager.hours_until_weekly_close', return_value=1.0)
    @patch('trading_bot.order_manager.send_pushover_notification')
    @patch('trading_bot.order_manager.generate_and_queue_orders', new_callable=AsyncMock)
    async def test_friday_too_late_blocks(
        self, mock_gen, mock_notify, mock_hours, mock_market
    ):
        """On weekly-close day with only 1h remaining, even friday threshold blocks."""
        from trading_bot.order_manager import generate_and_execute_orders
        config = {'risk_management': {
            'min_holding_hours': 6.0,
            'friday_min_holding_hours': 2.0,
        }, 'notifications': {}}
        await generate_and_execute_orders(config)
        # 1.0 < 2.0 so should be blocked
        mock_gen.assert_not_called()


if __name__ == '__main__':
    unittest.main()
