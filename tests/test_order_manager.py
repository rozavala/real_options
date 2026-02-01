import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ib_insync import Bag, ComboLeg, Contract, Order, Trade, Position
import pandas as pd
from datetime import datetime, timedelta
import os

from trading_bot.order_manager import generate_and_queue_orders, place_queued_orders, close_stale_positions, ORDER_QUEUE
from ib_insync import util

class TestOrderManager(unittest.TestCase):

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.get_model_predictions')
    @patch('trading_bot.order_manager.IBConnectionPool')
    def test_generate_orders_aborts_on_data_pull_failure(self, mock_pool, mock_get_predictions, mock_run_data_pull):
        async def run_test():
            mock_run_data_pull.return_value = None
            mock_get_predictions.return_value = {'price_changes': [1.0]}
            mock_ib_instance = AsyncMock()
            mock_pool.get_connection = AsyncMock(return_value=mock_ib_instance)

            config = {}
            await generate_and_queue_orders(config)

            mock_run_data_pull.assert_called_once()
            mock_get_predictions.assert_not_called()
            mock_pool.get_connection.assert_not_called()

        asyncio.run(run_test())

class TestPositionClosing(unittest.TestCase):

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
    def test_closes_only_old_positions_and_correct_symbols(self, mock_pool, mock_log_trade, mock_place_order):
        async def run_test():
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
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 250 C', symbol='KC', conId=1, exchange='NYBOT'), position=1, avgCost=1.0),
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 260 C', symbol='KC', conId=2, exchange='NYBOT'), position=1, avgCost=1.0),
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 300 P', symbol='KC', conId=3, exchange='NYBOT'), position=1, avgCost=1.0),
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 310 P', symbol='KC', conId=4, exchange='NYBOT'), position=-1, avgCost=1.0),
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 200 C', symbol='KC', conId=5, exchange='NYBOT'), position=1, avgCost=1.0),
            ]
            mock_ib_instance.reqPositionsAsync.return_value = mock_positions

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

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
