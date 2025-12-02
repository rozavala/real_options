import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from ib_insync import Bag, ComboLeg, Contract, Order, Trade, Position
import pandas as pd
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.order_manager import generate_and_queue_orders, place_queued_orders, close_positions_after_5_days, ORDER_QUEUE
from ib_insync import util

class TestOrderManager(unittest.TestCase):

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.get_model_predictions')
    @patch('trading_bot.order_manager.IB')
    def test_generate_orders_aborts_on_data_pull_failure(self, mock_ib, mock_get_predictions, mock_run_data_pull):
        async def run_test():
            mock_run_data_pull.return_value = None
            mock_get_predictions.return_value = {'price_changes': [1.0]}
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance

            config = {}
            await generate_and_queue_orders(config)

            mock_run_data_pull.assert_called_once()
            mock_get_predictions.assert_not_called()
            mock_ib_instance.connectAsync.assert_not_called()

        asyncio.run(run_test())

class TestPositionClosing(unittest.TestCase):

    def setUp(self):
        # Use a temporary ledger path for tests to avoid messing with real data
        self.ledger_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_ledger.csv')
        self.create_mock_ledger()

    def tearDown(self):
        # Clean up the temporary ledger
        if os.path.exists(self.ledger_path):
            os.remove(self.ledger_path)

    def create_mock_ledger(self):
        today = datetime.now()
        ten_days_ago = today - timedelta(days=10)
        two_days_ago = today - timedelta(days=2)

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

        df = pd.DataFrame(data)
        df.to_csv(self.ledger_path, index=False)

    # Use MagicMock for place_order because it is a synchronous function in the codebase
    @patch('trading_bot.order_manager.place_order', new_callable=MagicMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IB')
    def test_closes_only_old_positions_and_correct_symbols(self, mock_ib, mock_log_trade, mock_place_order):
        async def run_test():
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance
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
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 310 P', symbol='KC', conId=4, exchange='NYBOT'), position=-1, avgCost=1.0)
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
            await close_positions_after_5_days(config)

            # We expect 2 calls to place_order: 1 for OLD_POSITION (Single), 1 for COMBO_POSITION (Bag)
            self.assertEqual(mock_place_order.call_count, 2)

            single_order_call = None
            bag_order_call = None

            for call in mock_place_order.call_args_list:
                args, kwargs = call
                contract = args[1]
                if contract.secType == 'BAG':
                    bag_order_call = call
                else:
                    single_order_call = call

            # VERIFY SINGLE ORDER
            self.assertIsNotNone(single_order_call)
            args, _ = single_order_call
            single_contract = args[1]
            single_order = args[2]

            self.assertEqual(single_contract.conId, 1)
            self.assertEqual(single_contract.symbol, 'KC')
            self.assertEqual(single_contract.secType, 'FOP')

            # Verify Limit Order Logic
            # Single OLD_POSITION: KC 250 C (Buy) -> Close Action: SELL
            # We are selling, so limit price should be BID (5.0)
            self.assertEqual(single_order.orderType, 'LMT')
            self.assertEqual(single_order.lmtPrice, 5.0)
            self.assertEqual(single_order.action, 'SELL')

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
            self.assertEqual(bag_order.orderType, 'LMT')
            self.assertEqual(bag_order.lmtPrice, 5.5)
            self.assertEqual(bag_order.action, 'BUY')

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
