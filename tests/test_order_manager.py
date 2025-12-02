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
            'timestamp': [ten_days_ago.strftime('%Y-%m-%d %H:%M:%S'), two_days_ago.strftime('%Y-%m-%d %H:%M:%S')],
            'position_id': ['OLD_POSITION', 'NEW_POSITION'],
            'combo_id': [123, 456],
            'local_symbol': ['KC 26JUL24 250 C', 'KC 26JUL24 260 C'],
            'action': ['BUY', 'BUY'],
            'quantity': [1, 1],
            'avg_fill_price': [1.0, 1.0],
            'strike': [250, 260],
            'right': ['C', 'C'],
            'total_value_usd': [-100, -100],
            'reason': ['Strategy Execution', 'Strategy Execution']
        }
        df = pd.DataFrame(data)
        df.to_csv(self.ledger_path, index=False)

    # Use MagicMock for place_order because it is a synchronous function in the codebase
    @patch('trading_bot.order_manager.place_order', new_callable=MagicMock)
    @patch('trading_bot.order_manager.log_trade_to_ledger', new_callable=AsyncMock)
    @patch('trading_bot.order_manager.IB')
    def test_closes_only_old_positions(self, mock_ib, mock_log_trade, mock_place_order):
        async def run_test():
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance
            mock_ib_instance.connectAsync = AsyncMock()
            mock_ib_instance.reqPositionsAsync = AsyncMock()

            # Mock Positions: 1 Old, 1 New (from Setup)
            # The ledger says: OLD_POSITION is KC...250 C (10 days ago)
            #                  NEW_POSITION is KC...260 C (2 days ago)
            # We must set up live positions to match.
            # Live: 1 of 250 C, 1 of 260 C.
            # Position size +1 means Long. Ledger action 'BUY' confirms this.

            mock_positions = [
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 250 C', symbol='KC', conId=1), position=1, avgCost=1.0),
                Position(account='test_account', contract=Contract(localSymbol='KC 26JUL24 260 C', symbol='KC', conId=2), position=1, avgCost=1.0)
            ]
            mock_ib_instance.reqPositionsAsync.return_value = mock_positions

            mock_trade = MagicMock()
            mock_trade.orderStatus.status = 'Filled'
            # Mock fills so the P&L calculation doesn't fail
            mock_fill = MagicMock()
            mock_fill.commissionReport.realizedPNL = 50.0
            mock_trade.fills = [mock_fill]
            mock_trade.orderStatus.avgFillPrice = 1.0

            mock_place_order.return_value = mock_trade

            config = {'symbol': 'KC', 'exchange': 'NYBOT'}
            await close_positions_after_5_days(config)

            # We expect 1 call to place_order (for the old position 250 C)
            self.assertEqual(mock_place_order.call_count, 1)

            # Extract the contract from the call arguments
            # Call args: (ib, contract, order)
            args, kwargs = mock_place_order.call_args
            placed_order_contract = args[1]

            # Verify it closed the OLD position (250 C)
            # Note: The code creates a new Contract object, but we set symbol=localSymbol name in the mock data logic?
            # Actually, my code sets contract.symbol = "DA" or uses the symbol from Live Position?
            # In 'Single Order' path: contract.symbol = leg['symbol']
            # And leg['symbol'] comes from live_pos.contract.localSymbol
            self.assertEqual(placed_order_contract.symbol, 'KC 26JUL24 250 C')

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
