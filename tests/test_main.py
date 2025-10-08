import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from ib_insync import Future, PnLSingle

from trading_bot.main import main_runner

class TestMainRunner(unittest.TestCase):

    @patch('trading_bot.main.send_pushover_notification')
    @patch('trading_bot.main.send_heartbeat', new_callable=AsyncMock)
    @patch('trading_bot.main.get_active_futures', new_callable=AsyncMock)
    @patch('trading_bot.main.manage_existing_positions', new_callable=AsyncMock)
    @patch('trading_bot.main.build_option_chain', new_callable=AsyncMock)
    @patch('trading_bot.main.execute_directional_strategy', new_callable=AsyncMock)
    def test_main_runner_normalizes_price(self, mock_execute_strategy, mock_build_chain, mock_manage_positions, mock_get_futures, mock_heartbeat, mock_pushover):
        async def run_test():
            # --- Mocks ---
            config = {
                'symbol': 'KC', 'exchange': 'NYBOT', 'exchange_timezone': 'America/New_York',
                'connection': {}, 'notifications': {}
            }
            signals = [{'contract_month': '202512', 'prediction_type': 'DIRECTIONAL', 'direction': 'BULLISH'}]

            # Mock futures
            mock_future = Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='202512')
            mock_get_futures.return_value = [mock_future]

            # Mock IB connection and market data
            with patch('trading_bot.main.IB') as mock_ib_class:
                ib_instance = AsyncMock()
                # isConnected is called twice. First to check, second in the finally block.
                ib_instance.isConnected = MagicMock(side_effect=[False, True])
                ib_instance.disconnect = MagicMock()
                mock_ib_class.return_value = ib_instance

                # Mock market data ticker to return a magnified price
                mock_ticker = MagicMock()
                mock_ticker.marketPrice.return_value = 38290.0 # Magnified price (e.g., 382.90 * 100)
                # reqMktData is a synchronous method, so it needs a synchronous mock
                ib_instance.reqMktData = MagicMock(return_value=mock_ticker)

                # Mock downstream functions
                mock_manage_positions.return_value = True # Assume we should open a new trade
                mock_build_chain.return_value = {'some': 'chain'}

                # --- Run main_runner ---
                await main_runner(config, signals)

                # --- Assertions ---
                # Check that the strategy function was called with the NORMALIZED price
                mock_execute_strategy.assert_called_once()
                args, kwargs = mock_execute_strategy.call_args
                normalized_price = args[4] # price is the 5th positional argument
                self.assertAlmostEqual(normalized_price, 382.90)

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()