import asyncio
import unittest
from unittest.mock import patch, AsyncMock, MagicMock, call
from orchestrator import run_sentinels

class TestSentinelLoop(unittest.IsolatedAsyncioTestCase):

    @patch('orchestrator.IBConnectionPool')
    @patch('orchestrator.PriceSentinel')
    @patch('orchestrator.WeatherSentinel')
    @patch('orchestrator.LogisticsSentinel')
    @patch('orchestrator.NewsSentinel')
    @patch('orchestrator.XSentimentSentinel')
    @patch('orchestrator.PredictionMarketSentinel')
    @patch('orchestrator.MacroContagionSentinel')
    @patch('orchestrator.MicrostructureSentinel')
    @patch('orchestrator.is_market_open')
    @patch('orchestrator.is_trading_day')
    @patch('orchestrator.configure_market_data_type')
    @patch('orchestrator.GLOBAL_DEDUPLICATOR')
    @patch('orchestrator.get_active_futures', new_callable=AsyncMock)
    async def test_run_sentinels_lazy_init_and_market_hours(
        self, mock_get_futures, mock_dedup, mock_configure, mock_is_trading, mock_is_market_open,
        mock_micro_class, mock_macro_class, mock_prediction_class, mock_x_class, mock_news_class, mock_logistics_class,
        mock_weather_class, mock_price_class, mock_ib_pool
    ):
        # Setup mocks
        # Use MagicMock for connection object so methods like isConnected() are synchronous
        mock_ib_conn = MagicMock()
        mock_ib_conn.isConnected.return_value = True
        mock_ib_conn.connect = AsyncMock()
        mock_ib_conn.disconnect = MagicMock()

        # get_connection is awaited, so it must be an AsyncMock (or return a Future)
        # We replace the MagicMock attribute with an AsyncMock
        mock_ib_pool.get_connection = AsyncMock(return_value=mock_ib_conn)
        mock_ib_pool.release_connection = AsyncMock()

        mock_price_instance = AsyncMock()
        mock_price_class.return_value = mock_price_instance

        mock_get_futures.return_value = [MagicMock(localSymbol="KC_FUT")]

        # Scenario:
        # 1. Market Closed (Should NOT connect)
        # 2. Market Open (Should connect)
        # 3. Market Open (Should stay connected)
        # 4. Market Closed (Should disconnect)
        # 5. CancelledError (Stop loop)

        mock_is_market_open.side_effect = [False, True, True, False, asyncio.CancelledError]
        mock_is_trading.return_value = True

        # Need to mock sleep to run fast
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
             await run_sentinels(config={'symbol': 'KC', 'exchange': 'NYBOT'})

        # --- VERIFICATION ---

        # 1. Initialization
        # PriceSentinel should be initialized with None initially
        mock_price_class.assert_called_with({'symbol': 'KC', 'exchange': 'NYBOT'}, None)

        # 2. Iteration 1 (Market Closed)

        # 3. Iteration 2 (Market Open) -> Connection established
        mock_ib_pool.get_connection.assert_any_call("sentinel", {'symbol': 'KC', 'exchange': 'NYBOT'})
        mock_configure.assert_called_with(mock_ib_conn)

        # 4. Iteration 4 (Market Closed) -> Disconnect
        mock_ib_conn.disconnect.assert_called()

        # Verify clean up at the end
        self.assertIsNone(mock_price_instance.ib)

        print("Test passed: Lazy init, injection, and disconnect verified.")

if __name__ == '__main__':
    unittest.main()
