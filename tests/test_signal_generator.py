import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from ib_insync import Contract, Future

from trading_bot.signal_generator import generate_signals


class TestSignalGenerator(unittest.TestCase):

    def test_generate_signals(self):
        async def run_test():
            ib = AsyncMock()
            config = {'symbol': 'KC', 'exchange': 'NYBOT'}
            api_response = {
                "price_changes": [
                    10.0,  # BULLISH
                    -5.0,  # BEARISH
                    3.0,   # NO-TRADE
                    8.0,   # BULLISH
                    -1.0   # BEARISH
                ]
            }

            # Mock the active futures contracts, ensuring they have the necessary attributes
            # The sorting logic depends on lastTradeDateOrContractMonth
            mock_futures = [
                Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='20251212'), # Z25
                Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='20260312'), # H26
                Future(conId=3, symbol='KC', lastTradeDateOrContractMonth='20260512'), # K26
                Future(conId=4, symbol='KC', lastTradeDateOrContractMonth='20260712'), # N26
                Future(conId=5, symbol='KC', lastTradeDateOrContractMonth='20260912'), # U26
            ]
            ib.reqContractDetailsAsync.return_value = [MagicMock(contract=f) for f in mock_futures]

            signals = await generate_signals(ib, api_response, config)

            # We expect 4 signals (2 BULLISH, 2 BEARISH) because one price change is a no-trade
            self.assertEqual(len(signals), 4)

            # The order of signals should match the alphabetical order of month codes
            # H, K, N, U, Z -> maps to price changes at index 1, 2, 3, 4, 0

            # Expected order: H, K, N, U, Z
            # H26 -> -5.0 (BEARISH)
            # K26 -> 3.0 (NO-TRADE)
            # N26 -> 8.0 (BULLISH)
            # U26 -> -1.0 (BEARISH)
            # Z25 -> 10.0 (BULLISH)

            self.assertEqual(signals[0]['direction'], 'BULLISH') # Z25
            self.assertEqual(signals[0]['contract_month'], '202512')

            self.assertEqual(signals[1]['direction'], 'BEARISH') # H26
            self.assertEqual(signals[1]['contract_month'], '202603')

            self.assertEqual(signals[2]['direction'], 'BULLISH') # N26
            self.assertEqual(signals[2]['contract_month'], '202607')

            self.assertEqual(signals[3]['direction'], 'BEARISH') # U26
            self.assertEqual(signals[3]['contract_month'], '202609')


        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()