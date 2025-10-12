import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from ib_insync import Contract, Future

from trading_bot.signal_generator import generate_signals


class TestSignalGenerator(unittest.TestCase):

    def test_generate_signals(self):
        async def run_test():
            ib = AsyncMock()
            # Thresholds are > 7 for BULLISH, < 0 for BEARISH
            config = {'symbol': 'KC', 'exchange': 'NYBOT', 'signal_thresholds': {'bullish': 7, 'bearish': 0}}
            api_response = {
                "price_changes": [
                    10.0,  # Corresponds to H -> BULLISH
                    -5.0,  # Corresponds to K -> BEARISH
                    3.0,   # Corresponds to N -> NO-TRADE
                    8.0,   # Corresponds to U -> BULLISH
                    -1.0   # Corresponds to Z -> BEARISH
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

            # The contracts are sorted chronologically. Let's trace the logic:
            # 1. Z25 (202512) -> month 12 -> code 'Z' -> prediction -1.0 -> BEARISH
            # 2. H26 (202603) -> month 3  -> code 'H' -> prediction 10.0 -> BULLISH
            # 3. K26 (202605) -> month 5  -> code 'K' -> prediction -5.0 -> BEARISH
            # 4. N26 (202607) -> month 7  -> code 'N' -> prediction 3.0  -> NO-TRADE
            # 5. U26 (202609) -> month 9  -> code 'U' -> prediction 8.0  -> BULLISH

            # The final signals list should be ordered chronologically:
            self.assertEqual(signals[0]['direction'], 'BEARISH')
            self.assertEqual(signals[0]['contract_month'], '202512')

            self.assertEqual(signals[1]['direction'], 'BULLISH')
            self.assertEqual(signals[1]['contract_month'], '202603')

            self.assertEqual(signals[2]['direction'], 'BEARISH')
            self.assertEqual(signals[2]['contract_month'], '202605')

            self.assertEqual(signals[3]['direction'], 'BULLISH')
            self.assertEqual(signals[3]['contract_month'], '202609')


        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()