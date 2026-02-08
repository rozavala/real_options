import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from ib_insync import Contract, Bag, ComboLeg

from trading_bot.compliance import ComplianceGuardian


class TestComplianceVolume(unittest.TestCase):
    def setUp(self):
        self.config = {'compliance': {'max_volume_pct': 0.1}}
        with patch('trading_bot.compliance.HeterogeneousRouter'):
            self.guardian = ComplianceGuardian(self.config)

        self.mock_ib = MagicMock()
        self.mock_ib.isConnected.return_value = True

        self.mock_ib.reqContractDetailsAsync = AsyncMock()
        self.mock_ib.qualifyContractsAsync = AsyncMock()
        self.mock_ib.reqHistoricalDataAsync = AsyncMock()

    def test_volume_standard_future(self):
        """Standard future (FUT) - no underlying resolution needed."""
        async def run_test():
            contract = Contract(conId=123, secType='FUT', symbol='KC')

            mock_bar = MagicMock()
            mock_bar.volume = 100
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, contract)

            self.mock_ib.reqContractDetailsAsync.assert_not_called()
            self.mock_ib.reqHistoricalDataAsync.assert_called_once()
            args, kwargs = self.mock_ib.reqHistoricalDataAsync.call_args
            self.assertEqual(args[0], contract)
            self.assertEqual(vol, 100.0)

        asyncio.run(run_test())

    def test_volume_fop_resolution(self):
        """FOP contract - should resolve underlying future via underlyingConId."""
        async def run_test():
            fop_contract = Contract(conId=999, secType='FOP', symbol='KC')

            details = MagicMock()
            details.underConId = 12345
            self.mock_ib.reqContractDetailsAsync.return_value = [details]

            fut_contract = Contract(conId=12345, secType='FUT', symbol='KC', localSymbol='KCH6')
            self.mock_ib.qualifyContractsAsync.return_value = [fut_contract]

            mock_bar = MagicMock()
            mock_bar.volume = 500
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, fop_contract)

            self.mock_ib.reqContractDetailsAsync.assert_called()
            self.mock_ib.qualifyContractsAsync.assert_called()
            qualified_arg = self.mock_ib.qualifyContractsAsync.call_args[0][0]
            self.assertEqual(qualified_arg.conId, 12345)

            self.mock_ib.reqHistoricalDataAsync.assert_called()
            hist_arg = self.mock_ib.reqHistoricalDataAsync.call_args[0][0]
            self.assertEqual(hist_arg, fut_contract)

            self.assertEqual(vol, 500.0)

        asyncio.run(run_test())

    def test_volume_bag_resolution(self):
        """BAG contract - should resolve underlying via first leg."""
        async def run_test():
            leg1 = ComboLeg(conId=888)
            bag_contract = Bag(comboLegs=[leg1], symbol='KC')

            details = MagicMock()
            details.underConId = 12345
            self.mock_ib.reqContractDetailsAsync.return_value = [details]

            fut_contract = Contract(conId=12345, secType='FUT', symbol='KC', localSymbol='KCH6')
            self.mock_ib.qualifyContractsAsync.return_value = [fut_contract]

            mock_bar = MagicMock()
            mock_bar.volume = 1000
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, bag_contract)

            self.mock_ib.reqContractDetailsAsync.assert_called()
            checked_contract = self.mock_ib.reqContractDetailsAsync.call_args[0][0]
            self.assertEqual(checked_contract.conId, 888)

            self.assertEqual(vol, 1000.0)

        asyncio.run(run_test())

    @patch('trading_bot.utils.get_market_data_cached')
    def test_fallback_yfinance(self, mock_get_market_data):
        """Test fallback to YFinance when IB fails or returns no volume."""
        async def run_test():
            contract = Contract(conId=123, secType='FUT', symbol='KC')

            self.mock_ib.reqHistoricalDataAsync.side_effect = Exception("IB Error")

            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.columns = ['Volume']
            mock_series = MagicMock()
            mock_series.iloc.__getitem__.return_value = 750
            mock_df.__getitem__.return_value = mock_series

            mock_get_market_data.return_value = mock_df

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, contract)

            self.mock_ib.reqHistoricalDataAsync.assert_called()

            mock_get_market_data.assert_called()
            args = mock_get_market_data.call_args[0]
            self.assertIn("KC=F", args[0])

            self.assertEqual(vol, 750.0)

        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()
