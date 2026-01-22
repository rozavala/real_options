import sys
import asyncio
import unittest
import logging
from unittest.mock import MagicMock, patch, AsyncMock

# Configure logging to see output
logging.basicConfig(level=logging.DEBUG)

# 1. Mock dependencies BEFORE importing trading_bot modules
mock_ib_insync = MagicMock()
sys.modules["ib_insync"] = mock_ib_insync

mock_numpy = MagicMock()
mock_numpy.random.uniform.return_value = 1.0
mock_numpy.log.return_value = 0.0
mock_numpy.sqrt.return_value = 1.0
mock_numpy.exp.return_value = 1.0
sys.modules["numpy"] = mock_numpy

mock_scipy = MagicMock()
sys.modules["scipy"] = mock_scipy
sys.modules["scipy.stats"] = MagicMock()

mock_holidays = MagicMock()
sys.modules["holidays"] = mock_holidays

mock_pytz = MagicMock()
mock_pytz.UTC = 'UTC'
mock_pytz.timezone.return_value = 'UTC'
sys.modules["pytz"] = mock_pytz

# Define Mock classes that compliance.py and utils.py will use
class MockContract(MagicMock):
    def __init__(self, conId=None, **kwargs):
        super().__init__(**kwargs)
        self.conId = conId
        self.secType = kwargs.get('secType', 'FUT')
        self.symbol = kwargs.get('symbol', 'TEST')
        self.lastTradeDateOrContractMonth = kwargs.get('lastTradeDateOrContractMonth', '202603')
        self.localSymbol = kwargs.get('localSymbol', 'TEST')
        self.comboLegs = kwargs.get('comboLegs', [])
        self.multiplier = kwargs.get('multiplier', '37500')

    def __repr__(self):
        return f"MockContract(conId={self.conId}, secType={self.secType})"

class MockBag(MockContract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.secType = 'BAG'

class MockComboLeg(MagicMock):
    def __init__(self, conId=None, **kwargs):
        super().__init__(**kwargs)
        self.conId = conId

class MockContractDetails(MagicMock):
    def __init__(self, underConId=None):
        super().__init__()
        self.underConId = underConId
        self.contract = MockContract()

# Attach classes to the mock module
mock_ib_insync.Contract = MockContract
mock_ib_insync.Bag = MockBag
mock_ib_insync.ComboLeg = MockComboLeg
mock_ib_insync.FuturesOption = MagicMock()
mock_ib_insync.IB = MagicMock
mock_ib_insync.Trade = MagicMock
mock_ib_insync.Position = MagicMock
mock_ib_insync.Fill = MagicMock
mock_ib_insync.util = MagicMock()

# Now we can safely import trading_bot modules
# Explicitly import utils to ensure it is available for patching
import trading_bot.utils
from trading_bot.compliance import ComplianceGuardian

class TestComplianceVolume(unittest.TestCase):
    def setUp(self):
        self.config = {'compliance': {'max_volume_pct': 0.1}}
        # Mocking router in init to avoid side effects
        with patch('trading_bot.compliance.HeterogeneousRouter'):
            self.guardian = ComplianceGuardian(self.config)

        self.mock_ib = MagicMock()
        self.mock_ib.isConnected.return_value = True

        # Async mocks
        self.mock_ib.reqContractDetailsAsync = AsyncMock()
        self.mock_ib.qualifyContractsAsync = AsyncMock()
        self.mock_ib.reqHistoricalDataAsync = AsyncMock()

    def test_volume_standard_future(self):
        """Standard future (FUT) - no underlying resolution needed."""
        async def run_test():
            contract = MockContract(conId=123, secType='FUT', symbol='KC')

            # Setup historical data return
            mock_bar = MagicMock()
            mock_bar.volume = 100
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, contract)

            # Should NOT call reqContractDetailsAsync
            self.mock_ib.reqContractDetailsAsync.assert_not_called()

            # Should call reqHistoricalDataAsync with the original contract
            self.mock_ib.reqHistoricalDataAsync.assert_called_once()
            args, kwargs = self.mock_ib.reqHistoricalDataAsync.call_args
            self.assertEqual(args[0], contract)
            self.assertEqual(vol, 100.0)

        asyncio.run(run_test())

    def test_volume_fop_resolution(self):
        """FOP contract - should resolve underlying future via underlyingConId."""
        async def run_test():
            # Setup FOP contract
            fop_contract = MockContract(conId=999, secType='FOP', symbol='KC')

            # Setup ContractDetails return with underlyingConId
            details = MockContractDetails(underConId=12345)
            self.mock_ib.reqContractDetailsAsync.return_value = [details]

            # Setup qualified future contract
            fut_contract = MockContract(conId=12345, secType='FUT', symbol='KC', localSymbol='KCH6')
            self.mock_ib.qualifyContractsAsync.return_value = [fut_contract]

            # Setup volume return for the FUTURE
            mock_bar = MagicMock()
            mock_bar.volume = 500
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, fop_contract)

            # Verification
            # 1. reqContractDetails called for FOP
            self.mock_ib.reqContractDetailsAsync.assert_called()
            # 2. qualifyContracts called for underlying future (conId 12345)
            self.mock_ib.qualifyContractsAsync.assert_called()
            qualified_arg = self.mock_ib.qualifyContractsAsync.call_args[0][0]
            self.assertEqual(qualified_arg.conId, 12345)

            # 3. reqHistoricalData called for the QUALIFIED FUTURE
            self.mock_ib.reqHistoricalDataAsync.assert_called()
            hist_arg = self.mock_ib.reqHistoricalDataAsync.call_args[0][0]
            self.assertEqual(hist_arg, fut_contract)

            self.assertEqual(vol, 500.0)

        asyncio.run(run_test())

    def test_volume_bag_resolution(self):
        """BAG contract - should resolve underlying via first leg."""
        async def run_test():
            # Setup BAG contract
            leg1 = MockComboLeg(conId=888)
            bag_contract = MockBag(secType='BAG', comboLegs=[leg1], symbol='KC')

            # Setup ContractDetails for LEG 1
            details = MockContractDetails(underConId=12345)
            self.mock_ib.reqContractDetailsAsync.return_value = [details]

            # Setup qualified future
            fut_contract = MockContract(conId=12345, secType='FUT', symbol='KC', localSymbol='KCH6')
            self.mock_ib.qualifyContractsAsync.return_value = [fut_contract]

            # Volume
            mock_bar = MagicMock()
            mock_bar.volume = 1000
            self.mock_ib.reqHistoricalDataAsync.return_value = [mock_bar]

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, bag_contract)

            # Verify reqContractDetails called for LEG 1 (conId 888)
            self.mock_ib.reqContractDetailsAsync.assert_called()
            # Check what contract was passed to reqContractDetails
            # It should be a Contract with conId=888
            checked_contract = self.mock_ib.reqContractDetailsAsync.call_args[0][0]
            self.assertEqual(checked_contract.conId, 888)

            self.assertEqual(vol, 1000.0)

        asyncio.run(run_test())

    @patch('trading_bot.utils.get_market_data_cached')
    def test_fallback_yfinance(self, mock_get_market_data):
        """Test fallback to YFinance when IB fails or returns no volume."""
        async def run_test():
            contract = MockContract(conId=123, secType='FUT', symbol='KC')

            # IB fails
            self.mock_ib.reqHistoricalDataAsync.side_effect = Exception("IB Error")

            # Mock YFinance return
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.columns = ['Volume']
            # Mocking iloc access
            mock_series = MagicMock()
            mock_series.iloc.__getitem__.return_value = 750
            mock_df.__getitem__.return_value = mock_series

            mock_get_market_data.return_value = mock_df

            vol = await self.guardian._fetch_volume_stats(self.mock_ib, contract)

            # Verify IB called
            self.mock_ib.reqHistoricalDataAsync.assert_called()

            # Verify YFinance called
            mock_get_market_data.assert_called()
            args = mock_get_market_data.call_args[0]
            self.assertIn("KC=F", args[0])

            self.assertEqual(vol, 750.0)

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
