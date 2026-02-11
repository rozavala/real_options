import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import sys
import os


# Ensure python path is set correctly when running this
sys.path.append(os.getcwd())

from ib_insync import IB, Contract, FuturesOption, PnLSingle, Trade, OrderStatus
from trading_bot.risk_management import _check_risk_once

class TestIronCondorRisk(unittest.TestCase):
    def test_iron_condor_max_loss_calculation(self):
        async def run_test():
            ib = AsyncMock(spec=IB)
            ib.isConnected.return_value = True
            ib.managedAccounts.return_value = ['DU12345']

            # Setup Config
            config = {
                'notifications': {},
                'risk_management': {
                    'stop_loss_max_risk_pct': 0.40,  # 40% stop loss
                    'take_profit_capture_pct': 0.80,
                    'monitoring_grace_period_seconds': 0
                },
                'symbol': 'KC',
                'exchange': 'NYBOT'
            }

            # Setup Iron Condor Positions (4 legs)
            legs = []
            strikes = [90, 100, 110, 120]
            rights = ['P', 'P', 'C', 'C']
            positions = [1, -1, -1, 1]
            avg_costs = [1.0, 3.0, 3.0, 1.0]

            for i in range(4):
                leg = MagicMock()
                leg.contract = FuturesOption(symbol='KC', lastTradeDateOrContractMonth='202512', strike=strikes[i], right=rights[i], multiplier='1')
                leg.contract.conId = 100 + i
                leg.contract.localSymbol = f"KC_{rights[i]}_{strikes[i]}"
                leg.position = positions[i]
                leg.avgCost = avg_costs[i]
                legs.append(leg)

            # Fix: Ensure reqPositionsAsync returns a coroutine that yields the list
            f = asyncio.Future()
            f.set_result(legs)
            ib.reqPositionsAsync.return_value = f

            # Mock Contract Details to group them under same underlying
            # IB.reqContractDetailsAsync is also awaited
            f_details = asyncio.Future()
            f_details.set_result([MagicMock(underConId=999)])
            ib.reqContractDetailsAsync.side_effect = lambda c: f_details # return same future for all calls

            # Better way for side_effect with async:
            async def mock_contract_details(contract):
                return [MagicMock(underConId=999)]
            ib.reqContractDetailsAsync.side_effect = mock_contract_details

            # Map contract IDs to PnL values
            pnl_values = {
                100: -0.5,
                101: -1.0,
                102: -1.0,
                103: -0.5
            } # Total -3.0

            # Mock get_unrealized_pnl by inspecting the contract arg
            async def mock_get_pnl(ib_inst, contract):
                # Check conId
                return pnl_values.get(contract.conId, 0.0)

            # Patch get_unrealized_pnl directly to control PnL returns
            with patch('trading_bot.risk_management.get_unrealized_pnl', side_effect=mock_get_pnl) as mock_pnl_func:
                # Patch get_dollar_multiplier to return 1 (to simplify math)
                with patch('trading_bot.risk_management.get_dollar_multiplier', return_value=1.0):
                    # Patch get_trade_ledger_df to return empty dataframe (no grace period)
                    with patch('trading_bot.risk_management.get_trade_ledger_df') as mock_ledger:
                        mock_ledger.return_value.empty = True

                        await _check_risk_once(ib, config, set(), 0.0, 0.0)

            # Check if placeOrder was called (Stop Loss Triggered)
            # This assertion confirms that the fix is working.
            # Max Loss (6.0) -> Trigger @ 2.4 loss. Actual loss 3.0.
            # If bug present, Max Loss (26.0) -> Trigger @ 10.4 loss. Actual loss 3.0 -> No Trigger.
            self.assertTrue(ib.placeOrder.called, "Stop Loss should have been triggered for Iron Condor")

            # Verify details of the order if needed
            args, _ = ib.placeOrder.call_args
            contract = args[0]
            self.assertEqual(contract.secType, 'BAG')

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
