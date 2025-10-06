import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# --- Add project root to path to allow imports ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---

from trading.risk import is_trade_sane, monitor_positions_for_risk
from ib_insync import Position, Contract, PnLSingle

class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        """Set up a mock config for all tests."""
        self.config = {
            "risk_management": {
                "min_confidence_threshold": 0.70,
                "stop_loss_percentage": 20.0,
                "take_profit_percentage": 50.0,
                "check_interval_seconds": 1
            }
        }

    # --- Tests for is_trade_sane ---

    def test_is_trade_sane_passes_with_high_confidence(self):
        """Test that a signal with confidence above the threshold passes."""
        signal = {"confidence": 0.85}
        self.assertTrue(is_trade_sane(signal, self.config))

    def test_is_trade_sane_fails_with_low_confidence(self):
        """Test that a signal with confidence below the threshold fails."""
        signal = {"confidence": 0.65}
        self.assertFalse(is_trade_sane(signal, self.config))

    def test_is_trade_sane_fails_with_no_confidence(self):
        """Test that a signal with no confidence key fails."""
        signal = {"direction": "BULLISH"}
        self.assertFalse(is_trade_sane(signal, self.config))

    # --- Tests for monitor_positions_for_risk ---

    @patch('trading.risk.wait_for_fill', new_callable=AsyncMock)
    def test_stop_loss_trigger(self, mock_wait_for_fill):
        """Verify that a stop-loss is triggered and a closing order is placed."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']

        # Mock position: 1 contract, cost basis $1000
        mock_position = Position(
            account='U123456',
            contract=Contract(conId=1, symbol='KC'),
            position=1,
            avgCost=1000.0
        )
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])

        # Mock PnL: Unrealized loss of $250 (-25%)
        mock_pnl = PnLSingle(unrealizedPnL=-250.0)
        mock_ib.reqPnLSingleAsync = AsyncMock(return_value=mock_pnl)
        mock_ib.placeOrder = MagicMock()

        # --- Run Test ---
        async def run_monitor():
            # Run the monitor for a short time to allow one check cycle
            monitor_task = asyncio.create_task(monitor_positions_for_risk(mock_ib, self.config))
            await asyncio.sleep(1.5) # Wait for check interval + buffer
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_monitor())

        # --- Assertions ---
        # 1. Was placeOrder called?
        mock_ib.placeOrder.assert_called_once()
        # 2. Was it a SELL order to close the long position?
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].action, 'SELL')
        # 3. Was the quantity correct?
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].totalQuantity, 1)


    @patch('trading.risk.wait_for_fill', new_callable=AsyncMock)
    def test_take_profit_trigger(self, mock_wait_for_fill):
        """Verify that a take-profit is triggered and a closing order is placed."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']

        # Mock position: -2 contracts (short), cost basis $500/contract
        mock_position = Position(
            account='U123456',
            contract=Contract(conId=2, symbol='KC'),
            position=-2,
            avgCost=500.0
        )
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])

        # Mock PnL: Unrealized profit of $600 (total), which is $300/contract (+60%)
        # Note: PnL is positive for short positions when price goes down.
        mock_pnl = PnLSingle(unrealizedPnL=600.0)
        mock_ib.reqPnLSingleAsync = AsyncMock(return_value=mock_pnl)
        mock_ib.placeOrder = MagicMock()

        # --- Run Test ---
        async def run_monitor():
            monitor_task = asyncio.create_task(monitor_positions_for_risk(mock_ib, self.config))
            await asyncio.sleep(1.5)
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_monitor())

        # --- Assertions ---
        # 1. Was placeOrder called?
        mock_ib.placeOrder.assert_called_once()
        # 2. Was it a BUY order to close the short position?
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].action, 'BUY')
        # 3. Was the quantity correct?
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].totalQuantity, 2)


    def test_no_trigger(self):
        """Verify that no order is placed when PnL is within limits."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']
        mock_position = Position(account='U123456', contract=Contract(conId=3), position=1, avgCost=1000.0)
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])
        # PnL is a 10% loss, which is within the -20% stop loss limit
        mock_pnl = PnLSingle(unrealizedPnL=-100.0)
        mock_ib.reqPnLSingleAsync = AsyncMock(return_value=mock_pnl)
        mock_ib.placeOrder = MagicMock()

        # --- Run Test ---
        async def run_monitor():
            monitor_task = asyncio.create_task(monitor_positions_for_risk(mock_ib, self.config))
            await asyncio.sleep(1.5)
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_monitor())

        # --- Assertions ---
        mock_ib.placeOrder.assert_not_called()


if __name__ == '__main__':
    unittest.main()