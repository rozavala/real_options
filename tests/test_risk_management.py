import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# --- Add project root to path to allow imports ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---

from trading.risk import is_trade_sane, monitor_positions_for_risk
from ib_insync import Position, Contract, PnL, PnLSingle

class TestRiskManagement(unittest.TestCase):

    def setUp(self):
        """Set up a mock config for all tests."""
        self.config = {
            "risk_management": {
                "min_confidence_threshold": 0.70,
                "stop_loss_percentage": 20.0,
                "take_profit_percentage": 50.0,
                "check_interval_seconds": 1
            },
            "notifications": {} # Mock notifications to avoid errors
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
        signal = {"direction": "BULLISH", "confidence": None}
        self.assertFalse(is_trade_sane(signal, self.config))

    # --- Tests for monitor_positions_for_risk ---

    @patch('trading.risk.wait_for_fill', new_callable=AsyncMock)
    @patch('trading.risk.send_pushover_notification')
    def test_stop_loss_trigger(self, mock_send_notification, mock_wait_for_fill):
        """Verify that a stop-loss is triggered and a closing order is placed."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']

        mock_position = Position(account='U123456', contract=Contract(conId=1), position=1, avgCost=1000.0)
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])

        # NEW MOCK LOGIC: Simulate the subscription object returned by reqPnLSingle
        mock_pnl_subscription = MagicMock()
        # The PnL data is nested inside the .pnl attribute of the subscription
        mock_pnl_subscription.pnl = PnLSingle(unrealizedPnL=-250.0) # -25% loss
        mock_ib.reqPnLSingle.return_value = mock_pnl_subscription

        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelPnLSingle = MagicMock()

        # --- Run Test ---
        asyncio.run(self._run_monitor_cycle(mock_ib))

        # --- Assertions ---
        mock_ib.reqPnLSingle.assert_called_once()
        mock_ib.placeOrder.assert_called_once()
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].action, 'SELL')
        mock_ib.cancelPnLSingle.assert_called_once_with(mock_pnl_subscription)


    @patch('trading.risk.wait_for_fill', new_callable=AsyncMock)
    @patch('trading.risk.send_pushover_notification')
    def test_take_profit_trigger(self, mock_send_notification, mock_wait_for_fill):
        """Verify that a take-profit is triggered and a closing order is placed."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']
        mock_position = Position(account='U123456', contract=Contract(conId=2), position=-2, avgCost=500.0)
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])

        # NEW MOCK LOGIC
        mock_pnl_subscription = MagicMock()
        mock_pnl_subscription.pnl = PnLSingle(unrealizedPnL=600.0) # +60% profit per contract
        mock_ib.reqPnLSingle.return_value = mock_pnl_subscription

        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelPnLSingle = MagicMock()

        # --- Run Test ---
        asyncio.run(self._run_monitor_cycle(mock_ib))

        # --- Assertions ---
        mock_ib.reqPnLSingle.assert_called_once()
        mock_ib.placeOrder.assert_called_once()
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].action, 'BUY')
        self.assertEqual(mock_ib.placeOrder.call_args[0][1].totalQuantity, 2)
        mock_ib.cancelPnLSingle.assert_called_once_with(mock_pnl_subscription)


    def test_no_trigger(self):
        """Verify that no order is placed when PnL is within limits."""
        # --- Mocks Setup ---
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        mock_ib.managedAccounts.return_value = ['U123456']
        mock_position = Position(account='U123456', contract=Contract(conId=3), position=1, avgCost=1000.0)
        mock_ib.reqPositionsAsync = AsyncMock(return_value=[mock_position])

        # NEW MOCK LOGIC
        mock_pnl_subscription = MagicMock()
        mock_pnl_subscription.pnl = PnLSingle(unrealizedPnL=-100.0) # -10% loss (within -20% limit)
        mock_ib.reqPnLSingle.return_value = mock_pnl_subscription

        mock_ib.placeOrder = MagicMock()
        mock_ib.cancelPnLSingle = MagicMock()

        # --- Run Test ---
        asyncio.run(self._run_monitor_cycle(mock_ib))

        # --- Assertions ---
        mock_ib.reqPnLSingle.assert_called_once()
        mock_ib.placeOrder.assert_not_called()
        mock_ib.cancelPnLSingle.assert_called_once() # Should still be called to clean up

    async def _run_monitor_cycle(self, mock_ib):
        """Helper to run the monitor for one cycle."""
        monitor_task = asyncio.create_task(monitor_positions_for_risk(mock_ib, self.config))
        # This needs to be long enough for the monitor's internal loop to run at least once.
        # monitor's check_interval is 1s, and the PnL fetch sleeps for 1s.
        # So we need to wait > 2s. Let's use 2.5s to be safe.
        await asyncio.sleep(2.5)
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    unittest.main()