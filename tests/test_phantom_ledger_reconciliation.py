"""
Tests for the phantom ledger reconciliation feature (P2).

Validates that _reconcile_phantom_ledger_entries correctly identifies
position_ids with non-zero net quantity in the trade ledger (but no
matching IB positions) and writes synthetic close rows.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import asyncio
import tempfile
import csv
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd

from orchestrator import _reconcile_phantom_ledger_entries


class TestPhantomLedgerReconciliation(unittest.IsolatedAsyncioTestCase):
    """Tests for _reconcile_phantom_ledger_entries."""

    def setUp(self):
        """Set up a temporary directory for ledger writes."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {'notifications': {}, 'data_dir': self.temp_dir}

    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_empty_ledger_returns_zero(self):
        """An empty trade ledger should return 0 with no side effects."""
        trade_ledger = pd.DataFrame()
        tms = MagicMock()

        result = await _reconcile_phantom_ledger_entries(
            trade_ledger, tms, self.config
        )

        self.assertEqual(result, 0)
        tms.invalidate_thesis.assert_not_called()

    async def test_balanced_ledger_returns_zero(self):
        """A ledger where all positions net to zero should return 0."""
        trade_ledger = pd.DataFrame({
            'position_id': ['POS_001', 'POS_001'],
            'local_symbol': ['KCH6 C350', 'KCH6 C350'],
            'action': ['BUY', 'SELL'],
            'quantity': [1, 1],
            'timestamp': [datetime.now(), datetime.now()]
        })
        tms = MagicMock()

        result = await _reconcile_phantom_ledger_entries(
            trade_ledger, tms, self.config
        )

        self.assertEqual(result, 0)
        tms.invalidate_thesis.assert_not_called()

    @patch('trading_bot.utils._get_data_dir')
    @patch('orchestrator.send_pushover_notification')
    async def test_phantom_entry_creates_synthetic_close(
        self, mock_notify, mock_get_data_dir
    ):
        """A position with non-zero net qty should get a synthetic close row."""
        mock_get_data_dir.return_value = self.temp_dir

        # Create a ledger CSV with one open position (BUY 2, no close)
        ledger_path = os.path.join(self.temp_dir, 'trade_ledger.csv')
        fieldnames = [
            'timestamp', 'position_id', 'combo_id', 'local_symbol',
            'action', 'quantity', 'avg_fill_price', 'strike', 'right',
            'total_value_usd', 'reason'
        ]
        with open(ledger_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                'timestamp': '2026-02-20 10:00:00',
                'position_id': 'POS_PHANTOM',
                'combo_id': 'POS_PHANTOM',
                'local_symbol': 'KCH6 C350',
                'action': 'BUY',
                'quantity': 2,
                'avg_fill_price': 5.50,
                'strike': '350',
                'right': 'C',
                'total_value_usd': -412.50,
                'reason': 'Bull call spread entry'
            })

        trade_ledger = pd.DataFrame({
            'position_id': ['POS_PHANTOM'],
            'local_symbol': ['KCH6 C350'],
            'action': ['BUY'],
            'quantity': [2],
            'timestamp': [datetime(2026, 2, 20)]
        })

        tms = MagicMock()
        tms.invalidate_thesis = MagicMock()

        result = await _reconcile_phantom_ledger_entries(
            trade_ledger, tms, self.config
        )

        # Should find 1 phantom entry
        self.assertEqual(result, 1)

        # TMS thesis should be invalidated
        tms.invalidate_thesis.assert_called_once_with(
            'POS_PHANTOM',
            "Phantom reconciliation: ledger had non-zero qty with no IB position"
        )

        # Verify the synthetic close row was appended
        df = pd.read_csv(ledger_path)
        self.assertEqual(len(df), 2)  # original + synthetic
        close_row = df.iloc[-1]
        self.assertEqual(close_row['position_id'], 'POS_PHANTOM')
        self.assertEqual(close_row['action'], 'SELL')  # Reversal of BUY
        self.assertEqual(close_row['quantity'], 2)
        self.assertIn('PHANTOM_RECONCILIATION', close_row['reason'])

        # Notification should be sent
        mock_notify.assert_called_once()

    @patch('trading_bot.utils._get_data_dir')
    @patch('orchestrator.send_pushover_notification')
    async def test_multiple_phantoms_across_positions(
        self, mock_notify, mock_get_data_dir
    ):
        """Multiple phantom position_ids should all get synthetic closes."""
        mock_get_data_dir.return_value = self.temp_dir

        # Create ledger CSV
        ledger_path = os.path.join(self.temp_dir, 'trade_ledger.csv')
        fieldnames = [
            'timestamp', 'position_id', 'combo_id', 'local_symbol',
            'action', 'quantity', 'avg_fill_price', 'strike', 'right',
            'total_value_usd', 'reason'
        ]
        with open(ledger_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        # POS_A: BUY 1 (open long), POS_B: SELL 3 (open short)
        trade_ledger = pd.DataFrame({
            'position_id': ['POS_A', 'POS_B'],
            'local_symbol': ['KCH6 C350', 'KCH6 P340'],
            'action': ['BUY', 'SELL'],
            'quantity': [1, 3],
            'timestamp': [datetime(2026, 2, 20), datetime(2026, 2, 21)]
        })

        tms = MagicMock()
        tms.invalidate_thesis = MagicMock()

        result = await _reconcile_phantom_ledger_entries(
            trade_ledger, tms, self.config
        )

        # Should find 2 phantom entries (one per position_id/symbol)
        self.assertEqual(result, 2)

        # Both theses should be invalidated
        self.assertEqual(tms.invalidate_thesis.call_count, 2)
        invalidated_ids = {
            call.args[0] for call in tms.invalidate_thesis.call_args_list
        }
        self.assertEqual(invalidated_ids, {'POS_A', 'POS_B'})

        # Verify synthetic rows appended
        df = pd.read_csv(ledger_path)
        self.assertEqual(len(df), 2)  # 2 synthetic rows (original header only)
        # POS_A was BUY, so close is SELL
        pos_a_row = df[df['position_id'] == 'POS_A'].iloc[0]
        self.assertEqual(pos_a_row['action'], 'SELL')
        self.assertEqual(pos_a_row['quantity'], 1)
        # POS_B was SELL, so close is BUY
        pos_b_row = df[df['position_id'] == 'POS_B'].iloc[0]
        self.assertEqual(pos_b_row['action'], 'BUY')
        self.assertEqual(pos_b_row['quantity'], 3)


if __name__ == '__main__':
    unittest.main()
