"""
Integration tests for the Dynamic Exit Protocol.

These tests validate the complete thesis lifecycle from entry to invalidation.
They use mocked IB connections but real TMS storage (temp directory).
"""

import unittest
import asyncio
import tempfile
import shutil
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd

# Import the actual modules we're testing
from trading_bot.tms import TransactiveMemory
from trading_bot.order_manager import (
    record_entry_thesis_for_trade,
    _determine_guardian_from_reason,
    _build_invalidation_triggers
)
from trading_bot.strategy import validate_iron_condor_risk
from orchestrator import (
    _validate_thesis,
    _find_position_id_for_contract,
    run_position_audit_cycle
)
from trading_bot.sentinels import SentinelTrigger


class TestThesisLifecycleIntegration(unittest.IsolatedAsyncioTestCase):
    """
    Integration tests that validate the complete thesis lifecycle.
    Uses a temporary TMS directory to avoid polluting production data.
    """

    def setUp(self):
        """Create a temporary TMS directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.tms = TransactiveMemory(persist_path=self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_full_thesis_lifecycle_bull_spread(self):
        """
        End-to-end test: Bull Call Spread opened on frost thesis,
        closed when WeatherSentinel reports rain.
        """
        # === PHASE 1: TRADE ENTRY ===
        position_id = "TEST_POS_001"
        decision = {
            'direction': 'BULLISH',
            'reason': 'Frost risk in Minas Gerais coffee region',
            'regime': 'TRENDING',
            'volatility_sentiment': 'NEUTRAL',
            'confidence': 0.75
        }
        entry_price = 350.0

        # Record the thesis (uses our temp TMS)
        thesis_data = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'guardian_agent': _determine_guardian_from_reason(decision['reason']),
            'primary_rationale': decision['reason'],
            'invalidation_triggers': _build_invalidation_triggers('BULL_CALL_SPREAD', decision),
            'supporting_data': {
                'entry_price': entry_price,
                'entry_regime': decision['regime'],
                'volatility_sentiment': decision['volatility_sentiment'],
                'confidence': decision['confidence']
            },
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_regime': decision['regime']
        }

        self.tms.record_trade_thesis(position_id, thesis_data)

        # === PHASE 2: VERIFY THESIS STORED ===
        retrieved = self.tms.retrieve_thesis(position_id)

        self.assertIsNotNone(retrieved, "Thesis should be retrievable after recording")
        self.assertEqual(retrieved['strategy_type'], 'BULL_CALL_SPREAD')
        self.assertEqual(retrieved['guardian_agent'], 'Agronomist')
        self.assertIn('rain', retrieved['invalidation_triggers'])

        # === PHASE 3: VERIFY GUARDIAN LOOKUP ===
        affected = self.tms.get_active_theses_by_guardian('Agronomist')
        self.assertEqual(len(affected), 1, "Should find one thesis for Agronomist")
        self.assertEqual(affected[0]['primary_rationale'], decision['reason'])

        # === PHASE 4: SIMULATE SENTINEL TRIGGER ===
        # WeatherSentinel detects rain forecast
        trigger = SentinelTrigger(
            source='WeatherSentinel',
            reason='Heavy rain forecast for Minas Gerais next 48 hours',
            payload={'region': 'minas_gerais', 'precipitation_mm': 45},
            severity=7
        )

        # Check if trigger keywords match invalidation triggers
        trigger_keywords = trigger.reason.lower()
        thesis_invalidated = any(
            inv.lower() in trigger_keywords
            for inv in retrieved['invalidation_triggers']
        )

        # 'rain_forecast' should match 'rain forecast'
        self.assertTrue(thesis_invalidated, "Rain forecast should invalidate frost thesis")

        # === PHASE 5: INVALIDATE THESIS ===
        self.tms.invalidate_thesis(position_id, f"Sentinel: {trigger.source}")

        # Verify thesis is marked inactive
        # Note: retrieve_thesis still returns the thesis, but metadata.active = "false"
        # We verify by checking get_active_theses_by_guardian returns empty
        still_active = self.tms.get_active_theses_by_guardian('Agronomist')
        self.assertEqual(len(still_active), 0, "No active theses should remain for Agronomist")

    async def test_iron_condor_regime_breach_integration(self):
        """
        End-to-end test: Iron Condor entered in RANGE_BOUND regime,
        invalidated when regime shifts to HIGH_VOLATILITY.
        """
        position_id = "TEST_IC_001"

        thesis_data = {
            'strategy_type': 'IRON_CONDOR',
            'guardian_agent': 'VolatilityAnalyst',
            'primary_rationale': 'Premium harvest in range-bound market',
            'invalidation_triggers': _build_invalidation_triggers('IRON_CONDOR', {'reason': 'range'}),
            'supporting_data': {
                'entry_price': 345.0,
                'entry_regime': 'RANGE_BOUND',
                'volatility_sentiment': 'BULLISH',
                'confidence': 0.65
            },
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_regime': 'RANGE_BOUND'
        }

        self.tms.record_trade_thesis(position_id, thesis_data)

        # Mock position and IB for _validate_thesis
        mock_position = MagicMock()
        mock_position.contract = MagicMock()
        mock_position.position = 1

        mock_ib = MagicMock()
        mock_config = {'symbol': 'KC', 'exchange': 'NYBOT'}
        mock_council = MagicMock()

        # Test with RANGE_BOUND regime (should HOLD)
        with patch('orchestrator._get_current_regime', new_callable=AsyncMock) as mock_regime, \
             patch('orchestrator._get_current_price', new_callable=AsyncMock) as mock_price:
            mock_regime.return_value = 'RANGE_BOUND'
            mock_price.return_value = 345.0

            result = await _validate_thesis(
                thesis_data, mock_position, mock_council, mock_config, mock_ib
            )

            self.assertEqual(result['action'], 'HOLD', "Should HOLD when regime unchanged")

        # Test with HIGH_VOLATILITY regime (should CLOSE)
        with patch('orchestrator._get_current_regime', new_callable=AsyncMock) as mock_regime, \
             patch('orchestrator._get_current_price', new_callable=AsyncMock) as mock_price:
            mock_regime.return_value = 'HIGH_VOLATILITY'
            mock_price.return_value = 345.0

            result = await _validate_thesis(
                thesis_data, mock_position, mock_council, mock_config, mock_ib
            )

            self.assertEqual(result['action'], 'CLOSE', "Should CLOSE on regime breach")
            self.assertIn('REGIME BREACH', result['reason'])

    async def test_long_straddle_theta_burn_integration(self):
        """
        End-to-end test: Long Straddle held too long without price movement.
        """
        position_id = "TEST_LS_001"

        # Entry was 5 hours ago
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()

        thesis_data = {
            'strategy_type': 'LONG_STRADDLE',
            'guardian_agent': 'Master',
            'primary_rationale': 'Earnings catalyst expected',
            'invalidation_triggers': _build_invalidation_triggers('LONG_STRADDLE', {'reason': 'catalyst'}),
            'supporting_data': {
                'entry_price': 350.0,
                'entry_regime': 'TRENDING',
                'volatility_sentiment': 'NEUTRAL',
                'confidence': 0.70
            },
            'entry_timestamp': entry_time,
            'entry_regime': 'TRENDING'
        }

        self.tms.record_trade_thesis(position_id, thesis_data)

        mock_position = MagicMock()
        mock_position.contract = MagicMock()
        mock_ib = MagicMock()
        mock_config = {}
        mock_council = MagicMock()

        # Price barely moved (0.5% < 1% hurdle)
        with patch('orchestrator._get_current_price', new_callable=AsyncMock) as mock_price:
            mock_price.return_value = 351.75  # 0.5% move

            result = await _validate_thesis(
                thesis_data, mock_position, mock_council, mock_config, mock_ib
            )

            self.assertEqual(result['action'], 'CLOSE', "Should CLOSE on theta burn")
            self.assertIn('THETA BURN', result['reason'])

    def test_multiple_theses_different_guardians(self):
        """
        Test that multiple positions with different guardians are tracked correctly.
        """
        # Position 1: Weather-driven (Agronomist)
        self.tms.record_trade_thesis("POS_WEATHER", {
            'strategy_type': 'BULL_CALL_SPREAD',
            'guardian_agent': 'Agronomist',
            'primary_rationale': 'Drought risk',
            'invalidation_triggers': ['rain'],
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_regime': 'TRENDING',
            'supporting_data': {}
        })

        # Position 2: Logistics-driven (Logistics)
        self.tms.record_trade_thesis("POS_LOGISTICS", {
            'strategy_type': 'BULL_CALL_SPREAD',
            'guardian_agent': 'Logistics',
            'primary_rationale': 'Port strike',
            'invalidation_triggers': ['strike_resolution'],
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_regime': 'TRENDING',
            'supporting_data': {}
        })

        # Position 3: Volatility (VolatilityAnalyst)
        self.tms.record_trade_thesis("POS_VOL", {
            'strategy_type': 'IRON_CONDOR',
            'guardian_agent': 'VolatilityAnalyst',
            'primary_rationale': 'Range bound',
            'invalidation_triggers': ['regime_shift'],
            'entry_timestamp': datetime.now(timezone.utc).isoformat(),
            'entry_regime': 'RANGE_BOUND',
            'supporting_data': {}
        })

        # Verify isolation
        agro_theses = self.tms.get_active_theses_by_guardian('Agronomist')
        logistics_theses = self.tms.get_active_theses_by_guardian('Logistics')
        vol_theses = self.tms.get_active_theses_by_guardian('VolatilityAnalyst')

        self.assertEqual(len(agro_theses), 1)
        self.assertEqual(len(logistics_theses), 1)
        self.assertEqual(len(vol_theses), 1)

        # Invalidate Agronomist thesis
        self.tms.invalidate_thesis("POS_WEATHER", "Test invalidation")

        # Verify only Agronomist affected
        agro_after = self.tms.get_active_theses_by_guardian('Agronomist')
        logistics_after = self.tms.get_active_theses_by_guardian('Logistics')

        self.assertEqual(len(agro_after), 0, "Agronomist thesis should be inactive")
        self.assertEqual(len(logistics_after), 1, "Logistics thesis should be unaffected")


class TestPositionIdMapping(unittest.TestCase):
    """
    Tests for mapping live IB positions back to trade ledger position IDs.
    """

    def test_find_position_id_basic(self):
        """Test basic position ID lookup."""
        # Create mock position
        mock_position = MagicMock()
        mock_position.contract.localSymbol = "KCH6 C350"
        mock_position.position = 1

        # Create mock trade ledger
        trade_ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C350', 'KCH6 P340'],
            'position_id': ['POS_001', 'POS_002'],
            'action': ['BUY', 'BUY'],
            'quantity': [1, 1],
            'timestamp': [datetime.now(), datetime.now()]
        })

        result = _find_position_id_for_contract(mock_position, trade_ledger)

        # Should return the matching position_id
        self.assertEqual(result, 'POS_001')

    def test_find_position_id_no_match(self):
        """Test when no matching position exists."""
        mock_position = MagicMock()
        mock_position.contract.localSymbol = "KCZ6 C400"
        mock_position.position = 1

        trade_ledger = pd.DataFrame({
            'local_symbol': ['KCH6 C350'],
            'position_id': ['POS_001']
        })

        result = _find_position_id_for_contract(mock_position, trade_ledger)

        self.assertIsNone(result)


class TestIronCondorRiskValidation(unittest.TestCase):
    """
    Tests for Iron Condor position sizing validation.
    """

    def test_risk_within_limits(self):
        """Test that properly sized positions pass validation."""
        # 2% of $100,000 = $2,000 max loss
        self.assertTrue(
            validate_iron_condor_risk(1500, 100000, 0.02),
            "Position within limits should pass"
        )

    def test_risk_exceeds_limits(self):
        """Test that oversized positions fail validation."""
        self.assertFalse(
            validate_iron_condor_risk(2500, 100000, 0.02),
            "Position exceeding limits should fail"
        )

    def test_risk_at_boundary(self):
        """Test boundary condition."""
        # Exactly at limit should pass
        self.assertTrue(
            validate_iron_condor_risk(2000, 100000, 0.02),
            "Position exactly at limit should pass"
        )


if __name__ == '__main__':
    unittest.main()
