import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import json
from datetime import datetime, timezone, timedelta

# Import functions to test (assuming they are accessible or I mock them)
# Some functions are internal to orchestrator.py or order_manager.py
# I will import them or copy logic if they are strictly internal helpers not exposed.
# orchestrator.py helpers: _validate_thesis, _close_position_with_thesis_reason (hard to test without complex mocking)
# order_manager.py helpers: _determine_guardian_from_reason, _build_invalidation_triggers (I can import these if exposed)
# strategy.py: validate_iron_condor_risk (Importable)

# To import from orchestrator, I might need to deal with the top-level script nature.
# It is better to rely on what I can import.
# _determine_guardian_from_reason is inside order_manager.py.
# I will try to import it.

from trading_bot.order_manager import _determine_guardian_from_reason, _build_invalidation_triggers
from trading_bot.strategy import validate_iron_condor_risk
from trading_bot.tms import TransactiveMemory

# Mock for orchestrator._validate_thesis logic (since it's an async helper in a script)
# I will replicate the logic test or try to import it if possible.
# Importing orchestrator might run the script if not careful with if __name__ == "__main__".
# It has `if __name__ == "__main__":` so it should be safe.
from orchestrator import _validate_thesis

class TestExitLogic(unittest.IsolatedAsyncioTestCase):

    def test_guardian_mapping(self):
        """Test that reasons map to correct agents."""
        self.assertEqual(_determine_guardian_from_reason("Risk of frost in Minas Gerais"), "Agronomist")
        self.assertEqual(_determine_guardian_from_reason("Port of Santos strike"), "Logistics")
        self.assertEqual(_determine_guardian_from_reason("High volatility expected"), "VolatilityAnalyst")
        self.assertEqual(_determine_guardian_from_reason("BRL devaluing rapidly"), "Macro")
        self.assertEqual(_determine_guardian_from_reason("Twitter sentiment is bearish"), "Sentiment")
        self.assertEqual(_determine_guardian_from_reason("Unknown reason"), "Master")

    def test_invalidation_triggers(self):
        """Test trigger generation."""
        # Iron Condor
        triggers_ic = _build_invalidation_triggers('IRON_CONDOR', {'reason': 'Range bound'})
        self.assertIn('price_move_exceeds_2_percent', triggers_ic)
        self.assertIn('regime_shift_to_high_volatility', triggers_ic)

        # Long Straddle
        triggers_ls = _build_invalidation_triggers('LONG_STRADDLE', {'reason': 'Breakout'})
        self.assertIn('theta_burn_exceeds_hurdle', triggers_ls)

        # Bull Call Spread (Narrative)
        triggers_bcs = _build_invalidation_triggers('BULL_CALL_SPREAD', {'reason': 'Frost risk'})
        self.assertIn('rain', triggers_bcs)
        self.assertIn('warm front', triggers_bcs)

    def test_iron_condor_risk_validation(self):
        """Test risk check logic."""
        equity = 100000
        # 2% limit = 2000

        # Safe trade
        self.assertTrue(validate_iron_condor_risk(1000, equity, 0.02))

        # Risky trade
        self.assertFalse(validate_iron_condor_risk(2500, equity, 0.02))

    async def test_validate_thesis_iron_condor_regime_breach(self):
        """Test IC validation fails on regime change."""
        thesis = {
            'strategy_type': 'IRON_CONDOR',
            'entry_regime': 'RANGE_BOUND',
            'guardian_agent': 'VolatilityAnalyst',
            'supporting_data': {'entry_price': 100.0}
        }

        # Mock dependencies
        mock_ib = MagicMock()
        mock_config = {'symbol': 'KC', 'exchange': 'NYBOT'}
        mock_council = MagicMock()
        mock_position = MagicMock()

        # Mock _get_current_regime_and_iv to return HIGH_VOLATILITY with high IV rank
        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock) as mock_regime:
            mock_regime.return_value = ('HIGH_VOLATILITY', 75.0)

            result = await _validate_thesis(thesis, mock_position, mock_council, mock_config, mock_ib)

            self.assertEqual(result['action'], 'CLOSE')
            self.assertIn('REGIME BREACH', result['reason'])

    async def test_validate_thesis_long_straddle_theta_burn(self):
        """Test LS validation fails on theta burn."""
        # Entry 5 hours ago
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()

        thesis = {
            'strategy_type': 'LONG_STRADDLE',
            'entry_timestamp': entry_time,
            'supporting_data': {'entry_price': 100.0}
        }

        mock_ib = MagicMock()
        mock_position = MagicMock()
        mock_config = {}
        mock_council = MagicMock()

        # Mock _get_current_price to return 100.5 (0.5% move < 1% hurdle)
        with patch('orchestrator._get_current_price', new_callable=AsyncMock) as mock_price:
            mock_price.return_value = 100.5

            result = await _validate_thesis(thesis, mock_position, mock_council, mock_config, mock_ib)

            self.assertEqual(result['action'], 'CLOSE')
            self.assertIn('THETA BURN', result['reason'])

    async def test_validate_thesis_bull_spread_narrative(self):
        """Test narrative invalidation via Council."""
        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Frost risk',
            'invalidation_triggers': ['rain'],
            'guardian_agent': 'Agronomist'
        }

        mock_ib = MagicMock()
        mock_position = MagicMock()
        mock_config = {}
        mock_council = MagicMock()

        # Mock context fetch
        with patch('orchestrator._get_context_for_guardian', new_callable=AsyncMock) as mock_context:
            mock_context.return_value = "Forecast shows heavy rain next week."

            # Mock Router response
            mock_council.router = MagicMock()
            mock_council.router.route_and_call = AsyncMock(return_value=json.dumps({
                "verdict": "CLOSE",
                "confidence": 0.8,
                "reasoning": "Rain invalidates frost thesis."
            }))

            result = await _validate_thesis(thesis, mock_position, mock_council, mock_config, mock_ib)

            self.assertEqual(result['action'], 'CLOSE')
            self.assertIn('NARRATIVE INVALIDATION', result['reason'])

if __name__ == '__main__':
    unittest.main()
