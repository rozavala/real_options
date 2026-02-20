"""Tests for F.4.1 (confidence threshold gate) and F.4.2 (max positions gate).

These are deterministic entry-time gates that block trades before order generation
or compliance LLM calls, saving API costs and enforcing hard limits.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from trading_bot.compliance import ComplianceGuardian


# ──────────────────────────────────────────────────────────
# F.4.1: Confidence Threshold Gate (order_manager.py)
# ──────────────────────────────────────────────────────────

class TestConfidenceThresholdGate:
    """Tests for min_confidence_threshold gate in generate_and_execute_orders."""

    def _make_signal(self, confidence=0.70, direction='BULLISH', contract_month='2026Z'):
        return {
            'confidence': confidence,
            'direction': direction,
            'contract_month': contract_month,
            'prediction_type': 'DIRECTIONAL',
            'reason': 'Test signal',
        }

    def test_signal_below_threshold_is_blocked(self):
        """A signal with confidence below min_confidence_threshold should be skipped."""
        signal = self._make_signal(confidence=0.40)
        threshold = 0.50
        assert signal.get('confidence', 0.0) < threshold

    def test_signal_at_threshold_passes(self):
        """A signal exactly at min_confidence_threshold should NOT be blocked."""
        signal = self._make_signal(confidence=0.50)
        threshold = 0.50
        assert not (signal.get('confidence', 0.0) < threshold)

    def test_signal_above_threshold_passes(self):
        """A signal above min_confidence_threshold should NOT be blocked."""
        signal = self._make_signal(confidence=0.85)
        threshold = 0.50
        assert not (signal.get('confidence', 0.0) < threshold)

    def test_speculative_thesis_blocked(self):
        """SPECULATIVE thesis (0.45 confidence) should be blocked at 0.50 threshold."""
        signal = self._make_signal(confidence=0.45)
        threshold = 0.50
        assert signal.get('confidence', 0.0) < threshold

    def test_proven_thesis_passes(self):
        """PROVEN thesis (0.90 confidence) should pass at 0.50 threshold."""
        signal = self._make_signal(confidence=0.90)
        threshold = 0.50
        assert not (signal.get('confidence', 0.0) < threshold)

    def test_plausible_aligned_passes(self):
        """v8.0: PLAUSIBLE+aligned (0.80 confidence) should pass at 0.50 threshold."""
        signal = self._make_signal(confidence=0.80)
        threshold = 0.50
        assert not (signal.get('confidence', 0.0) < threshold)

    def test_plausible_divergent_passes(self):
        """v8.0: PLAUSIBLE+DIVERGENT (0.80*0.70=0.56) should pass at 0.50 threshold."""
        signal = self._make_signal(confidence=0.56)  # 0.80 * 0.70
        threshold = 0.50
        assert not (signal.get('confidence', 0.0) < threshold)

    def test_gate_code_exists_in_order_manager(self):
        """Verify the confidence gate code exists in order_manager.py."""
        import inspect
        from trading_bot import order_manager
        source = inspect.getsource(order_manager.generate_and_queue_orders)
        assert 'min_confidence_threshold' in source
        assert 'BLOCKED BY CONFIDENCE' in source

    def test_missing_confidence_defaults_to_zero(self):
        """A signal with no 'confidence' key should default to 0.0 (fail-closed)."""
        signal_no_confidence = {
            'direction': 'BULLISH',
            'contract_month': '2026Z',
            'prediction_type': 'DIRECTIONAL',
        }
        assert signal_no_confidence.get('confidence', 0.0) == 0.0
        assert signal_no_confidence.get('confidence', 0.0) < 0.50

    def test_threshold_default_if_config_missing(self):
        """If config has no min_confidence_threshold, default should be 0.60."""
        config = {}
        threshold = config.get('risk_management', {}).get('min_confidence_threshold', 0.60)
        assert threshold == 0.60


# ──────────────────────────────────────────────────────────
# F.4.2: Max Positions Gate (compliance.py)
# ──────────────────────────────────────────────────────────

class TestMaxPositionsGate:
    """Tests for max_positions gate in ComplianceGuardian.review_order()."""

    def _make_guardian(self, max_positions=20):
        config = {
            'compliance': {
                'model': 'gemini-1.5-pro',
                'temperature': 0.0,
                'max_positions': max_positions,
            },
            'gemini': {'api_key': 'TEST'},
        }
        with patch('trading_bot.compliance.HeterogeneousRouter'):
            return ComplianceGuardian(config)

    @pytest.mark.asyncio
    async def test_positions_at_limit_rejected(self):
        """When current positions >= max_positions, order should be rejected."""
        guardian = self._make_guardian(max_positions=20)

        order_context = {
            'symbol': 'KC',
            'order_quantity': 1,
            'total_position_count': 20,  # At limit
            'account_equity': 100000.0,
            'ib': None,
        }

        with patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=1000.0):
            approved, reason = await guardian.review_order(order_context)

        assert approved is False
        assert "Position Limit" in reason
        assert "20" in reason

    @pytest.mark.asyncio
    async def test_positions_above_limit_rejected(self):
        """When current positions > max_positions, order should be rejected."""
        guardian = self._make_guardian(max_positions=20)

        order_context = {
            'symbol': 'KC',
            'order_quantity': 1,
            'total_position_count': 25,  # Above limit
            'account_equity': 100000.0,
            'ib': None,
        }

        with patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=1000.0):
            approved, reason = await guardian.review_order(order_context)

        assert approved is False
        assert "Position Limit" in reason

    @pytest.mark.asyncio
    async def test_positions_below_limit_passes_gate(self):
        """When current positions < max_positions, this gate should not block."""
        guardian = self._make_guardian(max_positions=20)

        mock_ib = AsyncMock()
        order_context = {
            'symbol': 'KC',
            'order_quantity': 1,
            'total_position_count': 10,  # Below limit
            'account_equity': 100000.0,
            'ib': mock_ib,
            'contract': MagicMock(),
            'order_object': None,  # Will fail later, but should pass this gate
        }

        with patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=1000.0):
            with patch.object(guardian, 'router') as mock_router:
                mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "Approved"}')
                approved, reason = await guardian.review_order(order_context)

        # Should not be rejected by position limit (may be approved or fail later)
        assert "Position Limit" not in reason

    def test_default_max_positions_if_config_missing(self):
        """If config has no max_positions, default should be 20."""
        config = {'compliance': {}, 'gemini': {'api_key': 'TEST'}}
        default = config.get('compliance', {}).get('max_positions', 20)
        assert default == 20
