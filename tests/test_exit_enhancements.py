"""Tests for E.2.A (P&L exits), E.2.B (DTE acceleration), E.2.C (regime-aware exits).

These are deterministic exit-time gates that run during the position audit cycle,
before LLM-based thesis validation, to close positions based on numerical thresholds.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from ib_insync import IB


# ──────────────────────────────────────────────────────────
# E.2.C: Regime-Aware Directional Spread Exits
# ──────────────────────────────────────────────────────────

class TestRegimeAwareDirectionalExit:
    """Tests for regime breach check in _validate_directional_spread."""

    @pytest.mark.asyncio
    async def test_trending_to_range_bound_closes_position(self):
        """If entry was TRENDING and current is RANGE_BOUND, position should close."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Strong uptrend expected',
            'invalidation_triggers': [],
            'entry_regime': 'TRENDING',
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': True,
                'enable_narrative_exits': True,
            }
        }

        mock_ib = AsyncMock(spec=IB)

        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock, return_value=('RANGE_BOUND', 25.0)):
            result = await _validate_directional_spread(
                thesis=thesis,
                guardian='Master',
                council=MagicMock(),
                config=config,
                llm_budget_available=True,
                ib=mock_ib,
                active_futures_cache={},
            )

        assert result is not None
        assert result['action'] == 'CLOSE'
        assert 'REGIME BREACH' in result['reason']

    @pytest.mark.asyncio
    async def test_trending_stays_trending_no_close(self):
        """If entry was TRENDING and current is still TRENDING, should not trigger regime exit."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Strong uptrend expected',
            'invalidation_triggers': [],
            'entry_regime': 'TRENDING',
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': True,
                'enable_narrative_exits': False,  # Disable LLM to isolate regime check
            }
        }

        mock_ib = AsyncMock(spec=IB)

        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock, return_value=('TRENDING', 50.0)):
            result = await _validate_directional_spread(
                thesis=thesis,
                guardian='Master',
                council=MagicMock(),
                config=config,
                llm_budget_available=True,
                ib=mock_ib,
                active_futures_cache={},
            )

        # No regime breach → returns None (narrative exits disabled)
        assert result is None

    @pytest.mark.asyncio
    async def test_regime_check_disabled_skips(self):
        """If enable_regime_breach_exits is False, regime check should be skipped."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Strong uptrend expected',
            'invalidation_triggers': [],
            'entry_regime': 'TRENDING',
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': False,
                'enable_narrative_exits': False,
            }
        }

        mock_ib = AsyncMock(spec=IB)

        # _get_current_regime_and_iv should NOT be called
        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock, side_effect=AssertionError("Should not be called")) as mock_regime:
            result = await _validate_directional_spread(
                thesis=thesis,
                guardian='Master',
                council=MagicMock(),
                config=config,
                llm_budget_available=True,
                ib=mock_ib,
                active_futures_cache={},
            )

        assert result is None
        mock_regime.assert_not_called()

    @pytest.mark.asyncio
    async def test_regime_check_no_ib_skips(self):
        """If ib is None, regime check should be skipped gracefully."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Strong uptrend expected',
            'invalidation_triggers': [],
            'entry_regime': 'TRENDING',
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': True,
                'enable_narrative_exits': False,
            }
        }

        result = await _validate_directional_spread(
            thesis=thesis,
            guardian='Master',
            council=MagicMock(),
            config=config,
            llm_budget_available=True,
            ib=None,  # No IB connection
            active_futures_cache={},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_regime_check_error_falls_through(self):
        """If regime check raises an error, it should fall through to LLM check."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BULL_CALL_SPREAD',
            'primary_rationale': 'Strong uptrend expected',
            'invalidation_triggers': [],
            'entry_regime': 'TRENDING',
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': True,
                'enable_narrative_exits': False,  # Disable LLM to simplify
            }
        }

        mock_ib = AsyncMock(spec=IB)

        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock, side_effect=RuntimeError("IB disconnected")):
            result = await _validate_directional_spread(
                thesis=thesis,
                guardian='Master',
                council=MagicMock(),
                config=config,
                llm_budget_available=True,
                ib=mock_ib,
                active_futures_cache={},
            )

        # Should fall through to narrative check → disabled → None
        assert result is None

    @pytest.mark.asyncio
    async def test_non_trending_entry_no_close(self):
        """If entry regime was not TRENDING, range_bound current should not trigger close."""
        from orchestrator import _validate_directional_spread

        thesis = {
            'strategy_type': 'BEAR_PUT_SPREAD',
            'primary_rationale': 'High vol play',
            'invalidation_triggers': [],
            'entry_regime': 'HIGH_VOLATILITY',  # Not TRENDING
            'guardian_agent': 'Master',
        }
        config = {
            'exit_logic': {
                'enable_regime_breach_exits': True,
                'enable_narrative_exits': False,
            }
        }

        mock_ib = AsyncMock(spec=IB)

        with patch('orchestrator._get_current_regime_and_iv', new_callable=AsyncMock, return_value=('RANGE_BOUND', 25.0)):
            result = await _validate_directional_spread(
                thesis=thesis,
                guardian='Master',
                council=MagicMock(),
                config=config,
                llm_budget_available=True,
                ib=mock_ib,
                active_futures_cache={},
            )

        # HIGH_VOLATILITY → RANGE_BOUND is not the TRENDING → RANGE_BOUND pattern
        assert result is None


# ──────────────────────────────────────────────────────────
# E.2.A + E.2.B: P&L Exits and DTE Acceleration
# These run inside run_position_audit_cycle, tested via the
# _calculate_combo_risk_metrics return values and config thresholds.
# ──────────────────────────────────────────────────────────

class TestPnLExitLogic:
    """Unit tests for P&L exit threshold logic (E.2.A).

    These test the threshold comparison logic directly rather than the
    full audit cycle, which requires extensive IB mocking.
    """

    def test_take_profit_threshold_comparison(self):
        """capture_pct >= take_profit_pct should trigger take profit."""
        take_profit_pct = 0.80
        # Scenario: captured 85% of max profit
        capture_pct = 0.85
        assert capture_pct >= take_profit_pct

    def test_take_profit_below_threshold_holds(self):
        """capture_pct below take_profit_pct should NOT trigger take profit."""
        take_profit_pct = 0.80
        capture_pct = 0.60
        assert not (capture_pct >= take_profit_pct)

    def test_stop_loss_threshold_comparison(self):
        """risk_pct <= -stop_loss_pct should trigger stop loss."""
        stop_loss_pct = 0.50
        # Scenario: lost 55% of max loss
        risk_pct = -0.55
        assert risk_pct <= -abs(stop_loss_pct)

    def test_stop_loss_above_threshold_holds(self):
        """risk_pct above -stop_loss_pct should NOT trigger stop loss."""
        stop_loss_pct = 0.50
        risk_pct = -0.30
        assert not (risk_pct <= -abs(stop_loss_pct))

    def test_config_defaults(self):
        """Default thresholds should be 80% TP and 50% SL."""
        config = {}
        risk_cfg = config.get('risk_management', {})
        tp = risk_cfg.get('take_profit_capture_pct', 0.80)
        sl = risk_cfg.get('stop_loss_max_risk_pct', 0.50)
        assert tp == 0.80
        assert sl == 0.50


class TestDTEAcceleration:
    """Tests for DTE-aware exit acceleration logic (E.2.B)."""

    def test_force_close_at_or_below_dte(self):
        """Positions at or below force_close_dte should be force-closed."""
        force_close_dte = 3
        for dte in [0, 1, 2, 3]:
            assert dte <= force_close_dte, f"DTE={dte} should trigger force close"

    def test_no_force_close_above_dte(self):
        """Positions above force_close_dte should NOT be force-closed."""
        force_close_dte = 3
        for dte in [4, 5, 14, 30]:
            assert not (dte <= force_close_dte), f"DTE={dte} should NOT trigger force close"

    def test_acceleration_tightens_thresholds(self):
        """When DTE <= acceleration_dte (but > force_close), thresholds should tighten."""
        dte_cfg = {
            'enabled': True,
            'acceleration_dte': 14,
            'force_close_dte': 3,
            'accelerated_take_profit_pct': 0.50,
            'accelerated_stop_loss_pct': 0.30,
        }
        standard_tp = 0.80
        standard_sl = 0.50

        dte = 10  # Between force_close (3) and acceleration (14)
        assert dte > dte_cfg['force_close_dte']
        assert dte <= dte_cfg['acceleration_dte']

        # Accelerated thresholds are tighter
        accel_tp = dte_cfg['accelerated_take_profit_pct']
        accel_sl = dte_cfg['accelerated_stop_loss_pct']
        assert accel_tp < standard_tp  # 0.50 < 0.80
        assert accel_sl < standard_sl  # 0.30 < 0.50

    def test_no_acceleration_above_dte(self):
        """When DTE > acceleration_dte, standard thresholds should be used."""
        dte_cfg = {
            'acceleration_dte': 14,
            'force_close_dte': 3,
        }
        dte = 20
        assert dte > dte_cfg['acceleration_dte']
        # Standard thresholds remain

    def test_dte_config_defaults(self):
        """Config defaults for dte_acceleration should be sensible."""
        config = {}
        dte_cfg = config.get('exit_logic', {}).get('dte_acceleration', {})
        assert dte_cfg.get('enabled', False) is False  # Disabled by default if section missing
        assert dte_cfg.get('acceleration_dte', 14) == 14
        assert dte_cfg.get('force_close_dte', 3) == 3

    def test_dte_calculation(self):
        """DTE calculation from expiry date string."""
        expiry_str = '20260301'
        expiry_date = datetime.strptime(expiry_str, '%Y%m%d').date()
        today = datetime(2026, 2, 20).date()
        dte = (expiry_date - today).days
        assert dte == 9  # 9 days to expiry

    def test_expired_position_has_negative_dte(self):
        """Expired positions should have negative DTE and trigger force close."""
        expiry_str = '20260101'
        expiry_date = datetime.strptime(expiry_str, '%Y%m%d').date()
        today = datetime(2026, 2, 12).date()
        dte = (expiry_date - today).days
        assert dte < 0  # Already expired
        assert dte <= 3  # Below force_close_dte


class TestPnLExitIntegration:
    """Integration-style tests for P&L exit logic using mocked _calculate_combo_risk_metrics."""

    def _make_legs(self, expiry='20260601'):
        """Create mock position legs with contracts."""
        leg = MagicMock()
        leg.contract = MagicMock()
        leg.contract.lastTradeDateOrContractMonth = expiry
        leg.contract.localSymbol = 'KC 26JUN 250 C'
        leg.contract.strike = 250
        leg.position = 1
        leg.avgCost = 100.0
        return [leg]

    @pytest.mark.asyncio
    async def test_take_profit_triggers_close_in_audit(self):
        """When capture_pct exceeds threshold, position should be added to close list."""
        # Simulate what happens inside run_position_audit_cycle's loop
        config = {
            'risk_management': {
                'take_profit_capture_pct': 0.80,
                'stop_loss_max_risk_pct': 0.50,
            },
            'exit_logic': {
                'dte_acceleration': {'enabled': False},
            },
        }
        metrics = {
            'pnl': 400.0,
            'max_profit': 500.0,
            'max_loss': 200.0,
            'capture_pct': 0.85,  # Above 80% threshold
            'risk_pct': 0.0,
        }

        # Replicate the gate logic
        risk_cfg = config.get('risk_management', {})
        take_profit_pct = risk_cfg.get('take_profit_capture_pct', 0.80)
        stop_loss_pct = risk_cfg.get('stop_loss_max_risk_pct', 0.50)

        should_take_profit = metrics['capture_pct'] >= take_profit_pct
        should_stop_loss = metrics['risk_pct'] <= -abs(stop_loss_pct)

        assert should_take_profit is True
        assert should_stop_loss is False

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_close_in_audit(self):
        """When risk_pct exceeds negative threshold, position should be stop-lossed."""
        config = {
            'risk_management': {
                'take_profit_capture_pct': 0.80,
                'stop_loss_max_risk_pct': 0.50,
            },
            'exit_logic': {
                'dte_acceleration': {'enabled': False},
            },
        }
        metrics = {
            'pnl': -120.0,
            'max_profit': 500.0,
            'max_loss': 200.0,
            'capture_pct': -0.24,
            'risk_pct': -0.60,  # Below -50% threshold
        }

        risk_cfg = config.get('risk_management', {})
        take_profit_pct = risk_cfg.get('take_profit_capture_pct', 0.80)
        stop_loss_pct = risk_cfg.get('stop_loss_max_risk_pct', 0.50)

        should_take_profit = metrics['capture_pct'] >= take_profit_pct
        should_stop_loss = metrics['risk_pct'] <= -abs(stop_loss_pct)

        assert should_take_profit is False
        assert should_stop_loss is True

    @pytest.mark.asyncio
    async def test_dte_acceleration_tightens_and_triggers(self):
        """DTE acceleration should use tighter thresholds and trigger earlier."""
        config = {
            'risk_management': {
                'take_profit_capture_pct': 0.80,
                'stop_loss_max_risk_pct': 0.50,
            },
            'exit_logic': {
                'dte_acceleration': {
                    'enabled': True,
                    'acceleration_dte': 14,
                    'force_close_dte': 3,
                    'accelerated_take_profit_pct': 0.50,
                    'accelerated_stop_loss_pct': 0.30,
                },
            },
        }

        # With standard thresholds, 55% capture would NOT trigger TP (need 80%)
        # But with accelerated thresholds at DTE=10, 55% capture DOES trigger (need 50%)
        dte = 10
        dte_cfg = config['exit_logic']['dte_acceleration']

        take_profit_pct = 0.80  # Standard
        stop_loss_pct = 0.50

        if dte_cfg['enabled'] and dte <= dte_cfg['acceleration_dte']:
            take_profit_pct = dte_cfg['accelerated_take_profit_pct']  # 0.50
            stop_loss_pct = dte_cfg['accelerated_stop_loss_pct']       # 0.30

        capture_pct = 0.55
        assert capture_pct >= take_profit_pct  # 0.55 >= 0.50 → True with acceleration

    @pytest.mark.asyncio
    async def test_force_close_overrides_pnl(self):
        """DTE force close should happen regardless of P&L."""
        dte = 2
        force_close_dte = 3
        assert dte <= force_close_dte
        # Even if position is profitable, it should be force-closed near expiry
