"""
Tests for Article V VaR gate in trading_bot/compliance.py.

Tests 19-27: Enforcement modes, emergency bypass, staleness,
             startup grace period, gate ordering.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Reset boot time before importing compliance to ensure clean state
import trading_bot.compliance as compliance_module
compliance_module._ORCHESTRATOR_BOOT_TIME = None

from trading_bot.compliance import ComplianceGuardian, set_boot_time, _in_startup_grace_period
from trading_bot.var_calculator import VaRResult, PortfolioVaRCalculator


# --- Fixtures ---

@pytest.fixture
def base_config():
    return {
        "compliance": {
            "var_limit_pct": 0.03,
            "var_enforcement_mode": "log_only",
            "var_warning_pct": 0.02,
            "var_stale_seconds": 3600,
            "max_position_pct": 0.40,
            "max_straddle_pct": 0.55,
            "max_positions": 20,
            "max_volume_pct": 0.10,
            "max_brazil_concentration": 1.0,
            "model": "claude-sonnet-4-6",
            "temperature": 0.0,
        },
        "commodity": {"ticker": "KC"},
        "model_registry": {},
    }


@pytest.fixture
def mock_order_context():
    """Minimal order context that passes Article I and II."""
    mock_ib = AsyncMock()
    mock_contract = MagicMock()
    mock_contract.secType = "BAG"
    mock_contract.multiplier = "37500"

    mock_order = MagicMock()
    mock_order.totalQuantity = 1
    mock_order.orderType = "LMT"
    mock_order.lmtPrice = 2.0
    mock_order.action = "BUY"

    return {
        "symbol": "KC Bull Call Spread",
        "ib": mock_ib,
        "contract": mock_contract,
        "order_object": mock_order,
        "order_quantity": 1,
        "account_equity": 50000.0,
        "total_position_count": 2,
        "market_trend_pct": 0.01,
        "price": 2.0,
        "cycle_type": "SCHEDULED",
    }


@pytest.fixture(autouse=True)
def reset_boot_time():
    """Reset boot time and VaR-ready flag after each test."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = None
    compliance_module._VAR_READY = False
    yield
    compliance_module._ORCHESTRATOR_BOOT_TIME = None
    compliance_module._VAR_READY = False


def _make_cached_var(var_95_pct=0.02, computed_epoch=None):
    """Helper to create a cached VaR result."""
    return VaRResult(
        var_95=var_95_pct * 50000,
        var_99=var_95_pct * 50000 * 1.4,
        var_95_pct=var_95_pct,
        var_99_pct=var_95_pct * 1.4,
        equity=50000.0,
        position_count=3,
        commodities=["KC", "CC"],
        computed_epoch=computed_epoch or time.time(),
        timestamp="2026-02-18T10:00:00+00:00",
        last_attempt_status="OK",
    )


def _mock_compliance_through_article_i(mock_ib, mock_order_context):
    """Configure mocks so the test reaches Article V without being blocked earlier."""
    # Mock volume check to pass
    mock_ib.reqHistoricalDataAsync = AsyncMock(return_value=[
        MagicMock(volume=1000),
    ])

    # Mock spread max risk to return a small amount
    # We'll patch calculate_spread_max_risk directly
    return


# --- Test 19: log_only mode → never blocks ---

async def test_log_only_never_blocks(base_config, mock_order_context):
    """In log_only mode, VaR gate should never reject an order."""
    base_config["compliance"]["var_enforcement_mode"] = "log_only"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(var_95_pct=0.05)  # Over limit

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000), \
         patch.object(guardian, 'router') as mock_router:
        mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "OK"}')

        approved, reason = await guardian.review_order(mock_order_context)

    # Should not be blocked by VaR (LLM might still approve)
    assert approved is True or "Article V" not in reason


# --- Test 20: warn mode → logs warning, never blocks ---

async def test_warn_mode_never_blocks(base_config, mock_order_context):
    """In warn mode, VaR gate should log warning but never reject."""
    base_config["compliance"]["var_enforcement_mode"] = "warn"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(var_95_pct=0.05)  # Way over limit

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000), \
         patch.object(guardian, 'router') as mock_router:
        mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "OK"}')

        approved, reason = await guardian.review_order(mock_order_context)

    # Should not be blocked by VaR
    assert approved is True or "Article V" not in reason


# --- Test 21: enforce + VaR > limit → rejection ---

async def test_enforce_var_over_limit_rejects(base_config, mock_order_context):
    """In enforce mode, VaR over limit should reject the order."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(var_95_pct=0.04)  # Over 3% limit

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000):

        approved, reason = await guardian.review_order(mock_order_context)

    assert approved is False
    assert "Article V" in reason
    assert "Portfolio VaR" in reason


# --- Test 22: enforce + VaR < limit → approval (reaches LLM) ---

async def test_enforce_var_under_limit_passes(base_config, mock_order_context):
    """In enforce mode, VaR under limit should pass through to LLM review."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(var_95_pct=0.01)  # Under 3% limit

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000), \
         patch.object(guardian, 'router') as mock_router:
        mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "OK"}')

        approved, reason = await guardian.review_order(mock_order_context)

    assert approved is True


# --- Test 23: enforce + EMERGENCY → bypass ---

async def test_enforce_emergency_bypasses(base_config, mock_order_context):
    """In enforce mode, emergency cycles should bypass VaR blocking."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"
    mock_order_context["cycle_type"] = "EMERGENCY"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(var_95_pct=0.05)  # Way over

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000), \
         patch.object(guardian, 'router') as mock_router:
        mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "OK"}')

        approved, reason = await guardian.review_order(mock_order_context)

    # Emergency should not be blocked by VaR
    assert approved is True or "Article V" not in reason


# --- Test 24: enforce + no cached VaR → rejection (fail-closed) ---

async def test_enforce_no_cached_var_rejects(base_config, mock_order_context):
    """In enforce mode with no cached VaR, should fail-closed."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"

    guardian = ComplianceGuardian(base_config)

    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = None  # No VaR computed

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000):

        approved, reason = await guardian.review_order(mock_order_context)

    assert approved is False
    assert "Article V" in reason
    assert "fail-closed" in reason


# --- Test 25: enforce + stale + startup grace → allow ---

async def test_enforce_stale_startup_grace_allows(base_config, mock_order_context):
    """During startup grace, stale VaR should be allowed."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"

    # Set boot time to now (within grace period)
    compliance_module._ORCHESTRATOR_BOOT_TIME = time.time()

    guardian = ComplianceGuardian(base_config)

    # VaR is 3 hours old (stale beyond 2x var_stale_seconds)
    stale_epoch = time.time() - (3600 * 3)
    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(
        var_95_pct=0.01, computed_epoch=stale_epoch
    )

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000), \
         patch.object(guardian, 'router') as mock_router:
        mock_router.route = AsyncMock(return_value='{"approved": true, "reason": "OK"}')

        approved, reason = await guardian.review_order(mock_order_context)

    # Should pass because startup grace is active
    assert approved is True or "Article V" not in reason


# --- Test 26: enforce + stale + no grace → block ---

async def test_enforce_stale_no_grace_blocks(base_config, mock_order_context):
    """Without startup grace, stale VaR should block in enforce mode."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"

    # No boot time set → no grace period
    compliance_module._ORCHESTRATOR_BOOT_TIME = None

    guardian = ComplianceGuardian(base_config)

    stale_epoch = time.time() - (3600 * 3)  # 3 hours old
    mock_calc = MagicMock()
    mock_calc.get_cached_var.return_value = _make_cached_var(
        var_95_pct=0.01, computed_epoch=stale_epoch
    )

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch("trading_bot.var_calculator.get_var_calculator", return_value=mock_calc), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000):

        approved, reason = await guardian.review_order(mock_order_context)

    assert approved is False
    assert "Article V" in reason
    assert "stale" in reason.lower()


# --- Test 27: VaR gate fires AFTER Article I and max_positions ---

async def test_var_gate_ordering(base_config, mock_order_context):
    """VaR gate should fire after Article I (capital-at-risk) and max_positions."""
    base_config["compliance"]["var_enforcement_mode"] = "enforce"
    base_config["compliance"]["max_positions"] = 2  # Will be at limit

    # Set position count to limit
    mock_order_context["total_position_count"] = 2  # At max

    guardian = ComplianceGuardian(base_config)

    with patch("trading_bot.compliance.calculate_spread_max_risk", new_callable=AsyncMock, return_value=500.0), \
         patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock, return_value=10000):

        approved, reason = await guardian.review_order(mock_order_context)

    # Should be blocked by position limit, NOT by VaR
    assert approved is False
    assert "Position Limit" in reason
    assert "Article V" not in reason


# --- Helper tests ---

def test_startup_grace_not_active_by_default():
    """Without set_boot_time(), grace period is inactive."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = None
    assert _in_startup_grace_period() is False


def test_startup_grace_active_after_set():
    """After set_boot_time(), grace period is active."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = time.time()
    assert _in_startup_grace_period() is True


def test_startup_grace_extends_when_var_not_ready():
    """Grace period extends to 30 min if VaR hasn't computed."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = time.time() - 1000  # 16+ min ago
    compliance_module._VAR_READY = False
    assert _in_startup_grace_period() is True  # Extended grace


def test_startup_grace_expires_when_var_ready():
    """Grace period expires after 15 min once VaR has computed."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = time.time() - 1000  # 16+ min ago
    compliance_module._VAR_READY = True
    assert _in_startup_grace_period() is False


def test_startup_grace_hard_expires_at_30_min():
    """Grace period hard-expires at 30 min regardless of VaR status."""
    compliance_module._ORCHESTRATOR_BOOT_TIME = time.time() - 2000  # 33+ min ago
    compliance_module._VAR_READY = False
    assert _in_startup_grace_period() is False
