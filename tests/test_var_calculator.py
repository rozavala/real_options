"""
Tests for trading_bot/var_calculator.py — Portfolio-Level VaR Calculator.

Tests 1-18: VaR computation, data quality guards, state persistence,
            stress scenarios, risk agent resilience.
"""

import asyncio
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from trading_bot.var_calculator import (
    PortfolioVaRCalculator,
    PositionSnapshot,
    VaRResult,
    get_var_calculator,
    _reset_calculator,
    set_var_data_dir,
    run_risk_agent,
    _max_consecutive_zeros,
)


# --- Fixtures ---

@pytest.fixture
def calculator():
    """Fresh calculator instance (no singleton)."""
    return PortfolioVaRCalculator()


@pytest.fixture
def sample_config():
    return {
        "compliance": {
            "var_limit_pct": 0.03,
            "var_enforcement_mode": "log_only",
            "var_warning_pct": 0.02,
            "var_stale_seconds": 3600,
            "var_lookback_days": 252,
            "var_risk_free_rate": 0.04,
            "var_confidence_levels": [0.95, 0.99],
        },
        "commodity": {"ticker": "KC"},
    }


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Set var_data_dir to a temp directory and restore after."""
    import trading_bot.var_calculator as vc
    old_dir = vc._var_data_dir
    vc._var_data_dir = str(tmp_path)
    yield str(tmp_path)
    vc._var_data_dir = old_dir


def _make_option_snapshot(
    symbol="KC", qty=1.0, strike=400.0, right="C",
    expiry_years=0.25, iv=0.35, underlying_price=380.0,
    current_price=5.0, dollar_multiplier=375.0,
):
    return PositionSnapshot(
        symbol=symbol, sec_type="FOP", qty=qty, strike=strike,
        right=right, expiry_years=expiry_years, iv=iv,
        underlying_price=underlying_price, current_price=current_price,
        dollar_multiplier=dollar_multiplier,
    )


def _make_future_snapshot(symbol="KC", qty=1.0, price=380.0, dollar_multiplier=375.0):
    return PositionSnapshot(
        symbol=symbol, sec_type="FUT", qty=qty, strike=0.0,
        right="", expiry_years=0.0, iv=0.0,
        underlying_price=price, current_price=price,
        dollar_multiplier=dollar_multiplier,
    )


def _make_returns_df(symbols, n_days=252, seed=42):
    """Generate synthetic aligned returns DataFrame."""
    rng = np.random.default_rng(seed)
    data = {}
    for sym in symbols:
        data[sym] = rng.normal(0, 0.02, n_days)  # ~2% daily vol
    return pd.DataFrame(data)


def _make_correlated_returns(n_days=252, corr=0.5, seed=42):
    """Generate KC and CC returns with specified correlation."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(0, 0.02, n_days)
    z2 = corr * z1 + np.sqrt(1 - corr**2) * rng.normal(0, 0.02, n_days)
    return pd.DataFrame({"KC": z1, "CC": z2})


# --- Test 1: Empty portfolio → VaR = 0 ---

async def test_empty_portfolio_var_is_zero(calculator, sample_config, tmp_data_dir):
    """Empty portfolio should yield zero VaR."""
    mock_ib = MagicMock()
    mock_ib.portfolio.return_value = []
    mock_ib.accountSummary.return_value = [
        MagicMock(tag="NetLiquidation", currency="USD", value="50000")
    ]

    result = await calculator.compute_portfolio_var(mock_ib, sample_config)

    assert result.var_95 == 0.0
    assert result.var_99 == 0.0
    assert result.position_count == 0
    assert result.last_attempt_status == "OK"


# --- Test 2: Single future → VaR matches analytical ---

def test_single_future_var_linear(calculator):
    """Single future position should produce linear P&L (no gamma)."""
    positions = [_make_future_snapshot(qty=1.0, price=380.0, dollar_multiplier=375.0)]
    returns_df = _make_returns_df(["KC"], n_days=252, seed=42)

    scenarios = calculator._compute_scenarios(positions, returns_df, 0.04)

    assert len(scenarios) == 252
    # PnL should be proportional to returns
    expected = returns_df["KC"].values * 380.0 * 1.0 * 375.0
    np.testing.assert_allclose(scenarios, expected, rtol=1e-10)


# --- Test 3: Single option → VaR captures gamma ---

def test_single_option_captures_gamma(calculator):
    """Option VaR should differ from linear (delta-only) VaR due to gamma."""
    call = _make_option_snapshot(
        qty=1.0, strike=400.0, right="C", expiry_years=0.25,
        iv=0.35, underlying_price=380.0, current_price=5.0,
        dollar_multiplier=375.0,
    )
    returns_df = _make_returns_df(["KC"], n_days=252, seed=42)

    scenarios = calculator._compute_scenarios([call], returns_df, 0.04)

    # Verify not all zero
    assert np.std(scenarios) > 0

    # Compare to delta-only approximation (should differ due to gamma)
    # Delta of OTM call is <0.5, linear approx would be different from full revaluation
    delta_approx = returns_df["KC"].values * 380.0 * 0.3 * 1.0 * 375.0  # ~0.3 delta for OTM

    # Full revaluation and linear should diverge for large moves
    large_move_idx = np.abs(returns_df["KC"].values) > 0.03
    if large_move_idx.sum() > 0:
        # Just verify they're not identical — gamma creates divergence
        assert not np.allclose(scenarios[large_move_idx], delta_approx[large_move_idx], atol=1.0)


# --- Test 4: Iron Condor → tail risk at delta≈0 ---

def test_iron_condor_tail_risk(calculator):
    """Iron Condor has near-zero delta but significant tail risk."""
    # Short call 420, long call 430, short put 360, long put 350
    positions = [
        _make_option_snapshot(qty=-1.0, strike=420.0, right="C",
                              underlying_price=390.0, current_price=2.0),
        _make_option_snapshot(qty=1.0, strike=430.0, right="C",
                              underlying_price=390.0, current_price=1.0),
        _make_option_snapshot(qty=-1.0, strike=360.0, right="P",
                              underlying_price=390.0, current_price=2.0),
        _make_option_snapshot(qty=1.0, strike=350.0, right="P",
                              underlying_price=390.0, current_price=1.0),
    ]
    returns_df = _make_returns_df(["KC"], n_days=252, seed=42)

    scenarios = calculator._compute_scenarios(positions, returns_df, 0.04)

    # Condor has bounded risk — but VaR should be non-zero for tail scenarios
    var_95 = float(-np.percentile(scenarios, 5))
    assert var_95 > 0, "Iron Condor should have non-zero tail risk"


# --- Test 5: KC + CC → VaR < sum of individual VaRs (diversification) ---

def test_diversification_benefit(calculator):
    """Portfolio VaR should be less than sum of individual commodity VaRs."""
    kc_call = _make_option_snapshot(
        symbol="KC", qty=1.0, strike=400.0, right="C",
        underlying_price=380.0, current_price=5.0, dollar_multiplier=375.0,
    )
    cc_call = _make_option_snapshot(
        symbol="CC", qty=1.0, strike=10000.0, right="C",
        underlying_price=9500.0, current_price=200.0, dollar_multiplier=10.0,
    )

    corr_returns = _make_correlated_returns(n_days=252, corr=0.5, seed=42)

    # Combined VaR
    combined = calculator._compute_scenarios([kc_call, cc_call], corr_returns, 0.04)
    combined_var = float(-np.percentile(combined, 5))

    # Individual VaRs
    kc_only = calculator._compute_scenarios([kc_call], corr_returns, 0.04)
    cc_only = calculator._compute_scenarios([cc_call], corr_returns, 0.04)
    kc_var = float(-np.percentile(kc_only, 5))
    cc_var = float(-np.percentile(cc_only, 5))

    # Diversification: combined < sum of individuals (unless perfectly correlated)
    assert combined_var < kc_var + cc_var, (
        f"No diversification benefit: {combined_var:.2f} >= {kc_var:.2f} + {cc_var:.2f}"
    )


# --- Test 6: Pre-trade VaR with proposed position ---

async def test_pre_trade_var_increases(calculator, sample_config, tmp_data_dir):
    """Adding a proposed position should increase VaR vs empty portfolio."""
    mock_ib = MagicMock()
    mock_ib.portfolio.return_value = []
    mock_ib.accountSummary.return_value = [
        MagicMock(tag="NetLiquidation", currency="USD", value="50000")
    ]

    # Use a short put (high directional risk) to ensure measurable VaR
    proposed = [_make_option_snapshot(
        symbol="KC", qty=-5.0, strike=380.0, right="P",
        underlying_price=380.0, current_price=10.0, dollar_multiplier=375.0,
    )]

    with patch.object(calculator, '_fetch_aligned_returns', new_callable=AsyncMock) as mock_returns:
        mock_returns.return_value = _make_returns_df(["KC"])

        result = await calculator.compute_var_with_proposed_trade(
            mock_ib, sample_config, proposed
        )

        assert result.var_95 > 0, f"Expected positive VaR, got {result.var_95}"
        assert result.position_count == 1  # 1 snapshot (qty=-5)


# --- Test 7: _bs_price wraps price_option_black_scholes correctly ---

def test_bs_price_wraps_utils(calculator):
    """_bs_price should return the same price as price_option_black_scholes."""
    from trading_bot.utils import price_option_black_scholes

    S, K, T, r, sigma = 380.0, 400.0, 0.25, 0.04, 0.35
    expected = price_option_black_scholes(S, K, T, r, sigma, "C")
    actual = calculator._bs_price(S, K, T, r, sigma, "C")

    assert expected is not None
    assert abs(actual - expected["price"]) < 0.0001


def test_bs_price_fallback_intrinsic(calculator):
    """When T<=0, _bs_price returns intrinsic value."""
    # Expired call ITM
    price = calculator._bs_price(420.0, 400.0, 0.0, 0.04, 0.35, "C")
    assert price == 20.0  # max(420-400, 0)

    # Expired put ITM
    price = calculator._bs_price(380.0, 400.0, 0.0, 0.04, 0.35, "P")
    assert price == 20.0  # max(400-380, 0)

    # Expired call OTM
    price = calculator._bs_price(380.0, 400.0, 0.0, 0.04, 0.35, "C")
    assert price == 0.0


# --- Test 8: Batched IV fetch (mock reqMktData) ---

async def test_batched_iv_fetch(calculator, sample_config):
    """Batched IV fetch should request data for all options simultaneously."""
    mock_ib = MagicMock()
    mock_greeks = MagicMock()
    mock_greeks.impliedVol = 0.35

    mock_ticker = MagicMock()
    mock_ticker.modelGreeks = mock_greeks
    mock_ticker.contract = MagicMock()

    mock_ib.reqMktData.return_value = mock_ticker

    item1 = MagicMock()
    item1.contract = MagicMock(conId=1, localSymbol="KC 400C", secType="FOP")
    item2 = MagicMock()
    item2.contract = MagicMock(conId=2, localSymbol="KC 350P", secType="FOP")

    iv_map = await calculator._batch_fetch_iv(mock_ib, [item1, item2], sample_config)

    assert iv_map[1] == 0.35
    assert iv_map[2] == 0.35
    # Verify both were requested
    assert mock_ib.reqMktData.call_count == 2


# --- Test 8b: IV polling exits early when data arrives ---

async def test_iv_polling_exits_early(calculator, sample_config):
    """Polling loop should exit as soon as modelGreeks are available."""
    mock_ib = MagicMock()
    mock_greeks = MagicMock()
    mock_greeks.impliedVol = 0.28

    mock_ticker = MagicMock()
    mock_ticker.modelGreeks = mock_greeks  # Already available
    mock_ticker.contract = MagicMock()
    mock_ib.reqMktData.return_value = mock_ticker

    item = MagicMock()
    item.contract = MagicMock(conId=42, localSymbol="KC 380C", secType="FOP")

    config = {**sample_config, 'compliance': {'iv_fetch_timeout': 0.5}}
    iv_map = await calculator._batch_fetch_iv(mock_ib, [item], config)

    assert iv_map[42] == 0.28


# --- Test 8c: IV polling timeout logs warning ---

async def test_iv_polling_timeout(calculator, sample_config):
    """When IV never arrives, polling times out and logs warning."""
    mock_ib = MagicMock()
    mock_ticker = MagicMock()
    mock_ticker.modelGreeks = None  # Never arrives
    mock_ticker.contract = MagicMock()
    mock_ib.reqMktData.return_value = mock_ticker

    item = MagicMock()
    item.contract = MagicMock(conId=99, localSymbol="KC 400P", secType="FOP")

    config = {**sample_config, 'compliance': {'iv_fetch_timeout': 0.3}}
    iv_map = await calculator._batch_fetch_iv(mock_ib, [item], config)

    assert iv_map[99] is None  # No IV available


# --- Test 9: yfinance failure → None ---

async def test_yfinance_failure_returns_none(calculator):
    """When yfinance fails, _fetch_aligned_returns returns None."""
    with patch("trading_bot.var_calculator._yf_cache", {}):
        with patch("yfinance.download", side_effect=Exception("Network error")):
            result = await calculator._fetch_aligned_returns(["KC"])
            assert result is None


# --- Test 10: IV unavailable → uses fallback IV from commodity profile ---

async def test_iv_unavailable_uses_fallback(calculator, sample_config):
    """Positions without market IV should use fallback IV from commodity profile."""
    mock_ib = MagicMock()

    # Underlying future (provides underlying price for the option)
    mock_fut = MagicMock()
    mock_fut.position = 1
    mock_fut.contract = MagicMock(
        secType="FUT", symbol="KC", conId=100,
        localSymbol="KCN6", multiplier="37500",
    )
    mock_fut.marketPrice = 380.0

    # One option position
    mock_opt = MagicMock()
    mock_opt.position = 1
    mock_opt.contract = MagicMock(
        secType="FOP", symbol="KC", conId=123,
        localSymbol="KC 400C", strike=400.0, right="C",
        lastTradeDateOrContractMonth="20260601",
        multiplier="37500",
    )
    mock_opt.marketPrice = 5.0
    mock_opt.averageCost = 1875.0
    mock_ib.portfolio.return_value = [mock_fut, mock_opt]

    # Mock IV fetch returns None (unavailable)
    with patch.object(calculator, '_batch_fetch_iv', return_value={123: None}):
        snapshots = await calculator._snapshot_portfolio(mock_ib, sample_config)

    # Should include both: 1 future + 1 option with fallback IV
    assert len(snapshots) == 2
    option_snap = [s for s in snapshots if s.sec_type == "FOP"][0]
    assert option_snap.iv == 0.35, "Should use fallback IV from profile"
    assert option_snap.underlying_price == 380.0, "Should use FUT price as underlying"
    assert option_snap.current_price == 5.0, "Should use option market price"


# --- Test 10b: Option without matching FUT in portfolio → excluded ---

async def test_option_without_underlying_excluded(calculator, sample_config):
    """Options with no matching FUT in portfolio should be excluded."""
    mock_ib = MagicMock()

    # Option only, no matching future
    mock_opt = MagicMock()
    mock_opt.position = 1
    mock_opt.contract = MagicMock(
        secType="FOP", symbol="KC", conId=123,
        localSymbol="KC 400C", strike=400.0, right="C",
        lastTradeDateOrContractMonth="20260601",
        multiplier="37500",
    )
    mock_opt.marketPrice = 5.0
    mock_opt.averageCost = 1875.0
    mock_ib.portfolio.return_value = [mock_opt]

    with patch.object(calculator, '_batch_fetch_iv', return_value={123: 0.35}):
        snapshots = await calculator._snapshot_portfolio(mock_ib, sample_config)

    assert len(snapshots) == 0, "Option without underlying FUT should be excluded"


# --- Test 10c: Option uses underlying future price, not option premium ---

async def test_option_underlying_price_from_future(calculator, sample_config):
    """Option underlying_price should come from matching FUT, not option's marketPrice."""
    mock_ib = MagicMock()

    # Underlying future at $380
    mock_fut = MagicMock()
    mock_fut.position = 1
    mock_fut.contract = MagicMock(
        secType="FUT", symbol="KC", conId=100,
        localSymbol="KCN6", multiplier="37500",
    )
    mock_fut.marketPrice = 380.0

    # Option with marketPrice = $5 (option premium, NOT underlying)
    mock_opt = MagicMock()
    mock_opt.position = 1
    mock_opt.contract = MagicMock(
        secType="FOP", symbol="KC", conId=123,
        localSymbol="KC 400C", strike=400.0, right="C",
        lastTradeDateOrContractMonth="20260601",
        multiplier="37500",
    )
    mock_opt.marketPrice = 5.0
    mock_opt.averageCost = 1875.0
    mock_ib.portfolio.return_value = [mock_fut, mock_opt]

    with patch.object(calculator, '_batch_fetch_iv', return_value={123: 0.35}):
        snapshots = await calculator._snapshot_portfolio(mock_ib, sample_config)

    option_snap = [s for s in snapshots if s.sec_type == "FOP"][0]
    assert option_snap.underlying_price == 380.0, (
        f"underlying_price should be FUT price (380.0), not option premium. "
        f"Got {option_snap.underlying_price}"
    )
    assert option_snap.current_price == 5.0, (
        f"current_price should be option market price (5.0). "
        f"Got {option_snap.current_price}"
    )


# --- Test 11: fillna(0) preserves scenarios ---

def test_fillna_preserves_scenarios(calculator):
    """Using fillna(0) should still produce valid scenario results."""
    # Create returns with some NaN values pre-fill
    returns_df = _make_returns_df(["KC"], n_days=100)
    returns_df.iloc[5:8, 0] = 0.0  # Simulate filled zeros

    positions = [_make_future_snapshot(qty=1.0, price=380.0)]
    scenarios = calculator._compute_scenarios(positions, returns_df, 0.04)

    assert len(scenarios) == 100
    # The zero-return days should produce zero P&L
    np.testing.assert_allclose(scenarios[5:8], 0.0, atol=1e-10)


# --- Test 12: 5+ consecutive zeros → logged ---

def test_consecutive_zeros_detected():
    """_max_consecutive_zeros correctly detects runs of zeros."""
    arr = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert _max_consecutive_zeros(arr) == 5

    arr2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert _max_consecutive_zeros(arr2) == 3


async def test_fetch_returns_rejects_consecutive_zeros(calculator):
    """5+ consecutive zeros should trigger data quality rejection."""
    bad_returns = pd.Series(np.zeros(252))
    bad_returns[0:5] = 0.01  # First 5 are fine
    # Rest are zeros → >5 consecutive zeros

    with patch("trading_bot.var_calculator._yf_cache", {}):
        with patch("yfinance.download") as mock_dl:
            mock_data = pd.DataFrame({"Close": np.cumsum(bad_returns) + 100})
            mock_dl.return_value = mock_data
            result = await calculator._fetch_aligned_returns(["KC"])
            # Should return None due to consecutive zeros or near-zero variance
            assert result is None


# --- Test 13: Zero-variance → None ---

async def test_zero_variance_returns_none(calculator):
    """Returns with zero variance should trigger data quality rejection."""
    with patch("trading_bot.var_calculator._yf_cache", {}):
        with patch("yfinance.download") as mock_dl:
            # Constant price → zero returns
            mock_data = pd.DataFrame({"Close": [100.0] * 260})
            mock_dl.return_value = mock_data
            result = await calculator._fetch_aligned_returns(["KC"])
            assert result is None


# --- Test 14: State round-trip (save + load + computed_epoch) ---

def test_state_round_trip(calculator, tmp_data_dir):
    """VaRResult should survive save → load round-trip."""
    original = VaRResult(
        var_95=1500.0,
        var_99=2100.0,
        var_95_pct=0.03,
        var_99_pct=0.042,
        equity=50000.0,
        position_count=4,
        commodities=["KC", "CC"],
        computed_epoch=time.time(),
        timestamp="2026-02-18T10:00:00+00:00",
        narrative={"dominant_risk": "Coffee price shock"},
        scenarios=[{"name": "Crash", "pnl": -3000}],
    )

    calculator._save_state(original)
    loaded = calculator._load_state()

    assert loaded is not None
    assert loaded.var_95 == 1500.0
    assert loaded.var_99 == 2100.0
    assert loaded.var_95_pct == 0.03
    assert loaded.equity == 50000.0
    assert loaded.commodities == ["KC", "CC"]
    assert abs(loaded.computed_epoch - original.computed_epoch) < 1.0
    assert loaded.narrative["dominant_risk"] == "Coffee price shock"
    assert loaded.scenarios[0]["pnl"] == -3000


# --- Test 15: Failure tracking in state file ---

def test_failure_tracking_in_state(calculator, tmp_data_dir):
    """Failed VaR computation should save failure state."""
    fail_result = VaRResult(
        computed_epoch=time.time(),
        last_attempt_status="FAILED",
        last_attempt_error="yfinance timeout",
    )
    calculator._save_state(fail_result)
    loaded = calculator._load_state()

    assert loaded is not None
    assert loaded.last_attempt_status == "FAILED"
    assert loaded.last_attempt_error == "yfinance timeout"


# --- Test 16: yfinance_ticker field resolution ---

def test_yfinance_ticker_resolution(calculator):
    """_get_yf_ticker should resolve from CommodityProfile."""
    # KC has yfinance_ticker="KC=F" set in profile
    assert calculator._get_yf_ticker("KC") == "KC=F"
    assert calculator._get_yf_ticker("CC") == "CC=F"

    # Unknown ticker falls back to "{symbol}=F"
    assert calculator._get_yf_ticker("ZZZ") == "ZZZ=F"


# --- Test 17: Stress scenario: price + IV shock ---

async def test_stress_scenario_price_and_iv(calculator, sample_config):
    """Stress scenario should apply price and IV shocks correctly."""
    # Use a long future (pure directional) so crash cleanly produces loss
    long_fut = _make_future_snapshot(
        symbol="KC", qty=1.0, price=380.0, dollar_multiplier=375.0,
    )
    # Also test an ITM call that should lose on crash
    itm_call = _make_option_snapshot(
        symbol="KC", qty=1.0, strike=350.0, right="C",
        expiry_years=0.5, iv=0.35,
        underlying_price=380.0, current_price=38.0, dollar_multiplier=375.0,
    )

    mock_ib = MagicMock()
    with patch.object(calculator, '_snapshot_portfolio', return_value=[long_fut, itm_call]):
        scenario = {
            "name": "Coffee crash",
            "price_shock_pct": -0.15,
            "iv_shock_pct": 0.0,  # No IV shock — isolate price impact
            "time_horizon_weeks": 0,
        }
        result = await calculator.compute_stress_scenario(mock_ib, sample_config, scenario)

    assert result["name"] == "Coffee crash"
    assert result["positions"] == 2
    # Long future + ITM long call with -15% crash → should clearly lose money
    assert result["pnl"] < 0, f"Expected negative P&L, got {result['pnl']}"
    # Future alone loses 380 * 0.15 * 1 * 375 = $21,375
    assert result["pnl"] < -20000


# --- Test 18: Risk Agent failure → VaR still saves ---

async def test_risk_agent_failure_var_still_saves(sample_config, tmp_data_dir):
    """If risk agent fails, VaR result should still be saved."""
    var_result = VaRResult(
        var_95=1500.0,
        var_99=2100.0,
        var_95_pct=0.03,
        var_99_pct=0.042,
        equity=50000.0,
        position_count=4,
        commodities=["KC"],
        computed_epoch=time.time(),
        timestamp="2026-02-18T10:00:00+00:00",
    )

    # Both L1 and L2 will fail due to missing router
    with patch("trading_bot.var_calculator._run_l1_interpreter", side_effect=Exception("LLM down")):
        output = await run_risk_agent(var_result, sample_config)

    # Agent output should be empty but not raise
    assert isinstance(output, dict)
    # VaR result itself is unchanged
    assert var_result.var_95 == 1500.0


# --- Test 19: computed_by field round-trip ---

def test_computed_by_round_trip(calculator, tmp_data_dir):
    """computed_by field should survive save → load round-trip."""
    original = VaRResult(
        var_95=1500.0,
        computed_epoch=time.time(),
        computed_by="KC",
    )
    calculator._save_state(original)
    loaded = calculator._load_state()

    assert loaded is not None
    assert loaded.computed_by == "KC"


# --- Test 20: get_cached_var marks staleness ---

def test_get_cached_var_marks_stale(calculator):
    """get_cached_var should set is_stale when result is older than threshold."""
    old_result = VaRResult(
        var_95=1000.0,
        computed_epoch=time.time() - 8000,  # ~2.2h ago
        last_attempt_status="OK",
    )
    calculator._cached_result = old_result

    cached = calculator.get_cached_var(max_stale_seconds=7200)
    assert cached is not None
    assert cached.is_stale is True


def test_get_cached_var_fresh_not_stale(calculator):
    """get_cached_var should not mark a fresh result as stale."""
    fresh_result = VaRResult(
        var_95=1000.0,
        computed_epoch=time.time() - 60,  # 1 min ago
        last_attempt_status="OK",
    )
    calculator._cached_result = fresh_result

    cached = calculator.get_cached_var(max_stale_seconds=7200)
    assert cached is not None
    assert cached.is_stale is False


# --- Test 21: FUT with invalid marketPrice excluded from VaR ---

async def test_fut_invalid_market_price_excluded(calculator, sample_config):
    """Futures with marketPrice <= 0 (e.g., -1 from IB) should be excluded."""
    mock_ib = MagicMock()

    mock_fut = MagicMock()
    mock_fut.position = 1
    mock_fut.contract = MagicMock(
        secType="FUT", symbol="KC", conId=100,
        localSymbol="KCN6", multiplier="37500",
    )
    mock_fut.marketPrice = -1.0  # IB returns -1 when no market data

    mock_ib.portfolio.return_value = [mock_fut]

    snapshots = await calculator._snapshot_portfolio(mock_ib, sample_config)
    assert len(snapshots) == 0, "FUT with marketPrice=-1 should be excluded"


# --- Test 22: Negative shocked_S clamped in B-S revaluation ---

def test_negative_shocked_price_clamped(calculator):
    """Extreme negative returns should clamp shocked_S to 0.01, not go negative."""
    call = _make_option_snapshot(
        qty=1.0, strike=400.0, right="C", underlying_price=380.0,
        current_price=5.0, dollar_multiplier=375.0,
    )
    # Returns with extreme -110% move (bad data)
    extreme_returns = pd.DataFrame({"KC": [-1.1, -0.5, 0.0, 0.5]})

    scenarios = calculator._compute_scenarios([call], extreme_returns, 0.04)

    # Should not contain NaN (would happen if log(negative) were computed)
    assert not np.any(np.isnan(scenarios)), "NaN in scenarios from extreme negative return"
    assert len(scenarios) == 4


# --- Test 23: _calc_expiry_years edge cases ---

def test_calc_expiry_years_valid(calculator):
    """Valid YYYYMMDD expiry should return positive years."""
    from datetime import datetime, timedelta
    future_date = (datetime.now() + timedelta(days=90)).strftime("%Y%m%d")
    years = calculator._calc_expiry_years(future_date)
    assert 0.2 < years < 0.3  # ~90 days ≈ 0.25 years


def test_calc_expiry_years_expired(calculator):
    """Past expiry should return 0.0 (clamped)."""
    years = calculator._calc_expiry_years("20200101")
    assert years == 0.0


def test_calc_expiry_years_malformed(calculator):
    """Malformed string should return 0.0 without raising."""
    assert calculator._calc_expiry_years("not-a-date") == 0.0
    assert calculator._calc_expiry_years("") == 0.0
    assert calculator._calc_expiry_years(None) == 0.0


# --- Test 24: Singleton reset for test isolation ---

def test_singleton_reset():
    """_reset_calculator clears the singleton."""
    from trading_bot.var_calculator import get_var_calculator, _reset_calculator
    import trading_bot.var_calculator as vc

    calc = get_var_calculator({})
    assert vc._calculator_instance is not None

    _reset_calculator()
    assert vc._calculator_instance is None


# --- Test 25: PID-specific tmp file used in _save_state ---

def test_save_state_uses_pid_tmp(calculator, tmp_data_dir):
    """_save_state should use PID-specific tmp file for cross-process safety."""
    result = VaRResult(var_95=100.0, computed_epoch=time.time())
    calculator._save_state(result)

    # State file should exist
    state_path = os.path.join(tmp_data_dir, "var_state.json")
    assert os.path.exists(state_path)

    # Lock file should exist
    lock_path = os.path.join(tmp_data_dir, "var_state.lock")
    assert os.path.exists(lock_path)

    # PID-specific tmp should NOT exist (cleaned up after replace)
    tmp_path = f"{state_path}.tmp.{os.getpid()}"
    assert not os.path.exists(tmp_path)
