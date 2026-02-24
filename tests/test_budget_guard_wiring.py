"""Tests for Budget Guard wiring into HeterogeneousRouter.

Validates:
- Token extraction from each client's generate() returns (str, int, int)
- Cost calculation via calculate_api_cost()
- route() records cost after successful API call
- Budget throttling blocks low-priority calls
- Budget throttling allows critical calls
- Cache hits skip budget check
- Singleton factory returns same instance
- Graceful degradation when budget guard is None
- ROLE_PRIORITY covers all AgentRole values
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from trading_bot.heterogeneous_router import (
    HeterogeneousRouter, AgentRole, ModelProvider
)
from trading_bot.budget_guard import (
    BudgetGuard, BudgetThrottledError, CallPriority,
    ROLE_PRIORITY, get_budget_guard, calculate_api_cost,
    _load_cost_config,
)


MOCK_CONFIG = {
    'gemini': {'api_key': 'mock_gemini'},
    'openai': {'api_key': 'mock_openai'},
    'anthropic': {'api_key': 'mock_anthropic'},
    'xai': {'api_key': 'mock_xai'},
    'model_registry': {
        'openai': {'pro': 'gpt-4o', 'flash': 'gpt-4o-mini'},
        'anthropic': {'pro': 'claude-3-opus'},
        'gemini': {'pro': 'gemini-1.5-pro', 'flash': 'gemini-1.5-flash'},
        'xai': {'pro': 'grok-1', 'flash': 'grok-beta'}
    },
    'cost_management': {
        'daily_budget_usd': 15.0,
        'warning_threshold_pct': 0.75,
    }
}


@pytest.fixture
def router():
    with patch.dict('os.environ', {}, clear=True):
        return HeterogeneousRouter(MOCK_CONFIG)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset budget guard singleton between tests."""
    import trading_bot.budget_guard as bg
    bg._budget_guard_instance = None
    bg._cost_config_cache = None
    yield
    bg._budget_guard_instance = None
    bg._cost_config_cache = None


# --- Cost Calculation Tests ---

def test_calculate_api_cost_known_model():
    """calculate_api_cost uses model-specific rates from api_costs.json."""
    cost = calculate_api_cost("gpt-4o", 1000, 500)
    # gpt-4o: input=0.00250/1k, output=0.01000/1k
    # (1000/1000)*0.00250 + (500/1000)*0.01000 = 0.00250 + 0.00500 = 0.00750
    assert abs(cost - 0.00750) < 0.0001


def test_calculate_api_cost_default_fallback():
    """Unknown models use default rates."""
    cost = calculate_api_cost("some-unknown-model-xyz", 1000, 1000)
    # default: input=0.001/1k, output=0.002/1k
    # (1000/1000)*0.001 + (1000/1000)*0.002 = 0.003
    assert abs(cost - 0.003) < 0.0001


def test_calculate_api_cost_zero_tokens():
    """Zero tokens return zero cost."""
    cost = calculate_api_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_calculate_api_cost_gemini_flash():
    """Gemini Flash model matched by substring."""
    cost = calculate_api_cost("gemini-3-flash-preview", 10000, 5000)
    # gemini-3-flash-preview: input=0.00010/1k, output=0.00040/1k
    # (10000/1000)*0.00010 + (5000/1000)*0.00040 = 0.001 + 0.002 = 0.003
    assert abs(cost - 0.003) < 0.0001


# --- Singleton Factory Tests ---

def test_singleton_returns_none_without_config():
    """get_budget_guard() returns None when no config provided and not initialized."""
    result = get_budget_guard()
    assert result is None


def test_singleton_creates_instance():
    """get_budget_guard(config) creates instance on first call."""
    with patch.object(BudgetGuard, '_load_state'), \
         patch.object(BudgetGuard, '_check_reset'):
        bg = get_budget_guard(MOCK_CONFIG)
        assert bg is not None
        assert isinstance(bg, BudgetGuard)


def test_singleton_returns_same_instance():
    """Two get_budget_guard() calls return same instance."""
    with patch.object(BudgetGuard, '_load_state'), \
         patch.object(BudgetGuard, '_check_reset'):
        bg1 = get_budget_guard(MOCK_CONFIG)
        bg2 = get_budget_guard()
        assert bg1 is bg2


# --- ROLE_PRIORITY Coverage Tests ---

def test_role_priority_covers_all_agent_roles():
    """Every AgentRole.value has an entry in ROLE_PRIORITY."""
    for role in AgentRole:
        assert role.value in ROLE_PRIORITY, f"Missing ROLE_PRIORITY for {role.value}"


def test_role_priority_tier_assignments():
    """Verify priority tiers match architecture."""
    assert ROLE_PRIORITY['compliance'] == CallPriority.CRITICAL
    assert ROLE_PRIORITY['master'] == CallPriority.HIGH
    assert ROLE_PRIORITY['permabear'] == CallPriority.HIGH
    assert ROLE_PRIORITY['permabull'] == CallPriority.HIGH
    assert ROLE_PRIORITY['agronomist'] == CallPriority.NORMAL
    assert ROLE_PRIORITY['weather_sentinel'] == CallPriority.LOW


# --- Route Budget Wiring Tests ---

@pytest.mark.asyncio
async def test_route_records_cost_on_success(router):
    """route() calls record_cost after successful primary API call."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = ("Analysis result", 500, 200)

    mock_budget = MagicMock(spec=BudgetGuard)
    mock_budget.check_budget.return_value = True

    with patch.object(router, '_get_client', return_value=mock_client), \
         patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        result = await router.route(AgentRole.MASTER_STRATEGIST, "test prompt")

    assert result == "Analysis result"
    mock_budget.record_cost.assert_called_once()
    call_args = mock_budget.record_cost.call_args
    assert call_args[1].get('source') or 'router/master' in str(call_args)


@pytest.mark.asyncio
async def test_route_records_cost_on_fallback(router):
    """route() calls record_cost after successful fallback API call."""
    mock_fail = AsyncMock()
    mock_fail.generate.side_effect = Exception("Primary failed")

    mock_success = AsyncMock()
    mock_success.generate.return_value = ("Fallback result", 300, 150)

    mock_budget = MagicMock(spec=BudgetGuard)
    mock_budget.check_budget.return_value = True

    def client_side_effect(provider, model):
        if provider == ModelProvider.OPENAI:
            return mock_fail
        return mock_success

    with patch.object(router, '_get_client', side_effect=client_side_effect), \
         patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        result = await router.route(AgentRole.MASTER_STRATEGIST, "test")

    assert result == "Fallback result"
    mock_budget.record_cost.assert_called_once()


@pytest.mark.asyncio
async def test_budget_throttle_blocks_low_priority(router):
    """Budget throttling raises BudgetThrottledError for low-priority calls."""
    mock_budget = MagicMock(spec=BudgetGuard)
    mock_budget.check_budget.return_value = False  # Budget depleted

    with patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        with pytest.raises(BudgetThrottledError):
            await router.route(AgentRole.AGRONOMIST, "test")


@pytest.mark.asyncio
async def test_budget_throttle_allows_critical(router):
    """Budget allows CRITICAL priority calls even when throttling others."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = ("Compliance OK", 200, 100)

    mock_budget = MagicMock(spec=BudgetGuard)
    mock_budget.check_budget.return_value = True  # CRITICAL still allowed

    with patch.object(router, '_get_client', return_value=mock_client), \
         patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        result = await router.route(AgentRole.COMPLIANCE_OFFICER, "check")

    assert result == "Compliance OK"
    mock_budget.check_budget.assert_called_once_with(CallPriority.CRITICAL)


@pytest.mark.asyncio
async def test_cache_hit_skips_budget_check(router):
    """Cache hits return without budget check."""
    # Pre-populate cache
    router.cache.set(":test prompt", AgentRole.MASTER_STRATEGIST.value, "cached result")

    mock_budget = MagicMock(spec=BudgetGuard)

    with patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        result = await router.route(AgentRole.MASTER_STRATEGIST, "test prompt")

    assert result == "cached result"
    mock_budget.check_budget.assert_not_called()
    mock_budget.record_cost.assert_not_called()


@pytest.mark.asyncio
async def test_graceful_degradation_no_budget_guard(router):
    """route() works normally when get_budget_guard() returns None."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = ("No budget tracking", 100, 50)

    with patch.object(router, '_get_client', return_value=mock_client), \
         patch('trading_bot.budget_guard.get_budget_guard', return_value=None):
        result = await router.route(AgentRole.MASTER_STRATEGIST, "test")

    assert result == "No budget tracking"


@pytest.mark.asyncio
async def test_no_cost_recorded_on_zero_tokens(router):
    """route() skips record_cost when tokens are (0, 0)."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = ("result", 0, 0)

    mock_budget = MagicMock(spec=BudgetGuard)
    mock_budget.check_budget.return_value = True

    with patch.object(router, '_get_client', return_value=mock_client), \
         patch('trading_bot.budget_guard.get_budget_guard', return_value=mock_budget):
        await router.route(AgentRole.MASTER_STRATEGIST, "test")

    mock_budget.record_cost.assert_not_called()


# --- BudgetThrottledError in agents.py ---

@pytest.mark.asyncio
async def test_budget_throttle_not_caught_by_gemini_fallback():
    """BudgetThrottledError propagates through _route_call without Gemini fallback."""
    from trading_bot.agents import TradingCouncil

    mock_router = AsyncMock()
    mock_router.route.side_effect = BudgetThrottledError("Budget exhausted")

    with patch.object(TradingCouncil, '__init__', lambda self, *a, **kw: None):
        council = TradingCouncil.__new__(TradingCouncil)
        council.use_heterogeneous = True
        council.heterogeneous_router = mock_router
        council.response_tracker = MagicMock()
        council.response_tracker.record = MagicMock()

        with pytest.raises(BudgetThrottledError):
            await council._route_call(AgentRole.AGRONOMIST, "test prompt")


# --- Per-Engine Budget Guard Isolation Tests ---

def test_contextvar_budget_guard_preferred_over_singleton(tmp_path):
    """get_budget_guard() returns per-engine instance when ContextVar is set."""
    from trading_bot.data_dir_context import EngineRuntime, _engine_runtime

    # Create two BudgetGuards with different data dirs
    kc_dir = tmp_path / "KC"
    cc_dir = tmp_path / "CC"
    kc_dir.mkdir()
    cc_dir.mkdir()

    kc_config = {**MOCK_CONFIG, 'data_dir': str(kc_dir)}
    cc_config = {**MOCK_CONFIG, 'data_dir': str(cc_dir)}

    with patch.object(BudgetGuard, '_load_state'), \
         patch.object(BudgetGuard, '_check_reset'):
        kc_budget = BudgetGuard(kc_config)
        cc_budget = BudgetGuard(cc_config)

        # Set up ContextVar with CC's runtime
        rt = EngineRuntime(ticker="CC", budget_guard=cc_budget)
        token = _engine_runtime.set(rt)
        try:
            # get_budget_guard() should return CC's budget, not singleton
            result = get_budget_guard()
            assert result is cc_budget
            assert result is not kc_budget
        finally:
            _engine_runtime.reset(token)


def test_singleton_fallback_when_no_contextvar(tmp_path):
    """get_budget_guard() falls back to singleton when no ContextVar set."""
    from trading_bot.data_dir_context import _engine_runtime

    # Ensure no ContextVar is set (reset_singleton fixture already cleared singleton)
    kc_config = {**MOCK_CONFIG, 'data_dir': str(tmp_path)}

    with patch.object(BudgetGuard, '_load_state'), \
         patch.object(BudgetGuard, '_check_reset'):
        bg = get_budget_guard(kc_config)
        assert bg is not None
        # Without ContextVar, should get the singleton
        bg2 = get_budget_guard()
        assert bg2 is bg


def test_per_engine_budget_guard_writes_to_own_dir(tmp_path):
    """Per-engine BudgetGuard saves state to its own data directory."""
    kc_dir = tmp_path / "KC"
    cc_dir = tmp_path / "CC"
    kc_dir.mkdir()
    cc_dir.mkdir()

    kc_config = {**MOCK_CONFIG, 'data_dir': str(kc_dir)}
    cc_config = {**MOCK_CONFIG, 'data_dir': str(cc_dir)}

    kc_budget = BudgetGuard(kc_config)
    cc_budget = BudgetGuard(cc_config)

    kc_budget.record_cost(1.50, source="router/master")
    cc_budget.record_cost(2.25, source="router/master")

    assert (kc_dir / "budget_state.json").exists()
    assert (cc_dir / "budget_state.json").exists()

    import json
    kc_state = json.loads((kc_dir / "budget_state.json").read_text())
    cc_state = json.loads((cc_dir / "budget_state.json").read_text())

    assert abs(kc_state['daily_spend'] - 1.50) < 0.01
    assert abs(cc_state['daily_spend'] - 2.25) < 0.01
