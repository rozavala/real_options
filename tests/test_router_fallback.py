
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole, ModelProvider

# Mock config
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
    }
}

@pytest.fixture
def router():
    with patch.dict('os.environ', {}, clear=True):
        return HeterogeneousRouter(MOCK_CONFIG)

@pytest.mark.asyncio
async def test_tier3_primary_success(router):
    """Test Master Strategist uses Primary (OpenAI Pro) successfully."""
    mock_client = AsyncMock()
    mock_client.generate.return_value = ("Primary Success", 100, 50)

    with patch.object(router, '_get_client', return_value=mock_client) as mock_get_client:
        response = await router.route(AgentRole.MASTER_STRATEGIST, "test")

        assert response == "Primary Success"
        # Verify it tried OpenAI Pro (Primary for Master)
        mock_get_client.assert_called_with(ModelProvider.OPENAI, 'gpt-4o')

@pytest.mark.asyncio
async def test_tier3_fallback_success(router):
    """Test Master Strategist falls back to Anthropic Pro if OpenAI fails."""

    # Mock _get_client to return a failing client first, then a success client
    mock_fail = AsyncMock()
    mock_fail.generate.side_effect = Exception("OpenAI Failed")

    mock_success = AsyncMock()
    mock_success.generate.return_value = ("Fallback Success", 100, 50)

    def side_effect(provider, model):
        if provider == ModelProvider.OPENAI:
            return mock_fail
        elif provider == ModelProvider.ANTHROPIC:
            return mock_success
        return AsyncMock()

    with patch.object(router, '_get_client', side_effect=side_effect):
        response = await router.route(AgentRole.MASTER_STRATEGIST, "test")
        assert response == "Fallback Success"

@pytest.mark.asyncio
async def test_tier3_safety_failure(router):
    """Test Master Strategist raises error if ALL Pro models fail (NEVER uses Flash)."""

    mock_fail = AsyncMock()
    mock_fail.generate.side_effect = Exception("Model Failed")

    with patch.object(router, '_get_client', return_value=mock_fail) as mock_get_client:
        with pytest.raises(RuntimeError) as excinfo:
            await router.route(AgentRole.MASTER_STRATEGIST, "test")

        assert "All providers exhausted" in str(excinfo.value)

        # Verify it tried the Pro models but NOT Flash
        calls = mock_get_client.call_args_list
        providers_tried = [c[0][0] for c in calls]
        models_tried = [c[0][1] for c in calls]

        # It should try OpenAI, Anthropic, Gemini (all Pro)
        assert ModelProvider.OPENAI in providers_tried
        assert ModelProvider.ANTHROPIC in providers_tried
        assert ModelProvider.GEMINI in providers_tried

        # Ensure it NEVER tried a flash model
        assert 'gpt-4o-mini' not in models_tried
        assert 'gemini-1.5-flash' not in models_tried

@pytest.mark.asyncio
async def test_tier1_sentinel_fallback(router):
    """Test Sentinel falls back to OpenAI Flash if Gemini Flash fails."""

    mock_fail = AsyncMock()
    mock_fail.generate.side_effect = Exception("Gemini Failed")

    mock_success = AsyncMock()
    mock_success.generate.return_value = ("Sentinel Success", 100, 50)

    def side_effect(provider, model):
        if provider == ModelProvider.GEMINI: # Primary for Sentinel
            return mock_fail
        elif provider == ModelProvider.OPENAI: # Fallback 1
            return mock_success
        return AsyncMock()

    with patch.object(router, '_get_client', side_effect=side_effect):
        response = await router.route(AgentRole.PRICE_SENTINEL, "test")
        assert response == "Sentinel Success"
