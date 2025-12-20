
import os
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from trading_bot.agents import CoffeeCouncil

# --- Mocks for Google GenAI SDK ---

@pytest.fixture
def mock_genai_client():
    with patch("google.genai.Client") as mock_client_cls:
        # Create a mock client instance
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance

        # Mock the async methods: client.aio.models.generate_content
        mock_aio = MagicMock()
        mock_models = MagicMock()
        mock_generate_content = AsyncMock()

        mock_client_instance.aio = mock_aio
        mock_aio.models = mock_models
        mock_models.generate_content = mock_generate_content

        yield mock_client_cls, mock_client_instance, mock_generate_content

@pytest.fixture
def mock_config():
    return {
        "gemini": {
            "api_key": "TEST_KEY",
            "personas": {
                "meteorologist": "You are a weather expert.",
                "master": "You are the boss."
            }
        }
    }

@pytest.mark.asyncio
async def test_council_initialization(mock_genai_client, mock_config):
    mock_client_cls, _, _ = mock_genai_client

    council = CoffeeCouncil(mock_config)

    mock_client_cls.assert_called_with(api_key="TEST_KEY")
    assert council.personas['meteorologist'] == "You are a weather expert."
    assert council.agent_model_name == 'gemini-3.0-flash'
    assert council.master_model_name == 'gemini-2.5-pro'

@pytest.mark.asyncio
async def test_council_init_fallback(mock_genai_client):
    mock_client_cls, _, _ = mock_genai_client

    # Config with placeholder
    config = {
        "gemini": {
            "api_key": "YOUR_API_KEY_HERE"
        }
    }

    with patch.dict(os.environ, {"GEMINI_API_KEY": "ENV_KEY_123"}):
        CoffeeCouncil(config)
        mock_client_cls.assert_called_with(api_key="ENV_KEY_123")

@pytest.mark.asyncio
async def test_research_topic(mock_genai_client, mock_config):
    _, _, mock_generate_content = mock_genai_client

    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Rain is expected in Brazil."
    mock_generate_content.return_value = mock_response

    council = CoffeeCouncil(mock_config)

    result = await council.research_topic("meteorologist", "Check rain")

    assert result == "Rain is expected in Brazil."

    # Verify call arguments
    call_args = mock_generate_content.call_args
    assert call_args is not None
    kwargs = call_args.kwargs

    assert kwargs['model'] == 'gemini-3.0-flash'
    assert "You are a weather expert." in kwargs['contents']
    assert "Check rain" in kwargs['contents']

    # Verify config (Tools and Safety Settings)
    config_arg = kwargs.get('config')
    assert config_arg is not None
    assert config_arg.tools is not None
    # Check if google_search is present in tools
    # The structure depends on how the SDK constructs it, but we can check the repr or attribute
    assert str(config_arg.tools[0].google_search) is not None

    assert config_arg.safety_settings is not None
    assert len(config_arg.safety_settings) == 4

@pytest.mark.asyncio
async def test_decide_success(mock_genai_client, mock_config):
    _, _, mock_generate_content = mock_genai_client

    # Setup mock master response (JSON)
    mock_response = MagicMock()
    mock_response.text = '{"direction": "BULLISH", "confidence": 0.85, "reasoning": "Rain good.", "projected_price_5_day": 100.0}'
    mock_generate_content.return_value = mock_response

    council = CoffeeCouncil(mock_config)

    ml_signal = {"action": "LONG", "confidence": 0.6}
    reports = {"meteorologist": "Rainy"}
    market_context = "Market is up 5%"

    decision = await council.decide("KC H25", ml_signal, reports, market_context)

    assert decision['direction'] == "BULLISH"
    assert decision['confidence'] == 0.85
    assert decision['reasoning'] == "Rain good."

    # Verify call arguments
    call_args = mock_generate_content.call_args
    kwargs = call_args.kwargs

    assert kwargs['model'] == 'gemini-2.5-pro'

    # Verify JSON enforcement
    config_arg = kwargs.get('config')
    assert config_arg.response_mime_type == "application/json"

@pytest.mark.asyncio
async def test_decide_json_failure(mock_genai_client, mock_config):
    _, _, mock_generate_content = mock_genai_client

    # Setup invalid JSON response (or just a failure to parse)
    # Actually, the code calls json.loads(response.text).
    # If the model returns garbage despite mime_type, json.loads will raise.
    mock_response = MagicMock()
    mock_response.text = 'I think it is bullish because...' # Not JSON
    mock_generate_content.return_value = mock_response

    council = CoffeeCouncil(mock_config)

    ml_signal = {"action": "LONG", "confidence": 0.6, "expected_price": 100.0}
    reports = {"meteorologist": "Rainy"}
    market_context = "Market is up 5%"

    # Should not raise, but return fallback
    decision = await council.decide("KC H25", ml_signal, reports, market_context)

    assert decision['direction'] == "NEUTRAL"
    assert "Master Error" in decision['reasoning']
    # Should preserve ML signal price
    assert decision['projected_price_5_day'] == 100.0
