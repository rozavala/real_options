
import os
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from trading_bot.agents import CoffeeCouncil

# --- Mocks for Google Generative AI ---

@pytest.fixture
def mock_genai():
    with patch("google.generativeai.GenerativeModel") as mock_model_cls, \
         patch("google.generativeai.configure") as mock_configure:

        # Setup mock models
        mock_agent_instance = AsyncMock()
        mock_master_instance = AsyncMock()

        # Configure the class to return different instances based on args (simplified)
        # We'll just set side_effect to return agent then master, or just inspect calls
        # A simpler way is to patch the instances on the CoffeeCouncil object after creation,
        # or mock the constructor returns.

        # Let's mock the return of the constructor
        mock_model_cls.return_value = mock_agent_instance # Default return

        yield mock_configure, mock_model_cls, mock_agent_instance

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
async def test_council_initialization(mock_genai, mock_config):
    mock_configure, mock_model_cls, _ = mock_genai

    council = CoffeeCouncil(mock_config)

    mock_configure.assert_called_with(api_key="TEST_KEY")
    assert council.personas['meteorologist'] == "You are a weather expert."
    assert mock_model_cls.call_count == 2 # 1 for agent, 1 for master

@pytest.mark.asyncio
async def test_research_topic(mock_genai, mock_config):
    _, _, mock_agent_instance = mock_genai

    # Setup mock response
    mock_response = MagicMock()
    mock_response.text = "Rain is expected in Brazil."
    mock_agent_instance.generate_content_async.return_value = mock_response

    council = CoffeeCouncil(mock_config)
    # Manually set the agent model to our mock to be sure (since we reused the constructor return)
    council.agent_model = mock_agent_instance

    result = await council.research_topic("meteorologist", "Check rain")

    assert result == "Rain is expected in Brazil."
    assert "You are a weather expert." in str(mock_agent_instance.generate_content_async.call_args)
    assert "Check rain" in str(mock_agent_instance.generate_content_async.call_args)

@pytest.mark.asyncio
async def test_decide_success(mock_genai, mock_config):
    _, _, mock_instance = mock_genai

    # Setup mock master response (JSON)
    mock_response = MagicMock()
    mock_response.text = '{"direction": "BULLISH", "confidence": 0.85, "reasoning": "Rain good.", "projected_price_5_day": 100.0}'
    mock_instance.generate_content_async.return_value = mock_response

    council = CoffeeCouncil(mock_config)
    council.master_model = mock_instance # Force master model to be this mock

    ml_signal = {"action": "LONG", "confidence": 0.6}
    reports = {"meteorologist": "Rainy"}

    decision = await council.decide("KC H25", ml_signal, reports)

    assert decision['direction'] == "BULLISH"
    assert decision['confidence'] == 0.85
    assert decision['reasoning'] == "Rain good."

@pytest.mark.asyncio
async def test_decide_json_failure(mock_genai, mock_config):
    _, _, mock_instance = mock_genai

    # Setup invalid JSON response
    mock_response = MagicMock()
    mock_response.text = 'I think it is bullish because...' # Not JSON
    mock_instance.generate_content_async.return_value = mock_response

    council = CoffeeCouncil(mock_config)
    council.master_model = mock_instance

    ml_signal = {"action": "LONG", "confidence": 0.6, "expected_price": 100.0}
    reports = {"meteorologist": "Rainy"}

    # Should not raise, but return fallback
    decision = await council.decide("KC H25", ml_signal, reports)

    assert decision['direction'] == "NEUTRAL"
    assert "JSON Error" in decision['reasoning']
    # Should preserve ML signal price
    assert decision['projected_price_5_day'] == 100.0
