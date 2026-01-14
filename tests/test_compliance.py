import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.compliance import ComplianceOfficer

@pytest.fixture
def mock_config():
    return {
        'compliance': {'model': 'gemini-1.5-pro', 'temperature': 0.0},
        'gemini': {'api_key': 'TEST'}
    }

@pytest.mark.asyncio
async def test_compliance_position_limit(mock_config):
    with patch('google.genai.Client'):
        officer = ComplianceOfficer(mock_config)

        decision = {'direction': 'BULLISH', 'reasoning': 'Test'}
        # Simulate 5 existing positions
        context = {'total_position_count': 5, 'bid_ask_spread': 0.01}

        result = await officer.review(decision, context)
        assert "VETOED: Position Limit Exceeded" in result

@pytest.mark.asyncio
async def test_compliance_liquidity_check(mock_config):
    with patch('google.genai.Client'):
        officer = ComplianceOfficer(mock_config)

        decision = {'direction': 'BULLISH', 'reasoning': 'Test'}
        # Spread > 5 ticks (0.25). Let's say 0.30
        context = {'total_position_count': 0, 'bid_ask_spread': 0.30}

        result = await officer.review(decision, context)
        assert "VETOED: Liquidity Gap" in result

@pytest.mark.asyncio
async def test_compliance_sanity_check_veto(mock_config):
    with patch('google.genai.Client') as MockClient:
        # Mock LLM to Veto
        mock_instance = MockClient.return_value
        mock_instance.aio.models.generate_content = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = '{"approved": false, "flagged_reason": "Fighting the trend"}'
        mock_instance.aio.models.generate_content.return_value = mock_response

        officer = ComplianceOfficer(mock_config)

        decision = {'direction': 'BULLISH', 'reasoning': 'Buying the dip'}
        context = {'total_position_count': 0, 'bid_ask_spread': 0.05, 'market_trend_pct': -0.02} # -2% trend

        result = await officer.review(decision, context)
        assert "VETOED: Sanity Check Failed" in result

@pytest.mark.asyncio
async def test_compliance_approval(mock_config):
    with patch('google.genai.Client') as MockClient:
        # Mock LLM to Approve
        mock_instance = MockClient.return_value
        mock_instance.aio.models.generate_content = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = '{"approved": true, "flagged_reason": ""}'
        mock_instance.aio.models.generate_content.return_value = mock_response

        officer = ComplianceOfficer(mock_config)

        decision = {'direction': 'BULLISH', 'reasoning': 'Solid setup'}
        context = {'total_position_count': 0, 'bid_ask_spread': 0.05, 'market_trend_pct': 0.01}

        result = await officer.review(decision, context)
        assert result == "APPROVED"
