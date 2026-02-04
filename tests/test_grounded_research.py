import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from trading_bot.agents import CoffeeCouncil, GroundedDataPacket
from trading_bot.state_manager import StateManager
from trading_bot.heterogeneous_router import AgentRole

@pytest.fixture
def mock_config():
    return {
        'gemini': {
            'api_key': 'test_key',
            'agent_model': 'gemini-test',
            'master_model': 'gemini-master'
        },
        'model_registry': {}
    }

@pytest.mark.asyncio
async def test_grounded_data_packet_formatting():
    """Test that GroundedDataPacket formats correctly for prompts."""
    packet = GroundedDataPacket(
        search_query="Brazil coffee frost risk",
        raw_findings="Cold front approaching Minas Gerais...",
        extracted_facts=[
            {"date": "2026-01-14", "fact": "Temperatures to drop to 3°C", "source": "Somar Meteorologia"}
        ]
    )

    context = packet.to_context_block()

    assert "GROUNDED DATA PACKET" in context
    assert "2026-01-14" in context
    assert "3°C" in context
    assert "Somar Meteorologia" in context
    assert "Last 4 hours" in context

@pytest.mark.asyncio
async def test_gather_grounded_data_caching(mock_config):
    """Test 10-minute caching logic."""
    with patch('google.genai.Client'), \
         patch('trading_bot.semantic_router.SemanticRouter'), \
         patch('trading_bot.heterogeneous_router.HeterogeneousRouter'):

        council = CoffeeCouncil(mock_config)
        council.tms = MagicMock()

        # Mock API call
        mock_response = """
        {
            "raw_summary": "Test Summary",
            "dated_facts": [],
            "search_queries_used": [],
            "data_freshness": "today"
        }
        """
        council._call_model = AsyncMock(return_value=mock_response)

        # First call
        await council._gather_grounded_data("query1", "test_persona")
        assert council._call_model.call_count == 1

        # Second call (immediate) should cache
        await council._gather_grounded_data("query1", "test_persona")
        assert council._call_model.call_count == 1

        # Different query should call again
        await council._gather_grounded_data("query2", "test_persona")
        assert council._call_model.call_count == 2

@pytest.mark.asyncio
async def test_fallback_logic(mock_config):
    """Test fallback to Sentinel History when API fails."""
    with patch('google.genai.Client'), \
         patch('trading_bot.semantic_router.SemanticRouter'), \
         patch('trading_bot.heterogeneous_router.HeterogeneousRouter'), \
         patch('trading_bot.state_manager.StateManager.load_state') as mock_load_state:

        council = CoffeeCouncil(mock_config)

        # Mock API failure
        council._call_model = AsyncMock(side_effect=Exception("API Error"))

        # Mock StateManager return
        mock_load_state.return_value = {
            "events": [
                {"source": "WeatherSentinel", "reason": "Frost"}
            ]
        }

        packet = await council._gather_grounded_data("query_fail", "test_persona")

        assert "SEARCH FAILED" in packet.raw_findings
        assert "Frost" in packet.raw_findings
        assert packet.data_freshness_hours == 999

@pytest.mark.asyncio
async def test_research_topic_flow(mock_config):
    """Test the two-phase flow: Gather -> Analyze."""
    with patch('google.genai.Client'), \
         patch('trading_bot.semantic_router.SemanticRouter'), \
         patch('trading_bot.heterogeneous_router.HeterogeneousRouter'):

        council = CoffeeCouncil(mock_config)
        council.use_heterogeneous = True
        council.tms = MagicMock()

        # Mock Phase 1 (Data Gathering)
        mock_packet = GroundedDataPacket(search_query="q", raw_findings="data")
        council._gather_grounded_data = AsyncMock(return_value=mock_packet)

        # Mock Phase 2 (Routing)
        # Fix: research_topic calls heterogeneous_router.route directly, not _route_call
        council.heterogeneous_router.route = AsyncMock(return_value="Analysis Result")

        result = await council.research_topic("macro", "Analyze rates")

        # Check Phase 1 called
        council._gather_grounded_data.assert_called_once()

        # Check Phase 2 called with injected context
        council.heterogeneous_router.route.assert_called_once()
        call_args = council.heterogeneous_router.route.call_args
        prompt = call_args[0][1] # (role, prompt)

        assert "GROUNDED DATA PACKET" in prompt
        assert "Base your analysis ONLY on the data provided" in prompt
