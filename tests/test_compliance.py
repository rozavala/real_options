import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.compliance import ComplianceGuardian, AgentRole

@pytest.fixture
def mock_config():
    return {
        'compliance': {'model': 'gemini-1.5-pro', 'temperature': 0.0},
        'gemini': {'api_key': 'TEST'}
    }

@pytest.mark.asyncio
async def test_compliance_veto(mock_config):
    # Mock router
    with patch('trading_bot.compliance.HeterogeneousRouter') as MockRouter:
        mock_router_instance = MockRouter.return_value
        mock_router_instance.route = AsyncMock()
        mock_router_instance.route.return_value = '{"approved": false, "reason": "Vetoed: Position Limit"}'

        guardian = ComplianceGuardian(mock_config)

        # Mock fetch volume stats to return a valid volume
        with patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 1000.0

            context = {'symbol': 'KC', 'order_quantity': 1}
            approved, reason = await guardian.review_order(context)

            assert approved is False
            assert "Vetoed: Position Limit" in reason

@pytest.mark.asyncio
async def test_compliance_approval(mock_config):
    with patch('trading_bot.compliance.HeterogeneousRouter') as MockRouter:
        mock_router_instance = MockRouter.return_value
        mock_router_instance.route = AsyncMock()
        mock_router_instance.route.return_value = '{"approved": true, "reason": "Approved"}'

        guardian = ComplianceGuardian(mock_config)

        # Mock fetch volume stats to return a valid volume
        with patch.object(guardian, '_fetch_volume_stats', new_callable=AsyncMock) as mock_vol:
            mock_vol.return_value = 1000.0

            context = {'symbol': 'KC', 'order_quantity': 1}

            approved, reason = await guardian.review_order(context)
            assert approved is True
            assert "Approved" in reason

@pytest.mark.asyncio
async def test_audit_decision(mock_config):
    with patch('trading_bot.compliance.HeterogeneousRouter') as MockRouter:
        mock_router_instance = MockRouter.return_value
        mock_router_instance.route = AsyncMock()
        mock_router_instance.route.return_value = '{"approved": false, "flagged_reason": "Hallucination"}'

        guardian = ComplianceGuardian(mock_config)

        reports = {'agent1': 'report'}
        market_context = 'context'
        decision = {'direction': 'BULLISH', 'reasoning': 'Because'}

        result = await guardian.audit_decision(reports, market_context, decision, "")
        assert result['approved'] is False
        assert "Hallucination" in result['flagged_reason']
