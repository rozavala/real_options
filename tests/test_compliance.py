import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from trading_bot.compliance import ComplianceGuardian, ComplianceDecision, AgentRole

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


@pytest.mark.asyncio
async def test_audit_decision_with_debate_summary(mock_config):
    """v8.1: Verify debate_summary param is accepted and included in prompt."""
    with patch('trading_bot.compliance.HeterogeneousRouter') as MockRouter:
        mock_router_instance = MockRouter.return_value
        mock_router_instance.route = AsyncMock()
        mock_router_instance.route.return_value = '{"approved": true, "flagged_reason": ""}'

        guardian = ComplianceGuardian(mock_config)

        reports = {'agent1': 'report'}
        market_context = 'price data'
        decision = {'direction': 'BULLISH', 'reasoning': 'Permabear noted risk but Bull defense was stronger'}
        debate_summary = "BEAR ATTACK:\n{\"position\": \"BEARISH\"}\n\nBULL DEFENSE:\n{\"position\": \"BULLISH\"}"

        result = await guardian.audit_decision(
            reports, market_context, decision, "",
            debate_summary=debate_summary
        )
        assert result['approved'] is True

        # Verify debate summary was included in the prompt sent to LLM
        call_args = mock_router_instance.route.call_args
        prompt_sent = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('prompt', '')
        assert "BEAR ATTACK" in prompt_sent
        assert "BULL DEFENSE" in prompt_sent
        assert "Source 3: ADVERSARIAL DEBATE" in prompt_sent


# --- ComplianceDecision parser tests ---

class TestComplianceDecisionParser:
    """Tests for ComplianceDecision.from_llm_response multi-layer parser."""

    def test_clean_json_uses_layer1(self):
        """Clean JSON should be parsed by direct_json (Layer 1)."""
        resp = '{"approved": true, "reason": "All clear"}'
        d = ComplianceDecision.from_llm_response(resp)
        assert d.approved is True
        assert d.reason == "All clear"
        assert d.parse_method == "direct_json"

    def test_concatenated_json_uses_layer1_5(self):
        """Two concatenated JSON objects should be handled by Layer 1.5."""
        resp = '{"approved": false, "reason": "Risk too high"}{"extra": "data"}'
        d = ComplianceDecision.from_llm_response(resp)
        assert d.approved is False
        assert d.reason == "Risk too high"
        assert d.parse_method == "json_first_object"

    def test_json_with_trailing_prose(self):
        """JSON followed by prose explanation should use Layer 1.5."""
        resp = '{"approved": true, "reason": "Within limits"}\n\nThe order looks safe because...'
        d = ComplianceDecision.from_llm_response(resp)
        assert d.approved is True
        assert d.reason == "Within limits"
        assert d.parse_method == "json_first_object"

    def test_escaped_quotes_in_reason(self):
        """Escaped quotes inside reason field should parse correctly."""
        resp = r'{"approved": false, "reason": "Position \"KC\" exceeds limit"}'
        d = ComplianceDecision.from_llm_response(resp)
        assert d.approved is False
        assert "KC" in d.reason
        assert d.parse_method == "direct_json"

    def test_non_bool_approved_fails_closed(self):
        """Non-boolean approved value should fail closed."""
        resp = '{"approved": "yes", "reason": "Looks good"}'
        d = ComplianceDecision.from_llm_response(resp)
        assert d.approved is False

    def test_empty_response_fails_closed(self):
        """Empty response should fail closed."""
        d = ComplianceDecision.from_llm_response("")
        assert d.approved is False
        assert d.parse_method == "empty_response"

    def test_unparseable_fails_closed(self):
        """Completely unparseable text should fail closed."""
        d = ComplianceDecision.from_llm_response("I think this order is fine.")
        assert d.approved is False
        assert d.parse_method == "fail_closed"
