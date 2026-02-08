import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
from trading_bot.signal_generator import generate_signals

# Fixtures for setup
@pytest.fixture
def config():
    return {
        "gemini": {
            "api_key": "fake_key",
            "personas": {},
            "agent_model": "flash-model",
            "master_model": "pro-model"
        },
        "validation_thresholds": {
            "prediction_sanity_check_pct": 0.20
        },
        "symbol": "KC",
        "exchange": "NYBOT",
        "notifications": {}
    }

@pytest.fixture
def ib_mock():
    mock = MagicMock()
    # Historical bars for the "Reality Check" in signal_generator
    mock_bar = MagicMock()
    mock_bar.close = 100.0
    mock_bar.date = MagicMock()
    mock.reqHistoricalDataAsync = AsyncMock(return_value=[mock_bar, mock_bar])
    return mock

@pytest.fixture
def mocks(ib_mock):
    # Mock Contracts
    contracts = []
    for i in range(5):
        c = MagicMock()
        c.localSymbol = f"KC H2{i+5}"
        c.lastTradeDateOrContractMonth = f"20250{i+1}"
        contracts.append(c)

    patchers = []

    # Patch get_active_futures
    p = patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock)
    mock_get_active_futures = p.start()
    mock_get_active_futures.return_value = contracts
    patchers.append(p)

    # Patch TradingCouncil
    p = patch('trading_bot.signal_generator.TradingCouncil')
    MockTradingCouncil = p.start()
    council_instance = MockTradingCouncil.return_value
    patchers.append(p)

    # Patch ComplianceGuardian
    p = patch('trading_bot.signal_generator.ComplianceGuardian')
    MockCompliance = p.start()
    compliance_instance = MockCompliance.return_value
    patchers.append(p)

    # Patch build_all_market_contexts
    p = patch('trading_bot.signal_generator.build_all_market_contexts', new_callable=AsyncMock)
    mock_build_ctx = p.start()
    mock_build_ctx.return_value = [
        {'action': 'NEUTRAL', 'confidence': 0.5, 'expected_price': 100.0,
         'reason': 'N/A', 'price': 100.0, 'regime': 'NORMAL'}
    ] * 5
    patchers.append(p)

    # Patch calculate_weighted_decision
    p = patch('trading_bot.signal_generator.calculate_weighted_decision', new_callable=AsyncMock)
    mock_vote = p.start()
    mock_vote.return_value = {
        'direction': 'BULLISH', 'confidence': 0.8,
        'weighted_score': 0.8, 'dominant_agent': 'technical'
    }
    patchers.append(p)

    # Patch detect_market_regime_simple
    p = patch('trading_bot.signal_generator.detect_market_regime_simple', return_value='NORMAL')
    p.start()
    patchers.append(p)

    # Patch StateManager
    p = patch('trading_bot.signal_generator.StateManager')
    p.start()
    patchers.append(p)

    # Patch logging/recording helpers to avoid side effects
    for target in ['log_council_decision', 'log_decision_signal', 'record_agent_prediction']:
        p = patch(f'trading_bot.signal_generator.{target}')
        p.start()
        patchers.append(p)

    # Setup council mocks
    council_instance.research_topic = AsyncMock(return_value="Report Content")
    council_instance.research_topic_with_reflexion = AsyncMock(return_value="Report Content")
    council_instance.decide = AsyncMock()
    council_instance.personas = {'master': 'You are the boss.'}
    council_instance.run_devils_advocate = AsyncMock(return_value={'proceed': True})

    # Compliance
    compliance_instance.audit_decision = AsyncMock()

    yield {
        "council": council_instance,
        "compliance": compliance_instance,
        "get_active_futures": mock_get_active_futures,
        "vote": mock_vote,
    }

    # Cleanup
    for p in patchers:
        p.stop()

@pytest.mark.asyncio
async def test_compliance_audit_rejects(config, ib_mock, mocks):
    council_instance = mocks["council"]
    compliance_instance = mocks["compliance"]

    # Setup Master Decision
    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Because I say so',
        'thesis_strength': 'PROVEN',
    }

    # Setup Audit Rejection
    compliance_instance.audit_decision.return_value = {
        'approved': False,
        'flagged_reason': 'Hallucination detected'
    }

    results = await generate_signals(ib_mock, config)

    assert len(results) == 5
    result = results[0]
    # Compliance block sets direction=NEUTRAL, confidence=0.0
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_compliance_audit_fail_closed(config, ib_mock, mocks):
    council_instance = mocks["council"]
    compliance_instance = mocks["compliance"]

    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Valid',
        'thesis_strength': 'PROVEN',
    }

    # Simulate audit failure
    compliance_instance.audit_decision.return_value = {
        'approved': False,
        'flagged_reason': 'AUDIT SYSTEM FAILURE: Timeout'
    }

    results = await generate_signals(ib_mock, config)

    result = results[0]
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_price_sanity_check_fails(config, ib_mock, mocks):
    council_instance = mocks["council"]
    compliance_instance = mocks["compliance"]

    # 30% increase (Threshold is 20%)
    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'To the moon',
        'projected_price_5_day': 130.0,
        'thesis_strength': 'PROVEN',
    }

    compliance_instance.audit_decision.return_value = {'approved': True}

    results = await generate_signals(ib_mock, config)

    result = results[0]
    # Price sanity check sets direction=NEUTRAL, confidence=0.0
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0

@pytest.mark.asyncio
async def test_both_checks_pass(config, ib_mock, mocks):
    council_instance = mocks["council"]
    compliance_instance = mocks["compliance"]

    # 10% increase (Safe)
    projected_price = 110.0

    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Solid fundamentals',
        'projected_price_5_day': projected_price,
        'thesis_strength': 'PROVEN',
    }

    compliance_instance.audit_decision.return_value = {'approved': True}

    results = await generate_signals(ib_mock, config)

    result = results[0]
    assert result['direction'] == 'BULLISH'
    assert result['confidence'] == 0.9
    assert result['expected_price'] == projected_price
