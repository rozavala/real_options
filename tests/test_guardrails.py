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
    return MagicMock()

@pytest.fixture
def mocks(ib_mock):
    # Mock Contract
    contracts = []
    for i in range(5):
        c = MagicMock()
        c.localSymbol = f"KC H2{i+5}"
        c.lastTradeDateOrContractMonth = f"20250{i+1}"
        contracts.append(c)

    # Patch get_active_futures
    active_futures_patcher = patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock)
    mock_get_active_futures = active_futures_patcher.start()
    mock_get_active_futures.return_value = contracts

    # Patch CoffeeCouncil
    council_patcher = patch('trading_bot.signal_generator.CoffeeCouncil')
    MockCoffeeCouncil = council_patcher.start()
    council_instance = MockCoffeeCouncil.return_value

    # Setup common mocks
    council_instance.research_topic = AsyncMock(return_value="Report Content")
    council_instance.decide = AsyncMock()
    council_instance.audit_decision = AsyncMock()

    yield {
        "council": council_instance,
        "get_active_futures": mock_get_active_futures
    }

    # Cleanup
    active_futures_patcher.stop()
    council_patcher.stop()

@pytest.mark.asyncio
async def test_compliance_audit_rejects(config, ib_mock, mocks):
    council_instance = mocks["council"]

    # Setup ML Signal - Need 5 signals
    base_signal = {'confidence': 0.8, 'action': 'LONG', 'expected_price': 105.0, 'price': 100.0}
    signals_list = [base_signal.copy() for _ in range(5)]

    # Setup Master Decision
    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Because I say so',
        'projected_price_5_day': 105.0
    }

    # Setup Audit Rejection
    council_instance.audit_decision.return_value = {
        'approved': False,
        'flagged_reason': 'Hallucination detected'
    }

    results = await generate_signals(ib_mock, signals_list, config)

    assert len(results) == 5
    # Check the first one (they are processed in parallel/order)
    result = results[0]
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0
    assert 'BLOCKED BY AUDIT' in result['reason']

@pytest.mark.asyncio
async def test_compliance_audit_fail_closed(config, ib_mock, mocks):
    council_instance = mocks["council"]

    base_signal = {'confidence': 0.8, 'action': 'LONG', 'expected_price': 105.0, 'price': 100.0}
    signals_list = [base_signal.copy() for _ in range(5)]

    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Valid',
        'projected_price_5_day': 105.0
    }

    # Simulate the agents.py method returning False due to exception
    council_instance.audit_decision.return_value = {
        'approved': False,
        'flagged_reason': 'AUDIT SYSTEM FAILURE: Timeout'
    }

    results = await generate_signals(ib_mock, signals_list, config)

    result = results[0]
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0
    assert 'BLOCKED BY AUDIT' in result['reason']

@pytest.mark.asyncio
async def test_price_sanity_check_fails(config, ib_mock, mocks):
    council_instance = mocks["council"]

    # 30% increase (Threshold is 20%)
    current_price = 100.0
    projected_price = 130.0

    base_signal = {'confidence': 0.8, 'action': 'LONG', 'expected_price': 105.0, 'price': current_price}
    signals_list = [base_signal.copy() for _ in range(5)]

    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'To the moon',
        'projected_price_5_day': projected_price
    }

    council_instance.audit_decision.return_value = {'approved': True}

    results = await generate_signals(ib_mock, signals_list, config)

    result = results[0]
    assert result['direction'] == 'NEUTRAL'
    assert result['confidence'] == 0.0
    assert 'volatility safety limits' in result['reason']
    # Price should be reset to current
    assert result['expected_price'] == current_price

@pytest.mark.asyncio
async def test_both_checks_pass(config, ib_mock, mocks):
    council_instance = mocks["council"]

    # 10% increase (Safe)
    current_price = 100.0
    projected_price = 110.0

    base_signal = {'confidence': 0.8, 'action': 'LONG', 'expected_price': 105.0, 'price': current_price}
    signals_list = [base_signal.copy() for _ in range(5)]

    council_instance.decide.return_value = {
        'direction': 'BULLISH',
        'confidence': 0.9,
        'reasoning': 'Solid fundamentals',
        'projected_price_5_day': projected_price
    }

    council_instance.audit_decision.return_value = {'approved': True}

    results = await generate_signals(ib_mock, signals_list, config)

    result = results[0]
    assert result['direction'] == 'BULLISH'
    assert result['confidence'] == 0.9
    assert result['expected_price'] == projected_price
