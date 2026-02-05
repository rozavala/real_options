import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from ib_insync import Future, Contract
from trading_bot.signal_generator import generate_signals

@pytest.mark.asyncio
async def test_neutral_direction_triggers_volatility_signal_low():
    """When direction is NEUTRAL, regime RANGE_BOUND, and vol BULLISH, emit VOLATILITY/LOW signal."""

    # Mock dependencies
    ib_mock = MagicMock()
    config = {
        'symbol': 'KC',
        'exchange': 'NYBOT',
        'notifications': {},
        'gemini': {'api_key': 'fake'},
        'commodity': {'ticker': 'KC'}
    }

    # Mock market context
    mock_contexts = [{
        'action': 'NEUTRAL',
        'confidence': 0.5,
        'price': 100.0,
        'expected_price': 100.0,
        'regime': 'RANGE_BOUND',
        'reason': 'Context says Neutral',
        'volatility_5d': 0.015,
        'price_vs_sma': 0.0
    }] * 5

    # Mock Active Futures
    future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512', localSymbol='KCZ25')

    with patch('trading_bot.signal_generator.TradingCouncil') as council_cls_mock, \
         patch('trading_bot.signal_generator.ComplianceGuardian'), \
         patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock) as get_futures_mock, \
         patch('trading_bot.signal_generator.build_all_market_contexts', new_callable=AsyncMock) as build_ctx_mock, \
         patch('trading_bot.signal_generator.calculate_weighted_decision', new_callable=AsyncMock) as vote_mock, \
         patch('trading_bot.signal_generator.StateManager'), \
         patch('trading_bot.signal_generator.detect_market_regime_simple', return_value='RANGE_BOUND'):

        get_futures_mock.return_value = [future_contract]
        build_ctx_mock.return_value = mock_contexts

        # Setup Council Mock
        council_instance = council_cls_mock.return_value

        async def research_side_effect(agent, instruction):
            if agent == 'volatility':
                return {'data': 'Vol is low', 'confidence': 0.8, 'sentiment': 'BULLISH'}
            return {'data': 'Neutral report', 'confidence': 0.5, 'sentiment': 'NEUTRAL'}

        council_instance.research_topic.side_effect = research_side_effect
        council_instance.research_topic_with_reflexion.side_effect = research_side_effect

        council_instance.decide = AsyncMock(return_value={
            'direction': 'NEUTRAL',
            'confidence': 0.6,
            'reasoning': 'Market is flat'
        })
        council_instance.run_devils_advocate = AsyncMock(return_value={'proceed': True})

        vote_mock.return_value = {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'weighted_score': 0.0,
            'dominant_agent': 'None',
            'trigger_type': 'SCHEDULED'
        }

        signals = await generate_signals(ib_mock, config)

        assert len(signals) == 1
        sig = signals[0]
        assert sig['prediction_type'] == 'VOLATILITY'
        assert sig['level'] == 'LOW'
        assert sig['contract_month'] == '202512'

@pytest.mark.asyncio
async def test_neutral_direction_triggers_volatility_signal_high():
    """When direction is NEUTRAL and agents conflict, emit VOLATILITY/HIGH signal."""

    ib_mock = MagicMock()
    config = {
        'symbol': 'KC',
        'exchange': 'NYBOT',
        'notifications': {},
        'gemini': {'api_key': 'fake'},
        'commodity': {'ticker': 'KC'}
    }

    mock_contexts = [{
        'action': 'NEUTRAL',
        'confidence': 0.5,
        'price': 100.0,
        'expected_price': 100.0,
        'regime': 'UNKNOWN',
        'reason': 'Context says Neutral',
        'volatility_5d': 0.03,
        'price_vs_sma': 0.0
    }] * 5

    future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512', localSymbol='KCZ25')

    with patch('trading_bot.signal_generator.TradingCouncil') as council_cls_mock, \
         patch('trading_bot.signal_generator.ComplianceGuardian'), \
         patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock) as get_futures_mock, \
         patch('trading_bot.signal_generator.build_all_market_contexts', new_callable=AsyncMock) as build_ctx_mock, \
         patch('trading_bot.signal_generator.calculate_weighted_decision', new_callable=AsyncMock) as vote_mock, \
         patch('trading_bot.signal_generator.StateManager'), \
         patch('trading_bot.signal_generator.detect_market_regime_simple', return_value='UNKNOWN'):

        get_futures_mock.return_value = [future_contract]
        build_ctx_mock.return_value = mock_contexts

        council_instance = council_cls_mock.return_value

        async def research_side_effect(agent, instruction):
            if agent == 'agronomist': return "[SENTIMENT: BULLISH] Rain delay."
            if agent == 'macro': return "[SENTIMENT: BEARISH] BRL tanking."
            if agent == 'technical': return "[SENTIMENT: BULLISH] Support held."
            if agent == 'geopolitical': return "[SENTIMENT: BEARISH] Ports open."
            if agent == 'sentiment': return "[SENTIMENT: BULLISH] COT Long." # Changed to increase variance
            return "[SENTIMENT: NEUTRAL] ..."

        council_instance.research_topic.side_effect = research_side_effect
        council_instance.research_topic_with_reflexion.side_effect = research_side_effect

        council_instance.decide = AsyncMock(return_value={
            'direction': 'NEUTRAL',
            'confidence': 0.4,
            'reasoning': 'High conflict'
        })
        council_instance.run_devils_advocate = AsyncMock(return_value={'proceed': True})

        vote_mock.return_value = {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'weighted_score': 0.0,
            'dominant_agent': 'None',
            'trigger_type': 'SCHEDULED'
        }

        signals = await generate_signals(ib_mock, config)

        assert len(signals) == 1
        sig = signals[0]
        assert sig['prediction_type'] == 'VOLATILITY'
        assert sig['level'] == 'HIGH'
