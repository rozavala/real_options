import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from ib_insync import Future, Contract
from trading_bot.signal_generator import generate_signals

@pytest.mark.asyncio
async def test_neutral_direction_triggers_volatility_signal_low():
    """When direction is NEUTRAL, regime RANGE_BOUND, and vol BULLISH, emit VOLATILITY/LOW signal."""

    # Mock dependencies
    ib_mock = MagicMock()
    ib_mock.reqHistoricalDataAsync = AsyncMock(return_value=[]) # Mock history
    ib_mock.reqHistoricalDataAsync = AsyncMock(return_value=[]) # Mock history
    config = {
        'symbol': 'KC',
        'exchange': 'NYBOT',
        'notifications': {},
        'gemini': {'api_key': 'fake'}
    }

    # Mock signals_list from ML
    signals_list = [{
        'action': 'NEUTRAL',
        'confidence': 0.5,
        'price': 100.0,
        'expected_price': 100.0,
        'regime': 'RANGE_BOUND', # Explicitly providing regime
        'reason': 'ML says Neutral'
    }] * 5 # Pad to length 5 to pass validation

    # Mock Active Futures
    future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512', localSymbol='KCZ25')

    with patch('trading_bot.signal_generator.CoffeeCouncil') as council_cls_mock, \
         patch('trading_bot.signal_generator.ComplianceGuardian'), \
         patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock) as get_futures_mock, \
         patch('trading_bot.signal_generator.calculate_weighted_decision', new_callable=AsyncMock) as vote_mock, \
         patch('trading_bot.signal_generator.StateManager'), \
         patch('trading_bot.signal_generator.log_model_signal'), \
         patch('trading_bot.signal_generator.detect_market_regime_simple', return_value='RANGE_BOUND'): # Ensure regime matches

        # Setup Futures Mock
        get_futures_mock.return_value = [future_contract]

        # Setup Council Mock
        council_instance = council_cls_mock.return_value
        council_instance.research_topic.return_value = "Some report"
        council_instance.research_topic_with_reflexion.return_value = "Some report"

        # We need to ensure 'volatility' agent returns BULLISH sentiment
        # research_results returns a list of results corresponding to tasks.values()
        # The tasks dict in signal_generator has keys: agronomist, macro, geopolitical, supply_chain, inventory, sentiment, technical, volatility.
        # We can mock the side_effect of research_topic to return specific dicts.

        async def research_side_effect(agent, instruction):
            if agent == 'volatility':
                return {'data': 'Vol is low', 'confidence': 0.8, 'sentiment': 'BULLISH'}
            return {'data': 'Neutral report', 'confidence': 0.5, 'sentiment': 'NEUTRAL'}

        council_instance.research_topic.side_effect = research_side_effect
        council_instance.research_topic_with_reflexion.side_effect = research_side_effect

        # Setup Decide Mock (Master)
        council_instance.decide = AsyncMock(return_value={
            'direction': 'NEUTRAL',
            'confidence': 0.6,
            'reasoning': 'Market is flat',
            'projected_price_5_day': 100.0
        })

        # Setup Weighted Vote Mock
        vote_mock.return_value = {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'weighted_score': 0.0,
            'dominant_agent': 'None',
            'trigger_type': 'SCHEDULED'
        }

        # Run
        signals = await generate_signals(ib_mock, signals_list, config)

        # Assertions
        assert len(signals) == 1
        sig = signals[0]

        # Should emit VOLATILITY signal for Iron Condor
        assert sig['prediction_type'] == 'VOLATILITY'
        assert sig['level'] == 'LOW'
        assert sig['direction'] == 'VOLATILITY'
        assert sig['contract_month'] == '202512'

@pytest.mark.asyncio
async def test_neutral_direction_triggers_volatility_signal_high():
    """When direction is NEUTRAL and agents conflict, emit VOLATILITY/HIGH signal."""

    # Mock dependencies
    ib_mock = MagicMock()
    ib_mock.reqHistoricalDataAsync = AsyncMock(return_value=[]) # Mock history
    config = {
        'symbol': 'KC',
        'exchange': 'NYBOT',
        'notifications': {},
        'gemini': {'api_key': 'fake'}
    }

    signals_list = [{
        'action': 'NEUTRAL',
        'confidence': 0.5,
        'price': 100.0,
        'expected_price': 100.0,
        'regime': 'UNKNOWN',
        'reason': 'ML says Neutral'
    }] * 5 # Pad to length 5 to pass validation

    future_contract = Future(conId=1, lastTradeDateOrContractMonth='202512', localSymbol='KCZ25')

    with patch('trading_bot.signal_generator.CoffeeCouncil') as council_cls_mock, \
         patch('trading_bot.signal_generator.ComplianceGuardian'), \
         patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock) as get_futures_mock, \
         patch('trading_bot.signal_generator.calculate_weighted_decision', new_callable=AsyncMock) as vote_mock, \
         patch('trading_bot.signal_generator.StateManager'), \
         patch('trading_bot.signal_generator.log_model_signal'), \
         patch('trading_bot.signal_generator.detect_market_regime_simple', return_value='UNKNOWN'):

        get_futures_mock.return_value = [future_contract]
        council_instance = council_cls_mock.return_value

        # Create Conflict: Agronomist BULLISH, Macro BEARISH, etc.
        # Keys: 'macro_sentiment', 'technical_sentiment', 'geopolitical_sentiment', 'sentiment_sentiment', 'agronomist_sentiment'

        # We need to control the reports returned.
        # signal_generator loops over tasks keys and awaits gather.
        # It then reconstructs reports dict.
        # Then _calculate_agent_conflict reads from 'reports' (via extraction)

        # We mock research_topic/reflexion to return specific sentiments in the text or structured dict
        # The code extracts sentiment using regex [SENTIMENT: XXX] OR uses structured dict if provided?
        # Let's check signal_generator.py:
        # agent_data[f"{key}_sentiment"] = extract_sentiment(report)
        # extract_sentiment uses regex on str(report)
        # So we should return strings with tags.

        async def research_side_effect(agent, instruction):
            if agent == 'agronomist': return "[SENTIMENT: BULLISH] Rain delay."
            if agent == 'macro': return "[SENTIMENT: BEARISH] BRL tanking."
            if agent == 'technical': return "[SENTIMENT: BULLISH] Support held."
            if agent == 'geopolitical': return "[SENTIMENT: BEARISH] Ports open."
            if agent == 'sentiment': return "[SENTIMENT: NEUTRAL] COT flat."
            return "[SENTIMENT: NEUTRAL] ..."

        council_instance.research_topic.side_effect = research_side_effect
        council_instance.research_topic_with_reflexion.side_effect = research_side_effect

        council_instance.decide = AsyncMock(return_value={
            'direction': 'NEUTRAL',
            'confidence': 0.4,
            'reasoning': 'High conflict',
            'projected_price_5_day': 100.0
        })

        vote_mock.return_value = {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'weighted_score': 0.0,
            'dominant_agent': 'None',
            'trigger_type': 'SCHEDULED'
        }

        # Run
        signals = await generate_signals(ib_mock, signals_list, config)

        # Assertions
        assert len(signals) == 1
        sig = signals[0]

        # Should emit VOLATILITY signal for Long Straddle due to conflict
        assert sig['prediction_type'] == 'VOLATILITY'
        assert sig['level'] == 'HIGH'
