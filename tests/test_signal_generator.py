import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ib_insync import Contract, Future

from trading_bot.signal_generator import generate_signals

@pytest.mark.asyncio
async def test_generate_signals():
    ib = AsyncMock()
    config = {'symbol': 'KC', 'exchange': 'NYBOT', 'commodity': {'ticker': 'KC'}}

    # Mock market contexts
    mock_contexts = [
        {'action': 'NEUTRAL', 'confidence': 0.5, 'price': 100.0, 'sma_200': 90.0, 'expected_price': 100.0, 'predicted_return': 0.0, 'regime': 'BULL', 'reason': 'Test', 'volatility_5d': 0.02, 'price_vs_sma': 0.11, 'data_source': 'IBKR_LIVE', 'timestamp': '2026-01-28T00:00:00Z'},
        {'action': 'NEUTRAL', 'confidence': 0.5, 'price': 100.0, 'sma_200': 110.0, 'expected_price': 100.0, 'predicted_return': 0.0, 'regime': 'BEAR', 'reason': 'Test', 'volatility_5d': 0.02, 'price_vs_sma': -0.09, 'data_source': 'IBKR_LIVE', 'timestamp': '2026-01-28T00:00:00Z'},
        {'action': 'NEUTRAL', 'confidence': 0.5, 'price': 100.0, 'sma_200': 100.0, 'expected_price': 100.0, 'predicted_return': 0.0, 'regime': 'CHOP', 'reason': 'Test', 'volatility_5d': 0.02, 'price_vs_sma': 0.0, 'data_source': 'IBKR_LIVE', 'timestamp': '2026-01-28T00:00:00Z'},
        {'action': 'NEUTRAL', 'confidence': 0.5, 'price': 100.0, 'sma_200': 90.0, 'expected_price': 100.0, 'predicted_return': 0.0, 'regime': 'BULL', 'reason': 'Test', 'volatility_5d': 0.02, 'price_vs_sma': 0.11, 'data_source': 'IBKR_LIVE', 'timestamp': '2026-01-28T00:00:00Z'},
        {'action': 'NEUTRAL', 'confidence': 0.5, 'price': 100.0, 'sma_200': 110.0, 'expected_price': 100.0, 'predicted_return': 0.0, 'regime': 'BEAR', 'reason': 'Test', 'volatility_5d': 0.02, 'price_vs_sma': -0.09, 'data_source': 'IBKR_LIVE', 'timestamp': '2026-01-28T00:00:00Z'}
    ]

    mock_futures = [
        Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='20260312', localSymbol='KCH26'),
        Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='20260512', localSymbol='KCK26'),
        Future(conId=3, symbol='KC', lastTradeDateOrContractMonth='20260712', localSymbol='KCN26'),
        Future(conId=4, symbol='KC', lastTradeDateOrContractMonth='20260912', localSymbol='KCU26'),
        Future(conId=5, symbol='KC', lastTradeDateOrContractMonth='20261212', localSymbol='KCZ26'),
    ]

    with patch('trading_bot.signal_generator.get_active_futures', new_callable=AsyncMock) as mock_get_active_futures, \
         patch('trading_bot.signal_generator.build_all_market_contexts', new_callable=AsyncMock) as mock_build, \
         patch('trading_bot.signal_generator.TradingCouncil') as MockCouncil:

        mock_get_active_futures.return_value = mock_futures
        mock_build.return_value = mock_contexts

        # Mock Council behavior
        mock_council_instance = MockCouncil.return_value
        # Mock decide to return NEUTRAL
        mock_council_instance.decide = AsyncMock(return_value={
            'direction': 'NEUTRAL', 'confidence': 0.5, 'reasoning': 'Test'
        })
        mock_council_instance.run_devils_advocate = AsyncMock(return_value={'proceed': True})
        # Mock research_topic to return something valid
        mock_council_instance.research_topic = AsyncMock(return_value={'data': 'Test', 'confidence': 0.5, 'sentiment': 'NEUTRAL'})
        mock_council_instance.research_topic_with_reflexion = AsyncMock(return_value={'data': 'Test', 'confidence': 0.5, 'sentiment': 'NEUTRAL'})

        signals = await generate_signals(ib, config)

    assert len(signals) == 5
    assert signals[0]['contract_month'] == '202603'
    assert signals[1]['contract_month'] == '202605'
