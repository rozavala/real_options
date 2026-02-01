"""Tests for the commodity-agnostic Market Data Provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from trading_bot.market_data_provider import (
    build_market_context,
    build_all_market_contexts,
    format_market_context_for_prompt,
    _empty_market_context,
)


@pytest.fixture
def mock_ib():
    ib = AsyncMock()
    # reqMktData is synchronous in ib_insync
    ib.reqMktData = MagicMock()
    ib.cancelMktData = MagicMock()
    return ib


@pytest.fixture
def mock_contract():
    from ib_insync import Future
    return Future(
        conId=1,
        symbol='KC',
        lastTradeDateOrContractMonth='20260512',
        localSymbol='KCK6'
    )


@pytest.fixture
def mock_config():
    return {
        'commodity': {'ticker': 'KC'},
        'symbol': 'KC',
        'exchange': 'NYBOT',
    }


class TestBuildMarketContext:

    @pytest.mark.asyncio
    async def test_returns_required_keys(self, mock_ib, mock_contract, mock_config):
        """Market context must contain all keys the Council expects."""
        # Mock price ticker
        mock_ticker = MagicMock()
        mock_ticker.last = 335.0
        mock_ticker.close = 335.0
        mock_ib.reqMktData.return_value = mock_ticker

        # Mock historical bars
        mock_bars = [MagicMock(close=330 + i) for i in range(210)]
        mock_ib.reqHistoricalDataAsync.return_value = mock_bars

        with patch('trading_bot.market_data_provider.RegimeDetector') as mock_regime:
            mock_regime.detect_regime = AsyncMock(return_value='RANGE_BOUND')
            ctx = await build_market_context(mock_ib, mock_contract, mock_config)

        required_keys = [
            'price', 'sma_200', 'expected_price', 'predicted_return',
            'action', 'confidence', 'reason', 'regime'
        ]
        for key in required_keys:
            assert key in ctx, f"Missing required key: {key}"

    @pytest.mark.asyncio
    async def test_action_always_neutral(self, mock_ib, mock_contract, mock_config):
        """No ML prior — action should always be NEUTRAL."""
        mock_ticker = MagicMock()
        mock_ticker.last = 335.0
        mock_ib.reqMktData.return_value = mock_ticker
        mock_ib.reqHistoricalDataAsync.return_value = [MagicMock(close=330 + i) for i in range(210)]

        with patch('trading_bot.market_data_provider.RegimeDetector') as mock_regime:
            mock_regime.detect_regime = AsyncMock(return_value='RANGE_BOUND')
            ctx = await build_market_context(mock_ib, mock_contract, mock_config)

        assert ctx['action'] == 'NEUTRAL'
        assert ctx['confidence'] == 0.5

    @pytest.mark.asyncio
    async def test_handles_no_price(self, mock_ib, mock_contract, mock_config):
        """When IBKR returns no price, should return safe fallback."""
        import math
        mock_ticker = MagicMock()
        mock_ticker.last = float('nan')
        mock_ticker.close = float('nan')
        mock_ib.reqMktData.return_value = mock_ticker

        ctx = await build_market_context(mock_ib, mock_contract, mock_config)

        assert ctx['confidence'] == 0.0  # Zero confidence = won't trade
        assert ctx['data_source'] == 'FALLBACK'


class TestEmptyMarketContext:

    def test_has_all_required_keys(self):
        from ib_insync import Future
        contract = Future(localSymbol='KCK6', lastTradeDateOrContractMonth='20260512')
        ctx = _empty_market_context(contract)
        assert ctx['action'] == 'NEUTRAL'
        assert ctx['confidence'] == 0.0


class TestFormatForPrompt:

    def test_basic_format(self):
        ctx = {
            'price': 335.5,
            'sma_200': 340.0,
            'regime': 'RANGE_BOUND',
            'volatility_5d': 0.025,
            'price_vs_sma': -0.0132,
        }
        result = format_market_context_for_prompt(ctx)
        assert "335.50" in result
        assert "340.00" in result
        assert "RANGE_BOUND" in result
        assert "BELOW" in result

    def test_handles_missing_sma(self):
        ctx = {
            'price': 335.5,
            'sma_200': None,
            'regime': 'UNKNOWN',
            'volatility_5d': None,
            'price_vs_sma': None,
        }
        result = format_market_context_for_prompt(ctx)
        assert "335.50" in result
        assert "UNKNOWN" in result
        # Should not crash on None values


class TestBuildAllMarketContexts:

    @pytest.mark.asyncio
    async def test_parallel_fetch(self, mock_ib, mock_config):
        """Should build context for multiple contracts in parallel."""
        from ib_insync import Future

        contracts = [
            Future(conId=i, localSymbol=f'KC{m}6', lastTradeDateOrContractMonth=f'2026{m:02d}12')
            for i, m in enumerate([3, 5, 7, 9, 12], start=1)
        ]

        mock_ticker = MagicMock()
        mock_ticker.last = 335.0
        mock_ib.reqMktData.return_value = mock_ticker
        mock_ib.reqHistoricalDataAsync.return_value = [MagicMock(close=330 + i) for i in range(210)]

        with patch('trading_bot.market_data_provider.RegimeDetector') as mock_regime:
            mock_regime.detect_regime = AsyncMock(return_value='RANGE_BOUND')
            contexts = await build_all_market_contexts(mock_ib, contracts, mock_config)

        assert len(contexts) == 5
