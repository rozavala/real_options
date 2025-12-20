import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from ib_insync import Contract, Future

from trading_bot.signal_generator import generate_signals

@pytest.mark.asyncio
async def test_generate_signals():
    ib = AsyncMock()
    # Thresholds are > 7 for BULLISH, < 0 for BEARISH
    config = {'symbol': 'KC', 'exchange': 'NYBOT', 'signal_thresholds': {'bullish': 7, 'bearish': 0}}

    # New signal format (list of dicts)
    # The generator now expects a list of dictionaries with specific keys
    signals_input = [
        {'action': 'LONG', 'confidence': 0.10, 'reason': 'Test', 'regime': 'BULL', 'price': 100.0, 'sma_200': 90.0, 'expected_price': 110.0},
        {'action': 'SHORT', 'confidence': -0.05, 'reason': 'Test', 'regime': 'BEAR', 'price': 100.0, 'sma_200': 110.0, 'expected_price': 95.0},
        {'action': 'NEUTRAL', 'confidence': 0.03, 'reason': 'Test', 'regime': 'CHOP', 'price': 100.0, 'sma_200': 100.0, 'expected_price': 103.0},
        {'action': 'LONG', 'confidence': 0.08, 'reason': 'Test', 'regime': 'BULL', 'price': 100.0, 'sma_200': 90.0, 'expected_price': 108.0},
        {'action': 'SHORT', 'confidence': -0.01, 'reason': 'Test', 'regime': 'BEAR', 'price': 100.0, 'sma_200': 110.0, 'expected_price': 99.0}
    ]

    # Mock the active futures contracts, ensuring they have the necessary attributes
    # The sorting logic depends on lastTradeDateOrContractMonth
    # NOTE: Contracts must be > 45 days out from "now" (simulated as current date) to be considered active.
    # Assuming test runs around Dec 2025, we push these out to 2026 to ensure they are valid.
    mock_futures = [
        Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='20260312'), # H26
        Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='20260512'), # K26
        Future(conId=3, symbol='KC', lastTradeDateOrContractMonth='20260712'), # N26
        Future(conId=4, symbol='KC', lastTradeDateOrContractMonth='20260912'), # U26
        Future(conId=5, symbol='KC', lastTradeDateOrContractMonth='20261212'), # Z26
    ]
    ib.reqContractDetailsAsync.return_value = [MagicMock(contract=f) for f in mock_futures]

    signals = await generate_signals(ib, signals_input, config)

    # We expect 5 signals because NEUTRAL ones are now included
    assert len(signals) == 5

    # The contracts are sorted chronologically. The predictions should map to them in order.
    # 1. H26 (202603) -> LONG -> BULLISH
    # 2. K26 (202605) -> SHORT -> BEARISH
    # 3. N26 (202607) -> NEUTRAL -> NEUTRAL
    # 4. U26 (202609) -> LONG -> BULLISH
    # 5. Z26 (202612) -> SHORT -> BEARISH

    # The final signals list should be ordered chronologically:
    assert signals[0]['direction'] == 'BULLISH'
    assert signals[0]['contract_month'] == '202603'

    assert signals[1]['direction'] == 'BEARISH'
    assert signals[1]['contract_month'] == '202605'

    assert signals[2]['direction'] == 'NEUTRAL'
    assert signals[2]['contract_month'] == '202607'

    assert signals[3]['direction'] == 'BULLISH'
    assert signals[3]['contract_month'] == '202609'

    assert signals[4]['direction'] == 'BEARISH'
    assert signals[4]['contract_month'] == '202612'

