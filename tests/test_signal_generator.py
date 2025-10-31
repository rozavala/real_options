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
    api_response = {
        "price_changes": [
            10.0,  # front_month -> BULLISH
            -5.0,  # second_month -> BEARISH
            3.0,   # third_month -> NO-TRADE
            8.0,   # fourth_month -> BULLISH
            -1.0   # fifth_month -> BEARISH
        ]
    }

    # Mock the active futures contracts, ensuring they have the necessary attributes
    # The sorting logic depends on lastTradeDateOrContractMonth
    mock_futures = [
        Future(conId=1, symbol='KC', lastTradeDateOrContractMonth='20251212'), # Z25
        Future(conId=2, symbol='KC', lastTradeDateOrContractMonth='20260312'), # H26
        Future(conId=3, symbol='KC', lastTradeDateOrContractMonth='20260512'), # K26
        Future(conId=4, symbol='KC', lastTradeDateOrContractMonth='20260712'), # N26
        Future(conId=5, symbol='KC', lastTradeDateOrContractMonth='20260912'), # U26
    ]
    ib.reqContractDetailsAsync.return_value = [MagicMock(contract=f) for f in mock_futures]

    signals = await generate_signals(ib, api_response, config)

    # We expect 4 signals (2 BULLISH, 2 BEARISH) because one price change is a no-trade
    assert len(signals) == 4

    # The contracts are sorted chronologically. The predictions should map to them in order.
    # 1. Z25 (202512) -> front_month -> prediction 10.0 -> BULLISH
    # 2. H26 (202603) -> second_month -> prediction -5.0 -> BEARISH
    # 3. K26 (202605) -> third_month -> prediction 3.0 -> NO-TRADE
    # 4. N26 (202607) -> fourth_month -> prediction 8.0 -> BULLISH
    # 5. U26 (202609) -> fifth_month -> prediction -1.0 -> BEARISH

    # The final signals list should be ordered chronologically:
    assert signals[0]['direction'] == 'BULLISH'
    assert signals[0]['contract_month'] == '202512'

    assert signals[1]['direction'] == 'BEARISH'
    assert signals[1]['contract_month'] == '202603'

    assert signals[2]['direction'] == 'BULLISH'
    assert signals[2]['contract_month'] == '202607'

    assert signals[3]['direction'] == 'BEARISH'
    assert signals[3]['contract_month'] == '202609'

