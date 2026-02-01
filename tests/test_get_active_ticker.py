"""Test the get_active_ticker helper."""
from trading_bot.utils import get_active_ticker


def test_reads_from_commodity_ticker():
    config = {'commodity': {'ticker': 'CC'}, 'symbol': 'KC'}
    assert get_active_ticker(config) == 'CC'


def test_falls_back_to_symbol():
    config = {'symbol': 'KC'}
    assert get_active_ticker(config) == 'KC'


def test_ultimate_fallback():
    config = {}
    assert get_active_ticker(config) == 'KC'


def test_commodity_takes_precedence():
    """commodity.ticker should always win over symbol."""
    config = {'commodity': {'ticker': 'SB'}, 'symbol': 'KC'}
    assert get_active_ticker(config) == 'SB'
