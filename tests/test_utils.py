import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, mock_open
from datetime import datetime
import pandas as pd
import pytz
import os
from ib_insync import IB, Contract, FuturesOption, Bag, ComboLeg, OrderStatus, Trade, Order, Fill, Execution

from trading_bot.utils import (
    price_option_black_scholes,
    get_position_details,
    get_expiration_details,
    log_trade_to_ledger,
)


class TestUtils:

    def test_price_option_black_scholes(self):
        # Test with known values
        price = price_option_black_scholes(100, 100, 1, 0.05, 0.2, 'C')
        assert abs(price['price'] - 10.45) < 0.01

        # Test edge cases
        price = price_option_black_scholes(100, 100, 0, 0.05, 0.2, 'C')
        assert price is None

    async def test_get_position_details(self):
        ib = Mock()

        # --- Test with a single leg option ---
        position_single = Mock()
        position_single.contract = FuturesOption(symbol='KC', lastTradeDateOrContractMonth='202512', strike=3.5, right='C', exchange='NYBOT')
        details_single = await get_position_details(ib, position_single)
        assert details_single['type'] == 'SINGLE_LEG'
        assert details_single['key_strikes'] == [3.5]

        # --- Test with a bag option ---
        position_bag = Mock()
        leg1 = ComboLeg(conId=1, ratio=1, action='BUY', exchange='NYBOT')
        leg2 = ComboLeg(conId=2, ratio=1, action='SELL', exchange='NYBOT')
        position_bag.contract = Bag(symbol='KC', comboLegs=[leg1, leg2])

        # Mock the contract details resolution for the legs
        mock_cd1 = Mock()
        mock_cd1.contract = FuturesOption(conId=1, right='C', strike=3.5)
        mock_cd2 = Mock()
        mock_cd2.contract = FuturesOption(conId=2, right='C', strike=3.6)

        ib.reqContractDetailsAsync = AsyncMock(side_effect=[[mock_cd1], [mock_cd2]])

        details_bag = await get_position_details(ib, position_bag)
        assert details_bag['type'] == 'BULL_CALL_SPREAD'
        assert details_bag['key_strikes'] == [3.5, 3.6]

    def test_get_expiration_details(self):
        chain = {
            'expirations': ['20251120', '20251220'],
            'strikes_by_expiration': {
                '20251120': [3.4, 3.5, 3.6],
                '20251220': [3.4, 3.5, 3.6],
            }
        }
        details = get_expiration_details(chain, '202512')
        assert details['exp_date'] == '20251220'

    def test_round_to_tick(self):
        """Test tick size rounding for ICE Coffee options."""
        from trading_bot.utils import round_to_tick, COFFEE_OPTIONS_TICK_SIZE

        # BUY rounding (round DOWN)
        assert round_to_tick(17.98, COFFEE_OPTIONS_TICK_SIZE, 'BUY') == 17.95
        assert round_to_tick(18.805, COFFEE_OPTIONS_TICK_SIZE, 'BUY') == 18.80
        assert round_to_tick(41.405, COFFEE_OPTIONS_TICK_SIZE, 'BUY') == 41.40
        assert round_to_tick(39.95, COFFEE_OPTIONS_TICK_SIZE, 'BUY') == 39.95

        # SELL rounding (round UP)
        assert round_to_tick(17.98, COFFEE_OPTIONS_TICK_SIZE, 'SELL') == 18.00
        assert round_to_tick(18.801, COFFEE_OPTIONS_TICK_SIZE, 'SELL') == 18.85

    @patch('csv.DictWriter')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile', return_value=True)
    async def test_log_trade_to_ledger(self, mock_isfile, mock_open, mock_csv_writer):
        # --- Setup Mocks ---
        mock_ib = AsyncMock(spec=IB)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[
            FuturesOption(symbol='KC', localSymbol='KCH6 C3.5', strike=3.5, right='C', multiplier='37500'),
            FuturesOption(symbol='KC', localSymbol='KCH6 C3.6', strike=3.6, right='C', multiplier='37500')
        ])
        trade = Mock(spec=Trade)
        trade.orderStatus = Mock(spec=OrderStatus)
        trade.order = Mock(spec=Order, permId=123456, orderRef='test-uuid-string-longer-than-20-chars')
        trade.orderStatus.status = OrderStatus.Filled
        trade.contract = Bag(symbol='KC', comboLegs=[Mock(), Mock()]) # Required for position_id generation

        fill1 = Mock(spec=Fill)
        fill1.contract = FuturesOption(symbol='KC', localSymbol='KCH6 C3.5', strike=3.5, right='C', multiplier='37500')
        fill1.execution = Execution(execId='E1', time=datetime.now(), side='BOT', shares=1, price=0.5)

        fill2 = Mock(spec=Fill)
        fill2.contract = FuturesOption(symbol='KC', localSymbol='KCH6 C3.6', strike=3.6, right='C', multiplier='37500')
        fill2.execution = Execution(execId='E2', time=datetime.now(), side='SLD', shares=1, price=0.2)

        trade.fills = [fill1, fill2]

        mock_writer_instance = Mock()
        mock_csv_writer.return_value = mock_writer_instance

        # --- Call the function ---
        await log_trade_to_ledger(mock_ib, trade, "Test Combo")

        # --- Assertions ---
        # Check that the file was opened correctly
        expected_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_ledger.csv')
        mock_open.assert_called_once_with(expected_path, 'a', newline='')

        # Check that DictWriter was called with correct fieldnames
        expected_fieldnames = [
            'timestamp', 'position_id', 'combo_id', 'local_symbol', 'action', 'quantity',
            'avg_fill_price', 'strike', 'right', 'total_value_usd', 'reason'
        ]
        mock_csv_writer.assert_called_once_with(mock_open.return_value, fieldnames=expected_fieldnames)

        # Check that writerows was called once
        mock_writer_instance.writerows.assert_called_once()

        # Check the content of the written rows
        written_rows = mock_writer_instance.writerows.call_args[0][0]
        assert len(written_rows) == 2

        # Check the first leg (BUY order should have negative value)
        assert written_rows[0]['combo_id'] == 123456
        assert written_rows[0]['action'] == 'BUY'
        assert written_rows[0]['strike'] == 3.5
        assert abs(written_rows[0]['total_value_usd'] - -187.50) < 0.01

        # Check the second leg (SELL order should have positive value)
        assert written_rows[1]['combo_id'] == 123456
        assert written_rows[1]['action'] == 'SELL'
        assert written_rows[1]['strike'] == 3.6
        assert abs(written_rows[1]['total_value_usd'] - 75.0) < 0.01

    @patch('csv.DictWriter')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.isfile', return_value=True)
    async def test_log_trade_to_ledger_enriches_contract_details(self, mock_isfile, mock_open, mock_csv_writer):
        """
        Verify that if a fill contains an incomplete contract, the logger
        correctly uses the conId to find the full contract details from the
        ib.contracts cache.
        """
        # --- Setup Mocks ---
        # 1. Mock the IB object and its contracts cache
        mock_ib = AsyncMock(spec=IB)
        complete_contract = FuturesOption(
            conId=123, symbol='KC', localSymbol='KCH6 P3.2',
            strike=3.2, right='P', multiplier='37500'
        )
        mock_ib.contracts = {123: complete_contract} # Simulate the cache
        # Mock the async qualification to return the complete contract
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[complete_contract])

        # 2. Create a trade where the fill contains an INCOMPLETE contract
        trade = Mock(spec=Trade)
        trade.order = Mock(spec=Order, permId=98765, orderRef='test-uuid-string-longer-than-20-chars')
        trade.contract = Bag(symbol='KC', comboLegs=[Mock()])

        incomplete_contract = Mock(spec=Contract, conId=123)
        # We explicitly remove strike/right to simulate the problem
        del incomplete_contract.strike
        del incomplete_contract.right

        fill = Mock(spec=Fill)
        fill.contract = incomplete_contract
        fill.execution = Execution(execId='E3', time=datetime.now(), side='BOT', shares=1, price=0.8)
        trade.fills = [fill]

        mock_writer_instance = Mock()
        mock_csv_writer.return_value = mock_writer_instance

        # --- Call the function ---
        await log_trade_to_ledger(mock_ib, trade, "Test Enrichment")

        # --- Assertions ---
        # Verify that writerows was called and captured the data
        mock_writer_instance.writerows.assert_called_once()
        written_rows = mock_writer_instance.writerows.call_args[0][0]

        assert len(written_rows) == 1

        # CRITICAL: Assert that the logged data used the ENRICHED details
        # from the 'complete_contract' in the cache, not the incomplete one.
        assert written_rows[0]['strike'] == 3.2
        assert written_rows[0]['right'] == 'P'
        assert written_rows[0]['local_symbol'] == 'KCH6 P3.2'

    async def test_log_trade_to_ledger_race_condition(self):
        """
        Verify that when multiple coroutines call log_trade_to_ledger
        concurrently on a new file, the header is only written once.
        """
        # --- Setup ---
        # NOTE: This test writes to the actual trade_ledger.csv.
        # It's critical to clean it up before and after the test.
        ledger_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trade_ledger.csv')
        if os.path.exists(ledger_path):
            os.remove(ledger_path)

        # Mock IB and a sample trade
        mock_ib = AsyncMock(spec=IB)
        # Mock the async qualification to return a valid contract
        qualified_contract = FuturesOption(conId=123, localSymbol='KC H6 C3.5', strike=3.5, right='C', multiplier='37500')
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        trade = Mock(spec=Trade)
        trade.order = Mock(spec=Order, permId=123, orderRef='test-uuid-string-longer-than-20-chars')
        trade.contract = Bag(symbol='KC') # Simulate a combo to trigger qualification path
        trade.contract.comboLegs = [Mock()]

        fill = Mock(spec=Fill)
        fill.contract = Contract(conId=123)
        fill.execution = Execution(execId='E1', time=datetime.now(), side='BOT', shares=1, price=0.5)
        trade.fills = [fill]

        # --- Create concurrent logging tasks ---
        tasks = []
        for i in range(5):
            # Create a unique trade mock for each task to avoid shared state issues
            task_trade = Mock(spec=Trade)
            task_trade.order = Mock(spec=Order, permId=12345 + i, orderRef=f'test-uuid-string-longer-than-20-chars-{i}')
            task_trade.contract = trade.contract
            task_trade.fills = [fill]
            tasks.append(log_trade_to_ledger(mock_ib, task_trade, f"Concurrent Write {i}"))

        # --- Run tasks concurrently ---
        await asyncio.gather(*tasks)

        # --- Assertions ---
        # 1. Check the final content of the file
        with open(ledger_path, 'r') as f:
            content = f.read()

        # 2. Assert that the header appears exactly once
        header = "timestamp,position_id,combo_id,local_symbol,action,quantity,avg_fill_price,strike,right,total_value_usd,reason"
        assert content.count(header) == 1, "Header should only appear once in the ledger file."

        # 3. Assert that all 5 trade rows were written
        num_lines = len(content.strip().split('\n'))
        assert num_lines == 6, "File should contain 1 header line and 5 data lines."

        # --- Cleanup ---
        os.remove(ledger_path)

