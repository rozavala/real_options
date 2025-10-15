import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

import sys
import os
from ib_insync import Contract, ComboLeg

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_bot.order_manager import generate_and_queue_orders, _contract_to_dict
from trading_bot.utils import create_contract_from_dict


class TestOrderManager(unittest.TestCase):

    def test_contract_serialization_deserialization(self):
        """
        Verify that a complex combo contract can be serialized to a dict and
        then deserialized back into a fully functional Contract object.
        """
        # 1. Arrange: Create a complex Bag (combo) contract
        original_contract = Contract()
        original_contract.symbol = 'KC'
        original_contract.secType = 'BAG'
        original_contract.currency = 'USD'
        original_contract.exchange = 'NYBOT'
        leg1 = ComboLeg(conId=12345, ratio=1, action='BUY', exchange='NYBOT')
        leg2 = ComboLeg(conId=67890, ratio=1, action='SELL', exchange='NYBOT')
        original_contract.comboLegs = [leg1, leg2]

        # 2. Act: Serialize the contract to a dictionary
        contract_dict = _contract_to_dict(original_contract)

        # 3. Act: Deserialize the dictionary back to a contract object
        reconstructed_contract = create_contract_from_dict(contract_dict)

        # 4. Assert: Check that the reconstructed contract is identical to the original
        self.assertEqual(reconstructed_contract.symbol, original_contract.symbol)
        self.assertEqual(reconstructed_contract.secType, original_contract.secType)
        self.assertEqual(reconstructed_contract.currency, original_contract.currency)
        self.assertEqual(reconstructed_contract.exchange, original_contract.exchange)
        self.assertEqual(len(reconstructed_contract.comboLegs), 2)

        # Verify that the combo legs are actual ComboLeg objects with correct attributes
        self.assertIsInstance(reconstructed_contract.comboLegs[0], ComboLeg)
        self.assertIsInstance(reconstructed_contract.comboLegs[1], ComboLeg)
        self.assertEqual(reconstructed_contract.comboLegs[0].conId, leg1.conId)
        self.assertEqual(reconstructed_contract.comboLegs[0].action, leg1.action)
        self.assertEqual(reconstructed_contract.comboLegs[1].conId, leg2.conId)
        self.assertEqual(reconstructed_contract.comboLegs[1].action, leg2.action)

    @patch('trading_bot.order_manager.run_data_pull')
    @patch('trading_bot.order_manager.send_data_and_get_prediction')
    @patch('trading_bot.order_manager.IB')
    def test_generate_orders_uses_fallback_on_data_pull_failure(self, mock_ib, mock_send_data, mock_run_data_pull):
        """
        Verify that if run_data_pull fails, the process doesn't abort and
        instead proceeds to the next step (fetching predictions).
        """
        async def run_test():
            # Arrange: Simulate a data pull failure
            mock_run_data_pull.return_value = False

            # Arrange: Mock the subsequent functions to prevent them from running their full logic
            mock_send_data.return_value = {'price_changes': [1.0]} # Needs to return something to proceed
            mock_ib_instance = AsyncMock()
            mock_ib.return_value = mock_ib_instance

            config = {} # Dummy config

            # Act: Run the function
            await generate_and_queue_orders(config)

            # Assert: Check that the data pull was called
            mock_run_data_pull.assert_called_once()

            # Assert: Check that despite the failure, the process continued to the next step
            mock_send_data.assert_called_once()

            # Assert: Check that it tried to connect to IB, which is after the prediction step
            mock_ib_instance.connectAsync.assert_called_once()

        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()