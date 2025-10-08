import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import requests
import os

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from send_data_to_api import send_data_and_get_prediction, find_latest_data_file

class TestApiCommunication(unittest.TestCase):

    @patch('send_data_to_api.find_latest_data_file')
    @patch('pandas.read_csv')
    @patch('requests.post')
    @patch('requests.get')
    @patch('time.sleep', return_value=None) # To avoid waiting in tests
    def test_send_data_success(self, mock_sleep, mock_get, mock_post, mock_read_csv, mock_find_file):
        # --- Mock setup ---
        mock_find_file.return_value = 'dummy_path.csv'
        mock_read_csv.return_value = pd.DataFrame({'data': [1, 2, 3]})

        # Mock the POST request to create the job
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {'id': 'test_job_123'}
        mock_post.return_value = mock_post_response

        # Mock the GET request to poll for results
        mock_get_pending_response = MagicMock()
        mock_get_pending_response.status_code = 200
        mock_get_pending_response.json.return_value = {'status': 'pending'}

        mock_get_completed_response = MagicMock()
        mock_get_completed_response.status_code = 200
        mock_get_completed_response.json.return_value = {
            'status': 'completed',
            'result': {'prediction': 'BULLISH'}
        }
        # Simulate one 'pending' response, then a 'completed' one
        mock_get.side_effect = [mock_get_pending_response, mock_get_completed_response]

        # --- Run the function ---
        result = send_data_and_get_prediction()

        # --- Assertions ---
        self.assertIsNotNone(result)
        self.assertEqual(result, {'prediction': 'BULLISH'})
        mock_post.assert_called_once()
        self.assertEqual(mock_get.call_count, 2)

    @patch('send_data_to_api.find_latest_data_file')
    @patch('pandas.read_csv')
    @patch('requests.post')
    def test_send_data_api_failure(self, mock_post, mock_read_csv, mock_find_file):
        mock_find_file.return_value = 'dummy_path.csv'
        mock_read_csv.return_value = pd.DataFrame({'data': [1, 2, 3]})

        # Simulate a requests exception
        mock_post.side_effect = requests.exceptions.RequestException("API is down")

        result = send_data_and_get_prediction()

        self.assertIsNone(result)

    def test_find_latest_data_file(self):
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.getmtime') as mock_getmtime:

            mock_listdir.return_value = [
                'coffee_futures_data_2025-10-06.csv',
                'coffee_futures_data_2025-10-07.csv', # This one is the latest
                'other_file.txt'
            ]

            # The path needs to match what listdir returns
            def mtime_side_effect(path):
                if '2025-10-07' in path:
                    return 100
                return 50
            mock_getmtime.side_effect = mtime_side_effect

            latest_file = find_latest_data_file('.')
            self.assertIn('2025-10-07', latest_file)


if __name__ == '__main__':
    unittest.main()