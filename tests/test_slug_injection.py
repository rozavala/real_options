import pytest
import sys
import os
from unittest.mock import AsyncMock, patch, MagicMock
import urllib.parse

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.sentinels import PredictionMarketSentinel

@pytest.mark.asyncio
async def test_slug_injection_vulnerability():
    # Setup mock config
    mock_config = {
        'sentinels': {
            'prediction_markets': {
                'providers': {'polymarket': {'api_url': 'https://api.test'}},
                'enabled': True,
                'min_liquidity_usd': 0,
                'min_volume_usd': 0
            }
        },
        'commodity': {'ticker': 'KC'},
        'data_dir': '/tmp'
    }

    # Malicious slug attempting to inject a parameter
    # We use a slug that would alter the query string if not encoded
    malicious_slug = "event-slug&injected_param=true"

    with patch('trading_bot.sentinels.aiohttp.ClientSession') as MockSession, \
         patch('trading_bot.sentinels.get_commodity_profile'): # Mock profile to avoid loading issues

        mock_session_instance = AsyncMock()
        MockSession.return_value = mock_session_instance

        # Mock the context manager behavior of session.get()
        mock_get_ctx = MagicMock()
        mock_get_ctx.__aenter__ = AsyncMock()
        mock_get_ctx.__aexit__ = AsyncMock()
        mock_session_instance.get.return_value = mock_get_ctx

        # Mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])
        mock_get_ctx.__aenter__.return_value = mock_response

        sentinel = PredictionMarketSentinel(mock_config)

        # Inject our mock session directly to avoid _get_session logic creating a new one
        sentinel._session = mock_session_instance

        # Call the method
        await sentinel._fetch_by_slug(malicious_slug)

        # Check call arguments
        assert mock_session_instance.get.called
        args, kwargs = mock_session_instance.get.call_args
        url = args[0]
        params = kwargs.get('params', {})

        print(f"URL: {url}")
        print(f"Params: {params}")

        # VERIFICATION LOGIC:
        is_safe = False
        if 'slug' in params:
            # Safe usage of params dict
            assert params['slug'] == malicious_slug
            is_safe = True
        else:
            # Check for URL encoding in the string
            # If the slug was encoded manually or via some other safe method
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)

            # If safely encoded, 'slug' key should have the full malicious string as value
            # Note: parse_qs returns a list of values
            if 'slug' in query_params and query_params['slug'][0] == malicious_slug:
                 is_safe = True

            # Also ensure 'injected_param' is NOT a separate key
            if 'injected_param' in query_params:
                is_safe = False

        assert is_safe, f"Vulnerability detected! injected_param found in URL query parameters: {url}"
