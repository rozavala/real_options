import sys
import os
import json
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Mock chromadb and pysqlite3 before importing sentinels (which imports tms -> chromadb)
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["pysqlite3"] = MagicMock()

from trading_bot.sentinels import Sentinel, WeatherSentinel, XSentimentSentinel, PredictionMarketSentinel

class TestSentinelImpl(Sentinel):
    async def check(self):
        return None

@pytest.fixture
def mock_config():
    return {
        "sentinels": {
            "weather": {},
            "x_sentiment": {},
            "prediction_markets": {"providers": {"polymarket": {}}}
        },
        "commodity": {"ticker": "KC"},
        "xai": {"api_key": "dummy"},
        "x_api": {"bearer_token": "dummy"},
        "gemini": {"api_key": "dummy"}
    }

@pytest.mark.asyncio
async def test_sentinel_async_save_seen_cache(mock_config, tmp_path):
    # Setup
    sentinel = TestSentinelImpl(mock_config)
    sentinel.CACHE_DIR = str(tmp_path)
    sentinel._cache_file = tmp_path / "TestSentinelImpl_seen.json"

    sentinel._seen_links = {"link1", "link2"}
    sentinel._seen_timestamps = {"link1": 100, "link2": 200}

    # Execute
    await sentinel._save_seen_cache_async()

    # Verify
    assert sentinel._cache_file.exists()
    with open(sentinel._cache_file, "r") as f:
        data = json.load(f)
    assert "link1" in data
    assert "link2" in data

@pytest.mark.asyncio
async def test_weather_sentinel_async_save_alert_state(mock_config, tmp_path):
    # Setup
    sentinel = WeatherSentinel(mock_config)
    sentinel.ALERT_STATE_FILE = str(tmp_path / "weather_alerts.json")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    sentinel._active_alerts = {
        "TEST_ALERT": {"time": now, "value": 1.0}
    }

    # Execute
    await sentinel._save_alert_state_async()

    # Verify
    assert os.path.exists(sentinel.ALERT_STATE_FILE)
    with open(sentinel.ALERT_STATE_FILE, "r") as f:
        data = json.load(f)
    assert "TEST_ALERT" in data
    assert data["TEST_ALERT"]["value"] == 1.0

@pytest.mark.asyncio
async def test_x_sentiment_sentinel_async_save_volume(mock_config, tmp_path):
    # Setup
    sentinel = XSentimentSentinel(mock_config)
    sentinel.CACHE_DIR = str(tmp_path)
    sentinel._volume_state_file = tmp_path / "XSentimentSentinel_volume.json"

    sentinel.post_volume_history = [10, 20, 30]

    # Execute
    await sentinel._save_volume_state_async()

    # Verify
    assert sentinel._volume_state_file.exists()
    with open(sentinel._volume_state_file, "r") as f:
        data = json.load(f)
    assert data["history"] == [10, 20, 30]

@pytest.mark.asyncio
async def test_prediction_market_async_save(mock_config):
    # Setup
    sentinel = PredictionMarketSentinel(mock_config)
    sentinel.state_cache = {"topic1": {"slug": "slug1"}}

    with patch("trading_bot.state_manager.StateManager.save_state_async", new_callable=AsyncMock) as mock_save:
        # Execute
        await sentinel._save_state_cache_async()

        # Verify
        mock_save.assert_awaited_once()
        args, kwargs = mock_save.call_args
        assert args[0] == {"topic1": {"slug": "slug1"}}
        assert kwargs["namespace"] == "prediction_market_state"
