import pytest
import os
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from trading_bot.sentinels import Sentinel, LogisticsSentinel

class TestSentinel(Sentinel):
    def __init__(self, config):
        super().__init__(config)

    async def check(self):
        return None

def test_persistence(tmp_path):
    # Patch CACHE_DIR to use temp dir
    with pytest.MonkeyPatch.context() as m:
        cache_dir = tmp_path / "sentinel_caches"
        m.setattr(Sentinel, "CACHE_DIR", str(cache_dir))

        # 1. Initialize sentinel
        s1 = TestSentinel({})

        # 2. Add item to cache and save
        s1._seen_links.add("http://test.com/1")
        s1._save_seen_cache()

        # Verify file exists
        expected_file = cache_dir / "TestSentinel_seen.json"
        assert expected_file.exists()

        # 3. Initialize new sentinel instance
        s2 = TestSentinel({})

        # Verify it loaded the cache
        assert "http://test.com/1" in s2._seen_links

def test_logistics_sentinel_deduplication(tmp_path):
    # Test that LogisticsSentinel inherits and uses persistence
    with pytest.MonkeyPatch.context() as m:
        cache_dir = tmp_path / "sentinel_caches"
        m.setattr(Sentinel, "CACHE_DIR", str(cache_dir))

        config = {'gemini': {'api_key': 'fake'}, 'sentinels': {'logistics': {}}}
        ls = LogisticsSentinel(config)

        # Manually inject a link into seen_links (simulating fetch)
        ls._seen_links.add("http://link1")
        ls._save_seen_cache()

        # New instance
        ls2 = LogisticsSentinel(config)
        assert "http://link1" in ls2._seen_links

def test_duplicate_payload(tmp_path):
    with pytest.MonkeyPatch.context() as m:
        cache_dir = tmp_path / "sentinel_caches"
        m.setattr(Sentinel, "CACHE_DIR", str(cache_dir))

        s = TestSentinel({})

        payload = {"data": "test"}

        # First check
        assert s._is_duplicate_payload(payload) == False

        # Second check (duplicate)
        assert s._is_duplicate_payload(payload) == True

        # Different payload
        payload2 = {"data": "test2"}
        assert s._is_duplicate_payload(payload2) == False
