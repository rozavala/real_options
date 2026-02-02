import asyncio
import logging
import os
import hashlib
from pathlib import Path
import feedparser
import requests
import statistics
from openai import AsyncOpenAI
from email.utils import parsedate_to_datetime
from datetime import datetime, time, timezone, timedelta
import numpy as np
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types
import pytz
import aiohttp
import json
import re
from functools import wraps
from notifications import send_pushover_notification
from trading_bot.state_manager import StateManager
from trading_bot.rate_limiter import acquire_api_slot

logger = logging.getLogger(__name__)

def with_retry(max_attempts: int = 3, backoff: float = 2.0):
    """Decorator for retrying failed async operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    wait_time = backoff ** attempt
                    logger.warning(f"{func.__name__} failed (attempt {attempt+1}/{max_attempts}): {e}")
                    await asyncio.sleep(wait_time)

            # All retries exhausted
            logger.error(f"{func.__name__} failed after {max_attempts} attempts: {last_error}")
            return None
        return wrapper
    return decorator

class SentinelTrigger:
    """Represents an event triggered by a sentinel."""
    def __init__(self, source: str, reason: str, payload: Dict[str, Any] = None, severity: int = 5):
        self.source = source
        self.reason = reason
        self.payload = payload or {}
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)

    def __repr__(self):
        return f"SentinelTrigger(source='{self.source}', reason='{self.reason}')"

class Sentinel:
    """Base class for all sentinels."""
    CACHE_DIR = "data/sentinel_caches"

    def __init__(self, config: dict):
        self.config = config
        self.enabled = True
        self.last_triggered = 0 # Timestamp of last trigger
        self.last_payload_hash = None

        # Persistent seen cache
        self._cache_file = Path(self.CACHE_DIR) / f"{self.__class__.__name__}_seen.json"
        self._seen_links = self._load_seen_cache()

    def _validate_ai_response(self, data: Any, context: str = "") -> Optional[dict]:
        """
        Validate that an AI response is a dict.

        Use after any _analyze_with_ai() call that expects JSON objects.
        Returns the data if valid, None if not.

        Args:
            data: Response from _analyze_with_ai()
            context: Description for logging (e.g., "sentiment analysis")
        """
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.warning(
                f"{self.__class__.__name__} AI returned {type(data).__name__} "
                f"instead of dict{f' for {context}' if context else ''}: "
                f"{str(data)[:100]}"
            )
            return None
        return data

    def _load_seen_cache(self) -> set:
        """Load seen links from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                    # Only keep links from last 7 days
                    import time as time_module
                    cutoff = time_module.time() - (7 * 24 * 3600)
                    return {k for k, v in data.items() if v > cutoff}
            except Exception as e:
                logger.warning(f"Failed to load seen cache: {e}")
        return set()

    def _save_seen_cache(self):
        """Save seen links to disk with timestamps."""
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            import time as time_module
            # Store with timestamps for pruning
            data = {link: time_module.time() for link in self._seen_links}
            with open(self._cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save seen cache: {e}")

    def _is_duplicate_payload(self, payload: dict) -> bool:
        """Check if payload is semantically similar to last trigger."""
        # Normalize and hash
        normalized = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.md5(normalized.encode()).hexdigest()

        if current_hash == self.last_payload_hash:
            logger.info(f"{self.__class__.__name__}: Duplicate payload detected")
            return True

        self.last_payload_hash = current_hash
        return False

    async def check(self) -> Optional[SentinelTrigger]:
        """Performs the sentinel check. Returns a SentinelTrigger if fired, else None."""
        raise NotImplementedError

    async def _fetch_rss_safe(self, url: str, seen_cache: set, timeout: int = 10, max_age_hours: int = 48) -> List[str]:
        """Fetch RSS with timeout, validation, and DATE FILTERING using aiohttp."""
        headlines = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"RSS {url} returned {response.status}")
                        return []
                    content = await response.text()

            loop = asyncio.get_running_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, content)

            if feed.bozo:
                logger.warning(f"Malformed RSS from {url}: {feed.bozo_exception}")
                return []

            for entry in feed.entries[:10]:
                # === DATE FILTERING ===
                pub_date = None

                # Try multiple date fields (RSS feeds are inconsistent)
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        pass

                if not pub_date and hasattr(entry, 'published') and entry.published:
                    try:
                        pub_date = parsedate_to_datetime(entry.published)
                        if pub_date.tzinfo is None:
                            pub_date = pub_date.replace(tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        pass

                if not pub_date and hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    try:
                        pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        pass

                # HARD FILTER: No date or too old = skip
                if pub_date is None:
                    logger.debug(f"Skipping headline with no date: {entry.title[:50]}")
                    continue

                if pub_date < cutoff_time:
                    logger.debug(f"Skipping stale headline ({pub_date.date()}): {entry.title[:50]}")
                    continue

                # Use link as unique ID
                if entry.link not in seen_cache:
                    headlines.append(entry.title)
                    seen_cache.add(entry.link)

                if len(headlines) >= 5:
                    break

        except asyncio.TimeoutError:
            logger.error(f"RSS timeout for {url}")
        except Exception as e:
            logger.error(f"RSS fetch failed for {url}: {e}")

        return headlines

class PriceSentinel(Sentinel):
    """
    Watches for sudden volatility or liquidity gaps.
    Frequency: Every 1 Minute (Market Hours).
    """
    def __init__(self, config: dict, ib_instance):
        super().__init__(config)
        self.ib = ib_instance
        self.sentinel_config = config.get('sentinels', {}).get('price', {})
        self.pct_change_threshold = self.sentinel_config.get('pct_change_threshold', 1.5)
        self.symbol = config.get('symbol', 'KC')
        self.exchange = config.get('exchange', 'NYBOT')

    async def check(self, cached_contract=None) -> Optional[SentinelTrigger]:
        # Guard: Check connection before doing anything
        if self.ib is None or not self.ib.isConnected():
            return None

        import time as time_module
        now = time_module.time()

        # Cooldown Check (1 hour = 3600s)
        if (now - self.last_triggered) < 3600:
            return None

        # Market Hours Check: 09:00 - 17:00 NY, Mon-Fri
        # Logic: Calculate local NY times then convert to UTC for comparison
        utc = timezone.utc
        ny_tz = pytz.timezone('America/New_York')
        now_utc = datetime.now(utc)
        now_ny = now_utc.astimezone(ny_tz)

        if now_ny.weekday() >= 5: # Sat(5) or Sun(6)
            return None

        market_start_ny = now_ny.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end_ny = now_ny.replace(hour=17, minute=0, second=0, microsecond=0)

        market_start_utc = market_start_ny.astimezone(utc)
        market_end_utc = market_end_ny.astimezone(utc)

        if not (market_start_utc <= now_utc <= market_end_utc):
            return None

        try:
            contract = cached_contract
            if not contract:
                from trading_bot.ib_interface import get_active_futures
                active_futures = await get_active_futures(self.ib, self.symbol, self.exchange, count=1)

                if not active_futures:
                    return None

                contract = active_futures[0]

            # Detect Price Shock within the current session (Last 1 Hour)
            # Request 1-hour bar to see intraday move
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='3600 S', # Last 1 hour
                barSizeSetting='1 hour',
                whatToShow='TRADES',
                useRTH=True
            )

            if bars:
                last_bar = bars[-1]
                if last_bar.open > 0:
                    pct_change = ((last_bar.close - last_bar.open) / last_bar.open) * 100

                    if abs(pct_change) > self.pct_change_threshold:
                        msg = f"Price moved {pct_change:.2f}% in last hour (Threshold: {self.pct_change_threshold}%)"
                        logger.warning(f"PRICE SENTINEL TRIGGERED: {msg}")

                        # Update Trigger Timestamp
                        self.last_triggered = now
                        return SentinelTrigger("PriceSentinel", msg, {"contract": contract.localSymbol, "change": pct_change})

        except Exception as e:
            logger.error(f"Price Sentinel check failed: {e}")

        return None

class WeatherSentinel(Sentinel):
    """
    Monitors specific coffee-growing regions for frost or drought risks.
    Frequency: Every 4 Hours (24/7).
    """
    ALERT_STATE_FILE = "data/weather_sentinel_alerts.json"

    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('weather', {})
        self.api_url = self.sentinel_config.get('api_url', "https://api.open-meteo.com/v1/forecast")
        self.params = self.sentinel_config.get('params', "daily=temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=10")
        self.locations = self.sentinel_config.get('locations', [])
        self.triggers = self.sentinel_config.get('triggers', {})

        self._active_alerts = self._load_alert_state()
        self._alert_cooldown_hours = 24
        self._escalation_threshold = 0.05  # 5% worsening breaks cooldown

    def _load_alert_state(self) -> Dict[str, dict]:
        """Load alert state from disk."""
        if os.path.exists(self.ALERT_STATE_FILE):
            try:
                with open(self.ALERT_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Parse ISO timestamps back to datetime
                    for key, val in data.items():
                        if 'time' in val and isinstance(val['time'], str):
                            val['time'] = datetime.fromisoformat(val['time'])
                    logger.info(f"WeatherSentinel: Loaded {len(data)} active alerts from disk")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load weather alert state: {e}")
        return {}

    def _save_alert_state(self):
        """Persist alert state to disk."""
        try:
            os.makedirs(os.path.dirname(self.ALERT_STATE_FILE) or '.', exist_ok=True)
            # Serialize datetime to ISO format
            data = {}
            for key, val in self._active_alerts.items():
                data[key] = {
                    'time': val['time'].isoformat() if isinstance(val['time'], datetime) else val['time'],
                    'value': val['value']
                }
            with open(self.ALERT_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save weather alert state: {e}")

    async def check(self) -> Optional[SentinelTrigger]:
        for loc in self.locations:
            try:
                lat, lon = loc['lat'], loc['lon']
                name = loc['name']

                # Construct URL from config params
                url = f"{self.api_url}?latitude={lat}&longitude={lon}&{self.params}"

                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, requests.get, url)
                data = response.json()

                if 'daily' not in data:
                    continue

                min_temps = data['daily']['temperature_2m_min']
                precip = data['daily']['precipitation_sum']

                # Check Frost
                frost_threshold = self.triggers.get('min_temp_c', 4.0)
                min_temp = min(min_temps)
                if min_temp < frost_threshold:
                    alert_key = f"frost_{name}"
                    # Severity = distance below threshold. Higher is worse.
                    # E.g. Threshold 4. Temp 3 -> 1. Temp -1 -> 5.
                    severity_val = frost_threshold - min_temp

                    if self._should_alert(alert_key, severity_val):
                        self._active_alerts[alert_key] = {
                            "time": datetime.now(timezone.utc),
                            "value": severity_val
                        }
                        self._save_alert_state()
                        msg = f"Frost Risk in {name}: Min Temp {min_temp}°C < {frost_threshold}°C"
                        logger.warning(f"WEATHER SENTINEL DETECTED: {msg}")
                        return SentinelTrigger("WeatherSentinel", msg, {"location": name, "type": "FROST", "value": min_temp}, severity=8)

                # Check Drought
                drought_days_limit = self.triggers.get('drought_days', 10)
                rain_limit = self.triggers.get('drought_rain_mm', 5.0)

                low_rain_days = sum(1 for p in precip if p < rain_limit)

                if low_rain_days >= drought_days_limit:
                    alert_key = f"drought_{name}"
                    if self._should_alert(alert_key, low_rain_days):
                        self._active_alerts[alert_key] = {
                            "time": datetime.now(timezone.utc),
                            "value": low_rain_days
                        }
                        self._save_alert_state()
                        msg = f"Drought Risk in {name}: {low_rain_days} days with < {rain_limit}mm rain"
                        logger.warning(f"WEATHER SENTINEL DETECTED: {msg}")
                        return SentinelTrigger("WeatherSentinel", msg, {"location": name, "type": "DROUGHT", "days": low_rain_days}, severity=6)

            except Exception as e:
                logger.error(f"Weather Sentinel failed for {loc.get('name')}: {e}")

        return None

    def _should_alert(self, alert_key: str, current_value: float) -> bool:
        """
        Alert if:
        1. Never seen before, OR
        2. Cooldown expired, OR
        3. Situation has worsened by >5% (ESCALATION - breaks cooldown)
        """
        if alert_key not in self._active_alerts:
            return True

        prev = self._active_alerts[alert_key]
        hours_since = (datetime.now(timezone.utc) - prev["time"]).total_seconds() / 3600

        # Cooldown expired
        if hours_since > self._alert_cooldown_hours:
            return True

        # ESCALATION CHECK
        prev_value = prev["value"]
        if prev_value <= 0:
            return current_value > prev_value

        pct_change = (current_value - prev_value) / prev_value
        if pct_change > self._escalation_threshold:
            logger.warning(f"ESCALATION DETECTED for {alert_key}: "
                          f"{prev_value:.2f} -> {current_value:.2f} ({pct_change:.1%})")
            return True

        return False

class LogisticsSentinel(Sentinel):
    """
    Scans for supply chain disruptions using RSS + Gemini Flash.
    Frequency: Every 6 Hours (24/7).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('logistics', {})
        self.urls = self.sentinel_config.get('rss_urls', [])

        api_key = config.get('gemini', {}).get('api_key')
        self.client = genai.Client(api_key=api_key)
        self.model = self.sentinel_config.get('model', "gemini-3-flash-preview")

    @with_retry(max_attempts=3)
    async def _analyze_with_ai(self, prompt: str) -> Optional[str]:
        """AI analysis with retry logic."""
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip().upper()

    async def check(self) -> Optional[SentinelTrigger]:
        headlines = []
        for url in self.urls:
            new_titles = await self._fetch_rss_safe(url, self._seen_links, max_age_hours=48)
            headlines.extend(new_titles)

        # Save cache
        self._save_seen_cache()

        if not headlines:
            return None

        # === PAYLOAD DEDUPLICATION ===
        payload = {"headlines": headlines[:3]}
        if self._is_duplicate_payload(payload):
            return None

        prompt = (
            "Analyze these headlines for Critical Coffee Supply Chain Bottlenecks (Strikes, Port Closures, Blockades).\n"
            "Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
            "Question: Is there a CRITICAL disruption mentioned? Answer only 'YES' or 'NO'."
        )

        answer = await self._analyze_with_ai(prompt)

        if answer is None:
            # AI failed even after retries
            send_pushover_notification(
                self.config.get('notifications', {}),
                "Sentinel Degraded",
                f"LogisticsSentinel AI analysis failed. Manual monitoring recommended."
            )
            return None

        if "YES" in answer:
            msg = "Potential Supply Chain Disruption detected in headlines."
            logger.warning(f"LOGISTICS SENTINEL DETECTED: {msg}")
            return SentinelTrigger("LogisticsSentinel", msg, {"headlines": headlines[:3]}, severity=6)

        return None

class NewsSentinel(Sentinel):
    """
    Monitors broad market sentiment using RSS + Gemini Flash.
    Frequency: Every 1 Hour (Market Hours).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('news', {})
        self.urls = self.sentinel_config.get('rss_urls', [])
        self.threshold = self.sentinel_config.get('sentiment_magnitude_threshold', 8)

        api_key = config.get('gemini', {}).get('api_key')
        self.client = genai.Client(api_key=api_key)
        self.model = self.sentinel_config.get('model', "gemini-3-flash-preview")

    @with_retry(max_attempts=3)
    async def _analyze_with_ai(self, prompt: str) -> Optional[dict]:
        """AI analysis with retry logic."""
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[:-3]
        if text.endswith("```"): text = text[:-3]
        return json.loads(text)

    async def check(self) -> Optional[SentinelTrigger]:
        headlines = []
        for url in self.urls:
            new_titles = await self._fetch_rss_safe(url, self._seen_links, max_age_hours=48)
            headlines.extend(new_titles)

        # Save cache
        self._save_seen_cache()

        if not headlines:
            return None

        prompt = (
            "Analyze these headlines for EXTREME Market Sentiment regarding Coffee Futures.\n"
            "Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
            "Task: Score the 'Sentiment Magnitude' from 0 to 10 (where 10 is Market Crashing or Exploding panic/euphoria).\n"
            "Output JSON: {'score': int, 'summary': string}"
        )

        data = self._validate_ai_response(
            await self._analyze_with_ai(prompt),
            context="headline sentiment"
        )

        if data is None:
            # AI failed even after retries
            send_pushover_notification(
                self.config.get('notifications', {}),
                "Sentinel Degraded",
                f"NewsSentinel AI analysis failed. Manual monitoring recommended."
            )
            return None

        score = data.get('score', 0)
        if score >= self.threshold:
            msg = f"Extreme Sentiment Detected (Score: {score}/10): {data.get('summary')}"
            logger.warning(f"NEWS SENTINEL DETECTED: {msg}")
            return SentinelTrigger("NewsSentinel", msg, {"score": score, "summary": data.get('summary')}, severity=int(score))

        return None

class XSentimentSentinel(Sentinel):
    """
    Monitors X (Twitter) for coffee market sentiment using xAI Grok.

    ARCHITECTURE:
    - Frequency: Every 30 Minutes (Market Hours)
    - Uses Grok's X search + code execution for quantitative sentiment
    - Parallel query execution for low latency
    - Volume anomaly detection for early warning

    PRODUCTION FEATURES:
    - Explicit tool schemas (prevents hallucinations)
    - Weighted confidence scoring
    - Robust error handling & validation
    - Deduplication & history tracking
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('x_sentiment', {})

        # === SEARCH CONFIGURATION ===
        self.search_queries = self.sentinel_config.get('search_queries', [
            'coffee futures',
            'arabica prices',
            'KC futures',
            'robusta market',
            'coffee supply'
        ])
        self.from_handles = self.sentinel_config.get('from_handles', [])
        self.exclude_keywords = self.sentinel_config.get('exclude_keywords',
            ['meme', 'joke', 'spam', 'giveaway'])

        # === THRESHOLDS ===
        self.sentiment_threshold = self.sentinel_config.get('sentiment_threshold', 6.5)
        self.min_engagement = self.sentinel_config.get('min_engagement', 5)
        self.volume_spike_multiplier = self.sentinel_config.get('volume_spike_multiplier', 2.0)

        # === XAI CLIENT SETUP ===

        api_key = config.get('xai', {}).get('api_key')
        if not api_key or api_key == "LOADED_FROM_ENV":
            api_key = os.environ.get('XAI_API_KEY')

        if not api_key:
             raise ValueError("XAI_API_KEY not found. Set in .env or config.json")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=180.0  # Allow time for agentic loops
        )

        # === X API SETUP (for tool execution) ===
        self.x_bearer_token = config.get('x_api', {}).get('bearer_token')
        if not self.x_bearer_token or self.x_bearer_token == "LOADED_FROM_ENV":
            self.x_bearer_token = os.environ.get('X_BEARER_TOKEN')

        if not self.x_bearer_token:
            logger.warning("X_BEARER_TOKEN not found - X sentiment will use Grok's training knowledge only")

        # === MODEL SELECTION ===
        # CORRECTED: Use current production model
        self.model = self.sentinel_config.get('model', 'grok-4-1-fast-reasoning')

        # === MODEL VALIDATION ===
        KNOWN_GROK_MODELS = [
            'grok-4-1-fast-reasoning',
            'grok-4',
            'grok-3',
            'grok-2-1212',
            'grok-beta'
        ]
        if self.model not in KNOWN_GROK_MODELS:
            logger.warning(
                f"⚠️ Unrecognized Grok model '{self.model}' - verify with xAI API docs. "
                f"Known models: {KNOWN_GROK_MODELS}"
            )

        # === VOLUME TRACKING ===
        self.post_volume_history = []
        self.volume_mean = 0.0
        self.volume_stddev = 0.0

        # === CIRCUIT BREAKER ===
        self.consecutive_failures = 0
        self.degraded_until: Optional[datetime] = None
        self.sensor_status = "ONLINE"

        # === CONCURRENCY CONTROL (Protocol 1) ===
        # Semaphore removed in favor of GlobalRateLimiter
        self._request_interval = 1.5
        self._last_request_time = 0

        logger.info(f"XSentimentSentinel initialized with model: {self.model}")

    def get_sensor_status(self) -> dict:
        """Return sensor status for injection into Council context."""
        return {
            "sentiment_sensor_status": self.sensor_status,
            "consecutive_failures": self.consecutive_failures,
            "degraded_until": self.degraded_until.isoformat() if self.degraded_until else None
        }

    def _build_search_query(self, base_query: str) -> str:
        """Construct advanced X search query with filters."""
        query_parts = [base_query]

        # Add trusted handles if configured
        if self.from_handles:
            handle_filter = " OR ".join([f"from:{h.lstrip('@')}" for h in self.from_handles])
            query_parts.append(f"({handle_filter})")

        # Exclude noise keywords
        for keyword in self.exclude_keywords:
            query_parts.append(f"-{keyword}")

    # NOTE: min_faves removed - not available on Basic/Pro X API tiers
    # Engagement filtering is now done in post-processing (see _execute_x_search)

        return " ".join(query_parts)

    def _build_broad_search_query(self, base_query: str) -> str:
        """
        Construct a broader search query without author restrictions.
        Used as fallback when expert-filtered search returns no results.

        NOTE: min_faves operator removed - not available on Basic tier.
        Engagement filtering is done in post-processing via _fetch_x_posts with
        dynamic threshold passed from _execute_x_search.
        """
        query_parts = [base_query]

        # Exclude noise keywords
        for keyword in self.exclude_keywords:
            query_parts.append(f"-{keyword}")

        # REMOVED: min_faves operator (not available on Basic tier)
        # Threshold is now passed to _fetch_x_posts as min_likes parameter

        # Cap query length to prevent 400 errors (X API limit ~512 chars)
        full_query = " ".join(query_parts)
        if len(full_query) > 450:
            logger.warning(f"Broad query too long ({len(full_query)} chars), trimming exclusions")
            full_query = base_query

        return full_query

    async def _fetch_x_posts(self, query: str, limit: int, sort_order: str, min_likes: int = None) -> list:
        """
        Execute X API HTTP request and return posts.

        ENHANCED (X Expert): Added recency bound via start_time param.

        Args:
            query: Search query string
            limit: Max results to fetch
            sort_order: 'recency' or 'relevancy'
            min_likes: Minimum likes threshold for filtering. Defaults to self.min_engagement if None.

        Returns:
            list: Posts matching query (empty list on error/no results)
        """
        from datetime import timedelta

        headers = {
            "Authorization": f"Bearer {self.x_bearer_token}",
            "Content-Type": "application/json"
        }

        # === X EXPERT ENHANCEMENT: Add recency bound ===
        # Build query WITHOUT since: operator (not available on Basic tier)
        query_with_filters = f"{query} -is:retweet lang:en"

        # Use start_time parameter instead (ISO 8601 format)
        since_datetime = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "query": query_with_filters,
            "start_time": since_datetime,  # CORRECT: API parameter, not query operator
            "max_results": max(10, min(limit, 100)),
            "tweet.fields": "text,created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
            "sort_order": sort_order
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.twitter.com/2/tweets/search/recent",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 429:
                        logger.warning("X API rate limit hit")
                        return []

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"X API error {response.status}: {error_text}")
                        return []

                    data = await response.json()

                    posts = []
                    users = {}

                    if "includes" in data and "users" in data["includes"]:
                        for user in data["includes"]["users"]:
                            users[user["id"]] = user.get("username", "unknown")

                    for tweet in data.get("data", []):
                        metrics = tweet.get("public_metrics", {})
                        posts.append({
                            "text": tweet["text"][:400],
                            "author": users.get(tweet.get("author_id"), "unknown"),
                            "likes": metrics.get("like_count", 0),
                            "retweets": metrics.get("retweet_count", 0),
                            "created_at": tweet.get("created_at", "")
                        })

                    # Filter by engagement (post-processing since min_faves not available on Basic tier)
                    # Use passed threshold or fall back to instance default
                    likes_threshold = min_likes if min_likes is not None else self.min_engagement
                    posts = [p for p in posts if p.get('likes', 0) >= likes_threshold]

                    # === X EXPERT ENHANCEMENT: Log top post for debugging ===
                    if posts:
                        logger.debug(f"Top post: {posts[0]['text'][:50]}...")

                    return posts

        except asyncio.TimeoutError:
            logger.error("X API request timed out")
            return []
        except Exception as e:
            logger.error(f"X API request failed: {e}")
            return []

    def _update_volume_stats(self, new_volume: int):
        """Update rolling volume statistics for spike detection."""
        self.post_volume_history.append(new_volume)
        # Keep last 30 checks (~15 hours at 30min intervals)
        self.post_volume_history = self.post_volume_history[-30:]

        if len(self.post_volume_history) >= 5:
            # statistics imported at top
            self.volume_mean = statistics.mean(self.post_volume_history)
            self.volume_stddev = statistics.stdev(self.post_volume_history) if len(self.post_volume_history) > 1 else 0

    async def _sem_bound_search(self, query: str) -> Optional[dict]:
        """
        Wrapper with GlobalRateLimiter integration and circuit breaker.

        AMENDED: Now uses GlobalRateLimiter instead of local semaphore
        to prevent quota exhaustion race conditions with Council agents.

        Uses 'xai' provider bucket (25 RPM as configured in PROVIDER_LIMITS).
        """
        # === ACQUIRE SLOT FROM GLOBAL LIMITER ===
        slot_acquired = await acquire_api_slot('xai', timeout=30.0)

        if not slot_acquired:
            logger.warning(f"Rate limit exceeded for X Sentinel query: {query}")
            return None

        # Add jitter to prevent thundering herd
        jitter = np.random.uniform(0.5, 2.0)
        await asyncio.sleep(jitter)

        # Respect minimum interval between requests
        import time as time_module
        now = time_module.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request_time = time_module.time()

        try:
            result = await self._search_x_and_analyze(query)

            if result is not None:
                self.consecutive_failures = 0
                self.sensor_status = "ONLINE"

            return result

        except Exception as e:
            error_str = str(e)
            is_server_error = any(code in error_str for code in ['502', '503', '504', '500'])

            if is_server_error:
                self.consecutive_failures += 1
                logger.warning(
                    f"XSentimentSentinel: Server error. "
                    f"Consecutive failures: {self.consecutive_failures}"
                )

                if self.consecutive_failures >= 3:
                    self.degraded_until = datetime.now(timezone.utc) + timedelta(hours=1)
                    self.sensor_status = "OFFLINE"
                    logger.error(
                        f"XSentimentSentinel: CIRCUIT BREAKER TRIGGERED. "
                        f"Degraded until {self.degraded_until.isoformat()}"
                    )
                    StateManager.save_state(
                        {"x_sentiment": self.get_sensor_status()},
                        namespace="sensors"
                    )
            else:
                self.consecutive_failures += 1

            raise

    @with_retry(max_attempts=3)
    async def _search_x_and_analyze(self, query: str) -> Optional[dict]:
        """
        Use Grok with tool execution loop for live X sentiment analysis.

        ARCHITECTURE (per xAI expert feedback):
        1. Send query to Grok with x_search tool defined
        2. If Grok returns tool_calls, execute locally via X API
        3. Feed results back to Grok for analysis
        4. Loop until Grok returns final content
        """
        if not self.client:
            logger.error("XAI Client not initialized (missing API key)")
            return None

        search_query = self._build_search_query(query)

        # === TOOL DEFINITION ===
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "x_search",
                    "description": "Search X (Twitter) for posts related to a query to gather real-time sentiment data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'coffee futures sentiment' with operators like OR, -exclude)."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of posts to return (default 10, min 10, max 100)."
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["Top", "Latest"],
                                "description": "Sort preference: 'Latest' for most recent posts (recency), 'Top' for most relevant (relevancy). Maps to X API sort_order. Default 'Latest' for time-sensitive commodity sentiment."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # === SYSTEM PROMPT ===
        system_prompt = """You are an expert commodities market sentiment analyst.

Use the x_search tool to fetch live X data when needed for accurate sentiment analysis.
Analyze posts for bullish/bearish themes related to coffee futures.

IMPORTANT: Prioritize RECENT posts for time-sensitive commodity sentiment (weather events, crop reports, frost alerts, port disruptions, etc.). Use mode="Latest" unless specifically looking for high-engagement historical takes.

After analyzing posts, provide a JSON response with:
- sentiment_score: 0-10 (0=very bearish, 5=neutral, 10=very bullish)
- confidence: 0.0-1.0 based on data quality and volume
- summary: Executive summary (max 200 chars)
- post_volume: Number of relevant posts found
- key_themes: Top 3 themes detected
- notable_posts: Up to 3 notable posts with text snippet, engagement, sentiment

If the x_search tool returns no results or errors, provide neutral sentiment with low confidence."""

        # === MESSAGE LOOP ===
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze X sentiment for: {search_query}"}
        ]

        max_iterations = 5
        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if self.x_bearer_token else None,  # Only offer tools if we can execute them
                    tool_choice="auto" if self.x_bearer_token else None,
                    temperature=0.3,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )

                message = response.choices[0].message

                # === CASE 1: Grok provided final content ===
                if message.content and message.content.strip():
                    content = message.content.strip()

                    # Strip markdown wrappers
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    try:
                        data = json.loads(content)

                        # Validate required fields
                        required_fields = ['sentiment_score', 'confidence', 'summary', 'post_volume']
                        missing = [f for f in required_fields if f not in data]
                        if missing:
                            logger.warning(f"Missing fields {missing} in Grok response")
                            data.setdefault('sentiment_score', 5.0)
                            data.setdefault('confidence', 0.0)
                            data.setdefault('summary', 'Incomplete response')
                            data.setdefault('post_volume', 0)

                        # Type validation
                        data['sentiment_score'] = float(data['sentiment_score'])
                        data['confidence'] = float(data['confidence'])
                        data['post_volume'] = int(data['post_volume'])

                        # Update volume tracking
                        if data['post_volume'] > 0:
                            self._update_volume_stats(data['post_volume'])

                        logger.debug(f"X analysis for '{query}': sentiment={data['sentiment_score']:.1f}, "
                                   f"volume={data['post_volume']}, confidence={data['confidence']:.2f}")
                        return data

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from Grok: {e}")
                        return None

                # === CASE 2: Grok wants to use a tool ===
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name

                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {"query": query}

                    logger.debug(f"Grok requested tool: {function_name} with args: {args}")

                    if function_name == "x_search":
                        # Execute tool locally
                        tool_result = await self._execute_x_search(args)
                    else:
                        tool_result = {"error": f"Unknown tool: {function_name}"}

                    # === ERROR LOGGING (per expert feedback) ===
                    if "error" in tool_result:
                        logger.warning(f"Tool execution failed: {tool_result['error']}")
                        # Continue anyway - Grok will see the error and can provide fallback analysis

                    # Append tool call and result to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(tool_result)
                    })

                    continue  # Loop to get Grok's analysis of the results

                # === CASE 3: Empty response without tools ===
                logger.warning(f"Empty response from Grok without tool calls for '{query}'")
                return {
                    "sentiment_score": 5.0,
                    "confidence": 0.0,
                    "summary": "No data available",
                    "post_volume": 0,
                    "key_themes": [],
                    "notable_posts": []
                }

            # Max iterations reached
            logger.warning(f"Tool loop reached max iterations for '{query}'")
            return {
                "sentiment_score": 5.0,
                "confidence": 0.0,
                "summary": "Analysis loop timeout",
                "post_volume": 0,
                "key_themes": [],
                "notable_posts": []
            }

        except Exception as e:
            logger.error(f"X search failed for query '{query}': {e}", exc_info=True)
            return None


    async def _execute_x_search(self, args: dict) -> dict:
        """
        Execute X API search with two-stage fallback.

        ARCHITECTURE (AMENDED with X Expert feedback):
        Stage 1: Strict search (expert accounts from config)
        Stage 2: Broad search (general public with min_faves filter) if strict returns 0

        ENHANCED: Added data_quality flag for Grok confidence calibration.
        """
        if not self.x_bearer_token:
            return {
                "error": "X API not configured",
                "posts": [],
                "post_volume": 0,
                "data_quality": "unavailable"
            }

        query = args.get("query", "coffee futures")
        limit = args.get("limit", 10)

        # Map mode to X API sort_order
        mode = args.get("mode", "Latest")
        sort_order = "recency" if mode == "Latest" else "relevancy"

        # === NEW: SANITIZE LLM QUERY INPUT ===
        # Grok may pre-add exclusion keywords to the query. Remove them to prevent
        # duplication when _build_search_query adds them again.
        sanitized_query = query
        for keyword in self.exclude_keywords:
            # Remove both "-keyword" and "- keyword" patterns
            sanitized_query = sanitized_query.replace(f"-{keyword}", "").replace(f"- {keyword}", "")

        # Also remove any from: filters that Grok might have added
        # (we'll add our configured handles via _build_search_query)
        sanitized_query = re.sub(r'\(from:[^)]+\)', '', sanitized_query)
        sanitized_query = re.sub(r'from:\w+', '', sanitized_query)

        # Clean up extra whitespace
        sanitized_query = ' '.join(sanitized_query.split()).strip()

        if sanitized_query != query:
            logger.debug(f"Sanitized LLM query: '{query}' -> '{sanitized_query}'")

        # Track which stage succeeded for logging
        search_stage = "strict"

        # === STAGE 1: STRICT SEARCH (Expert Accounts) ===
        strict_query = self._build_search_query(sanitized_query)
        logger.debug(f"Stage 1 (strict) query: {strict_query}")
        posts = await self._fetch_x_posts(strict_query, limit, sort_order)

        # === STAGE 2: BROAD SEARCH FALLBACK ===
        if not posts:
            logger.info(f"Strict search for '{sanitized_query}' returned 0 posts. Trying broad search...")
            search_stage = "broad"
            broad_query = self._build_broad_search_query(sanitized_query)
            logger.debug(f"Stage 2 (broad) query: {broad_query}")

            # DYNAMIC THRESHOLD: Use config value for broad search (default: 3)
            # This is looser than strict search (default: 5) to capture more results
            broad_threshold = self.sentinel_config.get('broad_min_faves', 3)
            posts = await self._fetch_x_posts(broad_query, limit, sort_order, min_likes=broad_threshold)

            if posts:
                logger.info(f"Broad search returned {len(posts)} posts for '{sanitized_query}'")
            else:
                logger.warning(f"Both strict and broad search returned 0 posts for '{sanitized_query}'")
        else:
            logger.info(f"X API returned {len(posts)} posts for '{query}' (sort: {sort_order})")

        # === X EXPERT ENHANCEMENT: Data quality flag ===
        # Grok should reduce confidence when data is sparse
        data_quality = "high" if len(posts) >= 5 else ("low" if len(posts) < 3 else "medium")

        return {
            "posts": posts,
            "post_volume": len(posts),
            "query": query,
            "search_stage": search_stage,
            "data_quality": data_quality  # Grok uses this to calibrate confidence
        }

    async def check(self) -> Optional[SentinelTrigger]:
        """
        Check X sentiment across multiple queries IN PARALLEL.

        ENHANCEMENT: Parallel execution reduces latency from ~15s to ~3s.
        """
        # === MARKET HOURS GATE (Cost Optimization) ===
        from trading_bot.utils import is_market_open, is_trading_day

        if not is_trading_day():
            # Complete holiday/weekend - skip entirely
            logger.debug("Non-trading day - skipping X sentiment check")
            return None

        if not is_market_open():
            # Trading day but outside hours - reduced frequency
            if not hasattr(self, '_last_closed_market_check'):
                self._last_closed_market_check = None

            now = datetime.now(timezone.utc)
            if self._last_closed_market_check:
                hours_since_last = (now - self._last_closed_market_check).total_seconds() / 3600
                if hours_since_last < 4:
                    return None  # Skip check, still in cooldown

            self._last_closed_market_check = now
            logger.debug("Market closed - running periodic X sentiment check")

        # === CIRCUIT BREAKER CHECK ===
        if self.degraded_until and datetime.now(timezone.utc) < self.degraded_until:
            self.sensor_status = "OFFLINE"
            # Refresh state to ensure persistence knows we are still offline
            StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")
            return None

        # === PARALLEL QUERY EXECUTION ===
        tasks = [self._sem_bound_search(query) for query in self.search_queries]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # === FILTER SUCCESSFUL RESULTS ===
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Query '{self.search_queries[i]}' failed: {result}")
            elif result is not None:
                valid_results.append(result)

        if not valid_results:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.degraded_until = datetime.now(timezone.utc) + timedelta(hours=1)
                self.sensor_status = "OFFLINE"
                logger.warning("XSentimentSentinel entering degraded mode for 1 hour due to 3 consecutive failures")
                # Optionally send notification here

            # Save status
            StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")

            logger.warning("XSentimentSentinel: All X searches failed")
            return None

        # Success - Reset failure counter
        self.consecutive_failures = 0
        self.sensor_status = "ONLINE"
        # Save status
        StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")

        # === WEIGHTED AGGREGATION ===
        # ENHANCEMENT: Weight by confidence and post volume for better signal
        total_weight = 0.0
        weighted_sentiment = 0.0

        for r in valid_results:
            weight = r.get('confidence', 0.5) * max(1.0, r.get('post_volume', 1) / 10.0)
            weighted_sentiment += r.get('sentiment_score', 5.0) * weight
            total_weight += weight

        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 5.0

        # Simple average confidence
        avg_confidence = sum(r.get('confidence', 0.5) for r in valid_results) / len(valid_results)

        # === EXTRACT THEMES ===
        all_themes = []
        for r in valid_results:
            all_themes.extend(r.get('key_themes', []))

        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # === VOLUME SPIKE DETECTION ===
        # ENHANCEMENT: Use statistical thresholds (mean + 2*stddev)
        total_volume = sum(r.get('post_volume', 0) for r in valid_results)
        volume_spike_detected = False

        if len(self.post_volume_history) >= 5:
            # Simple multiplier check
            if total_volume > (self.volume_mean * self.volume_spike_multiplier):
                volume_spike_detected = True
                logger.warning(f"X volume spike: {total_volume} vs mean {self.volume_mean:.0f} (>{self.volume_spike_multiplier}x)")

            # Statistical outlier check (optional additional signal)
            if self.volume_stddev > 0:
                z_score = (total_volume - self.volume_mean) / self.volume_stddev
                if z_score > 2.0:
                    logger.info(f"X volume anomaly: z-score={z_score:.1f}")

        # === TRIGGER LOGIC ===
        trigger_reason = None
        severity = 0

        # Extreme sentiment triggers
        if avg_sentiment >= self.sentiment_threshold:
            trigger_reason = f"EXTREMELY BULLISH X sentiment (score: {avg_sentiment:.1f}/10, confidence: {avg_confidence:.0%})"
            severity = 7
        elif avg_sentiment <= (10 - self.sentiment_threshold):
            trigger_reason = f"EXTREMELY BEARISH X sentiment (score: {avg_sentiment:.1f}/10, confidence: {avg_confidence:.0%})"
            severity = 7

        # Volume spike trigger (even if sentiment neutral)
        if volume_spike_detected and not trigger_reason:
            trigger_reason = f"UNUSUAL X ACTIVITY SPIKE ({total_volume} posts, {self.volume_spike_multiplier:.1f}x baseline)"
            severity = 6

        if not trigger_reason:
            logger.debug(f"XSentimentSentinel: No trigger (sentiment={avg_sentiment:.1f}, volume={total_volume})")
            return None

        # === BUILD PAYLOAD ===
        payload = {
            "sentiment_score": round(avg_sentiment, 2),
            "confidence": round(avg_confidence, 2),
            "top_themes": [t[0] for t in top_themes],
            "post_volume": total_volume,
            "volume_baseline": round(self.volume_mean, 1),
            "volume_spike": volume_spike_detected,
            "query_results": [
                {
                    "query": self.search_queries[i],
                    "score": r.get('sentiment_score'),
                    "volume": r.get('post_volume'),
                    "confidence": r.get('confidence'),
                    "top_post": r.get('notable_posts', [{}])[0].get('text', 'N/A') if r.get('notable_posts') else None
                }
                for i, r in enumerate(valid_results)
            ][:3]  # Top 3 for context
        }

        # === DEDUPLICATION CHECK ===
        if self._is_duplicate_payload(payload):
            logger.info("XSentimentSentinel: Duplicate sentiment signal detected, skipping")
            return None

        logger.warning(f"🚨 X SENTINEL TRIGGERED: {trigger_reason}")

        return SentinelTrigger(
            source="XSentimentSentinel",
            reason=trigger_reason,
            payload=payload,
            severity=severity
        )

class PredictionMarketSentinel(Sentinel):
    """
    Monitors Prediction Markets via Dynamic Market Discovery.
    Automatically follows liquidity to the most active relevant market.

    ARCHITECTURE v2.0:
    - Frequency: Every 5 Minutes (24/7)
    - No LLM calls (pure API + threshold logic)
    - Zero maintenance: auto-discovers markets by topic/query

    KEY MECHANISMS:
    1. Dynamic Discovery: Searches by query, picks highest-liquidity active market
    2. Slug Consistency: Detects market rollover (June→July), resets baseline
    3. High-Water Mark: Prevents alert flapping on severity oscillation
    4. HWM Decay: Resets HWM after 24h to allow re-alerting on persistent situations

    COFFEE RELEVANCE:
    - Fed Policy → USD strength → Coffee export pricing
    - Brazil Rates → BRL/USD → Export competitiveness
    - Elections → Trade policy → Tariff risk
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('prediction_markets', {})

        # API Configuration
        polymarket_config = self.sentinel_config.get('providers', {}).get('polymarket', {})
        self.api_url = polymarket_config.get('api_url', "https://gamma-api.polymarket.com/events")
        self.search_limit = polymarket_config.get('search_limit', 10)

        # Topics to watch (queries, not slugs)
        self.topics = self.sentinel_config.get('topics_to_watch', [])

        # Liquidity Filters
        self.min_liquidity = self.sentinel_config.get('min_liquidity_usd', 50000)
        self.min_volume = self.sentinel_config.get('min_volume_usd', 10000)

        # High-Water Mark Decay (hours)
        self.hwm_decay_hours = self.sentinel_config.get('hwm_decay_hours', 24)

        # State Cache: { topic_query: { slug, price, timestamp, severity_hwm, hwm_timestamp } }
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self._load_state_cache()

        # Polling interval (internal rate limiting)
        self.poll_interval = self.sentinel_config.get('poll_interval_seconds', 300)
        self._last_poll_time = 0

        # Severity mapping
        self.severity_map = self.sentinel_config.get('severity_mapping', {
            '10_to_20_pct': 6,
            '20_to_30_pct': 7,
            '30_plus_pct': 9
        })

        # Track topics that persistently fail to find markets
        self._topic_failure_counts: Dict[str, int] = {}

        logger.info(
            f"PredictionMarketSentinel v2.0 initialized: "
            f"{len(self.topics)} topics | "
            f"min_liq=${self.min_liquidity:,} | "
            f"HWM decay={self.hwm_decay_hours}h"
        )

    def _load_state_cache(self):
        """Load state cache from StateManager for persistence across restarts."""
        try:
            # CRITICAL: Use load_state_raw() to avoid STALE string substitution
            # load_state() converts stale dicts to strings, which crashes
            # cached.get('slug') later in check()
            cached = StateManager.load_state_raw(namespace="prediction_market_state")
            if cached and isinstance(cached, dict):
                # Validate each entry is actually a dict (defense-in-depth)
                validated = {}
                for key, value in cached.items():
                    if isinstance(value, dict):
                        validated[key] = value
                    else:
                        logger.warning(
                            f"Skipping invalid state entry for '{key}': "
                            f"expected dict, got {type(value).__name__}"
                        )
                self.state_cache = validated
                logger.info(f"Loaded state for {len(self.state_cache)} prediction market topics")
        except Exception as e:
            logger.warning(f"Failed to load prediction market state: {e}")

    def _save_state_cache(self):
        """Persist state cache for restart resilience."""
        try:
            StateManager.save_state(self.state_cache, namespace="prediction_market_state")
        except Exception as e:
            logger.warning(f"Failed to save prediction market state: {e}")

    def _calculate_severity(self, delta_pct: float) -> int:
        """Map probability swing magnitude to severity level."""
        abs_delta = abs(delta_pct)
        if abs_delta >= 30:
            return self.severity_map.get('30_plus_pct', 9)
        elif abs_delta >= 20:
            return self.severity_map.get('20_to_30_pct', 7)
        else:
            return self.severity_map.get('10_to_20_pct', 6)

    def _should_decay_hwm(self, hwm_timestamp: Optional[str]) -> bool:
        """Check if High-Water Mark should decay (reset) based on time."""
        if not hwm_timestamp:
            return False

        try:
            hwm_time = datetime.fromisoformat(hwm_timestamp)
            age_hours = (datetime.now(timezone.utc) - hwm_time).total_seconds() / 3600
            return age_hours >= self.hwm_decay_hours
        except (ValueError, TypeError):
            return False

    async def _resolve_active_market(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        DYNAMIC MARKET DISCOVERY:
        Searches Polymarket for the query, filters by liquidity/volume,
        and returns the single most relevant ACTIVE market.

        This is the core mechanism that eliminates "slug rot" - we follow
        the liquidity, which naturally flows to the current active market.
        """
        params = {
            "q": query,
            "closed": "false",
            "active": "true",
            "limit": str(self.search_limit)
        }

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Polymarket search failed for '{query}': HTTP {response.status}"
                        )
                        return None

                    data = await response.json()

                    # Type guard: API should return a list of events
                    if not isinstance(data, list):
                        logger.warning(
                            f"Polymarket API returned {type(data).__name__} instead of list "
                            f"for query '{query}'"
                        )
                        return None

                    logger.debug(
                        f"Dynamic Discovery: '{query}' found {len(data)} markets. "
                        f"Top result: {[e.get('title') for e in data[:1] if isinstance(e, dict)]}"
                    )

                    if not data:
                        logger.debug(f"No markets found for query: '{query}'")
                        return None

                    # Filter and sort candidates by liquidity
                    candidates = []
                    for event in data:
                        if not isinstance(event, dict):
                            logger.debug(f"Skipping non-dict event: {type(event).__name__}")
                            continue

                        markets = event.get('markets', [])
                        if not markets:
                            continue

                        # Primary market (usually the main Yes/No question)
                        # NOTE: markets[0] heuristic is intentional per v2.0 design review.
                        # 95% of Polymarket events have the primary binary at index 0.
                        # If we find "Side Bet" lock-on issues later, add market['question'] filter.
                        market = markets[0]

                        if not isinstance(market, dict):
                            logger.debug(f"Skipping non-dict market: {type(market).__name__}")
                            continue

                        # Parse liquidity and volume safely
                        try:
                            liquidity = float(market.get('liquidity', 0) or 0)
                            volume = float(market.get('volume', 0) or 0)
                        except (ValueError, TypeError, AttributeError):
                            logger.warning(
                                f"Skipping malformed market data for '{query}': "
                                f"type={type(market).__name__}"
                            )
                            continue

                        # Apply filters
                        if liquidity < self.min_liquidity:
                            logger.debug(
                                f"Skipping '{event.get('slug')}': "
                                f"Low liquidity (${liquidity:,.0f} < ${self.min_liquidity:,})"
                            )
                            continue

                        if volume < self.min_volume:
                            logger.debug(
                                f"Skipping '{event.get('slug')}': "
                                f"Low volume (${volume:,.0f} < ${self.min_volume:,})"
                            )
                            continue

                        candidates.append({
                            'slug': event.get('slug'),
                            'title': event.get('title', ''),
                            'market': market,
                            'liquidity': liquidity,
                            'volume': volume
                        })

                    if not candidates:
                        return None

                    # --- RELEVANCE SCORING ---
                    # Get keywords from topic config (passed via new parameter)
                    relevance_keywords = kwargs.get('relevance_keywords', [])
                    min_relevance = kwargs.get('min_relevance_score', 1)

                    if relevance_keywords:
                        for candidate in candidates:
                            title_lower = candidate['title'].lower()
                            # Count keyword matches in event title
                            match_count = sum(
                                1 for kw in relevance_keywords
                                if kw.lower() in title_lower
                            )
                            candidate['relevance_score'] = match_count

                        # Prefer relevant candidates (must meet minimum score)
                        relevant = [
                            c for c in candidates
                            if c.get('relevance_score', 0) >= min_relevance
                        ]

                        if relevant:
                            # Among relevant: pick highest liquidity
                            relevant.sort(key=lambda x: x['liquidity'], reverse=True)
                            winner = relevant[0]
                            logger.debug(
                                f"Resolved '{query}' → '{winner['slug']}' "
                                f"(relevance={winner['relevance_score']}, "
                                f"liq=${winner['liquidity']:,.0f})"
                            )
                        else:
                            # FAIL-SAFE: No relevant candidates → return None
                            # This prevents tracking irrelevant markets (e.g., deportation
                            # market when searching for "Federal Reserve interest rate")
                            logger.warning(
                                f"⚠️ No relevant markets for '{query}' "
                                f"(keywords: {relevance_keywords[:5]}, "
                                f"min_score={min_relevance}). "
                                f"Skipping topic — will retry next cycle."
                            )
                            return None
                    else:
                        # No keywords configured — fall back to pure liquidity
                        candidates.sort(key=lambda x: x['liquidity'], reverse=True)
                        winner = candidates[0]

                    return winner

        except asyncio.TimeoutError:
            logger.warning(f"Polymarket API timeout for query: '{query}'")
        except aiohttp.ClientError as e:
            logger.warning(f"Polymarket network error for '{query}': {e}")
        except Exception as e:
            logger.error(f"Polymarket fetch error for '{query}': {e}")

        return None

    async def check(self) -> Optional[SentinelTrigger]:
        """
        Check prediction markets for significant probability swings.

        OPERATIONAL SAFEGUARDS:
        1. Dynamic Discovery: Auto-finds current active market per topic
        2. Slug Consistency: Resets baseline on market rollover
        3. High-Water Mark: Prevents alert flapping
        4. HWM Decay: Allows re-alerting after 24h
        """
        if not self.sentinel_config.get('enabled', True):
            return None

        if not self.topics:
            return None

        # Internal rate limiting
        import time as time_module
        now = time_module.time()
        if (now - self._last_poll_time) < self.poll_interval:
            return None

        # --- NON-TRADING-DAY FREQUENCY REDUCTION ---
        from trading_bot.utils import is_market_open, is_trading_day

        if not is_trading_day():
            # Weekend/holiday: check every 2 hours instead of every 5 minutes
            if not hasattr(self, '_last_non_trading_check'):
                self._last_non_trading_check = 0
            if (now - self._last_non_trading_check) < 7200:  # 2 hours
                return None
            self._last_non_trading_check = now
            logger.debug("Non-trading day: Running reduced-frequency prediction market check")
        elif not is_market_open():
            # Trading day but outside hours: check every 30 minutes
            if not hasattr(self, '_last_closed_market_check_pm'):
                self._last_closed_market_check_pm = 0
            if (now - self._last_closed_market_check_pm) < 1800:  # 30 min
                return None
            self._last_closed_market_check_pm = now

        self._last_poll_time = now

        triggers = []

        for topic in self.topics:
            query = topic.get('query')
            if not query:
                continue

            try:
                threshold = topic.get('trigger_threshold_pct', 10.0) / 100.0
                display_name = topic.get('display_name', query)
                tag = topic.get('tag', 'Unknown')
                importance = topic.get('importance', 'macro')
                commodity_impact = topic.get('commodity_impact',
                    topic.get('coffee_impact', 'Potential macro impact on commodity'))

                # === DYNAMIC MARKET RESOLUTION ===
                relevance_keywords = topic.get('relevance_keywords', [])
                min_relevance = topic.get('min_relevance_score',
                    self.sentinel_config.get('min_relevance_score', 1))
                market_data = await self._resolve_active_market(
                    query,
                    relevance_keywords=relevance_keywords,
                    min_relevance_score=min_relevance
                )

                if not market_data:
                    # Track persistent failures for alerting
                    self._topic_failure_counts[query] = self._topic_failure_counts.get(query, 0) + 1
                    if self._topic_failure_counts[query] == 10:  # Alert after ~50 min of failures
                        logger.warning(
                            f"⚠️ No markets found for topic '{display_name}' "
                            f"after {self._topic_failure_counts[query]} attempts. "
                            f"Consider reviewing query: '{query}'"
                        )
                    continue

                # Reset failure count on success
                self._topic_failure_counts[query] = 0

                current_slug = market_data['slug']
                current_title = market_data['title']

                # Parse price safely
                try:
                    outcomes = market_data['market'].get('outcomePrices', [])
                    if isinstance(outcomes, str):
                        try:
                            outcomes = json.loads(outcomes)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse outcomePrices JSON for '{query}'")
                            continue

                    if not outcomes:
                        continue
                    current_price = float(outcomes[0])  # Index 0 is typically "Yes"
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to parse price for '{query}': {e}")
                    continue

                # === COMPARE AGAINST CACHED STATE ===
                if query in self.state_cache:
                    cached = self.state_cache[query]
                    last_slug = cached.get('slug')
                    last_price = cached.get('price', current_price)
                    severity_hwm = cached.get('severity_hwm', 0)
                    hwm_timestamp = cached.get('hwm_timestamp')

                    # === SLUG CONSISTENCY CHECK ===
                    # If the resolved market changed (June → July rollover),
                    # reset the baseline and DO NOT trigger an alert
                    if last_slug and last_slug != current_slug:
                        logger.info(
                            f"📅 Market Rollover Detected for '{display_name}': "
                            f"'{last_slug}' → '{current_slug}'. Resetting baseline."
                        )

                        # Notify (informational, not alert)
                        send_pushover_notification(
                            self.config.get('notifications', {}),
                            f"Market Rollover: {display_name}",
                            f"Now tracking: {current_title[:50]}...",
                            priority=-1  # Low priority (informational)
                        )

                        # Reset state for this topic
                        self.state_cache[query] = {
                            'slug': current_slug,
                            'title': current_title,
                            'price': current_price,
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'severity_hwm': 0,
                            'hwm_timestamp': None
                        }
                        continue  # Skip trigger evaluation this cycle

                    # === HIGH-WATER MARK DECAY ===
                    # Reset HWM if enough time has passed (allows re-alerting)
                    if self._should_decay_hwm(hwm_timestamp):
                        logger.debug(
                            f"HWM decay for '{display_name}': "
                            f"Resetting from {severity_hwm} to 0 (>{self.hwm_decay_hours}h old)"
                        )
                        severity_hwm = 0
                        cached['severity_hwm'] = 0
                        cached['hwm_timestamp'] = None

                    # === RATE OF CHANGE DETECTION ===
                    delta = current_price - last_price
                    delta_pct = abs(delta) * 100

                    if abs(delta) >= threshold:
                        current_severity = self._calculate_severity(delta_pct)

                        # === HIGH-WATER MARK FLAPPING PROTECTION ===
                        # Only alert if severity INCREASES beyond the high-water mark
                        # This prevents: 15% alert → 21% alert → 19% RE-ALERT (bad)
                        if current_severity > severity_hwm:
                            direction = "JUMPED" if delta > 0 else "CRASHED"

                            msg = (
                                f"Prediction Market Alert: '{display_name}' {direction} "
                                f"{delta_pct:.1f}% (Now: {current_price*100:.0f}% → "
                                f"Was: {last_price*100:.0f}%)"
                            )

                            logger.warning(f"🎯 PREDICTION SENTINEL: {msg}")

                            triggers.append(SentinelTrigger(
                                source="PredictionMarketSentinel",
                                reason=msg,
                                payload={
                                    "topic": query,
                                    "tag": tag,
                                    "display_name": display_name,
                                    "slug": current_slug,
                                    "title": current_title,
                                    "prev_price": round(last_price, 4),
                                    "curr_price": round(current_price, 4),
                                    "delta_pct": round(delta * 100, 2),
                                    "importance": importance,
                                    "commodity_impact": commodity_impact,
                                    "volume": market_data.get('volume', 0),
                                    "liquidity": market_data.get('liquidity', 0)
                                },
                                severity=current_severity
                            ))

                            # Update High-Water Mark
                            cached['severity_hwm'] = current_severity
                            cached['hwm_timestamp'] = datetime.now(timezone.utc).isoformat()
                        else:
                            logger.debug(
                                f"Suppressed alert for '{display_name}': "
                                f"severity {current_severity} <= HWM {severity_hwm}"
                            )

                # === UPDATE STATE CACHE ===
                if query not in self.state_cache:
                    self.state_cache[query] = {
                        'severity_hwm': 0,
                        'hwm_timestamp': None
                    }

                self.state_cache[query].update({
                    'slug': current_slug,
                    'title': current_title,
                    'price': current_price,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })

            except Exception as topic_error:
                logger.warning(
                    f"Error processing prediction market topic '{query}': "
                    f"{type(topic_error).__name__}: {topic_error}"
                )
                self._topic_failure_counts[query] = self._topic_failure_counts.get(query, 0) + 1
                continue

        # --- DUPLICATE SLUG DETECTION ---
        # Alert if multiple topics resolve to the same market (indicates query failure)
        slug_map = {}
        for topic in self.topics:
            query = topic.get('query', '')
            cached = self.state_cache.get(query, {})
            slug = cached.get('slug') if isinstance(cached, dict) else None
            if slug:
                slug_map.setdefault(slug, []).append(topic.get('tag', query))

        duplicates = {slug: tags for slug, tags in slug_map.items() if len(tags) > 1}
        if duplicates:
            collision_details = []
            for slug, tags in duplicates.items():
                detail = f"{', '.join(tags)} → '{slug}'"
                collision_details.append(detail)
                logger.warning(
                    f"⚠️ TOPIC COLLISION: {', '.join(tags)} all resolved to "
                    f"'{slug}'. Dynamic Discovery is failing to differentiate topics."
                )

            # Alert operator on first detection (then suppress for 6 hours)
            if not hasattr(self, '_last_collision_alert_time'):
                self._last_collision_alert_time = 0

            import time as time_module
            now = time_module.time()
            if (now - self._last_collision_alert_time) > 21600:  # 6 hours
                send_pushover_notification(
                    self.config.get('notifications', {}),
                    "🔮 Prediction Market Topic Collision",
                    f"{len(duplicates)} collision(s) detected:\n"
                    + "\n".join(collision_details)
                    + "\n\nCheck config.json topics_to_watch queries.",
                    priority=0  # Normal priority
                )
                self._last_collision_alert_time = now

        # Persist state
        self._save_state_cache()

        # --- CYCLE SUMMARY (single INFO log replaces per-topic spam) ---
        active_topics = [
            topic.get('tag', topic.get('query', '?'))
            for topic in self.topics
            if topic.get('query') in self.state_cache
            and self.state_cache[topic['query']].get('slug')
        ]
        stale_topics = [
            topic.get('tag', topic.get('query', '?'))
            for topic in self.topics
            if topic.get('query') not in self.state_cache
            or not self.state_cache[topic['query']].get('slug')
        ]

        if active_topics or stale_topics:
            logger.info(
                f"PredictionMarket: {len(active_topics)} tracking, "
                f"{len(stale_topics)} unresolved | "
                f"Active: {', '.join(active_topics) if active_topics else 'none'}"
            )

        # Return the most severe trigger (if multiple)
        if triggers:
            triggers.sort(key=lambda t: t.severity, reverse=True)
            return triggers[0]

        return None
