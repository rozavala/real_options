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
from functools import wraps
from notifications import send_pushover_notification
from trading_bot.state_manager import StateManager

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
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('weather', {})
        self.api_url = self.sentinel_config.get('api_url', "https://api.open-meteo.com/v1/forecast")
        self.params = self.sentinel_config.get('params', "daily=temperature_2m_min,precipitation_sum&timezone=auto&forecast_days=10")
        self.locations = self.sentinel_config.get('locations', [])
        self.triggers = self.sentinel_config.get('triggers', {})

        self._active_alerts: Dict[str, dict] = {}  # {location_key: {"time": dt, "value": float}}
        self._alert_cooldown_hours = 24
        self._escalation_threshold = 0.05  # 5% worsening breaks cooldown

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
                        msg = f"Frost Risk in {name}: Min Temp {min_temp}°C < {frost_threshold}°C"
                        logger.warning(f"WEATHER SENTINEL TRIGGERED: {msg}")
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
                        msg = f"Drought Risk in {name}: {low_rain_days} days with < {rain_limit}mm rain"
                        logger.warning(f"WEATHER SENTINEL TRIGGERED: {msg}")
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

        # ESCALATION CHECK: If situation worsened by >5%, break cooldown
        # current_value > prev_value implies worsening (assuming metric is "badness")
        prev_value = prev["value"]

        # Avoid division by zero or negative base (metric should be positive severity)
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
            logger.warning(f"LOGISTICS SENTINEL TRIGGERED: {msg}")
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

        data = await self._analyze_with_ai(prompt)

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
            logger.warning(f"NEWS SENTINEL TRIGGERED: {msg}")
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

    def _update_volume_stats(self, new_volume: int):
        """Update rolling volume statistics for spike detection."""
        self.post_volume_history.append(new_volume)
        # Keep last 30 checks (~15 hours at 30min intervals)
        self.post_volume_history = self.post_volume_history[-30:]

        if len(self.post_volume_history) >= 5:
            # statistics imported at top
            self.volume_mean = statistics.mean(self.post_volume_history)
            self.volume_stddev = statistics.stdev(self.post_volume_history) if len(self.post_volume_history) > 1 else 0

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
        Execute X API search locally.

        Uses X API v2 recent search endpoint.
        Requires X_BEARER_TOKEN in environment.

        IMPORTANT: X API uses sort_order with values:
        - "recency" = Latest/chronological (newest first)
        - "relevancy" = Top/algorithmically relevant
        Default to "recency" for time-sensitive commodity sentiment.
        """
        if not self.x_bearer_token:
            return {
                "error": "X API not configured",
                "posts": [],
                "post_volume": 0
            }

        query = args.get("query", "coffee futures")
        limit = args.get("limit", 10)

        # === CRITICAL FIX: Correct sort_order mapping ===
        # Grok may send "Top" or "Latest" via the tool schema
        # Map to X API's actual parameter values
        mode = args.get("mode", "Latest")  # Default to Latest for freshness
        if mode == "Latest":
            sort_order = "recency"
        elif mode == "Top":
            sort_order = "relevancy"
        else:
            sort_order = "recency"  # Default to recency for time-sensitive sentiment

        headers = {
            "Authorization": f"Bearer {self.x_bearer_token}",
            "Content-Type": "application/json"
        }

        params = {
            "query": f"{query} -is:retweet lang:en",  # Filter retweets, English only
            "max_results": max(10, min(limit, 100)),  # X API requires 10 ≤ max_results ≤ 100
            "tweet.fields": "text,created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username",
            "sort_order": sort_order  # Always include sort_order
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
                        return {"error": "Rate limit exceeded", "posts": [], "post_volume": 0}

                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"X API error {response.status}: {error_text}")
                        return {"error": f"X API error: {response.status}", "posts": [], "post_volume": 0}

                    data = await response.json()

                    posts = []
                    users = {}

                    # Build user lookup
                    if "includes" in data and "users" in data["includes"]:
                        for user in data["includes"]["users"]:
                            users[user["id"]] = user.get("username", "unknown")

                    # Process tweets
                    for tweet in data.get("data", []):
                        metrics = tweet.get("public_metrics", {})
                        posts.append({
                            "text": tweet["text"][:400],  # Increased from 200 for better context
                            "author": users.get(tweet.get("author_id"), "unknown"),
                            "likes": metrics.get("like_count", 0),
                            "retweets": metrics.get("retweet_count", 0),
                            "created_at": tweet.get("created_at", "")
                        })

                # Filter by engagement (replaces min_faves operator)
                min_engagement = self.min_engagement
                posts = [p for p in posts if p.get('likes', 0) >= min_engagement]

                logger.info(f"X API returned {len(posts)} posts for '{query}' (sort: {sort_order})")

                return {
                    "posts": posts,
                    "post_volume": len(posts),
                    "query": query
                }

        except asyncio.TimeoutError:
            logger.error("X API request timed out")
            return {"error": "Request timeout", "posts": [], "post_volume": 0}
        except Exception as e:
            logger.error(f"X API request failed: {e}")
            return {"error": str(e), "posts": [], "post_volume": 0}

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
        tasks = [self._search_x_and_analyze(query) for query in self.search_queries]
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
