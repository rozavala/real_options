import asyncio
import logging
import feedparser
import requests
from datetime import datetime, time, timezone
import numpy as np
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types
import pytz
import aiohttp
import json
from functools import wraps
from notifications import send_pushover_notification

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
    def __init__(self, source: str, reason: str, payload: Dict[str, Any] = None):
        self.source = source
        self.reason = reason
        self.payload = payload or {}
        self.timestamp = datetime.now(timezone.utc)

    def __repr__(self):
        return f"SentinelTrigger(source='{self.source}', reason='{self.reason}')"

class Sentinel:
    """Base class for all sentinels."""
    def __init__(self, config: dict):
        self.config = config
        self.enabled = True
        self.last_triggered = 0 # Timestamp of last trigger

    async def check(self) -> Optional[SentinelTrigger]:
        """Performs the sentinel check. Returns a SentinelTrigger if fired, else None."""
        raise NotImplementedError

    async def _fetch_rss_safe(self, url: str, seen_cache: set, timeout: int = 10) -> List[str]:
        """Fetch RSS with timeout and validation using aiohttp."""
        headlines = []
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

            for entry in feed.entries[:5]:
                # Use link as unique ID
                if entry.link not in seen_cache:
                    headlines.append(entry.title)
                    seen_cache.add(entry.link)

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

    async def check(self) -> Optional[SentinelTrigger]:
        import time as time_module
        now = time_module.time()

        # Cooldown Check (1 hour = 3600s)
        if (now - self.last_triggered) < 3600:
            return None

        # Market Hours Check: 9:00 - 17:00 EST, Mon-Fri
        est = pytz.timezone('US/Eastern')
        now_est = datetime.now(est)

        if now_est.weekday() >= 5: # Sat(5) or Sun(6)
            return None

        market_start = now_est.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now_est.replace(hour=17, minute=0, second=0, microsecond=0)

        if not (market_start <= now_est <= market_end):
            return None

        try:
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
                    msg = f"Frost Risk in {name}: Min Temp {min_temp}°C < {frost_threshold}°C"
                    logger.warning(f"WEATHER SENTINEL TRIGGERED: {msg}")
                    return SentinelTrigger("WeatherSentinel", msg, {"location": name, "type": "FROST", "value": min_temp})

                # Check Drought
                drought_days_limit = self.triggers.get('drought_days', 10)
                rain_limit = self.triggers.get('drought_rain_mm', 5.0)

                low_rain_days = sum(1 for p in precip if p < rain_limit)

                if low_rain_days >= drought_days_limit:
                    msg = f"Drought Risk in {name}: {low_rain_days} days with < {rain_limit}mm rain"
                    logger.warning(f"WEATHER SENTINEL TRIGGERED: {msg}")
                    return SentinelTrigger("WeatherSentinel", msg, {"location": name, "type": "DROUGHT", "days": low_rain_days})

            except Exception as e:
                logger.error(f"Weather Sentinel failed for {loc.get('name')}: {e}")

        return None

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

        # Deduplication Cache (In-memory)
        self.seen_links = set()

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
            new_titles = await self._fetch_rss_safe(url, self.seen_links)
            headlines.extend(new_titles)

        if not headlines:
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
            return SentinelTrigger("LogisticsSentinel", msg, {"headlines": headlines[:3]})

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

        # Deduplication Cache (In-memory)
        self.seen_links = set()

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
            new_titles = await self._fetch_rss_safe(url, self.seen_links)
            headlines.extend(new_titles)

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
            return SentinelTrigger("NewsSentinel", msg, {"score": score, "summary": data.get('summary')})

        return None
