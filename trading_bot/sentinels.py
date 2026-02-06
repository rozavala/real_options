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
from config.commodity_profiles import get_commodity_profile, GrowingRegion

logger = logging.getLogger(__name__)

# Domain whitelists for prediction market filtering.
# IMPORTANT: Keep terms specific. Avoid generic words that appear in unrelated titles.
# Commodity-agnostic: each commodity profile can extend these via config.
DOMAIN_KEYWORD_WHITELISTS = {
    'coffee': [
        'coffee', 'arabica', 'robusta', 'kc', 'bean', 'starbucks',
        'roast', 'harvest', 'crop'
    ],
    'macro': [
        'fed', 'rate', 'inflation', 'cpi', 'fomc', 'powell',
        'yield', 'treasury', 'dollar', 'dxy', 'recession', 'gdp',
        'employment', 'jobs', 'economy', 'interest rate',
        'rate cut', 'rate hike', 'basis points', 'monetary'
    ],
    'geopolitical': [
        'war', 'conflict', 'tariff', 'sanction', 'trade war',
        'embargo', 'invasion', 'nato', 'military',
        'nuclear', 'ceasefire', 'peace deal'
    ],
    'weather': [
        'rain', 'drought', 'frost', 'el nino', 'la nina',
        'monsoon', 'hurricane', 'cyclone', 'typhoon',
        'precipitation', 'temperature'
    ],
    'logistics': [
        'port', 'shipping', 'container', 'freight', 'strike',
        'rail', 'supply chain', 'suez', 'panama canal'
    ]
}

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
        self._seen_timestamps = {}  # link -> first_seen_timestamp
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

                    # Handle legacy list format if encountered
                    if isinstance(data, list):
                        now = time_module.time()
                        self._seen_timestamps = {k: now for k in data}
                    else:
                        self._seen_timestamps = {k: v for k, v in data.items() if v > cutoff}

                    return set(self._seen_timestamps.keys())
            except Exception as e:
                logger.warning(f"Failed to load seen cache: {e}")
        return set()

    def _save_seen_cache(self):
        """Save seen links to disk with timestamps."""
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            import time as time_module
            now = time_module.time()

            # Rebuild dict ensuring only current _seen_links are saved,
            # preserving timestamps for existing ones.
            data_to_save = {}
            for link in self._seen_links:
                data_to_save[link] = self._seen_timestamps.get(link, now)

            # Sync internal map
            self._seen_timestamps = data_to_save

            with open(self._cache_file, 'w') as f:
                json.dump(data_to_save, f)
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

        # Load commodity profile (NEW)
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        self.profile = get_commodity_profile(ticker)

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

    async def _fetch_weather(self, region: GrowingRegion) -> List[Dict]:
        """Fetch weather data for a region."""
        try:
            url = f"{self.api_url}?latitude={region.latitude}&longitude={region.longitude}&{self.params}"

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, requests.get, url)
            data = response.json()

            if 'daily' not in data:
                return []

            # Map daily arrays to list of day objects
            daily = data['daily']
            result = []
            for i in range(len(daily.get('time', []))):
                result.append({
                    'time': daily.get('time')[i],
                    'precipitation_mm': daily.get('precipitation_sum')[i],
                    'min_temp_c': daily.get('temperature_2m_min')[i]
                })
            return result

        except Exception as e:
            logger.error(f"Weather fetch failed for {region.name}: {e}")
            return []

    def check_region_weather(self, region: GrowingRegion) -> Optional[Dict]:
        """
        Check weather for a single growing region.

        Flight Director: "Hardcoding agronomic calendar safer than dynamic inference" ✓

        Returns:
        - None if normal conditions
        - Dict with alert details if abnormal
        """
        # Note: _fetch_weather is async, but this method is called inside async check()
        # We'll need to redesign flow slightly since this logic was presented as synchronous in prompt.
        # But we can't await inside a non-async function if it calls async.
        # So we will make this method take the DATA, not fetch it.
        # OR make this method async. I'll make it async.
        raise NotImplementedError("Use async_check_region_weather")

    async def async_check_region_weather(self, region: GrowingRegion) -> Optional[Dict]:
        from datetime import datetime

        try:
            # Fetch weather data
            weather_data = await self._fetch_weather(region)
            if not weather_data:
                return None

            # Calculate rolling 7-day precipitation (using last 7 days of forecast/history)
            # API returns forecast_days=10. Let's look at the first 7 days (including today/forecast).
            weekly_precip = sum(day.get('precipitation_mm', 0) for day in weather_data[:7])

            # Get current month to determine agronomic stage
            current_month = datetime.now().month

            # Check Frost (if threshold defined)
            if region.frost_threshold_celsius is not None:
                min_temp_week = min(day.get('min_temp_c', 99) for day in weather_data[:7])
                if min_temp_week < region.frost_threshold_celsius:
                     return {
                        "type": "FROST",
                        "region": region.name,
                        "min_temp_c": min_temp_week,
                        "threshold": region.frost_threshold_celsius,
                        "stage": "ANY",
                        "direction": "BULLISH", # Frost is always Bullish
                        "severity": "CRITICAL"
                     }

            if current_month in region.flowering_months:
                stage = "FLOWERING"
            elif current_month in region.harvest_months:
                stage = "HARVEST"
            elif current_month in region.bean_filling_months:
                stage = "BEAN_FILLING"
            else:
                stage = "VEGETATIVE"

            # Drought Detection
            if weekly_precip < region.drought_threshold_mm:
                return {
                    "type": "DROUGHT",
                    "region": region.name,
                    "weekly_precip_mm": weekly_precip,
                    "threshold": region.drought_threshold_mm,
                    "stage": stage,
                    "direction": self._determine_drought_direction(stage),
                    "severity": "CRITICAL" if stage == "FLOWERING" else "HIGH"
                }

            # Flood Detection (NEW)
            if weekly_precip > region.flood_threshold_mm:
                return {
                    "type": "FLOOD",
                    "region": region.name,
                    "weekly_precip_mm": weekly_precip,
                    "threshold": region.flood_threshold_mm,
                    "stage": stage,
                    "direction": self._determine_flood_direction(stage),
                    "severity": "CRITICAL" if stage == "HARVEST" else "MODERATE"
                }

            # Goldilocks Detection (>150% of historical = relief from prior drought)
            if weekly_precip > region.historical_weekly_precip_mm * 1.5:
                return {
                    "type": "GOLDILOCKS_RAIN",
                    "region": region.name,
                    "weekly_precip_mm": weekly_precip,
                    "baseline": region.historical_weekly_precip_mm,
                    "stage": stage,
                    "direction": "BEARISH",  # Removes supply fear premium
                    "severity": "MODERATE"
                }

            return None

        except Exception as e:
            self.logger.error(f"Error checking weather for {region.name}: {e}")
            return None

    def _determine_drought_direction(self, stage: str) -> str:
        """
        Determine if drought is BULLISH or BEARISH based on agronomic stage.

        Flight Director: "Correct implementation" ✓
        """
        if stage == "FLOWERING":
            return "BULLISH"  # Drought during flowering = yield loss
        elif stage == "BEAN_FILLING":
            return "BULLISH"  # Drought during bean-filling = smaller beans
        elif stage == "HARVEST":
            return "BEARISH"  # Drought during harvest = ideal conditions
        else:
            return "NEUTRAL"

    def _determine_flood_direction(self, stage: str) -> str:
        """
        Determine if heavy rain is BULLISH or BEARISH based on agronomic stage.
        """
        if stage == "HARVEST":
            return "BULLISH"  # Rain during harvest = delays, quality loss
        elif stage == "FLOWERING":
            return "BEARISH"  # Excessive rain during flowering = too much water
        else:
            return "NEUTRAL"

    async def check(self) -> Optional[SentinelTrigger]:
        if not self.profile:
            return None

        for region in self.profile.primary_regions:
            try:
                alert = await self.async_check_region_weather(region)

                if alert:
                    # Alert Cooldown Logic
                    alert_key = f"{alert['type']}_{region.name}"
                    current_value = alert.get('weekly_precip_mm', 0)

                    # Calculate a numeric severity for internal logic
                    severity_val = 0
                    if alert['type'] == 'DROUGHT':
                        severity_val = alert['threshold'] - current_value
                    elif alert['type'] == 'FLOOD':
                        severity_val = current_value - alert['threshold']

                    if self._should_alert(alert_key, severity_val):
                        self._active_alerts[alert_key] = {
                            "time": datetime.now(timezone.utc),
                            "value": severity_val
                        }
                        self._save_alert_state()

                        msg = f"{alert['type']} in {alert['region']}: {alert.get('weekly_precip_mm',0):.1f}mm ({alert.get('direction')}, {alert.get('stage')} stage)"
                        logger.warning(f"WEATHER SENTINEL DETECTED: {msg}")

                        # Map severity string to int
                        sev_int = 8 if "CRITICAL" in alert.get('severity','') else (6 if "HIGH" in alert.get('severity','') else 4)

                        return SentinelTrigger("WeatherSentinel", msg, alert, severity=sev_int)

            except Exception as e:
                logger.error(f"Weather Sentinel failed for {region.name}: {e}")

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
        if isinstance(prev['time'], str):
             prev['time'] = datetime.fromisoformat(prev['time'])
             if prev['time'].tzinfo is None: prev['time'] = prev['time'].replace(tzinfo=timezone.utc)

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
    CIRCUIT_BREAKER_THRESHOLD = 3   # Trip after 3 consecutive failures
    CIRCUIT_BREAKER_RESET_S = 1800  # Reset after 30 minutes

    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('logistics', {})

        api_key = config.get('gemini', {}).get('api_key')
        self.client = genai.Client(api_key=api_key)
        self.model = self.sentinel_config.get('model', "gemini-3-flash-preview")

        # Commodity-agnostic: load profile
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        self.profile = get_commodity_profile(ticker)

        # Commodity-agnostic RSS URL generation
        self.urls = self._build_rss_urls()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_tripped_until = 0

    def _build_rss_urls(self) -> List[str]:
        """
        Generate RSS search URLs from commodity profile.
        Falls back to config if hubs are empty.
        """
        config_urls = self.sentinel_config.get('rss_urls_override', [])
        if config_urls:
            return config_urls

        base = "https://news.google.com/rss/search?q="
        commodity_name = self.profile.name.lower().replace(' ', '+')

        urls = []

        # Monitor specific logistics hubs defined in profile
        for hub in self.profile.logistics_hubs:
            hub_name = hub.name.replace(' ', '+')
            urls.append(f"{base}{hub_name}+logistics+{commodity_name}")

        # General supply chain search
        urls.append(f"{base}{commodity_name}+supply+chain+bottlenecks")
        urls.append(f"{base}Red+Sea+Suez+{commodity_name}+shipping+delays")

        if not urls:
             # Fallback to legacy key
             return self.sentinel_config.get('rss_urls', [])

        logger.info(f"LogisticsSentinel: Generated {len(urls)} RSS URLs for {self.profile.name}")
        return urls

    @with_retry(max_attempts=3)
    async def _analyze_with_ai(self, prompt: str) -> Optional[str]:
        """AI analysis with retry logic."""
        await acquire_api_slot('gemini')  # Respect global rate limits

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip().upper()

    async def check(self) -> Optional[SentinelTrigger]:
        # Circuit breaker check
        import time as time_module
        if time_module.time() < self._circuit_tripped_until:
            logger.info("LogisticsSentinel: Circuit breaker active, skipping AI analysis")
            return None

        if self._circuit_tripped_until > 0 and time_module.time() >= self._circuit_tripped_until:
             self._circuit_tripped_until = 0
             self._consecutive_failures = 0
             logger.info("LogisticsSentinel: Circuit breaker reset")

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

        commodity_name = self.profile.name
        prompt = (
            f"Do these headlines suggest a disruption to the {commodity_name} supply chain? "
            f"Consider port strikes, shipping delays, export bans, or logistics failures "
            f"affecting {commodity_name} producing or consuming regions.\n"
            f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
            "Question: Is there a CRITICAL disruption mentioned? Answer only 'YES' or 'NO'."
        )

        answer = await self._analyze_with_ai(prompt)

        if answer is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_tripped_until = time_module.time() + self.CIRCUIT_BREAKER_RESET_S
                logger.warning(
                    f"LogisticsSentinel: Circuit breaker TRIPPED after {self._consecutive_failures} "
                    f"consecutive failures. Cooling down for {self.CIRCUIT_BREAKER_RESET_S}s"
                )
                send_pushover_notification(
                    self.config.get('notifications', {}),
                    "Sentinel Circuit Breaker",
                    f"LogisticsSentinel AI circuit breaker tripped. Will retry in 30 min."
                )
            return None

        # Success
        self._consecutive_failures = 0

        if "YES" in answer:
            msg = "Potential Supply Chain Disruption detected in headlines."
            logger.warning(f"LOGISTICS SENTINEL DETECTED: {msg}")
            return SentinelTrigger("LogisticsSentinel", msg, {"headlines": headlines[:3]}, severity=6)

        return None

class NewsSentinel(Sentinel):
    """
    Monitors broad market sentiment using RSS + Gemini Flash.
    Frequency: Every 2 Hours (24/7).
    """
    CIRCUIT_BREAKER_THRESHOLD = 3   # Trip after 3 consecutive failures
    CIRCUIT_BREAKER_RESET_S = 1800  # Reset after 30 minutes

    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('news', {})
        self.threshold = self.sentinel_config.get('sentiment_magnitude_threshold', 8)

        api_key = config.get('gemini', {}).get('api_key')
        self.client = genai.Client(api_key=api_key)
        self.model = self.sentinel_config.get('model', "gemini-3-flash-preview")

        # Commodity-agnostic: load profile for prompt construction
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        self.profile = get_commodity_profile(ticker)

        # Commodity-agnostic RSS URL generation
        self.urls = self._build_rss_urls()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_tripped_until = 0

    def _build_rss_urls(self) -> List[str]:
        """
        Generate RSS search URLs from commodity profile.
        Falls back to config if profile keywords are empty.
        """
        # Allow config override for custom feeds
        config_urls = self.sentinel_config.get('rss_urls_override', [])
        if config_urls:
            return config_urls

        base = "https://news.google.com/rss/search?q="
        commodity_name = self.profile.name.lower().replace(' ', '+')
        keywords = self.profile.news_keywords or [commodity_name]

        urls = []

        # Core market feeds (site-restricted for quality)
        for source in ['reuters.com', 'bloomberg.com']:
            primary_kw = keywords[0].replace(' ', '+')
            urls.append(f"{base}{primary_kw}+markets+site:{source}")

        # Region-specific feeds (top 2 producing regions)
        sorted_regions = sorted(self.profile.primary_regions, key=lambda r: r.production_share, reverse=True)
        top_regions = sorted_regions[:2]

        for region in top_regions:
            region_name = region.name.replace(' ', '+')
            urls.append(f"{base}{region_name}+{commodity_name}")

        # General sentiment feed
        primary_kw = keywords[0].replace(' ', '+')
        urls.append(f"{base}{primary_kw}+futures+market+sentiment")

        if not urls:
            # Absolute fallback to config
            urls = self.sentinel_config.get('rss_urls', [])
            logger.warning("NewsSentinel: No profile keywords, falling back to config RSS URLs")

        logger.info(f"NewsSentinel: Generated {len(urls)} RSS URLs for {self.profile.name}")
        return urls

    @with_retry(max_attempts=3)
    async def _analyze_with_ai(self, prompt: str) -> Optional[dict]:
        """AI analysis with retry logic."""
        await acquire_api_slot('gemini')  # Respect global rate limits

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[:-3]
        if text.endswith("```"): text = text[:-3]

        parsed = json.loads(text)

        # Unwrap single-element arrays (Gemini sometimes wraps in [...])
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            logger.debug("NewsSentinel: Unwrapped single-element list from AI response")
            parsed = parsed[0]

        return parsed

    async def check(self) -> Optional[SentinelTrigger]:
        # Circuit breaker check
        import time as time_module
        if time_module.time() < self._circuit_tripped_until:
            logger.info("NewsSentinel: Circuit breaker active, skipping AI analysis")
            return None

        # Reset if circuit was tripped and cooldown expired
        if self._circuit_tripped_until > 0 and time_module.time() >= self._circuit_tripped_until:
            self._circuit_tripped_until = 0
            self._consecutive_failures = 0
            logger.info("NewsSentinel: Circuit breaker reset")

        headlines = []
        for url in self.urls:
            new_titles = await self._fetch_rss_safe(url, self._seen_links, max_age_hours=48)
            headlines.extend(new_titles)

        # Save cache
        self._save_seen_cache()

        if not headlines:
            return None

        # === PAYLOAD DEDUPLICATION ===
        payload_preview = {"headlines": headlines[:3]}
        if self._is_duplicate_payload(payload_preview):
            return None

        commodity_name = self.profile.name
        prompt = (
            f"Analyze these headlines for EXTREME Market Sentiment regarding {commodity_name} Futures.\n"
            f"Headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
            f"Task: Score the 'Sentiment Magnitude' from 0 to 10 "
            f"(where 10 is Market Crashing or Exploding panic/euphoria) "
            f"specifically as it relates to {commodity_name} markets.\n"
            f"If headlines are unrelated to {commodity_name} or commodities, score 0.\n"
            f"Output JSON: {{'score': int, 'summary': string}}"
        )

        raw_response = await self._analyze_with_ai(prompt)

        if raw_response is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.CIRCUIT_BREAKER_THRESHOLD:
                self._circuit_tripped_until = time_module.time() + self.CIRCUIT_BREAKER_RESET_S
                logger.warning(
                    f"NewsSentinel: Circuit breaker TRIPPED after {self._consecutive_failures} "
                    f"consecutive failures. Cooling down for {self.CIRCUIT_BREAKER_RESET_S}s"
                )
                send_pushover_notification(
                    self.config.get('notifications', {}),
                    "Sentinel Circuit Breaker",
                    f"NewsSentinel AI circuit breaker tripped. Will retry in 30 min."
                )
            return None

        # Success — reset counter
        self._consecutive_failures = 0

        data = self._validate_ai_response(raw_response, context="headline sentiment")

        if data is None:
            # AI returned something, but wrong format — log only, don't alarm
            logger.warning(
                f"NewsSentinel: AI returned unparseable format ({type(raw_response).__name__}). "
                f"Skipping this cycle."
            )
            return None

        score = data.get('score', 0)
        if score >= self.threshold:
            # Normalize: LLM score 8-10 maps to system severity 6-8
            severity = min(8, max(6, int(score - 2)))
            msg = f"Extreme Sentiment Detected (Score: {score}/10): {data.get('summary')}"
            logger.warning(f"NEWS SENTINEL DETECTED: {msg}")
            return SentinelTrigger("NewsSentinel", msg, {"score": score, "summary": data.get('summary')}, severity=severity)

        return None

class XSentimentSentinel(Sentinel):
    """
    Monitors X (Twitter) for coffee market sentiment using xAI Grok.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('x_sentiment', {})

        # Commodity-agnostic: load profile for search queries and prompts
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        self.profile = get_commodity_profile(ticker)

        self.search_queries = self.sentinel_config.get('search_queries')
        if not self.search_queries:
            # Fallback to profile queries if not explicitly overridden in config
            self.search_queries = self.profile.sentiment_search_queries

        self.from_handles = self.sentinel_config.get('from_handles', [])
        self.exclude_keywords = self.sentinel_config.get('exclude_keywords',
            ['meme', 'joke', 'spam', 'giveaway'])

        self.sentiment_threshold = self.sentinel_config.get('sentiment_threshold', 6.5)
        self.min_engagement = self.sentinel_config.get('min_engagement', 5)
        self.volume_spike_multiplier = self.sentinel_config.get('volume_spike_multiplier', 2.0)

        api_key = config.get('xai', {}).get('api_key')
        if not api_key or api_key == "LOADED_FROM_ENV":
            api_key = os.environ.get('XAI_API_KEY')

        if not api_key:
             raise ValueError("XAI_API_KEY not found. Set in .env or config.json")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=180.0
        )

        self.x_bearer_token = config.get('x_api', {}).get('bearer_token')
        if not self.x_bearer_token or self.x_bearer_token == "LOADED_FROM_ENV":
            self.x_bearer_token = os.environ.get('X_BEARER_TOKEN')

        if not self.x_bearer_token:
            logger.warning("X_BEARER_TOKEN not found - X sentiment will use Grok's training knowledge only")

        self.model = self.sentinel_config.get('model', 'grok-4-1-fast-reasoning')

        self.post_volume_history = []
        self.volume_mean = 0.0
        self.volume_stddev = 0.0

        self.consecutive_failures = 0
        self.degraded_until: Optional[datetime] = None
        self.sensor_status = "ONLINE"

        self._request_interval = 1.5
        self._last_request_time = 0

        logger.info(f"XSentimentSentinel initialized with model: {self.model}")

    def get_sensor_status(self) -> dict:
        return {
            "sentiment_sensor_status": self.sensor_status,
            "consecutive_failures": self.consecutive_failures,
            "degraded_until": self.degraded_until.isoformat() if self.degraded_until else None
        }

    def _build_search_query(self, base_query: str) -> str:
        query_parts = [base_query]
        if self.from_handles:
            handle_filter = " OR ".join([f"from:{h.lstrip('@')}" for h in self.from_handles])
            query_parts.append(f"({handle_filter})")
        for keyword in self.exclude_keywords:
            query_parts.append(f"-{keyword}")
        return " ".join(query_parts)

    def _build_broad_search_query(self, base_query: str) -> str:
        query_parts = [base_query]
        for keyword in self.exclude_keywords:
            query_parts.append(f"-{keyword}")
        full_query = " ".join(query_parts)
        if len(full_query) > 450:
            logger.warning(f"Broad query too long ({len(full_query)} chars), trimming exclusions")
            full_query = base_query
        return full_query

    async def _fetch_x_posts(self, query: str, limit: int, sort_order: str, min_likes: int = None) -> list:
        from datetime import timedelta
        headers = {
            "Authorization": f"Bearer {self.x_bearer_token}",
            "Content-Type": "application/json"
        }
        query_with_filters = f"{query} -is:retweet lang:en"
        since_datetime = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "query": query_with_filters,
            "start_time": since_datetime,
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
                    likes_threshold = min_likes if min_likes is not None else self.min_engagement
                    posts = [p for p in posts if p.get('likes', 0) >= likes_threshold]
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
        self.post_volume_history.append(new_volume)
        self.post_volume_history = self.post_volume_history[-30:]
        if len(self.post_volume_history) >= 5:
            self.volume_mean = statistics.mean(self.post_volume_history)
            self.volume_stddev = statistics.stdev(self.post_volume_history) if len(self.post_volume_history) > 1 else 0

    async def _sem_bound_search(self, query: str) -> Optional[dict]:
        slot_acquired = await acquire_api_slot('xai', timeout=30.0)
        if not slot_acquired:
            logger.warning(f"Rate limit exceeded for X Sentinel query: {query}")
            return None
        jitter = np.random.uniform(0.5, 2.0)
        await asyncio.sleep(jitter)
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
                logger.warning(f"XSentimentSentinel: Server error. Consecutive failures: {self.consecutive_failures}")
                if self.consecutive_failures >= 3:
                    self.degraded_until = datetime.now(timezone.utc) + timedelta(hours=1)
                    self.sensor_status = "OFFLINE"
                    logger.error(f"XSentimentSentinel: CIRCUIT BREAKER TRIGGERED. Degraded until {self.degraded_until.isoformat()}")
                    StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")
            else:
                self.consecutive_failures += 1
            raise

    @with_retry(max_attempts=3)
    async def _search_x_and_analyze(self, query: str) -> Optional[dict]:
        if not self.client:
            logger.error("XAI Client not initialized (missing API key)")
            return None
        search_query = self._build_search_query(query)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "x_search",
                    "description": "Search X (Twitter) for posts related to a query to gather real-time sentiment data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "limit": {"type": "integer", "description": "Number of posts to return."},
                            "mode": {"type": "string", "enum": ["Top", "Latest"], "description": "Sort preference."}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        commodity_name = self.profile.name
        system_prompt = f"""You are an expert commodities market sentiment analyst.
Use the x_search tool to fetch live X data when needed.
Analyze posts for bullish/bearish themes related to {commodity_name} futures.
IMPORTANT: Prioritize RECENT posts. Use mode="Latest".
After analyzing posts, provide a JSON response with:
- sentiment_score: 0-10
- confidence: 0.0-1.0
- summary: Executive summary
- post_volume: Number of relevant posts
- key_themes: Top 3 themes
- notable_posts: Up to 3 notable posts
If the x_search tool returns no results, provide neutral sentiment with low confidence."""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Analyze X sentiment for: {search_query}"}]
        max_iterations = 5
        iteration = 0
        try:
            while iteration < max_iterations:
                iteration += 1
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if self.x_bearer_token else None,
                    tool_choice="auto" if self.x_bearer_token else None,
                    temperature=0.3,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                message = response.choices[0].message
                if message.content and message.content.strip():
                    content = message.content.strip()
                    if content.startswith("```json"): content = content[7:]
                    if content.startswith("```"): content = content[3:]
                    if content.endswith("```"): content = content[:-3]
                    content = content.strip()
                    try:
                        data = json.loads(content)
                        required_fields = ['sentiment_score', 'confidence', 'summary', 'post_volume']
                        missing = [f for f in required_fields if f not in data]
                        if missing:
                            data.setdefault('sentiment_score', 5.0)
                            data.setdefault('confidence', 0.0)
                            data.setdefault('summary', 'Incomplete response')
                            data.setdefault('post_volume', 0)
                        data['sentiment_score'] = float(data['sentiment_score'])
                        data['confidence'] = float(data['confidence'])
                        data['post_volume'] = int(data['post_volume'])
                        if data['post_volume'] > 0: self._update_volume_stats(data['post_volume'])
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON from Grok: {e}")
                        return None
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {"query": query}
                    if function_name == "x_search":
                        tool_result = await self._execute_x_search(args)
                    else:
                        tool_result = {"error": f"Unknown tool: {function_name}"}
                    if "error" in tool_result:
                        logger.warning(f"Tool execution failed: {tool_result['error']}")
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": json.dumps(tool_result)})
                    continue
                return {"sentiment_score": 5.0, "confidence": 0.0, "summary": "No data available", "post_volume": 0, "key_themes": [], "notable_posts": []}
            return {"sentiment_score": 5.0, "confidence": 0.0, "summary": "Analysis loop timeout", "post_volume": 0, "key_themes": [], "notable_posts": []}
        except Exception as e:
            logger.error(f"X search failed for query '{query}': {e}", exc_info=True)
            return None

    async def _execute_x_search(self, args: dict) -> dict:
        if not self.x_bearer_token:
            return {"error": "X API not configured", "posts": [], "post_volume": 0, "data_quality": "unavailable"}
        query = args.get("query", "coffee futures")
        limit = args.get("limit", 10)
        mode = args.get("mode", "Latest")
        sort_order = "recency" if mode == "Latest" else "relevancy"
        sanitized_query = query
        for keyword in self.exclude_keywords:
            sanitized_query = sanitized_query.replace(f"-{keyword}", "").replace(f"- {keyword}", "")
        sanitized_query = re.sub(r'\(from:[^)]+\)', '', sanitized_query)
        sanitized_query = re.sub(r'from:\w+', '', sanitized_query)
        sanitized_query = ' '.join(sanitized_query.split()).strip()
        search_stage = "strict"
        strict_query = self._build_search_query(sanitized_query)
        posts = await self._fetch_x_posts(strict_query, limit, sort_order)
        if not posts:
            search_stage = "broad"
            broad_query = self._build_broad_search_query(sanitized_query)
            broad_threshold = self.sentinel_config.get('broad_min_faves', 3)
            posts = await self._fetch_x_posts(broad_query, limit, sort_order, min_likes=broad_threshold)
        data_quality = "high" if len(posts) >= 5 else ("low" if len(posts) < 3 else "medium")
        return {"posts": posts, "post_volume": len(posts), "query": query, "search_stage": search_stage, "data_quality": data_quality}

    async def check(self) -> Optional[SentinelTrigger]:
        from trading_bot.utils import is_market_open, is_trading_day
        if not is_trading_day(): return None
        if not is_market_open():
            if not hasattr(self, '_last_closed_market_check'): self._last_closed_market_check = None
            now = datetime.now(timezone.utc)
            if self._last_closed_market_check:
                hours_since_last = (now - self._last_closed_market_check).total_seconds() / 3600
                if hours_since_last < 4: return None
            self._last_closed_market_check = now
        if self.degraded_until and datetime.now(timezone.utc) < self.degraded_until:
            self.sensor_status = "OFFLINE"
            StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")
            return None
        tasks = [self._sem_bound_search(query) for query in self.search_queries]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Query '{self.search_queries[i]}' failed: {result}")
            elif isinstance(result, dict):
                valid_results.append(result)
            elif result is not None:
                logger.warning(
                    f"Query '{self.search_queries[i]}' returned non-dict type "
                    f"({type(result).__name__}): {str(result)[:100]}"
                )
        if not valid_results:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.degraded_until = datetime.now(timezone.utc) + timedelta(hours=1)
                self.sensor_status = "OFFLINE"
            StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")
            return None
        self.consecutive_failures = 0
        self.sensor_status = "ONLINE"
        StateManager.save_state({"x_sentiment": self.get_sensor_status()}, namespace="sensors")
        total_weight = 0.0
        weighted_sentiment = 0.0
        for r in valid_results:
            weight = r.get('confidence', 0.5) * max(1.0, r.get('post_volume', 1) / 10.0)
            weighted_sentiment += r.get('sentiment_score', 5.0) * weight
            total_weight += weight
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 5.0
        avg_confidence = sum(r.get('confidence', 0.5) for r in valid_results) / len(valid_results)
        all_themes = []
        for r in valid_results: all_themes.extend(r.get('key_themes', []))
        theme_counts = {}
        for theme in all_themes: theme_counts[theme] = theme_counts.get(theme, 0) + 1
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        total_volume = sum(r.get('post_volume', 0) for r in valid_results)
        volume_spike_detected = False
        if len(self.post_volume_history) >= 5:
            if total_volume > (self.volume_mean * self.volume_spike_multiplier):
                volume_spike_detected = True
        trigger_reason = None
        severity = 0
        if avg_sentiment >= self.sentiment_threshold:
            trigger_reason = f"EXTREMELY BULLISH X sentiment (score: {avg_sentiment:.1f}/10)"
            severity = 7
        elif avg_sentiment <= (10 - self.sentiment_threshold):
            trigger_reason = f"EXTREMELY BEARISH X sentiment (score: {avg_sentiment:.1f}/10)"
            severity = 7
        if volume_spike_detected and not trigger_reason:
            trigger_reason = f"UNUSUAL X ACTIVITY SPIKE ({total_volume} posts)"
            severity = 6
        if not trigger_reason: return None
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
            ][:3]
        }
        if self._is_duplicate_payload(payload): return None
        logger.warning(f"🚨 X SENTINEL TRIGGERED: {trigger_reason}")
        return SentinelTrigger(source="XSentimentSentinel", reason=trigger_reason, payload=payload, severity=severity)

class PredictionMarketSentinel(Sentinel):
    """
    Monitors Prediction Markets via Dynamic Market Discovery.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.sentinel_config = config.get('sentinels', {}).get('prediction_markets', {})
        polymarket_config = self.sentinel_config.get('providers', {}).get('polymarket', {})
        self.api_url = polymarket_config.get('api_url', "https://gamma-api.polymarket.com/events")
        self.search_limit = polymarket_config.get('search_limit', 10)
        self.min_liquidity = self.sentinel_config.get('min_liquidity_usd', 50000)
        self.min_volume = self.sentinel_config.get('min_volume_usd', 10000)
        self.hwm_decay_hours = self.sentinel_config.get('hwm_decay_hours', 24)
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self._load_state_cache()
        self._cleanup_misaligned_cache()
        self.topics = []
        self.reload_topics()
        self.poll_interval = self.sentinel_config.get('poll_interval_seconds', 300)
        self._last_poll_time = 0
        self.severity_map = self.sentinel_config.get('severity_mapping', {
            '10_to_20_pct': 6,
            '20_to_30_pct': 7,
            '30_plus_pct': 9
        })
        self._topic_failure_counts: Dict[str, int] = {}
        self._last_slug_check = datetime.now(timezone.utc)
        logger.info(f"PredictionMarketSentinel v2.0 initialized: {len(self.topics)} topics")

    @staticmethod
    def _word_boundary_match(keyword: str, text: str) -> bool:
        """Check if keyword matches in text using word-boundary matching.

        Handles both single words and multi-word phrases.
        Single words use plural-aware regex (appends optional 's').
        Multi-word phrases use substring match (natural word boundaries).

        Commodity-agnostic: works for any keyword vocabulary.
        """
        import re
        kw_lower = keyword.lower()
        text_lower = text.lower()

        if ' ' in kw_lower:
            # Multi-word phrase: substring match (natural boundaries)
            return kw_lower in text_lower
        else:
            # Single word: word-boundary match with optional plural 's'
            pattern = r'\b' + re.escape(kw_lower) + r's?\b'
            return bool(re.search(pattern, text_lower))

    def _passes_global_exclude_filter(self, title: str) -> bool:
        """
        Reject markets matching global exclude keywords.
        Uses plural-tolerant word-boundary matching for single words, substring for phrases.

        Plural-tolerant: \b{keyword}s?\b matches both "Bitcoin" and "Bitcoins",
        "Tariff" and "Tariffs", etc. — prevents the "Pluralization Trap" where
        Polymarket titles use plural forms that dodge exact word-boundary patterns.

        Commodity-agnostic: exclude list comes from config, not hardcoded.
        Returns True if market passes (is NOT excluded).
        """
        global_excludes = self.sentinel_config.get('global_exclude_keywords', [])

        for kw in global_excludes:
            if self._word_boundary_match(kw, title):
                return False
        return True

    def _validate_all_slugs(self):
        self._cleanup_misaligned_cache()

    def _cleanup_misaligned_cache(self):
        """Detect and clear cache entries where the resolved slug doesn't match the query.

        Uses plural-tolerant word-boundary matching for consistency with all other filter layers.
        """
        for topic_key, cached in list(self.state_cache.items()):
            if cached.get('slug') and cached.get('title'):
                query_lower = topic_key.lower()
                title_lower = cached['title'].lower()
                keywords = [kw for kw in query_lower.split() if len(kw) > 2]
                if not keywords:
                    continue

                has_match = any(
                    PredictionMarketSentinel._word_boundary_match(kw, title_lower)
                    for kw in keywords
                )

                if not has_match:
                    logger.warning(
                        f"Stale/misaligned slug for '{topic_key}': "
                        f"title='{cached['title']}'. Clearing cache."
                    )
                    self.state_cache.pop(topic_key)

    def _merge_discovered_topics(self, static_topics: List[Dict]) -> List[Dict]:
        discovered_file = "data/discovered_topics.json"
        if not os.path.exists(discovered_file): return static_topics
        try:
            with open(discovered_file, 'r') as f: discovered = json.load(f)
            merged = {t.get('tag', t.get('query')): t for t in static_topics}
            for topic in discovered:
                key = topic.get('tag', topic.get('query'))
                if key not in merged: merged[key] = topic
            return list(merged.values())
        except Exception as e:
            logger.warning(f"Failed to merge discovered topics: {e}")
            return static_topics

    def reload_topics(self):
        """Reload topics from config + discovered topics, pruning orphaned cache entries."""
        logger.info("Reloading prediction market topics...")
        static_topics = self.sentinel_config.get('topics_to_watch', [])
        self.topics = self._merge_discovered_topics(static_topics)

        # Prune orphaned cache entries for topics no longer active
        active_queries = {t.get('query', '') for t in self.topics}
        orphaned = [key for key in self.state_cache if key not in active_queries]
        for key in orphaned:
            logger.info(f"Pruning orphaned cache entry: '{key}'")
            self.state_cache.pop(key)

        if orphaned:
            self._save_state_cache()
            logger.info(f"Pruned {len(orphaned)} orphaned cache entries")

        logger.info(f"Topics reloaded: {len(self.topics)}")

    def _load_state_cache(self):
        try:
            cached = StateManager.load_state_raw(namespace="prediction_market_state")
            if cached and isinstance(cached, dict):
                validated = {}
                for key, value in cached.items():
                    if isinstance(value, dict): validated[key] = value
                self.state_cache = validated
                logger.info(f"Loaded state for {len(self.state_cache)} prediction market topics")
        except Exception as e:
            logger.warning(f"Failed to load prediction market state: {e}")

    def _save_state_cache(self):
        try:
            StateManager.save_state(self.state_cache, namespace="prediction_market_state")
        except Exception as e:
            logger.warning(f"Failed to save prediction market state: {e}")

    def _calculate_severity(self, delta_pct: float) -> int:
        abs_delta = abs(delta_pct)
        if abs_delta >= 30: return self.severity_map.get('30_plus_pct', 9)
        elif abs_delta >= 20: return self.severity_map.get('20_to_30_pct', 7)
        else: return self.severity_map.get('10_to_20_pct', 6)

    def _should_decay_hwm(self, hwm_timestamp: Optional[str]) -> bool:
        if not hwm_timestamp: return False
        try:
            hwm_time = datetime.fromisoformat(hwm_timestamp)
            age_hours = (datetime.now(timezone.utc) - hwm_time).total_seconds() / 3600
            return age_hours >= self.hwm_decay_hours
        except (ValueError, TypeError): return False

    def _infer_topic_category(self, query: str) -> str:
        """Infer topic category from query using plural-tolerant word-boundary matching.

        Commodity-agnostic: falls back to configured commodity name.
        """
        query_lower = query.lower()
        for category, keywords in DOMAIN_KEYWORD_WHITELISTS.items():
            for kw in keywords:
                if PredictionMarketSentinel._word_boundary_match(kw, query_lower):
                    return category
        commodity = self.config.get('commodity', {}).get('name', 'coffee').lower()
        return commodity if commodity in DOMAIN_KEYWORD_WHITELISTS else 'coffee'

    def _passes_domain_filter(self, market: dict, query: str, topic_category: str = None) -> bool:
        """Check if market title is relevant to the query domain.

        Uses plural-tolerant word-boundary matching for single words.
        Commodity-agnostic: whitelist is configurable per category.
        """
        title = market.get('title', '').lower()

        if topic_category and topic_category in DOMAIN_KEYWORD_WHITELISTS:
            required_keywords = DOMAIN_KEYWORD_WHITELISTS[topic_category]
        else:
            query_keywords = [kw for kw in query.lower().split() if len(kw) > 2]
            required_keywords = query_keywords

        for kw in required_keywords:
            if self._word_boundary_match(kw, title):
                return True

        return False

    async def _fetch_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific Polymarket event by slug.
        Used for slug pinning — avoids re-searching the API.

        Returns candidate dict or None if slug is invalid/closed/low-liquidity.
        """
        url = f"{self.api_url}?slug={slug}&closed=false&active=true&limit=1"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    data = await response.json()
                    if not isinstance(data, list) or not data:
                        return None

                    event = data[0]
                    if not isinstance(event, dict):
                        return None

                    markets = event.get('markets', [])
                    if not markets:
                        return None

                    # Find best market by liquidity
                    best_market = None
                    best_liq = -1
                    best_vol = 0
                    for m in markets:
                        if not isinstance(m, dict):
                            continue
                        try:
                            m_liq = float(m.get('liquidity', 0) or 0)
                        except (ValueError, TypeError):
                            continue
                        if m_liq > best_liq:
                            best_liq = m_liq
                            best_market = m
                            try:
                                best_vol = float(m.get('volume', 0) or 0)
                            except:
                                best_vol = 0

                    if best_market is None:
                        return None
                    if best_liq < self.min_liquidity:
                        return None
                    if best_vol < self.min_volume:
                        return None

                    return {
                        'slug': event.get('slug'),
                        'title': event.get('title', ''),
                        'market': best_market,
                        'liquidity': best_liq,
                        'volume': best_vol
                    }
        except Exception as e:
            logger.debug(f"Slug pin fetch failed for '{slug}': {e}")
            return None

    async def _resolve_active_market(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        params = {"q": query, "closed": "false", "active": "true", "limit": str(self.search_limit)}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Polymarket search failed for '{query}': HTTP {response.status}")
                        return None
                    data = await response.json()
                    if not isinstance(data, list): return None
                    if not data: return None
                    candidates = []
                    for event in data:
                        if not isinstance(event, dict): continue
                        markets = event.get('markets', [])
                        if not markets: continue
                        best_market = None
                        best_liq = -1
                        best_vol = 0
                        for m in markets:
                            if not isinstance(m, dict): continue
                            try:
                                m_liq = float(m.get('liquidity', 0) or 0)
                            except (ValueError, TypeError): continue
                            if m_liq > best_liq:
                                best_liq = m_liq
                                best_market = m
                                try: best_vol = float(m.get('volume', 0) or 0)
                                except: best_vol = 0
                        if best_market is None: continue
                        market = best_market
                        liquidity = best_liq
                        volume = best_vol
                        if liquidity < self.min_liquidity: continue
                        if volume < self.min_volume: continue
                        candidates.append({'slug': event.get('slug'), 'title': event.get('title', ''), 'market': market, 'liquidity': liquidity, 'volume': volume})
                    if not candidates: return None

                    # Layer 0: Global exclude filter (reject Bitcoin, sports, etc.)
                    candidates = [c for c in candidates if self._passes_global_exclude_filter(c.get('title', ''))]
                    if not candidates: return None

                    # Layer 1: Domain relevance filter
                    topic_category = self._infer_topic_category(query)
                    filtered_candidates = [m for m in candidates if self._passes_domain_filter(m, query, topic_category=topic_category)]
                    if not filtered_candidates: return None
                    candidates = filtered_candidates
                    relevance_keywords = kwargs.get('relevance_keywords', [])
                    min_relevance = kwargs.get('min_relevance_score', 2)  # Fail-safe: default to strict, not permissive
                    if relevance_keywords:
                        for candidate in candidates:
                            title_lower = candidate['title'].lower()
                            match_count = 0
                            for kw in relevance_keywords:
                                if self._word_boundary_match(kw, title_lower):
                                    match_count += 1
                            candidate['relevance_score'] = match_count
                        relevant = [c for c in candidates if c.get('relevance_score', 0) >= min_relevance]
                        if relevant:
                            relevant.sort(key=lambda x: x['liquidity'], reverse=True)
                            winner = relevant[0]
                        else: return None
                    else:
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
        if not self.sentinel_config.get('enabled', True): return None
        if not self.topics: return None
        import time as time_module
        now = time_module.time()
        if (now - self._last_poll_time) < self.poll_interval: return None
        slug_check_now = datetime.now(timezone.utc)
        if (slug_check_now - self._last_slug_check).total_seconds() > 14400:
            self._validate_all_slugs()
            self._last_slug_check = slug_check_now
        from trading_bot.utils import is_market_open, is_trading_day
        if not is_trading_day():
            if not hasattr(self, '_last_non_trading_check'): self._last_non_trading_check = 0
            if (now - self._last_non_trading_check) < 7200: return None
            self._last_non_trading_check = now
        elif not is_market_open():
            if not hasattr(self, '_last_closed_market_check_pm'): self._last_closed_market_check_pm = 0
            if (now - self._last_closed_market_check_pm) < 1800: return None
            self._last_closed_market_check_pm = now
        self._last_poll_time = now
        triggers = []
        for topic in self.topics:
            if not topic.get('enabled', True): continue
            query = topic.get('query')
            if not query: continue
            try:
                threshold = topic.get('trigger_threshold_pct', 10.0) / 100.0
                display_name = topic.get('display_name', query)
                tag = topic.get('tag', 'Unknown')
                importance = topic.get('importance', 'macro')
                commodity_impact = topic.get('commodity_impact', topic.get('coffee_impact', 'Potential macro impact'))
                relevance_keywords = topic.get('relevance_keywords', [])
                min_relevance = topic.get('min_relevance_score', self.sentinel_config.get('min_relevance_score', 1))

                # Try pinned slug first (from discovery), fall back to query search
                pinned_slug = topic.get('_discovery', {}).get('slug')
                market_data = None

                if pinned_slug:
                    market_data = await self._fetch_by_slug(pinned_slug)
                    if market_data:
                        # Validate pinned result still passes global excludes
                        if not self._passes_global_exclude_filter(market_data.get('title', '')):
                            logger.warning(f"Pinned slug '{pinned_slug}' failed global exclude. Falling back to search.")
                            market_data = None
                    else:
                        logger.info(f"Pinned slug '{pinned_slug}' no longer valid for '{display_name}'. Falling back to search.")

                if not market_data:
                    market_data = await self._resolve_active_market(query, relevance_keywords=relevance_keywords, min_relevance_score=min_relevance)

                if not market_data:
                    self._topic_failure_counts[query] = self._topic_failure_counts.get(query, 0) + 1
                    fail_count = self._topic_failure_counts[query]
                    MAX_CONSECUTIVE_FAILURES = 50
                    if fail_count >= MAX_CONSECUTIVE_FAILURES:
                        topic['enabled'] = False
                        continue
                    continue
                self._topic_failure_counts[query] = 0
                current_slug = market_data['slug']
                current_title = market_data['title']
                try:
                    outcomes = market_data['market'].get('outcomePrices', [])
                    if isinstance(outcomes, str):
                        try: outcomes = json.loads(outcomes)
                        except json.JSONDecodeError: continue
                    if not outcomes: continue
                    current_price = float(outcomes[0])
                except (ValueError, TypeError, IndexError): continue
                if query in self.state_cache:
                    cached = self.state_cache[query]
                    last_slug = cached.get('slug')
                    last_price = cached.get('price', current_price)
                    severity_hwm = cached.get('severity_hwm', 0)
                    hwm_timestamp = cached.get('hwm_timestamp')
                    if last_slug and last_slug != current_slug:
                        send_pushover_notification(self.config.get('notifications', {}), f"Market Rollover: {display_name}", f"Now tracking: {current_title[:50]}...", priority=-1)
                        self.state_cache[query] = {'slug': current_slug, 'title': current_title, 'price': current_price, 'timestamp': datetime.now(timezone.utc).isoformat(), 'severity_hwm': 0, 'hwm_timestamp': None}
                        continue
                    if self._should_decay_hwm(hwm_timestamp):
                        severity_hwm = 0
                        cached['severity_hwm'] = 0
                        cached['hwm_timestamp'] = None
                    delta = current_price - last_price
                    delta_pct = abs(delta) * 100
                    if abs(delta) >= threshold:
                        current_severity = self._calculate_severity(delta_pct)
                        if current_severity > severity_hwm:
                            direction = "JUMPED" if delta > 0 else "CRASHED"
                            msg = f"Prediction Market Alert: '{display_name}' {direction} {delta_pct:.1f}%"
                            logger.warning(f"🎯 PREDICTION SENTINEL: {msg}")
                            triggers.append(SentinelTrigger(source="PredictionMarketSentinel", reason=msg, payload={"topic": query, "slug": current_slug, "delta_pct": round(delta * 100, 2), "importance": importance, "commodity_impact": commodity_impact}, severity=current_severity))
                            cached['severity_hwm'] = current_severity
                            cached['hwm_timestamp'] = datetime.now(timezone.utc).isoformat()
                if query not in self.state_cache: self.state_cache[query] = {'severity_hwm': 0, 'hwm_timestamp': None}
                self.state_cache[query].update({'slug': current_slug, 'title': current_title, 'price': current_price, 'timestamp': datetime.now(timezone.utc).isoformat()})
            except Exception as topic_error:
                logger.warning(f"Error processing prediction market topic '{query}': {topic_error}")
                continue
        self._save_state_cache()
        if triggers:
            triggers.sort(key=lambda t: t.severity, reverse=True)
            return triggers[0]
        return None

class MacroContagionSentinel(Sentinel):
    """
    Detects macro shocks that cause cross-asset contagion.
    """
    def __init__(self, config, event_bus=None, logger=None):
        # event_bus and logger are passed by orchestrator, but we also inherit config from base Sentinel
        # Base Sentinel only takes config.
        # Orchestrator instantiates sentinels with (config).
        super().__init__(config)

        # Commodity-agnostic: load profile
        ticker = config.get('commodity', {}).get('ticker', 'KC')
        self.profile = get_commodity_profile(ticker)

        self.sentinel_config = config.get('sentinels', {}).get('MacroContagionSentinel', {})
        self.dxy_threshold_1d = 0.01  # 1% daily move
        self.dxy_threshold_2d = 0.02  # 2% two-day move
        self.last_dxy_value = None
        self.last_dxy_timestamp = None
        self.last_policy_check = None
        self.policy_check_interval = 86400  # Check policy news once per day

        api_key = config.get('gemini', {}).get('api_key')
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.0-flash-exp" # Flight Director specific model

    async def _get_history(self, ticker_symbol, period="5d", interval="1h"):
        import yfinance as yf
        loop = asyncio.get_running_loop()
        def fetch():
            t = yf.Ticker(ticker_symbol)
            return t.history(period=period, interval=interval)
        return await loop.run_in_executor(None, fetch)

    async def check_dxy_shock(self) -> Optional[Dict]:
        try:
            hist = await self._get_history("DX-Y.NYB", period="5d", interval="1h")

            if len(hist) < 48:
                return None

            current_value = hist['Close'].iloc[-1]
            value_24h_ago = hist['Close'].iloc[-24] if len(hist) >= 24 else None
            value_48h_ago = hist['Close'].iloc[-48] if len(hist) >= 48 else None

            pct_change_1d = ((current_value - value_24h_ago) / value_24h_ago) if value_24h_ago else 0
            pct_change_2d = ((current_value - value_48h_ago) / value_48h_ago) if value_48h_ago else 0

            self.last_dxy_value = current_value
            self.last_dxy_timestamp = datetime.now(timezone.utc)

            if abs(pct_change_1d) >= self.dxy_threshold_1d:
                direction = "SURGE" if pct_change_1d > 0 else "CRASH"
                return {
                    "type": "DXY_SHOCK_1D",
                    "direction": direction,
                    "current_dxy": current_value,
                    "pct_change": pct_change_1d,
                    "severity": "HIGH" if abs(pct_change_1d) >= 0.015 else "MODERATE"
                }

            if abs(pct_change_2d) >= self.dxy_threshold_2d:
                direction = "SURGE" if pct_change_2d > 0 else "CRASH"
                return {
                    "type": "DXY_SHOCK_2D",
                    "direction": direction,
                    "current_dxy": current_value,
                    "pct_change": pct_change_2d,
                    "severity": "CRITICAL" if abs(pct_change_2d) >= 0.025 else "HIGH"
                }

            return None

        except Exception as e:
            logger.error(f"Error checking DXY shock: {e}")
            return None

    async def check_cross_commodity_contagion(self) -> Optional[Dict]:
        try:
            tickers = {
                'coffee': 'KC=F',
                'gold': 'GC=F',
                'silver': 'SI=F',
                'wheat': 'ZW=F',
                'soybeans': 'ZS=F'
            }

            returns = {}
            for name, ticker in tickers.items():
                hist = await self._get_history(ticker, period="5d", interval="1d") # 1d for correlation
                if len(hist) < 2:
                    return None
                returns[name] = hist['Close'].pct_change().dropna()

            # Simple correlation (requires aligned dates, yfinance usually aligns or pandas handles it)
            import pandas as pd
            df = pd.DataFrame(returns)
            corr = df.corr()

            coffee_gold_corr = corr.loc['coffee', 'gold']
            coffee_silver_corr = corr.loc['coffee', 'silver']
            coffee_wheat_corr = corr.loc['coffee', 'wheat']
            coffee_soy_corr = corr.loc['coffee', 'soybeans']

            avg_precious_corr = (coffee_gold_corr + coffee_silver_corr) / 2
            avg_grain_corr = (coffee_wheat_corr + coffee_soy_corr) / 2

            if avg_precious_corr > 0.7 and avg_grain_corr < 0.3:
                return {
                    "type": "CROSS_COMMODITY_CONTAGION",
                    "correlation_precious": avg_precious_corr,
                    "correlation_grains": avg_grain_corr,
                    "interpretation": f"{self.profile.name} trading as RISK ASSET (like Gold/Silver), not AG COMMODITY",
                    "severity": "HIGH"
                }

            return None

        except Exception as e:
            logger.error(f"Error checking cross-commodity contagion: {e}")
            return None

    async def check_fed_policy_shock(self) -> Optional[Dict]:
        now = datetime.now(timezone.utc)
        if self.last_policy_check and (now - self.last_policy_check).total_seconds() < self.policy_check_interval:
            return None

        self.last_policy_check = now

        try:
            prompt = """
            Search for recent (past 48 hours) Federal Reserve policy surprises or shocks:
            - Fed Chair nominations or replacements
            - Unexpected FOMC rate decisions or hawkish/dovish pivots
            - Fed officials' speeches that moved markets

            If found, respond ONLY with JSON:
            {
                "shock_detected": true,
                "type": "CHAIR_NOMINATION" | "FOMC_SURPRISE" | "HAWKISH_PIVOT" | "DOVISH_PIVOT",
                "summary": "Brief description",
                "market_impact": "Expected impact on USD and commodities"
            }

            If no significant shock, respond: {"shock_detected": false}
            """

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt
            )

            # Clean JSON
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[:-3]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            if result.get('shock_detected'):
                return {
                    "type": "FED_POLICY_SHOCK",
                    "policy_type": result.get('type'),
                    "summary": result.get('summary'),
                    "market_impact": result.get('market_impact'),
                    "severity": "CRITICAL"
                }

            return None

        except Exception as e:
            logger.error(f"Error checking Fed policy shock: {e}")
            return None

    async def check(self) -> Optional[SentinelTrigger]:
        dxy_shock = await self.check_dxy_shock()
        if dxy_shock:
            return SentinelTrigger(
                source="MacroContagionSentinel",
                reason=f"DXY {dxy_shock['direction']}: {dxy_shock['pct_change']:.2%} ({dxy_shock['severity']})",
                payload=dxy_shock,
                severity=8 if dxy_shock['severity'] == 'CRITICAL' else 6
            )

        contagion = await self.check_cross_commodity_contagion()
        if contagion:
            return SentinelTrigger(
                source="MacroContagionSentinel",
                reason=f"Cross-Commodity Contagion: Coffee correlating with Gold/Silver ({contagion['correlation_precious']:.2f})",
                payload=contagion,
                severity=7
            )

        policy_shock = await self.check_fed_policy_shock()
        if policy_shock:
            return SentinelTrigger(
                source="MacroContagionSentinel",
                reason=f"Fed Policy Shock: {policy_shock['policy_type']} - {policy_shock['summary']}",
                payload=policy_shock,
                severity=9
            )

        return None

class FundamentalRegimeSentinel(Sentinel):
    """
    Determine if the market is in DEFICIT or SURPLUS regime.
    """
    def __init__(self, config, event_bus=None, logger=None):
        super().__init__(config)
        self.check_interval = 604800  # 1 week
        self.regime_file = Path("data/fundamental_regime.json")
        self.current_regime = self._load_regime()
        self.last_check = 0

    def _load_regime(self) -> Dict:
        if self.regime_file.exists():
            with open(self.regime_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "last_updated": None,
                "evidence": []
            }

    def _save_regime(self, regime: Dict):
        self.regime_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.regime_file, 'w') as f:
            json.dump(regime, f, indent=2, default=str)

    def check_ice_stocks_trend(self) -> str:
        # Placeholder logic
        return "BALANCED" # Assume balanced if no data

    def check_news_sentiment(self) -> str:
        try:
            surplus_url = "https://news.google.com/rss/search?q=coffee+market+surplus"
            deficit_url = "https://news.google.com/rss/search?q=coffee+market+deficit"

            surplus_feed = feedparser.parse(surplus_url)
            deficit_feed = feedparser.parse(deficit_url)

            if surplus_feed.bozo or deficit_feed.bozo:
                logger.warning("RSS feed error detected, preserving previous regime")
                return self.current_regime.get('regime', 'UNKNOWN')

            surplus_count = len(surplus_feed.entries)
            deficit_count = len(deficit_feed.entries)

            if surplus_count > deficit_count * 2:
                return "SURPLUS"
            elif deficit_count > surplus_count * 2:
                return "DEFICIT"
            else:
                return "BALANCED"

        except Exception as e:
            logger.error(f"Error checking news sentiment: {e}")
            return self.current_regime.get('regime', 'UNKNOWN')

    async def check(self) -> Optional[SentinelTrigger]:
        import time as time_module
        now = time_module.time()
        if (now - self.last_check) < self.check_interval:
            return None
        self.last_check = now

        loop = asyncio.get_running_loop()

        # Run blocking news check in executor
        news_regime = await loop.run_in_executor(None, self.check_news_sentiment)
        stocks_regime = self.check_ice_stocks_trend() # Placeholder is fast

        from collections import Counter

        if stocks_regime == news_regime:
            new_regime = stocks_regime
            confidence = 0.9
        else:
            regimes = [stocks_regime, stocks_regime, news_regime]
            new_regime = Counter(regimes).most_common(1)[0][0]
            confidence = 0.6

        if new_regime != self.current_regime.get('regime'):
            self.current_regime = {
                "regime": new_regime,
                "confidence": confidence,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "evidence": {
                    "ice_stocks": stocks_regime,
                    "news_sentiment": news_regime
                }
            }
            self._save_regime(self.current_regime)

            return SentinelTrigger(
                source="FundamentalRegimeSentinel",
                reason=f"Fundamental Regime Changed: {new_regime} (confidence: {confidence:.1%})",
                payload=self.current_regime,
                severity=6
            )

        return None
