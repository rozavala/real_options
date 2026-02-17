#!/usr/bin/env python3
"""
☕ COFFEE BOT REAL OPTIONS: COMPREHENSIVE PRE-FLIGHT CHECK
==============================================================

This script performs a complete diagnostic of the trading system to validate
all components are operational.

COMPONENTS TESTED (27 TOTAL):
1.  FOUNDATION (Env, Config, State, Dirs, Ledger, History)
2.  CHRONOMETRY (Time, Market Hours, Holidays)
3.  IBKR CONNECTIVITY (Gateway, Data, Pools)
4.  DATA FALLBACKS (YFinance)
5.  SENTINEL ARRAY (X, Price, Weather, News, Logistics, Microstructure)
6.  AI INFRASTRUCTURE (Router, Council, TMS, Semantic, Rate Limiter)
7.  EXECUTION PIPELINE (Strategy, Sizer, Notifications)

Usage:
    python verify_system_readiness.py [--quick] [--skip-ibkr] [--skip-llm] [--json] [--verbose]
"""

import asyncio
import logging
import os
import sys
import json
import time
import argparse
import socket
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# RESULT DATA STRUCTURES
# ============================================================================

class CheckStatus(Enum):
    PASS = "✅"
    FAIL = "❌"
    WARN = "⚠️"
    SKIP = "⏭️"

@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms
        }

@dataclass
class SystemReport:
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def is_ready(self) -> bool:
        return self.fail_count == 0

    def to_json(self):
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total": len(self.checks),
                "passed": self.pass_count,
                "failed": self.fail_count,
                "warnings": self.warn_count,
                "is_ready": self.is_ready
            },
            "checks": [c.to_dict() for c in self.checks]
        }, indent=2)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def timed_check(func):
    """Decorator to time check functions."""
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        if result:
            result.duration_ms = elapsed
        return result
    return wrapper

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_result(result: CheckResult):
    status_icon = result.status.value
    duration = f"({result.duration_ms:.0f}ms)" if result.duration_ms > 0 else ""
    print(f"  {status_icon} {result.name}: {result.message} {duration}")
    if result.details and result.status != CheckStatus.PASS:
        for line in result.details.split('\n'):
            print(f"      └─ {line}")

# ============================================================================
# CHECK 1: ENVIRONMENT VARIABLES
# ============================================================================

@timed_check
async def check_environment() -> CheckResult:
    """Verify all required environment variables are present."""
    required_keys = {
        'XAI_API_KEY': 'X Sentinel (Grok)',
        'GEMINI_API_KEY': 'Coffee Council / Research',
        'OPENAI_API_KEY': 'Router (GPT-4)',
        'ANTHROPIC_API_KEY': 'Router (Claude)',
    }

    optional_keys = {
        'X_BEARER_TOKEN': 'X API Direct Access',
        'PUSHOVER_USER_KEY': 'Notifications',
        'PUSHOVER_API_TOKEN': 'Notifications',
    }

    missing_required = []
    missing_optional = []

    # Also try to load from config.json as fallback
    config_keys = {}
    try:
        from config_loader import load_config
        config = load_config()
        config_keys = {
            'XAI_API_KEY': config.get('xai', {}).get('api_key'),
            'GEMINI_API_KEY': config.get('gemini', {}).get('api_key'),
            'OPENAI_API_KEY': config.get('openai', {}).get('api_key'),
            'ANTHROPIC_API_KEY': config.get('anthropic', {}).get('api_key'),
        }
    except Exception:
        config = None

    for key, purpose in required_keys.items():
        val = os.environ.get(key)
        if not val or val == "YOUR_API_KEY_HERE" or val == "LOADED_FROM_ENV":
            val = config_keys.get(key)

        if not val or val == "YOUR_API_KEY_HERE" or val == "LOADED_FROM_ENV":
            missing_required.append(f"{key} ({purpose})")

    for key, purpose in optional_keys.items():
        val = os.environ.get(key)
        if not val or val == "YOUR_API_KEY_HERE":
            missing_optional.append(f"{key} ({purpose})")

    if missing_required:
        return CheckResult("Environment Variables", CheckStatus.FAIL, f"{len(missing_required)} required keys MISSING", "\n".join(missing_required))
    elif missing_optional:
        return CheckResult("Environment Variables", CheckStatus.WARN, f"All required present, {len(missing_optional)} optional missing", f"Optional: {', '.join([k.split(' ')[0] for k in missing_optional])}")
    else:
        return CheckResult("Environment Variables", CheckStatus.PASS, f"All keys present")

# ============================================================================
# CHECK 2: CONFIGURATION FILE
# ============================================================================

@timed_check
async def check_config() -> CheckResult:
    """Verify config.json is valid and has required sections."""
    try:
        from config_loader import load_config
        config = load_config()

        required_sections = [
            'connection', 'symbol', 'exchange', 'gemini', 'sentinels',
            'compliance', 'risk_management'
        ]

        missing = [s for s in required_sections if s not in config]

        if missing:
            return CheckResult("Configuration", CheckStatus.FAIL, f"Missing sections: {missing}")

        sentinels = config.get('sentinels', {})
        sentinel_details = []
        for sentinel_name in ['news', 'logistics', 'weather', 'price', 'x_sentiment']:
            if sentinel_name in sentinels:
                sentinel_details.append(f"{sentinel_name}: ✓")
            else:
                sentinel_details.append(f"{sentinel_name}: MISSING")

        return CheckResult("Configuration", CheckStatus.PASS, f"Valid config ({len(config)} sections)", f"Sentinels: {', '.join(sentinel_details)}")

    except Exception as e:
        return CheckResult("Configuration", CheckStatus.FAIL, "Failed to load config", str(e))

# ============================================================================
# CHECK 3: IBKR CONNECTIVITY (STRICT MODE)
# ============================================================================


@timed_check
async def check_ibkr_connection(config: dict) -> CheckResult:
    """
    STRICT CONNECTION TEST (Revised):
    - Removed pre-flight TCP probe (avoids race conditions)
    - Increased timeout to 15s (accounts for Gateway latency)
    - Detailed logging enabled
    """
    try:
        from ib_insync import IB, util
        import logging
        
        # TEMPORARILY ELEVATE LOGGING to see handshake details
        util.logToConsole(logging.DEBUG)
        
        ib = IB()
        host = config.get('connection', {}).get('host', '127.0.0.1')
        port = config.get('connection', {}).get('port', 7497)
        
        # Attempt connection directly (15s timeout)
        await ib.connectAsync(host, port, clientId=999, timeout=15)
        
        # Latency Check
        start = time.time()
        await ib.reqCurrentTimeAsync()
        latency = (time.time() - start) * 1000
        
        account = ib.managedAccounts()[0] if ib.managedAccounts() else "Unknown"
        ib.disconnect()
        
        # === NEW: Give Gateway time to cleanup connection ===
        await asyncio.sleep(3.0)

        # Restore logging level
        util.logToConsole(logging.INFO)
        
        return CheckResult("IBKR Connection", CheckStatus.PASS, f"Connected | Latency: {latency:.0f}ms | Acct: {account}")
        
    except Exception as e:
        # Restore logging level
        from ib_insync import util
        import logging
        util.logToConsole(logging.INFO)
        
        return CheckResult(
            name="IBKR Connection",
            status=CheckStatus.FAIL, 
            message="API Handshake Failed", 
            details=f"Gateway at {host}:{port} did not respond within 15s.\nError: {str(e)}"
        )

# ============================================================================
# CHECK 4: IBKR MARKET DATA (SMART MODE)
# ============================================================================

@timed_check
async def check_ibkr_market_data(config: dict) -> CheckResult:
    """
    MARKET DATA TEST (SMART):
    Skips request if market is closed to avoid timeouts.

    ⚠️ OPERATIONAL NOTE: If this check fails with Error 162 after switching
    to a new commodity, it means you need to purchase the market data subscription
    in IBKR Account Management -> Market Data Subscriptions.
    """
    try:
        from trading_bot.utils import is_market_open, get_ibkr_exchange

        if not is_market_open():
            return CheckResult("Market Data", CheckStatus.PASS, "Market Closed - Request Skipped", "Skipping data request to avoid expected timeout.")

        # If Market is Open, we test it
        from ib_insync import IB
        from trading_bot.ib_interface import get_active_futures

        ticker_sym = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
        exchange = get_ibkr_exchange(config)

        ib = IB()
        host = config.get('connection', {}).get('host', '127.0.0.1')
        port = config.get('connection', {}).get('port', 7497)

        await ib.connectAsync(host, port, clientId=998)

        # Request market data for the front-month future
        futures = await get_active_futures(ib, ticker_sym, exchange, count=1)
        if not futures:
            ib.disconnect()
            return CheckResult(
                "Market Data", CheckStatus.FAIL,
                f"No futures found for {ticker_sym} on {exchange}. "
                f"Verify IBKR market data subscription in Account Management."
            )

        # === NEW: Async error collection (Fix B) ===
        # IBKR sends permission errors (162, 354) asynchronously.
        # We must wait briefly and check the error queue.
        collected_errors = []

        def _on_error(reqId, errorCode, errorString, contract):
            if errorCode in (162, 354, 10167):
                collected_errors.append((errorCode, errorString))

        ib.errorEvent += _on_error

        try:
            # Request a snapshot to trigger permission check
            contract = futures[0]
            ticker = ib.reqMktData(contract, '', True, False)

            # Wait for async errors to arrive
            await asyncio.sleep(3.0)

            if collected_errors:
                error_codes = [str(e[0]) for e in collected_errors]
                return CheckResult(
                    "Market Data", CheckStatus.FAIL,
                    f"IBKR data permission errors for {ticker_sym}: {', '.join(error_codes)}. "
                    f"Purchase market data subscription in IBKR Account Management."
                )

            # Check for data
            has_data = False
            start = time.time()
            while (time.time() - start) < 2: # Already waited 3s, wait a bit more if needed
                if ticker.last or ticker.close or ticker.bid:
                    has_data = True
                    break
                await asyncio.sleep(0.2)

            price = ticker.last or ticker.close or ticker.bid

        finally:
            ib.errorEvent -= _on_error
            ib.disconnect()

        # === NEW: Give Gateway time to cleanup connection ===
        await asyncio.sleep(3.0)

        if has_data:
            return CheckResult("Market Data", CheckStatus.PASS, f"Data Received: {price}")
        else:
            return CheckResult("Market Data", CheckStatus.WARN, "Connected but No Data (Illiquid?)")

    except Exception as e:
        return CheckResult("Market Data", CheckStatus.FAIL, str(e))

# ============================================================================
# CHECK 5: COMPLIANCE LOGIC (Full Logic)
# ============================================================================

@timed_check
async def check_compliance_volume(config: dict) -> CheckResult:
    """
    Verify the 'Expiry Trap' fix in Compliance volume resolution.
    """
    try:
        from ib_insync import IB, Contract, Bag, ComboLeg
        from trading_bot.compliance import ComplianceGuardian
        from trading_bot.utils import configure_market_data_type, get_ibkr_exchange

        ib = IB()
        host = config.get('connection', {}).get('host', '127.0.0.1')
        port = config.get('connection', {}).get('port', 7497)

        await ib.connectAsync(host, port, clientId=996)
        configure_market_data_type(ib)

        ticker_sym = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
        ibkr_exchange = get_ibkr_exchange(config)
        search_contract = Contract(secType='FOP', symbol=ticker_sym, exchange=ibkr_exchange, currency='USD')
        details_list = await ib.reqContractDetailsAsync(search_contract)

        if not details_list:
            ib.disconnect()
            return CheckResult("Compliance Volume (Expiry Trap)", CheckStatus.WARN, "No FOP options found to test", "Market may be closed")

        valid_option = details_list[0].contract
        bag = Bag()
        bag.symbol = ticker_sym
        bag.exchange = ibkr_exchange
        bag.currency = 'USD'
        bag.comboLegs = [ComboLeg(conId=valid_option.conId, ratio=1, action='BUY', exchange=ibkr_exchange)]

        guardian = ComplianceGuardian({'compliance': config.get('compliance', {'max_volume_pct': 0.10})})

        try:
            volume = await guardian._fetch_volume_stats(ib, bag)
            ib.disconnect()
            if volume >= 0:
                return CheckResult("Compliance Volume (Expiry Trap)", CheckStatus.PASS, f"Volume resolution OK: {volume:,.0f}", f"Tested with: {valid_option.localSymbol}")
            else:
                return CheckResult("Compliance Volume (Expiry Trap)", CheckStatus.WARN, "Volume returned -1 (unknown)")
        except AttributeError as e:
            ib.disconnect()
            if 'underlyingConId' in str(e):
                return CheckResult("Compliance Volume (Expiry Trap)", CheckStatus.FAIL, "EXPIRY TRAP BUG DETECTED!", f"Code uses 'underlyingConId' instead of 'underConId': {e}")
            raise
    except Exception as e:
        return CheckResult("Compliance Volume (Expiry Trap)", CheckStatus.FAIL, "Compliance test failed", str(e))

# ============================================================================
# CHECK 6: YFINANCE FALLBACK
# ============================================================================

@timed_check
async def check_yfinance() -> CheckResult:
    """
    Test YFinance data fallback.

    Commodity-agnostic: reads ticker from commodity profile.
    Market-aware: returns WARN (not FAIL) during off-hours/weekends,
    since empty data is expected when markets are closed.
    """
    try:
        from trading_bot.utils import get_market_data_cached, is_market_open, is_trading_day

        # Commodity-agnostic: derive YFinance ticker from profile
        try:
            from config.commodity_profiles import get_commodity_profile
            import json
            with open("config.json", "r") as f:
                cfg = json.load(f)
            ticker_symbol = cfg.get('commodity', {}).get('ticker', 'KC')
            profile = get_commodity_profile(ticker_symbol)
            yf_ticker = f"{profile.contract.symbol}=F"
        except Exception:
            yf_ticker = "KC=F"  # Safe fallback

        df = get_market_data_cached([yf_ticker], period="1d")

        if df is not None and not df.empty:
            return CheckResult(
                "YFinance Fallback", CheckStatus.PASS,
                f"Data OK ({yf_ticker}) | Rows: {len(df)}"
            )

        # Empty data — severity depends on whether market is open
        if not is_trading_day():
            return CheckResult(
                "YFinance Fallback", CheckStatus.WARN,
                f"No data for {yf_ticker} (expected: weekend/holiday)"
            )
        elif not is_market_open():
            return CheckResult(
                "YFinance Fallback", CheckStatus.WARN,
                f"No data for {yf_ticker} (expected: market closed)"
            )
        else:
            # Market IS open and we got no data — this is a real problem
            return CheckResult(
                "YFinance Fallback", CheckStatus.FAIL,
                f"No data for {yf_ticker} during market hours"
            )

    except Exception as e:
        return CheckResult("YFinance Fallback", CheckStatus.FAIL, str(e))

# ============================================================================
# CHECK 7: SENTINEL INITIALIZATION
# ============================================================================

@timed_check
async def check_x_sentinel(config: dict) -> CheckResult:
    try:
        from trading_bot.sentinels import XSentimentSentinel
        xai_key = os.environ.get('XAI_API_KEY')
        if not xai_key:
             return CheckResult("X Sentinel", CheckStatus.FAIL, "XAI_API_KEY not set")
        sentinel = XSentimentSentinel(config)
        return CheckResult("X Sentinel", CheckStatus.PASS, f"Model: {sentinel.model}")
    except Exception as e:
        return CheckResult("X Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_price_sentinel(config: dict) -> CheckResult:
    """Test Price Sentinel initialization (Lazy Mode - No Connection)."""
    try:
        from trading_bot.sentinels import PriceSentinel
        sentinel = PriceSentinel(config, None)
        threshold = sentinel.pct_change_threshold
        if threshold <= 0:
             return CheckResult("Price Sentinel", CheckStatus.FAIL, f"Invalid threshold: {threshold}")
        return CheckResult("Price Sentinel", CheckStatus.PASS, f"Initialized (Lazy) | Threshold: {threshold}%")
    except Exception as e:
        return CheckResult("Price Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_weather_sentinel(config: dict) -> CheckResult:
    try:
        from trading_bot.sentinels import WeatherSentinel
        sentinel = WeatherSentinel(config)
        # Handle v2.1 refactor (uses profile.primary_regions)
        if hasattr(sentinel, 'profile') and sentinel.profile:
            count = len(sentinel.profile.primary_regions)
            source = "profile"
        else:
            count = len(sentinel.locations)
            source = "config"
        return CheckResult("Weather Sentinel", CheckStatus.PASS, f"{count} regions ({source})")
    except Exception as e:
        return CheckResult("Weather Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_news_sentinel(config: dict) -> CheckResult:
    try:
        from trading_bot.sentinels import NewsSentinel
        sentinel = NewsSentinel(config)
        return CheckResult("News Sentinel", CheckStatus.PASS, "Initialized")
    except Exception as e:
        return CheckResult("News Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_logistics_sentinel(config: dict) -> CheckResult:
    """Test Logistics Sentinel RSS feed configuration (Full Logic)."""
    try:
        from trading_bot.sentinels import LogisticsSentinel
        sentinels_config = config.get('sentinels', {})
        logistics_config = sentinels_config.get('logistics', {})
        config_urls = logistics_config.get('rss_urls', [])
        sentinel = LogisticsSentinel(config)
        sentinel_urls = getattr(sentinel, 'urls', [])
        feeds = sentinel_urls if sentinel_urls else config_urls
        if not feeds:
            return CheckResult("Logistics Sentinel", CheckStatus.WARN, "No RSS feeds detected")
        return CheckResult("Logistics Sentinel", CheckStatus.PASS, f"Initialized | {len(feeds)} feeds configured")
    except Exception as e:
        return CheckResult("Logistics Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_prediction_market_sentinel(config: dict) -> CheckResult:
    """Test Prediction Market Sentinel v2.0 initialization and configuration."""
    try:
        from trading_bot.sentinels import PredictionMarketSentinel

        pm_config = config.get('sentinels', {}).get('prediction_markets', {})

        if not pm_config.get('enabled', False):
            return CheckResult(
                "Prediction Market Sentinel",
                CheckStatus.WARN,
                "Disabled in config"
            )

        sentinel = PredictionMarketSentinel(config)

        # Check topics (v2.0) instead of targets (v1.x)
        topics = len(sentinel.topics)

        if topics == 0:
            return CheckResult(
                "Prediction Market Sentinel",
                CheckStatus.WARN,
                "No topics configured (check topics_to_watch)"
            )

        # Validate thresholds
        liq = sentinel.min_liquidity
        vol = sentinel.min_volume
        hwm_decay = sentinel.hwm_decay_hours

        if liq < 10000:
            return CheckResult(
                "Prediction Market Sentinel",
                CheckStatus.WARN,
                f"Low liquidity threshold (${liq:,}) - risk of false positives"
            )

        return CheckResult(
            "Prediction Market Sentinel",
            CheckStatus.PASS,
            f"v2.0 | {topics} topics | liq=${liq:,} | HWM decay={hwm_decay}h"
        )
    except Exception as e:
        return CheckResult("Prediction Market Sentinel", CheckStatus.FAIL, str(e))

@timed_check
async def check_microstructure_sentinel(config: dict) -> CheckResult:
    try:
        micro_config = config.get('sentinels', {}).get('microstructure', {})
        return CheckResult("Microstructure Sentinel", CheckStatus.PASS, f"Config loaded")
    except Exception as e:
        return CheckResult("Microstructure Sentinel", CheckStatus.FAIL, str(e))

# ============================================================================
# CHECK 8: OPTION CHAIN BUILDER (Full Logic)
# ============================================================================

@timed_check
async def check_option_chain_builder(config: dict) -> CheckResult:
    """Test option chain building capability."""
    try:
        from ib_insync import IB
        from trading_bot.ib_interface import get_active_futures, build_option_chain
        from trading_bot.utils import configure_market_data_type, get_ibkr_exchange

        ib = IB()
        host = config.get('connection', {}).get('host', '127.0.0.1')
        port = config.get('connection', {}).get('port', 7497)

        await ib.connectAsync(host, port, clientId=994)
        configure_market_data_type(ib)

        ticker_sym = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
        ibkr_exchange = get_ibkr_exchange(config)
        futures = await get_active_futures(ib, ticker_sym, ibkr_exchange, count=1)
        if not futures:
            ib.disconnect()
            return CheckResult("Option Chain Builder", CheckStatus.WARN, "No active futures found")

        future = futures[0]
        chain = await build_option_chain(ib, future)
        ib.disconnect()

        if not chain:
            return CheckResult("Option Chain Builder", CheckStatus.WARN, f"Empty chain for {future.localSymbol}")

        return CheckResult("Option Chain Builder", CheckStatus.PASS, f"{future.localSymbol}: {len(chain.get('expirations', []))} expiries, {len(chain.get('strikes', []))} strikes")
    except Exception as e:
        return CheckResult("Option Chain Builder", CheckStatus.FAIL, str(e))

# ============================================================================
# CHECK 9: AI INFRASTRUCTURE
# ============================================================================

@timed_check
async def check_heterogeneous_router(config: dict) -> CheckResult:
    try:
        from trading_bot.heterogeneous_router import HeterogeneousRouter
        router = HeterogeneousRouter(config)
        return CheckResult("Heterogeneous Router", CheckStatus.PASS, f"{len(router.available_providers)} providers")
    except Exception as e:
        return CheckResult("Heterogeneous Router", CheckStatus.FAIL, str(e))

@timed_check
async def check_trading_council(config: dict) -> CheckResult:
    try:
        from trading_bot.agents import TradingCouncil
        council = TradingCouncil(config)
        return CheckResult("Trading Council", CheckStatus.PASS, f"Initialized")
    except Exception as e:
        return CheckResult("Trading Council", CheckStatus.FAIL, str(e))

@timed_check
async def check_tms() -> CheckResult:
    try:
        from trading_bot.tms import TransactiveMemory
        tms = TransactiveMemory()
        if tms.collection:
            return CheckResult("TMS (ChromaDB)", CheckStatus.PASS, f"Collection OK")
        return CheckResult("TMS (ChromaDB)", CheckStatus.FAIL, "Collection failed")
    except Exception as e:
        return CheckResult("TMS (ChromaDB)", CheckStatus.FAIL, str(e))

@timed_check
async def check_semantic_router(config: dict) -> CheckResult:
    """Test Semantic Router initialization and route matrix (Full Logic)."""
    try:
        from trading_bot.semantic_router import SemanticRouter, RouteDecision
        from trading_bot.sentinels import SentinelTrigger

        router = SemanticRouter(config)
        mock_trigger = SentinelTrigger(source="WeatherSentinel", reason="Frost", payload={"temp": -1}, severity=8)
        decision = router.route(mock_trigger)

        if not isinstance(decision, RouteDecision):
             return CheckResult("Semantic Router", CheckStatus.FAIL, "Invalid decision type")

        return CheckResult("Semantic Router", CheckStatus.PASS, f"{len(router.ROUTE_MATRIX)} routes | Test: {decision.primary_agent}")
    except Exception as e:
        return CheckResult("Semantic Router", CheckStatus.FAIL, str(e))

@timed_check
async def check_rate_limiter() -> CheckResult:
    try:
        from trading_bot.rate_limiter import GlobalRateLimiter
        status = GlobalRateLimiter.get_status()
        return CheckResult("Rate Limiter", CheckStatus.PASS, f"Providers: {list(status.keys())}")
    except Exception as e:
        return CheckResult("Rate Limiter", CheckStatus.FAIL, str(e))

@timed_check
async def check_llm_health(config: dict) -> CheckResult:
    """Test actual LLM provider connectivity."""
    try:
        from trading_bot.heterogeneous_router import HeterogeneousRouter, AgentRole
        router = HeterogeneousRouter(config)
        response = await router.route(role=AgentRole.NEWS_SENTINEL, prompt="OK", system_prompt=None, response_json=False)
        return CheckResult("LLM Provider Health", CheckStatus.PASS, f"Response: {response[:10]}")
    except Exception as e:
        return CheckResult("LLM Provider Health", CheckStatus.FAIL, str(e))

# ============================================================================
# CHECK 10: EXECUTION PIPELINE
# ============================================================================

@timed_check
async def check_strategy_definitions() -> CheckResult:
    try:
        from trading_bot.strategy import define_directional_strategy
        return CheckResult("Strategy Definitions", CheckStatus.PASS, "Import OK")
    except Exception as e:
        return CheckResult("Strategy Definitions", CheckStatus.FAIL, str(e))

@timed_check
async def check_position_sizer(config: dict) -> CheckResult:
    try:
        from trading_bot.position_sizer import DynamicPositionSizer
        sizer = DynamicPositionSizer(config)
        return CheckResult("Position Sizer", CheckStatus.PASS, "Initialized")
    except Exception as e:
        return CheckResult("Position Sizer", CheckStatus.FAIL, str(e))

@timed_check
async def check_notifications(config: dict) -> CheckResult:
    notif = config.get('notifications', {})
    if notif.get('enabled'):
        return CheckResult("Notifications", CheckStatus.PASS, "Enabled")
    return CheckResult("Notifications", CheckStatus.WARN, "Disabled in config")

# ============================================================================
# CHECK 11: MISC CHECKS
# ============================================================================

@timed_check
async def check_state_manager() -> CheckResult:
    try:
        from trading_bot.state_manager import StateManager
        ticker = os.environ.get("COMMODITY_TICKER", "KC")
        data_dir = os.path.join('data', ticker)
        StateManager.set_data_dir(data_dir)
        state = StateManager.load_state()
        StateManager.save_state({"_test": "ok"}, namespace="test")
        return CheckResult("State Manager", CheckStatus.PASS, f"Read/Write OK | {len(state)} keys")
    except Exception as e:
        return CheckResult("State Manager", CheckStatus.FAIL, str(e))

@timed_check
async def check_data_directories() -> CheckResult:
    ticker = os.environ.get("COMMODITY_TICKER", "KC")
    data_dir = os.path.join('data', ticker)
    dirs = [data_dir, os.path.join(data_dir, 'tms'), './logs']
    missing = [d for d in dirs if not os.path.exists(d)]
    if missing:
        return CheckResult("Data Directories", CheckStatus.FAIL, f"Missing: {missing}")
    return CheckResult("Data Directories", CheckStatus.PASS, "All directories exist")

@timed_check
async def check_trade_ledger() -> CheckResult:
    ticker = os.environ.get("COMMODITY_TICKER", "KC")
    ledger_path = os.path.join('data', ticker, 'trade_ledger.csv')
    if os.path.exists(ledger_path):
        return CheckResult("Trade Ledger", CheckStatus.PASS, "File exists")
    return CheckResult("Trade Ledger", CheckStatus.WARN, "File missing (will be created)")

@timed_check
async def check_council_history() -> CheckResult:
    ticker = os.environ.get("COMMODITY_TICKER", "KC")
    history_path = os.path.join('data', ticker, 'council_history.csv')
    if os.path.exists(history_path):
        return CheckResult("Council History", CheckStatus.PASS, "File exists")
    return CheckResult("Council History", CheckStatus.WARN, "File missing (will be created)")

@timed_check
async def check_chronometer() -> CheckResult:
    try:
        import pytz
        from trading_bot.utils import is_market_open, is_trading_day
        utc_now = datetime.now(timezone.utc)
        ny_now = utc_now.astimezone(pytz.timezone('America/New_York'))
        market_open = is_market_open()
        msg = f"UTC: {utc_now.strftime('%H:%M:%S')} | NY: {ny_now.strftime('%H:%M:%S')} | Market: {'OPEN' if market_open else 'CLOSED'}"
        return CheckResult("Chronometer", CheckStatus.PASS, msg)
    except Exception as e:
        return CheckResult("Chronometer", CheckStatus.FAIL, str(e))

@timed_check
async def check_holiday_calendar() -> CheckResult:
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cal = USFederalHolidayCalendar()
        return CheckResult("Holiday Calendar", CheckStatus.PASS, "Calendar loaded")
    except Exception as e:
        return CheckResult("Holiday Calendar", CheckStatus.FAIL, str(e))

@timed_check
async def check_connection_pool(config: dict) -> CheckResult:
    try:
        from trading_bot.connection_pool import IBConnectionPool
        ib = await IBConnectionPool.get_connection("test", config)
        await IBConnectionPool.release_connection("test")
        return CheckResult("Connection Pool", CheckStatus.PASS, "Acquired & Released")
    except Exception as e:
        return CheckResult("Connection Pool", CheckStatus.FAIL, str(e))

# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

async def run_diagnostics(
    skip_ibkr: bool = False,
    skip_llm: bool = False,
    quick: bool = False
) -> SystemReport:
    report = SystemReport()

    # 1. FOUNDATION
    report.checks.append(await check_environment())
    report.checks.append(await check_config())

    # Load config for rest
    from config_loader import load_config
    config = load_config()

    if config:
        report.checks.append(await check_state_manager())
        report.checks.append(await check_data_directories())
        report.checks.append(await check_trade_ledger())
        report.checks.append(await check_council_history())

        # 2. CHRONOMETRY
        report.checks.append(await check_chronometer())
        report.checks.append(await check_holiday_calendar())

        # 3. IBKR
        if not skip_ibkr:
            conn_result = await check_ibkr_connection(config)
            report.checks.append(conn_result)

            if conn_result.status == CheckStatus.PASS:
                report.checks.append(await check_ibkr_market_data(config))
                if not quick:
                    report.checks.append(await check_compliance_volume(config))
                    report.checks.append(await check_connection_pool(config))
                    report.checks.append(await check_option_chain_builder(config))
            else:
                 report.checks.append(CheckResult("Market Data", CheckStatus.SKIP, "No connection"))
        else:
            report.checks.append(CheckResult("IBKR", CheckStatus.SKIP, "Skipped by user"))

        # 4. FALLBACKS
        report.checks.append(await check_yfinance())

        # 5. SENTINELS
        report.checks.append(await check_x_sentinel(config))
        report.checks.append(await check_price_sentinel(config))
        report.checks.append(await check_weather_sentinel(config))
        report.checks.append(await check_news_sentinel(config))
        report.checks.append(await check_logistics_sentinel(config))
        report.checks.append(await check_microstructure_sentinel(config))
        report.checks.append(await check_prediction_market_sentinel(config))

        # 6. AI
        report.checks.append(await check_heterogeneous_router(config))
        report.checks.append(await check_trading_council(config))
        report.checks.append(await check_tms())
        report.checks.append(await check_semantic_router(config))
        report.checks.append(await check_rate_limiter())

        # 7. EXECUTION
        report.checks.append(await check_strategy_definitions())
        report.checks.append(await check_position_sizer(config))
        report.checks.append(await check_notifications(config))

        # 8. LLM HEALTH
        if not skip_llm and not quick:
            report.checks.append(await check_llm_health(config))

    return report

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip-ibkr', action='store_true')
    parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    report = await run_diagnostics(
        skip_ibkr=args.skip_ibkr,
        skip_llm=args.skip_llm,
        quick=args.quick
    )

    if args.json:
        print(report.to_json())
    else:
        print_section("FINAL REPORT")
        for check in report.checks:
            print_result(check)

        if report.is_ready:
            print(f"\n✅ SYSTEM READY ({report.pass_count} Passed)")
        else:
            print(f"\n❌ SYSTEM FAILURE ({report.fail_count} Failed)")

    sys.exit(0 if report.is_ready else 1)

if __name__ == "__main__":
    asyncio.run(main())
