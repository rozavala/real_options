"""The main orchestrator for the automated trading bot.

This script serves as the central nervous system of the application. It runs
as a long-lived process, responsible for scheduling and executing all the
different components of the trading pipeline at the correct times.

It now supports an Event-Driven architecture via a Sentinel Loop.
"""

import asyncio
import logging
import sys
import os
import json
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Callable
import time as time_module
from datetime import datetime, time, timedelta, timezone
import pytz
import pandas as pd
from ib_insync import IB, util, Contract, MarketOrder, LimitOrder, Order, Future

from config_loader import load_config
from trading_bot.logging_config import setup_logging
from notifications import send_pushover_notification
from performance_analyzer import main as run_performance_analysis
from reconcile_trades import main as run_reconciliation, reconcile_active_positions
from trading_bot.reconciliation import reconcile_council_history
from trading_bot.utils import log_council_decision
from trading_bot.decision_signals import log_decision_signal
from trading_bot.order_manager import (
    generate_and_execute_orders,
    close_stale_positions,
    cancel_all_open_orders,
    place_queued_orders,
    get_trade_ledger_df
)
from trading_bot.utils import archive_trade_ledger, configure_market_data_type, is_market_open, is_trading_day, round_to_tick, get_tick_size, word_boundary_match, hours_until_weekly_close
from equity_logger import log_equity_snapshot, sync_equity_from_flex
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, XSentimentSentinel, PredictionMarketSentinel, MacroContagionSentinel, SentinelTrigger, _sentinel_diag
from trading_bot.microstructure_sentinel import MicrostructureSentinel
from trading_bot.agents import TradingCouncil
from trading_bot.ib_interface import (
    get_active_futures, build_option_chain, create_combo_order_object, get_underlying_iv_metrics,
    place_order, close_spread_with_protection_cleanup
)
from trading_bot.strategy import define_directional_strategy, define_volatility_strategy
from trading_bot.state_manager import StateManager
from trading_bot.confidence_utils import parse_confidence
from trading_bot.connection_pool import IBConnectionPool
from trading_bot.compliance import ComplianceGuardian
from trading_bot.position_sizer import DynamicPositionSizer
from trading_bot.weighted_voting import RegimeDetector
from trading_bot.tms import TransactiveMemory
from trading_bot.budget_guard import BudgetGuard, get_budget_guard
from trading_bot.drawdown_circuit_breaker import DrawdownGuard
from trading_bot.cycle_id import generate_cycle_id
from trading_bot.strategy_router import route_strategy
from trading_bot.risk_management import _calculate_combo_risk_metrics
from trading_bot.task_tracker import record_task_completion, has_task_completed_today
from trading_bot.semantic_cache import get_semantic_cache
from trading_bot.utils import get_active_ticker
from trading_bot.sentinel_stats import SENTINEL_STATS

# --- Logging Setup ---
# NOTE: setup_logging() is called in main() after --commodity arg is parsed,
# so that the log file can be per-commodity (e.g. logs/orchestrator_kc.log).
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle for the monitor ---
monitor_process = None
GLOBAL_BUDGET_GUARD = None
GLOBAL_DRAWDOWN_GUARD = None
_STARTUP_DISCOVERY_TIME = 0  # Set to time.time() after successful startup topic discovery

# Module-level shutdown state
_SYSTEM_SHUTDOWN = False
_brier_zero_resolution_streak = 0

def is_system_shutdown() -> bool:
    """Check if the system has entered end-of-day shutdown."""
    return _SYSTEM_SHUTDOWN

def _record_sentinel_health(name: str, status: str, interval_seconds: int, error: str = None):
    """
    Record sentinel operational health to state.json for dashboard consumption.

    Args:
        name: Sentinel class name (e.g. 'WeatherSentinel')
        status: 'OK', 'ERROR', 'IDLE', 'INITIALIZING'
        interval_seconds: Expected check interval for staleness calculation
        error: Error message if status is 'ERROR'
    """
    try:
        health_data = {
            'status': status,
            'last_check_utc': datetime.now(timezone.utc).isoformat(),
            'interval_seconds': interval_seconds,
            'error': error,
        }
        StateManager.atomic_state_update("sentinel_health", name, health_data)
    except Exception as e:
        # Never let health reporting crash the sentinel loop
        logger.warning(f"Failed to record sentinel health for {name}: {e}")

class TriggerDeduplicator:
    def __init__(self, window_seconds: int = 7200, state_file=None, critical_severity_threshold: int = 9):
        if state_file is None:
            state_file = os.path.join("data", os.environ.get("COMMODITY_TICKER", "KC"), "deduplicator_state.json")
        self.state_file = state_file
        self.window = window_seconds
        self.critical_severity_threshold = critical_severity_threshold
        self.cooldowns = {} # Dictionary of cooldowns {source: until_timestamp}
        self.recent_triggers = deque(maxlen=50)
        self.metrics = {
            'total_triggers': 0,
            'filtered_global_cooldown': 0,
            'filtered_post_cycle': 0,
            'filtered_source_cooldown': 0,
            'filtered_duplicate_content': 0,
            'processed': 0,
            '_last_reset_date': datetime.now(timezone.utc).strftime('%Y-%m-%d')
        }
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Migrate old state if necessary or just load dict
                    if 'global_cooldown_until' in data:
                        self.cooldowns['GLOBAL'] = data['global_cooldown_until']
                    else:
                        self.cooldowns = data.get('cooldowns', {})

                    for t in data.get('recent_triggers', []):
                         self.recent_triggers.append(tuple(t)) # (hash, timestamp)

                    if 'metrics' in data and isinstance(data['metrics'], dict):
                        loaded_metrics = data['metrics']
                        # Daily reset: if metrics are from a previous day, reset counters
                        last_reset = loaded_metrics.get('_last_reset_date', '')
                        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                        if last_reset != today:
                            logger.info(f"Deduplicator metrics stale (last reset: {last_reset or 'never'}), resetting counters")
                            loaded_metrics = {}
                        loaded_metrics.pop('_last_reset_date', None)
                        self.metrics.update(loaded_metrics)
                        self.metrics['_last_reset_date'] = today
            except Exception as e:
                logger.warning(f"Failed to load deduplicator state: {e}")

    def _save_state(self):
        try:
            # Create data dir if not exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            data = {
                'cooldowns': self.cooldowns,
                'recent_triggers': list(self.recent_triggers),
                'metrics': self.metrics
            }
            tmp_path = self.state_file + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_file)
        except Exception as e:
             logger.warning(f"Failed to save deduplicator state: {e}")

    def should_process(self, trigger: SentinelTrigger) -> bool:
        self.metrics['total_triggers'] += 1
        now = time_module.time()

        # 1. Check Global Lock (Set by Scheduled Tasks)
        global_until = self.cooldowns.get('GLOBAL', 0)
        if now < global_until:
            logger.info(f"GLOBAL cooldown active until {datetime.fromtimestamp(global_until)}. Skipping {trigger.source}")
            self.metrics['filtered_global_cooldown'] += 1
            return False

        # 2. Check Post-Cycle Debounce (Set after ANY emergency cycle)
        post_cycle_until = self.cooldowns.get('POST_CYCLE', 0)
        if now < post_cycle_until:
            # Exception: Allow CRITICAL severity to bypass
            critical_threshold = self.critical_severity_threshold
            if getattr(trigger, 'severity', 0) < critical_threshold:
                logger.info(f"POST_CYCLE debounce active until {datetime.fromtimestamp(post_cycle_until)}. Skipping {trigger.source}")
                self.metrics['filtered_post_cycle'] += 1
                return False
            else:
                logger.warning(f"CRITICAL trigger {trigger.source} (Sev: {trigger.severity}) bypassing post-cycle debounce")

        # 3. Check Per-Source Lock (Set by Sentinel Self-Triggers)
        source_until = self.cooldowns.get(trigger.source, 0)
        if now < source_until:
            logger.info(f"{trigger.source} cooldown active until {datetime.fromtimestamp(source_until)}. Skipping.")
            self.metrics['filtered_source_cooldown'] += 1
            return False

        # Content-based deduplication
        # Sort keys to ensure stable hash
        # For persistent environmental conditions, append date so they can re-trigger daily
        date_suffix = ""
        if trigger.source in ("WeatherSentinel", "LogisticsSentinel"):
            date_suffix = datetime.now(timezone.utc).strftime("%Y%m%d")

        trigger_hash = hashlib.md5(
            f"{trigger.reason[:50]}{json.dumps(trigger.payload, sort_keys=True)}{date_suffix}".encode()
        ).hexdigest()[:8]

        # Prune old triggers
        cutoff = now - self.window
        while self.recent_triggers and self.recent_triggers[0][1] < cutoff:
             self.recent_triggers.popleft()

        if any(t[0] == trigger_hash for t in self.recent_triggers):
            logger.info(f"Duplicate trigger detected: {trigger_hash}")
            self.metrics['filtered_duplicate_content'] += 1
            return False

        self.recent_triggers.append((trigger_hash, now))
        self._save_state()
        self.metrics['processed'] += 1
        return True

    def set_cooldown(self, source: str, seconds: int = 900):
        """Sets a cooldown for a specific source (or 'GLOBAL')."""
        self.cooldowns[source] = time_module.time() + seconds
        self._save_state()

    def clear_cooldown(self, source: str):
        """Clears the cooldown for a specific source."""
        if source in self.cooldowns:
            del self.cooldowns[source]
            self._save_state()

# Global Deduplicator Instance — initialized in main() with commodity-specific state_file
GLOBAL_DEDUPLICATOR = None

# Concurrent Cycle Lock (Global)
EMERGENCY_LOCK = asyncio.Lock()

# Track fire-and-forget tasks so they can be cancelled on shutdown
_INFLIGHT_TASKS: set[asyncio.Task] = set()


def _extract_agent_prediction(report) -> tuple:
    """
    Extract direction and confidence from an agent report.

    Handles both structured dict reports and legacy string reports.
    Returns: (direction: str, confidence: float)
    """
    direction = 'NEUTRAL'
    confidence = 0.5

    if isinstance(report, dict):
        direction = report.get('sentiment', 'NEUTRAL')
        # v5.3.1 FIX: Parse band strings to float at the source
        confidence = parse_confidence(report.get('confidence', 0.5))

        # Fallback parsing if sentiment missing
        if not direction or direction in ('N/A', ''):
            report_str = str(report.get('data', '')).upper()
            if 'BULLISH' in report_str:
                direction = 'BULLISH'
            elif 'BEARISH' in report_str:
                direction = 'BEARISH'

    elif isinstance(report, str):
        report_str = report.upper()
        if 'BULLISH' in report_str:
            direction = 'BULLISH'
        elif 'BEARISH' in report_str:
            direction = 'BEARISH'

    return direction, confidence


def _route_emergency_strategy(decision: dict, market_context: dict, agent_reports: dict, config: dict) -> dict:
    """
    v7.1: Strategy routing for emergency cycles.

    Mirrors the v7.0 "Judge & Jury" routing logic from signal_generator.py
    to ensure emergency cycles respect thesis_strength, vol_sentiment,
    and regime-based strategy selection.

    Design principle: LLMs decide WHAT (direction + thesis).
                      Python code decides HOW (strategy type + sizing).

    Returns:
        dict with keys: prediction_type, vol_level, direction, confidence,
                        thesis_strength, conviction_multiplier, vol_sentiment,
                        regime, reason
    """
    direction = decision.get('direction', 'NEUTRAL')
    confidence = decision.get('confidence', 0.0)
    thesis_strength = decision.get('thesis_strength', 'SPECULATIVE')
    conviction_multiplier = decision.get('conviction_multiplier', 1.0)
    reasoning = decision.get('reasoning', '')

    # v7.0 SAFETY: Default to BEARISH (expensive) when vol data is missing.
    # Fail-safe, not fail-neutral.
    vol_sentiment = decision.get('volatility_sentiment', 'BEARISH')
    if not vol_sentiment or vol_sentiment == 'N/A':
        vol_sentiment = 'BEARISH'

    regime = market_context.get('regime', 'UNKNOWN')

    prediction_type = "DIRECTIONAL"
    vol_level = None
    reason = reasoning

    if direction == 'NEUTRAL':
        # === NEUTRAL PATH: Vol trade or No Trade ===
        # Simplified conflict detection for emergency path
        agent_conflict_score = _calculate_emergency_conflict(agent_reports)
        imminent_catalyst = _detect_emergency_catalyst(agent_reports)

        # PATH 1: IRON CONDOR — sell premium in range when vol is expensive
        if regime == 'RANGE_BOUND' and vol_sentiment == 'BEARISH':
            prediction_type = "VOLATILITY"
            vol_level = "LOW"
            reason = "Emergency Iron Condor: Range-bound + expensive vol (sell premium)"
            logger.info(f"EMERGENCY STRATEGY: IRON_CONDOR | regime={regime}, vol={vol_sentiment}")

        # PATH 2: LONG STRADDLE — expect big move, options not expensive
        elif (imminent_catalyst or agent_conflict_score > 0.6) and vol_sentiment != 'BEARISH':
            prediction_type = "VOLATILITY"
            vol_level = "HIGH"
            reason = f"Emergency Long Straddle: {imminent_catalyst or f'High conflict ({agent_conflict_score:.2f})'}"
            logger.info(f"EMERGENCY STRATEGY: LONG_STRADDLE | catalyst={imminent_catalyst}, conflict={agent_conflict_score:.2f}")

        # PATH 3: NO TRADE
        else:
            prediction_type = "DIRECTIONAL"
            vol_level = None
            reason = (
                f"Emergency NO TRADE: Direction neutral, no positive-EV vol trade. "
                f"(vol={vol_sentiment}, regime={regime}, conflict={agent_conflict_score:.2f})"
            )
            logger.info(f"EMERGENCY NO TRADE: vol={vol_sentiment}, regime={regime}")
    else:
        # === DIRECTIONAL PATH: Always defined-risk spreads ===
        if vol_sentiment == 'BEARISH':
            reason += f" [VOL WARNING: Options expensive. Spread costs elevated. Thesis: {thesis_strength}]"
        logger.info(f"EMERGENCY STRATEGY: DIRECTIONAL spread (thesis={thesis_strength}, vol={vol_sentiment})")

    return {
        'prediction_type': prediction_type,
        'vol_level': vol_level,
        'direction': direction if prediction_type != 'VOLATILITY' else 'VOLATILITY',
        'confidence': confidence,
        'thesis_strength': thesis_strength,
        'conviction_multiplier': conviction_multiplier,
        'volatility_sentiment': vol_sentiment,
        'regime': regime,
        'reason': reason,
    }


def _calculate_emergency_conflict(agent_reports: dict) -> float:
    """
    Quick conflict score for emergency path.

    Measures directional disagreement among cached agent reports.
    Returns 0.0 (full agreement) to 1.0 (full disagreement).
    """
    directions = []
    for key, report in agent_reports.items():
        if key in ('master_decision', 'master'):
            continue
        report_str = str(report.get('data', '') if isinstance(report, dict) else report).upper()
        if 'BULLISH' in report_str:
            directions.append(1)
        elif 'BEARISH' in report_str:
            directions.append(-1)
        # NEUTRAL agents don't contribute to conflict

    if len(directions) < 2:
        return 0.0

    avg = sum(directions) / len(directions)
    # Variance-like measure: how spread out are the directions?
    conflict = sum(abs(d - avg) for d in directions) / len(directions)
    return min(1.0, conflict)  # Normalize to [0, 1]


def _detect_emergency_catalyst(agent_reports: dict) -> str:
    """
    Check if any agent report mentions an imminent catalyst.

    Returns catalyst description string or empty string.
    """
    catalyst_keywords = [
        'USDA report', 'FOMC', 'frost', 'freeze', 'hurricane',
        'strike', 'embargo', 'election', 'earnings', 'inventory report',
        'COT report', 'export ban', 'tariff', 'sanctions'
    ]

    for key, report in agent_reports.items():
        report_text = str(report.get('data', '') if isinstance(report, dict) else report)
        for keyword in catalyst_keywords:
            if keyword.lower() in report_text.lower():
                return f"{keyword} (detected in {key} report)"

    return ""


def _infer_strategy_type(routed: dict) -> str:
    """Infer human-readable strategy type from routed signal."""
    if routed['prediction_type'] == 'VOLATILITY':
        if routed.get('vol_level') == 'HIGH':
            return 'LONG_STRADDLE'
        elif routed.get('vol_level') == 'LOW':
            return 'IRON_CONDOR'
    elif routed['direction'] in ('BULLISH', 'BEARISH'):
        return 'DIRECTIONAL'
    return 'NONE'


async def _detect_market_regime(config: dict, trigger=None, ib: IB = None, contract: Contract = None) -> str:
    """
    Detect current market regime for Brier scoring context.

    Uses actual market data if available, otherwise falls back to trigger source.
    Returns MarketRegime string value.
    """
    # 1. Try Actual Market Data
    if ib and contract:
        try:
            regime = await RegimeDetector.detect_regime(ib, contract)
            if regime != "UNKNOWN":
                # Normalize terminology
                if regime == "HIGH_VOLATILITY":
                    return "HIGH_VOL"
                return regime
        except Exception as e:
            logger.error(f"Regime detection via IB failed: {e}")

    # 2. Fallback to Trigger Source
    if trigger:
        source = getattr(trigger, 'source', '').lower()

        if 'weather' in source:
            return "WEATHER_EVENT"
        elif 'prediction_market' in source or 'macro' in source:
            return "MACRO_SHIFT"
        elif 'price' in source or 'microstructure' in source:
            return "HIGH_VOL"

    return "NORMAL"


async def _get_current_regime_and_iv(ib: IB, config: dict) -> tuple:
    """
    Estimates the current market regime and returns IV rank.

    Returns:
        (regime: str, iv_rank: float) — regime is one of
        'HIGH_VOLATILITY', 'RANGE_BOUND', 'TRENDING'; iv_rank is 0-100.
    """
    iv_rank = 50.0  # default
    try:
        futures = await asyncio.wait_for(get_active_futures(ib, config['symbol'], config['exchange'], count=1), timeout=15)
        if futures:
            metrics = await asyncio.wait_for(get_underlying_iv_metrics(ib, futures[0]), timeout=15)
            iv_rank = metrics.get('iv_rank', 50)
            if isinstance(iv_rank, str):
                iv_rank = 50.0  # Fallback
            iv_rank = float(iv_rank)
            iv_threshold = config.get('exit_logic', {}).get('condor_iv_rank_breach', 70)
            if iv_rank > iv_threshold:
                return 'HIGH_VOLATILITY', iv_rank
            elif iv_rank < 30:
                return 'RANGE_BOUND', iv_rank
    except Exception as e:
        logger.warning(f"IV regime classification failed: {e}")
    return 'TRENDING', iv_rank

async def _get_current_price(ib: IB, contract: Contract) -> float:
    """Fetches current price snapshot."""
    ticker = ib.reqMktData(contract, '', True, False)
    await asyncio.sleep(1)
    price = 0.0
    if not util.isNan(ticker.last):
        price = ticker.last
    elif not util.isNan(ticker.close):
        price = ticker.close
    # For snapshot requests, IB auto-cleans the ticker.
    # Do NOT call cancelMktData — it causes Error 300 spam.
    return price

async def _get_context_for_guardian(guardian: str, config: dict) -> str:
    """Fetches relevant context for a specific guardian's thesis validation.

    Combines three sources:
    1. Recent sentinel events (last 5 triggers from StateManager)
    2. Guardian's latest analyst report from state
    3. TMS memory for the guardian (ChromaDB vector store)
    """
    parts = []

    # 1. Recent sentinel events
    try:
        sentinel_state = StateManager.load_state_raw("sentinel_history")
        sentinel_events = sentinel_state.get("events", [])
        if sentinel_events:
            event_lines = []
            for evt in sentinel_events[-5:]:
                if isinstance(evt, dict):
                    event_lines.append(
                        f"  - [{evt.get('timestamp', '?')}] {evt.get('source', '?')}: {evt.get('reason', '?')}"
                    )
            if event_lines:
                parts.append("Recent Sentinel Events:\n" + "\n".join(event_lines))
    except Exception as e:
        logger.debug(f"Could not load sentinel history for guardian context: {e}")

    # 2. Guardian's latest report from state
    try:
        reports = StateManager.load_state_raw("reports")
        guardian_report = reports.get(guardian)
        if guardian_report:
            # Truncate if very long to fit LLM context
            report_str = str(guardian_report)
            if len(report_str) > 2000:
                report_str = report_str[:2000] + "... [truncated]"
            parts.append(f"Guardian '{guardian}' Latest Report:\n{report_str}")
    except Exception as e:
        logger.debug(f"Could not load guardian report for context: {e}")

    # 3. TMS memory (existing)
    try:
        tms = TransactiveMemory()
        tms_results = tms.retrieve(f"{guardian} analysis", n_results=3)
        if tms_results:
            parts.append(f"Memory Context:\n{tms_results}")
    except Exception as e:
        logger.debug(f"Could not query TMS for guardian context: {e}")

    if not parts:
        return f"No recent context available for {guardian}."

    return "\n\n".join(parts)

async def _validate_iron_condor(thesis: dict, config: dict, ib: IB, active_futures_cache: dict) -> dict:
    """Validate Iron Condor against regime, IV, and price breaches."""
    exit_cfg = config.get('exit_logic', {})
    regime_exits_enabled = exit_cfg.get('enable_regime_breach_exits', True)

    # Get current regime AND iv_rank
    current_regime, current_iv_rank = await _get_current_regime_and_iv(ib, config)
    entry_regime = thesis.get('entry_regime', 'RANGE_BOUND')
    iv_breach_threshold = exit_cfg.get('condor_iv_rank_breach', 70)

    if regime_exits_enabled:
        if current_regime == 'HIGH_VOLATILITY' and entry_regime == 'RANGE_BOUND':
            return {
                'action': 'CLOSE',
                'reason': "REGIME BREACH: Entered as RANGE_BOUND, now HIGH_VOLATILITY. Iron Condor invalid."
            }

        # Standalone IV rank check — high IV threatens all iron condors
        if current_iv_rank > iv_breach_threshold:
            return {
                'action': 'CLOSE',
                'reason': (
                    f"IV RANK BREACH: Current IV rank {current_iv_rank:.1f} exceeds "
                    f"threshold {iv_breach_threshold}. Iron Condor at risk."
                )
            }

    # === PRICE BREACH CHECK ===
    supporting_data = thesis.get('supporting_data', {})
    underlying_symbol = supporting_data.get('underlying_symbol', config.get('symbol', 'KC'))
    contract_month = supporting_data.get('contract_month')

    underlying_contract = None

    if contract_month:
        # Build underlying future contract from stored metadata
        underlying_contract = Future(
            symbol=underlying_symbol,
            lastTradeDateOrContractMonth=contract_month,
            exchange=config.get('exchange', 'NYBOT')
        )
        try:
            await asyncio.wait_for(ib.qualifyContractsAsync(underlying_contract), timeout=15)
        except Exception as e:
            logger.warning(f"Failed to qualify stored underlying contract: {e}")
            underlying_contract = None

    if not underlying_contract:
        # Fallback: Get front-month future (with caching)
        symbol = config.get('symbol', 'KC')
        if active_futures_cache and symbol in active_futures_cache:
            futures = active_futures_cache[symbol]
            underlying_contract = futures[0] if futures else None
            logger.debug(f"Using cached active future for {symbol}")
        else:
            try:
                futures = await asyncio.wait_for(get_active_futures(ib, symbol, config['exchange'], count=1), timeout=15)
                underlying_contract = futures[0] if futures else None
                if active_futures_cache is not None and futures:
                    active_futures_cache[symbol] = futures
            except Exception as e:
                logger.warning(f"Failed to get active futures for IC validation: {e}")

    if not underlying_contract:
        logger.error("Cannot validate IC thesis - no underlying contract available")
        return {'action': 'HOLD', 'reason': 'Unable to fetch underlying price for validation'}

    # Fetch current underlying price
    current_price = await _get_current_price(ib, underlying_contract)
    entry_price = supporting_data.get('entry_price', 0)

    # SANITY CHECK: Detect legacy theses where entry_price is a spread premium
    price_floor = config.get('validation', {}).get('underlying_price_floor', 100.0)

    if 0 < entry_price < price_floor:
        # spread_credit = supporting_data.get('spread_credit', entry_price)
        logger.warning(
            f"SEMANTIC GUARD: entry_price ${entry_price:.2f} < floor "
            f"${price_floor:.2f} for thesis {thesis.get('trade_id', '?')}. "
            f"Likely a spread premium, not underlying price. "
            f"Skipping price breach check. Run migration script."
        )
        entry_price = 0  # Disable price breach check for this thesis

    price_breach_pct = exit_cfg.get('condor_price_breach_pct', 2.0)
    if entry_price > 0 and current_price > 0:
        move_pct = abs((current_price - entry_price) / entry_price) * 100

        if move_pct > price_breach_pct:
            return {
                'action': 'CLOSE',
                'reason': f"PRICE BREACH: Underlying moved {move_pct:.2f}% from entry (threshold: {price_breach_pct}%). "
                          f"Entry: ${entry_price:.2f}, Current: ${current_price:.2f}"
            }
    else:
        logger.warning(f"Invalid prices for IC validation: entry={entry_price}, current={current_price}")

    return None


async def _validate_long_straddle(thesis: dict, position, config: dict, ib: IB, active_futures_cache: dict) -> dict:
    """Validate Long Straddle against theta burn."""
    exit_cfg = config.get('exit_logic', {})
    theta_check_enabled = exit_cfg.get('enable_theta_hurdle_check', True)
    if not theta_check_enabled:
        logger.debug("Theta hurdle check disabled via config — skipping LONG_STRADDLE validation")
        return None

    # Check theta efficiency
    theta_hours = exit_cfg.get('theta_hurdle_hours', 4)
    theta_min_move = exit_cfg.get('theta_minimum_move_pct', 1.0)

    entry_time_str = thesis.get('entry_timestamp', '')
    if entry_time_str:
        entry_time = datetime.fromisoformat(entry_time_str)
        hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        if hours_held > theta_hours:
            supporting_data = thesis.get('supporting_data', {})
            entry_price = supporting_data.get('entry_price', 0)
            if entry_price:
                # Fix 5: Fetch underlying future price, not option/combo price
                underlying_symbol = supporting_data.get('underlying_symbol', config.get('symbol', 'KC'))
                contract_month = supporting_data.get('contract_month')
                underlying_contract = None

                if contract_month:
                    underlying_contract = Future(
                        symbol=underlying_symbol,
                        lastTradeDateOrContractMonth=contract_month,
                        exchange=config.get('exchange', 'NYBOT')
                    )
                    try:
                        await asyncio.wait_for(ib.qualifyContractsAsync(underlying_contract), timeout=15)
                    except Exception as e:
                        logger.error(f"Failed to qualify underlying for straddle: {e}")
                        underlying_contract = None

                if not underlying_contract:
                    symbol = config.get('symbol', 'KC')
                    if active_futures_cache and symbol in active_futures_cache:
                        futures = active_futures_cache[symbol]
                        underlying_contract = futures[0] if futures else None
                    else:
                        try:
                            futures = await asyncio.wait_for(get_active_futures(ib, symbol, config['exchange'], count=1), timeout=15)
                            underlying_contract = futures[0] if futures else None
                            if active_futures_cache is not None and futures:
                                active_futures_cache[symbol] = futures
                        except Exception as e:
                            logger.error(f"Failed to get active futures for straddle validation: {e}")

                if underlying_contract:
                    current_price = await _get_current_price(ib, underlying_contract)
                else:
                    logger.warning("Cannot fetch underlying price for straddle — falling back to position contract")
                    current_price = await _get_current_price(ib, position.contract)

                if current_price > 0:
                    move_pct = abs((current_price - entry_price) / entry_price) * 100

                    if move_pct < theta_min_move:
                        return {
                            'action': 'CLOSE',
                            'reason': (
                                f"THETA BURN: {hours_held:.1f}h elapsed, only {move_pct:.2f}% move "
                                f"(threshold: {theta_min_move}%). Salvage residual value."
                            )
                        }
    return None


async def _validate_directional_spread(thesis: dict, guardian: str, council, config: dict, llm_budget_available: bool, ib: IB = None, active_futures_cache: dict = None) -> dict:
    """Validate Directional Spread using regime check + LLM narrative check."""
    exit_cfg = config.get('exit_logic', {})

    # E.2.C: Deterministic regime-aware exit (before LLM call)
    regime_exits_enabled = exit_cfg.get('enable_regime_breach_exits', True)
    if regime_exits_enabled and ib is not None:
        try:
            current_regime, iv_rank = await _get_current_regime_and_iv(ib, config)
            entry_regime = thesis.get('entry_regime', '').upper()
            # If entry was during a trending regime and market has shifted to range-bound,
            # the directional thesis is weakened — close the position
            if entry_regime == 'TRENDING' and current_regime == 'RANGE_BOUND':
                return {
                    'action': 'CLOSE',
                    'reason': (
                        f"REGIME BREACH: Entry regime was TRENDING, "
                        f"current regime is RANGE_BOUND (IV rank: {iv_rank:.0f}). "
                        f"Directional thesis weakened."
                    )
                }
        except Exception as e:
            logger.warning(f"Regime check failed in directional spread validation: {e} — continuing to LLM check")

    narrative_exits_enabled = exit_cfg.get('enable_narrative_exits', True)
    if not narrative_exits_enabled:
        logger.debug("Narrative exits disabled via config — skipping directional spread validation")
        return None
    elif not llm_budget_available:
        logger.info("LLM budget exhausted — skipping narrative thesis validation for directional spread")
        return None

    # Query the guardian agent for thesis validity
    primary_rationale = thesis.get('primary_rationale', '')
    invalidation_triggers = thesis.get('invalidation_triggers', [])

    # Get current sentinel data relevant to this thesis
    current_context = await _get_context_for_guardian(guardian, config)

    # Permabear Attack prompt
    attack_prompt = f"""You are stress-testing an existing position.

POSITION: {thesis.get('strategy_type')}
ORIGINAL THESIS: {primary_rationale}
KNOWN INVALIDATION TRIGGERS: {invalidation_triggers}

CURRENT MARKET CONTEXT:
{current_context}

QUESTION: Does the original thesis STILL HOLD?
- If ANY invalidation trigger has fired, return CLOSE.
- If the thesis is weakening but not dead, return HOLD with concerns.
- Be aggressive - we'd rather close early than ride a dead thesis.

Return JSON: {{"verdict": "HOLD" or "CLOSE", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
    confidence_threshold = exit_cfg.get('thesis_validation_confidence_threshold', 0.6)
    # Note: calling router directly requires accessing it from council or passing router
    if hasattr(council, 'router'):
        verdict_response = await council.router.route_and_call(
            model_key='permabear', # Use permabear model
            prompt=attack_prompt,
            response_model=None # Expecting JSON string or use strict mode if available
        )

        try:
            # Clean up response if needed (remove markdown)
            clean_response = verdict_response.replace('```json', '').replace('```', '')
            verdict_data = json.loads(clean_response)
            if not isinstance(verdict_data, dict):
                raise ValueError("Thesis validation returned non-dict JSON")
            if verdict_data.get('verdict') == 'CLOSE' and verdict_data.get('confidence', 0) > confidence_threshold:
                return {
                    'action': 'CLOSE',
                    'reason': f"NARRATIVE INVALIDATION: {verdict_data.get('reasoning', 'Thesis degraded')}"
                }
        except Exception as e:
            logger.error(f"Could not parse thesis validation response: {e}")

    return None


async def _validate_thesis(
    thesis: dict,
    position,
    council,
    config: dict,
    ib: IB,
    active_futures_cache: dict = None,
    llm_budget_available: bool = True
) -> dict:
    """
    Validates if a trade thesis still holds given current market conditions.

    Uses the Permabear/Permabull debate structure to stress-test the position.

    Returns:
        dict with 'action' ('HOLD', 'CLOSE', 'PRESS') and 'reason'
    """
    strategy_type = thesis.get('strategy_type', 'UNKNOWN')
    guardian = thesis.get('guardian_agent', 'Master')

    # A. REGIME-BASED VALIDATION (Iron Condor)
    if strategy_type == 'IRON_CONDOR':
        result = await _validate_iron_condor(thesis, config, ib, active_futures_cache)
        if result:
            return result

    # B. VOLATILITY/THETA VALIDATION (Long Straddle)
    elif strategy_type == 'LONG_STRADDLE':
        result = await _validate_long_straddle(thesis, position, config, ib, active_futures_cache)
        if result:
            return result

    # C. NARRATIVE-BASED VALIDATION (Directional Spreads)
    elif strategy_type in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
        result = await _validate_directional_spread(thesis, guardian, council, config, llm_budget_available, ib=ib, active_futures_cache=active_futures_cache)
        if result:
            return result

    # Default: HOLD
    return {'action': 'HOLD', 'reason': 'Thesis intact'}

def _find_position_id_for_contract(
    position,
    trade_ledger: pd.DataFrame,
    tms: TransactiveMemory = None
) -> str | None:
    """
    Map IB position to position_id using multi-strategy matching.

    v3.1 FIX: Three-tier matching strategy:
    1. Exact conId match with active thesis (highest confidence)
    2. Symbol + direction + open status match
    3. FIFO fallback with warning

    Args:
        position: IB Position object
        trade_ledger: DataFrame with trade history
        tms: TransactiveMemory for thesis status lookup (optional)

    Returns:
        position_id string or None
    """
    symbol = position.contract.localSymbol
    conId = position.contract.conId
    position_direction = 'BUY' if position.position > 0 else 'SELL'

    # Filter to matching symbol
    matches = trade_ledger[trade_ledger['local_symbol'] == symbol].copy()

    if matches.empty:
        logger.debug(f"No ledger entries found for symbol {symbol}")
        return None

    # === STRATEGY 1: Exact conId match with active thesis ===
    if tms is not None and conId:
        if 'conId' in matches.columns:
            conId_matches = matches[matches['conId'] == conId]
        else:
            conId_matches = pd.DataFrame()

        for pos_id in conId_matches['position_id'].unique():
            # Check if thesis is still active
            thesis = tms.retrieve_thesis(pos_id)
            if thesis and thesis.get('active', False):
                logger.debug(f"Matched {symbol} to {pos_id} via conId + active thesis")
                return pos_id

    # === STRATEGY 2: Symbol + direction + open status ===
    direction_matches = matches[matches['action'] == position_direction]
    if direction_matches.empty:
        direction_matches = matches

    # Find positions with non-zero net quantity
    open_positions = []
    for pos_id in direction_matches['position_id'].unique():
        pos_entries = trade_ledger[trade_ledger['position_id'] == pos_id]

        # Calculate net quantity for this symbol
        net_qty = 0
        for _, row in pos_entries.iterrows():
            if row['local_symbol'] == symbol:
                qty = row['quantity'] if row['action'] == 'BUY' else -row['quantity']
                net_qty += qty

        if net_qty != 0:
            entry_time = pos_entries['timestamp'].min()
            open_positions.append((pos_id, entry_time, net_qty))

    if open_positions:
        # Match by direction sign
        matching_sign = [
            p for p in open_positions
            if (p[2] > 0 and position.position > 0) or (p[2] < 0 and position.position < 0)
        ]

        if matching_sign:
            # If multiple matches, prefer the one with active thesis
            if tms is not None and len(matching_sign) > 1:
                for pos_id, _, _ in matching_sign:
                    thesis = tms.retrieve_thesis(pos_id)
                    if thesis and thesis.get('active', False):
                        logger.debug(f"Matched {symbol} to {pos_id} via direction + active thesis")
                        return pos_id

            # FIFO: oldest first
            matching_sign.sort(key=lambda x: x[1])
            logger.debug(f"Matched {symbol} to {matching_sign[0][0]} via direction + FIFO")
            return matching_sign[0][0]

        # No direction match - use oldest open
        open_positions.sort(key=lambda x: x[1])
        logger.warning(
            f"No direction match for {symbol}. Using oldest open position: {open_positions[0][0]}"
        )
        return open_positions[0][0]

    # === STRATEGY 3: FIFO fallback with warning ===
    logger.warning(f"No open position found for {symbol}. Using most recent entry (may be incorrect).")
    return matches.iloc[-1]['position_id']

def _build_thesis_invalidation_notification(
    position_id: str,
    thesis: dict,
    invalidation_reason: str,
    pnl: float = None
) -> tuple[str, str]:
    """
    Builds a rich notification for thesis invalidation.

    Returns:
        tuple of (title, message) for Pushover
    """
    strategy_type = thesis.get('strategy_type', 'Unknown')
    guardian = thesis.get('guardian_agent', 'Unknown')

    # Calculate holding time
    entry_time_str = thesis.get('entry_timestamp', '')
    if entry_time_str:
        entry_time = datetime.fromisoformat(entry_time_str)
        held_duration = datetime.now(timezone.utc) - entry_time
        held_hours = held_duration.total_seconds() / 3600
        held_str = f"{held_hours:.1f} hours"
    else:
        held_str = "Unknown"

    # Format P&L
    if pnl is not None:
        pnl_str = f"${pnl:+,.2f}"
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_section = f"<font color='{pnl_color}'><b>P&L: {pnl_str}</b></font>"
    else:
        pnl_section = "<i>P&L: Pending fill</i>"

    # Guardian icons
    guardian_icons = {
        'Agronomist': '🌱',
        'Logistics': '🚢',
        'VolatilityAnalyst': '📊',
        'Macro': '💹',
        'Sentiment': '🐦',
        'Master': '👑'
    }
    icon = guardian_icons.get(guardian, '🤖')

    # Build title
    title = f"🎯 Thesis Invalidated: {position_id}"

    # Build message
    message = f"""<b>Strategy:</b> {strategy_type.replace('_', ' ')}
<b>Guardian:</b> {icon} {guardian}

<b>📥 ENTRY THESIS:</b>
{thesis.get('primary_rationale', 'No rationale recorded')}

<b>❌ INVALIDATION REASON:</b>
{invalidation_reason}

<b>⏱️ Time Held:</b> {held_str}
<b>Entry Regime:</b> {thesis.get('entry_regime', 'Unknown')}
<b>Entry Confidence:</b> {thesis.get('supporting_data', {}).get('confidence', 0):.0%}

{pnl_section}"""

    return title, message

def _should_invalidate_futures_cache(cached_futures: list, config: dict) -> bool:
    """
    Check if if futures cache should be invalidated.

    v3.1: Invalidate when any cached contract is within 5 days of expiry.
    """
    if not cached_futures:
        return True

    now = datetime.now()
    warning_days = config.get('cache', {}).get('futures_expiry_warning_days', 5)

    for future in cached_futures:
        try:
            exp_str = future.lastTradeDateOrContractMonth
            if len(exp_str) == 8:
                exp_date = datetime.strptime(exp_str, '%Y%m%d')
                days_to_expiry = (exp_date - now).days

                if days_to_expiry <= warning_days:
                    logger.info(
                        f"Futures cache invalidation: {future.localSymbol} "
                        f"expires in {days_to_expiry} days"
                    )
                    return True
        except Exception as e:
            logger.debug(f"Could not parse expiry for {future}: {e}")

    return False


async def cleanup_orphaned_theses(config: dict):
    """
    Automated cleanup of orphaned theses.

    v3.1: Runs daily at 5 AM to clean up theses without IB positions.
    """
    logger.info("--- AUTOMATED THESIS CLEANUP ---")

    ib = None
    try:
        ib = await IBConnectionPool.get_connection("cleanup", config)
        tms = TransactiveMemory()
        trade_ledger = get_trade_ledger_df(config.get('data_dir'))

        cleaned = await _reconcile_orphaned_theses(ib, trade_ledger, tms, config)

        if cleaned > 0:
            logger.info(f"Thesis cleanup complete: {cleaned} orphaned theses invalidated")
            send_pushover_notification(
                config.get('notifications', {}),
                "🧹 Thesis Cleanup",
                f"Automated cleanup invalidated {cleaned} orphaned theses"
            )
        else:
            logger.info("Thesis cleanup: no orphans found")

        return cleaned

    except Exception as e:
        logger.error(f"Thesis cleanup failed: {e}")
        return 0
    finally:
        if ib is not None:
            try:
                await IBConnectionPool.release_connection("cleanup")
            except Exception:
                pass


async def check_and_recover_equity_data(config: dict) -> bool:
    """
    Check equity data freshness and trigger Flex query if stale.

    v3.1: Auto-recovery instead of just alerting.
    v7.2: Non-primary commodities copy equity from primary (account-wide metric).

    Returns:
        True if data is fresh or recovery succeeded
    """
    from equity_logger import sync_equity_from_flex
    from pathlib import Path
    import shutil

    data_dir = config.get('data_dir', 'data')
    equity_file = Path(os.path.join(data_dir, "daily_equity.csv"))
    max_staleness_hours = config.get('monitoring', {}).get('equity_max_staleness_hours', 24)
    is_primary = config.get('commodity', {}).get('is_primary', True)

    # Non-primary: equity is account-wide, copy from primary's data dir
    if not is_primary:
        primary_equity = Path("data/KC/daily_equity.csv")
        if primary_equity.exists():
            age_hours = (
                datetime.now() - datetime.fromtimestamp(primary_equity.stat().st_mtime)
            ).total_seconds() / 3600
            if age_hours <= max_staleness_hours:
                shutil.copy2(str(primary_equity), str(equity_file))
                logger.info(
                    f"Copied equity data from primary ({age_hours:.1f}h old)"
                )
                return True
            else:
                logger.warning(
                    f"Primary equity data is stale ({age_hours:.1f}h) — "
                    f"primary instance should refresh it"
                )
                # Still copy the stale data so we have something
                shutil.copy2(str(primary_equity), str(equity_file))
                return False
        else:
            logger.warning("Primary equity file not found")
            return False

    # Check staleness
    if equity_file.exists():
        file_age_hours = (
            datetime.now() - datetime.fromtimestamp(equity_file.stat().st_mtime)
        ).total_seconds() / 3600

        if file_age_hours <= max_staleness_hours:
            return True  # Data is fresh

        logger.warning(f"Equity data stale ({file_age_hours:.1f}h). Triggering Flex query...")
    else:
        logger.warning("Equity file missing. Triggering Flex query...")

    # Trigger recovery
    try:
        await sync_equity_from_flex(config)
        logger.info("Equity data recovery successful")
        return True
    except Exception as e:
        logger.error(f"Equity data recovery failed: {e}")
        send_pushover_notification(
            config.get('notifications', {}),
            "⚠️ Equity Data Stale",
            f"Auto-recovery failed. Manual intervention required.\nError: {e}"
        )
        return False


async def _reconcile_orphaned_theses(
    ib: IB,
    trade_ledger: pd.DataFrame,
    tms: TransactiveMemory,
    config: dict
) -> int:
    """
    Identifies and invalidates 'ghost theses' — TMS records with active=true
    that no longer have corresponding live positions in IB.

    This is a defense-in-depth safety net. Ghost theses are created when:
    - Post-close retry succeeds but thesis invalidation was skipped
    - Positions close externally (manual TWS, expiration, margin call)
    - System crash between fill and invalidation

    Returns the count of orphaned theses found and invalidated.

    Commodity-agnostic: Works for any symbol/exchange/strategy.
    """
    try:
        if not tms.collection:
            logger.debug("Reconciliation: TMS collection unavailable, skipping")
            return 0

        # 1. Get all active theses
        active_results = tms.collection.get(
            where={"active": "true"},
            include=['metadatas', 'documents']
        )

        active_thesis_ids = [
            meta.get('trade_id')
            for meta in active_results.get('metadatas', [])
            if meta.get('trade_id')
        ]

        if not active_thesis_ids:
            logger.debug("Reconciliation: No active theses to reconcile")
            return 0

        # 2. Build set of position_ids with live IB exposure
        try:
            live_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("Reconciliation: reqPositionsAsync timed out (30s), treating as empty")
            live_positions = []
        live_position_ids = set()

        for pos in live_positions:
            if pos.position == 0:
                continue
            pos_id = _find_position_id_for_contract(pos, trade_ledger)
            if pos_id:
                live_position_ids.add(pos_id)

        # 3. Identify orphans: active in TMS but not in IB
        orphaned_ids = [
            tid for tid in active_thesis_ids
            if tid not in live_position_ids
        ]

        if not orphaned_ids:
            logger.info(
                f"Reconciliation: All {len(active_thesis_ids)} active theses "
                f"have matching IB positions ✓"
            )
            return 0

        # 4. Invalidate orphans
        logger.warning(
            f"Reconciliation: Found {len(orphaned_ids)} orphaned theses "
            f"(active in TMS, no IB position): {orphaned_ids}"
        )

        for tid in orphaned_ids:
            try:
                tms.invalidate_thesis(
                    tid,
                    "Reconciliation: position closed externally or expired"
                )
                logger.info(f"Reconciliation: Invalidated ghost thesis {tid}")
            except Exception as e:
                logger.error(f"Reconciliation: Failed to invalidate {tid}: {e}")

        # 5. Notify
        try:
            summary = (
                f"🧹 Cleaned up {len(orphaned_ids)} ghost theses "
                f"(active in system, no matching IB positions).\n"
                f"Active theses checked: {len(active_thesis_ids)} | "
                f"Live IB positions: {len(live_position_ids)}"
            )
            send_pushover_notification(
                config.get('notifications', {}),
                "Thesis Reconciliation",
                summary
            )
        except Exception:
            pass  # Notification failure is non-fatal

        return len(orphaned_ids)

    except Exception as e:
        logger.error(f"Reconciliation failed (non-fatal): {e}")
        return 0


def _group_positions_by_thesis(
    positions: list,
    trade_ledger: object,
    tms: TransactiveMemory
) -> dict:
    """
    Groups IB position legs by their shared thesis (position_id).

    Returns:
        Dict mapping position_id → {
            'thesis': dict,         # The shared thesis
            'legs': list,           # List of IB Position objects
            'position_id': str      # The position_id key
        }

    Design: Commodity-agnostic — works for any multi-leg strategy.
    """
    groups = {}  # position_id → {thesis, legs}
    unmapped = []  # Legs we couldn't map

    for pos in positions:
        if pos.position == 0:
            continue

        # v3.1: Pass tms to _find_position_id_for_contract for robust matching
        position_id = _find_position_id_for_contract(pos, trade_ledger, tms)
        if not position_id:
            unmapped.append(pos)
            continue

        if position_id not in groups:
            thesis = tms.retrieve_thesis(position_id)
            groups[position_id] = {
                'thesis': thesis,
                'legs': [],
                'position_id': position_id
            }

        groups[position_id]['legs'].append(pos)

    if unmapped:
        logger.warning(
            f"Position audit: {len(unmapped)} legs could not be mapped to a thesis: "
            f"{[p.contract.localSymbol for p in unmapped]}"
        )

    return groups

async def _close_spread_position(
    ib: IB,
    legs: list,
    position_id: str,
    reason: str,
    config: dict,
    thesis: dict = None
):
    """
    Closes all legs of a spread position.

    Improvements over _close_position_with_thesis_reason:
    1. Qualifies contracts before placing orders (fixes Error 321)
    2. Verifies each order actually filled before claiming success
    3. Handles multi-leg positions atomically where possible
    4. Falls back to individual leg closure if BAG order fails

    Commodity-agnostic: Works for any multi-leg strategy on any exchange.
    """
    logger.info(
        f"Executing SPREAD CLOSE for {position_id} "
        f"({len(legs)} legs): {reason}"
    )

    # --- Step 1: Re-qualify ALL contracts by conId ---
    # CRITICAL FIX (2026-02-03): IBKR returns KC option positions with strikes
    # in cents (307.5), but the order API expects dollars (3.075). We MUST
    # re-qualify every contract using ONLY the conId so IB populates the
    # correct strike format. Previous code skipped this for contracts that
    # already had an exchange set, which was always the case for positions.
    qualified_legs = []
    for leg in legs:
        original_contract = leg.contract
        try:
            # Build a minimal contract with ONLY the conId.
            # This forces IB to populate all fields from its database,
            # including the correctly-formatted strike price.
            minimal = Contract(conId=original_contract.conId)
            qualified = await asyncio.wait_for(ib.qualifyContractsAsync(minimal), timeout=15)
            if qualified and qualified[0].conId != 0:
                qualified_legs.append(type(leg)(
                    account=leg.account,
                    contract=qualified[0],
                    position=leg.position,
                    avgCost=leg.avgCost
                ))
                logger.debug(
                    f"Re-qualified {original_contract.localSymbol}: "
                    f"strike {original_contract.strike} -> {qualified[0].strike}, "
                    f"exchange={qualified[0].exchange}"
                )
            else:
                logger.error(
                    f"Contract re-qualification returned invalid result for "
                    f"{original_contract.localSymbol} (conId={original_contract.conId}) "
                    f"— using original (may fail with Error 478)"
                )
                qualified_legs.append(leg)
        except Exception as e:
            logger.error(
                f"Contract re-qualification failed for "
                f"{original_contract.localSymbol}: {e} — using original"
            )
            qualified_legs.append(leg)

    # --- Step 2: Close each leg individually ---
    # (BAG orders require additional combo definition logic; individual
    #  leg closure is more reliable for thesis-based exits)
    successful_closes = []
    failed_closes = []

    for leg in qualified_legs:
        contract = leg.contract
        action = 'SELL' if leg.position > 0 else 'BUY'
        qty = abs(leg.position)

        try:
            order = MarketOrder(action, qty)
            trade = place_order(ib, contract, order)
            await asyncio.sleep(3)  # Allow time for fill

            # --- Step 3: Verify fill status ---
            if trade.orderStatus.status == 'Filled':
                successful_closes.append({
                    'symbol': contract.localSymbol,
                    'action': action,
                    'qty': qty,
                    'fill_price': trade.orderStatus.avgFillPrice
                })
                logger.info(
                    f"  ✅ {action} {qty}x {contract.localSymbol} "
                    f"@ {trade.orderStatus.avgFillPrice}"
                )
            else:
                failed_closes.append({
                    'symbol': contract.localSymbol,
                    'status': trade.orderStatus.status,
                    'error': str(trade.log[-1].message if trade.log else 'Unknown')
                })
                logger.error(
                    f"  ❌ {contract.localSymbol}: "
                    f"{trade.orderStatus.status} — "
                    f"{trade.log[-1].message if trade.log else 'No message'}"
                )
                # Cancel pending order to avoid rogue fills
                if trade.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive'):
                    try:
                        ib.cancelOrder(trade.order)
                    except Exception:
                        pass

        except Exception as e:
            failed_closes.append({
                'symbol': contract.localSymbol,
                'status': 'EXCEPTION',
                'error': str(e)
            })
            logger.error(f"  ❌ {contract.localSymbol}: Exception — {e}")

        await asyncio.sleep(0.5)  # Throttle between legs

    # --- Step 4: Send accurate notification ---
    total_legs = len(qualified_legs)
    success_count = len(successful_closes)
    # fail_count = len(failed_closes)

    if success_count == total_legs:
        # Full success
        leg_symbols = [sc['symbol'] for sc in successful_closes] if successful_closes else []
        readable_symbol = leg_symbols[0].split()[0] if leg_symbols else "Unknown"
        title = f"✅ Position Closed: {readable_symbol}"
        message = (
            f"Reason: {reason}\n"
            f"Closed {success_count}/{total_legs} legs successfully.\n"
        )
        for sc in successful_closes:
            message += f"  {sc['action']} {sc['qty']}x {sc['symbol']} @ ${sc['fill_price']:.4f}\n"
    elif success_count > 0:
        # Partial success — DANGEROUS, position is now unbalanced
        leg_symbols = [sc['symbol'] for sc in successful_closes] + [fc['symbol'] for fc in failed_closes]
        readable_symbol = leg_symbols[0].split()[0] if leg_symbols else "Unknown"
        title = f"⚠️ PARTIAL CLOSE: {readable_symbol}"
        message = (
            f"Reason: {reason}\n"
            f"⚠️ ONLY {success_count}/{total_legs} legs closed!\n"
            f"MANUAL INTERVENTION REQUIRED — position may have naked exposure.\n\n"
            f"Successful:\n"
        )
        for sc in successful_closes:
            message += f"  ✅ {sc['action']} {sc['qty']}x {sc['symbol']} @ ${sc['fill_price']:.4f}\n"
        message += "\nFailed:\n"
        for fc in failed_closes:
            message += f"  ❌ {fc['symbol']}: {fc['status']} — {fc['error']}\n"
    else:
        # Total failure — DO NOT invalidate thesis
        leg_symbols = [fc['symbol'] for fc in failed_closes]
        readable_symbol = leg_symbols[0].split()[0] if leg_symbols else "Unknown"
        title = f"❌ CLOSE FAILED: {readable_symbol}"
        message = (
            f"Reason: {reason}\n"
            f"ALL {total_legs} close orders FAILED.\n"
            f"Position remains open. Will retry on next audit cycle.\n\n"
        )
        for fc in failed_closes:
            message += f"  ❌ {fc['symbol']}: {fc['status']} — {fc['error']}\n"

    send_pushover_notification(config.get('notifications', {}), title, message)

    # --- Step 5: Cleanup catastrophe stops (only if fully closed) ---
    if success_count == total_legs:
        await close_spread_with_protection_cleanup(
            ib, None, f"CATASTROPHE_{position_id}"
        )

    # --- Step 6: Return success status ---
    # Caller should only invalidate thesis if ALL legs closed
    return success_count == total_legs


async def _reconcile_state_stores(
    ib: IB,
    trade_ledger: pd.DataFrame,
    tms: TransactiveMemory,
    config: dict
) -> dict:
    """
    Reconcile state across IBKR, CSV ledger, and TMS.

    v3.1: Three-body sync check runs before every position audit.
    IBKR is source of truth. Mismatches generate warnings for investigation.

    Returns:
        Dict with reconciliation results and any discrepancies found
    """
    results = {
        'ibkr_positions': 0,
        'ledger_open': 0,
        'tms_active': 0,
        'discrepancies': [],
        'reconciled': True
    }

    try:
        # 1. Count IBKR positions (ground truth) — use fresh data, not stale cache
        try:
            all_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("reqPositionsAsync timed out (30s) in state reconciliation")
            all_positions = ib.positions()  # Fall back to cached if timeout
        symbol = config.get('symbol', 'KC')
        ib_positions = [
            p for p in all_positions
            if p.position != 0 and p.contract.symbol == symbol
        ]
        results['ibkr_positions'] = len(ib_positions)

        # 2. Count open positions in ledger
        if not trade_ledger.empty and 'position_id' in trade_ledger.columns:
            # Group by position_id and check net quantity
            for pos_id in trade_ledger['position_id'].unique():
                pos_entries = trade_ledger[trade_ledger['position_id'] == pos_id]
                net_qty = 0
                for _, row in pos_entries.iterrows():
                    qty = row['quantity'] if row['action'] == 'BUY' else -row['quantity']
                    net_qty += qty
                if net_qty != 0:
                    results['ledger_open'] += 1

        # 3. Count active theses in TMS
        active_theses = tms.collection.get(
            where={"active": "true"},
            include=['metadatas']
        )
        results['tms_active'] = len(active_theses.get('metadatas', []))

        # 4. Check for discrepancies
        if results['ibkr_positions'] != results['ledger_open']:
            results['discrepancies'].append(
                f"IBKR has {results['ibkr_positions']} positions but ledger shows {results['ledger_open']} open"
            )

        if results['ibkr_positions'] != results['tms_active']:
            results['discrepancies'].append(
                f"IBKR has {results['ibkr_positions']} positions but TMS has {results['tms_active']} active theses"
            )

        if results['discrepancies']:
            results['reconciled'] = False
            for disc in results['discrepancies']:
                logger.warning(f"State sync discrepancy: {disc}")

            # Send notification for manual review
            send_pushover_notification(
                config.get('notifications', {}),
                f"⚠️ {symbol} State Sync Warning",
                f"[{symbol}] Discrepancies found:\n" + "\n".join(results['discrepancies'])
            )
        else:
            logger.info(
                f"State stores reconciled: {results['ibkr_positions']} positions across all stores"
            )

        return results

    except Exception as e:
        logger.error(f"State reconciliation failed: {e}")
        results['reconciled'] = False
        results['discrepancies'].append(f"Reconciliation error: {e}")
        return results


async def run_position_audit_cycle(config: dict, trigger_source: str = "Scheduled"):
    """
    Reviews all active positions against their original theses.
    Now operates on GROUPED positions (spread-aware).
    """
    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logger.info("[OFF] run_position_audit_cycle skipped — no positions exist in OFF mode")
        return

    logger.info(f"--- POSITION AUDIT CYCLE ({trigger_source}) ---")

    llm_budget_available = not (GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit)
    if not llm_budget_available:
        logger.info("Budget hit — position audit will skip LLM-based checks (IC/straddle checks still active)")

    ib = None
    try:
        ib = await IBConnectionPool.get_connection("audit", config)
        configure_market_data_type(ib)

        # === L5 FIX: Reconcile state stores before audit ===
        trade_ledger = get_trade_ledger_df()
        tms = TransactiveMemory()

        recon_results = await _reconcile_state_stores(ib, trade_ledger, tms, config)
        if not recon_results['reconciled']:
            logger.warning(
                f"Proceeding with audit despite {len(recon_results['discrepancies'])} discrepancies. "
                f"IBKR positions are source of truth."
            )

        # 1. Get current positions from IB (filtered to this commodity)
        commodity_symbol = config.get('symbol', 'KC')
        try:
            all_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("reqPositionsAsync timed out (30s) in position audit, treating as empty")
            all_positions = []
        live_positions = [p for p in all_positions if p.position != 0 and p.contract.symbol == commodity_symbol]
        if not live_positions:
            logger.info("No open positions to audit.")

            # Reconcile: If TMS has active theses but IB has no positions,
            # ALL active theses are ghosts
            try:
                orphan_count = await _reconcile_orphaned_theses(
                    ib, trade_ledger, tms, config
                )
                if orphan_count > 0:
                    logger.warning(
                        f"Reconciliation cleaned up {orphan_count} ghost theses "
                        f"(IB has zero positions)"
                    )
            except Exception as e:
                logger.warning(f"Post-audit reconciliation failed (non-fatal): {e}")

            return

        # 2. Initialize components
        tms = TransactiveMemory()
        council = TradingCouncil(config)
        positions_to_close = []

        # 3. Get trade ledger for position mapping
        trade_ledger = get_trade_ledger_df()

        # 4. === NEW: Group legs by thesis ===
        position_groups = _group_positions_by_thesis(live_positions, trade_ledger, tms)
        logger.info(
            f"Grouped {len(live_positions)} IB positions into "
            f"{len(position_groups)} thesis groups"
        )

        # 5. === NEW: Cache active futures (Issue 9 fix) ===
        active_futures_cache = {}
        try:
            symbol = config.get('symbol', 'KC')
            exchange = config.get('exchange', 'NYBOT')

            # === K1 FIX: Check cache validity before use ===
            cached_futures = active_futures_cache.get(symbol)

            if _should_invalidate_futures_cache(cached_futures, config):
                logger.info("Refreshing active futures cache")
                futures = await asyncio.wait_for(get_active_futures(ib, symbol, exchange, count=5), timeout=30)
                active_futures_cache[symbol] = futures
            else:
                futures = cached_futures

            # If still no futures (e.g. first run), fetch
            if not futures:
                 futures = await asyncio.wait_for(get_active_futures(ib, symbol, exchange, count=5), timeout=30)
                 active_futures_cache[symbol] = futures

        except Exception as e:
            logger.warning(f"Failed to pre-cache active futures: {e}")

        # 6. Audit each GROUP (not each leg)
        for position_id, group in position_groups.items():
            thesis = group['thesis']
            legs = group['legs']

            if not thesis:
                logger.info(
                    f"No thesis found for {position_id} "
                    f"({len(legs)} legs) — using default aging rules"
                )
                continue

            # === E.2.A + E.2.B: Deterministic P&L exits and DTE acceleration ===
            # Run BEFORE LLM thesis validation — cheaper and more authoritative for numerical exits
            try:
                metrics = await _calculate_combo_risk_metrics(ib, config, legs)
                if metrics:
                    risk_cfg = config.get('risk_management', {})
                    take_profit_pct = risk_cfg.get('take_profit_capture_pct', 0.80)
                    stop_loss_pct = risk_cfg.get('stop_loss_max_risk_pct', 0.50)

                    # E.2.B: DTE-aware exit acceleration
                    dte_cfg = config.get('exit_logic', {}).get('dte_acceleration', {})
                    dte_enabled = dte_cfg.get('enabled', False)
                    dte = None
                    if dte_enabled:
                        try:
                            first_leg = legs[0]
                            expiry_str = first_leg.contract.lastTradeDateOrContractMonth
                            expiry_date = datetime.strptime(expiry_str, '%Y%m%d').date()
                            dte = (expiry_date - datetime.now().date()).days

                            force_close_dte = dte_cfg.get('force_close_dte', 3)
                            if dte <= force_close_dte:
                                positions_to_close.append({
                                    'position_id': position_id,
                                    'legs': legs,
                                    'reason': f"DTE FORCE CLOSE: {dte} days to expiry (<= {force_close_dte})",
                                    'thesis': thesis
                                })
                                logger.warning(
                                    f"DTE FORCE CLOSE: {position_id} — "
                                    f"{dte} DTE (<= {force_close_dte})"
                                )
                                continue

                            accel_dte = dte_cfg.get('acceleration_dte', 14)
                            if dte <= accel_dte:
                                take_profit_pct = dte_cfg.get('accelerated_take_profit_pct', 0.50)
                                stop_loss_pct = dte_cfg.get('accelerated_stop_loss_pct', 0.30)
                                logger.info(
                                    f"DTE ACCELERATION: {position_id} — "
                                    f"{dte} DTE, using tightened thresholds "
                                    f"(TP={take_profit_pct:.0%}, SL={stop_loss_pct:.0%})"
                                )
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"DTE parsing failed for {position_id}: {e} — using standard thresholds")

                    capture_pct = metrics['capture_pct']
                    risk_pct = metrics['risk_pct']

                    if capture_pct >= take_profit_pct:
                        positions_to_close.append({
                            'position_id': position_id,
                            'legs': legs,
                            'reason': (
                                f"TAKE PROFIT: Captured {capture_pct:.1%} of max profit "
                                f"(threshold: {take_profit_pct:.0%})"
                                + (f", DTE={dte}" if dte is not None else "")
                            ),
                            'thesis': thesis
                        })
                        logger.warning(
                            f"TAKE PROFIT: {position_id} — "
                            f"capture {capture_pct:.1%} >= {take_profit_pct:.0%}"
                        )
                        continue

                    if risk_pct <= -abs(stop_loss_pct):
                        positions_to_close.append({
                            'position_id': position_id,
                            'legs': legs,
                            'reason': (
                                f"STOP LOSS: Risk at {risk_pct:.1%} of max loss "
                                f"(threshold: -{stop_loss_pct:.0%})"
                                + (f", DTE={dte}" if dte is not None else "")
                            ),
                            'thesis': thesis
                        })
                        logger.warning(
                            f"STOP LOSS: {position_id} — "
                            f"risk {risk_pct:.1%} <= -{stop_loss_pct:.0%}"
                        )
                        continue
            except Exception as e:
                logger.warning(f"P&L/DTE check failed for {position_id}: {e} — falling through to thesis validation")

            # Use first leg as representative for price checks
            # (underlying price is the same regardless of which leg we check)
            representative_leg = legs[0]

            # 7. Run thesis validation ONCE per group
            verdict = await _validate_thesis(
                thesis=thesis,
                position=representative_leg,
                council=council,
                config=config,
                ib=ib,
                active_futures_cache=active_futures_cache,  # Issue 9 fix
                llm_budget_available=llm_budget_available
            )

            if verdict['action'] == 'CLOSE':
                positions_to_close.append({
                    'position_id': position_id,
                    'legs': legs,       # ALL legs, not just one
                    'reason': verdict['reason'],
                    'thesis': thesis
                })
                logger.warning(
                    f"THESIS INVALIDATED: {position_id} "
                    f"({len(legs)} legs) — {verdict['reason']}"
                )

        # 8. Execute closures (spread-aware)
        for item in positions_to_close:
            fully_closed = await _close_spread_position(
                ib=ib,
                legs=item['legs'],
                position_id=item['position_id'],
                reason=item['reason'],
                config=config,
                thesis=item['thesis']
            )
            # CRITICAL: Only invalidate thesis if ALL legs actually closed
            if fully_closed:
                tms.invalidate_thesis(item['position_id'], item['reason'])
            else:
                logger.error(
                    f"Thesis {item['position_id']} NOT invalidated — "
                    f"close order did not fully succeed. "
                    f"Will retry on next audit cycle."
                )

        # 8.5 === Reconcile orphaned theses ===
        # After auditing known positions, check for ghost theses that
        # somehow stayed active despite their positions being closed.
        try:
            orphan_count = await _reconcile_orphaned_theses(
                ib, trade_ledger, tms, config
            )
            if orphan_count > 0:
                logger.warning(
                    f"Reconciliation cleaned up {orphan_count} ghost theses "
                    f"during position audit"
                )
        except Exception as e:
            logger.warning(f"Post-audit reconciliation failed (non-fatal): {e}")

        # 9. Summary notification
        if positions_to_close:
            summary = (
                f"Closed {len(positions_to_close)} positions "
                f"via thesis invalidation:\n"
            )
            summary += "\n".join([
                f"- {p['legs'][0].contract.localSymbol.split()[0] if p['legs'] else 'Unknown'} "
                f"({len(p['legs'])} legs): {p['reason']}"
                for p in positions_to_close
            ])
            send_pushover_notification(
                config.get('notifications', {}),
                "Position Audit Complete",
                summary
            )
        else:
            logger.info("Position audit complete. All theses remain valid.")

        # E.1: Post-audit VaR computation + AI Risk Agent
        try:
            from trading_bot.var_calculator import get_var_calculator, run_risk_agent
            var_calc = get_var_calculator(config)
            prev_var = var_calc.get_cached_var()
            var_result = await asyncio.wait_for(
                var_calc.compute_portfolio_var(ib, config), timeout=30.0
            )
            logger.info(
                f"Post-audit VaR: 95%={var_result.var_95_pct:.2%} "
                f"(${var_result.var_95:,.0f}), positions={var_result.position_count}"
            )

            # Run AI Risk Agent (L1 + L2)
            try:
                agent_output = await asyncio.wait_for(
                    run_risk_agent(var_result, config, ib, prev_var), timeout=30.0
                )
                if agent_output.get('interpretation'):
                    var_result.narrative = agent_output['interpretation']
                if agent_output.get('scenarios'):
                    var_result.scenarios = agent_output['scenarios']
                var_calc._save_state(var_result)
            except asyncio.TimeoutError:
                logger.warning("Risk Agent timed out after 30s (non-fatal, VaR saved)")
            except Exception as agent_e:
                logger.warning(f"Risk Agent failed (non-fatal, VaR saved): {agent_e}")

            # Pushover alert if VaR is elevated
            enforcement_mode = config.get('compliance', {}).get('var_enforcement_mode', 'log_only')
            var_warning = config.get('compliance', {}).get('var_warning_pct', 0.02)
            if enforcement_mode != 'log_only' and var_result.var_95_pct > var_warning:
                send_pushover_notification(
                    config.get('notifications', {}),
                    "Portfolio VaR Alert",
                    f"VaR(95%) = {var_result.var_95_pct:.1%} of equity "
                    f"(${var_result.var_95:,.0f}) — limit is "
                    f"{config.get('compliance', {}).get('var_limit_pct', 0.03):.0%}"
                )
        except asyncio.TimeoutError:
            logger.warning("Post-audit VaR computation timed out after 30s (non-fatal)")
        except Exception as var_e:
            logger.warning(f"Post-audit VaR computation failed (non-fatal): {var_e}")

    except Exception as e:
        logger.exception(f"Position Audit Cycle failed: {e}")
    finally:
        if ib is not None:
            try:
                await IBConnectionPool.release_connection("audit")
            except Exception:
                pass


async def log_stream(stream, logger_func):
    """Reads and logs lines from a subprocess stream."""
    while True:
        line = await stream.readline()
        if line:
            logger_func(line.decode('utf-8').strip())
        else:
            break


async def start_monitoring(config: dict):
    """Starts the `position_monitor.py` script as a background process."""
    global monitor_process, _SYSTEM_SHUTDOWN

    # Reset shutdown flag for new trading day
    _SYSTEM_SHUTDOWN = False
    logger.info("System shutdown flag CLEARED — new trading day beginning")

    # === EARLY EXIT: Don't start monitor on non-trading days ===
    if not is_market_open(config):
        logger.info("Market is closed (weekend/holiday). Skipping position monitoring startup.")
        return

    if monitor_process and monitor_process.returncode is None:
        logger.warning("Monitoring process is already running.")
        return

    try:
        logger.info("--- Starting position monitoring process ---")
        monitor_process = await asyncio.create_subprocess_exec(
            sys.executable, 'position_monitor.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE  # Capture both stdout and stderr
        )
        logger.info(f"Successfully started position monitor with PID: {monitor_process.pid}")

        # Create tasks to log the output from the monitor process in the background
        asyncio.create_task(log_stream(monitor_process.stdout, logger.info))
        asyncio.create_task(log_stream(monitor_process.stderr, logger.error))

        logger.info("Started position monitoring service.")
    except Exception as e:
        logger.critical(f"Failed to start position monitor: {e}", exc_info=True)
        send_pushover_notification(config.get('notifications', {}), "Orchestrator CRITICAL", "Failed to start position monitor.")


async def stop_monitoring(config: dict):
    """Stops the background position monitoring process."""
    global monitor_process
    if not monitor_process or monitor_process.returncode is not None:
        logger.warning("Monitoring process is not running or has already terminated.")
        return

    try:
        logger.info(f"--- Stopping position monitoring process (PID: {monitor_process.pid}) ---")
        monitor_process.terminate()
        await monitor_process.wait()
        logger.info("Position monitoring process has been successfully terminated.")
        monitor_process = None
    except ProcessLookupError:
        logger.warning("Process already terminated.")
    except Exception as e:
        logger.critical(f"An error occurred while stopping the monitor: {e}", exc_info=True)


async def cancel_and_stop_monitoring(config: dict):
    """Wrapper task to cancel open orders and then stop the monitor."""
    global _SYSTEM_SHUTDOWN

    logger.info("--- Initiating end-of-day shutdown sequence ---")
    _SYSTEM_SHUTDOWN = True  # Set BEFORE canceling orders
    logger.info("System shutdown flag SET — no new trades or emergency cycles will be processed")

    await cancel_all_open_orders(config)
    await stop_monitoring(config)

    # v5.4 Fix: Release pooled connections to prevent "Peer closed" errors
    # at 20:00 UTC Gateway restart. Post-shutdown tasks (equity logging,
    # reconciliation) use their own self-managed connections, not the pool.
    try:
        await IBConnectionPool.release_all()
        logger.info("Connection pool released — no stale connections for Gateway restart")
    except Exception as e:
        logger.warning(f"Pool cleanup during shutdown: {e}")

    logger.info("--- End-of-day shutdown sequence complete ---")


def get_next_task(now_utc: datetime, task_schedule):
    """Calculates the next scheduled task.

    The schedule times are in NY Local Time.
    We calculate the corresponding UTC run time dynamically to handle DST.
    Automatically skips weekends (Saturday/Sunday).

    Accepts either:
      - list[ScheduledTask] → returns (datetime, ScheduledTask)
      - dict{time: callable} → returns (datetime, callable)  [legacy compat]
    """
    # Normalize input: convert legacy dict to list of ScheduledTask
    if isinstance(task_schedule, dict):
        items = [
            ScheduledTask(
                id=func.__name__, time_et=rt, function=func,
                func_name=func.__name__, label=func.__name__,
            )
            for rt, func in task_schedule.items()
        ]
        _return_callable = True
    else:
        items = task_schedule
        _return_callable = False

    ny_tz = pytz.timezone('America/New_York')
    utc = pytz.UTC
    now_ny = now_utc.astimezone(ny_tz)

    # === WEEKEND SKIP LOGIC ===
    # If today is Saturday (5) or Sunday (6) in NY, advance to Monday
    if now_ny.weekday() == 5:  # Saturday
        days_to_monday = 2
        now_ny = (now_ny + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Weekend detected (Saturday). Next trading day: Monday {now_ny.strftime('%Y-%m-%d')}")
    elif now_ny.weekday() == 6:  # Sunday
        days_to_monday = 1
        now_ny = (now_ny + timedelta(days=days_to_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Weekend detected (Sunday). Next trading day: Monday {now_ny.strftime('%Y-%m-%d')}")

    next_run_utc, next_task = None, None

    for task in items:
        rt = task.time_et
        # Construct run time in NY for today
        try:
            candidate_ny = now_ny.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
        except ValueError:
            continue

        # If this time has passed in NY, move to tomorrow
        if candidate_ny <= now_ny:
             candidate_ny += timedelta(days=1)

        # === CHECK IF CANDIDATE IS ON WEEKEND ===
        if candidate_ny.weekday() == 5:  # Saturday -> Move to Monday
            candidate_ny += timedelta(days=2)
        elif candidate_ny.weekday() == 6: # Sunday -> Move to Monday
            candidate_ny += timedelta(days=1)

        # Convert to UTC
        candidate_utc = candidate_ny.astimezone(utc)

        if next_run_utc is None or candidate_utc < next_run_utc:
            next_run_utc = candidate_utc
            next_task = task

    if _return_callable and next_task is not None:
        return next_run_utc, next_task.function
    return next_run_utc, next_task


async def analyze_and_archive(config: dict):
    """
    Triggers the performance analysis and then archives the trade ledger.
    """
    logger.info("--- Initiating end-of-day analysis and archiving ---")
    try:
        await run_performance_analysis(config)
        archive_trade_ledger()

        # Log TMS Effectiveness
        try:
            tms = TransactiveMemory()
            stats = tms.get_collection_stats()
            logger.info(f"TMS Diagnostics: Status={stats.get('status')}, Docs={stats.get('document_count')}")
        except Exception as e:
            logger.warning(f"Failed to log TMS diagnostics: {e}")

        logger.info("--- End-of-day analysis and archiving complete ---")
    except Exception as e:
        logger.critical(f"An error occurred during the analysis and archiving process: {e}", exc_info=True)


async def reconcile_and_notify(config: dict):
    """Runs the trade reconciliation and sends a notification if discrepancies are found."""
    logger.info("--- Starting trade reconciliation ---")
    try:
        missing_df, superfluous_df = await run_reconciliation()

        if not missing_df.empty or not superfluous_df.empty:
            logger.warning("Trade reconciliation found discrepancies.")
            message = ""
            if not missing_df.empty:
                message += f"Found {len(missing_df)} missing trades in the local ledger.\n"
            if not superfluous_df.empty:
                message += f"Found {len(superfluous_df)} superfluous trades in the local ledger.\n"
            message += "Check the `archive_ledger` directory for details."

            send_pushover_notification(
                config.get('notifications', {}),
                "Trade Reconciliation Alert",
                message
            )
        else:
            logger.info("Trade reconciliation complete. No discrepancies found.")

        await reconcile_active_positions(config)
        await reconcile_council_history(config)

    except Exception as e:
        logger.critical(f"An error occurred during trade reconciliation: {e}", exc_info=True)


async def sentinel_effectiveness_check(config: dict):
    """
    Meta-monitor: alerts if significant price move occurred with zero sentinel trades.
    Runs daily at end of session.
    """
    logger.info("--- Sentinel Effectiveness Check ---")
    try:
        from config.commodity_profiles import get_commodity_profile

        ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
        profile = get_commodity_profile(ticker)
        significance_threshold = config.get('sentinels', {}).get('meta_monitor_threshold_pct', 5.0)

        # Get weekly price change from IB or yfinance
        weekly_change_pct = None
        try:
            import yfinance as yf
            yf_ticker = getattr(profile, 'yfinance_ticker', f"{profile.contract.symbol}=F")
            data = yf.Ticker(yf_ticker).history(period="5d")
            if data is not None and len(data) >= 2:
                weekly_change_pct = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        except Exception as e:
            logger.warning(f"yfinance fetch failed for meta-monitor: {e}")

        if weekly_change_pct is None:
            logger.info("Could not fetch weekly price data for meta-monitor. Skipping.")
            return

        # Check sentinel trade stats
        stats = SENTINEL_STATS.get_all()
        total_alerts = sum(s.get('total_alerts', 0) for s in stats.values())
        total_sentinel_trades = sum(s.get('trades_triggered', 0) for s in stats.values())

        if abs(weekly_change_pct) > significance_threshold and total_sentinel_trades == 0:
            severity = "🔴" if abs(weekly_change_pct) > 10.0 else "🟡"

            if total_alerts == 0:
                diagnosis = "No alerts fired — check sentinel thresholds and connectivity"
            else:
                # Check if council decisions were made but DA vetoed them
                da_veto_count = 0
                council_decisions_today = 0
                try:
                    import pandas as pd
                    ch_path = os.path.join(config.get('data_dir', 'data'), 'council_history.csv')
                    if os.path.exists(ch_path):
                        ch_df = pd.read_csv(ch_path)
                        if not ch_df.empty and 'timestamp' in ch_df.columns:
                            ch_df['timestamp'] = pd.to_datetime(ch_df['timestamp'], utc=True, errors='coerce')
                            today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                            today_mask = ch_df['timestamp'].dt.strftime('%Y-%m-%d') == today_str
                            council_decisions_today = today_mask.sum()
                            if 'master_reasoning' in ch_df.columns:
                                da_veto_count = (
                                    today_mask & ch_df['master_reasoning'].str.contains(
                                        r'\[DA VETO:', na=False, case=False
                                    )
                                ).sum()
                except Exception:
                    pass  # Fall through to generic diagnosis

                if da_veto_count > 0:
                    diagnosis = (
                        f"Pipeline working — DA vetoed {da_veto_count}/{council_decisions_today} "
                        f"council decisions today (risk management working as intended)"
                    )
                    severity = "🟡"  # Downgrade: DA vetoes are intentional, not a failure
                elif council_decisions_today > 0:
                    diagnosis = (
                        f"Council ran {council_decisions_today} decisions but no trades placed — "
                        f"check compliance/order generation"
                    )
                else:
                    diagnosis = "Alerts fired but no council decisions — check council pipeline and debounce"

            msg = (
                f"{severity} SENTINEL EFFECTIVENESS ALERT\n"
                f"{profile.name} weekly change: {weekly_change_pct:+.1f}%\n"
                f"Sentinel alerts: {total_alerts}, Trades from sentinels: {total_sentinel_trades}\n"
                f"Diagnosis: {diagnosis}"
            )
            logger.warning(msg)
            send_pushover_notification(
                config.get('notifications', {}),
                f"{severity} Sentinel Effectiveness",
                msg
            )
        else:
            logger.info(
                f"Sentinel effectiveness OK: weekly change {weekly_change_pct:+.1f}%, "
                f"alerts: {total_alerts}, sentinel trades: {total_sentinel_trades}"
            )

    except Exception as e:
        logger.error(f"Sentinel effectiveness check failed: {e}", exc_info=True)


async def reconcile_and_analyze(config: dict):
    """Runs reconciliation, then analysis and archiving."""
    logger.info("--- Kicking off end-of-day reconciliation and analysis process ---")

    await sync_equity_from_flex(config)

    # Check equity staleness
    try:
        equity_file = os.path.join(config.get('data_dir', 'data'), "daily_equity.csv")
        if os.path.exists(equity_file):
            import pandas as pd
            eq_df = pd.read_csv(equity_file)
            if not eq_df.empty and 'timestamp' in eq_df.columns:
                eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'], utc=True)
                last_ts = eq_df['timestamp'].max()
                now_utc = datetime.now(timezone.utc)
                age_hours = (now_utc - last_ts).total_seconds() / 3600

                # Check for staleness on weekdays (allow weekend staleness)
                is_weekday = now_utc.weekday() < 5
                if is_weekday and age_hours > 24:
                    msg = f"⚠️ Equity data is {age_hours:.1f} hours stale."
                    logger.warning(msg)
                    send_pushover_notification(config.get('notifications', {}), "Equity Data Stale", msg)
    except Exception as e:
        logger.warning(f"Failed to check equity staleness: {e}")

    # Isolate reconciliation failures
    # reconciliation_succeeded = False
    try:
        await reconcile_and_notify(config)
        # reconciliation_succeeded = True
    except Exception as e:
        logger.critical(f"Reconciliation FAILED: {e}", exc_info=True)
        send_pushover_notification(
            config.get('notifications', {}),
            "🚨 Reconciliation Failed",
            f"Council history reconciliation failed: {str(e)[:200]}\n"
            f"Brier scoring will be stale until this is fixed."
        )

    await analyze_and_archive(config)

    # NOTE: Feedback loop health check moved to run_brier_reconciliation (close+20min)
    # so it reports accurate counts AFTER CSV predictions are resolved.

    logger.info("--- End-of-day reconciliation and analysis process complete ---")


async def _check_feedback_loop_health(config: dict):
    """
    Check for stale PENDING predictions and alert if feedback loop is broken.

    This monitoring function detects when the prediction → reconciliation →
    Brier scoring pipeline has stalled. It would have caught the Jan 19-31
    failure within 48 hours instead of 12 days.

    DESIGN PRINCIPLES:
    - Fail-Safe: This function NEVER crashes the orchestrator. All errors are
      caught, logged, and the function exits gracefully.
    - Observable: Every branch logs its outcome for debugging.
    - Non-Blocking: Monitoring failures don't block trading operations.
    """
    logger.info("Running feedback loop health check...")

    try:
        structured_file = os.path.join(config.get('data_dir', 'data'), "agent_accuracy_structured.csv")
        if not os.path.exists(structured_file):
            logger.info("Feedback Loop Health: No structured predictions file yet (expected for new deployments)")
            return

        # Use pandas for analysis
        try:
            import pandas as pd
        except ImportError:
            logger.warning("Feedback Loop Health: pandas not available, skipping check")
            return

        df = pd.read_csv(structured_file)
        if df.empty:
            logger.info("Feedback Loop Health: Predictions file exists but is empty")
            return

        # === CORE METRICS ===
        pending_mask = df['actual'] == 'PENDING'
        pending_count = pending_mask.sum()
        total_count = len(df)

        if pending_count == 0:
            logger.info(f"Feedback Loop Health: All {total_count} predictions resolved ✓")
        else:
            # === PENDING PREDICTIONS EXIST - ANALYZE STALENESS ===

            # Defensive timestamp parsing: coerce unparseable values to NaT instead of crashing
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

            # Count and log corrupted rows
            corrupted_count = df['timestamp'].isna().sum()
            if corrupted_count > 0:
                logger.warning(
                    f"Feedback loop health: {corrupted_count} rows with unparseable timestamps "
                    f"(data corruption detected). Filtering them out."
                )

            # Filter out corrupted rows for analysis
            df_clean = df[df['timestamp'].notna()].copy()

            if df_clean.empty:
                logger.error("Feedback loop health: All rows had corrupted timestamps!")
                return

            # Recalculate pending mask on clean data
            pending_mask_clean = df_clean['actual'] == 'PENDING'

            # Calculate age of oldest PENDING prediction
            pending_timestamps = df_clean.loc[pending_mask_clean, 'timestamp']
            if not pending_timestamps.empty:
                oldest_pending = pending_timestamps.min()
                age_hours = (pd.Timestamp.now(tz='UTC') - oldest_pending).total_seconds() / 3600
            else:
                age_hours = 0

            # Calculate resolution rate (excluding orphans)
            orphaned_count = (df_clean['actual'] == 'ORPHANED').sum() if 'actual' in df_clean.columns else 0
            resolvable_count = total_count - orphaned_count
            resolution_rate = (
                (resolvable_count - pending_count) / resolvable_count * 100
                if resolvable_count > 0 else 0
            )

            logger.info(
                f"Feedback Loop Health: {pending_count}/{resolvable_count} PENDING "
                f"({orphaned_count} orphans excluded, "
                f"resolution rate: {resolution_rate:.0f}%)"
            )

            # === ALERT IF PREDICTIONS ARE STALE ===
            # Distinguish young PENDING (<24h, normal) from stale PENDING (>48h, needs attention)
            stale_threshold_hours = 48
            young_threshold_hours = 24
            stale_count = 0
            young_count = 0
            if not pending_timestamps.empty:
                now_utc = pd.Timestamp.now(tz='UTC')
                for ts in pending_timestamps:
                    hours_old = (now_utc - ts).total_seconds() / 3600
                    if hours_old > stale_threshold_hours:
                        stale_count += 1
                    elif hours_old <= young_threshold_hours:
                        young_count += 1

            if stale_count > 0:
                alert_msg = (
                    f"⚠️ FEEDBACK LOOP: {stale_count} stale predictions\n"
                    f"Oldest PENDING: {age_hours:.0f}h ago\n"
                    f"Stale (>48h): {stale_count} | Recent (<24h): {young_count}\n"
                    f"Resolution rate: {resolution_rate:.0f}%\n"
                    f"PENDING: {pending_count}/{total_count}"
                )
                logger.warning(alert_msg)
                send_pushover_notification(
                    config.get('notifications', {}),
                    "🟡 Feedback Loop Alert",
                    alert_msg
                )

        # === ENHANCED BRIER SYSTEM HEALTH ===
        try:
            from trading_bot.brier_bridge import get_calibration_data
            cal_data = get_calibration_data()

            if not cal_data:
                logger.info("Enhanced Brier: No calibration data yet (expected if newly deployed)")
            else:
                total_resolved = sum(
                    d.get('total_predictions', 0)
                    for d in cal_data.values()
                )
                logger.info(
                    f"Enhanced Brier Health: {len(cal_data)} agents tracked, "
                    f"{total_resolved} total resolved predictions"
                )

                # Alert if system has been running >7 days but no resolutions
                # Use age_hours from above if available
                if 'age_hours' in locals() and age_hours > 168 and total_resolved == 0:  # 7 days
                    alert_msg = (
                        "⚠️ ENHANCED BRIER STALLED\n"
                        "7+ days running but 0 predictions resolved.\n"
                        "Check reconciliation pipeline."
                    )
                    logger.warning(alert_msg)
                    send_pushover_notification(
                        config.get('notifications', {}),
                        "🟡 Brier System Alert",
                        alert_msg
                    )

        except ImportError:
            logger.debug("Enhanced Brier bridge not available - skipping advanced health check")
        except Exception as brier_err:
            logger.warning(f"Enhanced Brier health check failed (non-fatal): {brier_err}")

    except Exception as e:
        # === FAIL-SAFE: Log error but NEVER crash the orchestrator ===
        logger.error(
            f"Feedback loop health check failed (non-fatal): {e}. "
            f"The orchestrator will continue operating.",
            exc_info=True
        )
        # Optionally notify about the monitoring failure itself
        try:
            send_pushover_notification(
                config.get('notifications', {}),
                "⚠️ Health Check Error",
                f"Feedback loop health check failed: {str(e)[:100]}\n"
                f"Trading continues but monitoring is impaired."
            )
        except Exception:
            pass  # Don't let notification failure cause issues

    logger.info("Feedback loop health check complete.")


async def process_deferred_triggers(config: dict):
    """Process deferred triggers from overnight with deduplication."""
    if not is_market_open(config):
        logger.info("Market is closed. Skipping deferred trigger processing.")
        return

    logger.info("--- Processing Deferred Triggers ---")
    ib_conn = None
    try:
        deferred = StateManager.get_deferred_triggers()
        if not deferred:
            logger.info("No deferred triggers to process.")
            return

        logger.info(f"Processing {len(deferred)} deferred triggers from overnight")
        ib_conn = await IBConnectionPool.get_connection("deferred", config)

        # Load global excludes for payload validation
        pm_config = config.get('sentinels', {}).get('prediction_markets', {})
        global_excludes = pm_config.get('global_exclude_keywords', [])

        processed_count = 0
        skipped_count = 0
        rejected_count = 0

        for t in deferred:
            trigger = SentinelTrigger(t['source'], t['reason'], t['payload'])

            # === NEW: Validate PM trigger payloads against global excludes ===
            if trigger.source == "PredictionMarketSentinel" and global_excludes:
                payload_text = json.dumps(trigger.payload).lower()
                reason_lower = trigger.reason.lower()
                combined_text = f"{payload_text} {reason_lower}"

                is_contaminated = False
                for kw in global_excludes:
                    if word_boundary_match(kw, combined_text):
                        is_contaminated = True
                        break

                if is_contaminated:
                    logger.warning(
                        f"Rejecting contaminated deferred PM trigger: "
                        f"{trigger.reason[:80]} (matched global exclude '{kw}')"
                    )
                    rejected_count += 1
                    continue

            # === CRITICAL FIX: Check deduplicator before each cycle ===
            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                await run_emergency_cycle(trigger, config, ib_conn)
                processed_count += 1
                # Note: run_emergency_cycle sets POST_CYCLE debounce internally
            else:
                logger.info(f"Skipping deferred trigger (deduplicated): {trigger.source}")
                skipped_count += 1

        logger.info(
            f"Deferred triggers complete: {processed_count} processed, "
            f"{skipped_count} skipped, {rejected_count} rejected (contaminated)"
        )

    except Exception as e:
        logger.error(f"Failed to process deferred triggers: {e}")
    finally:
        if ib_conn is not None:
            try:
                await IBConnectionPool.release_connection("deferred")
            except Exception:
                pass


# --- SENTINEL LOGIC ---

def load_regime_context(config: dict = None) -> str:
    """
    Load current fundamental regime from FundamentalRegimeSentinel.

    Returns formatted string for prompt injection.
    """
    from pathlib import Path
    import json

    data_dir = config.get('data_dir', 'data') if config else 'data'
    regime_file = Path(os.path.join(data_dir, "fundamental_regime.json"))
    if regime_file.exists():
        try:
            with open(regime_file, 'r') as f:
                regime = json.load(f)
        except (json.JSONDecodeError, IOError, AttributeError) as e:
            logger.warning(f"Failed to load regime context: {e}")
            return ""

        regime_type = regime.get('regime', 'UNKNOWN')
        confidence = regime.get('confidence', 0.0)

        if regime_type == "DEFICIT":
            context = f"""
**CRITICAL CONTEXT: DEFICIT REGIME (Confidence: {confidence:.1%})**
The market is currently in a supply deficit. Any weather disruption, logistics bottleneck,
or demand spike will have AMPLIFIED price impact because there are no buffer stocks to absorb shocks.
Interpret bullish signals as higher conviction; bearish signals as potential mean reversion.
"""
        elif regime_type == "SURPLUS":
            context = f"""
**CRITICAL CONTEXT: SURPLUS REGIME (Confidence: {confidence:.1%})**
The market is currently in a supply surplus. Weather disruptions or logistics issues will have
MUTED price impact because ample global inventory can absorb shocks.
Interpret bearish signals as higher conviction; bullish signals as temporary.
"""
        else:
            context = f"""
**MARKET REGIME: BALANCED (Confidence: {confidence:.1%})**
Supply and demand are roughly in equilibrium. Price moves will be driven by marginal changes.
"""

        return context
    else:
        return ""

async def _is_signal_priced_in(trigger: SentinelTrigger, ib: IB, contract) -> tuple[bool, str]:
    """
    Check if the signal has already been priced into the market.

    Directional logic:
    - WeatherSentinel (typically bullish supply shock) → only priced-in if price already UP
    - PriceSentinel → skip check entirely (DEFCON-1 handles extremes)
    - NewsSentinel → only block on extreme moves (>5%) regardless of direction
    - MicrostructureSentinel → skip (structural, not directional)
    - Others → skip (let council decide)
    """
    PRICED_IN_THRESHOLD = 3.0
    EXTREME_MOVE_THRESHOLD = 5.0

    try:
        bars = await asyncio.wait_for(ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        ), timeout=15)
        if len(bars) < 2:
            return False, ""

        prev_close = bars[-2].close
        current_close = bars[-1].close
        change_pct = ((current_close - prev_close) / prev_close) * 100

        if trigger.source == 'WeatherSentinel':
            # Weather events are typically bullish (supply disruption)
            # Only "priced in" if price already surged UP
            if change_pct > PRICED_IN_THRESHOLD:
                return True, f"Price already +{change_pct:.1f}% — bullish weather shock likely priced in"
            return False, ""

        elif trigger.source == 'PriceSentinel':
            # Price sentinel IS the price move — skip priced-in check entirely
            # DEFCON-1 (>5% flash crash) is handled separately upstream
            return False, ""

        elif trigger.source == 'NewsSentinel':
            # News can be bullish or bearish — only block on extreme moves
            if abs(change_pct) > EXTREME_MOVE_THRESHOLD:
                return True, f"Price moved {change_pct:+.1f}% — extreme volatility, news likely priced in"
            return False, ""

        elif trigger.source == 'MicrostructureSentinel':
            # Structural signals — not directional, let council decide
            return False, ""

        else:
            # LogisticsSentinel, XSentimentSentinel, PredictionMarketSentinel, MacroContagionSentinel
            # Let the council evaluate these — too context-dependent for a simple gate
            return False, ""

    except Exception as e:
        logger.error(f"Priced-in check failed: {e}")
        return False, ""  # Fail open — council still evaluates the signal


async def run_emergency_cycle(trigger: SentinelTrigger, config: dict, ib: IB):
    """
    Runs a specialized cycle triggered by a Sentinel.
    Executes trades if the Council approves.
    """
    # === SHUTDOWN GATE ===
    if is_system_shutdown():
        logger.info(
            f"Emergency cycle BLOCKED (system shutdown): {trigger.source} — {trigger.reason[:100]}"
        )
        _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (system shutdown)")
        return

    # === NEW: MARKET HOURS GATE ===
    if not is_market_open(config):
        logger.info(f"Market closed. Queuing {trigger.source} alert for next session.")
        _sentinel_diag.info(f"OUTCOME {trigger.source}: DEFERRED (market closed)")
        StateManager.queue_deferred_trigger(trigger)
        return

    # === DAILY TRADING CUTOFF GATE ===
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

    cutoff_hour, cutoff_minute = get_trading_cutoff(config)

    if now_ny.hour > cutoff_hour or (now_ny.hour == cutoff_hour and now_ny.minute >= cutoff_minute):
        day_name = now_ny.strftime("%A")
        next_session = "Monday" if now_ny.weekday() == 4 else "tomorrow"
        logger.info(
            f"Emergency cycle BLOCKED (daily trading cutoff {cutoff_hour}:{cutoff_minute:02d} ET): "
            f"{trigger.source} — deferring to {next_session}"
        )
        _sentinel_diag.info(f"OUTCOME {trigger.source}: DEFERRED (daily cutoff {cutoff_hour}:{cutoff_minute:02d} ET)")
        StateManager.queue_deferred_trigger(trigger)

        # Notify operator for potential manual intervention
        # Include payload summary if available (e.g., Fed policy shock detail)
        reason_detail = trigger.reason[:200]
        if isinstance(getattr(trigger, 'payload', None), dict):
            _summary = trigger.payload.get('summary', '')
            if _summary:
                reason_detail = f"{reason_detail} - {_summary[:200]}"
        send_pushover_notification(
            config.get('notifications', {}),
            f"⏸️ Post-Cutoff: {trigger.source}",
            (
                f"{reason_detail}\n"
                f"Deferred to {next_session} ({day_name} {cutoff_hour}:{cutoff_minute:02d} ET cutoff).\n"
                f"Severity: {getattr(trigger, 'severity', 'N/A')}/10\n"
                f"Manual intervention available via dashboard."
            )
        )
        return

    # === HOLDING-TIME GATE (early exit — avoids burning API calls) ===
    # On weekly-close days (Friday / pre-holiday Thursday), skip the entire
    # council pipeline if there isn't enough time to hold a new position.
    remaining_hours = hours_until_weekly_close(config)
    if remaining_hours < float('inf'):
        min_holding = config.get('risk_management', {}).get('friday_min_holding_hours', 2.0)
    else:
        min_holding = config.get('risk_management', {}).get('min_holding_hours', 6.0)

    if remaining_hours < min_holding:
        logger.info(
            f"Emergency cycle BLOCKED (holding-time gate): Only {remaining_hours:.1f}h until weekly close "
            f"(minimum: {min_holding}h). Skipping council pipeline for {trigger.source}."
        )
        _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (holding-time gate: {remaining_hours:.1f}h < {min_holding}h)")
        StateManager.queue_deferred_trigger(trigger)
        send_pushover_notification(
            config.get('notifications', {}),
            f"📅 Deferred: {trigger.source}",
            f"Weekly close in {remaining_hours:.1f}h — below {min_holding}h minimum.\n"
            f"Trigger deferred (no API calls spent).\n"
            f"Reason: {trigger.reason[:200]}"
        )
        return

    # === NEW: Log Trigger for Fallback ===
    StateManager.log_sentinel_event(trigger)

    # Acquire Lock to prevent race conditions
    if EMERGENCY_LOCK.locked():
        logger.warning(f"Emergency cycle for {trigger.source} queued (Lock active).")

    try:
        await asyncio.wait_for(EMERGENCY_LOCK.acquire(), timeout=300)
    except asyncio.TimeoutError:
        severity = getattr(trigger, 'severity', 'N/A')
        logger.error(
            f"EMERGENCY_LOCK acquisition timed out (300s) for {trigger.source} "
            f"(severity={severity}). Deferring trigger instead of dropping."
        )
        _sentinel_diag.error(f"OUTCOME {trigger.source}: DEFERRED (lock timeout, severity={severity})")
        StateManager.queue_deferred_trigger(trigger)
        send_pushover_notification(
            config.get('notifications', {}),
            f"Lock Timeout: {trigger.source}",
            f"Emergency cycle blocked for 300s — trigger deferred.\n"
            f"Severity: {severity}/10\n"
            f"Reason: {trigger.reason[:200]}\n"
            f"Will retry at next deferred processing window.",
            priority=1 if severity != 'N/A' and int(severity) >= 8 else 0,
        )
        return
    try:
        cycle_actually_ran = False  # Did we reach the council decision?
        is_actionable = False       # Did the council produce a tradeable signal?
        try:
            # --- WEEKLY CLOSE WINDOW GUARD ---
            weekday = now_ny.weekday()

            # Align Friday cutoff with daily cutoff
            WEEKLY_CLOSE_CUTOFF_HOUR = cutoff_hour
            WEEKLY_CLOSE_CUTOFF_MINUTE = cutoff_minute

            is_in_close_window = False
            if weekday == 4:  # Friday
                close_cutoff = now_ny.replace(
                    hour=WEEKLY_CLOSE_CUTOFF_HOUR,
                    minute=WEEKLY_CLOSE_CUTOFF_MINUTE,
                    second=0
                )
                is_in_close_window = now_ny >= close_cutoff

            if is_in_close_window:
                logger.warning(
                    f"Emergency cycle blocked: Inside Friday close window "
                    f"({now_ny.strftime('%H:%M')} ET >= {WEEKLY_CLOSE_CUTOFF_HOUR}:{WEEKLY_CLOSE_CUTOFF_MINUTE:02d} ET). "
                    f"Trigger: {trigger.source} — {trigger.reason}"
                )
                # Still log the trigger for Monday analysis
                send_pushover_notification(
                    config.get('notifications', {}),
                    f"⏸️ Deferred: {trigger.source}",
                    f"Trigger deferred to Monday (Friday close window):\n{trigger.reason}"
                )
                _sentinel_diag.info(f"OUTCOME {trigger.source}: DEFERRED (Friday close window)")
                StateManager.queue_deferred_trigger(trigger)
                return

            # Check Budget
            if GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit:
                logger.warning("Budget hit — skipping Emergency Cycle (Sentinel-only mode)")
                _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (budget hit)")
                send_pushover_notification(config.get('notifications', {}), "Budget Guard",
                    "Daily API budget hit. Sentinel-only mode active.")
                return

            # === Drawdown Circuit Breaker ===
            if GLOBAL_DRAWDOWN_GUARD:
                # Update P&L and Check
                if not ib.isConnected():
                    logger.warning("IB disconnected — skipping drawdown P&L update (guard state preserved)")
                    status = "IB_DISCONNECTED"
                else:
                    status = await GLOBAL_DRAWDOWN_GUARD.update_pnl(ib)
                if not GLOBAL_DRAWDOWN_GUARD.is_entry_allowed():
                    logger.warning(f"Drawdown Circuit Breaker ACTIVE ({status}) - Skipping Emergency Cycle")
                    _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (drawdown breaker: {status})")
                    return

                if GLOBAL_DRAWDOWN_GUARD.should_panic_close():
                     logger.critical("Drawdown PANIC triggered during emergency cycle check. Triggering Hard Close.")
                     _sentinel_diag.critical(f"OUTCOME {trigger.source}: PANIC CLOSE (drawdown)")
                     await emergency_hard_close(config)
                     return

            # === Generate Cycle ID for prediction tracking ===
            active_ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
            cycle_id = generate_cycle_id(active_ticker)
            logger.info(f"🚨 EMERGENCY CYCLE TRIGGERED by {trigger.source}: {trigger.reason} (Cycle: {cycle_id})")
            _trigger_detail = trigger.reason
            if isinstance(getattr(trigger, 'payload', None), dict):
                _ps = trigger.payload.get('summary', '')
                if _ps:
                    _trigger_detail = f"{trigger.reason} - {_ps[:200]}"
            send_pushover_notification(config.get('notifications', {}), f"Sentinel Trigger: {trigger.source}", _trigger_detail)

            # --- DEFCON 1: Crash Protection ---
            # If price drops > 5% instantly, do NOT open new trades. Liquidation logic is complex, so we just Halt.
            if trigger.source == "PriceSentinel" and abs(trigger.payload.get('change', 0)) > 5.0:
                logger.critical("📉 FLASH CRASH DETECTED (>5%). Skipping Council. HALTING TRADING.")
                send_pushover_notification(config.get('notifications', {}), "FLASH CRASH ALERT", "Price moved >5%. Emergency Halt Triggered. No new orders.")
                # Future: await close_stale_positions(config, force=True)
                return

            # === NEW: DEFENSIVE CHECK (Defense Before Offense) ===
            # If a Sentinel fires, check existing positions FIRST
            if trigger.severity >= 6:
                # Map sentinel sources to guardian agents
                guardian_map = {
                    'WeatherSentinel': 'Agronomist',
                    'LogisticsSentinel': 'Logistics',
                    'NewsSentinel': 'Fundamentalist',
                    'PriceSentinel': 'VolatilityAnalyst',
                    'XSentimentSentinel': 'Sentiment',
                    'PredictionMarketSentinel': 'Macro'
                }

                guardian = guardian_map.get(trigger.source)
                if guardian:
                    tms = TransactiveMemory()
                    affected_theses = tms.get_active_theses_by_guardian(guardian)

                    if affected_theses:
                        # We have potential victims. We need to map them to live positions to close them.
                        try:
                            live_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=30)
                        except asyncio.TimeoutError:
                            logger.error("reqPositionsAsync timed out (30s) in defense check, skipping")
                            live_positions = None
                    if affected_theses and live_positions is not None:
                        trade_ledger = get_trade_ledger_df()

                        for thesis in affected_theses:
                          try:
                            # Check if this trigger invalidates the thesis
                            invalidation_triggers = thesis.get('invalidation_triggers', [])
                            trigger_keywords = trigger.reason.lower()

                            thesis_killed = False
                            for inv_trigger in invalidation_triggers:
                                if inv_trigger.lower() in trigger_keywords:
                                    thesis_killed = True
                                    logger.warning(f"SENTINEL INVALIDATION: {trigger.source} fired, killing thesis {thesis.get('trade_id')}")
                                    break

                            if thesis_killed:
                                thesis_id = thesis.get('trade_id')

                                # Collect ALL legs for this thesis (spreads have multiple legs)
                                thesis_legs = []
                                for pos in live_positions:
                                    if pos.position == 0:
                                        continue
                                    pos_id = _find_position_id_for_contract(pos, trade_ledger)
                                    if pos_id == thesis_id:
                                        thesis_legs.append(pos)

                                if thesis_legs:
                                    fully_closed = await _close_spread_position(
                                        ib=ib,
                                        legs=thesis_legs,
                                        position_id=thesis_id,
                                        reason=f"Sentinel Invalidation: {trigger.reason}",
                                        config=config,
                                        thesis=thesis
                                    )
                                    if fully_closed:
                                        tms.invalidate_thesis(thesis_id, f"Sentinel: {trigger.source}")
                                    else:
                                        logger.error(
                                            f"Thesis {thesis_id} NOT invalidated — "
                                            f"sentinel close did not fully succeed. "
                                            f"Will retry on next audit cycle."
                                        )
                                else:
                                    logger.warning(
                                        f"No live IB legs found for thesis {thesis_id}. "
                                        f"Position may have closed externally — invalidating thesis."
                                    )
                                    tms.invalidate_thesis(thesis_id, f"Sentinel: {trigger.source} (no IB position)")
                          except Exception as thesis_err:
                            logger.error(
                                f"Failed to process sentinel invalidation for thesis "
                                f"{thesis.get('trade_id', 'UNKNOWN')}: {thesis_err}"
                            )

            try:
                # 1. Initialize Council
                council = TradingCouncil(config)

                # 2. Get Active Futures (We need a target contract)
                # For simplicity, target the Front Month or the one from the trigger payload
                contract_name_hint = trigger.payload.get('contract')

                try:
                    active_futures = await asyncio.wait_for(
                        get_active_futures(ib, config['symbol'], config['exchange'], count=2), timeout=30
                    )
                except asyncio.TimeoutError:
                    logger.error("get_active_futures timed out (30s) in emergency cycle")
                    return
                if not active_futures:
                    logger.error("No active futures found for emergency cycle.")
                    return

                # Select contract
                target_contract = active_futures[0]
                if contract_name_hint:
                    # Try to match hint
                    for f in active_futures:
                        if f.localSymbol == contract_name_hint:
                            target_contract = f
                            break

                contract_name = f"{target_contract.localSymbol} ({target_contract.lastTradeDateOrContractMonth[:6]})"
                logger.info(f"Targeting contract: {contract_name}")

                # === NEW: PRICED IN CHECK ===
                priced_in, reason = await _is_signal_priced_in(trigger, ib, target_contract)
                if priced_in:
                    logger.warning(f"PRICED IN CHECK FAILED: {reason}. Skipping emergency cycle.")
                    _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (priced in: {reason})")
                    send_pushover_notification(config.get('notifications', {}), "Signal Priced In", reason)
                    return

                # 3. Get Market Context (Snapshot)
                ticker = ib.reqMktData(target_contract, '', True, False)
                await asyncio.sleep(2)

                # Fetch IV Metrics
                try:
                    iv_metrics = await asyncio.wait_for(
                        get_underlying_iv_metrics(ib, target_contract), timeout=30
                    )
                except asyncio.TimeoutError:
                    logger.warning("get_underlying_iv_metrics timed out (30s), using empty metrics")
                    iv_metrics = {'current_iv': 'N/A', 'iv_rank': 'N/A', 'iv_percentile': 'N/A'}

                market_context_str = (
                    f"Contract: {target_contract.localSymbol}\\n"
                    f"Current Price: {ticker.last if ticker.last else 'N/A'}\\n"
                    f"--- VOLATILITY METRICS (IBKR Live) ---\\n"
                    f"Current IV: {iv_metrics['current_iv']}\\n"
                    f"IV Rank: {iv_metrics['iv_rank']}\\n"
                    f"IV Percentile: {iv_metrics['iv_percentile']}\\n"
                    f"Note: If IV data shows N/A, analyst should search Barchart for KC IV Rank.\\n"
                )

                from trading_bot.market_data_provider import build_market_context
                try:
                    market_data = await asyncio.wait_for(
                        build_market_context(ib, target_contract, config), timeout=30
                    )
                except asyncio.TimeoutError:
                    logger.error("build_market_context timed out (30s) in emergency cycle")
                    return
                market_data['reason'] = f"Emergency Cycle triggered by {trigger.source}"
                logger.info(f"Emergency market context: price={market_data.get('price')}, regime={market_data.get('regime')}")

                # E.1: Inject VaR briefing into emergency context
                try:
                    from trading_bot.var_calculator import get_var_calculator
                    cached_var = get_var_calculator(config).get_cached_var()
                    var_limit = config.get('compliance', {}).get('var_limit_pct', 0.03)
                    var_warning = config.get('compliance', {}).get('var_warning_pct', 0.02)
                    if cached_var:
                        util = cached_var.var_95_pct / var_limit if var_limit else 0
                        market_context_str += (
                            f"\n--- PORTFOLIO STATE ---\n"
                            f"Positions: {cached_var.position_count} across "
                            f"{', '.join(cached_var.commodities)}\n"
                            f"VaR utilization: {util:.0%} of limit\n"
                            f"--- END PORTFOLIO STATE ---\n"
                        )
                        if cached_var.var_95_pct > var_warning:
                            market_context_str += (
                                f"\n--- PORTFOLIO RISK ALERT ---\n"
                                f"VaR: {cached_var.var_95_pct:.1%} (limit: {var_limit:.0%})\n"
                                f"INSTRUCTION: PREFER strategies that REDUCE correlation "
                                f"with existing positions.\n"
                                f"--- END RISK ALERT ---\n"
                            )
                except Exception:
                    pass  # Non-fatal

                # Load Regime Context
                regime_context = load_regime_context(config)

                # 5. Semantic Cache — sentinel fire invalidates other sources' cached decisions
                semantic_cache = get_semantic_cache(config)
                semantic_cache.invalidate_cross_source(contract_name, trigger.source)

                severity_threshold = config.get('semantic_cache', {}).get('severity_bypass_threshold', 8)
                cache_bypass = trigger.severity >= severity_threshold

                cached_decision = None
                if not cache_bypass:
                    cached_decision = semantic_cache.get(contract_name, trigger.source, market_data)

                if cached_decision:
                    decision = cached_decision
                    logger.info(f"SEMANTIC CACHE HIT: Reusing decision for {contract_name}/{trigger.source}")
                else:
                    decision = await council.run_specialized_cycle(
                        trigger,
                        contract_name,
                        market_data,
                        market_context_str,
                        ib=ib,
                        target_contract=target_contract,
                        cycle_id=cycle_id, # Pass cycle_id for logging if supported
                        regime_context=regime_context
                    )
                    semantic_cache.put(contract_name, trigger.source, market_data, decision)

                logger.info(f"Emergency Decision: {decision.get('direction')} ({decision.get('confidence')})")
                cycle_actually_ran = True

                # Amendment A: Inject polymarket context into the decision for thesis recording
                if trigger.source == "PredictionMarketSentinel":
                    decision['polymarket_slug'] = trigger.payload.get('slug', '')
                    decision['polymarket_title'] = trigger.payload.get('title', '')

                # === v7.1: Strategy Routing (aligns emergency with v7.0 Judge & Jury) ===
                # Previously: hardcoded DIRECTIONAL. Now: respects thesis, vol, regime.
                current_reports = StateManager.load_state()
                routed = _route_emergency_strategy(
                    decision=decision,
                    market_context=market_data,  # Contains regime, price, etc.
                    agent_reports=current_reports,
                    config=config
                )

                # === SHADOW RUN (Strategy Router) ===
                try:
                    routed_shadow = route_strategy(
                        direction=decision.get('direction', 'NEUTRAL'),
                        confidence=decision.get('confidence', 0.0),
                        vol_sentiment=decision.get('volatility_sentiment', 'BEARISH'),
                        regime=market_data.get('regime', 'UNKNOWN'),
                        thesis_strength=decision.get('thesis_strength', 'SPECULATIVE'),
                        conviction_multiplier=decision.get('conviction_multiplier', 1.0),
                        reasoning=decision.get('reasoning', ''),
                        agent_data=current_reports,
                        mode="emergency",
                    )

                    if routed['prediction_type'] != routed_shadow['prediction_type'] or \
                       routed['vol_level'] != routed_shadow['vol_level']:
                        logger.critical(
                            f"ROUTING MISMATCH (Emergency Shadow): "
                            f"Legacy=[{routed['prediction_type']}, {routed['vol_level']}], "
                            f"Router=[{routed_shadow['prediction_type']}, {routed_shadow['vol_level']}]"
                        )
                    else:
                        logger.info("Emergency Router Shadow Run: MATCH ✅")
                except Exception as e:
                    logger.error(f"Emergency Router Shadow Run FAILED: {e}")

                # === Log Emergency Decision to History ===
                # Reconstruct full log entry for history
                try:
                    # Reload reports to get sentiments
                    final_reports = StateManager.load_state()
                    agent_data = {}

                    def extract_sentiment(text):
                        if not text or not isinstance(text, str):
                            return "N/A"
                        import re
                        match = re.search(r'\[SENTIMENT: (\w+)\]', text)
                        return match.group(1) if match else "N/A"

                    for key, report in final_reports.items():
                        if isinstance(report, dict):
                            s = report.get('sentiment')
                            if not s or s == 'N/A':
                                s = extract_sentiment(report.get('data', ''))
                            agent_data[f"{key}_sentiment"] = s
                            agent_data[f"{key}_summary"] = str(report.get('data', 'N/A'))
                        else:
                            agent_data[f"{key}_sentiment"] = extract_sentiment(report)
                            agent_data[f"{key}_summary"] = str(report) if report else "N/A"

                    council_log_entry = {
                        "cycle_id": cycle_id,
                        "timestamp": datetime.now(timezone.utc),
                        "contract": contract_name,
                        "entry_price": market_data.get('price'),

                        "meteorologist_sentiment": agent_data.get("agronomist_sentiment"),
                        "meteorologist_summary": agent_data.get("agronomist_summary"),
                        "macro_sentiment": agent_data.get('macro_sentiment'),
                        "macro_summary": agent_data.get('macro_summary'),
                        "geopolitical_sentiment": agent_data.get('geopolitical_sentiment'),
                        "geopolitical_summary": agent_data.get('geopolitical_summary'),
                        "fundamentalist_sentiment": agent_data.get('inventory_sentiment'),
                        "fundamentalist_summary": agent_data.get('inventory_summary'),
                        "sentiment_sentiment": agent_data.get('sentiment_sentiment'),
                        "sentiment_summary": agent_data.get('sentiment_summary'),
                        "technical_sentiment": agent_data.get('technical_sentiment'),
                        "technical_summary": agent_data.get('technical_summary'),
                        "volatility_sentiment": agent_data.get('volatility_sentiment'),
                        "volatility_summary": agent_data.get('volatility_summary'),

                        "master_decision": decision.get('direction'),
                        "master_confidence": decision.get('confidence'),
                        "master_reasoning": decision.get('reasoning'),

                        # v7.1: Use routed prediction type, not raw decision
                        "prediction_type": routed['prediction_type'],
                        "volatility_level": routed.get('vol_level'),
                        "strategy_type": _infer_strategy_type(routed),

                        # v7.0 forensic fields
                        "thesis_strength": routed.get('thesis_strength', 'SPECULATIVE'),
                        "primary_catalyst": decision.get('primary_catalyst', 'N/A'),
                        "conviction_multiplier": routed.get('conviction_multiplier', 1.0),
                        "dissent_acknowledged": decision.get('dissent_acknowledged', 'N/A'),

                        "compliance_approved": True, # Assume true if we reached here, actually checked later
                        "trigger_type": trigger.source,

                        "vote_breakdown": json.dumps(decision.get('vote_breakdown', [])),
                        "dominant_agent": decision.get('dominant_agent', 'Unknown'),
                        "weighted_score": 0.0 # Not explicitly returned by run_specialized_cycle but embedded in vote
                    }
                    log_council_decision(council_log_entry)

                    # Decision Signal (Lightweight)
                    log_decision_signal(
                        cycle_id=cycle_id,
                        contract=contract_name,
                        signal=council_log_entry.get('master_decision', 'NEUTRAL'),
                        prediction_type=council_log_entry.get('prediction_type', 'DIRECTIONAL'),
                        strategy=council_log_entry.get('strategy_type', 'NONE'),
                        price=council_log_entry.get('entry_price'),
                        sma_200=market_data.get('sma_200'),
                        confidence=council_log_entry.get('master_confidence'),
                        regime=market_data.get('regime', 'UNKNOWN'),
                        trigger_type='EMERGENCY',
                    )
                except Exception as e:
                    logger.error(f"Failed to log emergency decision: {e}")

                # 6. Execute if Actionable
                direction = routed['direction']
                pred_type = routed['prediction_type']
                confidence = routed['confidence']
                threshold = config.get('strategy', {}).get('signal_threshold', 0.5)

                is_actionable = (
                    (direction in ['BULLISH', 'BEARISH'] and confidence > threshold) or
                    (direction == 'VOLATILITY' and pred_type == 'VOLATILITY' and confidence > threshold)
                )

                # Record sentinel stats
                SENTINEL_STATS.record_alert(
                    sentinel_name=trigger.source,
                    triggered_trade=is_actionable
                )

                if is_actionable:
                    logger.info(f"Emergency Cycle ACTION: {direction} ({pred_type})")

                    # === NEW: Compliance Audit ===
                    compliance = ComplianceGuardian(config)
                    # Load reports for audit (re-use final_reports from agents logic if possible, but here we only have decision)
                    # Ideally, run_specialized_cycle should return the full packet.
                    # For now, we reload state which is "close enough" as it was just updated.
                    current_reports = StateManager.load_state()

                    # v8.1: Build compliance context with IBKR market data
                    from trading_bot.market_data_provider import format_market_context_for_prompt
                    compliance_context = market_context_str
                    ibkr_data_str = format_market_context_for_prompt(market_data)
                    if ibkr_data_str:
                        compliance_context += f"\n--- IBKR MARKET DATA ---\n{ibkr_data_str}\n"
                    # Note: Semantic cache hits from pre-v8.1 won't have debate_summary (defaults to "")
                    debate_summary = decision.get('debate_summary', '')

                    audit = await compliance.audit_decision(
                        current_reports,
                        compliance_context,
                        decision,
                        council.personas.get('master', ''),
                        ib=ib,
                        debate_summary=debate_summary
                    )

                    if not audit.get('approved', True):
                        logger.warning(f"COMPLIANCE BLOCKED Emergency Trade: {audit.get('flagged_reason')}")
                        flagged_reason = audit.get('flagged_reason', 'Unknown')
                        # Truncate technical error traces for readability
                        if len(flagged_reason) > 200:
                            flagged_reason = flagged_reason[:200] + "..."

                        # Distinguish system errors from genuine compliance vetoes
                        if any(err_marker in flagged_reason for err_marker in ['Error:', 'Exception', 'object has no attribute', 'exhausted']):
                            title = "⚠️ Emergency Trade Blocked (System Error)"
                            message = (
                                f"Compliance check could not complete due to a system error.\n"
                                f"Trade was blocked as a safety precaution.\n"
                                f"Error: {flagged_reason[:150]}"
                            )
                        else:
                            title = "🛡️ Emergency Trade VETOED by Compliance"
                            message = f"Reason: {flagged_reason}"

                        send_pushover_notification(
                            config.get('notifications', {}),
                            title,
                            message
                        )
                        _sentinel_diag.info(f"OUTCOME {trigger.source}: BLOCKED (compliance: {flagged_reason[:100]})")
                        return

                    logger.info("Decision is actionable. Generating order...")

                    # === NEW: Dynamic Position Sizing ===
                    sizer = DynamicPositionSizer(config)

                    # Get account value
                    account_summary = await asyncio.wait_for(ib.accountSummaryAsync(), timeout=15)
                    net_liq_tag = next((v for v in account_summary if v.tag == 'NetLiquidation' and v.currency == 'USD'), None)
                    account_value = float(net_liq_tag.value) if net_liq_tag else 100000.0

                    vol_sentiment = decision.get('volatility_sentiment', 'NEUTRAL')
                    if not vol_sentiment and 'vote_breakdown' in decision:
                        # Try to extract from vote breakdown
                        pass # sizer defaults to NEUTRAL if passed string is None

                    # v7.1: Pass conviction_multiplier so divergent consensus = smaller position
                    _conviction = decision.get('conviction_multiplier', 1.0)
                    qty = await sizer.calculate_size(
                        ib, decision, vol_sentiment, account_value,
                        conviction_multiplier=_conviction
                    )

                    # Build Strategy
                    try:
                        chain = await asyncio.wait_for(
                            build_option_chain(ib, target_contract), timeout=45
                        )
                    except asyncio.TimeoutError:
                        logger.error("build_option_chain timed out (45s)")
                        return
                    if not chain:
                        logger.warning("No option chain available.")
                        return

                    signal_obj = {
                        "contract_month": target_contract.lastTradeDateOrContractMonth[:6],
                        "direction": routed['direction'],
                        "confidence": routed['confidence'],
                        "price": ticker.last,
                        "prediction_type": routed['prediction_type'],
                        "volatility_sentiment": routed['volatility_sentiment'],
                        "thesis_strength": routed['thesis_strength'],
                        "conviction_multiplier": routed['conviction_multiplier'],
                        "regime": routed['regime'],
                        "reason": routed['reason'],
                        "quantity": qty,
                    }

                    # Route to correct strategy definition function
                    if routed['prediction_type'] == 'VOLATILITY':
                        signal_obj['level'] = routed['vol_level']  # "HIGH" or "LOW"
                        strategy_def = define_volatility_strategy(config, signal_obj, chain, ticker.last, target_contract)
                    else:
                        strategy_def = define_directional_strategy(config, signal_obj, chain, ticker.last, target_contract)

                    if strategy_def:
                        strategy_def['quantity'] = qty # Force apply

                        order_objects = await create_combo_order_object(ib, config, strategy_def)
                        if order_objects:
                            contract, order = order_objects

                            # Queue and Execute (Fix Global Queue Collision)
                            # Pass specific list to place_queued_orders so we don't wipe the global queue
                            emergency_order_list = [(contract, order, decision)]
                            await place_queued_orders(config, orders_list=emergency_order_list)
                            logger.info(f"Emergency Order Placed (Qty: {qty}).")

                # === BRIER SCORE RECORDING (Dual-Write: Legacy CSV + Enhanced JSON) ===
                try:
                    from trading_bot.brier_bridge import record_agent_prediction
                    from trading_bot.agent_names import CANONICAL_AGENTS, DEPRECATED_AGENTS
                    final_reports_for_scoring = StateManager.load_state()

                    # Determine current regime from council context
                    current_regime = await _detect_market_regime(config, trigger, ib, target_contract)

                    BRIER_ELIGIBLE_AGENTS = set(CANONICAL_AGENTS) - {'master_decision'} - DEPRECATED_AGENTS

                    for agent_name, report in final_reports_for_scoring.items():
                        if agent_name not in BRIER_ELIGIBLE_AGENTS:
                            logger.debug(f"Skipping non-canonical agent '{agent_name}' for Brier recording")
                            continue

                        direction, confidence = _extract_agent_prediction(report)

                        record_agent_prediction(
                            agent=agent_name,
                            predicted_direction=direction,
                            predicted_confidence=parse_confidence(confidence),
                            cycle_id=cycle_id,
                            regime=current_regime,
                            contract=target_contract.lastTradeDateOrContractMonth[:6] if target_contract else "",
                        )

                except Exception as e:
                    logger.error(f"Brier recording failed: {e}")

                if not is_actionable:
                    logger.info(f"Emergency Cycle concluded with no action: {direction} ({pred_type})")

            except Exception as e:
                logger.exception(f"Emergency Cycle Failed: {e}")

        finally:
            # Graduated post-cycle debounce based on cycle outcome
            if not cycle_actually_ran:
                # Gate-blocked (budget, drawdown, Friday, priced-in) — don't punish next trigger
                debounce_seconds = 0
                debounce_reason = "gate-blocked (no debounce)"
            elif is_actionable:
                # Council produced a trade — full debounce to prevent overtrading
                debounce_seconds = config.get('sentinels', {}).get('post_cycle_debounce_seconds', 1800)
                debounce_reason = "actionable trade"
            else:
                # Council ran but decided NEUTRAL — short debounce then re-evaluate
                debounce_seconds = config.get('sentinels', {}).get('post_cycle_debounce_neutral_seconds', 300)
                debounce_reason = "neutral decision"

            if debounce_seconds > 0:
                GLOBAL_DEDUPLICATOR.set_cooldown("POST_CYCLE", debounce_seconds)
            else:
                GLOBAL_DEDUPLICATOR.clear_cooldown("POST_CYCLE")

            logger.info(f"Post-cycle debounce: {debounce_seconds}s ({debounce_reason})")
            if cycle_actually_ran:
                outcome = "TRADE" if is_actionable else "NEUTRAL"
                _sentinel_diag.info(f"OUTCOME {trigger.source}: {outcome} (debounce={debounce_seconds}s)")
    finally:
        EMERGENCY_LOCK.release()


def validate_trigger(trigger):
    """Defensive check for sentinel triggers."""
    if isinstance(trigger, list):
        logger.warning("Sentinel returned list instead of single trigger. Using first item.")
        trigger = trigger[0] if trigger else None

    if trigger is not None and not hasattr(trigger, 'source'):
        logger.error(f"Invalid trigger object (missing 'source'): {type(trigger)}")
        return None
    return trigger


def _log_task_exception(task: asyncio.Task, name: str):
    """Generic callback to log exceptions from fire-and-forget tasks."""
    _INFLIGHT_TASKS.discard(task)
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        logger.info(f"Task '{name}' was cancelled")
        return

    if exc is not None:
        logger.error(f"Fire-and-forget task '{name}' CRASHED: {type(exc).__name__}: {exc}")


def _emergency_cycle_done_callback(task: asyncio.Task, trigger_source: str, config: dict):
    """Callback to catch and report crashes in fire-and-forget emergency cycle tasks."""
    _INFLIGHT_TASKS.discard(task)
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        logger.info(f"Emergency cycle for {trigger_source} was cancelled")
        _sentinel_diag.info(f"OUTCOME {trigger_source}: CANCELLED")
        return

    if exc is not None:
        logger.error(
            f"🔥 Emergency cycle for {trigger_source} CRASHED: "
            f"{type(exc).__name__}: {exc}"
        )
        _sentinel_diag.error(f"OUTCOME {trigger_source}: CRASHED ({type(exc).__name__}: {exc})")
        SENTINEL_STATS.record_error(trigger_source, f"CRASH: {type(exc).__name__}")
        # Prevent crash-loop: cooldown so the sentinel doesn't immediately re-trigger
        GLOBAL_DEDUPLICATOR.set_cooldown(trigger_source, 1800)  # 30-min crash cooldown
        try:
            send_pushover_notification(
                config.get('notifications', {}),
                f"🔥 Emergency Cycle Crashed: {trigger_source}",
                f"Error: {str(exc)[:200]}\nSentinel loop continues normally."
            )
        except Exception:
            pass  # Don't let notification failure cascade


async def _run_periodic_sentinel(
    sentinel_instance,
    last_run_time: float,
    interval: int,
    timeout: int,
    config: dict,
    sentinel_ib,
    market_open: bool,
    cooldown_seconds: int = 900
) -> float:
    """
    Helper to run a periodic sentinel check.
    Returns the updated last_run_time (current timestamp if ran, else original).
    """
    import time as time_module
    now = time_module.time()

    if (now - last_run_time) <= interval:
        return last_run_time

    _health_error = None
    sentinel_name = sentinel_instance.__class__.__name__

    try:
        trigger = await asyncio.wait_for(sentinel_instance.check(), timeout=timeout)
        trigger = validate_trigger(trigger)
        if trigger:
            logger.info(f"{sentinel_name}: trigger detected (source={trigger.source}, severity={getattr(trigger, 'severity', '?')})")
            if market_open and sentinel_ib and sentinel_ib.isConnected():
                if GLOBAL_DEDUPLICATOR.should_process(trigger):
                    task = asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                    _INFLIGHT_TASKS.add(task)
                    task.add_done_callback(
                        lambda t, src=trigger.source, cfg=config: _emergency_cycle_done_callback(t, src, cfg)
                    )
                    GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, cooldown_seconds)
            else:
                # Record hash in deduplicator BEFORE deferring — prevents
                # duplicate notifications on restart (sentinel re-fires same
                # trigger, but deduplicator now recognizes it)
                if GLOBAL_DEDUPLICATOR.should_process(trigger):
                    StateManager.queue_deferred_trigger(trigger)
                    logger.info(f"Deferred {trigger.source} trigger for market open")
                else:
                    logger.info(f"Skipped duplicate deferred trigger: {trigger.source}")
    except asyncio.TimeoutError:
        logger.error(f"{sentinel_name} TIMED OUT after {timeout}s")
        _health_error = f"TIMEOUT after {timeout}s"
        SENTINEL_STATS.record_error(sentinel_name, "TIMEOUT")
    except Exception as e:
        logger.error(f"{sentinel_name} check failed: {type(e).__name__}: {e}")
        _health_error = str(e)
    finally:
        _record_sentinel_health(
            sentinel_name,
            "ERROR" if _health_error else "OK",
            interval,
            _health_error
        )
        return now


async def run_sentinels(config: dict):
    """
    Main loop for Sentinels. Runs concurrently with the scheduler.

    ARCHITECTURE (per Real Options):
    - Weather/News/Logistics/X sentinels run 24/7 (no IB needed)
    - Price/Microstructure sentinels only run during market hours (IB needed)
    - IB connection is LAZY: only established when market is actually open
    """
    logger.info("--- Starting Sentinel Array ---")

    # === 1. LAZY INITIALIZATION: Start with NO connection ===
    # Do NOT attempt connection at startup - wait for market open
    sentinel_ib = None

    # Initialize Price Sentinel with None - will inject connection later
    price_sentinel = PriceSentinel(config, None)

    # These sentinels don't need IB - initialize normally
    weather_sentinel = WeatherSentinel(config)
    logistics_sentinel = LogisticsSentinel(config)
    news_sentinel = NewsSentinel(config)
    x_sentinel = XSentimentSentinel(config)
    prediction_market_sentinel = PredictionMarketSentinel(config)
    macro_contagion_sentinel = MacroContagionSentinel(config)

    # Microstructure variables (also lazy)
    micro_sentinel = None
    micro_ib = None

    # X Sentinel Stats
    x_sentinel_stats = {
        'checks_today': 0,
        'triggers_today': 0,
        'estimated_tokens': 0,
        'estimated_cost_usd': 0.0,
        'last_reset': datetime.now().date()
    }

    # Timing state
    last_weather = 0
    last_logistics = 0
    last_news = 0
    last_x_sentiment = 0
    last_prediction_market = 0
    last_topic_discovery = _STARTUP_DISCOVERY_TIME  # 0 if startup scan failed/skipped → runs on first iteration
    last_macro_contagion = 0

    # Contract Cache
    cached_contract = None
    last_contract_refresh = 0
    CONTRACT_REFRESH_INTERVAL = 14400  # 4 hours

    # Outage Tracking (only relevant during market hours)
    last_successful_ib_time = None  # Will be set on first successful connection
    outage_notification_sent = False
    OUTAGE_THRESHOLD_SECONDS = 600  # 10 minutes

    # Record initial sentinel health state
    for name in ["WeatherSentinel", "LogisticsSentinel", "NewsSentinel",
                  "XSentimentSentinel", "PredictionMarketSentinel", "MacroContagionSentinel"]:
        _record_sentinel_health(name, "INITIALIZING", 0)

    _record_sentinel_health("PriceSentinel", "IDLE", 60)
    _record_sentinel_health("MicrostructureSentinel", "IDLE", 60)

    _sentinel_iteration = 0
    _HEARTBEAT_INTERVAL = 5  # Log heartbeat every 5 iterations (~5 min)

    while True:
        try:
            _sentinel_iteration += 1

            # v5.4: Shutdown gate — stop all sentinel activity post-shutdown
            if _SYSTEM_SHUTDOWN:
                # One-time microstructure cleanup (Issue 7 integrated)
                if micro_sentinel is not None:
                    logger.info("Shutdown: Gracefully disengaging Microstructure Sentinel")
                    try:
                        await micro_sentinel.unsubscribe_all()
                    except Exception as e:
                        logger.error(f"Shutdown microstructure cleanup error: {e}")
                    micro_sentinel = None
                    if micro_ib is not None:
                        try:
                            await IBConnectionPool.release_connection("microstructure")
                        except Exception:
                            pass
                        micro_ib = None
                await asyncio.sleep(60)
                continue

            now = time_module.time()
            market_open = is_market_open(config)
            trading_day = is_trading_day()

            # Heartbeat: confirm sentinel loop is alive
            if _sentinel_iteration % _HEARTBEAT_INTERVAL == 0:
                logger.info(
                    f"Sentinel loop heartbeat: iteration={_sentinel_iteration}, "
                    f"market_open={market_open}, trading_day={trading_day}"
                )

            # === 2. MARKET HOURS GATE: Only connect when market is OPEN ===
            should_connect = market_open  # NOT trading_day - must be is_market_open(config)

            if should_connect:
                # Market is open - we need IB for Price/Microstructure sentinels
                if sentinel_ib is None or not sentinel_ib.isConnected():
                    try:
                        logger.info("Market Open: Establishing Sentinel IB connection...")
                        sentinel_ib = await IBConnectionPool.get_connection("sentinel", config)
                        configure_market_data_type(sentinel_ib)

                        # === 3. CRITICAL: Inject connection into PriceSentinel ===
                        # Without this, price_sentinel.ib holds stale/None reference
                        price_sentinel.ib = sentinel_ib

                        logger.info("Sentinel IB connected and injected into PriceSentinel.")

                        # Reset outage tracking on success
                        last_successful_ib_time = time_module.time()
                        outage_notification_sent = False

                        # === SAFETY NET: Ensure position monitoring is running ===
                        # Covers the gap where bot starts just before market open
                        # and the scheduled start_monitoring was already missed.
                        if not _SYSTEM_SHUTDOWN and (
                            monitor_process is None or monitor_process.returncode is not None
                        ):
                            logger.info(
                                "🔄 SENTINEL SAFETY NET: Market open but position "
                                "monitor not running — starting it now"
                            )
                            try:
                                await start_monitoring(config)
                            except Exception as e:
                                logger.error(
                                    f"Safety net start_monitoring failed: {e}"
                                )

                    except Exception as e:
                        # Log as WARNING during market hours (connection should be possible)
                        logger.warning(f"Sentinel IB connection deferred: {e}")

                        # Track outage duration
                        if last_successful_ib_time is not None:
                            time_disconnected = time_module.time() - last_successful_ib_time

                            if time_disconnected > OUTAGE_THRESHOLD_SECONDS and not outage_notification_sent:
                                send_pushover_notification(
                                    config.get('notifications', {}),
                                    "🚨 IB CONNECTION CRITICAL",
                                    f"No IB connection for {time_disconnected/60:.0f} minutes during market hours. "
                                    f"Price/Microstructure sentinels OFFLINE. "
                                    f"Check Gateway status."
                                )
                                outage_notification_sent = True
                                logger.critical(f"IB outage notification sent after {time_disconnected/60:.0f} minutes")
                else:
                    # Connection is good - reset tracking
                    last_successful_ib_time = time_module.time()
                    outage_notification_sent = False

            else:
                # === 4. MARKET CLOSED: Disconnect to prevent zombie state ===
                if sentinel_ib is not None and sentinel_ib.isConnected():
                    logger.info("Market Closed: Releasing Sentinel IB connection to pool.")
                    try:
                        await IBConnectionPool.release_connection("sentinel")
                    except Exception as e:
                        logger.warning(f"Failed to release sentinel connection to pool, disconnecting directly: {e}")
                        sentinel_ib.disconnect()
                    # === Give Gateway time to cleanup ===
                    await asyncio.sleep(3.0)
                    sentinel_ib = None
                    price_sentinel.ib = None

                    # Reset outage tracking (not relevant when market is closed)
                    last_successful_ib_time = None
                    outage_notification_sent = False

                    _record_sentinel_health("PriceSentinel", "IDLE", 60)

            # === CONTRACT CACHE REFRESH ===
            if sentinel_ib and sentinel_ib.isConnected() and (now - last_contract_refresh > CONTRACT_REFRESH_INTERVAL):
                try:
                    active_futures = await asyncio.wait_for(get_active_futures(sentinel_ib, config['symbol'], config['exchange'], count=1), timeout=15)
                    cached_contract = active_futures[0] if active_futures else None
                    last_contract_refresh = now
                    logger.info(f"Refreshed contract cache: {cached_contract.localSymbol if cached_contract else 'None'}")
                except Exception as e:
                    logger.error(f"Failed to refresh contract cache: {e}")

            # === MICROSTRUCTURE SENTINEL LIFECYCLE ===
            gateway_available = sentinel_ib is not None and sentinel_ib.isConnected()

            if market_open and micro_sentinel is None and gateway_available:
                logger.info("Market Open: Engaging Microstructure Sentinel")
                try:
                    micro_ib = await IBConnectionPool.get_connection("microstructure", config)
                    configure_market_data_type(micro_ib)
                    micro_sentinel = MicrostructureSentinel(config, micro_ib)
                    _record_sentinel_health("MicrostructureSentinel", "OK", 60)

                    target = cached_contract
                    if not target and sentinel_ib.isConnected():
                        active_futures = await asyncio.wait_for(get_active_futures(sentinel_ib, config['symbol'], config['exchange'], count=1), timeout=15)
                        if active_futures:
                            target = active_futures[0]

                    if target:
                        await micro_sentinel.subscribe_contract(target)
                    else:
                        logger.warning("No active futures found for Microstructure Sentinel")

                except Exception as e:
                    logger.error(f"Failed to engage MicrostructureSentinel: {e}")
                    micro_sentinel = None
                    # Release connection on failure to prevent pool exhaustion
                    try:
                        await IBConnectionPool.release_connection("microstructure")
                    except Exception:
                        pass
                    micro_ib = None
                    _record_sentinel_health("MicrostructureSentinel", "ERROR", 60, str(e))

            elif market_open and micro_sentinel is None and not gateway_available:
                logger.debug("Skipping Microstructure engagement - Gateway unavailable")

            elif not market_open and micro_sentinel is not None:
                logger.info("Market Closed: Disengaging Microstructure Sentinel")
                try:
                    await micro_sentinel.unsubscribe_all()
                except Exception as e:
                    logger.error(f"Error unsubscribing microstructure: {e}")

                micro_sentinel = None
                await IBConnectionPool.release_connection("microstructure")
                micro_ib = None
                _record_sentinel_health("MicrostructureSentinel", "IDLE", 60)

            # === RUN SENTINELS ===

            # 1. Price Sentinel (Every 1 min) - ONLY if IB connected
            if sentinel_ib and sentinel_ib.isConnected():
                _health_error = None
                try:
                    trigger = await asyncio.wait_for(price_sentinel.check(cached_contract=cached_contract), timeout=30)
                    trigger = validate_trigger(trigger)

                    # === NEW: Price Move Triggers Position Audit ===
                    # If PriceSentinel detects significant move, proactively check theses
                    if trigger and trigger.source == 'PriceSentinel':
                        price_change = abs(trigger.payload.get('change', 0))
                        if price_change >= 1.5:  # Pre-emptive at 1.5% (before 2% breach)
                            logger.info(f"PriceSentinel detected {price_change:.1f}% move - triggering position audit")
                            audit_task = asyncio.create_task(run_position_audit_cycle(
                                config,
                                f"PriceSentinel trigger ({price_change:.1f}% move)"
                            ))
                            _INFLIGHT_TASKS.add(audit_task)
                            audit_task.add_done_callback(
                                lambda t: _log_task_exception(t, "position_audit_price_trigger")
                            )

                    if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                        task = asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                        _INFLIGHT_TASKS.add(task)
                        task.add_done_callback(
                            lambda t, src=trigger.source, cfg=config: _emergency_cycle_done_callback(t, src, cfg)
                        )
                        GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                except asyncio.TimeoutError:
                    logger.error("PriceSentinel TIMED OUT after 30s")
                    _health_error = "TIMEOUT after 30s"
                    SENTINEL_STATS.record_error("PriceSentinel", "TIMEOUT")
                except Exception as e:
                    logger.error(f"PriceSentinel check failed: {type(e).__name__}: {e}")
                    _health_error = str(e)
                finally:
                    _record_sentinel_health(
                        "PriceSentinel",
                        "ERROR" if _health_error else "OK",
                        60,
                        _health_error
                    )

            # 2. Weather Sentinel (Every 4 hours) - Runs 24/7, no IB needed
            last_weather = await _run_periodic_sentinel(
                weather_sentinel, last_weather, 14400, 60, config, sentinel_ib, market_open
            )

            # 3. Logistics Sentinel (Every 6 hours) - Runs 24/7, no IB needed
            last_logistics = await _run_periodic_sentinel(
                logistics_sentinel, last_logistics, 21600, 90, config, sentinel_ib, market_open
            )

            # 4. News Sentinel (Every 2 hours) - Runs 24/7, no IB needed
            last_news = await _run_periodic_sentinel(
                news_sentinel, last_news, 7200, 90, config, sentinel_ib, market_open
            )

            # 5. X Sentiment Sentinel (Every 90 min during market-adjacent hours on trading days)
            if trading_day and (now - last_x_sentiment) > 5400:
                # Only run during market-adjacent hours (6:00 AM - 4:30 PM ET)
                from datetime import time as dt_time
                et_now = datetime.now(pytz.timezone('US/Eastern'))
                x_start = dt_time(6, 0)
                x_end = dt_time(16, 30)

                if x_start <= et_now.time() <= x_end:
                    _health_error = None
                    try:
                        # Reset daily stats if new day
                        if datetime.now().date() != x_sentinel_stats['last_reset']:
                            x_sentinel_stats = {
                                'checks_today': 0,
                                'triggers_today': 0,
                                'estimated_tokens': 0,
                                'estimated_cost_usd': 0.0,
                                'last_reset': datetime.now().date()
                            }

                        trigger = await asyncio.wait_for(x_sentinel.check(), timeout=120)
                        trigger = validate_trigger(trigger)
                        x_sentinel_stats['checks_today'] += 1

                        if trigger:
                            logger.info(f"XSentimentSentinel: trigger detected (severity={getattr(trigger, 'severity', '?')})")
                            x_sentinel_stats['triggers_today'] += 1
                            if market_open and sentinel_ib and sentinel_ib.isConnected():
                                if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                    task = asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                    _INFLIGHT_TASKS.add(task)
                                    task.add_done_callback(
                                        lambda t, src=trigger.source, cfg=config: _emergency_cycle_done_callback(t, src, cfg)
                                    )
                                    GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                            else:
                                StateManager.queue_deferred_trigger(trigger)
                                logger.info(f"Deferred {trigger.source} trigger for market open")
                    except asyncio.TimeoutError:
                        logger.error("XSentimentSentinel TIMED OUT after 120s")
                        _health_error = "TIMEOUT after 120s"
                        SENTINEL_STATS.record_error("XSentimentSentinel", "TIMEOUT")
                    except Exception as e:
                        logger.error(f"XSentimentSentinel check failed: {type(e).__name__}: {e}")
                        _health_error = str(e)
                    finally:
                        last_x_sentiment = now
                        _record_sentinel_health(
                            "XSentimentSentinel",
                            "ERROR" if _health_error else "OK",
                            5400,
                            _health_error
                        )
                else:
                    # Outside operating window - IDLE
                    _record_sentinel_health("XSentimentSentinel", "IDLE", 5400)
                    last_x_sentiment = now

            elif not trading_day and (now - last_x_sentiment) > 5400:
                # Weekend/Holiday update - IDLE
                _record_sentinel_health("XSentimentSentinel", "IDLE", 5400)
                last_x_sentiment = now

            # 6. Prediction Market Sentinel (Every 5 minutes) - Runs 24/7, no IB needed
            prediction_config = config.get('sentinels', {}).get('prediction_markets', {})
            prediction_interval = prediction_config.get('poll_interval_seconds', 300)

            last_prediction_market = await _run_periodic_sentinel(
                prediction_market_sentinel, last_prediction_market, prediction_interval, 120, config, sentinel_ib, market_open, cooldown_seconds=1800
            )

            # 7. Macro Contagion Sentinel (Every 4 hours) - Runs 24/7, no IB needed
            last_macro_contagion = await _run_periodic_sentinel(
                macro_contagion_sentinel, last_macro_contagion, 14400, 60, config, sentinel_ib, market_open
            )

            # 8. Topic Discovery Agent (Every 12 hours) - Runs 24/7, no IB needed
            discovery_config = config.get('sentinels', {}).get('prediction_markets', {}).get('discovery_agent', {})
            discovery_interval = discovery_config.get('scan_interval_hours', 12) * 3600

            if discovery_config.get('enabled', False) and (now - last_topic_discovery) > discovery_interval:
                try:
                    from trading_bot.topic_discovery import TopicDiscoveryAgent
                    # Inject GLOBAL_BUDGET_GUARD (dependency injection)
                    discovery_agent = TopicDiscoveryAgent(config, budget_guard=GLOBAL_BUDGET_GUARD)
                    result = await discovery_agent.run_scan()
                    logger.info(
                        f"TopicDiscovery: {result['metadata']['topics_discovered']} topics, "
                        f"{result['changes']['summary']}"
                    )

                    # If topics changed, hot-reload the sentinel (preserves state)
                    if result.get('changes', {}).get('has_changes'):
                        prediction_market_sentinel.reload_topics()
                except Exception as e:
                    logger.error(f"TopicDiscoveryAgent scan failed: {type(e).__name__}: {e}")
                    # Retry in 1 hour on failure, not the full 12-hour interval
                    last_topic_discovery = now - discovery_interval + 3600
                else:
                    last_topic_discovery = now

            # 9. Microstructure Sentinel (Every 1 min with Price Sentinel)
            if micro_sentinel and micro_ib and micro_ib.isConnected():
                _health_error = None
                try:
                    micro_trigger = await asyncio.wait_for(micro_sentinel.check(), timeout=30)
                    if micro_trigger:
                        from trading_bot.notifications import get_notification_tier, NotificationTier
                        tier = get_notification_tier(micro_trigger.severity)

                        if tier == NotificationTier.CRITICAL:
                            # Severity 9-10: Full emergency cycle (must still pass deduplicator)
                            logger.warning(f"MICROSTRUCTURE CRITICAL: {micro_trigger.reason}")
                            if GLOBAL_DEDUPLICATOR.should_process(micro_trigger):
                                task = asyncio.create_task(run_emergency_cycle(micro_trigger, config, sentinel_ib))
                                _INFLIGHT_TASKS.add(task)
                                task.add_done_callback(
                                    lambda t, src=micro_trigger.source, cfg=config: _emergency_cycle_done_callback(t, src, cfg)
                                )
                                GLOBAL_DEDUPLICATOR.set_cooldown(micro_trigger.source, 900)
                        elif tier == NotificationTier.PUSHOVER:
                            # Severity 7-8: Log + Pushover but NO emergency cycle
                            # Liquidity depletion is informational, not actionable by council
                            logger.warning(f"MICROSTRUCTURE ALERT: {micro_trigger.reason}")
                            if GLOBAL_DEDUPLICATOR.should_process(micro_trigger):
                                StateManager.log_sentinel_event(micro_trigger)
                                SENTINEL_STATS.record_alert(micro_trigger.source, triggered_trade=False)
                                send_pushover_notification(
                                    config.get('notifications', {}),
                                    "Microstructure Alert",
                                    f"{micro_trigger.reason} (Severity: {micro_trigger.severity:.1f})"
                                )
                                GLOBAL_DEDUPLICATOR.set_cooldown(micro_trigger.source, 1800)
                        elif tier == NotificationTier.DASHBOARD:
                            # Severity 5-6: Log only (visible in dashboard via sentinel history)
                            logger.info(f"MICROSTRUCTURE NOTE: {micro_trigger.reason} (Sev: {micro_trigger.severity})")
                            StateManager.log_sentinel_event(micro_trigger)
                        else:
                            # Severity 0-4: Debug log only
                            logger.debug(f"MICROSTRUCTURE LOW: {micro_trigger.reason}")
                except asyncio.TimeoutError:
                    logger.error("MicrostructureSentinel TIMED OUT after 30s")
                    _health_error = "TIMEOUT after 30s"
                    SENTINEL_STATS.record_error("MicrostructureSentinel", "TIMEOUT")
                except Exception as e:
                    logger.error(f"MicrostructureSentinel check failed: {type(e).__name__}: {e}")
                    _health_error = str(e)
                finally:
                    _record_sentinel_health(
                        "MicrostructureSentinel",
                        "ERROR" if _health_error else "OK",
                        60,
                        _health_error
                    )

            await asyncio.sleep(60)  # Loop tick

        except asyncio.CancelledError:
            logger.info("Sentinel Loop Cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in Sentinel Loop: {e}")
            await asyncio.sleep(60)

    # Close shared aiohttp sessions on all sentinels
    for s in [weather_sentinel, logistics_sentinel, news_sentinel,
              x_sentinel, prediction_market_sentinel, macro_contagion_sentinel,
              price_sentinel]:
        try:
            await s.close()
        except Exception:
            pass
    if micro_sentinel is not None:
        try:
            await micro_sentinel.close()
        except Exception:
            pass


# --- Main Schedule ---
# IMPORTANT: Keys are New York Local Time.
# The orchestrator dynamically converts these to UTC based on DST.

async def emergency_hard_close(config: dict):
    """
    Last-resort position closure at 13:15 ET using MARKET orders.

    This runs 45 minutes before market close. If we still have open positions
    at this point, limit orders have failed and we accept slippage to protect
    against overnight/weekend risk.
    """
    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logger.info("[OFF] emergency_hard_close skipped — no positions exist in OFF mode")
        return

    logger.info("--- Emergency Hard Close Check (T-45 min) ---")

    ib = None
    try:
        ib = await IBConnectionPool.get_connection("orchestrator_orders", config)
        configure_market_data_type(ib)

        commodity_symbol = config.get('symbol', 'KC')
        try:
            live_positions = await asyncio.wait_for(ib.reqPositionsAsync(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("reqPositionsAsync timed out (30s) in emergency hard close, aborting")
            return
        open_positions = [
            p for p in live_positions
            if p.position != 0 and p.contract.symbol == commodity_symbol
        ]

        if not open_positions:
            logger.info("Emergency hard close: No open positions. All clear. ✓")
            return

        position_count = len(open_positions)
        logger.warning(
            f"🚨 EMERGENCY HARD CLOSE: {position_count} positions still open at T-45! "
            f"Closing with MARKET orders (slippage accepted)."
        )

        send_pushover_notification(
            config.get('notifications', {}),
            "🚨 Emergency Hard Close Triggered",
            f"{position_count} positions still open at 13:15 ET.\n"
            f"Closing with MARKET orders. Slippage expected."
        )

        closed = 0
        failed = 0

        for pos in open_positions:
            try:
                close_action = 'SELL' if pos.position > 0 else 'BUY'
                qty = abs(pos.position)

                # CRITICAL FIX: Re-qualify by conId only to get correct strike format
                minimal = Contract(conId=pos.contract.conId)
                try:
                    qualified = await asyncio.wait_for(
                        ib.qualifyContractsAsync(minimal), timeout=15
                    )
                except asyncio.TimeoutError:
                    logger.error(f"qualifyContractsAsync timed out (15s) for {pos.contract.localSymbol}")
                    failed += 1
                    continue
                if not qualified or qualified[0].conId == 0:
                    logger.error(f"Could not qualify {pos.contract.localSymbol} (conId={pos.contract.conId})")
                    failed += 1
                    continue
                contract = qualified[0]

                # L3 FIX: Protective Close Order
                # Fetch price for limit cap
                ticker = ib.reqMktData(contract, '', True, False)
                await asyncio.sleep(1)
                last_price = ticker.last if not util.isNan(ticker.last) else ticker.close

                if last_price and not util.isNan(last_price) and last_price > 0:
                    slippage_pct = config.get('execution', {}).get('max_slippage_pct', 0.02)
                    if close_action == 'BUY':
                        limit_price = last_price * (1 + slippage_pct)
                    else:
                        limit_price = last_price * (1 - slippage_pct)

                    tick_size = get_tick_size(config)
                    limit_price = round_to_tick(limit_price, tick_size)

                    order = LimitOrder(close_action, qty, limit_price)
                    order.tif = 'GTC'
                    logger.info(f"L3: Protective close at {limit_price:.4f} (last: {last_price:.4f}, cap: {slippage_pct:.1%})")
                else:
                    logger.warning(f"L3: No price for {contract.localSymbol}, using market order")
                    order = MarketOrder(close_action, qty)
                    order.tif = 'GTC'

                trade = ib.placeOrder(contract, order)
                logger.info(f"Emergency close: {close_action} {qty} {contract.localSymbol}")

                for _ in range(60):
                    await asyncio.sleep(1)
                    if trade.isDone():
                        break

                if trade.orderStatus.status == 'Filled':
                    closed += 1
                    logger.info(f"Emergency fill: {contract.localSymbol} @ {trade.orderStatus.avgFillPrice}")
                else:
                    failed += 1
                    logger.error(f"Emergency close incomplete: {contract.localSymbol}")

            except Exception as e:
                failed += 1
                logger.error(f"Emergency close failed for {pos.contract.localSymbol}: {e}")

        summary = f"Emergency hard close: {closed} closed, {failed} failed"
        logger.info(summary)
        send_pushover_notification(config.get('notifications', {}), "Emergency Close Result", summary)

        # Sweep-invalidate active theses for positions we just closed.
        # Without this, ghost theses remain active until the 5AM cleanup job.
        if closed > 0:
            try:
                sweep_tms = TransactiveMemory()
                if sweep_tms.collection:
                    active_results = sweep_tms.collection.get(
                        where={"active": "true"},
                        include=['metadatas']
                    )
                    swept = 0
                    for meta in active_results.get('metadatas', []):
                        tid = meta.get('trade_id')
                        if tid:
                            sweep_tms.invalidate_thesis(
                                tid,
                                "Emergency hard close sweep"
                            )
                            swept += 1
                    if swept > 0:
                        logger.warning(
                            f"Emergency hard close: Swept {swept} active theses"
                        )
            except Exception as sweep_err:
                logger.warning(f"Thesis sweep after emergency hard close failed (non-fatal): {sweep_err}")

    except Exception as e:
        logger.critical(f"Emergency hard close FAILED entirely: {e}")
        send_pushover_notification(
            config.get('notifications', {}),
            "🚨🚨 EMERGENCY CLOSE FAILED",
            f"Could not execute emergency hard close: {str(e)[:200]}\n"
            f"MANUAL INTERVENTION REQUIRED IMMEDIATELY."
        )
    finally:
        if ib is not None:
            try:
                await IBConnectionPool.release_connection("orchestrator_orders")
            except Exception:
                pass

async def close_stale_positions_fallback(config: dict):
    """Fallback close attempt at 12:45 ET. Only acts if 11:00 primary close missed anything."""
    from trading_bot.utils import is_trading_off
    if is_trading_off():
        logger.info("[OFF] close_stale_positions_fallback skipped — no positions exist in OFF mode")
        return

    logger.info("--- Fallback Close Attempt (12:45 ET) ---")
    logger.info("This is a retry for any positions the 11:00 primary close failed to handle.")
    await close_stale_positions(config)

async def run_brier_reconciliation(config: dict):
    """Automated Brier prediction reconciliation."""
    try:
        from trading_bot.brier_reconciliation import resolve_with_cycle_aware_match
        resolved = resolve_with_cycle_aware_match(dry_run=False)
        logger.info(f"Brier reconciliation complete: {resolved} predictions resolved")

        # v5.5: Enhanced Brier catch-up — resolve JSON predictions from CSV data.
        # This fixes the pipeline gap where resolve_with_cycle_aware_match resolves
        # the CSV but doesn't update enhanced_brier.json.
        try:
            from trading_bot.brier_bridge import backfill_enhanced_from_csv, reset_enhanced_tracker
            backfilled = backfill_enhanced_from_csv()
            if backfilled > 0:
                logger.info(f"Enhanced Brier backfill: {backfilled} predictions caught up from CSV")
                reset_enhanced_tracker()  # Reset singleton so voting picks up new scores
        except Exception as backfill_e:
            logger.warning(f"Enhanced Brier backfill failed (non-fatal): {backfill_e}")

        # v5.4: Stall detection — alert after 3 consecutive days with 0 resolutions
        global _brier_zero_resolution_streak
        try:
            import json
            brier_path = os.path.join(config.get('data_dir', 'data'), 'enhanced_brier.json')
            if os.path.exists(brier_path):
                with open(brier_path, 'r') as f:
                    brier_data = json.load(f)
                pending = sum(
                    1 for p in brier_data.get('predictions', [])
                    if p.get('resolved_at') is None
                )
                resolved_today = sum(
                    1 for p in brier_data.get('predictions', [])
                    if p.get('resolved_at') and p['resolved_at'].startswith(
                        datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    )
                )
                if resolved_today == 0 and pending > 0:
                    _brier_zero_resolution_streak += 1
                    logger.warning(
                        f"Brier stall: {pending} pending, 0 resolved today "
                        f"(streak: {_brier_zero_resolution_streak} days)"
                    )
                    if _brier_zero_resolution_streak >= 3:
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "⚠️ Brier Reconciliation Stall",
                            f"{pending} predictions pending, 0 resolved for "
                            f"{_brier_zero_resolution_streak} consecutive days. "
                            f"Check council_history backfill and schedule ordering."
                        )
                else:
                    _brier_zero_resolution_streak = 0
                    if resolved_today > 0:
                        logger.info(f"Brier reconciliation: {resolved_today} predictions resolved today")
        except Exception as e:
            logger.debug(f"Brier stall check error (non-critical): {e}")

    except Exception as e:
        logger.error(f"Brier reconciliation failed (non-fatal): {e}")

    # Feedback loop health check — runs here (after CSV + Enhanced Brier resolution)
    # so it reports accurate post-resolution counts.
    await _check_feedback_loop_health(config)

async def guarded_generate_orders(config: dict):
    """Generate orders with budget and cutoff guards."""

    # v3.1: Check equity data freshness before cycle
    await check_and_recover_equity_data(config)

    if GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit:
        logger.warning("Budget hit - skipping scheduled orders.")
        return

    # --- QUARANTINE HEALTH CHECK (Fix B2) ---
    # Ensure any agents past their quarantine cooldown are released
    # before the order generation cycle begins.
    try:
        from trading_bot.agents import TradingCouncil
        temp_council = TradingCouncil(config)
        if hasattr(temp_council, 'observability') and temp_council.observability:
            hub = temp_council.observability
            if hasattr(hub, 'hallucination_detector'):
                detector = hub.hallucination_detector
                for agent_name in list(detector.quarantined_agents):
                    recent_flags = [
                        f for f in detector.agent_flags.get(agent_name, [])
                        if (datetime.now(timezone.utc) - f.timestamp).days < 7
                    ]
                    if not recent_flags:
                        detector.quarantined_agents.discard(agent_name)
                        logger.info(
                            f"Pre-cycle quarantine release: {agent_name} "
                            f"(no flags in 7-day window)"
                        )
                    else:
                        most_recent = max(f.timestamp for f in recent_flags)
                        hours_since = (datetime.now(timezone.utc) - most_recent).total_seconds() / 3600
                        if hours_since > 48:
                            detector.quarantined_agents.discard(agent_name)
                            logger.info(
                                f"Pre-cycle quarantine release: {agent_name} "
                                f"(clean for {hours_since:.1f}h)"
                            )
                detector._save_state()  # Persist release
    except Exception as e:
        logger.warning(f"Pre-cycle quarantine check failed: {e}")

    # === Shutdown proximity check ===
    # Use cutoff + buffer as proxy for shutdown time (session-aware)
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)
    co_h, co_m = get_trading_cutoff(config)
    shutdown_time = time(co_h, co_m)
    minutes_to_shutdown = (
        datetime.combine(now_ny.date(), shutdown_time) -
        datetime.combine(now_ny.date(), now_ny.time())
    ).total_seconds() / 60

    if 0 < minutes_to_shutdown < 30:
        logger.warning(
            f"Order generation SKIPPED: Only {minutes_to_shutdown:.0f} min to shutdown. "
            f"Insufficient time for full council cycle."
        )
        return

    # Daily cutoff check
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)
    cutoff_hour, cutoff_minute = get_trading_cutoff(config)

    if now_ny.hour > cutoff_hour or (now_ny.hour == cutoff_hour and now_ny.minute >= cutoff_minute):
        logger.info(f"Order generation BLOCKED: Past daily cutoff ({cutoff_hour}:{cutoff_minute:02d} ET)")
        return

    # Check Drawdown Guard
    if GLOBAL_DRAWDOWN_GUARD:
        ib = None
        try:
             # Need a connection to check P&L.
             # generate_and_execute_orders creates its own connection.
             # We will rely on order_manager to check again, or check here if possible.
             # Ideally we check here to avoid spinning up the order generation logic.
             ib = await IBConnectionPool.get_connection("drawdown_check", config)
             if not ib.isConnected():
                 logger.warning("IB disconnected — skipping drawdown P&L update (guard state preserved)")
             else:
                 await GLOBAL_DRAWDOWN_GUARD.update_pnl(ib)
             if not GLOBAL_DRAWDOWN_GUARD.is_entry_allowed():
                 logger.warning("Order generation BLOCKED: Drawdown Guard Active")
                 return
        except Exception as e:
             logger.error(f"Drawdown guard check failed (fail-closed): {e}")
             send_pushover_notification(
                 config.get('notifications', {}),
                 "⚠️ Drawdown Check Failed",
                 f"Order generation blocked — drawdown guard unreachable: {e}"
             )
             return
        finally:
             if ib is not None:
                 try:
                     await IBConnectionPool.release_connection("drawdown_check")
                 except Exception:
                     pass

    await generate_and_execute_orders(config, shutdown_check=is_system_shutdown)

    # Invalidate semantic cache after scheduled cycle (fresh analysis supersedes cached)
    try:
        sc = get_semantic_cache(config)
        ticker = get_active_ticker(config)
        sc.invalidate_by_ticker(ticker)
    except Exception as e:
        logger.warning(f"Failed to invalidate semantic cache post-scheduled: {e}")

@dataclass
class ScheduledTask:
    """A single scheduled task with a unique ID for per-instance tracking."""
    id: str           # Unique task ID (e.g., "signal_early")
    time_et: time     # NY local time
    function: Callable
    func_name: str    # function.__name__, for RECOVERY_POLICY lookup
    label: str        # Human-readable label for dashboard/logs

# Maps config.json function name strings to actual Python callables.
FUNCTION_REGISTRY = {
    'start_monitoring': start_monitoring,
    'process_deferred_triggers': process_deferred_triggers,
    'cleanup_orphaned_theses': cleanup_orphaned_theses,
    'guarded_generate_orders': guarded_generate_orders,
    'run_position_audit_cycle': run_position_audit_cycle,
    'close_stale_positions': close_stale_positions,
    'close_stale_positions_fallback': close_stale_positions_fallback,
    'emergency_hard_close': emergency_hard_close,
    'cancel_and_stop_monitoring': cancel_and_stop_monitoring,
    'log_equity_snapshot': log_equity_snapshot,
    'reconcile_and_analyze': reconcile_and_analyze,
    'run_brier_reconciliation': run_brier_reconciliation,
    'sentinel_effectiveness_check': sentinel_effectiveness_check,
}


def _build_default_schedule() -> list:
    """Backward-compatible default schedule (19 tasks) as list[ScheduledTask]."""
    defaults = [
        ("start_monitoring",          time(3, 30),  start_monitoring,              "Start Position Monitoring"),
        ("process_deferred_triggers", time(3, 31),  process_deferred_triggers,     "Process Deferred Triggers"),
        ("cleanup_orphaned_theses",   time(5, 0),   cleanup_orphaned_theses,       "Daily Thesis Cleanup"),
        ("signal_early",              time(9, 0),   guarded_generate_orders,       "Signal: Early Session (09:00 ET)"),
        ("signal_euro",               time(11, 0),  guarded_generate_orders,       "Signal: EU Overlap (11:00 ET)"),
        ("signal_us_open",            time(13, 0),  guarded_generate_orders,       "Signal: US Open (13:00 ET)"),
        ("signal_peak",               time(15, 0),  guarded_generate_orders,       "Signal: Peak Liquidity (15:00 ET)"),
        ("signal_settlement",         time(17, 0),  guarded_generate_orders,       "Signal: Settlement (17:00 ET)"),
        ("audit_morning",             time(13, 30), run_position_audit_cycle,      "Audit: Midday (13:30 ET)"),
        ("close_stale_primary",       time(15, 30), close_stale_positions,         "Close Stale: Primary (15:30 ET)"),
        ("audit_post_close",          time(15, 45), run_position_audit_cycle,      "Audit: Post-Close (15:45 ET)"),
        ("close_stale_fallback",      time(16, 30), close_stale_positions_fallback,"Close Stale: Fallback (16:30 ET)"),
        ("audit_pre_close",           time(17, 15), run_position_audit_cycle,      "Audit: Pre-Shutdown (17:15 ET)"),
        ("emergency_hard_close",      time(17, 30), emergency_hard_close,          "Emergency Hard Close (17:30 ET)"),
        ("eod_shutdown",              time(18, 0),  cancel_and_stop_monitoring,    "End-of-Day Shutdown (18:00 ET)"),
        ("log_equity_snapshot",       time(18, 20), log_equity_snapshot,           "Log Equity Snapshot (18:20 ET)"),
        ("reconcile_and_analyze",     time(18, 25), reconcile_and_analyze,         "Reconcile & Analyze (18:25 ET)"),
        ("brier_reconciliation",      time(18, 35), run_brier_reconciliation,      "Brier Reconciliation (18:35 ET)"),
        ("sentinel_effectiveness",    time(18, 40), sentinel_effectiveness_check,  "Sentinel Effectiveness Check (18:40 ET)"),
    ]
    return [
        ScheduledTask(id=tid, time_et=t, function=fn, func_name=fn.__name__, label=lbl)
        for tid, t, fn, lbl in defaults
    ]


def _build_session_schedule(config: dict) -> list:
    """Build schedule from session_template, anchored to commodity trading hours.

    Derives all task times from the active commodity profile's trading_hours_et
    field instead of absolute clock times.

    Returns list[ScheduledTask] sorted by time_et.
    """
    from config.commodity_profiles import get_active_profile, parse_trading_hours

    profile = get_active_profile(config)
    open_t, close_t = parse_trading_hours(profile.contract.trading_hours_et)

    tmpl = config['schedule']['session_template']
    today = datetime.now(timezone.utc).date()

    open_dt = datetime.combine(today, open_t)
    close_dt = datetime.combine(today, close_t)
    session_minutes = (close_dt - open_dt).total_seconds() / 60

    result = []
    seen_ids = set()

    def _add_task(task_id, task_time, func_name, label):
        if task_id in seen_ids:
            raise ValueError(f"Duplicate schedule task ID: '{task_id}'")
        seen_ids.add(task_id)
        func = FUNCTION_REGISTRY.get(func_name)
        if func is None:
            logger.warning(f"Unknown function '{func_name}' in session task '{task_id}' — skipping")
            return
        result.append(ScheduledTask(
            id=task_id,
            time_et=task_time,
            function=func,
            func_name=func_name,
            label=label,
        ))

    # 1. Pre-open tasks: open_time + offset_minutes (negative offsets = before open)
    for entry in tmpl.get('pre_open_tasks', []):
        dt = open_dt + timedelta(minutes=entry['offset_minutes'])
        _add_task(entry['id'], dt.time(), entry['function'], entry.get('label', entry['id']))

    # 2. Signal generation: evenly distributed between start_pct and end_pct
    signal_count = tmpl.get('signal_count', 4)
    start_pct = tmpl.get('signal_start_pct', 0.05)
    end_pct = tmpl.get('signal_end_pct', 0.80)

    signal_names = ['signal_open', 'signal_early', 'signal_mid', 'signal_late', 'signal_5']
    signal_labels = ['Signal: Open', 'Signal: Early', 'Signal: Mid', 'Signal: Late', 'Signal: 5']

    if signal_count == 1:
        pcts = [start_pct]
    else:
        step = (end_pct - start_pct) / (signal_count - 1)
        pcts = [start_pct + i * step for i in range(signal_count)]

    for i, pct in enumerate(pcts):
        dt = open_dt + timedelta(minutes=session_minutes * pct)
        sid = signal_names[i] if i < len(signal_names) else f'signal_{i+1}'
        slbl = signal_labels[i] if i < len(signal_labels) else f'Signal: {i+1}'
        _add_task(sid, dt.time(), 'guarded_generate_orders', slbl)

    # 3. Intra-session tasks: open_time + (session_minutes * session_pct)
    for entry in tmpl.get('intra_session_tasks', []):
        dt = open_dt + timedelta(minutes=session_minutes * entry['session_pct'])
        _add_task(entry['id'], dt.time(), entry['function'], entry.get('label', entry['id']))

    # 4. Pre-close tasks: close_time + offset_minutes (negative offsets = before close)
    for entry in tmpl.get('pre_close_tasks', []):
        dt = close_dt + timedelta(minutes=entry['offset_minutes'])
        _add_task(entry['id'], dt.time(), entry['function'], entry.get('label', entry['id']))

    # 5. Post-close tasks: close_time + offset_minutes (positive offsets = after close)
    for entry in tmpl.get('post_close_tasks', []):
        dt = close_dt + timedelta(minutes=entry['offset_minutes'])
        _add_task(entry['id'], dt.time(), entry['function'], entry.get('label', entry['id']))

    result.sort(key=lambda t: (t.time_et.hour, t.time_et.minute))
    logger.info(f"Built session schedule: {len(result)} tasks for {profile.ticker} ({open_t.strftime('%H:%M')}-{close_t.strftime('%H:%M')} ET)")
    return result


def get_trading_cutoff(config: dict) -> tuple:
    """Get the daily trading cutoff as (hour, minute) in ET.

    In session mode: close_time - cutoff_before_close_minutes.
    Fallback: daily_trading_cutoff_et from config.
    """
    schedule_cfg = config.get('schedule', {})
    mode = schedule_cfg.get('mode', 'absolute')
    tmpl = schedule_cfg.get('session_template')

    if mode == 'session' and tmpl:
        try:
            from config.commodity_profiles import get_active_profile, parse_trading_hours
            profile = get_active_profile(config)
            _, close_t = parse_trading_hours(profile.contract.trading_hours_et)
            cutoff_minutes = tmpl.get('cutoff_before_close_minutes', 78)
            today = datetime.now(timezone.utc).date()
            close_dt = datetime.combine(today, close_t)
            cutoff_dt = close_dt - timedelta(minutes=cutoff_minutes)
            return (cutoff_dt.time().hour, cutoff_dt.time().minute)
        except Exception as e:
            logger.warning(f"Failed to compute session cutoff: {e}")

    # Fallback to absolute cutoff
    cutoff_cfg = schedule_cfg.get('daily_trading_cutoff_et', {'hour': 10, 'minute': 45})
    return (cutoff_cfg.get('hour', 10), cutoff_cfg.get('minute', 45))


def build_schedule(config: dict) -> list:
    """Build schedule from config, falling back to defaults.

    Supports two modes:
    - 'session': Derives times from commodity profile trading hours (preferred)
    - 'absolute': Uses explicit time_et values from tasks array (legacy)

    Returns list[ScheduledTask] sorted by time_et.
    Raises ValueError on duplicate task IDs.
    Logs warning and skips unknown function names.
    """
    schedule_cfg = config.get('schedule', {})
    mode = schedule_cfg.get('mode', 'absolute')

    # Session mode: derive times from commodity trading hours
    if mode == 'session' and schedule_cfg.get('session_template'):
        return _build_session_schedule(config)

    # Absolute mode: explicit times from tasks array
    tasks_cfg = schedule_cfg.get('tasks')
    if not tasks_cfg:
        logger.info("No schedule.tasks in config — using built-in default schedule")
        return _build_default_schedule()

    result = []
    seen_ids = set()
    for entry in tasks_cfg:
        task_id = entry['id']
        if task_id in seen_ids:
            raise ValueError(f"Duplicate schedule task ID: '{task_id}'")
        seen_ids.add(task_id)

        func_name = entry['function']
        func = FUNCTION_REGISTRY.get(func_name)
        if func is None:
            logger.warning(f"Unknown function '{func_name}' in schedule task '{task_id}' — skipping")
            continue

        h, m = map(int, entry['time_et'].split(':'))
        result.append(ScheduledTask(
            id=task_id,
            time_et=time(h, m),
            function=func,
            func_name=func_name,
            label=entry.get('label', task_id),
        ))

    result.sort(key=lambda t: (t.time_et.hour, t.time_et.minute))
    logger.info(f"Built config-driven schedule: {len(result)} tasks")
    return result


# Backward-compat: module-level schedule dict (used by test_weekend_behavior.py
# and sequential_main). Keys are time objects, values are callables — note that
# dict keys collapse duplicate times (e.g., multiple guarded_generate_orders).
schedule = {task.time_et: task.function for task in _build_default_schedule()}


def apply_schedule_offset(original_schedule, offset_minutes: int):
    """Shift schedule times by offset_minutes.

    Accepts list[ScheduledTask] (preferred) or dict (legacy).
    Returns same type as input.
    """
    today = datetime.now(timezone.utc).date()

    if isinstance(original_schedule, list):
        result = []
        for task in original_schedule:
            dt_original = datetime.combine(today, task.time_et)
            dt_shifted = dt_original + timedelta(minutes=offset_minutes)
            result.append(ScheduledTask(
                id=task.id,
                time_et=dt_shifted.time(),
                function=task.function,
                func_name=task.func_name,
                label=task.label,
            ))
        return result

    # Legacy dict path
    new_schedule = {}
    for run_time, task_func in original_schedule.items():
        dt_original = datetime.combine(today, run_time)
        dt_shifted = dt_original + timedelta(minutes=offset_minutes)
        new_schedule[dt_shifted.time()] = task_func
    return new_schedule

# =========================================================================
# RECOVERY POLICIES — Controls which missed tasks run on late/restart startup
# =========================================================================
# policy:
#   MARKET_OPEN   — Run if market is currently open (task has internal guards)
#   BEFORE_CUTOFF — Run if market is open AND before daily trading cutoff
#   ALWAYS        — Run regardless of market state (data/analysis tasks)
#   NEVER         — Do not auto-recover (shutdown tasks)
#
# idempotent:
#   True  — Safe to re-run (checks current state internally). No completion check needed.
#   False — NOT safe to re-run. Recovery will skip if already completed today.
#
# IMPORTANT: Keys MUST match the function names in `schedule` exactly.
# If a function is renamed, update this dict to match. The default fallback
# for unknown tasks is {'policy': 'MARKET_OPEN', 'idempotent': False} which
# is fail-safe (won't force-run a dangerous task), but recovery won't work
# optimally for the renamed function until this dict is updated.
#
RECOVERY_POLICY = {
    'start_monitoring':               {'policy': 'MARKET_OPEN',   'idempotent': True},
    'process_deferred_triggers':      {'policy': 'MARKET_OPEN',   'idempotent': True},
    'cleanup_orphaned_theses':        {'policy': 'ALWAYS',        'idempotent': True},
    'run_position_audit_cycle':       {'policy': 'MARKET_OPEN',   'idempotent': True},
    'guarded_generate_orders':        {'policy': 'BEFORE_CUTOFF', 'idempotent': False},  # 5 daily cycles (09:00-17:00 UTC)
    'close_stale_positions':          {'policy': 'MARKET_OPEN',   'idempotent': True},
    'close_stale_positions_fallback': {'policy': 'MARKET_OPEN',   'idempotent': True},
    'emergency_hard_close':           {'policy': 'MARKET_OPEN',   'idempotent': True},
    'cancel_and_stop_monitoring':     {'policy': 'NEVER',         'idempotent': False},
    'log_equity_snapshot':            {'policy': 'ALWAYS',        'idempotent': False},
    'reconcile_and_analyze':          {'policy': 'ALWAYS',        'idempotent': False},
    'run_brier_reconciliation':       {'policy': 'ALWAYS',        'idempotent': True},
    'sentinel_effectiveness_check':   {'policy': 'ALWAYS',        'idempotent': False},
}

# Startup validation: warn if default schedule has tasks not covered by RECOVERY_POLICY
_schedule_func_names = {task.func_name for task in _build_default_schedule()}
_policy_names = set(RECOVERY_POLICY.keys())
_uncovered = _schedule_func_names - _policy_names
if _uncovered:
    import logging as _log
    _log.getLogger(__name__).warning(
        f"⚠️ Schedule tasks without RECOVERY_POLICY entries (will use safe defaults): "
        f"{_uncovered}"
    )


async def recover_missed_tasks(missed_tasks: list, config: dict):
    """
    Generic recovery for missed scheduled tasks on late/restart startup.

    Runs missed tasks in chronological order based on their recovery policy.
    For non-idempotent tasks, checks the completion tracker to avoid
    re-executing tasks that already ran before a crash.

    Each unique task_id recovers independently — if 3 signal cycles were
    missed, all 3 are evaluated (not deduplicated). RECOVERY_POLICY is
    looked up by func_name (policy is about function behavior, not instance).

    IMPORTANT — Crash-during-execution edge case:
    If the orchestrator crashes AFTER a task sends an IB order but BEFORE
    record_task_completion writes, recovery will re-run that task. This is
    acceptable because non-idempotent tasks like guarded_generate_orders
    have internal guards (checking existing positions and open orders before
    submitting). Do NOT remove those internal guards — they are the last
    line of defense for this edge case.

    Args:
        missed_tasks: List of ScheduledTask objects (or legacy (time, name, func) tuples)
        config: Application config dict
    """
    if not missed_tasks:
        return

    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

    # Cutoff check for BEFORE_CUTOFF policy
    cutoff_hour, cutoff_minute = get_trading_cutoff(config)
    before_cutoff = (
        now_ny.hour < cutoff_hour or
        (now_ny.hour == cutoff_hour and now_ny.minute < cutoff_minute)
    )
    market_open = is_market_open(config)

    # Normalize: support both ScheduledTask and legacy (time, name, func) tuples
    normalized = []
    for item in missed_tasks:
        if isinstance(item, ScheduledTask):
            normalized.append(item)
        else:
            # Legacy tuple: (time, task_name, task_func)
            rt, name, func = item
            normalized.append(ScheduledTask(
                id=name, time_et=rt, function=func,
                func_name=name, label=name,
            ))

    # Sort chronologically
    sorted_missed = sorted(normalized, key=lambda t: (t.time_et.hour, t.time_et.minute))

    # Deduplicate by task_id — each instance recovers independently
    seen_ids = set()
    recovered = 0
    skipped = 0
    already_ran = 0

    logger.info(f"--- Recovery: Evaluating {len(sorted_missed)} missed tasks ---")

    for task in sorted_missed:
        if task.id in seen_ids:
            logger.debug(
                f"  ⏭️ {task.id} @ {task.time_et.strftime('%H:%M')} ET "
                f"(duplicate ID, already recovered)"
            )
            continue
        seen_ids.add(task.id)

        # Look up policy by func_name (policy is about function behavior)
        policy_entry = RECOVERY_POLICY.get(
            task.func_name, {'policy': 'MARKET_OPEN', 'idempotent': False}
        )
        policy = policy_entry['policy']
        is_idempotent = policy_entry['idempotent']

        # --- Completion check for non-idempotent tasks ---
        if not is_idempotent and has_task_completed_today(task.id):
            logger.info(
                f"✅ ALREADY RAN: {task.id} completed earlier today "
                f"(before restart). Skipping re-execution."
            )
            already_ran += 1
            continue

        # --- Policy check ---
        should_run = False
        skip_reason = ""

        if policy == 'NEVER':
            skip_reason = "policy=NEVER (shutdown task)"
        elif policy == 'ALWAYS':
            should_run = True
        elif policy == 'MARKET_OPEN':
            if market_open:
                should_run = True
            else:
                skip_reason = "market closed"
        elif policy == 'BEFORE_CUTOFF':
            if market_open and before_cutoff:
                should_run = True
            elif not market_open:
                skip_reason = "market closed"
            else:
                skip_reason = f"past cutoff ({cutoff_hour}:{cutoff_minute:02d} ET)"

        if should_run:
            logger.info(
                f"🔄 RECOVERY: Running missed {task.id} [{task.func_name}] "
                f"(was scheduled {task.time_et.strftime('%H:%M')} ET)"
            )
            try:
                await task.function(config)
                record_task_completion(task.id)
                logger.info(f"✅ RECOVERY: {task.id} completed")
                recovered += 1
            except Exception as e:
                logger.exception(f"❌ RECOVERY: {task.id} failed: {e}")
        else:
            logger.info(f"⏭️ RECOVERY SKIP: {task.id} — {skip_reason}")
            skipped += 1

    total_evaluated = recovered + skipped + already_ran
    summary = (
        f"System restarted with {total_evaluated} missed tasks.\n"
        f"✅ {recovered} recovered | ⏭️ {skipped} skipped | ✔️ {already_ran} already ran"
    )
    logger.info(summary)

    # Always send ONE notification on restart (even if nothing recovered)
    send_pushover_notification(
        config.get('notifications', {}),
        f"🔄 System Restart: {recovered} Tasks Recovered",
        summary
    )


async def main(commodity_ticker: str = None):
    """The main long-running orchestrator process."""
    # --- Multi-commodity path isolation ---
    ticker = commodity_ticker or os.environ.get("COMMODITY_TICKER", "KC")
    ticker = ticker.upper()
    os.environ["COMMODITY_TICKER"] = ticker  # Expose to notifications and dashboard
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', ticker)
    os.makedirs(data_dir, exist_ok=True)

    # Per-commodity logging (must be first — before any logger.info calls)
    setup_logging(log_file=f"logs/orchestrator_{ticker.lower()}.log")

    logger.info("=============================================")
    logger.info(f"=== Starting Trading Bot Orchestrator [{ticker}] ===")
    logger.info("=============================================")
    logger.info(f"Data directory: {data_dir}")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot start without a valid configuration.")
        return

    # Apply per-commodity config overrides (e.g. commodity_overrides.CC)
    from config_loader import deep_merge
    commodity_overrides = config.get('commodity_overrides', {}).get(ticker, {})
    if commodity_overrides:
        config = deep_merge(config, commodity_overrides)
        logger.info(f"Applied commodity overrides for {ticker}: {list(commodity_overrides.keys())}")

    # Inject data_dir and ticker into config for downstream modules
    config['data_dir'] = data_dir
    config['symbol'] = ticker
    config.setdefault('commodity', {})['ticker'] = ticker
    # Primary commodity owns account-wide equity tracking (NetLiquidation).
    # Non-primary commodities skip equity snapshots to avoid duplicate data.
    config['commodity']['is_primary'] = (ticker == 'KC')

    # --- Initialize all path-dependent modules BEFORE anything else ---
    from trading_bot.state_manager import StateManager
    from trading_bot.task_tracker import set_data_dir as set_tracker_dir
    from trading_bot.decision_signals import set_data_dir as set_signals_dir
    from trading_bot.order_manager import set_capital_state_dir
    from trading_bot.sentinel_stats import set_data_dir as set_stats_dir
    from trading_bot.utils import set_data_dir as set_utils_data_dir
    from trading_bot.tms import set_data_dir as set_tms_dir
    from trading_bot.brier_bridge import set_data_dir as set_brier_bridge_dir
    from trading_bot.brier_scoring import set_data_dir as set_brier_scoring_dir
    from trading_bot.weighted_voting import set_data_dir as set_weighted_voting_dir
    from trading_bot.brier_reconciliation import set_data_dir as set_brier_recon_dir
    from trading_bot.router_metrics import set_data_dir as set_router_metrics_dir
    from trading_bot.agents import set_data_dir as set_agents_dir

    StateManager.set_data_dir(data_dir)
    set_tracker_dir(data_dir)
    set_signals_dir(data_dir)
    set_capital_state_dir(data_dir)
    set_stats_dir(data_dir)
    set_utils_data_dir(data_dir)
    set_tms_dir(data_dir)
    set_brier_bridge_dir(data_dir)
    set_brier_scoring_dir(data_dir)
    set_weighted_voting_dir(data_dir)
    set_brier_recon_dir(data_dir)
    set_router_metrics_dir(data_dir)
    set_agents_dir(data_dir)

    # E.1: Portfolio VaR — shared data dir (NOT per-commodity)
    from trading_bot.var_calculator import set_var_data_dir
    from trading_bot.compliance import set_boot_time
    set_var_data_dir(os.path.dirname(data_dir))  # data/{ticker}/ → data/
    set_boot_time()

    global GLOBAL_DEDUPLICATOR
    GLOBAL_DEDUPLICATOR = TriggerDeduplicator(
        state_file=os.path.join(data_dir, 'deduplicator_state.json')
    )

    # Initialize trading mode
    from trading_bot.utils import set_trading_mode, is_trading_off
    set_trading_mode(config)
    if is_trading_off():
        logger.warning("=" * 60)
        logger.warning("  TRADING MODE: OFF — Training/Observation Only")
        logger.warning("  No real orders will be placed via IB")
        logger.warning("=" * 60)
        send_pushover_notification(
            config.get('notifications', {}),
            "Orchestrator Started (OFF Mode)",
            "Trading mode is OFF. Analysis pipeline runs normally. No orders will be placed."
        )

    # Remote Gateway indicator
    ib_host = config.get('connection', {}).get('host', '127.0.0.1')
    is_paper = config.get('connection', {}).get('paper', False)
    if ib_host not in ('127.0.0.1', 'localhost', '::1'):
        gw_label = "REMOTE/PAPER" if is_paper else "REMOTE"
        logger.warning("=" * 60)
        logger.warning(f"  IB GATEWAY: {gw_label} ({ib_host})")
        logger.warning(f"  Client IDs: DEV range (10-79)")
        logger.warning(f"  Trading Mode: {config.get('trading_mode', 'LIVE')}")
        logger.warning("=" * 60)
        if not is_trading_off() and not is_paper:
            logger.critical(
                "SAFETY: Remote gateway with TRADING_MODE=LIVE! "
                "Set TRADING_MODE=OFF or IB_PAPER=true in .env for dev environments."
            )
            send_pushover_notification(
                config.get('notifications', {}),
                "REMOTE GW + LIVE MODE",
                f"Orchestrator on remote GW ({ib_host}) with TRADING_MODE=LIVE. "
                "Likely a misconfiguration — set TRADING_MODE=OFF or IB_PAPER=true.",
                priority=1
            )

    # Update deduplicator with config values
    GLOBAL_DEDUPLICATOR.critical_severity_threshold = config.get('sentinels', {}).get('critical_severity_threshold', 9)

    # Initialize Budget Guard (singleton — shared with heterogeneous_router)
    global GLOBAL_BUDGET_GUARD
    GLOBAL_BUDGET_GUARD = get_budget_guard(config)
    logger.info(f"Budget Guard initialized. Daily limit: ${GLOBAL_BUDGET_GUARD.daily_budget}")

    # Initialize Drawdown Guard
    global GLOBAL_DRAWDOWN_GUARD
    GLOBAL_DRAWDOWN_GUARD = DrawdownGuard(config)
    logger.info("Drawdown Guard initialized.")

    # M6 FIX: Validate expiry overlap
    from config import get_active_profile
    profile = get_active_profile(config)
    if profile.min_dte >= profile.max_dte:
        raise ValueError(
            f"M6: Expiry filter overlap impossible: "
            f"min_dte ({profile.min_dte}) >= max_dte ({profile.max_dte})"
        )
    logger.info(f"Expiry filter window: {profile.min_dte}-{profile.max_dte} DTE")

    # Process deferred triggers from overnight - ONLY if market is open
    if is_market_open(config):
        await process_deferred_triggers(config)
    else:
        logger.info("Market Closed. Deferred triggers will remain queued.")

    env_name = os.getenv("ENV_NAME", "DEV") 
    is_prod = env_name.startswith("PROD")

    # Build config-driven schedule (falls back to defaults if no tasks in config)
    task_list = build_schedule(config)

    # Runtime RECOVERY_POLICY validation for config-driven schedule
    _cfg_func_names = {t.func_name for t in task_list}
    _cfg_uncovered = _cfg_func_names - set(RECOVERY_POLICY.keys())
    if _cfg_uncovered:
        logger.warning(
            f"Config schedule has functions without RECOVERY_POLICY entries "
            f"(will use safe defaults): {_cfg_uncovered}"
        )

    if not is_prod:
        schedule_offset_minutes = config.get('schedule', {}).get('dev_offset_minutes', -30)
        logger.info(f"Environment: {env_name}. Applying {schedule_offset_minutes} minute 'Civil War' avoidance offset.")
        task_list = apply_schedule_offset(task_list, offset_minutes=schedule_offset_minutes)
    else:
        schedule_offset_minutes = 0
        logger.info("Environment: PROD 🚀. Using standard master schedule.")

    # === WRITE ACTIVE SCHEDULE FOR DASHBOARD ===
    # The dashboard can't import orchestrator.py (it would pull in IB, agents, etc.)
    # so we write the effective schedule to a JSON file it can read independently.
    try:
        schedule_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "env": env_name,
            "offset_minutes": 0 if is_prod else schedule_offset_minutes,
            "tasks": [
                {
                    "id": task.id,
                    "time_et": task.time_et.strftime('%H:%M'),
                    "name": task.func_name,
                    "label": task.label,
                }
                for task in task_list
            ]
        }
        schedule_file = os.path.join(data_dir, 'active_schedule.json')
        os.makedirs(os.path.dirname(schedule_file), exist_ok=True)
        with open(schedule_file, 'w') as f:
            json.dump(schedule_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        logger.debug(f"Active schedule written: {len(schedule_data['tasks'])} tasks")
    except Exception as e:
        logger.warning(f"Failed to write active schedule (non-fatal): {e}")

    # === MISSED TASK DETECTION & RECOVERY ===
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

    missed_tasks = []
    for task in task_list:
        task_ny = now_ny.replace(hour=task.time_et.hour, minute=task.time_et.minute, second=0, microsecond=0)
        if task_ny < now_ny:
            missed_tasks.append(task)

    if missed_tasks:
        missed_names = [f"  - {t.time_et.strftime('%H:%M')} ET: {t.id}" for t in missed_tasks]
        logger.warning(
            f"⚠️ LATE START DETECTED: {len(missed_tasks)} scheduled tasks already passed:\n"
            + "\n".join(missed_names)
        )

        # === GENERIC RECOVERY: Run all valid missed tasks ===
        await recover_missed_tasks(missed_tasks, config)

    # === STARTUP: Run Topic Discovery Agent immediately ===
    # Ensures PredictionMarketSentinel has topics before sentinel loop begins.
    # The sentinel loop will handle subsequent 12-hour refreshes.
    global _STARTUP_DISCOVERY_TIME
    discovery_config = config.get('sentinels', {}).get('prediction_markets', {}).get('discovery_agent', {})
    if discovery_config.get('enabled', False):
        try:
            from trading_bot.topic_discovery import TopicDiscoveryAgent
            logger.info("Running TopicDiscoveryAgent on startup...")
            startup_discovery = TopicDiscoveryAgent(config, budget_guard=GLOBAL_BUDGET_GUARD)
            result = await startup_discovery.run_scan()
            logger.info(
                f"Startup TopicDiscovery: {result['metadata']['topics_discovered']} topics, "
                f"{result['changes']['summary']}"
            )
            _STARTUP_DISCOVERY_TIME = time_module.time()
        except Exception as e:
            logger.warning(f"Startup TopicDiscovery failed (sentinel loop will retry): {e}")

    # Start Sentinels in background
    sentinel_task = asyncio.create_task(run_sentinels(config))
    sentinel_task.add_done_callback(
        lambda t: logger.critical(f"SENTINEL TASK DIED: {t.exception()}") if not t.cancelled() and t.exception() else None
    )

    # Start Self-Healing Monitor
    from trading_bot.self_healing import SelfHealingMonitor
    healer = SelfHealingMonitor(config)
    healing_task = asyncio.create_task(healer.run())

    try:
        while True:
            try:
                now_utc = datetime.now(pytz.UTC)
                next_run_time, next_task = get_next_task(now_utc, task_list)
                wait_seconds = (next_run_time - now_utc).total_seconds()

                logger.info(f"Next task '{next_task.id}' ({next_task.label}) scheduled for "
                            f"{next_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                            f"Waiting for {wait_seconds / 3600:.2f} hours.")

                await asyncio.sleep(wait_seconds)

                logger.info(f"--- Running scheduled task: {next_task.id} [{next_task.func_name}] ---")

                # Set global cooldown during scheduled cycle (e.g. 10 mins)
                # This prevents Sentinels from firing Emergency Cycles while we are busy
                GLOBAL_DEDUPLICATOR.set_cooldown("GLOBAL", 600)

                try:
                    await next_task.function(config)
                    record_task_completion(next_task.id)
                finally:
                    # Clear cooldown immediately after task finishes
                    GLOBAL_DEDUPLICATOR.clear_cooldown("GLOBAL")

            except asyncio.CancelledError:
                logger.info("Orchestrator main loop cancelled.")
                break
            except Exception as e:
                error_msg = f"A critical error occurred in the main orchestrator loop: {e}"
                logger.critical(error_msg, exc_info=True)
                await asyncio.sleep(60)
    finally:
        logger.info("Orchestrator shutting down. Ensuring monitor is stopped.")
        healer.stop()
        healing_task.cancel()
        sentinel_task.cancel()

        # Cancel any in-flight fire-and-forget tasks (emergency cycles, audits)
        # before releasing connections they may be using
        if _INFLIGHT_TASKS:
            logger.info(f"Cancelling {len(_INFLIGHT_TASKS)} in-flight tasks...")
            for t in list(_INFLIGHT_TASKS):
                t.cancel()
            # Give tasks a moment to handle CancelledError gracefully
            await asyncio.sleep(1)
            _INFLIGHT_TASKS.clear()

        if monitor_process and monitor_process.returncode is None:
            await stop_monitoring(config)
        await IBConnectionPool.release_all()


async def sequential_main():
    for task in schedule.values():
        await task()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trading Bot Orchestrator")
    parser.add_argument(
        '--commodity', type=str,
        default=os.environ.get("COMMODITY_TICKER", "KC"),
        help="Commodity ticker (e.g. KC, CC). Default: $COMMODITY_TICKER or KC"
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help="Run tasks sequentially (legacy mode)"
    )
    args = parser.parse_args()

    if args.sequential:
        asyncio.run(sequential_main())
    else:
        loop = asyncio.get_event_loop()
        main_task = None
        try:
            main_task = loop.create_task(main(commodity_ticker=args.commodity))
            loop.run_until_complete(main_task)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Orchestrator stopped by user.")
            if main_task:
                main_task.cancel()
                loop.run_until_complete(main_task)
        finally:
            loop.close()
