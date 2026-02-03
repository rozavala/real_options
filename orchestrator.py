"""The main orchestrator for the automated trading bot.

This script serves as the central nervous system of the application. It runs
as a long-lived process, responsible for scheduling and executing all the
different components of the trading pipeline at the correct times.

It now supports an Event-Driven architecture via a Sentinel Loop.
"""

import asyncio
import logging
import sys
import traceback
import os
import random
import json
import hashlib
from collections import deque
import time as time_module
from datetime import datetime, time, timedelta, timezone
import pytz
from ib_insync import IB, util, Contract, MarketOrder, OrderStatus, Future

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
    ORDER_QUEUE,
    get_trade_ledger_df
)
from trading_bot.utils import archive_trade_ledger, configure_market_data_type, is_market_open, is_trading_day
from equity_logger import log_equity_snapshot, sync_equity_from_flex
from trading_bot.sentinels import PriceSentinel, WeatherSentinel, LogisticsSentinel, NewsSentinel, XSentimentSentinel, PredictionMarketSentinel, SentinelTrigger
from trading_bot.microstructure_sentinel import MicrostructureSentinel
from trading_bot.agents import TradingCouncil
from trading_bot.ib_interface import (
    get_active_futures, build_option_chain, create_combo_order_object, get_underlying_iv_metrics,
    place_order, close_spread_with_protection_cleanup
)
from trading_bot.strategy import define_directional_strategy, define_volatility_strategy
from trading_bot.state_manager import StateManager
from trading_bot.connection_pool import IBConnectionPool
from trading_bot.compliance import ComplianceGuardian
from trading_bot.position_sizer import DynamicPositionSizer
from trading_bot.weighted_voting import RegimeDetector
from trading_bot.brier_scoring import get_brier_tracker
from trading_bot.tms import TransactiveMemory
from trading_bot.budget_guard import BudgetGuard
from trading_bot.drawdown_circuit_breaker import DrawdownGuard
from trading_bot.cycle_id import generate_cycle_id

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("Orchestrator")

# --- Global Process Handle for the monitor ---
monitor_process = None
GLOBAL_BUDGET_GUARD = None
GLOBAL_DRAWDOWN_GUARD = None

# Module-level shutdown state
_SYSTEM_SHUTDOWN = False

def is_system_shutdown() -> bool:
    """Check if the system has entered end-of-day shutdown."""
    return _SYSTEM_SHUTDOWN

class TriggerDeduplicator:
    def __init__(self, window_seconds: int = 7200, state_file="data/deduplicator_state.json"):
        self.state_file = state_file
        self.window = window_seconds
        self.cooldowns = {} # Dictionary of cooldowns {source: until_timestamp}
        self.recent_triggers = deque(maxlen=50)
        self.metrics = {
            'total_triggers': 0,
            'filtered_global_cooldown': 0,
            'filtered_post_cycle': 0,
            'filtered_source_cooldown': 0,
            'filtered_duplicate_content': 0,
            'processed': 0
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
                        self.metrics.update(data['metrics'])
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
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
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
            critical_threshold = 9
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
        trigger_hash = hashlib.md5(
            f"{trigger.reason[:50]}{json.dumps(trigger.payload, sort_keys=True)}".encode()
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

# Global Deduplicator Instance
GLOBAL_DEDUPLICATOR = TriggerDeduplicator()

# Concurrent Cycle Lock (Global)
EMERGENCY_LOCK = asyncio.Lock()


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
        confidence = report.get('confidence', 0.5)

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
            reason = f"Emergency Iron Condor: Range-bound + expensive vol (sell premium)"
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
                if regime == "HIGH_VOLATILITY": return "HIGH_VOL"
                return regime
        except Exception as e:
            logger.warning(f"Regime detection via IB failed: {e}")

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


async def _get_current_regime(ib: IB, config: dict) -> str:
    """Estimates the current market regime based on simple VIX/Price metrics."""
    # Simplified placeholder logic - ideally fetch from Volatility Agent or State
    # Here we check IV Rank of the front month
    try:
        futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=1)
        if futures:
            metrics = await get_underlying_iv_metrics(ib, futures[0])
            iv_rank = metrics.get('iv_rank', 50)
            if isinstance(iv_rank, str): iv_rank = 50 # Fallback
            if iv_rank > 70:
                return 'HIGH_VOLATILITY'
            elif iv_rank < 30:
                return 'RANGE_BOUND'
    except Exception:
        pass
    return 'TRENDING' # Default

async def _get_current_price(ib: IB, contract: Contract) -> float:
    """Fetches current price snapshot."""
    ticker = ib.reqMktData(contract, '', True, False)
    await asyncio.sleep(1)
    price = 0.0
    if not util.isNan(ticker.last): price = ticker.last
    elif not util.isNan(ticker.close): price = ticker.close
    # For snapshot requests, IB auto-cleans the ticker.
    # Do NOT call cancelMktData — it causes Error 300 spam.
    return price

async def _get_context_for_guardian(guardian: str, config: dict) -> str:
    """Fetches relevant context (news/weather) for a specific guardian."""
    # Simplified: Returns recent sentinel alerts or agent memory
    # In a full impl, query the specific agent's memory
    tms = TransactiveMemory()
    return f"Context for {guardian}: " + str(tms.retrieve(f"{guardian} context", n_results=3))

async def _validate_thesis(
    thesis: dict,
    position,
    council,
    config: dict,
    ib: IB,
    active_futures_cache: dict = None
) -> dict:
    """
    Validates if a trade thesis still holds given current market conditions.

    Uses the Permabear/Permabull debate structure to stress-test the position.

    Returns:
        dict with 'action' ('HOLD', 'CLOSE', 'PRESS') and 'reason'
    """
    strategy_type = thesis.get('strategy_type', 'UNKNOWN')
    guardian = thesis.get('guardian_agent', 'Master')

    # A. REGIME-BASED VALIDATION (Iron Condor / Long Straddle)
    if strategy_type == 'IRON_CONDOR':
        # Get current regime
        current_regime = await _get_current_regime(ib, config)
        entry_regime = thesis.get('entry_regime', 'RANGE_BOUND')

        if current_regime == 'HIGH_VOLATILITY' and entry_regime == 'RANGE_BOUND':
            return {
                'action': 'CLOSE',
                'reason': f"REGIME BREACH: Entered as RANGE_BOUND, now HIGH_VOLATILITY. Iron Condor invalid."
            }

        # === PRICE BREACH CHECK ===
        # FIX: Get underlying future price, NOT combo contract price
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
                await ib.qualifyContractsAsync(underlying_contract)
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
                    futures = await get_active_futures(ib, symbol, config['exchange'], count=1)
                    underlying_contract = futures[0] if futures else None
                    if active_futures_cache is not None and futures:
                         active_futures_cache[symbol] = futures
                except Exception as e:
                    logger.warning(f"Failed to get active futures for IC validation: {e}")

        if not underlying_contract:
            logger.error(f"Cannot validate IC thesis - no underlying contract available")
            return {'action': 'HOLD', 'reason': 'Unable to fetch underlying price for validation'}

        # Fetch current underlying price
        current_price = await _get_current_price(ib, underlying_contract)
        entry_price = supporting_data.get('entry_price', 0)

        # SANITY CHECK: Detect legacy theses where entry_price is a spread premium
        price_floor = config.get('validation', {}).get('underlying_price_floor', 100.0)

        if 0 < entry_price < price_floor:
            spread_credit = supporting_data.get('spread_credit', entry_price)
            logger.warning(
                f"SEMANTIC GUARD: entry_price ${entry_price:.2f} < floor "
                f"${price_floor:.2f} for thesis {thesis.get('trade_id', '?')}. "
                f"Likely a spread premium, not underlying price. "
                f"Skipping price breach check. Run migration script."
            )
            entry_price = 0  # Disable price breach check for this thesis

        if entry_price > 0 and current_price > 0:
            move_pct = abs((current_price - entry_price) / entry_price) * 100

            if move_pct > 2.0:
                return {
                    'action': 'CLOSE',
                    'reason': f"PRICE BREACH: Underlying moved {move_pct:.2f}% from entry (threshold: 2%). "
                              f"Entry: ${entry_price:.2f}, Current: ${current_price:.2f}"
                }
        else:
            logger.warning(f"Invalid prices for IC validation: entry={entry_price}, current={current_price}")

    elif strategy_type == 'LONG_STRADDLE':
        # Check theta efficiency
        entry_time_str = thesis.get('entry_timestamp', '')
        if entry_time_str:
            entry_time = datetime.fromisoformat(entry_time_str)
            hours_held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

            # If held > 4 hours with minimal movement, consider closing
            if hours_held > 4:
                entry_price = thesis.get('supporting_data', {}).get('entry_price', 0)
                if entry_price:
                    current_price = await _get_current_price(ib, position.contract)
                    if current_price > 0:
                        move_pct = abs((current_price - entry_price) / entry_price) * 100

                        # Theta hurdle: should have moved at least 1% to justify theta burn
                        if move_pct < 1.0:
                            return {
                                'action': 'CLOSE',
                                'reason': f"THETA BURN: {hours_held:.1f}h elapsed, only {move_pct:.2f}% move. Salvage residual value."
                            }

    # B. NARRATIVE-BASED VALIDATION (Directional Spreads)
    elif strategy_type in ['BULL_CALL_SPREAD', 'BEAR_PUT_SPREAD']:
        # Query the guardian agent for thesis validity
        primary_rationale = thesis.get('primary_rationale', '')
        invalidation_triggers = thesis.get('invalidation_triggers', [])

        # Get current sentinel data relevant to this thesis
        current_context = await _get_context_for_guardian(guardian, config)

        # Permabear Attack prompt
        attack_prompt = f"""You are stress-testing an existing position.

POSITION: {strategy_type}
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
                if verdict_data.get('verdict') == 'CLOSE' and verdict_data.get('confidence', 0) > 0.6:
                    return {
                        'action': 'CLOSE',
                        'reason': f"NARRATIVE INVALIDATION: {verdict_data.get('reasoning', 'Thesis degraded')}"
                    }
            except Exception as e:
                logger.warning(f"Could not parse thesis validation response: {e}")

    # Default: HOLD
    return {'action': 'HOLD', 'reason': 'Thesis intact'}

def _find_position_id_for_contract(position, trade_ledger) -> str | None:
    """Maps a live position contract to a position_id from the ledger."""
    symbol = position.contract.localSymbol
    position_direction = 'BUY' if position.position > 0 else 'SELL'

    matches = trade_ledger[trade_ledger['local_symbol'] == symbol].copy()

    if matches.empty:
        logger.debug(f"No ledger entries found for symbol {symbol}")
        return None

    direction_matches = matches[matches['action'] == position_direction]
    if direction_matches.empty:
        direction_matches = matches

    # Find OPEN positions (net quantity != 0)
    open_positions = []
    for pos_id in direction_matches['position_id'].unique():
        pos_entries = trade_ledger[trade_ledger['position_id'] == pos_id]
        net_qty = 0
        for _, row in pos_entries.iterrows():
            net_qty += row['quantity'] if row['action'] == 'BUY' else -row['quantity']

        if net_qty != 0:
            entry_time = pos_entries['timestamp'].min()
            open_positions.append((pos_id, entry_time, net_qty))

    if open_positions:
        # Match by direction sign (FIFO)
        matching_sign = [p for p in open_positions
                        if (p[2] > 0 and position.position > 0) or (p[2] < 0 and position.position < 0)]
        if matching_sign:
            matching_sign.sort(key=lambda x: x[1])
            return matching_sign[0][0]
        open_positions.sort(key=lambda x: x[1])
        return open_positions[0][0]

    logger.warning(f"No open position found for {symbol}, using most recent")
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

        position_id = _find_position_id_for_contract(pos, trade_ledger)
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

    # --- Step 1: Qualify all contracts ---
    qualified_legs = []
    for leg in legs:
        contract = leg.contract
        if not contract.exchange:
            try:
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified:
                    # Create a new leg-like object with qualified contract
                    qualified_legs.append(type(leg)(
                        account=leg.account,
                        contract=qualified[0],
                        position=leg.position,
                        avgCost=leg.avgCost
                    ))
                    logger.debug(
                        f"Qualified {contract.localSymbol}: "
                        f"exchange={qualified[0].exchange}"
                    )
                else:
                    logger.error(
                        f"Contract qualification returned empty for "
                        f"{contract.localSymbol} — skipping leg"
                    )
                    qualified_legs.append(leg)  # Try anyway
            except Exception as e:
                logger.error(
                    f"Contract qualification failed for "
                    f"{contract.localSymbol}: {e}"
                )
                qualified_legs.append(leg)  # Try anyway
        else:
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
    fail_count = len(failed_closes)

    if success_count == total_legs:
        # Full success
        title = f"✅ Position Closed: {position_id}"
        message = (
            f"Reason: {reason}\n"
            f"Closed {success_count}/{total_legs} legs successfully.\n"
        )
        for sc in successful_closes:
            message += f"  {sc['action']} {sc['qty']}x {sc['symbol']} @ ${sc['fill_price']:.4f}\n"
    elif success_count > 0:
        # Partial success — DANGEROUS, position is now unbalanced
        title = f"⚠️ PARTIAL CLOSE: {position_id}"
        message = (
            f"Reason: {reason}\n"
            f"⚠️ ONLY {success_count}/{total_legs} legs closed!\n"
            f"MANUAL INTERVENTION REQUIRED — position may have naked exposure.\n\n"
            f"Successful:\n"
        )
        for sc in successful_closes:
            message += f"  ✅ {sc['action']} {sc['qty']}x {sc['symbol']} @ ${sc['fill_price']:.4f}\n"
        message += f"\nFailed:\n"
        for fc in failed_closes:
            message += f"  ❌ {fc['symbol']}: {fc['status']} — {fc['error']}\n"
    else:
        # Total failure — DO NOT invalidate thesis
        title = f"❌ CLOSE FAILED: {position_id}"
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


async def run_position_audit_cycle(config: dict, trigger_source: str = "Scheduled"):
    """
    Reviews all active positions against their original theses.
    Now operates on GROUPED positions (spread-aware).
    """
    logger.info(f"--- POSITION AUDIT CYCLE ({trigger_source}) ---")

    if GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit:
        logger.warning("Budget hit — skipping Position Audit (LLM disabled)")
        return

    try:
        ib = await IBConnectionPool.get_connection("audit", config)
        configure_market_data_type(ib)

        # 1. Get current positions from IB
        live_positions = await ib.reqPositionsAsync()
        if not live_positions or all(p.position == 0 for p in live_positions):
            logger.info("No open positions to audit.")
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
            futures = await get_active_futures(ib, symbol, exchange, count=5)
            if futures:
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
                active_futures_cache=active_futures_cache  # Issue 9 fix
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

        # 9. Summary notification
        if positions_to_close:
            summary = (
                f"Closed {len(positions_to_close)} positions "
                f"via thesis invalidation:\n"
            )
            summary += "\n".join([
                f"- {p['position_id']} ({len(p['legs'])} legs): {p['reason']}"
                for p in positions_to_close
            ])
            send_pushover_notification(
                config.get('notifications', {}),
                "Position Audit Complete",
                summary
            )
        else:
            logger.info("Position audit complete. All theses remain valid.")

    except Exception as e:
        logger.error(f"Position Audit Cycle failed: {e}\n{traceback.format_exc()}")


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
    if not is_market_open():
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

        send_pushover_notification(config.get('notifications', {}), "Orchestrator", "Started position monitoring service.")
    except Exception as e:
        logger.critical(f"Failed to start position monitor: {e}\n{traceback.format_exc()}")
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
        send_pushover_notification(config.get('notifications', {}), "Orchestrator", "Stopped position monitoring service.")
        monitor_process = None
    except ProcessLookupError:
        logger.warning("Process already terminated.")
    except Exception as e:
        logger.critical(f"An error occurred while stopping the monitor: {e}\n{traceback.format_exc()}")


async def cancel_and_stop_monitoring(config: dict):
    """Wrapper task to cancel open orders and then stop the monitor."""
    global _SYSTEM_SHUTDOWN

    logger.info("--- Initiating end-of-day shutdown sequence ---")
    _SYSTEM_SHUTDOWN = True  # Set BEFORE canceling orders
    logger.info("System shutdown flag SET — no new trades or emergency cycles will be processed")

    await cancel_all_open_orders(config)
    await stop_monitoring(config)
    logger.info("--- End-of-day shutdown sequence complete ---")


def get_next_task(now_utc: datetime, schedule: dict) -> tuple[datetime, callable]:
    """Calculates the next scheduled task.

    The schedule keys are in NY Local Time.
    We calculate the corresponding UTC run time dynamically to handle DST.
    Automatically skips weekends (Saturday/Sunday).
    """
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

    # Sort by time
    sorted_times = sorted(schedule.keys(), key=lambda t: (t.hour, t.minute))

    for rt in sorted_times:
        # Construct run time in NY for today
        try:
            candidate_ny = now_ny.replace(hour=rt.hour, minute=rt.minute, second=0, microsecond=0)
        except ValueError:
            # Handle rare edge cases like 2:30 AM on DST change day (doesn't exist)
            # by normalizing via adding 0 timedelta (or just skip)
            # But simple replace usually works or raises.
            # Ideally we use localize, but replace is robust enough for standard trading hours.
            continue

        # If this time has passed in NY, move to tomorrow
        if candidate_ny <= now_ny:
             candidate_ny += timedelta(days=1)

        # === CHECK IF CANDIDATE IS ON WEEKEND ===
        # After potentially moving to tomorrow, check if we landed on a weekend
        if candidate_ny.weekday() == 5:  # Saturday -> Move to Monday
            candidate_ny += timedelta(days=2)
            # logger.info(f"Task scheduled for Saturday moved to Monday {candidate_ny.strftime('%Y-%m-%d')}")
        elif candidate_ny.weekday() == 6: # Sunday -> Move to Monday
            candidate_ny += timedelta(days=1)
            # logger.info(f"Task scheduled for Sunday moved to Monday {candidate_ny.strftime('%Y-%m-%d')}")

        # Convert to UTC
        candidate_utc = candidate_ny.astimezone(utc)

        if next_run_utc is None or candidate_utc < next_run_utc:
            next_run_utc = candidate_utc
            next_task = schedule[rt]

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
        logger.critical(f"An error occurred during the analysis and archiving process: {e}\n{traceback.format_exc()}")


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
        logger.critical(f"An error occurred during trade reconciliation: {e}\n{traceback.format_exc()}")


async def reconcile_and_analyze(config: dict):
    """Runs reconciliation, then analysis and archiving."""
    logger.info("--- Kicking off end-of-day reconciliation and analysis process ---")

    await sync_equity_from_flex(config)

    # Check equity staleness
    try:
        equity_file = "data/daily_equity.csv"
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
    reconciliation_succeeded = False
    try:
        await reconcile_and_notify(config)
        reconciliation_succeeded = True
    except Exception as e:
        logger.critical(f"Reconciliation FAILED: {e}\n{traceback.format_exc()}")
        send_pushover_notification(
            config.get('notifications', {}),
            "🚨 Reconciliation Failed",
            f"Council history reconciliation failed: {str(e)[:200]}\n"
            f"Brier scoring will be stale until this is fixed."
        )

    await analyze_and_archive(config)

    # Feedback loop health check — report reconciliation status too
    await _check_feedback_loop_health(config)

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
        structured_file = "data/agent_accuracy_structured.csv"
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
                logger.warning("Feedback loop health: All rows had corrupted timestamps!")
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
            if age_hours > 48:
                alert_msg = (
                    f"⚠️ FEEDBACK LOOP STALE\n"
                    f"Oldest PENDING: {age_hours:.0f}h ago\n"
                    f"PENDING: {pending_count}/{total_count}\n"
                    f"Resolution rate: {resolution_rate:.0f}%\n"
                    f"Agent learning is NOT occurring!"
                )
                logger.warning(alert_msg)
                send_pushover_notification(
                    config.get('notifications', {}),
                    "🔴 Feedback Loop Alert",
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
            f"Feedback loop health check failed (non-fatal): {e}\n"
            f"The orchestrator will continue operating.\n"
            f"{traceback.format_exc()}"
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
    if not is_market_open():
        logger.info("Market is closed. Skipping deferred trigger processing.")
        return

    logger.info("--- Processing Deferred Triggers ---")
    try:
        deferred = StateManager.get_deferred_triggers()
        if not deferred:
            logger.info("No deferred triggers to process.")
            return

        logger.info(f"Processing {len(deferred)} deferred triggers from overnight")
        ib_conn = await IBConnectionPool.get_connection("deferred", config)

        processed_count = 0
        skipped_count = 0

        for t in deferred:
            trigger = SentinelTrigger(t['source'], t['reason'], t['payload'])

            # === CRITICAL FIX: Check deduplicator before each cycle ===
            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                await run_emergency_cycle(trigger, config, ib_conn)
                processed_count += 1
                # Note: run_emergency_cycle sets POST_CYCLE debounce internally
            else:
                logger.info(f"Skipping deferred trigger (deduplicated): {trigger.source}")
                skipped_count += 1

        logger.info(f"Deferred triggers complete: {processed_count} processed, {skipped_count} skipped")

    except Exception as e:
        logger.error(f"Failed to process deferred triggers: {e}")


# --- SENTINEL LOGIC ---

async def _is_signal_priced_in(trigger: SentinelTrigger, ml_signal: dict, ib: IB, contract) -> tuple[bool, str]:
    """Check if the signal has already been priced into the market."""
    try:
        # Get 24h price change
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        if len(bars) >= 2:
            prev_close = bars[-2].close
            current_close = bars[-1].close
            change_pct = ((current_close - prev_close) / prev_close) * 100

            # PRICED IN: If bullish trigger + price already up >3%
            if trigger.source in ['WeatherSentinel', 'NewsSentinel']:
                if change_pct > 3.0:
                    return True, f"Price already +{change_pct:.1f}% - signal likely priced in"
                elif change_pct < -3.0:
                    # Note: Original requirement said "If '24h Change' is UP >3% and agents report Bullish news"
                    # We implement a symmetric check for now (extreme moves).
                    return True, f"Price already {change_pct:.1f}% - signal likely priced in"
        return False, ""
    except Exception as e:
        logger.warning(f"Priced-in check failed: {e}")
        return False, ""  # Fail open


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
        return

    # === NEW: MARKET HOURS GATE ===
    if not is_market_open():
        logger.info(f"Market closed. Queuing {trigger.source} alert for next session.")
        StateManager.queue_deferred_trigger(trigger)
        return

    # === DAILY TRADING CUTOFF GATE ===
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)

    # Read from config — default to 10:45
    trading_cutoff = config.get('schedule', {}).get('daily_trading_cutoff_et', {'hour': 10, 'minute': 45})
    cutoff_hour = trading_cutoff.get('hour', 10)
    cutoff_minute = trading_cutoff.get('minute', 45)

    if now_ny.hour > cutoff_hour or (now_ny.hour == cutoff_hour and now_ny.minute >= cutoff_minute):
        logger.info(
            f"Emergency cycle BLOCKED (daily trading cutoff {cutoff_hour}:{cutoff_minute:02d} ET): "
            f"{trigger.source} — deferring to next session"
        )
        StateManager.queue_deferred_trigger(trigger)
        return

    # === NEW: Log Trigger for Fallback ===
    StateManager.log_sentinel_event(trigger)

    # Acquire Lock to prevent race conditions
    if EMERGENCY_LOCK.locked():
        logger.warning(f"Emergency cycle for {trigger.source} queued (Lock active).")

    async with EMERGENCY_LOCK:
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
                StateManager.queue_deferred_trigger(trigger)
                return

            # Check Budget
            if GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit:
                logger.warning("Budget hit — skipping Emergency Cycle (Sentinel-only mode)")
                send_pushover_notification(config.get('notifications', {}), "Budget Guard",
                    f"Daily API budget hit. Sentinel-only mode active.")
                return

            # === Drawdown Circuit Breaker ===
            if GLOBAL_DRAWDOWN_GUARD:
                # Update P&L and Check
                status = await GLOBAL_DRAWDOWN_GUARD.update_pnl(ib)
                if not GLOBAL_DRAWDOWN_GUARD.is_entry_allowed():
                    logger.warning(f"Drawdown Circuit Breaker ACTIVE ({status}) - Skipping Emergency Cycle")
                    return

                if GLOBAL_DRAWDOWN_GUARD.should_panic_close():
                     logger.critical(f"Drawdown PANIC triggered during emergency cycle check. Triggering Hard Close.")
                     await emergency_hard_close(config)
                     return

            # === Generate Cycle ID for prediction tracking ===
            active_ticker = config.get('commodity', {}).get('ticker', config.get('symbol', 'KC'))
            cycle_id = generate_cycle_id(active_ticker)
            logger.info(f"🚨 EMERGENCY CYCLE TRIGGERED by {trigger.source}: {trigger.reason} (Cycle: {cycle_id})")
            send_pushover_notification(config.get('notifications', {}), f"Sentinel Trigger: {trigger.source}", trigger.reason)

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
                        live_positions = await ib.reqPositionsAsync()
                        trade_ledger = get_trade_ledger_df()

                        for thesis in affected_theses:
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
                                # Find the matching live position
                                target_pos = None
                                thesis_id = thesis.get('trade_id')

                                for pos in live_positions:
                                    if pos.position == 0: continue
                                    # Map pos to ID
                                    pos_id = _find_position_id_for_contract(pos, trade_ledger)
                                    if pos_id == thesis_id:
                                        target_pos = pos
                                        break

                                if target_pos:
                                    await _close_position_with_thesis_reason(
                                        ib=ib,
                                        position=target_pos,
                                        position_id=thesis_id,
                                        reason=f"Sentinel Invalidation: {trigger.reason}",
                                        config=config,
                                        thesis=thesis
                                    )
                                else:
                                    logger.warning(f"Could not find live position for invalidated thesis {thesis_id}. Marking thesis as invalid anyway.")

                                tms.invalidate_thesis(thesis_id, f"Sentinel: {trigger.source}")

            try:
                # 1. Initialize Council
                council = TradingCouncil(config)

                # 2. Get Active Futures (We need a target contract)
                # For simplicity, target the Front Month or the one from the trigger payload
                contract_name_hint = trigger.payload.get('contract')

                active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=2)
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
                priced_in, reason = await _is_signal_priced_in(trigger, {}, ib, target_contract)
                if priced_in:
                    logger.warning(f"PRICED IN CHECK FAILED: {reason}. Skipping emergency cycle.")
                    send_pushover_notification(config.get('notifications', {}), "Signal Priced In", reason)
                    return

                # 3. Get Market Context (Snapshot)
                ticker = ib.reqMktData(target_contract, '', True, False)
                await asyncio.sleep(2)

                # Fetch IV Metrics
                iv_metrics = await get_underlying_iv_metrics(ib, target_contract)

                market_context_str = (
                    f"Contract: {target_contract.localSymbol}\n"
                    f"Current Price: {ticker.last if ticker.last else 'N/A'}\n"
                    f"--- VOLATILITY METRICS (IBKR Live) ---\n"
                    f"Current IV: {iv_metrics['current_iv']}\n"
                    f"IV Rank: {iv_metrics['iv_rank']}\n"
                    f"IV Percentile: {iv_metrics['iv_percentile']}\n"
                    f"Note: If IV data shows N/A, analyst should search Barchart for KC IV Rank.\n"
                )

                # 4. Build live market context from IBKR (replaces cached ML signal)
                from trading_bot.market_data_provider import build_market_context
                ml_signal = await build_market_context(ib, target_contract, config)
                ml_signal['reason'] = f"Emergency Cycle triggered by {trigger.source}"
                logger.info(f"Emergency market context: price={ml_signal.get('price')}, regime={ml_signal.get('regime')}")

                # 5. Run Specialized Cycle
                decision = await council.run_specialized_cycle(
                    trigger,
                    contract_name,
                    ml_signal,
                    market_context_str,
                    ib=ib,
                    target_contract=target_contract,
                    cycle_id=cycle_id # Pass cycle_id for logging if supported
                )

                logger.info(f"Emergency Decision: {decision.get('direction')} ({decision.get('confidence')})")

                # === v7.1: Strategy Routing (aligns emergency with v7.0 Judge & Jury) ===
                # Previously: hardcoded DIRECTIONAL. Now: respects thesis, vol, regime.
                current_reports = StateManager.load_state()
                routed = _route_emergency_strategy(
                    decision=decision,
                    market_context=ml_signal,  # Contains regime, price, etc.
                    agent_reports=current_reports,
                    config=config
                )

                # === Log Emergency Decision to History ===
                # Reconstruct full log entry for history
                try:
                    # Reload reports to get sentiments
                    final_reports = StateManager.load_state()
                    agent_data = {}

                    def extract_sentiment(text):
                        if not text or not isinstance(text, str): return "N/A"
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
                        "entry_price": ml_signal.get('price'),

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
                        sma_200=None,
                        confidence=council_log_entry.get('master_confidence'),
                        regime='UNKNOWN',
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

                if is_actionable:
                    logger.info(f"Emergency Cycle ACTION: {direction} ({pred_type})")

                    # === NEW: Compliance Audit ===
                    compliance = ComplianceGuardian(config)
                    # Load reports for audit (re-use final_reports from agents logic if possible, but here we only have decision)
                    # Ideally, run_specialized_cycle should return the full packet.
                    # For now, we reload state which is "close enough" as it was just updated.
                    current_reports = StateManager.load_state()

                    audit = await compliance.audit_decision(
                        current_reports,
                        market_context_str,
                        decision,
                        council.personas.get('master', ''),
                        ib=ib
                    )

                    if not audit.get('approved', True):
                        logger.warning(f"COMPLIANCE BLOCKED Emergency Trade: {audit.get('flagged_reason')}")
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "Emergency Trade VETOED by Compliance",
                            f"Reason: {audit.get('flagged_reason')}"
                        )
                        return

                    logger.info("Decision is actionable. Generating order...")

                    # === NEW: Dynamic Position Sizing ===
                    sizer = DynamicPositionSizer(config)

                    # Get account value
                    account_summary = await ib.accountSummaryAsync()
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
                    chain = await build_option_chain(ib, target_contract)
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
                            predicted_confidence=float(confidence),
                            cycle_id=cycle_id,
                            regime=current_regime,
                            contract=target_contract.lastTradeDateOrContractMonth[:6] if target_contract else "",
                        )

                except Exception as e:
                    logger.error(f"Brier recording failed: {e}")

                if not is_actionable:
                    logger.info(f"Emergency Cycle concluded with no action: {direction} ({pred_type})")

            except Exception as e:
                logger.error(f"Emergency Cycle Failed: {e}\n{traceback.format_exc()}")

        finally:
            # Global post-cycle debounce (blocks ALL sentinels)
            debounce_seconds = config.get('sentinels', {}).get('post_cycle_debounce_seconds', 1800)
            GLOBAL_DEDUPLICATOR.set_cooldown("POST_CYCLE", debounce_seconds)
            logger.info(f"Post-cycle debounce set for {debounce_seconds}s")


def validate_trigger(trigger):
    """Defensive check for sentinel triggers."""
    if isinstance(trigger, list):
        logger.warning(f"Sentinel returned list instead of single trigger. Using first item.")
        trigger = trigger[0] if trigger else None

    if trigger is not None and not hasattr(trigger, 'source'):
        logger.error(f"Invalid trigger object (missing 'source'): {type(trigger)}")
        return None
    return trigger


async def run_sentinels(config: dict):
    """
    Main loop for Sentinels. Runs concurrently with the scheduler.

    ARCHITECTURE (per Mission Control):
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

    # Contract Cache
    cached_contract = None
    last_contract_refresh = 0
    CONTRACT_REFRESH_INTERVAL = 14400  # 4 hours

    # Outage Tracking (only relevant during market hours)
    last_successful_ib_time = None  # Will be set on first successful connection
    outage_notification_sent = False
    OUTAGE_THRESHOLD_SECONDS = 600  # 10 minutes

    while True:
        try:
            now = time_module.time()
            market_open = is_market_open()
            trading_day = is_trading_day()

            # === 2. MARKET HOURS GATE: Only connect when market is OPEN ===
            should_connect = market_open  # NOT trading_day - must be is_market_open()

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
                    logger.info("Market Closed: Disconnecting Sentinel IB to prevent zombie state.")
                    sentinel_ib.disconnect()
                    # === NEW: Give Gateway time to cleanup ===
                    await asyncio.sleep(3.0)
                    sentinel_ib = None
                    price_sentinel.ib = None

                    # Reset outage tracking (not relevant when market is closed)
                    last_successful_ib_time = None
                    outage_notification_sent = False

            # === CONTRACT CACHE REFRESH ===
            if sentinel_ib and sentinel_ib.isConnected() and (now - last_contract_refresh > CONTRACT_REFRESH_INTERVAL):
                try:
                    active_futures = await get_active_futures(sentinel_ib, config['symbol'], config['exchange'], count=1)
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

                    target = cached_contract
                    if not target and sentinel_ib.isConnected():
                        active_futures = await get_active_futures(sentinel_ib, config['symbol'], config['exchange'], count=1)
                        if active_futures:
                            target = active_futures[0]

                    if target:
                        await micro_sentinel.subscribe_contract(target)
                    else:
                        logger.warning("No active futures found for Microstructure Sentinel")

                except Exception as e:
                    logger.error(f"Failed to engage MicrostructureSentinel: {e}")
                    micro_sentinel = None

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

            # === RUN SENTINELS ===

            # 1. Price Sentinel (Every 1 min) - ONLY if IB connected
            if sentinel_ib and sentinel_ib.isConnected():
                try:
                    trigger = await price_sentinel.check(cached_contract=cached_contract)
                    trigger = validate_trigger(trigger)

                    # === NEW: Price Move Triggers Position Audit ===
                    # If PriceSentinel detects significant move, proactively check theses
                    if trigger and trigger.source == 'PriceSentinel':
                        price_change = abs(trigger.payload.get('change', 0))
                        if price_change >= 1.5:  # Pre-emptive at 1.5% (before 2% breach)
                            logger.info(f"PriceSentinel detected {price_change:.1f}% move - triggering position audit")
                            asyncio.create_task(run_position_audit_cycle(
                                config,
                                f"PriceSentinel trigger ({price_change:.1f}% move)"
                            ))

                    if trigger and GLOBAL_DEDUPLICATOR.should_process(trigger):
                        asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                        GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                except Exception as e:
                    logger.error(f"PriceSentinel check failed: {type(e).__name__}: {e}")

            # 2. Weather Sentinel (Every 4 hours) - Runs 24/7, no IB needed
            if (now - last_weather) > 14400:
                try:
                    trigger = await weather_sentinel.check()
                    trigger = validate_trigger(trigger)
                    if trigger:
                        if market_open and sentinel_ib and sentinel_ib.isConnected():
                            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                        else:
                            # Defer for market open
                            StateManager.queue_deferred_trigger(trigger)
                            logger.info(f"Deferred {trigger.source} trigger for market open")
                except Exception as e:
                    logger.error(f"WeatherSentinel check failed: {type(e).__name__}: {e}")
                finally:
                    last_weather = now

            # 3. Logistics Sentinel (Every 6 hours) - Runs 24/7, no IB needed
            if (now - last_logistics) > 21600:
                try:
                    trigger = await logistics_sentinel.check()
                    trigger = validate_trigger(trigger)
                    if trigger:
                        if market_open and sentinel_ib and sentinel_ib.isConnected():
                            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                        else:
                            StateManager.queue_deferred_trigger(trigger)
                            logger.info(f"Deferred {trigger.source} trigger for market open")
                except Exception as e:
                    logger.error(f"LogisticsSentinel check failed: {type(e).__name__}: {e}")
                finally:
                    last_logistics = now

            # 4. News Sentinel (Every 2 hours) - Runs 24/7, no IB needed
            if (now - last_news) > 7200:
                try:
                    trigger = await news_sentinel.check()
                    trigger = validate_trigger(trigger)
                    if trigger:
                        if market_open and sentinel_ib and sentinel_ib.isConnected():
                            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                        else:
                            StateManager.queue_deferred_trigger(trigger)
                            logger.info(f"Deferred {trigger.source} trigger for market open")
                except Exception as e:
                    logger.error(f"NewsSentinel check failed: {type(e).__name__}: {e}")
                finally:
                    last_news = now

            # 5. X Sentiment Sentinel (Every 90 min during trading day)
            if trading_day and (now - last_x_sentiment) > 5400:
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

                    trigger = await x_sentinel.check()
                    trigger = validate_trigger(trigger)
                    x_sentinel_stats['checks_today'] += 1

                    if trigger:
                        x_sentinel_stats['triggers_today'] += 1
                        if market_open and sentinel_ib and sentinel_ib.isConnected():
                            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 900)
                        else:
                            StateManager.queue_deferred_trigger(trigger)
                            logger.info(f"Deferred {trigger.source} trigger for market open")
                except Exception as e:
                    logger.error(f"XSentimentSentinel check failed: {type(e).__name__}: {e}")
                finally:
                    last_x_sentiment = now

            # 6. Prediction Market Sentinel (Every 5 minutes) - Runs 24/7, no IB needed
            prediction_config = config.get('sentinels', {}).get('prediction_markets', {})
            prediction_interval = prediction_config.get('poll_interval_seconds', 300)

            if (now - last_prediction_market) > prediction_interval:
                try:
                    trigger = await prediction_market_sentinel.check()
                    trigger = validate_trigger(trigger)
                    if trigger:
                        if market_open and sentinel_ib and sentinel_ib.isConnected():
                            if GLOBAL_DEDUPLICATOR.should_process(trigger):
                                asyncio.create_task(run_emergency_cycle(trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(trigger.source, 1800)  # 30 min cooldown
                        else:
                            StateManager.queue_deferred_trigger(trigger)
                            logger.info(f"Deferred {trigger.source} trigger for market open")
                except Exception as e:
                    logger.error(f"PredictionMarketSentinel check failed: {type(e).__name__}: {e}")
                finally:
                    last_prediction_market = now

            # 7. Microstructure Sentinel (Every 1 min with Price Sentinel)
            if micro_sentinel and micro_ib and micro_ib.isConnected():
                try:
                    micro_trigger = await micro_sentinel.check()
                    if micro_trigger:
                        from trading_bot.notifications import get_notification_tier, NotificationTier
                        tier = get_notification_tier(micro_trigger.severity)

                        if tier == NotificationTier.CRITICAL:
                            # Severity 9-10: Full emergency cycle (must still pass deduplicator)
                            logger.warning(f"MICROSTRUCTURE CRITICAL: {micro_trigger.reason}")
                            if GLOBAL_DEDUPLICATOR.should_process(micro_trigger):
                                asyncio.create_task(run_emergency_cycle(micro_trigger, config, sentinel_ib))
                                GLOBAL_DEDUPLICATOR.set_cooldown(micro_trigger.source, 900)
                        elif tier == NotificationTier.PUSHOVER:
                            # Severity 7-8: Log + Pushover but NO emergency cycle
                            # Liquidity depletion is informational, not actionable by council
                            logger.warning(f"MICROSTRUCTURE ALERT: {micro_trigger.reason}")
                            if GLOBAL_DEDUPLICATOR.should_process(micro_trigger):
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
                except Exception as e:
                    logger.error(f"MicrostructureSentinel check failed: {type(e).__name__}: {e}")

            await asyncio.sleep(60)  # Loop tick

        except asyncio.CancelledError:
            logger.info("Sentinel Loop Cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in Sentinel Loop: {e}")
            await asyncio.sleep(60)

    # IBConnectionPool manages disconnects, so explicit disconnect is not strictly needed,
    # but good practice if task is cancelled.
    # However, pool is shared (conceptually not shared here but...)
    # We leave it connected.


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
    logger.info("--- Emergency Hard Close Check (T-45 min) ---")

    try:
        ib = await IBConnectionPool.get_connection("orchestrator_orders", config)
        configure_market_data_type(ib)

        live_positions = await ib.reqPositionsAsync()
        open_positions = [p for p in live_positions if p.position != 0]

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

                contract = pos.contract
                qualified = await ib.qualifyContractsAsync(contract)
                if not qualified:
                    logger.error(f"Could not qualify {contract.localSymbol}")
                    failed += 1
                    continue

                # Market order with GTC
                from ib_insync import MarketOrder
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

    except Exception as e:
        logger.critical(f"Emergency hard close FAILED entirely: {e}")
        send_pushover_notification(
            config.get('notifications', {}),
            "🚨🚨 EMERGENCY CLOSE FAILED",
            f"Could not execute emergency hard close: {str(e)[:200]}\n"
            f"MANUAL INTERVENTION REQUIRED IMMEDIATELY."
        )

async def close_stale_positions_fallback(config: dict):
    """Fallback close attempt at 12:45 ET. Only acts if 11:00 primary close missed anything."""
    logger.info("--- Fallback Close Attempt (12:45 ET) ---")
    logger.info("This is a retry for any positions the 11:00 primary close failed to handle.")
    await close_stale_positions(config)

async def run_brier_reconciliation(config: dict):
    """Automated Brier prediction reconciliation."""
    try:
        from scripts.fix_brier_data import resolve_with_cycle_aware_match
        resolved = resolve_with_cycle_aware_match(dry_run=False)
        logger.info(f"Brier reconciliation complete: {resolved} predictions resolved")
    except Exception as e:
        logger.error(f"Brier reconciliation failed (non-fatal): {e}")

async def guarded_generate_orders(config: dict):
    """Generate orders with budget and cutoff guards."""
    if GLOBAL_BUDGET_GUARD and GLOBAL_BUDGET_GUARD.is_budget_hit:
        logger.warning("Budget hit - skipping scheduled orders.")
        return

    # === NEW: Shutdown proximity check ===
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(timezone.utc).astimezone(ny_tz)
    shutdown_time = time(13, 45)  # cancel_and_stop_monitoring time
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
    trading_cutoff = config.get('schedule', {}).get('daily_trading_cutoff_et', {'hour': 10, 'minute': 45})
    cutoff_hour = trading_cutoff.get('hour', 10)
    cutoff_minute = trading_cutoff.get('minute', 45)

    if now_ny.hour > cutoff_hour or (now_ny.hour == cutoff_hour and now_ny.minute >= cutoff_minute):
        logger.info(f"Order generation BLOCKED: Past daily cutoff ({cutoff_hour}:{cutoff_minute:02d} ET)")
        return

    # Check Drawdown Guard
    if GLOBAL_DRAWDOWN_GUARD:
        try:
             # Need a connection to check P&L.
             # generate_and_execute_orders creates its own connection.
             # We will rely on order_manager to check again, or check here if possible.
             # Ideally we check here to avoid spinning up the order generation logic.
             ib = await IBConnectionPool.get_connection("drawdown_check", config)
             await GLOBAL_DRAWDOWN_GUARD.update_pnl(ib)
             if not GLOBAL_DRAWDOWN_GUARD.is_entry_allowed():
                 logger.warning("Order generation BLOCKED: Drawdown Guard Active")
                 return
        except Exception as e:
             logger.warning(f"Failed to check drawdown guard before orders: {e}")

    await generate_and_execute_orders(config, shutdown_check=is_system_shutdown)

schedule = {
    time(3, 30): start_monitoring,
    time(3, 31): process_deferred_triggers,
    time(8, 30): run_position_audit_cycle,
    time(9, 0): guarded_generate_orders,
    time(11, 0): close_stale_positions,          # PRIMARY: Peak liquidity
    time(11, 15): run_position_audit_cycle,       # Post-close verification
    time(12, 45): close_stale_positions_fallback, # FALLBACK: Retry failures
    time(13, 0): run_position_audit_cycle,        # Pre-close audit
    time(13, 15): emergency_hard_close,           # SAFETY: Market orders
    time(13, 45): cancel_and_stop_monitoring,
    time(14, 5): log_equity_snapshot,
    time(14, 10): run_brier_reconciliation,       # After equity logging, before end-of-day
    time(14, 15): reconcile_and_analyze,
}

def apply_schedule_offset(original_schedule: dict, offset_minutes: int) -> dict:
    new_schedule = {}
    today = datetime.now(timezone.utc).date()
    for run_time, task_func in original_schedule.items():
        dt_original = datetime.combine(today, run_time)
        dt_shifted = dt_original + timedelta(minutes=offset_minutes)
        new_schedule[dt_shifted.time()] = task_func
    return new_schedule

async def main():
    """The main long-running orchestrator process."""
    logger.info("=============================================")
    logger.info("=== Starting the Trading Bot Orchestrator ===")
    logger.info("=============================================")

    config = load_config()
    if not config:
        logger.critical("Orchestrator cannot start without a valid configuration."); return
    
    # Initialize Budget Guard
    global GLOBAL_BUDGET_GUARD
    GLOBAL_BUDGET_GUARD = BudgetGuard(config)
    logger.info(f"Budget Guard initialized. Daily limit: ${GLOBAL_BUDGET_GUARD.daily_budget}")

    # Initialize Drawdown Guard
    global GLOBAL_DRAWDOWN_GUARD
    GLOBAL_DRAWDOWN_GUARD = DrawdownGuard(config)
    logger.info("Drawdown Guard initialized.")

    # Process deferred triggers from overnight - ONLY if market is open
    if is_market_open():
        await process_deferred_triggers(config)
    else:
        logger.info("Market Closed. Deferred triggers will remain queued.")

    env_name = os.getenv("ENV_NAME", "DEV") 
    is_prod = env_name == "PROD 🚀"

    current_schedule = schedule
    if not is_prod:
        schedule_offset_minutes = -10
        logger.info(f"Environment: {env_name}. Applying {schedule_offset_minutes} minute 'Civil War' avoidance offset.")
        current_schedule = apply_schedule_offset(schedule, offset_minutes=schedule_offset_minutes)
    else:
        logger.info("Environment: PROD 🚀. Using standard master schedule.")

    # Start Sentinels in background
    sentinel_task = asyncio.create_task(run_sentinels(config))

    try:
        while True:
            try:
                now_utc = datetime.now(pytz.UTC)
                next_run_time, next_task_func = get_next_task(now_utc, current_schedule)
                wait_seconds = (next_run_time - now_utc).total_seconds()

                task_name = next_task_func.__name__
                logger.info(f"Next task '{task_name}' scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}. "
                            f"Waiting for {wait_seconds / 3600:.2f} hours.")

                await asyncio.sleep(wait_seconds)

                logger.info(f"--- Running scheduled task: {task_name} ---")

                # Set global cooldown during scheduled cycle (e.g. 10 mins)
                # This prevents Sentinels from firing Emergency Cycles while we are busy
                GLOBAL_DEDUPLICATOR.set_cooldown("GLOBAL", 600)

                try:
                    await next_task_func(config)
                finally:
                    # Clear cooldown immediately after task finishes
                    GLOBAL_DEDUPLICATOR.clear_cooldown("GLOBAL")

            except asyncio.CancelledError:
                logger.info("Orchestrator main loop cancelled."); break
            except Exception as e:
                error_msg = f"A critical error occurred in the main orchestrator loop: {e}\n{traceback.format_exc()}"
                logger.critical(error_msg)
                await asyncio.sleep(60)
    finally:
        logger.info("Orchestrator shutting down. Ensuring monitor is stopped.")
        sentinel_task.cancel()
        if monitor_process and monitor_process.returncode is None:
            await stop_monitoring(config)
        await IBConnectionPool.release_all()


async def sequential_main():
    for task in schedule.values():
        await task()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        asyncio.run(sequential_main())
    else:
        loop = asyncio.get_event_loop()
        main_task = None
        try:
            main_task = loop.create_task(main())
            loop.run_until_complete(main_task)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Orchestrator stopped by user.")
            if main_task:
                main_task.cancel()
                loop.run_until_complete(main_task)
        finally:
            loop.close()
