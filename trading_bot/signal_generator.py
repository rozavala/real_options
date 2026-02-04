
import logging
import asyncio
import traceback
import re
import json
from datetime import datetime, timezone
from ib_insync import IB
from trading_bot.agents import TradingCouncil
from trading_bot.ib_interface import get_active_futures # CORRECTED IMPORT
from trading_bot.market_data_provider import build_all_market_contexts
from trading_bot.utils import (
    log_council_decision,
    get_active_ticker
)
from trading_bot.decision_signals import log_decision_signal
from notifications import send_pushover_notification
from trading_bot.state_manager import StateManager # Import StateManager
from trading_bot.compliance import ComplianceGuardian
from trading_bot.weighted_voting import calculate_weighted_decision, determine_trigger_type, TriggerType, detect_market_regime_simple
from trading_bot.heterogeneous_router import CriticalRPCError
from trading_bot.cycle_id import generate_cycle_id
from trading_bot.brier_bridge import record_agent_prediction
from trading_bot.strategy_router import route_strategy, infer_strategy_type  # NEW

logger = logging.getLogger(__name__)

def _calculate_agent_conflict(agent_data: dict) -> float:
    """
    Returns 0.0-1.0 score indicating how much agents disagree.
    High conflict = potential inflection point = good for straddle.
    """
    sentiments = []
    for key in ['macro_sentiment', 'technical_sentiment', 'geopolitical_sentiment',
                'sentiment_sentiment', 'agronomist_sentiment']:
        s = agent_data.get(key, 'NEUTRAL')
        if s == 'BULLISH':
            sentiments.append(1)
        elif s == 'BEARISH':
            sentiments.append(-1)
        else:
            sentiments.append(0)

    if not sentiments:
        return 0.0

    # High variance = high conflict
    avg = sum(sentiments) / len(sentiments)
    variance = sum((s - avg) ** 2 for s in sentiments) / len(sentiments)
    # Normalize: max variance with [-1, 1] values is 1.0
    return min(1.0, variance)


def _detect_imminent_catalyst(agent_data: dict) -> str | None:
    """
    Scans agent reports for signs of IMMINENT, UNPRICED market-moving events.
    Returns catalyst description if found, None otherwise.

    v7.0: Previous version triggered on generic keywords like "drought" and "rain"
    which appear in EVERY coffee report. Now requires:
    1. Compound keywords (e.g. "frost warning") trigger directly — high confidence
    2. Single keywords (e.g. "drought") require urgency co-occurrence
    3. Resolved markers ("ended", "improving", "priced in") suppress false positives

    This prevents the system from perpetually thinking there's an "imminent catalyst"
    when reports are simply discussing normal weather conditions.
    """
    # Compound keywords — trigger directly (high confidence)
    catalyst_keywords = [
        ('frost warning', 'Active frost warning in growing regions'),
        ('freeze warning', 'Active freeze warning'),
        ('frost risk', 'Elevated frost risk detected'),
        ('conab report', 'CONAB report imminent'),
        ('usda report', 'USDA report pending'),
        ('strike action', 'Labor strike in progress or imminent'),
        ('port closure', 'Port closure affecting exports'),
        ('export ban', 'Export restriction announced'),
    ]

    # Single-word triggers — REQUIRE urgency co-occurrence
    conditional_keywords = [
        ('drought', 'Drought conditions'),
        ('frost', 'Frost conditions'),
        ('freeze', 'Freeze conditions'),
        ('strike', 'Labor disruption'),
    ]

    # Urgency markers — catalyst must be ACTIVE, not historical
    urgency_markers = [
        'imminent', 'warning', 'developing', 'worsening', 'escalating',
        'confirmed', 'breaking', 'emergency', 'unprecedented', 'severe',
        'expected this week', 'forecast for', 'risk elevated',
    ]

    # Resolved markers — catalyst is OVER, not actionable
    resolved_markers = [
        'ended', 'resolved', 'passed', 'recovered', 'abated', 'eased',
        'improving', 'relief', 'normalized', 'subsided', 'historical',
        'priced in', 'already reflected',
    ]

    # Check agronomist and geopolitical summaries
    reports_to_check = [
        agent_data.get('agronomist_summary', ''),
        agent_data.get('geopolitical_summary', ''),
        agent_data.get('sentiment_summary', ''),
    ]

    combined_text = ' '.join(str(r).lower() for r in reports_to_check)

    # Check if text contains resolved markers (reduces false positives)
    has_resolved = any(marker in combined_text for marker in resolved_markers)

    # Phase 1: Check compound keywords (high confidence, don't need urgency)
    for keyword, description in catalyst_keywords:
        if keyword in combined_text and not has_resolved:
            logger.info(f"CATALYST DETECTED (compound): '{keyword}' → {description}")
            return description

    # Phase 2: Check single keywords (require urgency co-occurrence)
    if not has_resolved:
        has_urgency = any(marker in combined_text for marker in urgency_markers)
        if has_urgency:
            for keyword, description in conditional_keywords:
                if keyword in combined_text:
                    logger.info(f"CATALYST DETECTED (conditional+urgent): '{keyword}' → {description}")
                    return f"{description} (urgent)"

    return None

async def generate_signals(ib: IB, config: dict, shutdown_check=None) -> list:
    """
    Generates trading signals via the Council's multi-agent analysis.

    Args:
        ib: Interactive Brokers connection
        config: Trading configuration
        shutdown_check: Optional callable returning True if system is shutting down.
                       Injected by orchestrator to avoid circular import.

    Market data is fetched directly from IBKR for each active contract.
    The Council (8 specialized agents + adversarial debate + weighted voting)
    is the sole decision-maker — no ML prior or anchoring bias.

    Args:
        ib: Connected IB instance
        config: Application config with commodity, strategy, and agent settings

    Returns:
        List of signal dicts, one per active contract
    """
    logger.info("--- Starting Multi-Agent Signal Generation (Council) ---")

    # === SHUTDOWN GATE (R1: Dependency Injection) ===
    if shutdown_check and shutdown_check():
        logger.warning("Signal generation SKIPPED: System in shutdown mode")
        return []

    # 1. Initialize Council & Compliance
    council = None
    compliance = None
    try:
        council = TradingCouncil(config)
        # --- CACHE INVALIDATION (Fix B1) ---
        # Force fresh agent searches for order generation cycles.
        if hasattr(council, 'invalidate_grounded_data_cache'):
            council.invalidate_grounded_data_cache()

        compliance = ComplianceGuardian(config)
    except Exception as e:
        logger.error(f"Failed to initialize Trading Council: {e}. Cannot proceed without Council.")
        send_pushover_notification(
            config.get('notifications', {}),
            "Trading Council Startup Failed",
            f"Error: {e}. No ML fallback — aborting signal generation."
        )
        return []  # No ML fallback — Council is required

    # 2. Get Active Futures
    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    sorted_contracts = sorted(active_futures, key=lambda c: c.lastTradeDateOrContractMonth)

    if not sorted_contracts:
        logger.error("No active futures found. Cannot generate signals.")
        return []

    # 3. Build Market Context for All Contracts (replaces ML inference)
    market_contexts = await build_all_market_contexts(ib, sorted_contracts, config)

    if not market_contexts:
        logger.error("Failed to build market context. Cannot proceed.")
        return []

    logger.info(f"Market context built for {len(market_contexts)} contracts")

    final_signals = []

    # Semaphore to prevent hitting Gemini rate limits (max 3 concurrent contract analysis groups)
    sem = asyncio.Semaphore(3)

    # 4. Define the async processor for a single contract
    async def process_contract(contract, market_ctx):
        contract_name = f"{contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]})"

        # Default values
        final_data = {
            "action": "NEUTRAL",
            "confidence": market_ctx.get('confidence', 0.5),
            "expected_price": market_ctx.get('expected_price'),
            "reason": market_ctx.get('reason', 'N/A')
        }

        # If Council is offline, we already aborted above, but keeping safety check logic if reused
        if not council:
            return {}

        # === Generate Cycle ID for this contract's decision ===
        cycle_id = generate_cycle_id(get_active_ticker(config))

        agent_data = {} # Initialize outside try/except to avoid UnboundLocalError
        regime_for_voting = 'UNKNOWN' # Initialize here to avoid scope issues
        reports = {} # Initialize here to ensure scope for return

        async with sem:
            try:
                logger.info(f"Convening Council for {contract_name} (Cycle: {cycle_id})...")

                # A. Define Research Tasks
                # Note: We use specific keywords to guide the Flash model search
                # We use 'research_topic_with_reflexion' for critical analysts to reduce hallucinations
                tasks = {
                    "agronomist": council.research_topic_with_reflexion("agronomist",
                        f"Search for 'current 10-day weather forecast Minas Gerais coffee zone' and 'NOAA Brazil precipitation anomaly'. Analyze if recent rains are beneficial for flowering or excessive."),
                    "macro": council.research_topic("macro",
                        f"Search for 'USD BRL exchange rate forecast' and 'Brazil Central Bank Selic rate outlook'. Determine if the BRL is trending to encourage farmer selling."),
                    "geopolitical": council.research_topic("geopolitical",
                        f"Search for 'Red Sea shipping coffee delays', 'Brazil port of Santos wait times', and 'EUDR regulation delay latest news'. Determine if there are logistical bottlenecks."),
                    "supply_chain": council.research_topic("supply_chain",
                        f"Search for 'Cecafé Brazil coffee export report latest', 'Global container freight index rates', and 'Green coffee shipping manifest trends'. Analyze flow volume vs port capacity."),
                    "inventory": council.research_topic("inventory",
                        f"Search for 'ICE Arabica Certified Stocks level current' and 'GCA Green Coffee stocks report latest'. Look for 'Backwardation' in the forward curve."),
                    "sentiment": council.research_topic("sentiment",
                        f"Search for 'Coffee COT report non-commercial net length'. Determine if market is overbought."),
                    "technical": council.research_topic_with_reflexion("technical",
                        f"Search for 'Coffee futures technical analysis {contract.localSymbol}' and '{contract.localSymbol} support resistance levels'. "
                        f"Look for 'RSI divergence' or 'Moving Average crossover'. "
                        f"IMPORTANT: You MUST find and explicitly state the current value of the '200-day Simple Moving Average (SMA)'."),

                    # [NEW] Volatility Agent Task
                    "volatility": council.research_topic("volatility",
                        f"Search for 'Coffee Futures Implied Volatility Rank current' and '{contract.localSymbol} option volatility skew'. "
                        f"Determine if option premiums are cheap (Bullish for buying) or expensive (Bearish for buying) relative to historical volatility.")
                }

                # B. Execute Research (Parallel) with Rate Limit Protection
                # Add slight stagger to avoid instantaneous burst
                await asyncio.sleep(0.5)
                research_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

                # Check for CriticalRPCError (Executor Shutdown)
                for res in research_results:
                    if isinstance(res, CriticalRPCError):
                        raise res

                # C. Format Reports
                reports = {}
                for key, res in zip(tasks.keys(), research_results):
                    # === FLIGHT DIRECTOR MANDATE: Type Guard for List Results ===
                    if isinstance(res, list):
                        # === CRITICAL: Log multi-element lists for investigation ===
                        if len(res) > 1:
                            logger.error(
                                f"🚨 MULTI-REPORT ANOMALY: Agent {key} returned {len(res)} results. "
                                f"Only using first element. Full list keys: {[r.get('sentiment', 'N/A') if isinstance(r, dict) else type(r) for r in res]}. "
                                f"INVESTIGATE: LLM may be generating contradictory reports."
                            )
                        elif len(res) == 0:
                            logger.warning(f"Agent {key} returned empty list.")
                            res = {}
                        else:
                            logger.warning(f"Agent {key} returned single-element list instead of dict.")

                        res = res[0] if res else {}

                    if isinstance(res, Exception):
                        logger.error(f"Agent {key} failed for {contract_name}: {res}")
                        reports[key] = "Data Unavailable"
                    else:
                        reports[key] = res

                # --- METRICS LOGGING ---
                successful_agents = sum(1 for r in reports.values() if isinstance(r, dict) and r.get('confidence', 0) != 0.5)
                total_agents = len(reports)
                logger.info(f"Agent research complete for {contract_name}: {successful_agents}/{total_agents} returned non-default results")

                if successful_agents < total_agents * 0.5:
                    logger.warning(f"⚠️ LOW AGENT YIELD for {contract_name}: Only {successful_agents}/{total_agents} agents returned meaningful data")

                # --- DATA FRESHNESS GATE (v2: Two-Tier) ---
                # Tier 1: HARD GATE — Reject if data is catastrophically stale (>24h)
                #   This catches overnight/weekend data that is genuinely dangerous.
                # Tier 2: SOFT PENALTY — Degrade confidence if data is moderately stale (>4h)
                #   This allows trading on calm days when news has not updated.

                HARD_FRESHNESS_LIMIT_HOURS = config.get(
                    'risk_management', {}
                ).get('hard_freshness_limit_hours', 24)

                SOFT_FRESHNESS_LIMIT_HOURS = config.get(
                    'risk_management', {}
                ).get('soft_freshness_limit_hours', 4)

                critically_stale_agents = []
                moderately_stale_agents = []

                for k, v in reports.items():
                    if isinstance(v, dict) and 'data_freshness_hours' in v:
                        freshness = v['data_freshness_hours']
                        if freshness > HARD_FRESHNESS_LIMIT_HOURS:
                            critically_stale_agents.append(k)
                            v['is_stale'] = True
                        elif freshness > SOFT_FRESHNESS_LIMIT_HOURS:
                            moderately_stale_agents.append(k)
                            v['is_stale'] = True  # Mark for weighted voting penalty

                # HARD GATE: Only reject if >50% are CRITICALLY stale (>24h)
                if len(critically_stale_agents) > total_agents * 0.5:
                    logger.warning(
                        f"HARD FRESHNESS GATE: {len(critically_stale_agents)}/{total_agents} "
                        f"agents have critically stale data (>{HARD_FRESHNESS_LIMIT_HOURS}h). "
                        f"Skipping {contract_name}."
                    )
                    return {
                        **market_ctx,
                        **final_data,
                        "contract_month": contract.lastTradeDateOrContractMonth[:6],
                        "prediction_type": "NEUTRAL",
                        "direction": "NEUTRAL",
                        "confidence": 0.0,
                        "reason": (
                            f"Hard Freshness Gate: {len(critically_stale_agents)} agents "
                            f"returned critically stale data (>{HARD_FRESHNESS_LIMIT_HOURS}h)"
                        ),
                        "_agent_reports": reports
                    }

                # SOFT PENALTY: Log warning but proceed with degraded confidence
                if moderately_stale_agents:
                    logger.info(
                        f"SOFT FRESHNESS PENALTY: {len(moderately_stale_agents)}/{total_agents} "
                        f"agents have moderately stale data (>{SOFT_FRESHNESS_LIMIT_HOURS}h). "
                        f"Proceeding with confidence penalty for {contract_name}."
                    )
                    # Stale agents already marked with is_stale=True
                    # The weighted voting system will apply reduced weight to these agents

                # --- SAVE STATE (For Sentinel Context) ---
                # DISABLED: Moved to post-gather consolidation to prevent race condition
                # See Fix #2C in JULES_IMPLEMENTATION_GUIDE.md
                # StateManager.save_state(reports)

                # --- WEIGHTED VOTING (New) ---
                # Determine trigger type from context
                trigger_source = locals().get('trigger_reason', None)  # From sentinel if triggered
                trigger_type = determine_trigger_type(trigger_source)

                # Detect Regime for Voting
                # Use volatility from market context directly
                price_change = 0.0
                if market_ctx and 'volatility_5d' in market_ctx:
                    # Use actual measured volatility instead of ML prediction delta
                    price_change = market_ctx.get('volatility_5d', 0.0) or 0.0

                vol_report = reports.get('volatility', '')
                vol_report_str = vol_report.get('data', vol_report) if isinstance(vol_report, dict) else str(vol_report)
                regime_for_voting = detect_market_regime_simple(vol_report_str, price_change)

                # Calculate weighted consensus
                min_quorum = config.get('strategy', {}).get('min_voter_quorum', 3)
                weighted_result = await calculate_weighted_decision(
                    agent_reports=reports,
                    trigger_type=trigger_type,
                    ml_signal=market_ctx,
                    ib=ib,
                    contract=contract,
                    regime=regime_for_voting,
                    min_quorum=min_quorum
                )

                if weighted_result.get('quorum_failure'):
                    logger.warning(
                        f"Quorum failure for {contract_name}: "
                        f"Only {weighted_result.get('voters_present', [])} voted. "
                        f"Skipping council for this contract."
                    )
                    return {
                        **market_ctx,
                        **final_data,
                        "contract_month": contract.lastTradeDateOrContractMonth[:6],
                        "prediction_type": "NEUTRAL",
                        "direction": "NEUTRAL",
                        "confidence": 0.0,
                        "reason": f"Quorum Failure: Insufficient agent participation (Need {min_quorum})",
                        "_agent_reports": reports
                    }

                # Inject weighted result into market context for Master
                # v7.0: Present vote as INFORMATION, not instruction.
                # The Master should consider this as one data point, not a binding directive.
                weighted_context = (
                    f"\n\n--- AGENT SENTIMENT THERMOMETER (Informational Only) ---\n"
                    f"Agent Consensus: {weighted_result['direction']}\n"
                    f"Consensus Strength: {weighted_result['confidence']:.2f}\n"
                    f"Dominant Voice: {weighted_result['dominant_agent']}\n"
                    f"NOTE: This is a summary of agent votes. You are NOT bound by this.\n"
                    f"Your job is to evaluate the QUALITY of arguments, not count votes.\n"
                    f"A strong thesis with weak consensus is still a valid trade.\n"
                )

                # D. Master Decision
                # --- NEW: GET MARKET CONTEXT (The "Reality Check") ---
                market_context_str = "MARKET CONTEXT: Data unavailable."
                live_price_val = None # Capture live price for Sanity Check

                try:
                    # End time = now, Duration = 1 week, Bar size = 1 day
                    bars = await ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime='',
                        durationStr='1 W',
                        barSizeSetting='1 day',
                        whatToShow='TRADES',
                        useRTH=True
                    )

                    if bars and len(bars) >= 2:
                        current_close = bars[-1].close
                        live_price_val = current_close # Update live price

                        prev_close = bars[-2].close
                        price_5d_ago = bars[0].close

                        pct_1d = (current_close - prev_close) / prev_close
                        pct_5d = (current_close - price_5d_ago) / price_5d_ago

                        market_context_str = (
                            f"MARKET REALITY CHECK:\n"
                            f"- Current Price: {current_close}\n"
                            f"- 24h Change: {pct_1d:+.2%} (Did the market already react?)\n"
                            f"- 5-Day Trend: {pct_5d:+.2%} (Is the move extended?)"
                        )

                        # "Priced In" Guardrail
                        if final_data["action"] == "BULLISH" and pct_1d > 0.03:
                            logger.warning(f"Signal PRICED IN. Market up {pct_1d:.1%} in 24h. Forcing NEUTRAL.")
                            final_data["action"] = "NEUTRAL"
                            final_data["reason"] = "Signal Priced In (24h move > 3%)"
                            return {
                                **market_ctx,
                                **final_data,
                                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                                "prediction_type": "NEUTRAL",  # <-- FIX: Add missing key
                                "direction": "NEUTRAL",
                                "confidence": 0.0,
                                "reason": "Signal Priced In (24h move > 3%)",
                                "_agent_reports": reports
                            }

                        elif final_data["action"] == "BEARISH" and pct_1d < -0.03:
                            logger.warning(f"Signal PRICED IN. Market down {pct_1d:.1%} in 24h. Forcing NEUTRAL.")
                            final_data["action"] = "NEUTRAL"
                            final_data["reason"] = "Signal Priced In (24h move < -3%)"
                            return {
                                **market_ctx,
                                **final_data,
                                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                                "prediction_type": "NEUTRAL",  # <-- FIX: Add missing key
                                "direction": "NEUTRAL",
                                "confidence": 0.0,
                                "reason": "Signal Priced In (24h move < -3%)",
                                "_agent_reports": reports
                            }

                except Exception as e:
                    logger.error(f"Failed to fetch history for context: {e}")

                # Append weighted voting result to context
                market_context_str += weighted_context

                # === INJECT SENSOR STATUS ===
                # Check X Sentiment status from StateManager
                try:
                    sensor_state = StateManager.load_state("sensors", max_age=3600)
                    x_status = sensor_state.get("x_sentiment", {})
                    # Handle both dict and potential stale string
                    if isinstance(x_status, dict) and x_status.get("sentiment_sensor_status") == "OFFLINE":
                        market_context_str += "\n\n⚠️ WARNING: X Sentiment Sensor is OFFLINE. " \
                                              "Do not hallucinate sentiment. Treat social sentiment as UNKNOWN."
                except Exception as e:
                    logger.warning(f"Failed to check sensor status: {e}")

                # Call decided (which now includes the Hegelian Loop)
                decision = await council.decide(contract_name, market_ctx, reports, market_context_str)

                # Ensure decision confidence is valid
                try:
                    decision['confidence'] = max(0.0, min(1.0, float(decision.get('confidence', 0.0))))
                except (ValueError, TypeError):
                    decision['confidence'] = 0.0

                # --- v7.0: CONSENSUS SENSOR (Replaces Vote Override) ---
                # The Master's DIRECTION is trusted. The Vote only adjusts CONVICTION.
                # Rationale: LLMs are better at qualitative reasoning (thesis quality)
                # than at producing calibrated numerical confidence scores. The vote's
                # value is measuring how much the agents agree, not which direction is right.
                vote_dir = weighted_result['direction']
                master_dir = decision.get('direction', 'NEUTRAL')

                if master_dir == vote_dir or vote_dir == 'NEUTRAL':
                    conviction_multiplier = 1.0
                    consensus_note = f"Consensus ALIGNED (Vote={vote_dir})"
                elif vote_dir != master_dir and master_dir != 'NEUTRAL':
                    conviction_multiplier = 0.5
                    consensus_note = f"Consensus DIVERGENT (Master={master_dir}, Vote={vote_dir}) → half size"
                else:
                    conviction_multiplier = 0.75
                    consensus_note = f"Consensus PARTIAL (Master={master_dir}, Vote={vote_dir})"

                # Adjust confidence by consensus (preserves thesis_strength bucket but modulates)
                decision['confidence'] = round(decision.get('confidence', 0.5) * conviction_multiplier, 2)
                decision['conviction_multiplier'] = conviction_multiplier
                decision['reasoning'] += f" [{consensus_note}]"

                logger.info(
                    f"Consensus Sensor: Master={master_dir}, Vote={vote_dir} → "
                    f"multiplier={conviction_multiplier}, adj_confidence={decision['confidence']}"
                )

                # === Devil's Advocate Check (Emergency Only — v7.0) ===
                # DA is skipped for scheduled cycles to prevent excessive vetoing.
                # Scheduled cycles already have: 7 agents + debate + compliance.
                # DA adds value for emergency/sentinel triggers where unusual events
                # need extra scrutiny that the routine pipeline may miss.
                is_emergency = trigger_type in [
                    TriggerType.WEATHER, TriggerType.PRICE_SPIKE,
                    TriggerType.NEWS_SENTIMENT, TriggerType.LOGISTICS,
                    TriggerType.MICROSTRUCTURE
                ]

                if is_emergency and decision.get('direction') != 'NEUTRAL' and decision.get('confidence', 0) > 0.5:
                    da_review = await council.run_devils_advocate(
                        decision, str(reports), market_context_str
                    )
                    if not da_review.get('proceed', True):
                        logger.warning(f"DA VETOED emergency trade: {da_review.get('recommendation')}")
                        decision['direction'] = 'NEUTRAL'
                        decision['confidence'] = 0.0
                        decision['reasoning'] += f" [DA VETO: {da_review.get('risks', ['Unknown'])[0]}]"
                    else:
                        logger.info("DA CHECK PASSED for emergency cycle.")

                    # --- R3: Propagate Bypass Flag ---
                    if da_review.get('da_bypassed'):
                        decision['da_bypassed'] = True
                elif not is_emergency:
                    logger.info("DA SKIPPED: Scheduled cycle (v7.0 — DA reserved for emergency triggers)")

                # --- E.1: EARLY COMPLIANCE AUDIT (Hallucination Check) ---
                compliance_approved = True
                compliance_reason = "N/A"

                if decision.get('direction') != 'NEUTRAL' and compliance:
                    audit = await compliance.audit_decision(
                        reports,
                        market_context_str,
                        decision,
                        council.personas.get('master', '')
                    )

                    if not audit.get('approved', True):
                        logger.warning(f"COMPLIANCE BLOCKED {contract_name}: {audit.get('flagged_reason')}")
                        compliance_approved = False
                        compliance_reason = audit.get('flagged_reason', 'N/A')

                        # Override Decision
                        decision['direction'] = "NEUTRAL"
                        decision['confidence'] = 0.0
                        decision['reasoning'] = f"BLOCKED BY COMPLIANCE: {compliance_reason}"

                        # Notify
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "Compliance Veto Triggered",
                            f"Trade for {contract_name} blocked.\nReason: {compliance_reason}"
                        )

                # --- E.2: PRICE SANITY CHECK (The "Fat Finger" Guardrail) ---
                # Robust extraction of price
                proj_price = decision.get('projected_price_5_day') or decision.get('projection_price_5_day')

                # Use Live Price if available, else signal price
                current_price = live_price_val if live_price_val is not None else market_ctx.get('price')

                if proj_price and current_price:
                    # Calculate percent change
                    pct_change = abs((proj_price - current_price) / current_price)

                    # Sanity Check Threshold from Config
                    sanity_threshold = config.get('validation_thresholds', {}).get('prediction_sanity_check_pct', 0.20)

                    # RULE: If price moves > X% in 5 days, it's highly suspicious for Coffee futures
                    if pct_change > sanity_threshold:
                        logger.warning(f"⚠️ PRICE SANITY CHECK FAILED: {contract_name} predicted {pct_change:.1%} move.")
                        # Clamp the price or invalidate the signal
                        decision['direction'] = "NEUTRAL"
                        decision['reasoning'] = "Price projection exceeded volatility safety limits (Hallucination Risk)."
                        decision['confidence'] = 0.0
                        proj_price = current_price # Reset to current

                # E. Update Final Data
                # Robust key extraction for 'projected' vs 'projection'
                price = proj_price

                final_data["action"] = decision.get('direction', final_data["action"])

                # Confidence Clamping with Logging
                raw_conf = decision.get('confidence', final_data["confidence"])
                try:
                    conf_val = float(raw_conf)
                except (ValueError, TypeError):
                    conf_val = 0.5

                clamped_value = max(0.0, min(1.0, abs(conf_val)))

                if conf_val < 0.0 or conf_val > 1.0:
                    logger.warning(f"Confidence Clamping Triggered: {conf_val} -> {clamped_value}. Check model outputs.")

                final_data["confidence"] = clamped_value
                final_data["reason"] = decision.get('reasoning', final_data["reason"])
                if price:
                    final_data["expected_price"] = price

                logger.info(f"Council Decision {contract_name}: {final_data['action']} ({final_data['confidence']})")

                # F. Council Logging (Structured)
                try:
                    def extract_sentiment(text):
                        if not text or not isinstance(text, str): return "N/A"
                        match = re.search(r'\[SENTIMENT: (\w+)\]', text)
                        return match.group(1) if match else "N/A"

                    # Initialize logging variables early to prevent UnboundLocalError
                    prediction_type_log = "DIRECTIONAL"
                    vol_level_log = None
                    strategy_type_log = "NONE"

                    # Store structured data for logging
                    agent_data = {}
                    for key, report in reports.items():
                        if isinstance(report, dict):
                            # Try explicit sentiment first, then regex on data
                            s = report.get('sentiment')
                            if not s or s == 'N/A':
                                s = extract_sentiment(report.get('data', ''))
                            agent_data[f"{key}_sentiment"] = s
                            agent_data[f"{key}_summary"] = str(report.get('data', 'N/A'))
                        else:
                            agent_data[f"{key}_sentiment"] = extract_sentiment(report)
                            agent_data[f"{key}_summary"] = str(report) if report else "N/A"

                    # === DECISION LOGIC FOR PREDICTION TYPE ===
                    # Moved inside logging block to capture state for DB
                    final_direction_log = final_data["action"]
                    # v7.0 SAFETY: Match execution path default
                    vol_sentiment_log = agent_data.get('volatility_sentiment', 'BEARISH')
                    regime_log = regime_for_voting if regime_for_voting != 'UNKNOWN' else market_ctx.get('regime', 'UNKNOWN')

                    if final_direction_log == 'NEUTRAL':
                        agent_conflict_score_log = _calculate_agent_conflict(agent_data)
                        imminent_catalyst_log = _detect_imminent_catalyst(agent_data)

                        # v7.0: Mirror execution-path logic exactly
                        if regime_log == 'RANGE_BOUND' and vol_sentiment_log == 'BEARISH':
                            prediction_type_log = "VOLATILITY"
                            vol_level_log = "LOW"
                            strategy_type_log = "IRON_CONDOR"
                        elif (imminent_catalyst_log or agent_conflict_score_log > 0.6) and vol_sentiment_log != 'BEARISH':
                            prediction_type_log = "VOLATILITY"
                            vol_level_log = "HIGH"
                            strategy_type_log = "LONG_STRADDLE"
                        else:
                            prediction_type_log = "DIRECTIONAL"
                            vol_level_log = None
                            strategy_type_log = "NONE"
                    else:
                        if final_direction_log == 'BULLISH':
                            strategy_type_log = 'BULL_CALL_SPREAD'
                        elif final_direction_log == 'BEARISH':
                            strategy_type_log = 'BEAR_PUT_SPREAD'

                    council_log_entry = {
                        "cycle_id": cycle_id,
                        "timestamp": datetime.now(timezone.utc),
                        "contract": contract_name,
                        "entry_price": market_ctx.get('price'),

                        # Unpack agent data (FULL TEXT, NO TRUNCATION)
                        # NOTE: Agent keys map to UI-friendly names
                        # agronomist (weather analyst) -> meteorologist (dashboard display)
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

                        # [NEW] Volatility Logging
                        "volatility_sentiment": agent_data.get('volatility_sentiment'),
                        "volatility_summary": agent_data.get('volatility_summary'),

                        "master_decision": decision.get('direction'),
                        "master_confidence": decision.get('confidence'),
                        "master_reasoning": decision.get('reasoning'),

                        # v7.0: New thesis-based fields
                        "thesis_strength": decision.get('thesis_strength', 'SPECULATIVE'),
                        "primary_catalyst": decision.get('primary_catalyst', 'Not specified'),
                        "conviction_multiplier": decision.get('conviction_multiplier', 1.0),
                        "dissent_acknowledged": decision.get('dissent_acknowledged', 'None stated'),

                        # [NEW] Prediction & Strategy Fields
                        "prediction_type": prediction_type_log,
                        "volatility_level": vol_level_log,
                        "strategy_type": strategy_type_log,

                        "compliance_approved": compliance_approved,
                        "compliance_reason": compliance_reason if not compliance_approved else "N/A",

                        # [NEW] Weighted Voting Data
                        "vote_breakdown": json.dumps(weighted_result.get('vote_breakdown', [])),
                        "dominant_agent": weighted_result.get('dominant_agent', 'Unknown'),
                        "weighted_score": weighted_result.get('weighted_score', 0.0),
                        "trigger_type": weighted_result.get('trigger_type', 'scheduled'),
                    }
                    log_council_decision(council_log_entry)

                    # G. Decision Signal (Lightweight summary for quick-check)
                    log_decision_signal(
                        cycle_id=cycle_id,
                        contract=contract_name,
                        signal=final_direction_log if final_direction_log != 'NEUTRAL' or prediction_type_log != 'VOLATILITY' else 'VOLATILITY',
                        prediction_type=prediction_type_log,
                        strategy=strategy_type_log,
                        price=market_ctx.get('price'),
                        sma_200=market_ctx.get('sma_200'),
                        confidence=final_data.get('confidence'),
                        regime=regime_log,
                        trigger_type=trigger_type.value.upper() if isinstance(trigger_type, TriggerType) else str(trigger_type or 'SCHEDULED').upper(),
                    )

                    # === BRIER SCORE RECORDING (Dual-Write: Legacy CSV + Enhanced JSON) ===
                    try:
                        from trading_bot.agent_names import CANONICAL_AGENTS, DEPRECATED_AGENTS
                        BRIER_ELIGIBLE_AGENTS = set(CANONICAL_AGENTS) - {'master_decision'} - DEPRECATED_AGENTS
                        timestamp_now = datetime.now(timezone.utc)

                        # Record each agent's prediction
                        for agent_name, report in reports.items():
                            if agent_name not in BRIER_ELIGIBLE_AGENTS:
                                continue

                            # Default
                            direction = 'NEUTRAL'
                            confidence = 0.5

                            if isinstance(report, dict):
                                # Structured report
                                direction = report.get('sentiment', 'NEUTRAL')
                                confidence = report.get('confidence', 0.5)
                                # Fallback parsing if sentiment missing or raw string inside dict
                                if not direction or direction == 'N/A':
                                    report_str = str(report.get('data', '')).upper()
                                    if 'BULLISH' in report_str: direction = 'BULLISH'
                                    elif 'BEARISH' in report_str: direction = 'BEARISH'
                            else:
                                # Legacy string report
                                report_str = str(report).upper()
                                if 'BULLISH' in report_str: direction = 'BULLISH'
                                elif 'BEARISH' in report_str: direction = 'BEARISH'

                            record_agent_prediction(
                                agent=agent_name,
                                predicted_direction=direction,
                                predicted_confidence=float(confidence),
                                cycle_id=cycle_id,
                                regime=regime_log,
                                contract=contract_name,
                                timestamp=timestamp_now,
                            )

                        # Also record Master Decision
                        record_agent_prediction(
                            agent='master',
                            predicted_direction=decision.get('direction', 'NEUTRAL'),
                            predicted_confidence=float(decision.get('confidence', 0.5)),
                            cycle_id=cycle_id,
                            regime=regime_log,
                            contract=contract_name,
                            timestamp=timestamp_now,
                        )

                    except Exception as brier_e:
                        logger.error(f"Failed to record Brier predictions for {contract_name}: {brier_e}")

                except Exception as log_e:
                    logger.error(f"Failed to log council decision for {contract_name}: {log_e}")

            except CriticalRPCError as e:
                logger.critical(
                    f"CRITICAL: Executor shutdown for {contract_name}. "
                    f"Aborting cycle — will not produce garbage decisions. Error: {e}"
                )
                return {
                    **market_ctx,
                    "contract_month": contract.lastTradeDateOrContractMonth[:6],
                    "prediction_type": "NEUTRAL",
                    "direction": "NEUTRAL",
                    "confidence": 0.0,
                    "reason": f"CYCLE ABORTED: {e}",
                    "_aborted": True,  # Flag for downstream
                }

            except Exception as e:
                logger.error(f"Council execution failed for {contract_name}: {e}")
                send_pushover_notification(
                    config.get('notifications', {}),
                    "Trading Council Error",
                    f"Council failed for {contract_name}. Using ML fallback. Error: {e}"
                )
                # We silently fall back to the ML defaults set at start of function

        # === v7.0: HARD-CODED STRATEGY SELECTION ("Judge & Jury" Protocol) ===
        # Strategy type is determined by HARD DATA (vol_sentiment, regime),
        # NOT by LLM suggestion. The Master's volatility_play output is IGNORED.
        #
        # Design principle: LLMs decide WHAT (direction + thesis).
        # Python code decides HOW (strategy type + sizing).
        #
        # Origin: Merges "Cash Is a Position" hotfix + Judge & Jury routing
        # + conservative vol defaults (Addendum PATCH 2).

        final_direction = final_data["action"]

        # v7.0 SAFETY: Default to BEARISH (expensive) when vol data is missing.
        # Rationale: On a $50K account, assume worst-case (expensive options)
        # rather than neutral. Fail-safe, not fail-neutral.
        vol_sentiment = agent_data.get('volatility_sentiment', 'BEARISH')

        regime = regime_for_voting if regime_for_voting != 'UNKNOWN' else market_ctx.get('regime', 'UNKNOWN')
        thesis_strength = decision.get('thesis_strength', 'SPECULATIVE') if 'decision' in dir() else 'SPECULATIVE'

        prediction_type = "DIRECTIONAL"
        vol_level = None
        reason = final_data["reason"]

        if final_direction == 'NEUTRAL':
            # === NEUTRAL PATH: Vol trade or No Trade ===
            agent_conflict_score = _calculate_agent_conflict(agent_data)
            imminent_catalyst = _detect_imminent_catalyst(agent_data)

            # PATH 1: IRON CONDOR — sell premium in a range when vol is expensive
            if regime == 'RANGE_BOUND' and vol_sentiment == 'BEARISH':
                prediction_type = "VOLATILITY"
                vol_level = "LOW"
                reason = f"Iron Condor: Range-bound + expensive vol (sell premium)"
                logger.info(f"STRATEGY SELECTED: IRON_CONDOR | regime={regime}, vol={vol_sentiment}")

            # PATH 2: LONG STRADDLE — expect big move, but only if options aren't expensive
            elif (imminent_catalyst or agent_conflict_score > 0.6) and vol_sentiment != 'BEARISH':
                prediction_type = "VOLATILITY"
                vol_level = "HIGH"
                reason = f"Long Straddle: {imminent_catalyst or f'High conflict ({agent_conflict_score:.2f})'}"
                logger.info(f"STRATEGY SELECTED: LONG_STRADDLE | catalyst={imminent_catalyst}, conflict={agent_conflict_score:.2f}")

            # PATH 3: NO TRADE — insufficient conviction for any trade
            else:
                prediction_type = "DIRECTIONAL"
                vol_level = None
                reason = (
                    f"NO TRADE: Direction neutral, no positive-EV vol trade available. "
                    f"(vol={vol_sentiment}, regime={regime}, conflict={agent_conflict_score:.2f})"
                )
                logger.info(f"NO TRADE: vol={vol_sentiment}, regime={regime}, catalyst={imminent_catalyst}")

        else:
            # === DIRECTIONAL PATH: Always use defined-risk spreads ===
            # Strategy type forced by code. On a $50K account, we NEVER trade naked options.
            # Vol sentiment only affects whether we LOG a warning, not the strategy itself.
            if vol_sentiment == 'BEARISH':
                reason += f" [VOL WARNING: Options expensive. Spread costs elevated. Thesis: {thesis_strength}]"
                logger.info(f"STRATEGY: Directional spread with expensive vol (thesis={thesis_strength})")
            else:
                logger.info(f"STRATEGY: Directional spread (thesis={thesis_strength}, vol={vol_sentiment})")

        # === SHADOW RUN (v7.0 Strategy Router) ===
        # Verify the new router produces identical results to the legacy logic
        try:
            routed_shadow = route_strategy(
                direction=final_direction,
                confidence=final_data["confidence"],
                vol_sentiment=vol_sentiment,
                regime=regime,
                thesis_strength=thesis_strength,
                conviction_multiplier=decision.get('conviction_multiplier', 1.0) if 'decision' in dir() else 1.0,
                reasoning=final_data["reason"],
                agent_data=agent_data,
                mode="scheduled",
            )

            # Assertion: Check invariant
            mismatch = False
            if routed_shadow['prediction_type'] != prediction_type:
                mismatch = True
            if routed_shadow['vol_level'] != vol_level:
                mismatch = True

            if mismatch:
                logger.critical(
                    f"ROUTING MISMATCH (Shadow Run): "
                    f"Legacy=[{prediction_type}, {vol_level}], "
                    f"Router=[{routed_shadow['prediction_type']}, {routed_shadow['vol_level']}]"
                )
            else:
                logger.info("Strategy Router Shadow Run: MATCH ✅")

        except Exception as e:
            logger.error(f"Strategy Router Shadow Run FAILED: {e}")

        # Construct final signal object
        if prediction_type == "VOLATILITY":
            return {
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "VOLATILITY",
                "level": vol_level,  # "HIGH" or "LOW"
                "direction": "VOLATILITY", # Needed for logging and to bypass order_manager skip
                "reason": reason,
                "regime": regime,
                "volatility_sentiment": vol_sentiment,
                "confidence": final_data["confidence"],
                "price": market_ctx.get('price'),
                "sma_200": market_ctx.get('sma_200'),
                "expected_price": final_data["expected_price"],
                "predicted_return": market_ctx.get('predicted_return'),
                # v7.0: Carry forward for Phase 5 position sizing
                "thesis_strength": decision.get('thesis_strength', 'SPECULATIVE') if 'decision' in dir() else 'SPECULATIVE',
                "conviction_multiplier": decision.get('conviction_multiplier', 1.0) if 'decision' in dir() else 1.0,
                "_agent_reports": reports
            }
        else:
            return {
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "DIRECTIONAL",
                "direction": final_direction,
                "reason": reason,
                "regime": regime,
                "volatility_sentiment": vol_sentiment,
                "confidence": final_data["confidence"],
                "price": market_ctx.get('price'),
                "sma_200": market_ctx.get('sma_200'),
                "expected_price": final_data["expected_price"],
                "predicted_return": market_ctx.get('predicted_return'),
                # v7.0: Carry forward for Phase 5 position sizing
                "thesis_strength": decision.get('thesis_strength', 'SPECULATIVE') if 'decision' in dir() else 'SPECULATIVE',
                "conviction_multiplier": decision.get('conviction_multiplier', 1.0) if 'decision' in dir() else 1.0,
                "_agent_reports": reports
            }

    # 5. Execute for all contracts
    tasks = []
    for i, contract in enumerate(sorted_contracts):
        market_ctx = market_contexts[i] if i < len(market_contexts) else {}
        tasks.append(process_contract(contract, market_ctx))

    final_signals_list = await asyncio.gather(*tasks)

    # --- CONSOLIDATED STATE SAVE (Fix Race Condition) ---
    # Extract reports from each signal and save once
    # NOTE: Currently using "General Market Context" approach where
    # last contract's reports are used. This is acceptable because
    # specialist agents analyze market-wide conditions, not contract-specific.
    # TODO: For contract-specific state, key by contract ID:
    #       e.g., "reports_KCZ25_agronomist" instead of "agronomist"

    consolidated_reports = {}
    for sig in final_signals_list:
        if isinstance(sig, dict) and "_agent_reports" in sig:
            # Last write wins (General Context approach)
            consolidated_reports.update(sig.pop("_agent_reports"))

    # Single atomic save
    if consolidated_reports:
        try:
            StateManager.save_state(consolidated_reports)
            logger.info(f"Persisted {len(consolidated_reports)} agent reports to state.")
        except Exception as e:
            logger.error(f"Failed to persist agent reports: {e}")

    return final_signals_list
