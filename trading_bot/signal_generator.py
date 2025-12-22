
import logging
import asyncio
import traceback
import re
from datetime import datetime
from ib_insync import IB
from trading_bot.agents import CoffeeCouncil
from trading_bot.ib_interface import get_active_futures # CORRECTED IMPORT
from trading_bot.model_signals import log_model_signal # Keep existing logging
from trading_bot.utils import log_council_decision
from notifications import send_pushover_notification

logger = logging.getLogger(__name__)

async def generate_signals(ib: IB, signals_list: list, config: dict) -> list:
    """
    Generates trading signals by combining the existing ML signals (signals_list)
    with the Gemini Coffee Council's qualitative analysis.
    """
    # 1. Validation
    if not signals_list or not isinstance(signals_list, list) or len(signals_list) != 5:
        logger.error(f"Invalid signals list from inference engine: {signals_list}")
        return []

    logger.info("--- Starting Multi-Agent Signal Generation (Council) ---")

    # 2. Initialize Council
    council = None
    try:
        council = CoffeeCouncil(config)
    except Exception as e:
        logger.error(f"Failed to initialize Coffee Council: {e}. Falling back to raw ML signals.")
        send_pushover_notification(
            config.get('notifications', {}),
            "Coffee Council Startup Failed",
            f"Error: {e}. Bot will use raw ML signals."
        )
        # We proceed; logic below handles council=None by skipping agent calls

    # 3. Get Active Futures (using passed-in IB instance)
    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)

    # Sort chronologically to match the assumed order of signals_list
    sorted_contracts = sorted(active_futures, key=lambda c: c.lastTradeDateOrContractMonth)

    if len(sorted_contracts) < len(signals_list):
        logger.warning(f"Mismatch: {len(signals_list)} signals but only {len(sorted_contracts)} active contracts found.")

    final_signals = []

    # Semaphore to prevent hitting Gemini rate limits (max 3 concurrent contract analysis groups)
    sem = asyncio.Semaphore(3)

    # 4. Define the async processor for a single contract
    async def process_contract(contract, ml_signal):
        contract_name = f"{contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]})"

        # Default values (Fallback to ML)
        final_data = {
            "action": "NEUTRAL",
            "confidence": ml_signal.get('confidence', 0.0),
            "expected_price": ml_signal.get('expected_price'),
            "reason": ml_signal.get('reason', 'N/A')
        }

        # Map ML 'LONG'/'SHORT' to Council's 'BULLISH'/'BEARISH'
        raw_action = ml_signal.get('action', 'NEUTRAL')
        if raw_action == "LONG": final_data["action"] = "BULLISH"
        elif raw_action == "SHORT": final_data["action"] = "BEARISH"

        # If Council is offline, return ML defaults immediately
        if not council:
            return {
                **ml_signal,
                **final_data,
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "direction": final_data["action"]  # Ensure 'direction' key exists for logging
            }

        async with sem:
            try:
                logger.info(f"Convening Council for {contract_name}...")

                # A. Define Research Tasks
                # Note: We use specific keywords to guide the Flash model search
                tasks = {
                    "meteorologist": council.research_topic("meteorologist",
                        f"Search for 'current 10-day weather forecast Minas Gerais coffee zone' and 'NOAA Brazil precipitation anomaly'. Analyze if recent rains are beneficial for flowering or excessive."),
                    "macro": council.research_topic("macro",
                        f"Search for 'USD BRL exchange rate forecast' and 'Brazil Central Bank Selic rate outlook'. Determine if the BRL is trending to encourage farmer selling."),
                    "geopolitical": council.research_topic("geopolitical",
                        f"Search for 'Red Sea shipping coffee delays', 'Brazil port of Santos wait times', and 'EUDR regulation delay latest news'. Determine if there are logistical bottlenecks."),
                    "sentiment": council.research_topic("sentiment",
                        f"Search for 'Coffee COT report non-commercial net length'. Determine if market is overbought.")
                }

                # B. Execute Research (Parallel) with Rate Limit Protection
                # Add slight stagger to avoid instantaneous burst
                await asyncio.sleep(0.5)
                research_results = await asyncio.gather(*tasks.values(), return_exceptions=True)

                # C. Format Reports
                reports = {}
                for key, res in zip(tasks.keys(), research_results):
                    if isinstance(res, Exception):
                        logger.error(f"Agent {key} failed for {contract_name}: {res}")
                        reports[key] = "Data Unavailable"
                    else:
                        reports[key] = res

                # D. Master Decision
                # --- NEW: GET MARKET CONTEXT (The "Reality Check") ---
                market_context_str = "MARKET CONTEXT: Data unavailable."
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
                except Exception as e:
                    logger.error(f"Failed to fetch history for context: {e}")

                decision = await council.decide(contract_name, ml_signal, reports, market_context_str)

                # --- E.1: THE COMPLIANCE AUDIT (New Step) ---
                audit = await council.audit_decision(contract_name, reports, decision, market_context=market_context_str)

                if not audit.get('approved', True):
                    logger.warning(f"🚨 COMPLIANCE BLOCKED {contract_name}: {audit.get('flagged_reason')}")
                    # Downgrade to NEUTRAL if the AI was hallucinating facts
                    decision['direction'] = "NEUTRAL"
                    decision['reasoning'] = f"BLOCKED BY AUDIT: {audit.get('flagged_reason')}"
                    decision['confidence'] = 0.0

                # --- E.2: PRICE SANITY CHECK (The "Fat Finger" Guardrail) ---
                # Robust extraction of price
                proj_price = decision.get('projected_price_5_day') or decision.get('projection_price_5_day')
                current_price = ml_signal.get('price')

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
                final_data["confidence"] = decision.get('confidence', final_data["confidence"])
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

                    # Store structured data for logging
                    agent_data = {}
                    for key, report in reports.items():
                        agent_data[f"{key}_sentiment"] = extract_sentiment(report)
                        agent_data[f"{key}_summary"] = str(report) if report else "N/A"  # SAVE FULL TEXT

                    council_log_entry = {
                        "timestamp": datetime.now(),
                        "contract": contract_name,
                        "entry_price": ml_signal.get('price'),
                        "ml_signal": raw_action,
                        "ml_confidence": ml_signal.get('confidence'),

                        # Unpack agent data (FULL TEXT, NO TRUNCATION)
                        "meteorologist_sentiment": agent_data.get('meteorologist_sentiment'),
                        "meteorologist_summary": agent_data.get('meteorologist_summary'),
                        "macro_sentiment": agent_data.get('macro_sentiment'),
                        "macro_summary": agent_data.get('macro_summary'),
                        "geopolitical_sentiment": agent_data.get('geopolitical_sentiment'),
                        "geopolitical_summary": agent_data.get('geopolitical_summary'),
                        "sentiment_sentiment": agent_data.get('sentiment_sentiment'),
                        "sentiment_summary": agent_data.get('sentiment_summary'),

                        "master_decision": decision.get('direction'),
                        "master_confidence": decision.get('confidence'),
                        "master_reasoning": decision.get('reasoning'),
                        "compliance_approved": audit.get('approved', True)
                    }
                    log_council_decision(council_log_entry)
                except Exception as log_e:
                    logger.error(f"Failed to log council decision for {contract_name}: {log_e}")

            except Exception as e:
                logger.error(f"Council execution failed for {contract_name}: {e}")
                send_pushover_notification(
                    config.get('notifications', {}),
                    "Coffee Council Error",
                    f"Council failed for {contract_name}. Using ML fallback. Error: {e}"
                )
                # We silently fall back to the ML defaults set at start of function

        # Construct final signal object
        return {
            "contract_month": contract.lastTradeDateOrContractMonth[:6],
            "prediction_type": "DIRECTIONAL",
            "direction": final_data["action"],
            "reason": final_data["reason"],
            "regime": ml_signal.get('regime', 'N/A'),
            "confidence": final_data["confidence"],
            "price": ml_signal.get('price'),
            "sma_200": ml_signal.get('sma_200'),
            "expected_price": final_data["expected_price"]
        }

    # 5. Execute for all contracts
    tasks = []
    for i, contract in enumerate(sorted_contracts):
        # Match contract to the corresponding ML signal by index
        # (Assuming signals_list is sorted chronologically like active_futures)
        ml_signal = signals_list[i] if i < len(signals_list) else {}
        tasks.append(process_contract(contract, ml_signal))

    final_signals_list = await asyncio.gather(*tasks)

    # 6. Logging & Return
    for sig in final_signals_list:
        # Log to CSV/File using existing utility
        log_model_signal(
            sig['contract_month'],
            sig['direction'],
            price=sig.get('price'),
            sma_200=sig.get('sma_200'),
            expected_price=sig.get('expected_price'),
            confidence=sig.get('confidence')
        )

    return final_signals_list
