
import logging
import asyncio
import traceback
from ib_insync import IB
from trading_bot.logging_config import setup_logging
from trading_bot.ib_interface import get_active_futures
from trading_bot.model_signals import log_model_signal
from trading_bot.agents import CoffeeCouncil
from notifications import send_pushover_notification

setup_logging()
logger = logging.getLogger(__name__)

async def generate_signals(ib: IB, signals_list: list, config: dict) -> list:
    """
    Generates structured trading signals from the inference engine's output
    and the Gemini Coffee Council's analysis.

    Maps detailed signals (dicts) to active futures contracts chronologically.
    """
    if not signals_list or not isinstance(signals_list, list) or len(signals_list) != 5:
        logger.error(f"Invalid signals list from inference engine: {signals_list}")
        return []

    logger.info("Generating trading signals from Inference Engine + Coffee Council...")

    # Initialize Council
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

    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    if len(active_futures) < len(signals_list):
        logger.error(f"Could not retrieve enough active futures. "
                      f"Needed {len(signals_list)}, found {len(active_futures)}.")
        return []

    # Sort contracts chronologically
    sorted_contracts = sorted(active_futures, key=lambda c: c.lastTradeDateOrContractMonth)

    generated_signals = []

    # Process each contract
    for i, contract in enumerate(sorted_contracts):
        ml_signal = signals_list[i]
        contract_name = f"{contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]})"

        # Default to ML signal values
        final_direction = "NEUTRAL"
        raw_action = ml_signal.get('action', 'NEUTRAL')
        if raw_action == "LONG": final_direction = "BULLISH"
        elif raw_action == "SHORT": final_direction = "BEARISH"

        final_confidence = ml_signal.get('confidence', 0.0)
        final_reason = ml_signal.get('reason', 'N/A')
        final_expected_price = ml_signal.get('expected_price')

        # --- COUNCIL EXECUTION ---
        if council:
            try:
                logger.info(f"Convening Coffee Council for {contract_name}...")

                # 1. Parallel Research Tasks
                research_tasks = {
                    "meteorologist": council.research_topic("meteorologist",
                        "Search for 'current 10-day weather forecast Minas Gerais coffee zone' and 'NOAA Brazil precipitation anomaly'. Analyze if recent rains are beneficial for flowering or excessive."),
                    "macro": council.research_topic("macro",
                        "Search for 'USD BRL exchange rate forecast' and 'Brazil Central Bank Selic rate outlook'. Determine if the BRL is trending to encourage farmer selling."),
                    "geopolitical": council.research_topic("geopolitical",
                        "Search for 'Red Sea shipping coffee delays', 'Brazil port of Santos wait times', and 'EUDR regulation delay latest news'. Determine if there are logistical bottlenecks forming."),
                    "sentiment": council.research_topic("sentiment",
                        "Search for 'Coffee COT report non-commercial net length'. Determine if market is overbought.")
                }

                # Execute all research in parallel
                results = await asyncio.gather(*research_tasks.values(), return_exceptions=True)

                # Unpack results into a dictionary, handling exceptions
                research_reports = {}
                for key, result in zip(research_tasks.keys(), results):
                    if isinstance(result, Exception):
                        logger.error(f"Agent {key} failed: {result}")
                        research_reports[key] = "Agent failed to report."
                    else:
                        research_reports[key] = result

                # 2. Master Strategist Decision
                decision = await council.decide(contract_name, ml_signal, research_reports)

                # 3. Update Final Signal
                final_direction = decision.get('direction', final_direction)
                final_confidence = decision.get('confidence', final_confidence)
                final_reason = decision.get('reasoning', f"Master Decision (Fallback Reason). {decision}")
                final_expected_price = decision.get('projected_price_5_day', final_expected_price)

                logger.info(f"Council Decision for {contract_name}: {final_direction} ({final_confidence:.2%}) due to: {final_reason}")

            except Exception as e:
                logger.error(f"Council execution failed for {contract_name}: {e}\n{traceback.format_exc()}")
                # Fallback is already set to ML defaults
                send_pushover_notification(
                    config.get('notifications', {}),
                    "Coffee Council Error",
                    f"Council failed for {contract_name}. Using ML fallback. Error: {e}"
                )

        # Log the final signal (whether from Council or ML)
        log_model_signal(
            contract.lastTradeDateOrContractMonth[:6],
            final_direction,
            price=ml_signal.get('price'),
            sma_200=ml_signal.get('sma_200'),
            expected_price=final_expected_price,
            confidence=final_confidence
        )

        logger.info(f"Final Signal {contract_name}: {final_direction} ({final_confidence:.2%}) | {final_reason}")

        generated_signals.append({
            "contract_month": contract.lastTradeDateOrContractMonth[:6],
            "prediction_type": "DIRECTIONAL",
            "direction": final_direction,
            "reason": final_reason,
            "regime": ml_signal.get('regime', 'N/A'), # Keep regime from ML/Trend
            "confidence": final_confidence,
            "price": ml_signal.get('price'),
            "sma_200": ml_signal.get('sma_200'),
            "expected_price": final_expected_price
        })

    return generated_signals
