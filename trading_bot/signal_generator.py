
import logging
from ib_insync import IB, Contract, Future
from logging_config import setup_logging
from trading_bot.ib_interface import get_active_futures
from trading_bot.model_signals import log_model_signal

setup_logging()

async def generate_signals(ib: IB, api_response: dict, config: dict) -> list:
    """
    Generates structured trading signals from raw API price predictions.
    Maps predictions to active futures contracts chronologically.
    """
    price_changes = api_response.get("price_changes")
    if not price_changes or not isinstance(price_changes, list) or len(price_changes) != 5:
        logging.error(f"Invalid 'price_changes' in API response: {api_response}")
        return []

    logging.info("Generating trading signals from API response...")

    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    if len(active_futures) < len(price_changes):
        logging.error(f"Could not retrieve enough active futures. "
                      f"Needed {len(price_changes)}, found {len(active_futures)}.")
        return []

    # Sort contracts chronologically to align with prediction order (front_month, second_month, etc.)
    sorted_contracts = sorted(active_futures, key=lambda c: c.lastTradeDateOrContractMonth)

    signals = []
    for i, contract in enumerate(sorted_contracts):
        price_change = price_changes[i]

        thresholds = config.get('signal_thresholds', {})
        bullish_threshold = thresholds.get('bullish', 7)
        bearish_threshold = thresholds.get('bearish', 0)

        direction = "NEUTRAL"
        if price_change > bullish_threshold:
            direction = "BULLISH"
        elif price_change < bearish_threshold:
            direction = "BEARISH"

        logging.info(f"Contract {contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]}): "
                     f"Price Change = {price_change:.2f} -> {direction}")

        log_model_signal(contract.lastTradeDateOrContractMonth[:6], direction)

        if direction != "NEUTRAL":
            signals.append({
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "DIRECTIONAL",
                "direction": direction
            })

    return signals
