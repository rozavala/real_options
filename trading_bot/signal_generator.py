
import logging
from ib_insync import IB, Contract, Future
from logging_config import setup_logging
from trading_bot.ib_interface import get_active_futures
from trading_bot.model_signals import log_model_signal

setup_logging()

async def generate_signals(ib: IB, signals_list: list, config: dict) -> list:
    """
    Generates structured trading signals from the inference engine's output.
    Maps detailed signals (dicts) to active futures contracts chronologically.
    """
    if not signals_list or not isinstance(signals_list, list) or len(signals_list) != 5:
        logging.error(f"Invalid signals list from inference engine: {signals_list}")
        return []

    logging.info("Generating trading signals from Inference Engine output...")

    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    if len(active_futures) < len(signals_list):
        logging.error(f"Could not retrieve enough active futures. "
                      f"Needed {len(signals_list)}, found {len(active_futures)}.")
        return []

    # Sort contracts chronologically to align with prediction order (front_month, second_month, etc.)
    sorted_contracts = sorted(active_futures, key=lambda c: c.lastTradeDateOrContractMonth)

    generated_signals = []
    for i, contract in enumerate(sorted_contracts):
        signal_data = signals_list[i]

        action = signal_data.get('action', 'NEUTRAL')
        confidence = signal_data.get('confidence', 0.0)
        reason = signal_data.get('reason', 'N/A')
        regime = signal_data.get('regime', 'N/A')

        # Map 'action' to Direction
        direction = "NEUTRAL"
        if action == "LONG":
            direction = "BULLISH"
        elif action == "SHORT":
            direction = "BEARISH"

        logging.info(f"Contract {contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]}): "
                     f"{action} ({confidence:.2%}) | {reason}")

        # Log the signal regardless of whether we trade it
        log_model_signal(
            contract.lastTradeDateOrContractMonth[:6],
            direction,
            price=signal_data.get('price'),
            sma_200=signal_data.get('sma_200'),
            expected_price=signal_data.get('expected_price'),
            confidence=signal_data.get('confidence')
        )

        if direction != "NEUTRAL":
            generated_signals.append({
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "DIRECTIONAL",
                "direction": direction,
                "reason": reason,
                "regime": regime,
                "confidence": confidence
            })

    return generated_signals
