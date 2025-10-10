"""Translates raw API predictions into structured trading signals.

This module is responsible for taking the numerical price change predictions
from the prediction API and converting them into a list of structured,
actionable trading signals that the rest of the trading bot can understand
and execute.
"""

import logging
from datetime import datetime

from ib_insync import IB, Contract

from logging_config import setup_logging
from trading_bot.ib_interface import get_active_futures

# --- Logging Setup ---
setup_logging()


async def generate_signals(ib: IB, api_response: dict, config: dict) -> list:
    """Generates structured trading signals from raw API price predictions.

    This function maps the `price_changes` from the API response to the
    active coffee futures contracts. It sorts the contracts chronologically
    to ensure the predictions align with the correct contract months. Based on
    pre-defined thresholds in the configuration, it determines whether a
    price change constitutes a 'BULLISH' or 'BEARISH' directional signal.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
        api_response (dict): The JSON response from the prediction API.
            It is expected to have a "price_changes" key with a list of floats.
        config (dict): The application's configuration dictionary, used to
            access signal thresholds and contract symbol information.

    Returns:
        A list of signal dictionaries. Each dictionary represents an
        actionable signal and contains the contract month, prediction type
        (always 'DIRECTIONAL'), and the determined direction ('BULLISH' or
        'BEARISH'). Returns an empty list if the API response is invalid or
        not enough active futures can be fetched.
    """
    price_changes = api_response.get("price_changes")
    if not price_changes or not isinstance(price_changes, list) or len(price_changes) != 5:
        logging.error(f"Invalid or missing 'price_changes' in API response: {api_response}")
        return []

    logging.info("Generating trading signals from API response...")

    # The API returns predictions sorted alphabetically by month code (H, K, N, U, Z)
    # Create a mapping from the month code to its corresponding prediction.
    month_codes_alpha = ['H', 'K', 'N', 'U', 'Z']
    predictions_by_month = dict(zip(month_codes_alpha, price_changes))
    logging.info(f"Received predictions mapped to month codes: {predictions_by_month}")


    # Fetch the active futures contracts to map predictions to
    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    if len(active_futures) < len(price_changes):
        logging.error(f"Could not retrieve enough active futures contracts. "
                      f"Needed {len(price_changes)}, found {len(active_futures)}.")
        return []

    # Define the standard chronological order of futures month codes
    month_code_order = {'H': 0, 'K': 1, 'N': 2, 'U': 3, 'Z': 4}
    month_to_code = {3: 'H', 5: 'K', 7: 'N', 9: 'U', 12: 'Z'}

    def get_sort_key(contract: Contract) -> tuple[int, int]:
        """Creates a sort key based on year and the month code order.

        Args:
            contract (Contract): The futures contract to generate a key for.

        Returns:
            A tuple (year, month_order) used for sorting.
        """
        try:
            year = int(contract.lastTradeDateOrContractMonth[:4])
            month = int(contract.lastTradeDateOrContractMonth[4:6])
            code = month_to_code.get(month)
            if code is None:
                return (year, 99)  # Place at the end if code is unknown
            return (year, month_code_order[code])
        except (ValueError, IndexError):
            logging.warning(f"Could not parse date for contract: {contract.localSymbol}")
            return (9999, 99) # Place problematic contracts at the end

    # Sort contracts chronologically to ensure we are evaluating the nearest expirations
    sorted_contracts = sorted(active_futures, key=get_sort_key)

    signals = []
    for contract in sorted_contracts:
        # Extract the month from the contract's expiration date string
        try:
            month = int(contract.lastTradeDateOrContractMonth[4:6])
            contract_month_code = month_to_code.get(month)
        except (ValueError, IndexError):
            logging.warning(f"Could not determine month code for contract: {contract.localSymbol}. Skipping.")
            continue

        # Look up the correct prediction using the contract's month code
        price_change = predictions_by_month.get(contract_month_code)
        if price_change is None:
            logging.warning(f"No prediction found for month code '{contract_month_code}' in contract {contract.localSymbol}. Skipping.")
            continue

        # Determine signal direction based on configured thresholds
        thresholds = config.get('signal_thresholds', {})
        bullish_threshold = thresholds.get('bullish', 7)
        bearish_threshold = thresholds.get('bearish', 0)

        log_msg = (f"Evaluating contract {contract.localSymbol} ({contract.lastTradeDateOrContractMonth[:6]}): "
                   f"Price Change = {price_change:.2f}. "
                   f"Thresholds (Bullish > {bullish_threshold}, Bearish < {bearish_threshold}).")

        direction = None
        if price_change > bullish_threshold:
            direction = "BULLISH"
            log_msg += " -> RESULT: BULLISH"
        elif price_change < bearish_threshold:
            direction = "BEARISH"
            log_msg += " -> RESULT: BEARISH"
        else:
            log_msg += " -> RESULT: NO-TRADE"

        logging.info(log_msg)

        if direction:
            signal = {
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "DIRECTIONAL",
                "direction": direction
            }
            signals.append(signal)

    return signals
