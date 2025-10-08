import logging
from datetime import datetime

from ib_insync import IB, Contract

from logging_config import setup_logging
from trading_bot.ib_interface import get_active_futures

# --- Logging Setup ---
setup_logging()


async def generate_signals(ib: IB, api_response: dict, config: dict) -> list:
    """
    Generates structured trading signals from the raw API price change predictions.

    The function maps the `price_changes` from the API to the five active
    coffee futures contracts, sorted alphabetically by their month code (H, K, N, U, Z).
    """
    price_changes = api_response.get("price_changes")
    if not price_changes or not isinstance(price_changes, list):
        logging.error(f"Invalid or missing 'price_changes' in API response: {api_response}")
        return []

    logging.info("Generating trading signals from API response...")

    # Fetch the five active futures contracts
    active_futures = await get_active_futures(ib, config['symbol'], config['exchange'], count=5)
    if len(active_futures) < len(price_changes):
        logging.error(f"Could not retrieve enough active futures contracts. "
                      f"Needed {len(price_changes)}, found {len(active_futures)}.")
        return []

    # Define the alphabetical order of month codes for sorting
    month_code_order = {'H': 0, 'K': 1, 'N': 2, 'U': 3, 'Z': 4}
    month_to_code = {3: 'H', 5: 'K', 7: 'N', 9: 'U', 12: 'Z'}

    def get_sort_key(contract: Contract):
        """Creates a sort key based on year and the alphabetical month code order."""
        try:
            year = int(contract.lastTradeDateOrContractMonth[:4])
            month = int(contract.lastTradeDateOrContractMonth[4:6])
            code = month_to_code.get(month)
            if code is None:
                return (year, 99)  # Should not happen, place at the end
            return (year, month_code_order[code])
        except (ValueError, IndexError):
            logging.warning(f"Could not parse date for contract: {contract.localSymbol}")
            return (9999, 99) # Place problematic contracts at the end

    # Sort the contracts to match the order of the API's price_changes (alphabetical by month code)
    sorted_contracts = sorted(active_futures, key=get_sort_key)

    signals = []
    for i, contract in enumerate(sorted_contracts):
        if i >= len(price_changes):
            break

        price_change = price_changes[i]
        direction = None

        thresholds = config.get('signal_thresholds', {})
        bullish_threshold = thresholds.get('bullish', 7)
        bearish_threshold = thresholds.get('bearish', 0)

        if price_change > bullish_threshold:
            direction = "BULLISH"
        elif price_change < bearish_threshold:
            direction = "BEARISH"

        if direction:
            signal = {
                "contract_month": contract.lastTradeDateOrContractMonth[:6],
                "prediction_type": "DIRECTIONAL",
                "direction": direction
            }
            signals.append(signal)
            logging.info(f"Generated signal for {signal['contract_month']}: {signal['direction']}")

    return signals