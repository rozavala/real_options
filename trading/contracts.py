import logging
from datetime import datetime
from ib_insync import *

# --- Contract & Option Chain Utilities ---

async def get_active_futures(ib: IB, symbol: str, exchange: str, count: int = 5) -> list[Contract]:
    """
    Fetches the next 'count' active futures contracts for a given symbol.
    Contracts are sorted by expiration date, soonest first.
    """
    logging.info(f"Fetching {count} active futures contracts for {symbol} on {exchange}...")
    try:
        # Request contract details for all futures of the given symbol
        cds = await ib.reqContractDetailsAsync(Future(symbol, exchange=exchange))

        # Filter for contracts that have not yet expired and sort them
        active_contracts = sorted(
            [cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth > datetime.now().strftime('%Y%m')],
            key=lambda c: c.lastTradeDateOrContractMonth
        )

        logging.info(f"Found {len(active_contracts)} active contracts. Returning the first {count}.")
        return active_contracts[:count]
    except Exception as e:
        logging.error(f"Error fetching active futures: {e}")
        return []

async def build_option_chain(ib: IB, future_contract: Contract) -> dict | None:
    """
    Fetches the full option chain for a given underlying future contract.
    """
    logging.info(f"Fetching option chain for future {future_contract.localSymbol}...")
    try:
        # Request security definition option parameters
        chains = await ib.reqSecDefOptParamsAsync(
            underlyingSymbol=future_contract.symbol,
            futFopExchange=future_contract.exchange,
            underlyingSecType='FUT',
            underlyingConId=future_contract.conId
        )

        if not chains:
            logging.warning(f"No option chain parameters found for {future_contract.localSymbol}.")
            return None

        # Find the primary chain, usually matching the future's exchange, or take the first one
        chain = next((c for c in chains if c.exchange == future_contract.exchange), chains[0])

        # Structure the chain data for easier access
        option_chain = {
            'exchange': chain.exchange,
            'tradingClass': chain.tradingClass,
            'multiplier': chain.multiplier,
            'expirations': sorted(chain.expirations),
            'strikes_by_expiration': {exp: sorted(chain.strikes) for exp in chain.expirations}
        }
        logging.info(f"Successfully built option chain for {future_contract.localSymbol} with {len(option_chain['expirations'])} expirations.")
        return option_chain
    except Exception as e:
        logging.error(f"Failed to build option chain for {future_contract.localSymbol}: {e}")
        return None

def get_expiration_details(chain: dict, future_exp_str: str) -> dict | None:
    """
    Selects the most appropriate option expiration date from the chain.
    It prefers expirations that occur on or before the future's expiration.
    """
    # Find all expirations that are valid (i.e., before or on the same month as the future expires)
    valid_expirations = [exp for exp in sorted(chain['expirations']) if exp <= future_exp_str]

    if not valid_expirations:
        # If no options expire before the future, it might be a quarterly. Use the nearest available option.
        logging.warning(f"No option expiration found before future '{future_exp_str}'. Searching for the closest available.")
        valid_expirations = [exp for exp in sorted(chain['expirations']) if exp > datetime.now().strftime('%Y%m%d')]
        if not valid_expirations:
            logging.error(f"No suitable option expirations found for future {future_exp_str}.")
            return None

    # Choose the latest valid expiration date
    chosen_exp = valid_expirations[-1]

    try:
        days_to_expiry = (datetime.strptime(chosen_exp, '%Y%m%d').date() - datetime.now().date()).days
        if days_to_expiry < 0:
            logging.warning(f"Chosen expiration {chosen_exp} is in the past. This may indicate an issue.")
            return None # Avoid trading expired options

        return {
            'exp_date': chosen_exp,
            'days_to_exp': days_to_expiry,
            'strikes': chain['strikes_by_expiration'][chosen_exp]
        }
    except (ValueError, KeyError) as e:
        logging.error(f"Could not process expiration details for {chosen_exp}: {e}")
        return None