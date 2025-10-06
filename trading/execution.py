import asyncio
import logging
import time
import traceback
import os
import csv
from datetime import datetime
from ib_insync import *

from notifications import send_pushover_notification

# --- Order Execution & Management ---

async def wait_for_fill(ib: IB, trade: Trade, config: dict, timeout: int = 180, reason: str = "Strategy Execution"):
    """
    Waits for a trade to be filled. If it times out, the order is canceled.
    Logs the trade to the ledger upon successful fill.
    """
    logging.info(f"Waiting for order {trade.order.orderId} ({reason}) to fill...")
    start_time = time.time()

    while not trade.isDone():
        await asyncio.sleep(1)
        if (time.time() - start_time) > timeout:
            logging.warning(f"Order {trade.order.orderId} did not fill within {timeout}s. Canceling.")
            ib.cancelOrder(trade.order)
            send_pushover_notification(
                config,
                "Order Canceled (Timeout)",
                f"Order {trade.order.orderId} for {trade.contract.localSymbol} was canceled due to timeout."
            )
            break # Exit the loop after canceling

    if trade.orderStatus.status == OrderStatus.Filled:
        logging.info(f"Order {trade.order.orderId} has been successfully filled.")
        log_trade_to_ledger(trade, reason)
    elif trade.orderStatus.status == OrderStatus.Cancelled:
        logging.info(f"Order {trade.order.orderId} was canceled.")
    else:
        logging.warning(f"Order {trade.order.orderId} finished with status: {trade.orderStatus.status}")


def log_trade_to_ledger(trade: Trade, reason: str):
    """
    Writes the details of a filled trade to a CSV ledger file.
    """
    if trade.orderStatus.status != OrderStatus.Filled:
        return

    # For coffee options (KC), the price is in cents per pound and the
    # contract size is 37,500 lbs. The standard multiplier is 375.
    # The avgFillPrice is in points (which are cents for KC), so the
    # total dollar value of the trade is price * quantity * multiplier.
    # This applies to both single-leg options and multi-leg spreads.
    multiplier = 375.0

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'local_symbol': trade.contract.localSymbol,
        'action': trade.order.action,
        'quantity': trade.orderStatus.filled,
        'avg_fill_price': trade.orderStatus.avgFillPrice,
        'total_value_usd': trade.orderStatus.avgFillPrice * trade.orderStatus.filled * multiplier,
        'order_id': trade.order.orderId,
        'reason': reason
    }

    ledger_path = 'trade_ledger.csv'
    try:
        # Check if file exists to write header
        file_exists = os.path.isfile(ledger_path)
        with open(ledger_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logging.info(f"Logged trade to ledger ({reason}): {row['action']} {row['quantity']} of {row['local_symbol']} @ {row['avg_fill_price']}")
    except (IOError, csv.Error) as e:
        logging.error(f"Error writing to trade ledger: {e}")


async def place_combo_order(ib: IB, config: dict, action: str, legs_def: list, exp_details: dict, chain: dict) -> Trade | None:
    """
    Constructs and places a complex combo (bag) order.
    `legs_def` should be a list of tuples: (right: 'C'|'P', leg_action: 'BUY'|'SELL', strike: float)
    """
    logging.info(f"--- Constructing {action} Combo Order ---")

    # Create the Bag contract that will hold the legs
    combo = Bag(
        symbol=config['symbol'],
        exchange=chain['exchange'],
        currency='USD'
    )

    # Create and qualify each leg, then add it to the Bag
    try:
        for right, leg_action, strike in legs_def:
            leg_contract = FuturesOption(
                symbol=config['symbol'],
                lastTradeDateOrContractMonth=exp_details['exp_date'],
                strike=strike,
                right=right,
                exchange=chain['exchange'],
                tradingClass=chain['tradingClass']
            )
            # It's crucial to qualify the contract to get its conId
            await ib.qualifyContractsAsync(leg_contract)

            combo.comboLegs.append(
                ComboLeg(
                    conId=leg_contract.conId,
                    ratio=1,
                    action=leg_action,
                    exchange=chain['exchange']
                )
            )
        logging.info(f"Successfully qualified and added {len(legs_def)} legs to the combo.")
    except Exception as e:
        logging.error(f"Failed to create or qualify combo legs. Aborting order. Error: {e}\n{traceback.format_exc()}")
        return None

    # --- Order Pricing and Placement ---
    # To get a decent limit price, we can request market data for the combo itself.
    await ib.qualifyContractsAsync(combo)
    ticker = ib.reqMktData(combo, '', False, False)
    await asyncio.sleep(2) # Allow time for data to arrive

    bid_price = ticker.bid
    ask_price = ticker.ask

    ib.cancelMktData(combo) # Clean up market data request

    if util.isNan(bid_price) or util.isNan(ask_price) or bid_price <= 0 or ask_price <= 0:
        logging.error(f"Could not get valid bid/ask prices for the combo {combo.localSymbol}. Aborting order.")
        return None

    # For a BUY order (debit), our limit is the ask. For a SELL (credit), our limit is the bid.
    # This is an aggressive limit price designed to get filled quickly.
    limit_price = ask_price if action == 'BUY' else bid_price

    logging.info(f"Combo market prices: Bid={bid_price}, Ask={ask_price}. Setting Limit Price to {limit_price:.2f}")

    order = LimitOrder(
        action,
        config['strategy']['quantity'],
        round(limit_price, 2) # Ensure price is rounded to 2 decimal places
    )
    order.tif = 'GTC' # Good 'Til Canceled

    try:
        trade = ib.placeOrder(combo, order)
        logging.info(f"Placed combo order {trade.order.orderId} for {combo.localSymbol}.")
        # The wait_for_fill logic will handle monitoring and logging
        return trade
    except Exception as e:
        logging.error(f"Failed to place combo order: {e}\n{traceback.format_exc()}")
        return None