"""Manages the entire lifecycle of trading orders.

This module is responsible for generating, queuing, placing, and canceling
orders, as well as closing open positions at the end of the trading day. It
acts as the central hub for all order-related activities, decoupling the
strategy definition from the execution timing.
"""
import asyncio
import logging
import random
import traceback
from ib_insync import *

from config_loader import load_config
from coffee_factors_data_pull_new import main as run_data_pull
from send_data_to_api import send_data_and_get_prediction
from notifications import send_pushover_notification

from trading_bot.ib_interface import (
    get_active_futures, build_option_chain, create_combo_order_object, place_order
)
from trading_bot.signal_generator import generate_signals
from trading_bot.strategy import define_directional_strategy, define_volatility_strategy
from trading_bot.utils import normalize_strike, log_trade_to_ledger

# --- Module-level storage for orders ---
ORDER_QUEUE = []

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrderManager")


async def generate_and_queue_orders(config: dict):
    """
    Generates trading strategies based on market data and API predictions,
    and queues them for later execution.
    """
    logger.info("--- Starting order generation and queuing process ---")
    global ORDER_QUEUE
    ORDER_QUEUE.clear()  # Clear queue at the start of each day

    try:
        logger.info("Step 1: Running data pull...")
        if not run_data_pull(config):
            logger.error("Data pull failed. Aborting order generation."); return
        logger.info("Data pull complete.")

        logger.info("Step 2: Fetching predictions from API...")
        predictions = send_data_and_get_prediction(config)
        if not predictions:
            send_pushover_notification(config.get('notifications', {}), "Order Generation Failure", "Failed to get predictions from API.")
            return
        logger.info(f"Successfully received predictions from API: {predictions}")

        ib = IB()
        try:
            conn_settings = config.get('connection', {})
            await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000))
            logger.info("Connected to IB for signal generation.")

            logger.info("Step 2.5: Generating structured signals from predictions...")
            signals = await generate_signals(ib, predictions, config)
            logger.info(f"Generated {len(signals)} signals: {signals}")
            if not signals:
                logger.info("No actionable trading signals were generated. Concluding.")
                return

            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
            if not active_futures:
                raise ConnectionError("Could not find any active futures contracts.")

            for signal in signals:
                future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future:
                    logger.warning(f"No active future for signal month {signal.get('contract_month')}."); continue

                logger.info(f"Processing signal for {future.localSymbol}")
                ticker = ib.reqMktData(future, '', False, False)
                await asyncio.sleep(1)
                price = ticker.marketPrice()
                ib.cancelMktData(future)
                if util.isNan(price):
                    logger.error(f"Failed to get market price for {future.localSymbol}."); continue

                price = normalize_strike(price)
                chain = await build_option_chain(ib, future)
                if not chain: continue

                define_func = define_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else define_volatility_strategy
                strategy_def = define_func(config, signal, chain, price, future)

                if strategy_def:
                    order_objects = await create_combo_order_object(ib, config, strategy_def)
                    if order_objects:
                        ORDER_QUEUE.append(order_objects)
                        logger.info(f"Successfully queued order for {future.localSymbol}.")
        finally:
            if ib.isConnected():
                ib.disconnect()

        logger.info(f"--- Order generation complete. {len(ORDER_QUEUE)} orders queued. ---")
        send_pushover_notification(config.get('notifications', {}), "Orders Queued", f"{len(ORDER_QUEUE)} strategies have been generated and are ready for placement.")

    except Exception as e:
        msg = f"A critical error occurred during order generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Generation CRITICAL", msg)


async def place_queued_orders(config: dict):
    """
    Connects to IB and places all orders currently in the queue.
    """
    logger.info(f"--- Placing {len(ORDER_QUEUE)} queued orders ---")
    if not ORDER_QUEUE:
        logger.info("Order queue is empty. Nothing to place.")
        return

    ib = IB()
    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000))
        logger.info("Connected to IB for order placement.")

        placed_trades = []
        for contract, order in ORDER_QUEUE:
            trade = place_order(ib, contract, order)
            placed_trades.append(trade)
            await asyncio.sleep(0.5) # Small delay between placements

        logger.info(f"--- Finished placing {len(placed_trades)} orders. ---")
        send_pushover_notification(config.get('notifications', {}), "Orders Placed", f"{len(placed_trades)} DAY orders have been submitted to the exchange.")
        ORDER_QUEUE.clear()

    except Exception as e:
        msg = f"A critical error occurred during order placement: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Placement CRITICAL", msg)
    finally:
        if ib.isConnected():
            ib.disconnect()

async def close_all_open_positions(config: dict):
    """
    Fetches all open positions and places opposing orders to close them.
    """
    logger.info("--- Initiating end-of-day position closing ---")
    ib = IB()
    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000))
        logger.info("Connected to IB for closing positions.")

        positions = await ib.reqPositionsAsync()
        if not positions:
            logger.info("No open positions to close.")
            return

        logger.info(f"Found {len(positions)} positions to close.")
        closed_count = 0
        for pos in positions:
            if pos.position == 0: continue
            action = 'SELL' if pos.position > 0 else 'BUY'
            order = MarketOrder(action, abs(pos.position))
            trade = place_order(ib, pos.contract, order)
            if trade.orderStatus.status == OrderStatus.Filled:
                log_trade_to_ledger(trade, "Daily Close")
                closed_count += 1
            await asyncio.sleep(1) # Pace the closing orders

        logger.info(f"--- Finished closing {closed_count}/{len(positions)} positions ---")
        send_pushover_notification(config.get('notifications', {}), "Positions Closed", f"Attempted to close {len(positions)} positions. See logs for details.")

    except Exception as e:
        msg = f"A critical error occurred while closing positions: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Position Closing CRITICAL", msg)
    finally:
        if ib.isConnected():
            ib.disconnect()

async def cancel_all_open_orders(config: dict):
    """
    Fetches and cancels all open (non-filled) orders.
    """
    logger.info("--- Canceling any remaining open DAY orders ---")
    ib = IB()
    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000))
        logger.info("Connected to IB for canceling open orders.")

        open_trades = ib.reqOpenOrders()
        if not open_trades:
            logger.info("No open orders found to cancel.")
            return

        logger.info(f"Found {len(open_trades)} open orders to cancel.")
        for trade in open_trades:
            ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order ID {trade.order.orderId}.")
            await asyncio.sleep(0.2)

        logger.info(f"--- Finished canceling {len(open_trades)} orders ---")
        send_pushover_notification(config.get('notifications', {}), "Open Orders Canceled", f"Canceled {len(open_trades)} unfilled DAY orders.")

    except Exception as e:
        msg = f"A critical error occurred while canceling orders: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Cancellation CRITICAL", msg)
    finally:
        if ib.isConnected():
            ib.disconnect()