"""Manages the entire lifecycle of trading orders.

This module is responsible for generating, queuing, placing, and canceling
orders, as well as closing open positions at the end of the trading day. It
acts as the central hub for all order-related activities, decoupling the
strategy definition from the execution timing.
"""
import asyncio
import json
import logging
import os
import random
import time
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
from trading_bot.utils import log_trade_to_ledger, log_order_event

# --- Constants ---
ORDER_QUEUE_DIR = "order_queue"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrderManager")


def _contract_to_dict(contract: Contract) -> dict:
    """Converts an IB-insync Contract object to a serializable dictionary."""
    # A set of attributes to include in the dictionary
    attrs = {
        'conId', 'symbol', 'secType', 'lastTradeDateOrContractMonth',
        'strike', 'right', 'multiplier', 'exchange', 'currency', 'localSymbol',
        'tradingClass'
    }
    contract_dict = {attr: getattr(contract, attr) for attr in attrs if hasattr(contract, attr)}

    # Handle combo legs for Bag contracts
    if contract.secType == 'BAG' and contract.comboLegs:
        contract_dict['comboLegs'] = [
            {
                'conId': leg.conId, 'ratio': leg.ratio, 'action': leg.action,
                'exchange': leg.exchange
            }
            for leg in contract.comboLegs
        ]
    return contract_dict


def _order_to_dict(order: Order) -> dict:
    """Converts an IB-insync Order object to a serializable dictionary."""
    attrs = {
        'orderId', 'action', 'totalQuantity', 'orderType', 'lmtPrice', 'tif'
    }
    return {attr: getattr(order, attr) for attr in attrs if hasattr(order, attr)}


async def generate_and_queue_orders(config: dict):
    """
    Generates trading strategies, serializes them as JSON, and saves them
    as individual files in the order queue directory for the 'listener'
    process to pick up.
    """
    logger.info("--- Starting order generation and queuing process ---")
    os.makedirs(ORDER_QUEUE_DIR, exist_ok=True)
    queued_orders_count = 0

    try:
        logger.info("Step 1: Running data pull...")
        if not run_data_pull(config):
            logger.warning("Data pull failed. Using most recent data file as fallback.")
        else:
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
            await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
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

                logger.info(f"Requesting market price for {future.localSymbol}...")
                ticker = ib.reqMktData(future, '', False, False)

                price = float('nan')
                try:
                    # Wait for the ticker to update with a valid price, with a timeout
                    start_time = time.time()
                    while util.isNan(ticker.marketPrice()):
                        await asyncio.sleep(0.1) # Use asyncio's sleep to allow ib_insync to process messages
                        if (time.time() - start_time) > 30: # 30-second timeout
                            logger.error(f"Timeout waiting for market price for {future.localSymbol}.")
                            logger.error(f"Ticker data received: {ticker}")
                            break # Exit loop, price remains NaN
                    else: # Loop completed without break
                        market_price = ticker.marketPrice()
                        if not util.isNan(market_price):
                            price = market_price
                            logger.info(f"Successfully received market price for {future.localSymbol}: {price}")
                        else:
                            logger.error(f"Received invalid NaN price for {future.localSymbol}. Ticker: {ticker}")
                finally:
                    ib.cancelMktData(future)
                if util.isNan(price):
                    continue

                chain = await build_option_chain(ib, future)
                if not chain: continue

                define_func = define_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else define_volatility_strategy
                strategy_def = define_func(config, signal, chain, price, future)

                if strategy_def:
                    order_objects = await create_combo_order_object(ib, config, strategy_def)
                    if order_objects:
                        # Serialize the contract and order objects
                        contract_dict = _contract_to_dict(order_objects[0])
                        order_dict = _order_to_dict(order_objects[1])
                        order_data = {"contract": contract_dict, "order": order_dict}

                        # Save to a unique file in the queue directory
                        timestamp = int(time.time() * 1000)
                        filename = os.path.join(ORDER_QUEUE_DIR, f"order_{timestamp}.json")
                        with open(filename, 'w') as f:
                            json.dump(order_data, f, indent=4)

                        queued_orders_count += 1
                        logger.info(f"Successfully queued order for {future.localSymbol} to file: {filename}")
        finally:
            if ib.isConnected():
                ib.disconnect()

        logger.info(f"--- Order generation complete. {queued_orders_count} orders queued. ---")

        # Notification is simplified as the listener will provide detailed execution reports
        message = f"Order generation cycle complete. {queued_orders_count} new strategies were identified and queued for placement."
        send_pushover_notification(config.get('notifications', {}), f"{queued_orders_count} Orders Queued", message)

    except Exception as e:
        msg = f"A critical error occurred during order generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Generation CRITICAL", msg)

async def close_all_open_positions(config: dict):
    """
    Fetches all open positions, places opposing orders to close them, and
    sends a detailed notification of the outcome.
    """
    logger.info("--- Initiating end-of-day position closing ---")
    ib = IB()
    closed_position_details = []
    failed_closes = []

    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
        logger.info("Connected to IB for closing positions.")

        positions = await ib.reqPositionsAsync()
        if not positions:
            logger.info("No open positions to close.")
            send_pushover_notification(config.get('notifications', {}), "Position Closing", "No open positions were found to close.")
            return

        symbol_to_close = config.get('symbol', 'KC')
        relevant_positions = [p for p in positions if p.contract.symbol == symbol_to_close and p.position != 0]

        if not relevant_positions:
            logger.info("No relevant positions to close for the target symbol.")
            return

        logger.info(f"Found {len(relevant_positions)} positions for symbol '{symbol_to_close}' to close.")

        for pos in relevant_positions:
            pos.contract.exchange = config.get('exchange')
            if not pos.contract.exchange:
                logger.error(f"Exchange not found in config. Cannot close position for {pos.contract.localSymbol}")
                failed_closes.append(f"{pos.contract.localSymbol}: Missing exchange in config")
                continue

            action = 'SELL' if pos.position > 0 else 'BUY'
            order = MarketOrder(action, abs(pos.position))
            trade = place_order(ib, pos.contract, order)

            await asyncio.sleep(2)  # Wait for order status updates

            if trade.orderStatus.status == OrderStatus.Filled and trade.fills:
                fill = trade.fills[0]
                realized_pnl = fill.commissionReport.realizedPNL
                fill_price = fill.execution.avgPrice
                log_trade_to_ledger(trade, "Daily Close")
                closed_position_details.append({
                    "symbol": pos.contract.localSymbol,
                    "action": action,
                    "quantity": pos.position,
                    "price": fill_price,
                    "pnl": realized_pnl
                })
                logger.info(f"Successfully closed position for {pos.contract.localSymbol}.")
            else:
                logger.warning(f"Failed to close position for {pos.contract.localSymbol}. Final status: {trade.orderStatus.status}")
                failed_closes.append(f"{pos.contract.localSymbol}: Failed with status {trade.orderStatus.status}")

            await asyncio.sleep(1)

        logger.info(f"--- Finished closing process. Success: {len(closed_position_details)}, Failed: {len(failed_closes)} ---")

        # --- Build and send detailed notification ---
        message_parts = []
        total_pnl = 0
        if closed_position_details:
            message_parts.append(f"<b>✅ Successfully closed {len(closed_position_details)} positions:</b>")
            for detail in closed_position_details:
                pnl_str = f"${detail['pnl']:,.2f}"
                pnl_color = "green" if detail['pnl'] >= 0 else "red"
                message_parts.append(
                    f"  - <b>{detail['symbol']}</b>: {detail['action']} {abs(detail['quantity'])} "
                    f"@ ${detail['price']:.2f}. P&L: <font color='{pnl_color}'>{pnl_str}</font>"
                )
                total_pnl += detail.get('pnl', 0)
            message_parts.append(f"<b>Total Realized P&L: ${total_pnl:,.2f}</b>")


        if failed_closes:
            message_parts.append(f"\n<b>❌ Failed to close {len(failed_closes)} positions:</b>")
            message_parts.extend([f"  - {reason}" for reason in failed_closes])

        if not message_parts:
            message = "No positions were processed for closing."
        else:
            message = "\n".join(message_parts)

        notification_title = f"Position Close Report: P&L ${total_pnl:,.2f}"
        send_pushover_notification(config.get('notifications', {}), notification_title, message)

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
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
        logger.info("Connected to IB for canceling open orders.")

        # Use ib.reqAllOpenOrdersAsync() to get all open orders from the broker
        open_trades = await ib.reqAllOpenOrdersAsync()
        if not open_trades:
            logger.info("No open orders found to cancel.")
            return

        logger.info(f"Found {len(open_trades)} open orders to cancel.")
        for trade in list(open_trades): # Iterate over a copy
            ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order ID {trade.order.orderId}.")
            await asyncio.sleep(0.2)

        logger.info(f"--- Finished canceling {len(open_trades)} orders ---")
        # Create a detailed notification message
        if open_trades:
            summary_items = [f"  - {trade.contract.localSymbol} ({trade.order.action} {trade.order.totalQuantity}) ID: {trade.order.orderId}" for trade in open_trades]
            message = f"<b>Canceled {len(open_trades)} unfilled DAY orders:</b>\n" + "\n".join(summary_items)
        else:
            message = "No open orders were found to cancel."

        send_pushover_notification(config.get('notifications', {}), "Open Orders Canceled", message)

    except Exception as e:
        msg = f"A critical error occurred while canceling orders: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Cancellation CRITICAL", msg)
    finally:
        if ib.isConnected():
            ib.disconnect()

