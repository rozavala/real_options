"""Manages the entire lifecycle of trading orders.

This module is responsible for generating, queuing, placing, and canceling
orders, as well as closing open positions at the end of the trading day. It
acts as the central hub for all order-related activities, decoupling the
strategy definition from the execution timing.
"""
import asyncio
import logging
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
                        ORDER_QUEUE.append(order_objects)
                        logger.info(f"Successfully queued order for {future.localSymbol}.")
        finally:
            if ib.isConnected():
                ib.disconnect()

        logger.info(f"--- Order generation complete. {len(ORDER_QUEUE)} orders queued. ---")

        # Create a detailed notification message
        if ORDER_QUEUE:
            summary_items = []
            for contract, order in ORDER_QUEUE:
                summary = (
                    f"<b>{contract.localSymbol}</b> ({order.action} {order.totalQuantity}): "
                    f"LMT @ ${order.lmtPrice}, "
                    f"{len(contract.comboLegs)} legs"
                )
                summary_items.append(summary)
            message = f"<b>{len(ORDER_QUEUE)} strategies generated and queued:</b>\n" + "\n".join(summary_items)
        else:
            message = "No new orders were generated or queued."

        send_pushover_notification(config.get('notifications', {}), "Orders Queued", message)

    except Exception as e:
        msg = f"A critical error occurred during order generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Generation CRITICAL", msg)


async def place_queued_orders(config: dict):
    """
    Connects to IB, places all queued orders, and monitors them until they
    are filled or reach a terminal state, ensuring all fills are logged.
    """
    logger.info(f"--- Placing and monitoring {len(ORDER_QUEUE)} queued orders ---")
    if not ORDER_QUEUE:
        logger.info("Order queue is empty. Nothing to place.")
        return

    ib = IB()
    live_orders = {} # Dictionary to track order status by orderId

    # --- Event Handlers ---
    def on_order_status(trade: Trade):
        """Handles order status updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        if order_id in live_orders and live_orders[order_id]['status'] != status:
            live_orders[order_id]['status'] = status
            logger.info(f"Order {order_id} status update: {status}")
            log_order_event(trade, status)

    def on_exec_details(trade: Trade, fill: Fill):
        """Handles trade execution details and logs the trade."""
        order_id = trade.order.orderId
        if order_id in live_orders and not live_orders[order_id].get('is_filled', False):
            logger.info(f"Order {order_id} FILLED. Logging to trade ledger.")
            # Log the trade to the ledger. The fill object has all necessary details.
            log_trade_to_ledger(trade, "Strategy Execution")
            live_orders[order_id]['is_filled'] = True # Mark as filled to avoid duplicate logging

    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
        logger.info("Connected to IB for order placement and monitoring.")

        # Attach event handlers
        ib.orderStatusEvent += on_order_status
        ib.execDetailsEvent += on_exec_details

        placed_trades_summary = []
        for contract, order in ORDER_QUEUE:
            trade = place_order(ib, contract, order)
            # Initial log upon submission
            log_order_event(trade, trade.orderStatus.status)
            live_orders[trade.order.orderId] = {'trade': trade, 'status': trade.orderStatus.status, 'is_filled': False}
            placed_trades_summary.append(f"  - {trade.contract.localSymbol} ({trade.order.action} {trade.order.totalQuantity}) ID: {trade.order.orderId}")
            await asyncio.sleep(0.5)

        if placed_trades_summary:
            message = f"<b>{len(live_orders)} DAY orders submitted, now monitoring:</b>\n" + "\n".join(placed_trades_summary)
            send_pushover_notification(config.get('notifications', {}), "Orders Placed & Monitored", message)

        # --- Monitoring Loop ---
        start_time = time.time()
        monitoring_duration = 900  # 15 minutes
        logger.info(f"Monitoring orders for up to {monitoring_duration / 60} minutes...")
        while time.time() - start_time < monitoring_duration:
            if not live_orders: break # Exit if all orders were somehow removed
            all_done = all(
                details['status'] in OrderStatus.DoneStates
                for details in live_orders.values()
            )
            if all_done:
                logger.info("All orders have reached a terminal state. Concluding monitoring.")
                break
            await asyncio.sleep(15) # Check status every 15 seconds
        else:
            logger.warning("Monitoring loop timed out. Some orders may not be in a terminal state.")

        # --- Cancellation and Final Reporting ---
        filled_orders = []
        cancelled_orders = []
        for order_id, details in live_orders.items():
            trade = details['trade']
            if details['status'] == OrderStatus.Filled:
                filled_orders.append(f"  - <b>{trade.contract.localSymbol}</b> ({trade.order.action})")
            elif details['status'] not in OrderStatus.DoneStates:
                logger.info(f"Order {order_id} did not fill. Cancelling it now.")
                ib.cancelOrder(trade.order)
                cancelled_orders.append(f"  - <b>{trade.contract.localSymbol}</b> ({trade.order.action}) ID: {order_id}")
                await asyncio.sleep(0.2) # Small delay for cancellation request

        # Build and send the final summary notification
        message_parts = []
        if filled_orders:
            message_parts.append(f"<b>✅ {len(filled_orders)} orders were successfully filled:</b>")
            message_parts.extend(filled_orders)
        if cancelled_orders:
            message_parts.append(f"\n<b>❌ {len(cancelled_orders)} unfilled orders were cancelled:</b>")
            message_parts.extend(cancelled_orders)

        if not message_parts:
            summary_message = "Order monitoring complete. No orders were filled or needed cancellation."
        else:
            summary_message = "\n".join(message_parts)

        send_pushover_notification(config.get('notifications', {}), "Order Monitoring Complete", summary_message)

        ORDER_QUEUE.clear()
        logger.info("--- Finished monitoring and cleanup. ---")

    except Exception as e:
        msg = f"A critical error occurred during order placement/monitoring: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Placement CRITICAL", msg)
    finally:
        if ib.isConnected():
            # It's good practice to remove the handlers when done
            ib.orderStatusEvent -= on_order_status
            ib.execDetailsEvent -= on_exec_details
            ib.disconnect()
            logger.info("Disconnected from IB.")

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

