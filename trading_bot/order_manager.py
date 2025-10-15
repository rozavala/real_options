"""Manages the generation and queuing of trading signals.

This module is responsible for orchestrating the data pull, fetching predictions
from an external API, and generating high-level trading signals based on those
predictions. The generated signals (not complete orders) are then serialized
and placed into a queue directory for the 'listener' process to pick up and
execute.
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

# Note: The ib_interface imports are now handled by the listener process
from trading_bot.signal_generator import generate_signals

# --- Constants ---
ORDER_QUEUE_DIR = "order_queue"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrderManager")


async def generate_and_queue_orders(config: dict):
    """
    Generates trading signals based on API predictions and queues them as
    JSON files for the listener process to execute.
    """
    logger.info("--- Starting signal generation and queuing process ---")
    os.makedirs(ORDER_QUEUE_DIR, exist_ok=True)
    queued_signals_count = 0

    try:
        logger.info("Step 1: Running data pull...")
        if not run_data_pull(config):
            logger.warning("Data pull failed. Using most recent data file as fallback.")
        else:
            logger.info("Data pull complete.")

        logger.info("Step 2: Fetching predictions from API...")
        predictions = send_data_and_get_prediction(config)
        if not predictions:
            send_pushover_notification(config.get('notifications', {}), "Signal Generation Failure", "Failed to get predictions from API.")
            return
        logger.info(f"Successfully received predictions from API: {predictions}")

        ib = IB()
        try:
            conn_settings = config.get('connection', {})
            await ib.connectAsync(
                host=conn_settings.get('host', '127.0.0.1'),
                port=conn_settings.get('port', 7497),
                clientId=random.randint(200, 2000),
                timeout=30
            )
            logger.info("Connected to IB for signal generation.")

            logger.info("Step 3: Generating structured signals from predictions...")
            signals = await generate_signals(ib, predictions, config)
            logger.info(f"Generated {len(signals)} signals: {signals}")

            if not signals:
                logger.info("No actionable trading signals were generated. Concluding.")
                return

            # Queue each generated signal into a file
            for signal in signals:
                timestamp = int(time.time() * 1000)
                # Using a more descriptive filename
                filename = os.path.join(ORDER_QUEUE_DIR, f"signal_{signal['contract_month']}_{timestamp}.json")
                with open(filename, 'w') as f:
                    json.dump(signal, f, indent=4)
                queued_signals_count += 1
                logger.info(f"Successfully queued signal for {signal['contract_month']} to file: {filename}")

        finally:
            if ib.isConnected():
                ib.disconnect()

        logger.info(f"--- Signal generation complete. {queued_signals_count} signals queued. ---")
        message = (
            f"Signal generation cycle complete. {queued_signals_count} new trading "
            f"signals were identified and queued for execution."
        )
        send_pushover_notification(config.get('notifications', {}), f"{queued_signals_count} Signals Queued", message)

    except Exception as e:
        msg = f"A critical error occurred during signal generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Signal Generation CRITICAL", msg)


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
            # The contract from reqPositionsAsync() might be incomplete.
            # We must explicitly set the exchange for the closing order.
            pos.contract.exchange = config.get('exchange')
            if not pos.contract.exchange:
                logger.error(f"Exchange not found in config. Cannot close position for {pos.contract.localSymbol}")
                failed_closes.append(f"{pos.contract.localSymbol}: Missing exchange in config")
                continue

            action = 'SELL' if pos.position > 0 else 'BUY'
            order = MarketOrder(action, abs(pos.position))
            # Use a generic place_order function from ib_interface
            from trading_bot.ib_interface import place_order
            trade = place_order(ib, pos.contract, order)

            # Wait for the trade to be filled to report P&L
            await asyncio.sleep(2)  # Simple wait for fill status

            if trade.isDone() and trade.orderStatus.status == OrderStatus.Filled and trade.fills:
                fill = trade.fills[0]
                realized_pnl = fill.commissionReport.realizedPNL if fill.commissionReport else 0.0
                fill_price = fill.execution.avgPrice
                # Log the closed trade to the ledger
                from trading_bot.utils import log_trade_to_ledger
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
                logger.warning(f"Failed to confirm close for {pos.contract.localSymbol}. Final status: {trade.orderStatus.status}")
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

        open_trades = await ib.reqAllOpenOrdersAsync()
        if not open_trades:
            logger.info("No open orders found to cancel.")
            return

        logger.info(f"Found {len(open_trades)} open orders to cancel.")
        for trade in list(open_trades):
            ib.cancelOrder(trade.order)
            logger.info(f"Cancelled order ID {trade.order.orderId}.")
            await asyncio.sleep(0.2)

        logger.info(f"--- Finished canceling {len(open_trades)} orders ---")
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