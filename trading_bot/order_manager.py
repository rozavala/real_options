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
from collections import defaultdict
from ib_insync import *

from config_loader import load_config
from coffee_factors_data_pull_new import main as run_data_pull
from trading_bot.inference import get_model_predictions
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


async def generate_and_execute_orders(config: dict):
    """
    Generates, queues, and immediately places orders.
    This ensures that order placement isn't skipped if generation takes
    longer than the gap between scheduled times.
    """
    logger.info(">>> Starting combined task: Generate and Execute Orders <<<")
    await generate_and_queue_orders(config)

    # Only proceed to placement if orders were actually queued
    if ORDER_QUEUE:
        await place_queued_orders(config)
    else:
        logger.info("No orders were queued. Skipping placement step.")
    logger.info(">>> Combined task 'Generate and Execute Orders' complete <<<")


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
        data_df = run_data_pull(config)
        if data_df is None:
            logger.error("Data pull failed. Cannot proceed with order generation.")
            send_pushover_notification(config.get('notifications', {}), "Order Generation Failure", "Data pull failed. Aborting.")
            return
        logger.info("Data pull complete.")

        logger.info("Step 2: Running local model inference...")
        predictions = get_model_predictions(data_df)
        if not predictions:
            send_pushover_notification(config.get('notifications', {}), "Order Generation Failure", "Failed to get predictions from local model.")
            return
        logger.info(f"Successfully received predictions from local model: {predictions}")

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
                    await asyncio.sleep(2) # Give it a moment to populate
                    # Wait for the ticker to update with a valid bid/ask spread
                    start_time = time.time()
                    while ticker.bid <= 0 or ticker.ask <= 0 or util.isNan(ticker.bid) or util.isNan(ticker.ask):
                        await asyncio.sleep(0.1)
                        if (time.time() - start_time) > 30:
                            logger.error(f"Timeout waiting for valid bid/ask for {future.localSymbol}.")
                            logger.error(f"Ticker data received: {ticker}")
                            break

                    # Fallback Price Discovery Logic
                    if ticker.bid > 0 and ticker.ask > 0:
                        price = (ticker.bid + ticker.ask) / 2
                        logger.info(f"Using bid/ask midpoint for {future.localSymbol}: {price}")
                    elif not util.isNan(ticker.last):
                        price = ticker.last
                        logger.info(f"Using last price for {future.localSymbol}: {price}")
                    elif not util.isNan(ticker.close):
                        price = ticker.close
                        logger.info(f"Using close price for {future.localSymbol}: {price}")
                    else:
                        logger.error(f"Could not determine a valid price for {future.localSymbol}. Ticker: {ticker}")
                finally:
                    ib.cancelMktData(future)

                if util.isNan(price):
                    logger.warning(f"Skipping strategy for {future.localSymbol} due to missing price.")
                    continue

                chain = await build_option_chain(ib, future)
                if not chain:
                    logger.warning(f"Skipping {future.localSymbol}: Failed to build option chain.")
                    continue

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

        # --- Create a detailed notification message ---

        # 1. Summarize the API signals
        signal_summary_parts = []
        if signals:
            for signal in signals:
                future_month = signal.get("contract_month", "N/A")
                if signal['prediction_type'] == 'DIRECTIONAL':
                    direction = signal.get("direction", "N/A")
                    reason = signal.get("reason", "N/A")
                    signal_summary_parts.append(f"  - <b>{future_month}</b>: {direction} ({reason})")
                elif signal['prediction_type'] == 'VOLATILITY':
                    vol_level = signal.get("level", "N/A")
                    signal_summary_parts.append(f"  - <b>{future_month}</b>: {vol_level} Volatility signal")

        signal_summary = "<b>Signals from Two-Brain Model:</b>\n" + "\n".join(signal_summary_parts)

        # 2. Summarize the queued orders
        order_summary_parts = []
        if ORDER_QUEUE:
            for contract, order in ORDER_QUEUE:
                # Determine the strategy name from the contract symbol if possible
                strategy_name = "Strategy" # Default
                if "PUT" in contract.localSymbol and "CALL" in contract.localSymbol:
                     strategy_name = "IRON_CONDOR" if "IRON" in contract.localSymbol else "STRADDLE/STRANGLE" # Placeholder
                elif "PUT" in contract.localSymbol:
                    strategy_name = "BEAR_PUT_SPREAD"
                elif "CALL" in contract.localSymbol:
                    strategy_name = "BULL_CALL_SPREAD"

                price_info = f"LMT @ ${order.lmtPrice:.2f}" if order.orderType == "LMT" else "MKT"
                order_summary_parts.append(
                    f"  - <b>{strategy_name}</b> for {contract.localSymbol}: {order.action} {order.totalQuantity} @ {price_info}"
                )
            order_summary = f"\n<b>{len(ORDER_QUEUE)} strategies generated and queued:</b>\n" + "\n".join(order_summary_parts)
        else:
            order_summary = "\nNo new orders were generated or queued based on these signals."

        # 3. Combine and send
        message = signal_summary + order_summary
        send_pushover_notification(config.get('notifications', {}), "Trading Orders Queued", message)

    except Exception as e:
        msg = f"A critical error occurred during order generation: {e}\n{traceback.format_exc()}"
        logger.critical(msg)
        send_pushover_notification(config.get('notifications', {}), "Order Generation CRITICAL", msg)


async def _get_market_data_for_leg(ib: IB, leg: ComboLeg) -> str:
    """
    Fetches comprehensive market data for a single contract leg with a timeout.
    Returns a formatted string for logging and notifications.
    """
    try:
        contract = await ib.reqContractDetailsAsync(Contract(conId=leg.conId))
        if not contract:
            logger.warning(f"Could not resolve contract for leg conId {leg.conId}")
            return f"Leg conId {leg.conId}: N/A"

        ticker = ib.reqMktData(contract[0].contract, '', False, False)
        try:
            await asyncio.sleep(2) # Give it a moment to populate

            # Wait for the ticker to update with a valid price, with a timeout
            start_time = time.time()
            while util.isNan(ticker.bid) and util.isNan(ticker.ask) and util.isNan(ticker.last):
                await asyncio.sleep(0.1)
                if (time.time() - start_time) > 15: # 15-second timeout
                    logger.error(f"Timeout waiting for market data for {contract[0].contract.localSymbol}.")
                    # ib.cancelMktData(ticker.contract) # Handled in finally
                    return f"Leg {contract[0].contract.localSymbol}: Timeout"

            # Fetch all data points
            leg_symbol = contract[0].contract.localSymbol
            leg_bid = ticker.bid if not util.isNan(ticker.bid) else "N/A"
            leg_ask = ticker.ask if not util.isNan(ticker.ask) else "N/A"
            leg_vol = ticker.volume if not util.isNan(ticker.volume) else "N/A"
            leg_last = ticker.last if not util.isNan(ticker.last) else "N/A"
        finally:
            ib.cancelMktData(ticker.contract)
        
        # Return a formatted string
        return f"Leg {leg_symbol}: {leg_bid}x{leg_ask}, V:{leg_vol}, L:{leg_last}"

    except Exception as e:
        logger.error(f"Error fetching market data for leg {leg.conId}: {e}")
        return f"Leg conId {leg.conId}: Error"


async def _handle_and_log_fill(ib: IB, trade: Trade, fill: Fill, combo_id: int, position_uuid: str):
    """
    Handles the logic for processing a trade fill, ensuring the contract
    details are complete before logging.
    """
    try:
        if isinstance(fill.contract, Bag):
            logger.info(f"Skipping ledger log for summary combo fill with conId: {fill.contract.conId}")
            return

        detailed_contract = Contract(conId=fill.contract.conId)
        await ib.qualifyContractsAsync(detailed_contract)

        corrected_fill = Fill(
            contract=detailed_contract,
            execution=fill.execution,
            commissionReport=fill.commissionReport,
            time=fill.time
        )

        if isinstance(trade.contract, Bag):
            for leg in trade.contract.comboLegs:
                if leg.conId == fill.contract.conId:
                    logger.info(f"Matched fill conId {fill.contract.conId} to leg in {trade.contract.localSymbol}.")
                    break
            else:
                logger.warning(f"Could not match fill conId {fill.contract.conId} to any leg in {trade.contract.localSymbol}.")

        # This now calls the async version of the function, passing the unique position ID
        await log_trade_to_ledger(ib, trade, "Strategy Execution", specific_fill=corrected_fill, combo_id=combo_id, position_id=position_uuid)
        logger.info(f"Successfully logged fill for {detailed_contract.localSymbol} (Order ID: {trade.order.orderId}, Position ID: {position_uuid})")

    except Exception as e:
        logger.error(f"Error processing and logging fill for order {trade.order.orderId}: {e}\n{traceback.format_exc()}")


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
        """
        Handles trade execution details for each leg, logs the trade, and
        tracks the overall fill status of the combo order.
        """
        order_id = trade.order.orderId
        if order_id in live_orders:
            leg_con_id = fill.contract.conId
            order_details = live_orders[order_id]

            if leg_con_id in order_details['filled_legs']:
                logger.info(f"Duplicate fill event for leg {leg_con_id} in order {order_id}. Ignoring.")
                return

            parent_trade = order_details['trade']
            combo_perm_id = parent_trade.order.permId
            # The unique position ID is the orderRef from the PARENT order object.
            position_uuid = parent_trade.order.orderRef

            logger.info(f"Fill received for leg {leg_con_id} in order {order_id}. Processing with Position ID {position_uuid}.")
            asyncio.create_task(_handle_and_log_fill(ib, trade, fill, combo_perm_id, position_uuid))

            order_details['filled_legs'].add(leg_con_id)
            order_details['fill_prices'][leg_con_id] = fill.execution.avgPrice

            # --- Check if all legs of the combo are now filled ---
            if len(order_details['filled_legs']) == order_details['total_legs']:
                logger.info(f"All {order_details['total_legs']} legs of combo order {order_id} are now filled.")
                order_details['is_filled'] = True # Mark the entire order as filled

    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
        logger.info("Connected to IB for order placement and monitoring.")

        # Attach event handlers
        ib.orderStatusEvent += on_order_status
        ib.execDetailsEvent += on_exec_details

        placed_trades_summary = []
        tuning = config.get('strategy_tuning', {})
        adaptive_interval = tuning.get('adaptive_step_interval_seconds', 15)
        adaptive_pct = tuning.get('adaptive_step_percentage', 0.05)
        
        # --- PLACEMENT LOOP (ENHANCED LOGGING & NOTIFICATIONS) ---
        for contract, order in ORDER_QUEUE:
            
            # --- 1. Get BAG (Spread) Market Data with a robust polling loop ---
            bag_ticker = ib.reqMktData(contract, '', False, False)
            try:
                await asyncio.sleep(2) # Give it a moment to populate
                start_time = time.time()
                while util.isNan(bag_ticker.bid) and util.isNan(bag_ticker.ask) and util.isNan(bag_ticker.last):
                    await asyncio.sleep(0.1)
                    if (time.time() - start_time) > 15: # 15-second timeout
                        logger.error(f"Timeout waiting for market data for BAG {contract.localSymbol}.")
                        break

                bag_bid = bag_ticker.bid if not util.isNan(bag_ticker.bid) else "N/A"
                bag_ask = bag_ticker.ask if not util.isNan(bag_ticker.ask) else "N/A"
                bag_vol = bag_ticker.volume if not util.isNan(bag_ticker.volume) else "N/A"
                bag_last = bag_ticker.last if not util.isNan(bag_ticker.last) else "N/A"
                bag_last_time = bag_ticker.time.strftime('%H:%M:%S') if bag_ticker.time else "N/A"
            finally:
                ib.cancelMktData(contract) # Cleanup
            
            bag_state_str = f"BAG: {bag_bid}x{bag_ask}, V:{bag_vol}, L:{bag_last}@{bag_last_time}"

            # --- 2. Get LEG Market Data ---
            leg_state_strings = await asyncio.gather(
                *[_get_market_data_for_leg(ib, leg) for leg in contract.comboLegs]
            )
            leg_state_for_log = ", ".join(leg_state_strings)

            # --- 3. Create Enhanced Log and Place Order ---
            price_info_log = f"Limit: {order.lmtPrice:.2f}" if order.orderType == "LMT" else "Market Order"
            market_state_message = (
                f"Placing Order for {contract.localSymbol}. {price_info_log}. "
                f"{bag_state_str}. "
                f"LEGs: {leg_state_for_log}"
            )
            logger.info(market_state_message)

            # --- 4. Place the actual order ---
            trade = place_order(ib, contract, order)
            log_order_event(trade, trade.orderStatus.status, market_state_message)
            
            live_orders[trade.order.orderId] = {
                'trade': trade,
                'status': trade.orderStatus.status,
                'is_filled': False,
                'total_legs': len(contract.comboLegs) if isinstance(contract, Bag) else 1,
                'filled_legs': set(),
                'fill_prices': {}, # Stores fill prices by conId
                'last_update_time': time.time(),
                'adaptive_ceiling_price': getattr(order, 'adaptive_limit_price', None) # Store ceiling/floor
            }

            # --- 5. Build Notification String (ENHANCED FOR ALL DATA) ---
            price_info_notify = f"LMT: {order.lmtPrice:.2f}" if order.orderType == "LMT" else "MKT"
            leg_state_for_notify = "<br>    ".join(leg_state_strings) # HTML newline
            summary_line = (
                f"  - <b>{contract.localSymbol}</b> (ID: {trade.order.orderId})<br>"
                f"    {price_info_notify}<br>"
                f"    {bag_state_str}<br>"
                f"    {leg_state_for_notify}"
            )
            placed_trades_summary.append(summary_line)
            await asyncio.sleep(0.5) # Throttle order placement

        if placed_trades_summary:
            message = f"<b>{len(live_orders)} DAY orders submitted, now monitoring:</b>\n" + "\n".join(placed_trades_summary)
            send_pushover_notification(config.get('notifications', {}), "Orders Placed & Monitored", message)

        # --- Monitoring Loop ---
        start_time = time.time()
        monitoring_duration = 1800  # 30 minutes
        logger.info(f"Monitoring orders for up to {monitoring_duration / 60} minutes...")
        while time.time() - start_time < monitoring_duration:
            if not live_orders: break

            active_orders_count = 0
            for order_id, details in list(live_orders.items()):
                trade = details['trade']
                # Use the live status from the trade object to ensure accuracy
                status = trade.orderStatus.status

                if status in OrderStatus.DoneStates:
                    continue

                active_orders_count += 1

                # --- Adaptive "Walking" Logic ---
                if trade.order.orderType == 'LMT' and details['adaptive_ceiling_price'] is not None:
                    if (time.time() - details['last_update_time']) >= adaptive_interval:
                        current_limit = trade.order.lmtPrice
                        ceiling = details['adaptive_ceiling_price']
                        action = trade.order.action

                        new_price = None
                        step_amount = abs(current_limit) * adaptive_pct

                        if action == 'BUY':
                            if current_limit < ceiling:
                                # Increase price
                                proposed_price = current_limit + step_amount
                                # Round to 2 decimals and ensure we don't exceed ceiling
                                new_price = min(round(proposed_price, 2), ceiling)
                                # Ensure we actually move by at least 0.05 or 1 tick if calculated change is too small
                                if new_price <= current_limit and current_limit < ceiling:
                                     new_price = min(current_limit + 0.05, ceiling)

                        else: # SELL
                            if current_limit > ceiling: # Here 'ceiling' is actually floor
                                # Decrease price
                                proposed_price = current_limit - step_amount
                                new_price = max(round(proposed_price, 2), ceiling)
                                if new_price >= current_limit and current_limit > ceiling:
                                     new_price = max(current_limit - 0.05, ceiling)

                        if new_price is not None and new_price != current_limit:
                            logger.info(f"Adaptive Update for Order {order_id} ({trade.contract.localSymbol}): {current_limit} -> {new_price} (Cap/Floor: {ceiling})")
                            trade.order.lmtPrice = new_price
                            ib.placeOrder(trade.contract, trade.order) # Update the order
                            details['last_update_time'] = time.time()
                            log_order_event(trade, "Updated", f"Adaptive Walk: Price updated to {new_price}")
                        elif new_price is not None and new_price == current_limit:
                             # Reached cap/floor
                             pass

            if active_orders_count == 0:
                logger.info("All orders have reached a terminal state. Concluding monitoring.")
                break

            await asyncio.sleep(1) # Check every second, but update logic respects adaptive_interval
        else:
            logger.warning("Monitoring loop timed out. Some orders may not be in a terminal state.")

        # --- CANCELLATION AND FINAL REPORTING (ENHANCED LOGGING & NOTIFICATIONS) ---
        filled_orders = []
        cancelled_orders = []
        for order_id, details in live_orders.items():
            trade = details['trade']
            # Check the new 'is_filled' flag which confirms all legs are filled
            if details.get('is_filled', False):
                # --- Build Enhanced Fill Notification ---
                price_info = f"LMT: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "MKT"

                # Use the official avgFillPrice from the parent trade status, which is correct for both combos and single legs
                avg_fill_price = trade.orderStatus.avgFillPrice
                fill_line = (
                    f"  - <b>{trade.contract.localSymbol}</b> ({trade.order.action})<br>"
                    f"    {price_info} -> FILLED @ avg ${avg_fill_price:.2f}"
                )
                filled_orders.append(fill_line)
            
            elif details['status'] not in OrderStatus.DoneStates:
                
                # --- 1. Get FINAL BAG (Spread) Market Data with a robust polling loop ---
                final_bag_ticker = ib.reqMktData(trade.contract, '', False, False)
                try:
                    await asyncio.sleep(2) # Give it a moment to populate
                    start_time = time.time()
                    while util.isNan(final_bag_ticker.bid) and util.isNan(final_bag_ticker.ask) and util.isNan(final_bag_ticker.last):
                        await asyncio.sleep(0.1)
                        if (time.time() - start_time) > 15: # 15-second timeout
                            logger.error(f"Timeout waiting for final market data for BAG {trade.contract.localSymbol}.")
                            break

                    final_bag_bid = final_bag_ticker.bid if not util.isNan(final_bag_ticker.bid) else "N/A"
                    final_bag_ask = final_bag_ticker.ask if not util.isNan(final_bag_ticker.ask) else "N/A"
                    final_bag_vol = final_bag_ticker.volume if not util.isNan(final_bag_ticker.volume) else "N/A"
                    final_bag_last = final_bag_ticker.last if not util.isNan(final_bag_ticker.last) else "N/A"
                    final_bag_last_time = final_bag_ticker.time.strftime('%H:%M:%S') if final_bag_ticker.time else "N/A"
                finally:
                    ib.cancelMktData(trade.contract)
                
                final_bag_state = (
                    f"BAG: {final_bag_bid}x{final_bag_ask}, V:{final_bag_vol}, L:{final_bag_last}@{final_bag_last_time}"
                )

                # --- 2. Get FINAL LEG Market Data ---
                final_leg_state_strings = await asyncio.gather(
                    *[_get_market_data_for_leg(ib, leg) for leg in trade.contract.comboLegs]
                )
                final_leg_state_for_log = ", ".join(final_leg_state_strings)

                # --- 3. Create Enhanced Log and Cancel Order ---
                price_info_log = f"Original Limit: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "Original Order: MKT"
                log_message = (
                    f"Order {trade.order.orderId} ({trade.contract.localSymbol}) TIMED OUT. "
                    f"{price_info_log}. "
                    f"Final {final_bag_state}. "
                    f"Final LEGs: {final_leg_state_for_log}"
                )
                logger.warning(log_message)
                log_order_event(trade, "TimedOut", log_message)

                ib.cancelOrder(trade.order)

                # --- 4. Update Notification String (ENHANCED FOR ALL DATA) ---
                price_info_notify = f"LMT: {trade.order.lmtPrice:.2f}" if trade.order.orderType == "LMT" else "MKT"
                final_leg_state_for_notify = "<br>    ".join(final_leg_state_strings) # HTML newline
                cancel_line = (
                    f"  - <b>{trade.contract.localSymbol}</b> (ID: {order_id})<br>"
                    f"    {price_info_notify}<br>"
                    f"    Final {final_bag_state}<br>"
                    f"    Final {final_leg_state_for_notify}"
                )
                cancelled_orders.append(cancel_line)
                await asyncio.sleep(0.2)

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
            ib.orderStatusEvent -= on_order_status
            ib.execDetailsEvent -= on_exec_details
            ib.disconnect()
            logger.info("Disconnected from IB.")

import pandas as pd
from datetime import datetime, date
import os
from pandas.tseries.holiday import USFederalHolidayCalendar

def get_trade_ledger_df():
    """
    Reads and consolidates the main and archived trade ledgers for analysis.
    This function is now robust to historical ledgers that may be missing
    the 'position_id' column, using 'combo_id' as a fallback.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive_ledger')

    dataframes = []

    def load_and_prepare_ledger(file_path):
        """Loads a single ledger file and ensures it has a position_id."""
        df = pd.read_csv(file_path)
        if 'position_id' not in df.columns:
            logger.warning(f"File '{file_path}' is missing 'position_id'. Using 'combo_id' as fallback.")
            if 'combo_id' in df.columns:
                df['position_id'] = df['combo_id']
            else:
                logger.error(f"Cannot create fallback 'position_id' as 'combo_id' is also missing in '{file_path}'.")
                # Return an empty dataframe with the expected columns if a critical column is missing
                return pd.DataFrame(columns=['timestamp', 'position_id']) # Ensure it has minimal columns
        return df

    if os.path.exists(ledger_path):
        dataframes.append(load_and_prepare_ledger(ledger_path))

    if os.path.exists(archive_dir):
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        for file in archive_files:
            dataframes.append(load_and_prepare_ledger(file))

    if not dataframes:
        return pd.DataFrame()

    full_ledger = pd.concat(dataframes, ignore_index=True)

    # Ensure timestamp column exists and is in datetime format
    if 'timestamp' in full_ledger.columns:
        full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp'])
        return full_ledger.sort_values(by='timestamp').reset_index(drop=True)
    else:
        logger.error("Consolidated ledger is missing the 'timestamp' column.")
        return pd.DataFrame()


async def close_positions_after_5_days(config: dict):
    """
    Closes any open positions that have been held for 5 or more trading days.
    This function now uses the live portfolio from IB as the source of truth,
    reconstructs position ages using a FIFO stack from the ledger, and
    groups closing orders by Position ID to ensure spreads are closed as combos.
    """
    logger.info("--- Initiating position closing based on 5-day holding period ---")
    ib = IB()
    closed_position_details = []
    failed_closes = []
    kept_positions_details = []
    orphaned_positions = []

    try:
        # --- 1. Connect to IB and get the ground truth of current positions ---
        conn_settings = config.get('connection', {})
        await ib.connectAsync(host=conn_settings.get('host', '127.0.0.1'), port=conn_settings.get('port', 7497), clientId=random.randint(200, 2000), timeout=30)
        logger.info("Connected to IB for closing positions.")

        live_positions = await ib.reqPositionsAsync()
        if not live_positions:
            logger.info("No open positions found in the IB account. Nothing to close.")
            send_pushover_notification(config.get('notifications', {}), "Position Closing", "No open IB positions were found to close.")
            return

        logger.info(f"Found {len(live_positions)} live position legs in the IB account.")

        # --- 2. Load trade ledger to get historical data (open dates) ---
        trade_ledger = get_trade_ledger_df()
        if trade_ledger.empty:
            logger.warning("Trade ledger is empty. Cannot determine position ages. Skipping closure.")
            return

        # Prepare Ledger: Sort by Timestamp DESCENDING (Newest First) for reconstruction
        trade_ledger.sort_values('timestamp', ascending=False, inplace=True)

        # --- 3. Calculate trading day logic ---
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=date(date.today().year - 1, 1, 1), end=date.today()).to_pydatetime().tolist()
        today = datetime.now().date()
        custom_bday = pd.offsets.CustomBusinessDay(holidays=holidays)

        # Dictionary to store legs identified for closing, grouped by Position ID
        # Structure: { position_id: [ {contract_details}, ... ] }
        positions_to_close = defaultdict(list)

        # --- 4. FIFO Reconstruction and Age Verification ---
        # Map unique local symbols to live positions
        # Note: If multiple accounts, this assumes unique local_symbol per account or aggregated.
        live_pos_map = {p.contract.localSymbol: p for p in live_positions if p.position != 0}

        for symbol, pos in live_pos_map.items():
            live_qty = abs(pos.position)
            live_action_sign = 1 if pos.position > 0 else -1 # 1 for Long, -1 for Short

            # Filter ledger for this symbol and matching direction
            # If Live is Long (+), we look for 'BUY' trades in ledger.
            # If Live is Short (-), we look for 'SELL' trades in ledger.
            target_ledger_action = 'BUY' if live_action_sign > 0 else 'SELL'

            symbol_ledger = trade_ledger[
                (trade_ledger['local_symbol'] == symbol) &
                (trade_ledger['action'] == target_ledger_action) &
                (trade_ledger['reason'] == 'Strategy Execution') # Only count opening trades
            ].copy()

            if symbol_ledger.empty:
                orphaned_positions.append(f"{symbol} (Qty: {pos.position})")
                continue

            # Reconstruct the stack (Newest -> Oldest)
            accumulated_qty = 0

            for _, trade in symbol_ledger.iterrows():
                if accumulated_qty >= live_qty:
                    break # We have accounted for all live shares

                trade_qty = abs(trade['quantity'])
                qty_needed = live_qty - accumulated_qty
                qty_to_attribute = min(trade_qty, qty_needed)

                accumulated_qty += qty_to_attribute

                # Check Age
                trade_date = trade['timestamp']
                trading_days = pd.date_range(start=trade_date.date(), end=today, freq=custom_bday)
                age_in_trading_days = len(trading_days)

                logger.info(f"Reconstruction: {symbol} | Trade Date: {trade_date.date()} | Age: {age_in_trading_days} | Qty: {qty_to_attribute}")

                if age_in_trading_days >= 5:
                    # This specific lot is expired. Mark for closure.
                    # We need to INVERT the action to close.
                    close_action = 'SELL' if target_ledger_action == 'BUY' else 'BUY'

                    positions_to_close[trade['position_id']].append({
                        'conId': pos.contract.conId,
                        'symbol': symbol,
                        'exchange': pos.contract.exchange,
                        'action': close_action,
                        'quantity': qty_to_attribute,
                        'multiplier': pos.contract.multiplier,
                        'currency': pos.contract.currency
                    })
                else:
                    kept_positions_details.append(f"{trade['position_id']} ({symbol}) - Age: {age_in_trading_days} days")

        # --- 5. Execution Phase (Grouping by Position ID) ---
        for pos_id, legs in positions_to_close.items():
            logger.info(f"Processing closure for Position ID {pos_id} with {len(legs)} legs.")

            try:
                # Deduplicate legs if necessary (though our logic shouldn't produce duplicates for same symbol/pos_id combo usually)
                # But we might have multiple 'lots' for the same symbol if they are split in ledger?
                # Actually, if we have 2 expired lots for same symbol/pos_id, we should sum them up.
                combined_legs = defaultdict(lambda: {'quantity': 0, 'details': None})
                for leg in legs:
                    key = leg['conId']
                    combined_legs[key]['quantity'] += leg['quantity']
                    combined_legs[key]['details'] = leg

                final_legs_list = [v['details'] for v in combined_legs.values()]
                # Update quantities in the list
                for leg in final_legs_list:
                    leg['quantity'] = combined_legs[leg['conId']]['quantity']

                if len(final_legs_list) > 1:
                    # Create COMBO (Bag) Order
                    contract = Contract()
                    contract.symbol = config.get('symbol', 'KC')
                    contract.secType = "BAG"
                    contract.currency = final_legs_list[0]['currency']
                    contract.exchange = final_legs_list[0]['exchange'] or config.get('exchange', 'SMART')

                    combo_legs = []
                    for leg in final_legs_list:
                        combo_leg = ComboLeg()
                        combo_leg.conId = leg['conId']
                        combo_leg.ratio = int(leg['quantity']) # Assuming integer ratios for now
                        combo_leg.action = leg['action']
                        combo_leg.exchange = leg['exchange'] or config.get('exchange', 'SMART')
                        combo_legs.append(combo_leg)

                    contract.comboLegs = combo_legs

                    # For BAG orders, the totalQuantity is usually 1 (multiplied by ratio)
                    # Use the GCD of quantities if possible, but for simple closes, usually ratio is quantity and bag size is 1.
                    # Or ratio is 1 and bag size is quantity?
                    # If we have 5 spreads, we usually set ratio 1, size 5.
                    # Here we might have different quantities for different legs? (Unbalanced close).
                    # If quantities differ (e.g. 2 calls, 1 put), we can't do a simple 1:1 spread.
                    # We will assume balanced spread or use ratio=qty, size=1.

                    # Logic: Find GCD of quantities?
                    # Simplification: Set Ratio = Quantity, Bag Order Size = 1.
                    # This works for "Closing what we have".

                    order_size = 1

                    logger.info(f"Constructed BAG order for Pos ID {pos_id}: {[l.action + ' ' + str(l.ratio) + ' x ' + str(l.conId) for l in combo_legs]}")

                else:
                    # Single Leg Order
                    leg = final_legs_list[0]
                    contract = Contract()
                    contract.conId = leg['conId']
                    contract.symbol = config.get('symbol', 'KC')
                    contract.secType = "FOP" # Assumption, or fetch from pos?
                    # Actually, if we supply conId, IB usually resolves. But secType helps.
                    # We can get secType from the live position contract if needed, but we didn't store it.
                    # Let's rely on conId.
                    contract.exchange = leg['exchange'] or config.get('exchange', 'SMART')

                    order_size = leg['quantity']
                    logger.info(f"Constructed SINGLE order for Pos ID {pos_id}: {leg['action']} {order_size} {leg['symbol']}")

                # Place Order
                # For Bag, action is usually BUY or SELL. If we set leg actions explicitly, Bag action determines the side of the trade?
                # "For a Combo order, the action is determined by the actions of the legs."
                # Actually, IB Bag orders have an action. "BUY" a spread usually means Buy Leg 1, Sell Leg 2.
                # Here we set explicit actions in ComboLegs. The Bag order action is effectively "BUY" (to execute the legs as defined) or we match.
                # Standard practice: Set Bag Action to "BUY" and define leg actions as absolute.

                bag_action = 'BUY'
                if len(final_legs_list) == 1:
                    bag_action = final_legs_list[0]['action']

                # Fetch Market Data for Limit Order
                ticker = ib.reqMktData(contract, '', False, False)
                try:
                    await asyncio.sleep(2) # Give it a moment to populate
                    start_time = time.time()
                    # Wait for valid bid/ask
                    while util.isNan(ticker.bid) and util.isNan(ticker.ask):
                        await asyncio.sleep(0.1)
                        if (time.time() - start_time) > 10: # 10s timeout
                            break

                    # Determine Price
                    lmt_price = 0.0
                    if bag_action == 'BUY':
                        if not util.isNan(ticker.ask) and ticker.ask > 0:
                            lmt_price = ticker.ask
                        elif not util.isNan(ticker.last) and ticker.last > 0:
                            lmt_price = ticker.last
                        else:
                            lmt_price = ticker.close if not util.isNan(ticker.close) else 0.0
                    else: # SELL
                        if not util.isNan(ticker.bid) and ticker.bid > 0:
                            lmt_price = ticker.bid
                        elif not util.isNan(ticker.last) and ticker.last > 0:
                            lmt_price = ticker.last
                        else:
                            lmt_price = ticker.close if not util.isNan(ticker.close) else 0.0
                finally:
                    ib.cancelMktData(contract)

                if util.isNan(lmt_price) or lmt_price == 0.0:
                    logger.warning(f"Could not determine limit price for {contract.symbol}. Using Market Order.")
                    order = MarketOrder(bag_action, order_size)
                else:
                    logger.info(f"Placing LIMIT order for {contract.symbol}. Price: {lmt_price}")
                    order = LimitOrder(bag_action, order_size, lmt_price)

                order.outsideRth = True

                trade = place_order(ib, contract, order)
                logger.info(f"Placed close order {trade.order.orderId} for Pos ID {pos_id}. Waiting for fill...")

                # Poll for fill
                fill_detected = False
                for _ in range(30):
                    await asyncio.sleep(1)
                    if trade.orderStatus.status in [OrderStatus.Filled, OrderStatus.Cancelled, OrderStatus.Inactive]:
                        fill_detected = True
                        break

                if fill_detected and trade.orderStatus.status == OrderStatus.Filled:
                    # Log to ledger.
                    # IMPORTANT: For Bag orders, we want to log the LEGS.
                    # log_trade_to_ledger handles extraction of fills if we pass the trade.
                    # But trade.fills for Bag might be empty if we didn't receive execDetails?
                    # IB returns separate execDetails for legs.
                    # We need to make sure we capture them.
                    # Since we are not running a full event loop listener here (just polling status),
                    # trade.fills might NOT be populated if 'execDetails' event didn't fire in the background.
                    # ib.sleep() or asyncio.sleep() in ib_insync allows background processing?
                    # Yes, asyncio.sleep() allows loop to run. ib_insync updates trade.fills automatically via events.

                    # We need to wait a bit more for fills to populate?
                    await asyncio.sleep(1)

                    await log_trade_to_ledger(ib, trade, "5-Day Close", position_id=pos_id)

                    # Calculate PnL for report
                    # Sum realizedPNL from all fills
                    pnl = sum(f.commissionReport.realizedPNL for f in trade.fills if f.commissionReport)
                    price_str = f"{trade.orderStatus.avgFillPrice}"

                    desc = "Combo" if len(final_legs_list) > 1 else final_legs_list[0]['symbol']
                    closed_position_details.append({
                        "symbol": desc,
                        "action": "CLOSE",
                        "quantity": order_size, # Might be 1 for bag
                        "price": trade.orderStatus.avgFillPrice,
                        "pnl": pnl
                    })
                    logger.info(f"Successfully closed {desc} (Pos ID: {pos_id}). P&L: {pnl}")

                else:
                    logger.warning(f"Order {trade.order.orderId} for Pos ID {pos_id} timed out or failed. Status: {trade.orderStatus.status}")
                    failed_closes.append(f"Pos ID {pos_id}: Status {trade.orderStatus.status}")
                    if trade.orderStatus.status not in OrderStatus.DoneStates:
                        ib.cancelOrder(trade.order)

                await asyncio.sleep(1) # Throttle

            except Exception as ex:
                logger.error(f"Error closing position {pos_id}: {ex}")
                failed_closes.append(f"Pos ID {pos_id}: Error {ex}")

        # --- 6. Build and send a comprehensive notification ---
        message_parts = []
        total_pnl = 0
        if closed_position_details:
            message_parts.append(f"<b>✅ Successfully closed {len(closed_position_details)} positions (5-day rule):</b>")
            for detail in closed_position_details:
                pnl_str = f"${detail['pnl']:,.2f}"
                pnl_color = "green" if detail['pnl'] >= 0 else "red"
                message_parts.append(
                    f"  - <b>{detail['symbol']}</b>: {detail['action']} {abs(detail['quantity'])} "
                    f"@ {detail['price']}. P&L: <font color='{pnl_color}'>{pnl_str}</font>"
                )
                total_pnl += detail.get('pnl', 0)
            message_parts.append(f"<b>Total Realized P&L: ${total_pnl:,.2f}</b>")

        if failed_closes:
            message_parts.append(f"\n<b>❌ Failed to close {len(failed_closes)} positions:</b>")
            message_parts.extend([f"  - {reason}" for reason in failed_closes])

        if kept_positions_details:
            unique_kept = sorted(list(set(kept_positions_details)))
            message_parts.append(f"\n<b> Positions held ({len(unique_kept)} unique legs/combos):</b>")
            message_parts.extend([f"  - {reason}" for reason in unique_kept])

        if orphaned_positions:
             message_parts.append(f"\n<b>️ Orphaned Positions Found:</b>")
             message_parts.extend([f"  - {pos}" for pos in orphaned_positions])

        if not message_parts:
            message = "No positions were eligible for closing today based on the 5-day rule."
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
