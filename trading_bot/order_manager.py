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

        # --- Create a detailed notification message ---

        # 1. Summarize the API signals
        signal_summary_parts = []
        if signals:
            for signal in signals:
                future_month = signal.get("contract_month", "N/A")
                if signal['prediction_type'] == 'DIRECTIONAL':
                    direction = signal.get("direction", "N/A")
                    signal_summary_parts.append(f"  - <b>{future_month}</b>: {direction} signal")
                elif signal['prediction_type'] == 'VOLATILITY':
                    vol_level = signal.get("level", "N/A")
                    signal_summary_parts.append(f"  - <b>{future_month}</b>: {vol_level} Volatility signal")

        signal_summary = "<b>Signals from Prediction API:</b>\n" + "\n".join(signal_summary_parts)

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

        # Wait for the ticker to update with a valid price, with a timeout
        start_time = time.time()
        while util.isNan(ticker.bid) and util.isNan(ticker.ask) and util.isNan(ticker.last):
            await asyncio.sleep(0.1)
            if (time.time() - start_time) > 5: # 5-second timeout
                logger.error(f"Timeout waiting for market data for {contract[0].contract.localSymbol}.")
                ib.cancelMktData(ticker.contract)
                return f"Leg {contract[0].contract.localSymbol}: Timeout"

        # Fetch all data points
        leg_symbol = contract[0].contract.localSymbol
        leg_bid = ticker.bid if not util.isNan(ticker.bid) else "N/A"
        leg_ask = ticker.ask if not util.isNan(ticker.ask) else "N/A"
        leg_vol = ticker.volume if not util.isNan(ticker.volume) else "N/A"
        leg_last = ticker.last if not util.isNan(ticker.last) else "N/A"
        
        ib.cancelMktData(ticker.contract)
        
        # Return a formatted string
        return f"Leg {leg_symbol}: {leg_bid}x{leg_ask}, V:{leg_vol}, L:{leg_last}"

    except Exception as e:
        logger.error(f"Error fetching market data for leg {leg.conId}: {e}")
        return f"Leg conId {leg.conId}: Error"


async def _handle_and_log_fill(ib: IB, trade: Trade, fill: Fill, combo_id: int):
    """
    Handles the logic for processing a trade fill, ensuring the contract
    details are complete before logging.
    """
    try:
        # With combo orders, IB sends fills for the parent 'Bag' in addition
        # to the individual legs. These Bag fills are redundant for our ledger
        # and trying to qualify them will cause an API error. We identify and
        # skip them here.
        if isinstance(fill.contract, Bag):
            logger.info(f"Skipping ledger log for summary combo fill with conId: {fill.contract.conId}")
            return

        # The contract object in the Fill can be incomplete. We need to get the
        # full contract details to log accurately, especially for options.
        # We create a new contract object from the conId and qualify it.
        detailed_contract = Contract(conId=fill.contract.conId)
        await ib.qualifyContractsAsync(detailed_contract)

        # Create a new Fill object with the detailed contract, preserving the
        # original execution and commission report details. This avoids the
        # "can't set attribute" error on the original immutable Fill object.
        corrected_fill = Fill(
            contract=detailed_contract,
            execution=fill.execution,
            commissionReport=fill.commissionReport,
            time=fill.time
        )

        # If the parent trade is a Bag, find the specific leg that was filled
        # to ensure context is maintained.
        if isinstance(trade.contract, Bag):
            for leg in trade.contract.comboLegs:
                if leg.conId == fill.contract.conId:
                    logger.info(f"Matched fill conId {fill.contract.conId} to leg in {trade.contract.localSymbol}.")
                    break
            else:
                logger.warning(f"Could not match fill conId {fill.contract.conId} to any leg in {trade.contract.localSymbol}.")

        log_trade_to_ledger(ib, trade, "Strategy Execution", specific_fill=corrected_fill, combo_id=combo_id)
        logger.info(f"Successfully logged fill for {detailed_contract.localSymbol} (Order ID: {trade.order.orderId}, Combo ID: {combo_id})")

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

            # Check if this leg has already been processed to avoid duplicates
            if leg_con_id in order_details['filled_legs']:
                logger.info(f"Duplicate fill event for leg {leg_con_id} in order {order_id}. Ignoring.")
                return

            # The trade object here is for the specific leg. We need the permId
            # of the parent combo order, which is stored in our tracking dict.
            parent_trade = order_details['trade']
            combo_perm_id = parent_trade.order.permId

            logger.info(f"Fill received for leg {leg_con_id} in order {order_id}. Processing with Combo ID {combo_perm_id}.")
            asyncio.create_task(_handle_and_log_fill(ib, trade, fill, combo_perm_id))

            # --- Track the fill details for this leg ---
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
        
        # --- PLACEMENT LOOP (ENHANCED LOGGING & NOTIFICATIONS) ---
        for contract, order in ORDER_QUEUE:
            
            # --- 1. Get BAG (Spread) Market Data ---
            bag_ticker = ib.reqMktData(contract, '', False, False)
            await asyncio.sleep(2) # Wait for ticker to populate
            
            bag_bid = bag_ticker.bid if not util.isNan(bag_ticker.bid) else "N/A"
            bag_ask = bag_ticker.ask if not util.isNan(bag_ticker.ask) else "N/A"
            bag_vol = bag_ticker.volume if not util.isNan(bag_ticker.volume) else "N/A"
            bag_last = bag_ticker.last if not util.isNan(bag_ticker.last) else "N/A"
            bag_last_time = bag_ticker.time.strftime('%H:%M:%S') if bag_ticker.time else "N/A"
            
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
                'fill_prices': {} # Stores fill prices by conId
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
        monitoring_duration = 900  # 15 minutes
        logger.info(f"Monitoring orders for up to {monitoring_duration / 60} minutes...")
        while time.time() - start_time < monitoring_duration:
            if not live_orders: break
            all_done = all(
                details['status'] in OrderStatus.DoneStates
                for details in live_orders.values()
            )
            if all_done:
                logger.info("All orders have reached a terminal state. Concluding monitoring.")
                break
            await asyncio.sleep(15)
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

                # Since this is a combo, we report the fills for each leg
                if details['total_legs'] > 1:
                    avg_fill_price = sum(details['fill_prices'].values()) / len(details['fill_prices'])
                    fill_line = (
                        f"  - <b>{trade.contract.localSymbol}</b> ({trade.order.action})<br>"
                        f"    {price_info} -> FILLED @ avg ${avg_fill_price:.2f}"
                    )
                else: # Single leg order
                    fill_price = next(iter(details['fill_prices'].values()), 0.0)
                    fill_line = (
                        f"  - <b>{trade.contract.localSymbol}</b> ({trade.order.action})<br>"
                        f"    {price_info} -> FILLED @ ${fill_price:.2f}"
                    )
                filled_orders.append(fill_line)
            
            elif details['status'] not in OrderStatus.DoneStates:
                
                # --- 1. Get FINAL BAG (Spread) Market Data ---
                final_bag_ticker = ib.reqMktData(trade.contract, '', False, False)
                await asyncio.sleep(2)
                
                final_bag_bid = final_bag_ticker.bid if not util.isNan(final_bag_ticker.bid) else "N/A"
                final_bag_ask = final_bag_ticker.ask if not util.isNan(final_bag_ticker.ask) else "N/A"
                final_bag_vol = final_bag_ticker.volume if not util.isNan(final_bag_ticker.volume) else "N/A"
                final_bag_last = final_bag_ticker.last if not util.isNan(final_bag_ticker.last) else "N/A"
                final_bag_last_time = final_bag_ticker.time.strftime('%H:%M:%S') if final_bag_ticker.time else "N/A"
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
                # Pass the permId explicitly as the combo_id for consistency.
                log_trade_to_ledger(ib, trade, "Daily Close", combo_id=trade.order.permId)
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
