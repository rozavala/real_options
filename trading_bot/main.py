"""Main entry point for the core trading bot logic.

This module orchestrates the primary trading cycle. It connects to the
Interactive Brokers (IB) Gateway, processes incoming trading signals,
and executes trades based on the defined strategies.
"""

import asyncio
import json
import os
import random
import logging
import traceback

from ib_insync import *

from logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.ib_interface import get_active_futures, build_option_chain
from trading_bot.risk_management import manage_existing_positions
from trading_bot.strategy import execute_directional_strategy, execute_volatility_strategy
from trading_bot.utils import is_market_open, normalize_strike

# --- Logging Setup ---
setup_logging()
ib_logger = logging.getLogger('ib_insync')
ib_logger.setLevel(logging.INFO)


async def send_heartbeat(ib: IB):
    """Sends a periodic request to the IB Gateway to keep the connection alive.

    This function runs in a loop, sending a `reqCurrentTimeAsync` request
    every 5 minutes (300 seconds) to prevent the connection from timing out.

    Args:
        ib (IB): The connected `ib_insync.IB` instance.
    """
    logging.info("Heartbeat task started.")
    while True:
        try:
            await asyncio.sleep(300)
            if ib.isConnected():
                server_time = await ib.reqCurrentTimeAsync()
                logging.info(f"Heartbeat: Connection is alive. Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except asyncio.CancelledError:
            logging.info("Heartbeat task cancelled.")
            break
        except Exception as e:
            logging.error(f"Error in heartbeat loop: {e}")


async def main_runner(config: dict, signals: list = None):
    """The main execution function for the trading bot.

    This function connects to IB, processes a list of trading signals, and
    manages the entire lifecycle of a trading cycle. For each signal, it:
    1. Finds the corresponding active futures contract.
    2. Fetches the current market price for the future.
    3. Calls `manage_existing_positions` to align the portfolio.
    4. If a new trade is warranted, it builds the option chain and executes
       the appropriate strategy (directional or volatility).
    5. Sends a summary notification of all actions taken during the cycle.

    Args:
        config (dict): The application configuration dictionary.
        signals (list, optional): A list of trading signals to be processed.
            If None or empty, the function will raise a ValueError.

    Raises:
        ValueError: If no signals are provided.
        ConnectionError: If no active futures contracts can be found.
    """
    ib = IB()
    heartbeat_task = None

    try:
        conn_settings = config.get('connection', {})
        host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)

        if not ib.isConnected():
            logging.info(f"Connecting to {host}:{port}...")
            # Use a random client ID to avoid conflicts with the monitor
            client_id = conn_settings.get('clientId', 10) + random.randint(1, 1000)
            await ib.connectAsync(host, port, clientId=client_id)
            send_pushover_notification(config.get('notifications', {}), "Trading Bot Started", "Trading logic has successfully connected to IBKR.")

            heartbeat_task = asyncio.create_task(send_heartbeat(ib))

        if not signals:
            raise ValueError("No signals provided to execute trades.")

        active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
        if not active_futures:
            raise ConnectionError("Could not find any active futures contracts.")

        trades_summary = {'filled': [], 'cancelled': [], 'aligned': 0}

        for signal in signals:
            future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
            if not future:
                logging.warning(f"No active future for signal month {signal.get('contract_month')}."); continue

            logging.info(f"\n===== Processing Signal for {future.localSymbol} =====")
            ticker = ib.reqMktData(future, '', False, False)
            await asyncio.sleep(1)
            price = ticker.marketPrice()
            if util.isNan(price):
                logging.error(f"Failed to get market price for {future.localSymbol}."); continue

            price = normalize_strike(price)

            logging.info(f"Current normalized price for {future.localSymbol}: {price}")

            # Step 1: Align portfolio. This will close any misaligned or orphaned positions.
            should_open_new_trade = await manage_existing_positions(ib, config, signal, price, future)

            if not should_open_new_trade:
                trades_summary['aligned'] += 1
                continue

            # Step 2: Place new trades. The is_market_open check is removed to allow pre-market order placement.
            logging.info(f"Proceeding to place new trade for {future.localSymbol}.")
            chain = await build_option_chain(ib, future)
            if not chain: continue

            trade = await (execute_directional_strategy if signal['prediction_type'] == 'DIRECTIONAL' else execute_volatility_strategy)(ib, config, signal, chain, price, future)
            if trade:
                trades_summary['filled' if trade.orderStatus.status == OrderStatus.Filled else 'cancelled'].append(trade)

        # End of cycle summary notification
        summary_msg = "<b>-- Trading Cycle Summary --</b>"
        if trades_summary['filled']:
            summary_msg += "\n<b>Positions Opened:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.orderStatus.filled} @ ${t.orderStatus.avgFillPrice:.2f}\n" for t in trades_summary['filled']])
        if trades_summary['cancelled']:
            summary_msg += "\n<b>Orders Cancelled:</b>\n" + "".join([f"- {t.contract.localSymbol}: {t.order.action} {t.order.totalQuantity}\n" for t in trades_summary['cancelled']])
        if trades_summary['aligned']:
            summary_msg += f"\n- {trades_summary['aligned']} position(s) were already aligned."
        if not trades_summary['filled'] and not trades_summary['cancelled'] and not trades_summary['aligned']:
            summary_msg += "\nNo positions were opened, closed, or aligned."

        send_pushover_notification(config.get('notifications', {}), "Trading Cycle Complete", summary_msg)

    except Exception as e:
        msg = f"An unexpected critical error occurred in the trading bot: {e}"
        logging.critical(f"{msg}\n{traceback.format_exc()}")
        send_pushover_notification(config.get('notifications', {}), "Trading Bot CRITICAL ERROR", msg)
    finally:
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        if ib.isConnected():
            ib.disconnect()
        logging.info("Trading bot has finished its execution cycle and disconnected.")