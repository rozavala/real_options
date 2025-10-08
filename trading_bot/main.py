import asyncio
import json
import os
import random
import logging
import traceback
from datetime import datetime, timedelta

import pytz
from ib_insync import *

from notifications import send_pushover_notification
from trading_bot.ib_interface import get_active_futures, build_option_chain
from trading_bot.risk_management import manage_existing_positions, monitor_positions_for_risk
from trading_bot.strategy import execute_directional_strategy, execute_volatility_strategy
from trading_bot.utils import is_market_open, calculate_wait_until_market_open

# --- Logging Setup ---
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ])
ib_logger = logging.getLogger('ib_insync')
ib_logger.setLevel(logging.INFO)


async def send_heartbeat(ib: IB):
    logging.info("Heartbeat task started.")
    while True:
        try:
            await asyncio.sleep(300)
            if ib.isConnected():
                server_time = await ib.reqCurrentTimeAsync()
                logging.info(f"Heartbeat: Connection is alive. Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logging.error(f"Error in heartbeat loop: {e}")


async def main_runner(signals: list = None):
    ib = IB()
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.json')
    monitor_task, heartbeat_task = None, None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        conn_settings = config.get('connection', {})
        host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)

        if not ib.isConnected():
            logging.info(f"Connecting to {host}:{port}...")
            await ib.connectAsync(host, port, clientId=conn_settings.get('clientId', random.randint(1, 1000)))
            send_pushover_notification(config.get('notifications', {}), "Trading Bot Started", "Trading logic has successfully connected to IBKR.")

            # These tasks will run for the duration of the main_runner execution.
            monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
            heartbeat_task = asyncio.create_task(send_heartbeat(ib))

        active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
        if not active_futures:
            raise ConnectionError("Could not find any active futures contracts.")

        front_month_details = (await ib.reqContractDetailsAsync(active_futures[0]))[0]
        if not is_market_open(front_month_details, config['exchange_timezone']):
            logging.info("Market is closed. No trades will be placed.")
            return

        if not signals:
            raise ValueError("No signals provided to execute trades.")

        trades_summary = {'filled': [], 'cancelled': []}
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

            logging.info(f"Current price for {future.localSymbol}: {price}")

            if await manage_existing_positions(ib, config, signal, price, future):
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
        if not trades_summary['filled'] and not trades_summary['cancelled']:
            summary_msg += "\nNo new positions were opened or attempted."
        send_pushover_notification(config.get('notifications', {}), "Trading Cycle Complete", summary_msg)

    except Exception as e:
        msg = f"An unexpected critical error occurred in the trading bot: {e}"
        logging.critical(f"{msg}\n{traceback.format_exc()}")
        send_pushover_notification(config.get('notifications', {}), "Trading Bot CRITICAL ERROR", msg)
    finally:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
        if ib.isConnected():
            ib.disconnect()
        logging.info("Trading bot has finished its execution cycle and disconnected.")