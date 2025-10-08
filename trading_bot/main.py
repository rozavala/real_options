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


def is_trade_sane(signal: dict, config: dict) -> bool:
    min_confidence = config.get('risk_management', {}).get('min_confidence_threshold', 0.0)
    if signal['confidence'] < min_confidence:
        logging.warning(f"Sanity Check FAILED: Signal confidence {signal['confidence']} is below threshold {min_confidence}.")
        return False
    logging.info("Sanity Check PASSED."); return True


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

    while True:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            conn_settings = config.get('connection', {})
            host, port = conn_settings.get('host', '127.0.0.1'), conn_settings.get('port', 7497)

            if not ib.isConnected():
                logging.info(f"Connecting to {host}:{port}...")
                await ib.connectAsync(host, port, clientId=conn_settings.get('clientId', random.randint(1, 1000)))
                send_pushover_notification(config.get('notifications', {}), "Script Started", "Trading script has successfully connected to IBKR.")
                if monitor_task is None or monitor_task.done():
                    monitor_task = asyncio.create_task(monitor_positions_for_risk(ib, config))
                if heartbeat_task is None or heartbeat_task.done():
                    heartbeat_task = asyncio.create_task(send_heartbeat(ib))

            active_futures = await get_active_futures(ib, config['symbol'], config['exchange'])
            if not active_futures:
                raise ConnectionError("Could not find any active futures contracts.")

            front_month_details = (await ib.reqContractDetailsAsync(active_futures[0]))[0]
            if not is_market_open(front_month_details, config['exchange_timezone']):
                await asyncio.sleep(calculate_wait_until_market_open(front_month_details, config['exchange_timezone']))
                continue

            if not signals:
                raise ValueError("Could not get a valid signal list from the API.")

            trades_summary = {'filled': [], 'cancelled': []}
            for signal in signals:
                future = next((f for f in active_futures if f.lastTradeDateOrContractMonth.startswith(signal.get("contract_month", ""))), None)
                if not future:
                    logging.warning(f"No active future for signal month {signal.get('contract_month')}."); continue

                logging.info(f"\n===== Processing Signal for {future.localSymbol} =====")
                ticker = ib.reqMktData(future, '', False, False);
                await asyncio.sleep(1)
                price = ticker.marketPrice()
                if util.isNan(price):
                    logging.error(f"Failed to get market price for {future.localSymbol}."); continue

                logging.info(f"Current price for {future.localSymbol}: {price}")

                if await manage_existing_positions(ib, config, signal, price, future) and is_trade_sane(signal, config):
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

            ny_tz = pytz.timezone(config['exchange_timezone'])
            now_ny = datetime.now(ny_tz)
            h, m = map(int, config.get('trade_execution_time_ny', '04:26').split(':'))
            next_trade = now_ny.replace(hour=h, minute=m, second=0, microsecond=0)
            if now_ny >= next_trade:
                next_trade += timedelta(days=1)
            wait = (next_trade - now_ny).total_seconds()
            logging.info(f"\nTrading cycle complete. Waiting {wait / 3600:.2f} hours until {next_trade:%Y-%m-%d %H:%M:%S} NY time.")
            await asyncio.sleep(wait)

        except (ConnectionError, OSError, asyncio.TimeoutError, asyncio.CancelledError) as e:
            msg = f"Connection error: {e}. Reconnecting..."
            logging.error(msg)
            send_pushover_notification(config.get('notifications', {}), "Connection Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)
        except Exception as e:
            msg = f"An unexpected critical error occurred: {e}. Restarting logic in 1 min..."
            logging.critical(f"{msg}\n{traceback.format_exc()}")
            send_pushover_notification(config.get('notifications', {}), "Critical Error", msg)
            if monitor_task and not monitor_task.done(): monitor_task.cancel()
            if heartbeat_task and not heartbeat_task.done(): heartbeat_task.cancel()
            monitor_task, heartbeat_task = None, None
            if ib.isConnected(): ib.disconnect()
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        # This is a placeholder for testing. The orchestrator will call main_runner with real signals.
        mock_signals = [
            {"contract_month": "202512", "prediction_type": "DIRECTIONAL", "direction": "BULLISH", "confidence": 0.65},
            {"contract_month": "202603", "prediction_type": "DIRECTIONAL", "direction": "BEARISH", "confidence": 0.88},
        ]
        util.run(main_runner(signals=mock_signals))
    except (KeyboardInterrupt, SystemExit):
        logging.info("Script stopped manually.")