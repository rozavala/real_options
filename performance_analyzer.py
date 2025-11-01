"""Analyzes and reports the performance of trading activities."""

import pandas as pd
from datetime import datetime
import os
import logging
import asyncio
import random
from ib_insync import IB

from logging_config import setup_logging
from notifications import send_pushover_notification
from trading_bot.model_signals import get_model_signals_df
from trading_bot.performance_graphs import generate_performance_charts
from config_loader import load_config

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("PerformanceAnalyzer")


def get_trade_ledger_df():
    """Reads and consolidates the main and archived trade ledgers for analysis."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive')

    dataframes = []

    if os.path.exists(ledger_path):
        dataframes.append(pd.read_csv(ledger_path))
    if os.path.exists(archive_dir):
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        if archive_files:
            df_list = [pd.read_csv(file) for file in archive_files]
            dataframes.extend(df_list)

    if not dataframes:
        return pd.DataFrame()

    full_ledger = pd.concat(dataframes, ignore_index=True)
    full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp'])

    return full_ledger.sort_values(by='timestamp').reset_index(drop=True)


async def get_account_pnl_and_positions(config: dict) -> dict | None:
    """Connects to IB and fetches P&L and current positions."""
    ib = IB()
    summary = {}
    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=random.randint(200, 2000),
            timeout=10
        )

        # Use reqAccountSummary with a subscription to get live updates
        # The values will arrive in the accountSummary event
        ib.reqAccountSummary('All', 'NetLiquidation,DailyPnL,RealizedPnL,UnrealizedPnL')

        # Give the subscription a moment to deliver the first batch of data
        await asyncio.sleep(2.5)

        account_values = ib.accountSummary()
        positions = await ib.reqPositionsAsync()

        summary = {
            'daily_pnl': next((float(v.value) for v in account_values if v.tag == 'DailyPnL'), 0.0),
            'positions': positions
        }
        logger.info(f"Successfully fetched account P&L from IB: ${summary['daily_pnl']:.2f}")

    except Exception as e:
        logger.error(f"Failed to get account summary from IB: {e}", exc_info=True)
        return None
    finally:
        if ib.isConnected():
            ib.disconnect()

    return summary

def generate_executive_summary(trade_df: pd.DataFrame, signals_df: pd.DataFrame, today_date: datetime.date, live_daily_pnl: float | None) -> tuple[str, float]:
    """Generates Section 1: The Executive Summary."""

    # --- Helper function to calculate metrics from the trade ledger ---
    def calculate_ledger_metrics(trades, signals):
        if trades.empty:
            return {
                "pnl": 0, "signals_fired": len(signals), "trades_executed": 0,
                "execution_rate": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0
            }

        # Create a stable position identifier
        combo_legs_map = trades.groupby('combo_id')['local_symbol'].unique().apply(lambda legs: tuple(sorted(legs)))
        trades['position_id'] = trades['combo_id'].map(combo_legs_map)

        # Calculate P&L for each closed position
        closed_positions = trades.groupby('position_id').filter(lambda x: (x['action'].eq('BUY').astype(int) - x['action'].eq('SELL').astype(int)).sum() == 0)
        pnl_per_position = closed_positions.groupby('position_id')['total_value_usd'].sum()

        net_pnl = pnl_per_position.sum()
        opening_trades = trades[trades['reason'] == 'Strategy Execution']
        trades_executed = len(opening_trades.groupby('position_id'))

        wins = pnl_per_position[pnl_per_position > 0]
        losses = pnl_per_position[pnl_per_position <= 0]

        win_rate = len(wins) / trades_executed if trades_executed > 0 else 0
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = losses.mean() if not losses.empty else 0
        signals_fired = len(signals[signals['signal'] != 'NEUTRAL'])
        execution_rate = trades_executed / signals_fired if signals_fired > 0 else 0

        return {
            "pnl": net_pnl, "signals_fired": signals_fired, "trades_executed": trades_executed,
            "execution_rate": execution_rate, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss
        }

    # --- Calculate metrics ---
    today_trades = trade_df[trade_df['timestamp'].dt.date == today_date]
    today_signals = signals_df[signals_df['timestamp'].dt.date == today_date]

    # LTD metrics are always from the ledger
    ltd_metrics = calculate_ledger_metrics(trade_df, signals_df)

    # Today's metrics are from the ledger, but P&L is overridden by the live value
    today_metrics = calculate_ledger_metrics(today_trades, today_signals)
    if live_daily_pnl is not None:
        today_metrics['pnl'] = live_daily_pnl

    # --- Build Report ---
    report = f"{'Metric':<18} {'Today':>12} {'LTD':>12}\n"
    report += "-" * 44 + "\n"
    rows = {
        "Net P&L": (f"${today_metrics['pnl']:,.2f}", f"${ltd_metrics['pnl']:,.2f}"),
        "Signals Fired": (f"{today_metrics['signals_fired']}", f"{ltd_metrics['signals_fired']}"),
        "Trades Executed": (f"{today_metrics['trades_executed']}", f"{ltd_metrics['trades_executed']}"),
        "Exec. Rate": (f"{today_metrics['execution_rate']:.1%}", f"{ltd_metrics['execution_rate']:.1%}"),
        "Win Rate": (f"{today_metrics['win_rate']:.1%}", f"{ltd_metrics['win_rate']:.1%}"),
        "Avg. Win/Loss": (f"${today_metrics['avg_win']:,.0f}/${today_metrics['avg_loss']:,.0f}", f"${ltd_metrics['avg_win']:,.0f}/${ltd_metrics['avg_loss']:,.0f}")
    }
    for metric, (today_val, ltd_val) in rows.items():
        report += f"{metric:<18} {today_val:>12} {ltd_val:>12}\n"

    final_pnl_for_title = today_metrics['pnl']
    return report, final_pnl_for_title

def generate_morning_signals_report(signals_df: pd.DataFrame, today_date: datetime.date) -> str:
    """Creates a report of the signals generated this morning."""
    today_signals = signals_df[signals_df['timestamp'].dt.date == today_date]
    if today_signals.empty:
        return "No model signals for today.\n"

    report = f"{'Contract':<12} {'Signal':<10}\n"
    report += "-" * 22 + "\n"
    for _, row in today_signals.iterrows():
        report += f"{row['contract']:<12} {row['signal']:<10}\n"
    return report

def generate_open_positions_report(positions: list) -> str:
    """Creates a summary of all currently open positions from IB."""
    if not positions:
        return "No open positions.\n"

    report = f"{'Symbol':<25} {'Qty':>5} {'Avg Cost':>10} {'Unreal. P&L':>15}\n"
    report += "-" * 57 + "\n"
    for pos in positions:
        # Skip positions with zero quantity
        if pos.position == 0:
            continue
        symbol = pos.contract.localSymbol
        qty = int(pos.position)
        avg_cost = f"${pos.avgCost/100.0:,.2f}" # Assuming cost is in cents
        # P&L for positions is not directly provided in this object, would need another call
        # For now, we will mark it as N/A
        unreal_pnl_str = "N/A" # This would require fetching live market data
        
        report += f"{symbol:<25} {qty:>5} {avg_cost:>10} {unreal_pnl_str:>15}\n"

    return report

def generate_closed_positions_report(trade_df: pd.DataFrame, today_date: datetime.date) -> str:
    """Creates a report of positions closed today, based on the trade ledger."""
    today_trades = trade_df[trade_df['timestamp'].dt.date == today_date].copy()
    if today_trades.empty:
        return "No trades executed today.\n"

    # Identify closed positions for today
    combo_legs_map = today_trades.groupby('combo_id')['local_symbol'].unique().apply(lambda legs: tuple(sorted(legs)))
    today_trades['position_id'] = today_trades['combo_id'].map(combo_legs_map)
    closed_positions = today_trades.groupby('position_id').filter(lambda x: (x['action'].eq('BUY').astype(int) - x['action'].eq('SELL').astype(int)).sum() == 0)

    if closed_positions.empty:
        return "No positions were closed today.\n"

    pnl_per_position = closed_positions.groupby('position_id')['total_value_usd'].sum()

    report = f"{'Position':<25} {'Net P&L':>12}\n"
    report += "-" * 39 + "\n"
    for pos_id, pnl in pnl_per_position.items():
        # Shorten the position id for display
        pos_str = ' / '.join(pos_id)[:23] + '..' if len(' / '.join(pos_id)) > 25 else ' / '.join(pos_id)
        pnl_str = f"${pnl:,.2f}"
        report += f"{pos_str:<25} {pnl_str:>12}\n"

    return report

async def generate_system_status_report(config: dict) -> tuple[str, bool]:
    """Generates Section 3: System Status Check."""
    report = "Section 3: System Status\n"
    is_ok = True

    # 1. Position Check (now done via live data in open_positions_report)
    # This check can be simplified or removed if the live data is trusted.
    # For now, we'll check for pending orders.

    # 2. Pending Orders Check
    try:
        open_orders = await check_for_open_orders(config)
        if open_orders:
            report += "!! WARNING: PENDING ORDERS !!\n"
            for order in open_orders:
                report += f"- {order.action} {order.totalQuantity} {order.contract.localSymbol} ({order.orderStatus.status})\n"
            is_ok = False
        else:
            report += "Pending Orders: PASS (None found)\n"
    except Exception as e:
        logger.error(f"Failed to check for open orders: {e}")
        report += "Pending Orders: FAIL (No connection)\n"
        is_ok = False

    return report, is_ok

async def check_for_open_orders(config: dict) -> list:
    """Connects to IB and checks for any open orders."""
    ib = IB()
    try:
        conn_settings = config.get('connection', {})
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=random.randint(200, 2000),
            timeout=10
        )
        open_orders = await ib.reqAllOpenOrdersAsync()
        return open_orders
    finally:
        if ib.isConnected():
            ib.disconnect()

async def analyze_performance(config: dict) -> dict | None:
    """
    Analyzes trading performance and generates a dictionary of report parts.
    """
    trade_df = get_trade_ledger_df()
    signals_df = get_model_signals_df()

    if trade_df.empty:
        logger.error("Trade ledger is empty or not found. Cannot analyze performance.")
        return None

    logger.info("--- Starting Daily Performance Analysis ---")

    try:
        # Determine the analysis date from the ledger
        today_date = trade_df['timestamp'].max().date()
        today_str = today_date.strftime('%Y-%m-%d')

        # Fetch live data from IB
        account_summary = await get_account_pnl_and_positions(config)
        live_pnl = account_summary.get('daily_pnl') if account_summary else None
        live_positions = account_summary.get('positions') if account_summary else []

        # --- Generate Report Sections ---
        exec_summary, pnl_for_title = generate_executive_summary(trade_df, signals_df, today_date, live_pnl)
        morning_signals = generate_morning_signals_report(signals_df, today_date)
        open_positions = generate_open_positions_report(live_positions)
        closed_positions = generate_closed_positions_report(trade_df, today_date)

        # --- Generate Charts ---
        chart_paths = generate_performance_charts(trade_df, signals_df)

        logger.info("--- Analysis Complete ---")
        
        return {
            "title": f"Daily Report: P&L ${pnl_for_title:,.2f}",
            "date": today_str,
            "reports": {
                "Exec. Summary": exec_summary,
                "Morning Signals": morning_signals,
                "Open Positions": open_positions,
                "Closed Positions": closed_positions
            },
            "charts": chart_paths
        }

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)
        return None

async def main():
    """
    Main function to run analysis and send notifications in multiple parts.
    """
    config = load_config()
    if not config:
        logger.critical("Failed to load configuration. Exiting."); return

    analysis_result = await analyze_performance(config)

    if analysis_result:
        notification_config = config.get('notifications', {})

        # --- Send Main Title ---
        send_pushover_notification(notification_config, analysis_result['title'], f"Trading Performance Report for {analysis_result['date']}")

        # --- Send Report Sections ---
        for title, content in analysis_result['reports'].items():
            # The content is wrapped in <pre> tags for monospaced formatting
            message = f"<pre>{content}</pre>"
            send_pushover_notification(
                notification_config,
                f"Report Section: {title}",
                message,
                monospace=True
            )
            await asyncio.sleep(1) # Small delay to ensure notifications arrive in order

        # --- Send Charts ---
        for i, chart_path in enumerate(analysis_result['charts']):
            chart_title = os.path.splitext(os.path.basename(chart_path))[0].replace('_', ' ').title()
            send_pushover_notification(
                notification_config,
                f"Chart {i+1}/{len(analysis_result['charts'])}: {chart_title}",
                f"See attached chart: {chart_title}",
                attachment_path=chart_path
            )
            await asyncio.sleep(1)

    else:
        send_pushover_notification(
            config.get('notifications', {}),
            "Performance Analysis FAILED",
            "The performance analysis script failed to run. Check logs for details."
        )

if __name__ == "__main__":
    asyncio.run(main())
