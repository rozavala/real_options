"""Analyzes and reports the performance of trading activities."""

import pandas as pd
from datetime import datetime
import os
import logging
import asyncio
import random
import math
from ib_insync import IB, PortfolioItem

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


async def get_live_account_data(config: dict) -> dict | None:
    """
    Connects to IB and fetches all live data needed for the daily report:
    - P&L from account summary.
    - Open positions from the portfolio.
    - Today's trade executions (fills).
    """
    ib = IB()
    summary_data = {}
    conn_settings = config.get('connection', {})
    account = conn_settings.get('account_number', '')
    try:
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=random.randint(200, 2000),
            timeout=15
        )

        if not account:
            accounts = ib.managedAccounts()
            if accounts:
                account = accounts[0]
                logger.info(f"Account number not in config, using managed account: {account}")
            else:
                logger.error("No account number in config and no managed accounts found.")
                return None

        # 1. Fetch account summary to calculate daily P&L from equity change.
        summary = await ib.accountSummaryAsync()
        account_summary_values = [v for v in summary if v.account == account]
        summary_dict = {v.tag: v.value for v in account_summary_values}

        daily_pnl = 0.0
        try:
            current_equity = float(summary_dict['EquityWithLoanValue'])
            previous_equity = float(summary_dict['PreviousDayEquityWithLoanValue'])
            daily_pnl = current_equity - previous_equity
        except (ValueError, KeyError) as e:
            logger.error(f"Could not calculate daily P&L from account summary: {e}")

        # 2. Fetch open positions
        portfolio_items = ib.portfolio()

        # 3. Fetch today's executions for live realized P&L calculation
        executions = await ib.reqExecutionsAsync()
        today_date = datetime.now().date()
        todays_fills = [f for f in executions if f.time.date() == today_date]
        logger.info(f"Successfully fetched {len(todays_fills)} execution(s) for today from IB.")

        summary_data = {
            'daily_pnl': daily_pnl,
            'portfolio': portfolio_items,
            'executions': todays_fills
        }
        logger.info(f"Successfully fetched live account data. Daily P&L: ${daily_pnl:,.2f}")

    except asyncio.TimeoutError:
        logger.error("Connection to IB timed out during live data fetch.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to get live account data from IB: {e}", exc_info=True)
        return None
    finally:
        if ib.isConnected():
            if account:
                ib.cancelPnL(account)
            ib.disconnect()

    return summary_data

def generate_executive_summary(
    trade_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    today_date: datetime.date,
    total_daily_pnl: float | None,
    realized_daily_pnl: float,
    unrealized_daily_pnl: float
) -> tuple[str, float]:
    """Generates Section 1: The Executive Summary, separating Realized and Unrealized P&L."""

    # --- Helper function to calculate metrics from the trade ledger ---
    def calculate_ledger_metrics(trades, signals):
        # This function primarily calculates LTD realized P&L and other stats
        if trades.empty:
            return {
                "pnl": 0, "signals_fired": len(signals), "trades_executed": 0,
                "execution_rate": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0
            }

        combo_legs_map = trades.groupby('combo_id')['local_symbol'].unique().apply(lambda legs: tuple(sorted(legs)))
        trades['position_id'] = trades['combo_id'].map(combo_legs_map)

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

    # LTD metrics are always from the ledger (realized P&L)
    ltd_metrics = calculate_ledger_metrics(trade_df, signals_df)

    # Today's non-P&L stats are from the daily ledger slice
    today_stats = calculate_ledger_metrics(today_trades, today_signals)

    # --- Build Report ---
    report = f"{'Metric':<18} {'Today':>12} {'LTD':>12}\n"
    report += "-" * 44 + "\n"

    # P&L rows are handled separately to show the breakdown
    today_total_pnl_str = f"${total_daily_pnl:,.2f}" if total_daily_pnl is not None else "N/A"
    today_realized_pnl_str = f"${realized_daily_pnl:,.2f}"
    today_unrealized_pnl_str = f"${unrealized_daily_pnl:,.2f}"
    ltd_realized_pnl_str = f"${ltd_metrics['pnl']:,.2f}"

    pnl_rows = {
        "Total P&L": (today_total_pnl_str, ltd_realized_pnl_str),
        " Realized P&L": (today_realized_pnl_str, ltd_realized_pnl_str), # LTD is always realized
        " Unrealized P&L": (today_unrealized_pnl_str, "$0.00"), # No unrealized concept for LTD
    }
    for metric, (today_val, ltd_val) in pnl_rows.items():
        report += f"{metric:<18} {today_val:>12} {ltd_val:>12}\n"

    report += "-" * 44 + "\n"

    # Other metric rows
    other_rows = {
        "Signals Fired": (f"{today_stats['signals_fired']}", f"{ltd_metrics['signals_fired']}"),
        "Trades Executed": (f"{today_stats['trades_executed']}", f"{ltd_metrics['trades_executed']}"),
        "Exec. Rate": (f"{today_stats['execution_rate']:.1%}", f"{ltd_metrics['execution_rate']:.1%}"),
        "Win Rate": (f"{today_stats['win_rate']:.1%}", f"{ltd_metrics['win_rate']:.1%}"),
    }
    for metric, (today_val, ltd_val) in other_rows.items():
        report += f"{metric:<18} {today_val:>12} {ltd_val:>12}\n"

    final_pnl_for_title = total_daily_pnl if total_daily_pnl is not None else 0.0
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

def generate_open_positions_report(portfolio: list) -> tuple[str, float]:
    """
    Creates a summary of all currently open positions and calculates the total unrealized P&L.
    Returns the report string and the total unrealized P&L amount.
    """
    # Filter out positions that are flat (position == 0)
    open_positions = [p for p in portfolio if p.position != 0]

    if not open_positions:
        return "No open positions.\n", 0.0

    report = f"{'Symbol':<25} {'Qty':>5} {'Avg Cost':>10} {'Unreal. P&L':>15}\n"
    report += "-" * 57 + "\n"
    total_unrealized_pnl = 0
    for pos in open_positions:
        symbol = pos.contract.localSymbol
        qty = int(pos.position)
        avg_cost = f"${pos.averageCost:,.2f}"
        unreal_pnl = pos.unrealizedPNL
        unreal_pnl_str = f"${unreal_pnl:,.2f}" if isinstance(unreal_pnl, float) else "N/A"

        report += f"{symbol:<25} {qty:>5} {avg_cost:>10} {unreal_pnl_str:>15}\n"
        if isinstance(unreal_pnl, float):
            total_unrealized_pnl += unreal_pnl

    # --- Add Total Row ---
    report += "-" * 57 + "\n"
    total_pnl_str = f"${total_unrealized_pnl:,.2f}"
    report += f"{'TOTAL':<42} {total_pnl_str:>15}\n"

    return report, total_unrealized_pnl

def generate_closed_positions_report(fills: list) -> tuple[str, float]:
    """
    Creates a report of positions closed today from live execution data from IB.
    Calculates realized P&L from the fills.
    """
    if not fills:
        return "No positions were closed today.\n", 0.0

    # Group fills by contract to determine net positions for the day
    positions = {}
    for fill in fills:
        conId = fill.contract.conId
        if conId not in positions:
            positions[conId] = {'qty': 0, 'cost': 0, 'symbol': fill.contract.localSymbol}

        # 'BOT' is a buy, 'SLD' is a sell
        action_multiplier = 1 if fill.execution.side == 'BOT' else -1
        positions[conId]['qty'] += fill.execution.shares * action_multiplier
        positions[conId]['cost'] += fill.execution.shares * fill.execution.price * action_multiplier

    # A position is considered closed today if its net quantity change for the day is zero.
    # Note: This simple logic assumes no overnight holdings of the same contract.
    # For more complex scenarios, cross-referencing with open positions would be needed.
    total_realized_pnl = 0
    closed_positions_details = []
    for conId, data in positions.items():
        if data['qty'] == 0:
            # The cost is negative for buys, positive for sells. Summing them gives the P&L.
            # We multiply by -1 because a net cost of -100 (bought for 100) and
            # a final value of +110 (sold for 110) should result in a profit of 10.
            pnl = data['cost'] * -1
            total_realized_pnl += pnl
            closed_positions_details.append({'symbol': data['symbol'], 'pnl': pnl})

    if not closed_positions_details:
        return "No positions were closed today.\n", 0.0

    # --- Build Report String ---
    report = f"{'Position':<25} {'Net P&L':>12}\n"
    report += "-" * 39 + "\n"
    for pos in sorted(closed_positions_details, key=lambda x: x['symbol']):
        pos_str = pos['symbol'][:25]
        pnl_str = f"${pos['pnl']:,.2f}"
        report += f"{pos_str:<25} {pnl_str:>12}\n"

    # --- Add Total Row ---
    report += "-" * 39 + "\n"
    total_pnl_str = f"${total_realized_pnl:,.2f}"
    report += f"{'TOTAL':<25} {total_pnl_str:>12}\n"

    return report, total_realized_pnl

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
    # Ledger is still needed for LTD stats and charts
    trade_df = get_trade_ledger_df()
    signals_df = get_model_signals_df()

    logger.info("--- Starting Daily Performance Analysis ---")

    try:
        # Use the current date for the report
        today_date = datetime.now().date()
        today_str = today_date.strftime('%Y-%m-%d')

        # Fetch live data from IB for all daily metrics
        account_summary = await get_live_account_data(config)
        total_daily_pnl = account_summary.get('daily_pnl') if account_summary else None
        live_portfolio = account_summary.get('portfolio') if account_summary else []
        todays_fills = account_summary.get('executions') if account_summary else []

        # --- Generate Report Sections and get P&L values from LIVE data ---
        open_positions_report, unrealized_pnl = generate_open_positions_report(live_portfolio)
        closed_positions_report, realized_pnl = generate_closed_positions_report(todays_fills)
        morning_signals_report = generate_morning_signals_report(signals_df, today_date)

        # Generate the executive summary with a mix of live daily data and historical ledger data
        exec_summary, pnl_for_title = generate_executive_summary(
            trade_df,           # Used for LTD stats
            signals_df,         # Used for LTD stats
            today_date,
            total_daily_pnl,    # Live from IB
            realized_pnl,       # Live from IB fills
            unrealized_pnl      # Live from IB portfolio
        )

        # --- Generate Charts ---
        # Do not generate charts if the ledger is empty to avoid errors
        chart_paths = []
        if not trade_df.empty:
            chart_paths = generate_performance_charts(trade_df, signals_df)
        else:
            logger.warning("Trade ledger is empty, skipping chart generation.")

        logger.info("--- Analysis Complete ---")

        return {
            "title": f"Daily Report: Total P&L ${pnl_for_title:,.2f}",
            "date": today_str,
            "reports": {
                "Exec. Summary": exec_summary,
                "Morning Signals": morning_signals_report,
                "Open Positions": open_positions_report,
                "Closed Positions": closed_positions_report
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
            send_pushover_notification(
                notification_config,
                f"Report Section: {title}",
                content,  # Content is already formatted
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
