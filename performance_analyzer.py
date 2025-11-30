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

# --- Constants ---
DEFAULT_STARTING_CAPITAL = 250000


def get_trade_ledger_df():
    """Reads and consolidates the main and archived trade ledgers for analysis."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive_ledger')

    dataframes = []
    logger.info("--- Consolidating Trade Ledgers ---")

    if os.path.exists(ledger_path):
        logger.info(f"Loading main trade ledger: {os.path.basename(ledger_path)}")
        dataframes.append(pd.read_csv(ledger_path))
    else:
        logger.warning("Main trade_ledger.csv not found.")

    if os.path.exists(archive_dir):
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        if archive_files:
            logger.info(f"Found {len(archive_files)} archived trade ledger(s).")
            for file in archive_files:
                logger.info(f"Loading archived ledger: {os.path.basename(file)}")
                dataframes.append(pd.read_csv(file))
        else:
            logger.info("No archived trade ledgers found.")
    else:
        logger.info("Archive directory not found, skipping.")


    if not dataframes:
        logger.warning("No trade ledger data found to consolidate.")
        return pd.DataFrame()

    full_ledger = pd.concat(dataframes, ignore_index=True)
    full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp'])

    # Coerce P&L column to numeric, turning any non-numeric values into NaN
    full_ledger['total_value_usd'] = pd.to_numeric(full_ledger['total_value_usd'], errors='coerce')

    logger.info(f"Consolidated a total of {len(full_ledger)} trade records.")
    return full_ledger.sort_values(by='timestamp').reset_index(drop=True)


async def get_live_account_data(config: dict) -> dict | None:
    """
    Connects to IB and fetches all live data needed for the daily report:
    - Open positions from the portfolio.
    - Today's trade executions (fills) and their commission reports.
    - Net Liquidation Value (Account Summary).
    """
    ib = IB()
    live_data = {}
    conn_settings = config.get('connection', {})

    try:
        await ib.connectAsync(
            host=conn_settings.get('host', '127.0.0.1'),
            port=conn_settings.get('port', 7497),
            clientId=random.randint(200, 2000),
            timeout=15
        )

        # 1. Fetch Account Summary (Net Liquidation Value)
        account_summary = await ib.accountSummaryAsync()
        net_liquidation = next((float(v.value) for v in account_summary if v.tag == 'NetLiquidation'), None)
        if net_liquidation:
             logger.info(f"Fetched Net Liquidation Value: ${net_liquidation:,.2f}")
        else:
             logger.warning("Could not fetch Net Liquidation Value.")

        # 2. Fetch open positions (PortfolioItems have P&L data)
        # Wait a moment for portfolio data to stream in after connection.
        await asyncio.sleep(2)
        portfolio_items = ib.portfolio()

        # 3. Fetch today's executions for live realized P&L calculation
        fills = await ib.reqExecutionsAsync()
        today_date = datetime.now().date()
        todays_fills = [f for f in fills if f.time.date() == today_date]
        logger.info(f"Successfully fetched {len(todays_fills)} execution(s) for today from IB.")

        live_data = {
            'portfolio': portfolio_items,
            'executions': todays_fills,
            'net_liquidation': net_liquidation
        }

    except asyncio.TimeoutError:
        logger.error("Connection to IB timed out during live data fetch.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Failed to get live account data from IB: {e}", exc_info=True)
        return None
    finally:
        if ib.isConnected():
            ib.disconnect()

    return live_data

def generate_executive_summary(
    trade_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    today_date: datetime.date,
    todays_fills: list,
    total_daily_pnl: float | None,
    realized_daily_pnl: float,
    unrealized_daily_pnl: float,
    ltd_total_pnl: float | None = None
) -> tuple[str, float]:
    """Generates Section 1: The Executive Summary, separating Realized and Unrealized P&L."""

    # --- Helper function to calculate metrics from the trade ledger ---
    def calculate_ledger_metrics(trades: pd.DataFrame, signals: pd.DataFrame):
        if trades.empty:
            return {"pnl": 0, "trades_executed": 0, "win_rate": 0}

        trades['position_id'] = trades.apply(lambda row: tuple(sorted(str(row['combo_id']).split(','))), axis=1)
        closed_positions = trades.groupby('position_id').filter(lambda x: x['action'].eq('BUY').count() == x['action'].eq('SELL').count())
        pnl_per_position = closed_positions.groupby('position_id')['total_value_usd'].sum()

        net_pnl = pnl_per_position.sum()
        trades_executed = len(pnl_per_position)
        win_rate = (pnl_per_position > 0).mean() if trades_executed > 0 else 0

        return {"pnl": net_pnl, "trades_executed": trades_executed, "win_rate": win_rate}

    # --- Calculate metrics ---
    today_signals = signals_df[signals_df['timestamp'].dt.date == today_date]
    signals_fired_today = len(today_signals[today_signals['signal'] != 'NEUTRAL'])
    signals_fired_ltd = len(signals_df[signals_df['signal'] != 'NEUTRAL'])

    # LTD metrics are always from the ledger (realized P&L)
    ltd_metrics = calculate_ledger_metrics(trade_df, signals_df)

    # Today's trades executed count is from live fills
    trades_executed_today = len(set(f.execution.permId for f in todays_fills))

    # Win rate for today can't be calculated without P&L for each live trade, so we omit it.

    execution_rate_today = trades_executed_today / signals_fired_today if signals_fired_today > 0 else 0
    execution_rate_ltd = ltd_metrics['trades_executed'] / signals_fired_ltd if signals_fired_ltd > 0 else 0

    # --- Build Report ---
    report = f"{'Metric':<18} {'Today':>12} {'LTD':>12}\n"
    report += "-" * 44 + "\n"

    # P&L rows are handled separately to show the breakdown
    today_total_pnl_str = f"${total_daily_pnl:,.2f}" if total_daily_pnl is not None else "N/A"
    today_realized_pnl_str = f"${realized_daily_pnl:,.2f}"
    today_unrealized_pnl_str = f"${unrealized_daily_pnl:,.2f}"

    # Calculate LTD values
    # If ltd_total_pnl (Equity P&L) is provided, use it. Otherwise fallback to ledger realized.
    if ltd_total_pnl is not None:
        ltd_total_pnl_str = f"${ltd_total_pnl:,.2f}"
        # Derived Unrealized = Total (Equity) - Realized (Ledger)
        # Note: This might not match live portfolio sum exactly due to fees/timing, but it balances the report.
        ltd_unrealized_val = ltd_total_pnl - ltd_metrics['pnl']
        ltd_unrealized_pnl_str = f"${ltd_unrealized_val:,.2f}"
    else:
        ltd_total_pnl_str = f"${ltd_metrics['pnl']:,.2f}"
        ltd_unrealized_pnl_str = "$0.00"

    ltd_realized_pnl_str = f"${ltd_metrics['pnl']:,.2f}"

    pnl_rows = {
        "Total P&L": (today_total_pnl_str, ltd_total_pnl_str),
        " Realized P&L": (today_realized_pnl_str, ltd_realized_pnl_str),
        " Unrealized P&L": (today_unrealized_pnl_str, ltd_unrealized_pnl_str),
    }
    for metric, (today_val, ltd_val) in pnl_rows.items():
        report += f"{metric:<18} {today_val:>12} {ltd_val:>12}\n"

    report += "-" * 44 + "\n"

    # Other metric rows
    other_rows = {
        "Signals Fired": (f"{signals_fired_today}", f"{signals_fired_ltd}"),
        "Trades Executed": (f"{trades_executed_today}", f"{ltd_metrics['trades_executed']}"),
        "Exec. Rate": (f"{execution_rate_today:.1%}", f"{execution_rate_ltd:.1%}"),
        "Win Rate": ("N/A", f"{ltd_metrics['win_rate']:.1%}"),
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
    Creates a summary of all currently open positions, grouped by contract month,
    and calculates the total unrealized P&L.
    """
    open_positions = [p for p in portfolio if p.position != 0]

    if not open_positions:
        return "No open positions.\n", 0.0

    # Group positions by the root of the local symbol (e.g., 'KOZ5')
    grouped_positions = {}
    for pos in open_positions:
        # Extract the common root (e.g., 'KOZ5' from 'KOZ5 C4.15')
        root_symbol = pos.contract.localSymbol.split(' ')[0]
        if root_symbol not in grouped_positions:
            grouped_positions[root_symbol] = []
        grouped_positions[root_symbol].append(pos)

    report = f"{'Symbol':<25} {'Qty':>5} {'Avg Cost':>10} {'Unreal. P&L':>15}\n"
    report += "-" * 57 + "\n"
    total_unrealized_pnl = 0

    # Sort groups by symbol for consistent ordering
    for root_symbol in sorted(grouped_positions.keys()):
        positions_in_group = grouped_positions[root_symbol]
        subtotal_pnl = 0
        
        # Sort positions within the group for readability
        for pos in sorted(positions_in_group, key=lambda p: p.contract.localSymbol):
            symbol = pos.contract.localSymbol
            qty = int(pos.position)
            avg_cost = f"${pos.averageCost:,.2f}"
            unreal_pnl = pos.unrealizedPNL if isinstance(pos.unrealizedPNL, float) else 0.0
            unreal_pnl_str = f"${unreal_pnl:,.2f}"

            report += f"{symbol:<25} {qty:>5} {avg_cost:>10} {unreal_pnl_str:>15}\n"
            total_unrealized_pnl += unreal_pnl
            subtotal_pnl += unreal_pnl

        # Add a subtotal for the group
        subtotal_pnl_str = f"${subtotal_pnl:,.2f}"
        report += f"{'':<42} {'-'*15}\n"
        report += f"{'Subtotal for ' + root_symbol:<42} {subtotal_pnl_str:>15}\n"
        report += "\n" # Add a blank line for spacing between groups

    # --- Add Grand Total Row ---
    report += "=" * 57 + "\n"
    total_pnl_str = f"${total_unrealized_pnl:,.2f}"
    report += f"{'GRAND TOTAL':<42} {total_pnl_str:>15}\n"

    return report, total_unrealized_pnl

def generate_closed_positions_report(fills: list) -> tuple[str, float]:
    """
    Creates a report of positions closed today from live execution data from IB.
    Calculates realized P&L from the commission reports of each fill.
    """
    if not fills:
        return "No trades resulting in a closed position today.\n", 0.0

    # Sum realized P&L directly from commission reports. This is the source of truth.
    total_realized_pnl = sum(f.commissionReport.realizedPNL for f in fills if f.commissionReport and f.commissionReport.realizedPNL != 0.0)

    # For the report body, group fills by contract to show P&L per position.
    positions = {}
    for f in fills:
        if f.commissionReport and f.commissionReport.realizedPNL != 0.0:
            symbol = f.contract.localSymbol
            if symbol not in positions:
                positions[symbol] = 0.0
            positions[symbol] += f.commissionReport.realizedPNL

    if not positions:
        return "No trades resulting in a closed position today.\n", total_realized_pnl

    # --- Build Report String ---
    report = f"{'Position':<25} {'Net P&L':>12}\n"
    report += "-" * 39 + "\n"
    # Sort by symbol for consistent ordering
    for symbol, pnl in sorted(positions.items()):
        pos_str = symbol[:25]
        pnl_str = f"${pnl:,.2f}"
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
        today_date = datetime.now().date()
        today_str = today_date.strftime('%Y-%m-%d')

        # Fetch live data from IB for all daily metrics
        live_data = await get_live_account_data(config)
        live_portfolio = live_data.get('portfolio') if live_data else []
        todays_fills = live_data.get('executions') if live_data else []
        live_net_liq = live_data.get('net_liquidation') if live_data else None

        # Load equity history if available
        equity_df = pd.DataFrame()
        equity_file = os.path.join("data", "daily_equity.csv")
        starting_capital = DEFAULT_STARTING_CAPITAL

        if os.path.exists(equity_file):
            logger.info("Loading daily_equity.csv for equity curve.")
            equity_df = pd.read_csv(equity_file)
            if not equity_df.empty:
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                equity_df = equity_df.sort_values('timestamp')
                starting_capital = equity_df.iloc[0]['total_value_usd']
                logger.info(f"Dynamic Starting Capital from History: ${starting_capital:,.2f}")
            else:
                logger.warning("daily_equity.csv is empty, using default starting capital.")
        else:
            logger.warning(f"daily_equity.csv not found, using default starting capital: ${starting_capital:,.2f}")

        # --- Generate Report Sections and get P&L values from LIVE data ---
        open_positions_report, unrealized_pnl = generate_open_positions_report(live_portfolio)
        closed_positions_report, realized_pnl = generate_closed_positions_report(todays_fills)
        morning_signals_report = generate_morning_signals_report(signals_df, today_date)

        # Calculate Total P&L as the sum of its parts for consistency
        total_daily_pnl = realized_pnl + unrealized_pnl

        # Calculate LTD Total P&L from Equity (if available)
        ltd_total_pnl = None
        if live_net_liq:
            ltd_total_pnl = live_net_liq - starting_capital
            logger.info(f"Calculated LTD Total P&L (Equity): ${ltd_total_pnl:,.2f}")
        elif not equity_df.empty:
            # Fallback to last recorded equity
            last_equity = equity_df['total_value_usd'].iloc[-1]
            ltd_total_pnl = last_equity - starting_capital
            logger.warning("Using last recorded equity for LTD P&L (Live NetLiq unavailable).")

        # Generate the executive summary with a mix of live daily data and historical ledger data
        exec_summary, pnl_for_title = generate_executive_summary(
            trade_df,
            signals_df,
            today_date,
            todays_fills,
            total_daily_pnl,          # Sum of realized and unrealized
            realized_pnl,       # Live from IB fills
            unrealized_pnl,      # Live from IB portfolio
            ltd_total_pnl       # Equity-based LTD P&L
        )

        # --- Generate Charts ---
        # Do not generate charts if the ledger is empty to avoid errors
        chart_paths = []
        if not trade_df.empty:
            chart_paths = generate_performance_charts(trade_df, signals_df, equity_df, starting_capital)
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
