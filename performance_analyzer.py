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


def generate_executive_summary(trade_df: pd.DataFrame, signals_df: pd.DataFrame, today_date: datetime.date) -> tuple[str, float]:
    """Generates Section 1: The Executive Summary."""

    # --- Helper function to calculate metrics for a given period ---
    def calculate_metrics(trades, signals):
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
        # Trades executed should be based on the number of opening trades, not the total number of trades
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

    # --- Calculate for Today and Life-to-Date ---
    today_trades = trade_df[trade_df['timestamp'].dt.date == today_date]
    today_signals = signals_df[signals_df['timestamp'].dt.date == today_date]

    today_metrics = calculate_metrics(today_trades, today_signals)
    ltd_metrics = calculate_metrics(trade_df, signals_df)

    # --- Build Monospaced Text Report Table ---
    report = "Section 1: Exec. Summary\n"
    report += f"{'Metric':<18} {'Today':>12} {'LTD':>12}\n"
    report += "-" * 44 + "\n"

    # --- Rows (with abbreviated metric names) ---
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

    report += "\n"

    return report, today_metrics['pnl']


def generate_model_performance_report(trade_df: pd.DataFrame, signals_df: pd.DataFrame, today_date: datetime.date) -> tuple[str, float]:
    """Generates Section 2: Model Performance & Attribution."""
    today_signals = signals_df[signals_df['timestamp'].dt.date == today_date].copy()
    today_trades = trade_df[trade_df['timestamp'].dt.date == today_date].copy()

    if today_signals.empty:
        return "Section 2: Model Performance\nNo model signals for today.\n\n", 0.0

    # 1. Process trade data to get P&L per position
    if not today_trades.empty:
        combo_legs_map = today_trades.groupby('combo_id')['local_symbol'].unique().apply(lambda legs: tuple(sorted(legs)))
        today_trades['position_id'] = today_trades['combo_id'].map(combo_legs_map)

        closed_positions = today_trades.groupby('position_id').filter(lambda x: (x['action'].eq('BUY').astype(int) - x['action'].eq('SELL').astype(int)).sum() == 0)
        pnl_per_position = closed_positions.groupby('position_id')['total_value_usd'].sum().reset_index()
        pnl_per_position = pnl_per_position.rename(columns={'total_value_usd': 'net_pnl'})

        # Get exit reason and contract for each position
        def get_position_details(group):
            exit_reason = group[group['reason'] != 'Strategy Execution']['reason'].iloc[0] if not group[group['reason'] != 'Strategy Execution'].empty else 'N/A'
            # Extract underlying, forcing 'KC' as the prefix because the ledger incorrectly uses 'KO'
            local_sym_prefix = group['local_symbol'].iloc[0].split(' ')[0][:4]
            contract = "KC" + local_sym_prefix[2:]
            return pd.Series({'exit_reason': exit_reason, 'contract': contract})

        position_details = closed_positions.groupby('position_id').apply(get_position_details).reset_index()
        pnl_per_position = pd.merge(pnl_per_position, position_details, on='position_id')

    else:
        pnl_per_position = pd.DataFrame(columns=['position_id', 'net_pnl', 'exit_reason', 'contract'])

    # 2. Join signal data with trade data
    def get_contract_prefix(contract_string):
        # Extracts 'KOM6' from 'KOM6 C1.23'
        return contract_string.split(' ')[0][:4]

    pnl_per_position['contract_prefix'] = pnl_per_position['contract'].apply(get_contract_prefix)

    # The signal 'contract' is like '202512', need to map to month codes
    month_code_map = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    today_signals['month'] = pd.to_datetime(today_signals['contract'], format='%Y%m').dt.month
    today_signals['year_last_digit'] = pd.to_datetime(today_signals['contract'], format='%Y%m').dt.strftime('%y').str[-1]
    # Corrected 'KO' to 'KC'
    today_signals['signal_contract_prefix'] = "KC" + today_signals['month'].map(month_code_map) + today_signals['year_last_digit']

    report_df = pd.merge(today_signals, pnl_per_position, left_on='signal_contract_prefix', right_on='contract_prefix', how='left')

    # 3. Populate report columns
    report_df['signal'] = report_df['signal'].str.replace('BEA', 'BEARISH', case=False)
    report_df['Trade Executed?'] = report_df['net_pnl'].notna().map({True: 'Yes', False: 'No'})
    report_df['Position'] = report_df['position_id'].apply(lambda x: ' / '.join(x) if isinstance(x, tuple) else 'N/A')
    report_df['Net P&L'] = report_df['net_pnl'].fillna(0)
    report_df['Exit Reason'] = report_df['exit_reason'].fillna('N/A')

    def determine_signal_hit(row):
        if row['Trade Executed?'] == 'No':
            return 'N/A'
        return 'Yes' if row['Net P&L'] > 0 else 'No'
        
    report_df['Signal Hit?'] = report_df.apply(determine_signal_hit, axis=1)

    # --- Build Monospaced Text Report Table ---
    report = "Section 2: Model Performance\n"
    header = f"{'Contract':<9} {'Signal':<9} {'Exec?':<6} {'P&L':>9} {'Exit':<10} {'Hit?':<5}"
    report += header + "\n"
    report += "-" * len(header) + "\n"

    for _, row in report_df.iterrows():
        pnl_str = f"${row['Net P&L']:,.0f}" # Abbreviated to no decimals
        executed_str = row['Trade Executed?']
        # Abbreviate Exit Reason
        exit_reason_str = row['Exit Reason'].replace('Execution', 'Exec').replace('Management', 'Mgmt')
        signal_hit_str = row['Signal Hit?']
        signal_str = row['signal'].replace('BEARISH', 'BEAR') # Abbreviate signal

        report += (f"{row['contract_x']:<9} {signal_str:<9} {executed_str:<6} "
                   f"{pnl_str:>9} {exit_reason_str:<10} {signal_hit_str:<5}\n")

    report += "\n"

    model_pnl_sum = report_df['Net P&L'].sum()

    return report, model_pnl_sum


async def generate_system_status_report(trade_df: pd.DataFrame, config: dict) -> tuple[str, bool]:
    """Generates Section 3: System Status Check."""
    report = "Section 3: System Status\n"
    is_ok = True

    # 1. Position Check
    trade_df['signed_quantity'] = trade_df.apply(
        lambda row: row['quantity'] if row['action'] == 'SELL' else -row['quantity'], axis=1
    )
    net_positions = trade_df.groupby('local_symbol')['signed_quantity'].sum()
    open_positions = net_positions[net_positions != 0]

    if not open_positions.empty:
        report += "!! WARNING: OPEN POSITIONS !!\n"
        for symbol, qty in open_positions.items():
            action = "SELL" if qty > 0 else "BUY"
            report += f"- {action} {int(abs(qty))} {symbol}\n"
        is_ok = False
    else:
        report += "Position Check: PASS (All flat)\n"

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

async def analyze_performance(config: dict) -> tuple[str, float, list[str]] | None:
    """
    Analyzes the trade ledger to report on daily trading performance.
    Orchestrates the generation of all report sections.
    """
    trade_df = get_trade_ledger_df()
    signals_df = get_model_signals_df()

    if trade_df.empty:
        logger.error("Trade ledger is empty or not found. Cannot analyze performance.")
        return None

    logger.info("--- Starting Daily Performance Analysis ---")

    try:
        today_date = trade_df['timestamp'].max().date()
        today_str = today_date.strftime('%Y-%m-%d')

        # --- Generate Sections ---
        summary_report, daily_pnl = generate_executive_summary(trade_df, signals_df, today_date)
        model_report, model_pnl = generate_model_performance_report(trade_df, signals_df, today_date)
        status_report, is_status_ok = await generate_system_status_report(trade_df, config)

        # --- Data Integrity Check ---
        integrity_check_msg = "\nData Integrity Check: "
        if not is_status_ok or abs(daily_pnl - model_pnl) > 0.01: # Using a small tolerance for float comparison
            integrity_check_msg += "FAIL"
            logger.error(f"Data integrity check failed: Exec Summary P&L (${daily_pnl}) != Model Perf P&L (${model_pnl})")
        else:
            integrity_check_msg += "PASS"


        # --- Assemble Final Report ---
        full_report = f"Trading Performance Report: {today_str}\n\n"
        full_report += summary_report
        full_report += model_report
        full_report += status_report
        full_report += integrity_check_msg

        # --- Generate Performance Charts ---
        chart_paths = []
        try:
            chart_paths = generate_performance_charts(trade_df, signals_df) # Updated function call
            if chart_paths:
                logger.info(f"Performance charts generated.")
            else:
                logger.warning("Performance charts could not be generated.")
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}", exc_info=True)


        logger.info("--- Analysis Complete ---")
        print(full_report)
        
        return full_report, daily_pnl, chart_paths

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)
        return None

async def main():
    """
    Main function to run the performance analysis and send a notification.
    """
    config = load_config()
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        return

    analysis_result = await analyze_performance(config)

    if analysis_result:
        report, total_pnl, chart_paths = analysis_result
        title = f"Daily Report: P&L ${total_pnl:,.2f}"

        # --- Truncate report if it exceeds Pushover's limit (1024 chars) ---
        if len(report) > 1024:
            report = report[:1000] + "\n... (report truncated)"

        # Send the main report with monospaced formatting
        send_pushover_notification(
            config.get('notifications', {}),
            title,
            report,
            monospace=True # Enable monospaced formatting for the report
        )
        # --- Send Charts (one by one to avoid message size issues) ---
        for i, chart_path in enumerate(chart_paths):
            chart_title = os.path.splitext(os.path.basename(chart_path))[0].replace('_', ' ').title()
            send_pushover_notification(
                config.get('notifications', {}),
                f"Chart {i+1}/{len(chart_paths)}: {chart_title}",
                f"See attached chart: {chart_title}",
                attachment_path=chart_path
            )
    else:
        send_pushover_notification(
            config.get('notifications', {}),
            "Performance Analysis FAILED",
            "The performance analysis script failed to run. Check logs for details."
        )

if __name__ == "__main__":
    asyncio.run(main())
