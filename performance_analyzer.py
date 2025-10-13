"""Analyzes and reports the performance of trading activities.

This script reads the `trade_ledger.csv` file to calculate and summarize
the performance of trading strategies. It groups trades by their combo ID
to correctly attribute profit and loss for multi-leg positions. The script
generates a daily report that includes the net P&L for positions closed
that day and a list of all currently open positions.
"""

import pandas as pd
from datetime import datetime
import os
import logging
from logging_config import setup_logging
from notifications import send_pushover_notification

# --- Logging Setup ---
setup_logging()
logger = logging.getLogger("PerformanceAnalyzer")


def get_trade_ledger_df():
    """Reads and consolidates trade ledgers for analysis.

    This function first looks for the main `trade_ledger.csv`. If it exists,
    it reads it. If not, it scans the `archive` directory for all ledger files,
    combines them into a single DataFrame, and returns it. This ensures that
    performance analysis can run on the most recent (unarchived) data or on
    the entire history of archived data.

    Returns:
        pandas.DataFrame: A DataFrame containing all trade data, or None if
        no trade ledger files can be found.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive')

    if os.path.exists(ledger_path):
        logger.info("Reading main trade ledger...")
        return pd.read_csv(ledger_path)

    elif os.path.exists(archive_dir):
        logger.info("Main ledger not found. Reading from archive...")
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        if not archive_files:
            return None

        df_list = [pd.read_csv(file) for file in archive_files]
        return pd.concat(df_list, ignore_index=True)

    else:
        return None


def analyze_performance(config: dict) -> tuple[str, float] | None:
    """Analyzes the trade ledger to report on daily trading performance.

    Args:
        config (dict): The application configuration dictionary.

    Returns:
        A tuple containing the formatted report string and the total P&L,
        or None if analysis cannot be completed.
    """
    df = get_trade_ledger_df()
    if df is None or df.empty:
        logger.error("Trade ledger is empty or not found. Cannot analyze performance.")
        return None

    logger.info("--- Starting Daily Performance Analysis ---")

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        grouped = df.groupby('combo_id')
        total_pnl = 0
        closed_positions_summary = []
        open_positions_summary = []

        df['signed_quantity'] = df.apply(lambda row: -row['quantity'] if row['action'] == 'BUY' else row['quantity'], axis=1)

        for combo_id, group in grouped:
            leg_quantities = group.groupby('local_symbol')['signed_quantity'].sum()

            if (leg_quantities == 0).all():
                if group['timestamp'].dt.strftime('%Y-%m-%d').max() == today_str:
                    combo_pnl = group['total_value_usd'].sum()
                    total_pnl += combo_pnl
                    summary_line = (
                        f"  - Combo {combo_id}: Net P&L = ${combo_pnl:,.2f} "
                        f"(Closed {group['timestamp'].max().strftime('%H:%M')})"
                    )
                    closed_positions_summary.append(summary_line)
            else:
                entry_cost = -group['total_value_usd'].sum()
                position_details = [f"{row['action']} {int(row['quantity'])} {row['local_symbol']}" for _, row in group.iterrows()]
                summary_line = f"  - {' | '.join(position_details)} (Entry Cost: ${entry_cost:,.2f})"
                open_positions_summary.append(summary_line)

        report = f"<b>Trading Performance Report: {today_str}</b>\n\n"
        report += f"<b>Daily Net P&L: ${total_pnl:,.2f}</b>\n\n"
        
        if closed_positions_summary:
            report += "<b>Positions Closed Today:</b>\n" + "\n".join(closed_positions_summary) + "\n\n"
        else:
            report += "No positions were closed today.\n\n"

        if open_positions_summary:
            report += "<b>Currently Open Positions:</b>\n" + "\n".join(open_positions_summary)
        else:
            report += "No currently open positions."

        logger.info("--- Analysis Complete ---")
        print(report)
        
        return report, total_pnl

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)
        return None