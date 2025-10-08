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


def analyze_performance(config: dict):
    """Analyzes the trade ledger to report on daily trading performance.

    This function reads the trade ledger, calculates the total profit or loss
    for all combo positions that were closed on the current day, and identifies
    all currently open positions. It then formats this information into a
    report and sends it as a Pushover notification.

    The P&L for a combo is calculated by summing the `total_value_usd` for
    all its legs. A position is considered closed if the sum of its signed
    quantities (where buys are negative and sells are positive) for each
    leg is zero.

    Args:
        config (dict): The application configuration dictionary, used for
            sending notifications.
    """
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    if not os.path.exists(ledger_path):
        logger.error("Trade ledger not found. Cannot analyze performance.")
        return

    logger.info("--- Starting Daily Performance Analysis ---")
    
    try:
        df = pd.read_csv(ledger_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # --- Analyze all trades by grouping them into combos ---
        grouped = df.groupby('combo_id')

        total_pnl = 0
        closed_positions_summary = []
        open_positions_summary = []

        # Use a signed quantity to determine if a position is open or closed
        # BUY actions decrease position (cost), SELL actions increase it (credit)
        # So we treat BUY as negative and SELL as positive to sum up to zero.
        df['signed_quantity'] = df.apply(lambda row: -row['quantity'] if row['action'] == 'BUY' else row['quantity'], axis=1)

        for combo_id, group in grouped:
            # A position is closed if the quantities for each leg cancel out.
            leg_quantities = group.groupby('local_symbol')['signed_quantity'].sum()

            if (leg_quantities == 0).all():
                # --- This is a closed position ---
                # Include it in today's P&L report if it was closed today.
                if group['timestamp'].dt.strftime('%Y-%m-%d').max() == today_str:
                    combo_pnl = group['total_value_usd'].sum()
                    total_pnl += combo_pnl

                    summary_line = (
                        f"  - Combo {combo_id}: Net P&L = ${combo_pnl:,.2f} "
                        f"(Closed {group['timestamp'].max().strftime('%H:%M')})"
                    )
                    closed_positions_summary.append(summary_line)
            else:
                # --- This is an open position ---
                entry_cost = -group['total_value_usd'].sum()
                position_details = []
                for _, row in group.iterrows():
                    position_details.append(f"{row['action']} {int(row['quantity'])} {row['local_symbol']}")

                summary_line = f"  - {' | '.join(position_details)} (Entry Cost: ${entry_cost:,.2f})"
                open_positions_summary.append(summary_line)

        # --- Construct the final report ---
        report = f"<b>Trading Performance Report: {today_str}</b>\n\n"
        report += f"<b>Daily Net P&L: ${total_pnl:,.2f}</b>\n\n"
        
        if closed_positions_summary:
            report += "<b>Positions Closed Today:</b>\n"
            report += "\n".join(closed_positions_summary)
            report += "\n\n"
        else:
            report += "No positions were closed today.\n\n"

        if open_positions_summary:
            report += "<b>Currently Open Positions:</b>\n"
            report += "\n".join(open_positions_summary)
        else:
            report += "No currently open positions."

        logger.info("--- Analysis Complete ---")
        print(report) # Print report to console/log
        
        # --- Send Notification ---
        send_pushover_notification(
            config.get('notifications', {}),
            title=f"Daily Report: P&L ${total_pnl:,.2f}",
            message=report
        )

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)