import pandas as pd
from datetime import datetime
import os
import logging

# This is a bit of a hack to make sure the custom module can be found
# when running tests from the root directory.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from notifications import send_pushover_notification
from config_loader import load_config

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PerformanceAnalyzer")


def analyze_performance(config: dict):
    """
    Analyzes the trade ledger to provide a summary of the day's trading performance.
    It groups trades by combo_id to correctly calculate P&L for multi-leg strategies.
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

        # Adjust quantity based on action for accurate P&L and position state calculation
        df['signed_quantity'] = df.apply(lambda row: -row['quantity'] if row['action'] == 'BUY' else row['quantity'], axis=1)

        for combo_id, group in grouped:
            # Check if a position is closed by seeing if quantities cancel out
            leg_quantities = group.groupby('local_symbol')['signed_quantity'].sum()

            if (leg_quantities == 0).all():
                # --- This is a closed position ---
                # Check if the closing trade was today
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
        if config:
            send_pushover_notification(
                config.get('notifications', {}),
                title=f"Daily Report: P&L ${total_pnl:,.2f}",
                message=report
            )

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)


if __name__ == "__main__":
    config = load_config()
    if config:
        analyze_performance(config)