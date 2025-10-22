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
    """Reads and consolidates the main and archived trade ledgers for analysis.

    This function combines data from `trade_ledger.csv` and all CSV files
    in the `archive/` directory into a single DataFrame. This ensures that
    performance analysis is always run on the complete history of trades.

    Returns:
        pandas.DataFrame: A DataFrame containing all trade data, or an empty
        DataFrame if no trade ledger files are found.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(base_dir, 'trade_ledger.csv')
    archive_dir = os.path.join(base_dir, 'archive')

    dataframes = []

    # Load the main ledger if it exists
    if os.path.exists(ledger_path):
        logger.info("Reading main trade ledger...")
        dataframes.append(pd.read_csv(ledger_path))

    # Load archived ledgers
    if os.path.exists(archive_dir):
        logger.info("Reading archived trade ledgers...")
        archive_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.startswith('trade_ledger_') and f.endswith('.csv')]
        if archive_files:
            df_list = [pd.read_csv(file) for file in archive_files]
            dataframes.extend(df_list)

    if not dataframes:
        logger.warning("No trade ledger data found.")
        return pd.DataFrame()

    # Combine all ledgers and sort by timestamp
    logger.info(f"Consolidating {len(dataframes)} ledger file(s).")
    full_ledger = pd.concat(dataframes, ignore_index=True)
    full_ledger['timestamp'] = pd.to_datetime(full_ledger['timestamp'])
    return full_ledger.sort_values(by='timestamp').reset_index(drop=True)


from trading_bot.performance_graphs import generate_performance_chart
from config_loader import load_config


def analyze_performance(config: dict) -> tuple[str, float, str | None] | None:
    """Analyzes the trade ledger to report on daily trading performance.

    Args:
        config (dict): The application configuration dictionary.

    Returns:
        A tuple containing the report string, total P&L, and chart path,
        or None if analysis fails.
    """
    df = get_trade_ledger_df()
    if df is None or df.empty:
        logger.error("Trade ledger is empty or not found. Cannot analyze performance.")
        return None

    logger.info("--- Starting Daily Performance Analysis ---")

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        today_date = datetime.now().date()
        today_str = today_date.strftime('%Y-%m-%d')
        
        grouped = df.groupby('combo_id')
        total_pnl = 0
        closed_positions_summary = []
        open_positions_summary = []

        # Create a signed value based on action for correct P&L and cost calculation
        df['signed_value_usd'] = df.apply(
            lambda row: -row['total_value_usd'] if row['action'] == 'BUY' else row['total_value_usd'],
            axis=1
        )
        df['signed_quantity'] = df.apply(lambda row: -row['quantity'] if row['action'] == 'BUY' else row['quantity'], axis=1)

        for combo_id, group in grouped:
            leg_quantities = group.groupby('local_symbol')['signed_quantity'].sum()

            # Check if the position is currently flat
            if (leg_quantities == 0).all():
                # To be considered "closed today", it must have trades today AND not have been closed before today.
                trades_today = group[group['timestamp'].dt.date == today_date]
                if not trades_today.empty:
                    trades_before_today = group[group['timestamp'].dt.date < today_date]

                    is_closed_before_today = False
                    if not trades_before_today.empty:
                        qtys_before_today = trades_before_today.groupby('local_symbol')['signed_quantity'].sum()
                        # Reindex to handle legs that only appeared today
                        qtys_before_today = qtys_before_today.reindex(leg_quantities.index, fill_value=0)
                        if (qtys_before_today == 0).all():
                            is_closed_before_today = True

                    if not is_closed_before_today:
                        # This position was officially closed today. P&L is the sum of all its transactions.
                        combo_pnl = group['signed_value_usd'].sum()
                        total_pnl += combo_pnl

                        if isinstance(combo_id, str) and '-' in combo_id:
                            underlying_symbol = combo_id.split('-')[0]
                            display_id = f"{underlying_symbol} ({combo_id})"
                        else:
                            display_id = f"Combo {combo_id}"

                        summary_line = (
                            f"  - {display_id}: Net P&L = ${combo_pnl:,.2f} "
                            f"(Closed {group['timestamp'].max().strftime('%H:%M')})"
                        )
                        closed_positions_summary.append(summary_line)
            else:
                # It's an open position. Summarize it correctly.
                entry_cost = group['signed_value_usd'].sum()

                position_details = []
                net_leg_quantities = group.groupby('local_symbol')['signed_quantity'].sum()
                for symbol, qty in net_leg_quantities.items():
                    if qty == 0:
                        continue
                    action = 'SELL' if qty > 0 else 'BUY'
                    position_details.append(f"{action} {int(abs(qty))} {symbol}")

                cost_type = "Credit" if entry_cost > 0 else "Debit"
                summary_line = f"  - {' | '.join(position_details)} (Net {cost_type}: ${abs(entry_cost):,.2f})"
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

        # --- Generate Performance Chart ---
        chart_path = None
        try:
            # Pass the dataframe to the chart generation function
            chart_path = generate_performance_chart(df)
            if chart_path:
                logger.info(f"Performance chart generated at: {chart_path}")
            else:
                logger.warning("Performance chart could not be generated.")
        except Exception as e:
            # The original traceback is preserved with exc_info=True
            logger.error(f"Error generating performance chart: {e}", exc_info=True)


        logger.info("--- Analysis Complete ---")
        print(report) # Keep for console output
        
        return report, total_pnl, chart_path

    except Exception as e:
        logger.error(f"An error occurred during performance analysis: {e}", exc_info=True)
        return None

def main():
    """
    Main function to run the performance analysis and send a notification.
    """
    config = load_config()
    if not config:
        logger.critical("Failed to load configuration. Exiting.")
        return

    analysis_result = analyze_performance(config)

    if analysis_result:
        report, total_pnl, chart_path = analysis_result
        title = f"Daily Report: P&L ${total_pnl:,.2f}"

        # Send notification with the chart if available
        send_pushover_notification(
            config.get('notifications', {}),
            title,
            report,
            attachment_path=chart_path
        )
    else:
        # Send a failure notification
        send_pushover_notification(
            config.get('notifications', {}),
            "Performance Analysis FAILED",
            "The performance analysis script failed to run. Check logs for details."
        )

if __name__ == "__main__":
    main()