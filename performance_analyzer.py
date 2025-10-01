import pandas as pd
from datetime import datetime
import os

def analyze_daily_performance():
    """
    Analyzes the trade ledger to provide a summary of the day's trading performance,
    broken down by strategy and risk management actions.
    """
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    if not os.path.exists(ledger_path):
        print("Trade ledger not found. No analysis to perform.")
        return

    print("--- Daily Performance Analysis ---")
    
    try:
        df = pd.read_csv(ledger_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter for today's trades
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"Analyzing trades for: {today_str}\n")
        
        today_df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == today_str].copy()

        if today_df.empty:
            print("No trades executed today.")
            return
            
        # --- Overall P&L Calculation ---
        # To calculate P&L, we pair opening ('Strategy Execution') and closing trades.
        # This is a simplified approach assuming FIFO (First-In, First-Out) for positions.
        
        total_pnl = 0
        
        # Separate opening and closing trades
        opened_trades = today_df[today_df['reason'] == 'Strategy Execution'].copy()
        closed_trades = today_df[today_df['reason'].isin(['Stop-Loss', 'Take-Profit', 'Position Misaligned'])].copy()
        
        # Simplified P&L: Sum of all transaction values
        # Note: A more accurate P&L would track individual positions.
        # For combo trades, 'BUY' can be a debit or credit, so we check action and value.
        
        # A SELL action on a combo is typically opening a credit spread (positive cashflow)
        # A BUY action is typically opening a debit spread (negative cashflow)
        today_df['signed_value'] = today_df.apply(
            lambda row: row['total_value_usd'] if row['action'] == 'SELL' else -row['total_value_usd'],
            axis=1
        )
        total_pnl = today_df['signed_value'].sum()

        print(f"** Overall Summary **")
        print(f"Total Net P&L for Today: ${total_pnl:,.2f}")
        print(f"Total Trades Executed: {len(today_df)}")
        print("-" * 30)
        
        # --- Breakdown by Reason ---
        print("** Breakdown by Trade Reason **")
        reason_counts = today_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"- {reason}: {count} trade(s)")

        print("\n" + "-" * 30)
        
        # --- Detailed Trade Log ---
        print("** Today's Trade Log **")
        print(today_df[['timestamp', 'local_symbol', 'action', 'quantity', 'avg_fill_price', 'reason']].to_string(index=False))
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    analyze_daily_performance()
