import pandas as pd
from datetime import datetime
import os

def analyze_daily_performance():
    """
    Analyzes the trade ledger to provide a summary of the day's trading performance,
    broken down by strategy and underlying symbol.
    """
    ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_ledger.csv')
    if not os.path.exists(ledger_path):
        print("Trade ledger not found.")
        return

    print("--- Daily Performance Analysis ---")
    
    try:
        df = pd.read_csv(ledger_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"Analyzing trades for: {today_str}\n")
        
        today_df = df[df['timestamp'].dt.strftime('%Y-%m-%d') == today_str].copy()

        if today_df.empty:
            print("No trades executed today.")
            return
            
        # --- Correct P&L Calculation ---
        # BUY action is a debit (cash out, negative value)
        # SELL action is a credit (cash in, positive value)
        today_df['cash_flow'] = today_df.apply(
            lambda row: -row['total_value_usd'] if row['action'] == 'BUY' else row['total_value_usd'],
            axis=1
        )
        total_pnl = today_df['cash_flow'].sum()

        print(f"** Overall Summary **")
        print(f"Total Net P&L (Cash Flow) for Today: ${total_pnl:,.2f}")
        print(f"Total Trades Executed: {len(today_df)}")
        print("-" * 40)
        
        # --- Breakdown by Strategy ---
        print("\n** P&L Breakdown by Strategy **")
        strategy_pnl = today_df.groupby('strategy_type')['cash_flow'].sum()
        strategy_counts = today_df['strategy_type'].value_counts()
        strategy_summary = pd.DataFrame({
            'Net P&L': strategy_pnl,
            'Trade Count': strategy_counts
        })
        print(strategy_summary.to_string())
        print("-" * 40)

        # --- Breakdown by Underlying ---
        print("\n** P&L Breakdown by Underlying Symbol **")
        underlying_pnl = today_df.groupby('underlying_symbol')['cash_flow'].sum()
        underlying_counts = today_df['underlying_symbol'].value_counts()
        underlying_summary = pd.DataFrame({
            'Net P&L': underlying_pnl,
            'Trade Count': underlying_counts
        })
        print(underlying_summary.to_string())
        print("-" * 40)
        
        # --- Detailed Trade Log ---
        print("\n** Today's Detailed Trade Log **")
        display_cols = ['timestamp', 'underlying_symbol', 'strategy_type', 'strikes', 
                        'action', 'quantity', 'avg_fill_price', 'cash_flow', 'reason']
        print(today_df[display_cols].to_string(index=False))
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    analyze_daily_performance()
