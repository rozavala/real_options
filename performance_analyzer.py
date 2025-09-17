import asyncio
import os
import random
from collections import deque

import pandas as pd
from ib_insync import *


def analyze_ledger_file(ledger_path):
    """
    Analyzes the trade_ledger.csv file and returns calculated P&L and
    a list of positions that the ledger *thinks* are still open.
    """
    if not os.path.isfile(ledger_path):
        print("Trade ledger file not found. No historical trades to analyze.")
        return [], 0, {}

    try:
        # Add on_bad_lines='warn' to skip corrupted rows instead of crashing.
        df = pd.read_csv(ledger_path, on_bad_lines='warn')
        print(f"Loaded {len(df)} transactions from the ledger.")

        # Make timestamp parsing more robust to handle different formats
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        
        # Drop any rows where the timestamp could not be parsed
        original_rows = len(df)
        df = df.dropna(subset=['timestamp'])
        if len(df) < original_rows:
            print(f"Warning: Dropped {original_rows - len(df)} rows due to unparseable timestamps.")

        df = df.sort_values(by='timestamp').reset_index(drop=True)

        completed_trades = []
        open_positions = {}
        total_realized_pnl = 0

        # Use FIFO logic to calculate P&L
        for symbol, group in df.groupby('local_symbol'):
            position_queue = deque()
            for index, row in group.iterrows():
                if row['action'] in ['SELL', 'SLD']:
                    position_queue.append(row)
                elif row['action'] in ['BUY', 'BOT']:
                    if not position_queue:
                        continue
                    open_trade = position_queue.popleft()
                    pnl = open_trade['total_value_usd'] - row['total_value_usd']
                    total_realized_pnl += pnl
                    completed_trades.append({
                        'Symbol': symbol,
                        'Open Time': open_trade['timestamp'], # Keep as datetime for sorting
                        'Close Time': row['timestamp'],   # Keep as datetime for sorting
                        'P&L (USD)': f"${pnl:,.2f}"
                    })
            if position_queue:
                open_positions[symbol] = len(position_queue)
        
        return completed_trades, total_realized_pnl, open_positions
    except Exception as e:
        print(f"An error occurred during ledger analysis: {e}")
        return [], 0, {}


async def main():
    """
    Main function to run ledger analysis and then connect to IBKR for
    live position and P&L reconciliation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(script_dir, 'trade_ledger.csv')
    
    # --- 1. Analyze the Local Ledger File ---
    completed_trades, total_pnl_ledger, ledger_open_positions = analyze_ledger_file(ledger_path)

    print("\n--- Strategy Performance Summary (from Ledger) ---")
    if not completed_trades:
        print("No completed round-trip trades found in ledger.")
    else:
        summary_df = pd.DataFrame(completed_trades)
        summary_df = summary_df.sort_values(by='Open Time').reset_index(drop=True)
        summary_df['Open Time'] = summary_df['Open Time'].dt.strftime('%Y-%m-%d %H:%M')
        summary_df['Close Time'] = summary_df['Close Time'].dt.strftime('%Y-%m-%d %H:%M')
        print(summary_df.to_string(index=False))

    print("-------------------------------------------------")
    print(f"Total Realized P&L (from Ledger): ${total_pnl_ledger:,.2f}")
    print("-------------------------------------------------")

    # --- 2. Connect to IBKR for Live Position and P&L Data ---
    ib = IB()
    try:
        client_id = random.randint(1, 1000)
        print(f"\nConnecting to IBKR with Client ID {client_id} for live data...")
        await ib.connectAsync('127.0.0.1', 7497, clientId=client_id)
        
        # Fetch Account P&L
        account_summary = await ib.reqAccountSummaryAsync()
        realized_pnl_api = 0
        unrealized_pnl_api = 0
        for acc_val in account_summary:
            if acc_val.tag == 'RealizedPnL' and acc_val.currency == 'USD':
                realized_pnl_api = float(acc_val.value)
            if acc_val.tag == 'UnrealizedPnL' and acc_val.currency == 'USD':
                unrealized_pnl_api = float(acc_val.value)

        # Fetch Live Positions
        live_positions = await ib.reqPositionsAsync()
        live_open_positions = {
            pos.contract.localSymbol: pos.position
            for pos in live_positions if isinstance(pos.contract, FuturesOption) and pos.contract.symbol == 'KC' and pos.position != 0
        }
        
        print("\n--- Live Open Positions (Source of Truth: IBKR API) ---")
        if not live_open_positions:
            print("No open coffee option positions found.")
        else:
            for symbol, qty in live_open_positions.items():
                print(f"  - {symbol}: {abs(qty)} contract(s)")
        print("-------------------------------------------------------")

        print("\n--- Live P&L (Source of Truth: IBKR API) ---")
        print(f"Total Realized P&L (Daily Account): ${realized_pnl_api:,.2f}")
        print(f"Total Unrealized P&L (Daily Account): ${unrealized_pnl_api:,.2f}")
        print("---------------------------------------------")

        # --- 3. Reconciliation and Discrepancy Report ---
        ledger_symbols = set(ledger_open_positions.keys())
        live_symbols = set(live_open_positions.keys())

        if ledger_symbols != live_symbols:
             print("\n!!! WARNING: Discrepancy found between ledger and live positions. !!!")
             print("This confirms your trade_ledger.csv is missing some historical transactions.")

             in_ledger_not_live = ledger_symbols - live_symbols
             if in_ledger_not_live:
                 print("\nPositions considered OPEN by ledger but are CLOSED in IBKR:")
                 # Sort for consistent output
                 for symbol in sorted(list(in_ledger_not_live)):
                     # Show the quantity to make it easier to fix
                     qty = ledger_open_positions[symbol]
                     print(f"  - {symbol}: {qty} contract(s) (Missing closing 'BUY' transaction(s) in ledger)")

             live_not_in_ledger = live_symbols - ledger_symbols
             if live_not_in_ledger:
                 print("\nPositions OPEN in IBKR but not tracked in ledger:")
                 for symbol in sorted(list(live_not_in_ledger)):
                     print(f"  - {symbol} (Missing opening 'SELL' transaction in ledger)")


    except Exception as e:
        print(f"\nCould not connect to IBKR for live data: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    try:
        util.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nScript stopped manually.")

