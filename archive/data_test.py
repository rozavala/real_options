# Filename: data_test.py

from ib_insync import *
import pandas as pd

# This script is designed to test basic data subscriptions.

def run_data_test():
    """
    Connects to TWS and attempts to download data for two common instruments
    to diagnose subscription and API connectivity issues.
    """
    ib = IB()
    try:
        print("Connecting to TWS...")
        ib.connect('127.0.0.1', 7497, clientId=30) # Using a different clientId
        print("Connection successful.")

        # --- Test Case 1: E-mini S&P 500 Future (ES) on CME ---
        print("\n----- [TEST 1] Attempting to fetch E-mini S&P 500 (ES) futures data... -----")
        try:
            # Note: We need to find the front-month contract first.
            es_generic = Future(symbol='ES', exchange='CME')
            es_chains = ib.reqContractDetails(es_generic)
            
            if not es_chains:
                raise ConnectionError("Could not retrieve futures chain for ES.")

            es_contracts = sorted([c.contract for c in es_chains], key=lambda x: x.lastTradeDateOrContractMonth)
            es_front_month = es_contracts[0]
            
            print(f"  Found front-month contract: {es_front_month.localSymbol}")
            print(f"  Requesting 5 days of historical data for {es_front_month.localSymbol}...")

            bars = ib.reqHistoricalData(
                es_front_month, endDateTime='', durationStr='5 D',
                barSizeSetting='1 day', whatToShow='TRADES',
                useRTH=True
            )

            if bars:
                print(f"  ✅ SUCCESS: Successfully downloaded {len(bars)} bars of data for ES future.")
            else:
                print("  ❌ FAILURE: Request was successful, but no data was returned.")
        
        except Exception as e:
            print(f"  ❌ CRITICAL FAILURE for ES Future: {e}")

        # --- Test Case 2: SPY ETF (Stock) on ARCA ---
        print("\n----- [TEST 2] Attempting to fetch SPY ETF data... -----")
        try:
            spy_contract = Stock('SPY', 'ARCA', 'USD')
            print("  Requesting 5 days of historical data for SPY...")
            
            bars = ib.reqHistoricalData(
                spy_contract, endDateTime='', durationStr='5 D',
                barSizeSetting='1 day', whatToShow='TRADES',
                useRTH=True
            )

            if bars:
                print(f"  ✅ SUCCESS: Successfully downloaded {len(bars)} bars of data for SPY ETF.")
            else:
                print("  ❌ FAILURE: Request was successful, but no data was returned.")

        except Exception as e:
            print(f"  ❌ CRITICAL FAILURE for SPY ETF: {e}")

    except Exception as e:
        print(f"A critical connection error occurred: {e}")
    finally:
        print("\nData test complete. Disconnecting from TWS.")
        ib.disconnect()

if __name__ == "__main__":
    run_data_test()
