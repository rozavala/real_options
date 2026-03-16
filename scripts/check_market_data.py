#!/usr/bin/env python3
"""
Quick diagnostic: connect to IB, request quotes for active commodities,
and report whether data is LIVE or DELAYED.

Usage:
    python scripts/check_market_data.py
"""

import asyncio
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ib_insync import IB, Future
from config_loader import load_config


# Contracts to check — front-month futures for each active commodity
CONTRACTS = [
    Future(symbol="KC", exchange="NYBOT", currency="USD"),
    Future(symbol="NG", exchange="NYMEX", currency="USD"),
]


async def main():
    config = load_config()
    host = config["connection"]["host"]
    port = config["connection"]["port"]

    force_delayed = os.getenv("FORCE_DELAYED_DATA", "0")
    data_type = 3 if force_delayed == "1" else 1

    print(f"IB Gateway:         {host}:{port}")
    print(f"FORCE_DELAYED_DATA: {force_delayed}")
    print(f"Requesting:         Type {data_type} ({'DELAYED' if data_type == 3 else 'LIVE'})")
    print()

    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=999, timeout=10)
    except Exception as e:
        print(f"CONNECTION FAILED: {e}")
        sys.exit(1)

    ib.reqMarketDataType(data_type)
    print(f"Connected  (server v{ib.client.serverVersion()})")
    print("-" * 60)

    for contract in CONTRACTS:
        # Qualify to get the front-month expiry
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            print(f"{contract.symbol:4s}  Could not qualify contract")
            continue

        c = qualified[0]
        ticker = ib.reqMktData(c, genericTickList="", snapshot=False)

        # Wait up to 5 seconds for data to arrive
        for _ in range(50):
            await asyncio.sleep(0.1)
            if ticker.last is not None or ticker.bid is not None:
                break

        # Determine data type from marketDataType field
        # 1 = LIVE, 2 = FROZEN, 3 = DELAYED, 4 = DELAYED_FROZEN
        md_type = getattr(ticker, "marketDataType", None)
        type_labels = {1: "LIVE", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED_FROZEN"}
        type_str = type_labels.get(md_type, f"UNKNOWN({md_type})")

        last = ticker.last if ticker.last is not None else "-"
        bid = ticker.bid if ticker.bid is not None else "-"
        ask = ticker.ask if ticker.ask is not None else "-"
        volume = ticker.volume if ticker.volume is not None else "-"

        print(f"{c.symbol:4s}  {c.localSymbol:12s}  "
              f"last={last!s:>8s}  bid={bid!s:>8s}  ask={ask!s:>8s}  "
              f"vol={volume!s:>10s}  dataType={type_str}")

        ib.cancelMktData(c)

    print("-" * 60)
    ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
