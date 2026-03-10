import time
import pandas as pd
import random

data = {
    'contract': ['contract_' + str(i) for i in range(1000)],
    'signal': [random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']) for _ in range(1000)]
}
df = pd.DataFrame(data)

def method_iterrows(today_signals):
    report = f"{'Contract':<12} {'Signal':<10}\n"
    report += "-" * 22 + "\n"
    for _, row in today_signals.iterrows():
        contract_col = 'contract' if 'contract' in row.index else 'symbol'
        signal_col = 'signal' if 'signal' in row.index else 'master_decision'
        report += f"{row.get(contract_col, 'N/A'):<12} {row.get(signal_col, 'N/A'):<10}\n"
    return report

def method_vectorized(today_signals):
    report = f"{'Contract':<12} {'Signal':<10}\n"
    report += "-" * 22 + "\n"
    contract_col = 'contract' if 'contract' in today_signals.columns else 'symbol'
    signal_col = 'signal' if 'signal' in today_signals.columns else 'master_decision'

    contracts = today_signals.get(contract_col, pd.Series(['N/A'] * len(today_signals)))
    signals = today_signals.get(signal_col, pd.Series(['N/A'] * len(today_signals)))

    for c, s in zip(contracts, signals):
        report += f"{c:<12} {s:<10}\n"
    return report

import timeit
print("iterrows:", timeit.timeit(lambda: method_iterrows(df), number=10))
print("vectorized:", timeit.timeit(lambda: method_vectorized(df), number=10))

assert method_iterrows(df) == method_vectorized(df)
