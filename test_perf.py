import time
import pandas as pd

# Creating dummy DataFrame
data = {
    'contract': ['contract_' + str(i) for i in range(1000)],
    'signal': ['BULLISH', 'BEARISH', 'NEUTRAL'] * 333 + ['NEUTRAL']
}
df = pd.DataFrame(data)

def method_iterrows():
    report = f"{'Contract':<12} {'Signal':<10}\n"
    report += "-" * 22 + "\n"
    for _, row in df.iterrows():
        contract_col = 'contract' if 'contract' in row.index else 'symbol'
        signal_col = 'signal' if 'signal' in row.index else 'master_decision'
        report += f"{row.get(contract_col, 'N/A'):<12} {row.get(signal_col, 'N/A'):<10}\n"
    return report

def method_vectorized():
    report = f"{'Contract':<12} {'Signal':<10}\n"
    report += "-" * 22 + "\n"
    contract_col = 'contract' if 'contract' in df.columns else 'symbol'
    signal_col = 'signal' if 'signal' in df.columns else 'master_decision'

    # We can just iterate over zip or use map? Let's try zip which is fast.
    for c, s in zip(df.get(contract_col, ['N/A']*len(df)), df.get(signal_col, ['N/A']*len(df))):
        report += f"{c:<12} {s:<10}\n"
    return report

start = time.time()
report1 = method_iterrows()
time1 = time.time() - start

start = time.time()
report2 = method_vectorized()
time2 = time.time() - start

print(f"iterrows: {time1:.4f}s")
print(f"vectorized: {time2:.4f}s")
assert report1 == report2
