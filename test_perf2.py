import pandas as pd
import timeit
import numpy as np

# Simulate a large dataframe from 'ds_df' or 'signals_df'
data = {
    'contract': ['contract_' + str(i) for i in range(1000)],
    'signal': ['BULLISH', 'BEARISH', 'NEUTRAL'] * 333 + ['NEUTRAL'],
    'timestamp': pd.date_range(start='1/1/2021', periods=1000, freq='D')
}
df = pd.DataFrame(data)

def method_iterrows():
    cnt = 0
    for _, sig_row in df.iterrows():
        cnt += 1
    return cnt

def method_itertuples():
    cnt = 0
    for sig_row in df.itertuples():
        cnt += 1
    return cnt

print("iterrows:", timeit.timeit(method_iterrows, number=100))
print("itertuples:", timeit.timeit(method_itertuples, number=100))
