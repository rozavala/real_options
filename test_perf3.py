import pandas as pd
import timeit
import numpy as np

# Simulate a large dataframe
data = {
    'Symbol': ['contract_' + str(i) for i in range(100)],
    'Quantity_ib': np.random.randn(100),
    'Quantity_local': np.random.randn(100),
}
df = pd.DataFrame(data)

def method_iterrows():
    message_lines = ["Active Position Discrepancies Found:"]
    for _, row in df.iterrows():
        sym = row['Symbol']
        ib_qty = row['Quantity_ib']
        loc_qty = row['Quantity_local']
        message_lines.append(f"- {sym}: IB={ib_qty}, Local={loc_qty}")
    return "\n".join(message_lines)

def method_vectorized():
    message_lines = ["Active Position Discrepancies Found:"]
    # zip is fast
    for sym, ib_qty, loc_qty in zip(df['Symbol'], df['Quantity_ib'], df['Quantity_local']):
        message_lines.append(f"- {sym}: IB={ib_qty}, Local={loc_qty}")
    return "\n".join(message_lines)

print("iterrows:", timeit.timeit(method_iterrows, number=100))
print("vectorized:", timeit.timeit(method_vectorized, number=100))

assert method_iterrows() == method_vectorized()
