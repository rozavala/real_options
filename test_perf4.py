import pandas as pd
import timeit
import numpy as np

# Simulate a large dataframe
data = {
    'local_symbol': ['contract_' + str(i) for i in range(100)],
    'position_id': ['pos_' + str(i) for i in range(100)],
}
df = pd.DataFrame(data)

def method_iterrows():
    result = df.copy()
    pid_lookup = {
        'contract_5': 'new_pos_5',
        'contract_10': 'new_pos_10',
    }

    for idx, row in result.iterrows():
        symbol = row.get('local_symbol', '')
        if not symbol:
            continue

        original_pid = pid_lookup.get(symbol)
        if original_pid and str(original_pid).strip():
            old_pid = row.get('position_id', '')
            result.at[idx, 'position_id'] = original_pid
    return result

def method_vectorized():
    result = df.copy()
    pid_lookup = {
        'contract_5': 'new_pos_5',
        'contract_10': 'new_pos_10',
    }

    # Fast map lookup
    mapped = result['local_symbol'].map(pid_lookup)
    mask = mapped.notna() & mapped.astype(str).str.strip().astype(bool)
    result.loc[mask, 'position_id'] = mapped[mask]
    return result

print("iterrows:", timeit.timeit(method_iterrows, number=100))
print("vectorized:", timeit.timeit(method_vectorized, number=100))

# verify outputs are same
pd.testing.assert_frame_equal(method_iterrows(), method_vectorized())
