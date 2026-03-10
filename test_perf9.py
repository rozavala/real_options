import pandas as pd
import timeit
import numpy as np

# Simulate a large dataframe for ib_trades
data_ib = {
    'local_symbol': ['contract_' + str(i % 10) for i in range(1000)],
    'action': ['BUY', 'SELL'] * 500,
    'quantity': [1] * 1000,
    'timestamp': pd.date_range(start='1/1/2021', periods=1000, freq='s'),
    'dummy': range(1000)
}
ib_df = pd.DataFrame(data_ib)

# local
data_local = {
    'local_symbol': ['contract_' + str(i % 10) for i in range(1000)],
    'action': ['BUY', 'SELL'] * 500,
    'quantity': [1] * 1000,
    'timestamp': pd.date_range(start='1/1/2021', periods=1000, freq='s')
}
local_df = pd.DataFrame(data_local)

def method_fast():
    # Keep local trades in a list of dicts for fast matching and removal
    from collections import defaultdict

    local_trades = local_df.to_dict('records')
    for i, t in enumerate(local_trades):
        t['_idx'] = local_df.index[i]
        t['_matched'] = False

    local_lookup = defaultdict(list)
    for t in local_trades:
        key = (t['local_symbol'], t['action'], t['quantity'])
        local_lookup[key].append(t)

    missing_from_local = []
    ib_dicts = ib_df.to_dict('records')

    for ib_trade in ib_dicts:
        is_matched = False
        key = (ib_trade['local_symbol'], ib_trade['action'], ib_trade['quantity'])

        if key in local_lookup:
            # Find the first unmatched trade within 30s
            potential_matches = [t for t in local_lookup[key] if not t['_matched']]

            for t in potential_matches:
                time_diff = abs((t['timestamp'] - ib_trade['timestamp']).total_seconds())
                if time_diff <= 30:
                    t['_matched'] = True
                    is_matched = True
                    break

        if not is_matched:
            # Append the original series object to retain data structure
            missing_from_local.append(ib_df.loc[ib_df.index[ib_dicts.index(ib_trade)]])

    unmatched_local_indices = [t['_idx'] for t in local_trades if not t['_matched']]
    local_trades_unmatched = local_df.loc[unmatched_local_indices]

    return missing_from_local

def method_fast2():
    # Keep local trades in a list of dicts for fast matching and removal
    from collections import defaultdict

    local_trades = local_df.to_dict('records')
    for i, t in enumerate(local_trades):
        t['_idx'] = local_df.index[i]
        t['_matched'] = False

    local_lookup = defaultdict(list)
    for t in local_trades:
        key = (t['local_symbol'], t['action'], t['quantity'])
        local_lookup[key].append(t)

    missing_from_local_indices = []
    ib_dicts = ib_df.to_dict('records')

    for i, ib_trade in enumerate(ib_dicts):
        is_matched = False
        key = (ib_trade['local_symbol'], ib_trade['action'], ib_trade['quantity'])

        if key in local_lookup:
            # Find the first unmatched trade within 30s
            potential_matches = [t for t in local_lookup[key] if not t['_matched']]

            for t in potential_matches:
                time_diff = abs((t['timestamp'] - ib_trade['timestamp']).total_seconds())
                if time_diff <= 30:
                    t['_matched'] = True
                    is_matched = True
                    break

        if not is_matched:
            missing_from_local_indices.append(i)

    unmatched_local_indices = [t['_idx'] for t in local_trades if not t['_matched']]
    local_trades_unmatched = local_df.loc[unmatched_local_indices]

    missing_from_local = [ib_df.iloc[idx] for idx in missing_from_local_indices]

    return missing_from_local

print("fast:", timeit.timeit(method_fast, number=10))
print("fast2:", timeit.timeit(method_fast2, number=10))
