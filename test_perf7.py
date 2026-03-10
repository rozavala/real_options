import pandas as pd
import timeit
import numpy as np

# Simulate a large dataframe for ib_trades
data_ib = {
    'local_symbol': ['contract_' + str(i) for i in range(100)],
    'action': ['BUY', 'SELL'] * 50,
    'quantity': [1] * 100,
    'timestamp': pd.date_range(start='1/1/2021', periods=100, freq='s'),
    'dummy': range(100)
}
ib_df = pd.DataFrame(data_ib)

# local
data_local = {
    'local_symbol': ['contract_' + str(i) for i in range(100)],
    'action': ['BUY', 'SELL'] * 50,
    'quantity': [1] * 100,
    'timestamp': pd.date_range(start='1/1/2021', periods=100, freq='s')
}
local_df = pd.DataFrame(data_local)

def method_iterrows():
    local_trades_unmatched = local_df.copy()
    missing_from_local = []

    for ib_index, ib_trade in ib_df.iterrows():
        is_matched = False
        potential_matches = local_trades_unmatched[
            (local_trades_unmatched['local_symbol'] == ib_trade['local_symbol']) &
            (local_trades_unmatched['action'] == ib_trade['action']) &
            (local_trades_unmatched['quantity'] == ib_trade['quantity'])
        ]

        if not potential_matches.empty:
            time_diff = (potential_matches['timestamp'] - ib_trade['timestamp']).abs()
            match_indices = potential_matches[time_diff <= pd.Timedelta(seconds=30)].index

            if len(match_indices) > 0:
                local_trades_unmatched.drop(match_indices[0], inplace=True)
                is_matched = True

        if not is_matched:
            missing_from_local.append(ib_trade)

    return missing_from_local

def method_zip():
    local_trades_unmatched = local_df.copy()
    missing_from_local_indices = []

    # Fast column extraction
    ib_symbols = ib_df['local_symbol']
    ib_actions = ib_df['action']
    ib_quantities = ib_df['quantity']
    ib_timestamps = ib_df['timestamp']

    # Using zip is much faster than iterrows
    for idx, (sym, action, qty, ts) in enumerate(zip(ib_symbols, ib_actions, ib_quantities, ib_timestamps)):
        is_matched = False

        # Filtering local_trades_unmatched
        potential_matches = local_trades_unmatched[
            (local_trades_unmatched['local_symbol'] == sym) &
            (local_trades_unmatched['action'] == action) &
            (local_trades_unmatched['quantity'] == qty)
        ]

        if not potential_matches.empty:
            time_diff = (potential_matches['timestamp'] - ts).abs()
            match_indices = potential_matches[time_diff <= pd.Timedelta(seconds=30)].index

            if len(match_indices) > 0:
                local_trades_unmatched.drop(match_indices[0], inplace=True)
                is_matched = True

        if not is_matched:
            missing_from_local_indices.append(idx)

    missing_from_local = [ib_df.iloc[idx] for idx in missing_from_local_indices]
    return missing_from_local

print("iterrows:", timeit.timeit(method_iterrows, number=10))
print("zip:", timeit.timeit(method_zip, number=10))
