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

def method_itertuples():
    local_trades_unmatched = local_df.copy()
    missing_from_local = []

    # We can pre-extract columns for faster iteration

    # Using iteruples is ~5-10x faster than iterrows.
    for ib_trade in ib_df.itertuples(index=False):
        is_matched = False
        potential_matches = local_trades_unmatched[
            (local_trades_unmatched['local_symbol'] == ib_trade.local_symbol) &
            (local_trades_unmatched['action'] == ib_trade.action) &
            (local_trades_unmatched['quantity'] == ib_trade.quantity)
        ]

        if not potential_matches.empty:
            time_diff = (potential_matches['timestamp'] - ib_trade.timestamp).abs()
            match_indices = potential_matches[time_diff <= pd.Timedelta(seconds=30)].index

            if len(match_indices) > 0:
                local_trades_unmatched.drop(match_indices[0], inplace=True)
                is_matched = True

        if not is_matched:
            missing_from_local.append(ib_trade._asdict()) # to match row format roughly

    return missing_from_local

print("iterrows:", timeit.timeit(method_iterrows, number=10))
print("itertuples:", timeit.timeit(method_itertuples, number=10))
