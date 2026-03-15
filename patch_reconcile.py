import pandas as pd

def patch_file():
    with open('reconcile_trades.py', 'r') as f:
        content = f.read()

    search = """    local_sorted = local_trades_unmatched.sort_values('timestamp')
    local_grouped = local_sorted.groupby(['local_symbol', 'action', 'quantity'])
    local_indices_to_drop = set()

    missing_from_local = []

    ib_sorted = ib_trades_df.sort_values('timestamp')
    for keys, ib_group in ib_sorted.groupby(['local_symbol', 'action', 'quantity']):
        if keys not in local_grouped.groups:
            # No matching local trades at all for this group
            missing_from_local.extend(ib_group.to_dict('records'))
            continue

        local_group = local_grouped.get_group(keys).copy()

        # For each IB trade in this group, find the closest local trade within 30s
        for _, ib_trade in ib_group.iterrows():
            if local_group.empty:
                missing_from_local.append(ib_trade.to_dict())
                continue

            time_diff = (local_group['timestamp'] - ib_trade['timestamp']).abs()
            valid_matches = time_diff[time_diff <= pd.Timedelta(seconds=30)]

            if not valid_matches.empty:
                # Find index of minimum time diff
                best_match_idx = valid_matches.idxmin()
                local_indices_to_drop.add(best_match_idx)
                # Remove from local_group so it can't be matched again
                local_group.drop(best_match_idx, inplace=True)
            else:
                missing_from_local.append(ib_trade.to_dict())

    local_trades_unmatched.drop(list(local_indices_to_drop), inplace=True)"""

    replace = """    local_sorted = local_trades_unmatched.sort_values('timestamp')
    local_grouped = local_sorted.groupby(['local_symbol', 'action', 'quantity'], dropna=False)
    local_indices_to_drop = set()

    missing_from_local = []

    ib_sorted = ib_trades_df.sort_values('timestamp')
    for keys, ib_group in ib_sorted.groupby(['local_symbol', 'action', 'quantity'], dropna=False):
        if keys not in local_grouped.groups:
            # No matching local trades at all for this group
            # Append original Series objects to preserve indices
            for _, ib_trade in ib_group.iterrows():
                missing_from_local.append(ib_trade)
            continue

        local_group = local_grouped.get_group(keys).copy()

        # For each IB trade in this group, find the closest local trade within 30s
        for _, ib_trade in ib_group.iterrows():
            if local_group.empty:
                missing_from_local.append(ib_trade)
                continue

            time_diff = (local_group['timestamp'] - ib_trade['timestamp']).abs()
            valid_matches = time_diff[time_diff <= pd.Timedelta(seconds=30)]

            if not valid_matches.empty:
                # Find index of minimum time diff
                best_match_idx = valid_matches.idxmin()
                local_indices_to_drop.add(best_match_idx)
                # Remove from local_group so it can't be matched again
                local_group.drop(best_match_idx, inplace=True)
            else:
                missing_from_local.append(ib_trade)

    local_trades_unmatched.drop(list(local_indices_to_drop), inplace=True)

    # Restore original chronological ordering for missing_from_local
    if missing_from_local:
        missing_from_local.sort(key=lambda x: x.name)"""

    content = content.replace(search, replace)
    with open('reconcile_trades.py', 'w') as f:
        f.write(content)

patch_file()
