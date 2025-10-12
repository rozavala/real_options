import os
os.system('clear')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

# Read the CSV file and attempt to infer timestamp format
df = pd.read_csv('trade_ledger.csv')

# Sum SELL and BUY in total_value_usd
sell_total = df[df['action'] == 'SELL']['total_value_usd'].sum()
buy_total = df[df['action'] == 'BUY']['total_value_usd'].sum()
net_profit = sell_total - buy_total
print(f"Buy total: ${buy_total:,.2f}")
print(f"Sell total: ${sell_total:,.2f}")
print(f"Net Profit: ${net_profit:,.2f}")

# Set SELL as positive cashflows and BUY as negative cashflows
df['net_value'] = df.apply(
    lambda row: row['total_value_usd'] if row['action'] == 'SELL' else -row['total_value_usd'],
    axis=1
)

# Group or aggregate net_value by timestamp
df['net_value_grouped'] = df.groupby('timestamp')['net_value'].transform('sum')

# Calculates the cumulative net value
df['cumulative_net_value'] = df['net_value'].cumsum()

# Create a single figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# Plot the cashflows on the top subplot
ax1.bar(df['timestamp'], df['net_value'], color=['green' if x > 0 else 'red' for x in df['net_value']], label='Cashflows')
ax1.set_title('Cashflows (SELL positive, BUY negative)')
ax1.set_ylabel('Cashflows (USD)')
ax1.legend()
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
ax1.tick_params(axis='x', rotation=45)
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Plot the cumulative net value on the bottom subplot
ax2.plot(df['timestamp'], df['cumulative_net_value'], color='#1E90FF', label='Cumulative Net Value')
ax2.set_title('Cumulative cashflows')
ax2.set_xlabel('Date & time')
ax2.set_ylabel('Cumulative cashflows (USD)')
ax2.legend()
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.tick_params(axis='x', rotation=45)
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()