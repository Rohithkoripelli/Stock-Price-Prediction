"""
Simple macro and sector data collection
"""

import yfinance as yf
import pandas as pd
import os

print("Collecting macro and sector data...")

START_DATE = '2019-01-01'
END_DATE = '2024-12-10'

os.makedirs('data/enhanced/macro', exist_ok=True)
os.makedirs('data/enhanced/market', exist_ok=True)

# Collect Nifty
print("\n1. Collecting Nifty...")
nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False)
nifty_df = pd.DataFrame({
    'date': nifty.index,
    'nifty_close': nifty['Close'].iloc[:, 0] if len(nifty['Close'].shape) > 1 else nifty['Close'],
})
nifty_df['nifty_return'] = nifty_df['nifty_close'].pct_change() * 100

# Collect Bank Nifty
print("2. Collecting Bank Nifty...")
banknifty = yf.download('^NSEBANK', start=START_DATE, end=END_DATE, progress=False)
banknifty_df = pd.DataFrame({
    'date': banknifty.index,
    'banknifty_close': banknifty['Close'].iloc[:, 0] if len(banknifty['Close'].shape) > 1 else banknifty['Close'],
    'banknifty_volume': banknifty['Volume'].iloc[:, 0] if len(banknifty['Volume'].shape) > 1 else banknifty['Volume'],
})
banknifty_df['banknifty_return'] = banknifty_df['banknifty_close'].pct_change() * 100

# Collect USD/INR
print("3. Collecting USD/INR...")
usdinr = yf.download('INR=X', start=START_DATE, end=END_DATE, progress=False)
usdinr_df = pd.DataFrame({
    'date': usdinr.index,
    'usdinr': usdinr['Close'].iloc[:, 0] if len(usdinr['Close'].shape) > 1 else usdinr['Close'],
})
usdinr_df['usdinr_change'] = usdinr_df['usdinr'].pct_change() * 100

# Merge all macro data
macro_data = nifty_df.merge(banknifty_df, on='date', how='inner')
macro_data = macro_data.merge(usdinr_df, on='date', how='inner')

macro_data.to_csv('data/enhanced/macro/macro_indicators.csv', index=False)
print(f"✓ Saved macro data: {len(macro_data)} days")

# Create sector data
sector_data = pd.DataFrame({
    'date': banknifty_df['date'],
    'sector_close': banknifty_df['banknifty_close'],
    'sector_volume': banknifty_df['banknifty_volume'],
    'sector_return': banknifty_df['banknifty_return'],
})
sector_data['sector_volatility'] = sector_data['sector_return'].rolling(20).std()

sector_data.to_csv('data/enhanced/market/banking_sector.csv', index=False)
print(f"✓ Saved sector data: {len(sector_data)} days")

print("\n✓ Macro and sector data collection complete!")
