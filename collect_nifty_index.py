"""
Collect Nifty Bank Index and BSE Bankex data to use as market indicators
"""

import yfinance as yf
import pandas as pd
import os

print("=" * 80)
print("COLLECTING BANK INDICES DATA".center(80))
print("=" * 80)

# Bank Index symbols
NIFTY_BANK = "^NSEBANK"
BSE_BANKEX = "BANKEX.BO"  # BSE Bankex symbol (alternative format)
START_DATE = "2019-01-01"
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

os.makedirs('data/market_index', exist_ok=True)

print(f"\n📊 Downloading Nifty Bank Index data...")
print(f"   Symbol: {NIFTY_BANK}")
print(f"   Period: {START_DATE} to {END_DATE}")

try:
    nifty_data = yf.download(NIFTY_BANK, start=START_DATE, end=END_DATE, progress=False)

    if len(nifty_data) == 0:
        print("   ⚠ No data found, trying alternative symbol...")
        # Try alternative
        nifty_data = yf.download("NIFTY_BANK.NS", start=START_DATE, end=END_DATE, progress=False)

    if len(nifty_data) > 0:
        # Save the data
        output_file = 'data/market_index/NIFTY_BANK_index.csv'
        nifty_data.to_csv(output_file)

        print(f"\n   ✓ Total Records: {len(nifty_data)}")
        print(f"   ✓ Date Range: {nifty_data.index[0].date()} to {nifty_data.index[-1].date()}")
        print(f"   ✓ Current Level: {nifty_data['Close'].iloc[-1]:.2f}")
        print(f"   ✓ Range: {nifty_data['Close'].min():.2f} - {nifty_data['Close'].max():.2f}")
        print(f"   ✓ Saved to: {output_file}")

        # Calculate recent returns
        returns_1d = ((nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-2]) - 1) * 100
        returns_5d = ((nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-6]) - 1) * 100
        returns_20d = ((nifty_data['Close'].iloc[-1] / nifty_data['Close'].iloc[-21]) - 1) * 100

        print(f"\n   📈 Recent Performance:")
        print(f"      1-day: {returns_1d:+.2f}%")
        print(f"      5-day: {returns_5d:+.2f}%")
        print(f"      20-day: {returns_20d:+.2f}%")

    else:
        print("   ✗ Failed to download data")

except Exception as e:
    print(f"   ✗ Error: {e}")

# Collect BSE Bankex
print(f"\n📊 Downloading BSE Bankex Index data...")
print(f"   Symbol: {BSE_BANKEX}")
print(f"   Period: {START_DATE} to {END_DATE}")

try:
    bankex_data = yf.download(BSE_BANKEX, start=START_DATE, end=END_DATE, progress=False)

    if len(bankex_data) > 0:
        # Save the data
        output_file = 'data/market_index/BSE_BANKEX_index.csv'
        bankex_data.to_csv(output_file)

        print(f"\n   ✓ Total Records: {len(bankex_data)}")
        print(f"   ✓ Date Range: {bankex_data.index[0].date()} to {bankex_data.index[-1].date()}")
        print(f"   ✓ Current Level: {bankex_data['Close'].iloc[-1]:.2f}")
        print(f"   ✓ Range: {bankex_data['Close'].min():.2f} - {bankex_data['Close'].max():.2f}")
        print(f"   ✓ Saved to: {output_file}")

        # Calculate recent returns
        if len(bankex_data) >= 21:
            returns_1d = ((bankex_data['Close'].iloc[-1] / bankex_data['Close'].iloc[-2]) - 1) * 100
            returns_5d = ((bankex_data['Close'].iloc[-1] / bankex_data['Close'].iloc[-6]) - 1) * 100
            returns_20d = ((bankex_data['Close'].iloc[-1] / bankex_data['Close'].iloc[-21]) - 1) * 100

            print(f"\n   📈 Recent Performance:")
            print(f"      1-day: {returns_1d:+.2f}%")
            print(f"      5-day: {returns_5d:+.2f}%")
            print(f"      20-day: {returns_20d:+.2f}%")

    else:
        print("   ✗ Failed to download Bankex data")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ BANK INDICES COLLECTION COMPLETE".center(80))
print("=" * 80)
