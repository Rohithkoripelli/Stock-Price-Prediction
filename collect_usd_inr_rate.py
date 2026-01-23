"""
Collect USD/INR Exchange Rate Data
Weakening INR (increasing USD/INR) often indicates:
- Bearish market sentiment
- FII selling pressure
- Market downturn risk
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

print("=" * 80)
print("COLLECTING USD/INR EXCHANGE RATE DATA".center(80))
print("=" * 80)

# USD/INR symbol on Yahoo Finance
USD_INR = "USDINR=X"
START_DATE = "2019-01-01"
END_DATE = pd.Timestamp.now().strftime('%Y-%m-%d')

os.makedirs('data/forex', exist_ok=True)

print(f"\n📊 Downloading USD/INR exchange rate data...")
print(f"   Symbol: {USD_INR}")
print(f"   Period: {START_DATE} to {END_DATE}")

try:
    # Download USD/INR data
    usd_inr_data = yf.download(USD_INR, start=START_DATE, end=END_DATE, progress=False)

    if len(usd_inr_data) == 0:
        print("   ⚠ No data found, trying alternative symbol...")
        # Try alternative symbol
        usd_inr_data = yf.download("INR=X", start=START_DATE, end=END_DATE, progress=False)

    if len(usd_inr_data) > 0:
        # Calculate key features
        forex_features = pd.DataFrame(index=usd_inr_data.index)

        # Close price (exchange rate)
        forex_features['usd_inr_rate'] = usd_inr_data['Close']

        # Daily change in INR (positive = INR weakening = bearish)
        forex_features['usd_inr_change_1d'] = usd_inr_data['Close'].pct_change(1)

        # 5-day change (short-term trend)
        forex_features['usd_inr_change_5d'] = usd_inr_data['Close'].pct_change(5)

        # 20-day change (medium-term trend)
        forex_features['usd_inr_change_20d'] = usd_inr_data['Close'].pct_change(20)

        # Momentum indicator (rate of change)
        forex_features['usd_inr_momentum'] = forex_features['usd_inr_change_1d'].rolling(window=5).mean()

        # Volatility (higher volatility = uncertainty)
        forex_features['usd_inr_volatility'] = forex_features['usd_inr_change_1d'].rolling(window=20).std()

        # INR weakness score (normalized)
        # Positive = INR weakening = bearish signal
        forex_features['inr_weakness_score'] = (
            forex_features['usd_inr_change_1d'] * 0.4 +  # Recent change (40% weight)
            forex_features['usd_inr_change_5d'] * 0.3 +   # Short-term trend (30% weight)
            forex_features['usd_inr_momentum'] * 0.3      # Momentum (30% weight)
        )

        # Save the data
        output_file = 'data/forex/USD_INR_rates.csv'
        forex_features.to_csv(output_file)

        print(f"\n   ✓ Total Records: {len(forex_features)}")
        print(f"   ✓ Date Range: {forex_features.index[0].date()} to {forex_features.index[-1].date()}")
        print(f"   ✓ Current USD/INR Rate: ₹{forex_features['usd_inr_rate'].iloc[-1]:.4f}")
        print(f"   ✓ Features: {len(forex_features.columns)}")
        print(f"   ✓ Saved to: {output_file}")

        # Calculate recent trends
        if len(forex_features) >= 21:
            current_rate = forex_features['usd_inr_rate'].iloc[-1]
            change_1d = forex_features['usd_inr_change_1d'].iloc[-1] * 100
            change_5d = forex_features['usd_inr_change_5d'].iloc[-1] * 100
            change_20d = forex_features['usd_inr_change_20d'].iloc[-1] * 100
            weakness_score = forex_features['inr_weakness_score'].iloc[-1] * 100

            print(f"\n   📈 Recent INR Performance:")
            print(f"      Current Rate: ₹{current_rate:.4f}")
            print(f"      1-day change: {change_1d:+.2f}% {'(INR weakening ⚠️)' if change_1d > 0 else '(INR strengthening ✓)'}")
            print(f"      5-day change: {change_5d:+.2f}% {'(INR weakening ⚠️)' if change_5d > 0 else '(INR strengthening ✓)'}")
            print(f"      20-day change: {change_20d:+.2f}% {'(INR weakening ⚠️)' if change_20d > 0 else '(INR strengthening ✓)'}")
            print(f"      Weakness Score: {weakness_score:+.3f}% {'(Bearish for market ⚠️)' if weakness_score > 0 else '(Bullish for market ✓)'}")

            # Market impact assessment
            print(f"\n   💡 Market Impact Assessment:")
            if weakness_score > 0.05:
                print(f"      🔴 STRONG BEARISH: Significant INR weakness may trigger FII selling")
            elif weakness_score > 0.02:
                print(f"      🟠 MODERATELY BEARISH: INR weakness may pressure markets")
            elif weakness_score > -0.02:
                print(f"      🟡 NEUTRAL: Stable INR, minimal FII impact expected")
            elif weakness_score > -0.05:
                print(f"      🟢 MODERATELY BULLISH: INR strength may attract FII buying")
            else:
                print(f"      🟢 STRONG BULLISH: Strong INR may boost FII inflows")

        print(f"\n   ✓ Feature Breakdown:")
        print(f"      - USD/INR Rate: 1 feature")
        print(f"      - Daily/Weekly/Monthly Changes: 3 features")
        print(f"      - Momentum: 1 feature")
        print(f"      - Volatility: 1 feature")
        print(f"      - INR Weakness Score: 1 feature")
        print(f"      Total: 7 forex features")

    else:
        print("   ✗ Failed to download USD/INR data")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("✓ USD/INR DATA COLLECTION COMPLETE".center(80))
print("=" * 80)
