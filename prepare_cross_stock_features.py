"""
Add Cross-Stock Features (V8 Enhancement)

Uses other bank stocks' movements as features for each stock
Banks are correlated - if ICICI/AXIS/HDFC all go up, KOTAK likely follows
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

STOCKS = {
    'HDFCBANK': 'HDFC Bank',
    'ICICIBANK': 'ICICI Bank',
    'KOTAKBANK': 'Kotak Mahindra Bank',
    'AXISBANK': 'Axis Bank',
    'SBIN': 'State Bank of India',
    'PNB': 'Punjab National Bank',
    'BANKBARODA': 'Bank of Baroda',
    'CANBK': 'Canara Bank'
}

print("=" * 80)
print("CREATING CROSS-STOCK FEATURES".center(80))
print("=" * 80)

# =============================================================================
# LOAD ALL STOCKS' DATA
# =============================================================================

print(f"\n1. Loading data for all stocks...")

all_data = {}
for ticker in STOCKS.keys():
    file_path = f'data/enhanced_model_ready/{ticker}_enhanced.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            all_data[ticker] = pickle.load(f)
        print(f"   ✓ Loaded {ticker}")
    else:
        print(f"   ✗ Missing {ticker}")

if len(all_data) < 8:
    print(f"\n   WARNING: Only {len(all_data)}/8 stocks loaded")
    print(f"   Run prepare_enhanced_features.py first for missing stocks")

# =============================================================================
# EXTRACT DAILY RETURNS FOR ALL STOCKS
# =============================================================================

print(f"\n2. Extracting daily returns for cross-stock features...")

def extract_daily_returns(data, ticker):
    """Extract daily percentage changes for all splits"""
    returns = {}

    for split in ['train', 'val', 'test']:
        dates = pd.to_datetime(data[split]['dates'])
        y_pct = data[split]['y']  # Already percentage changes

        df = pd.DataFrame({
            'date': dates,
            f'{ticker}_return': y_pct
        })
        returns[split] = df.set_index('date')

    return returns

# Get returns for all stocks
stock_returns = {}
for ticker in all_data.keys():
    stock_returns[ticker] = extract_daily_returns(all_data[ticker], ticker)
    print(f"   ✓ Extracted returns for {ticker}")

# =============================================================================
# CREATE CROSS-STOCK FEATURES
# =============================================================================

print(f"\n3. Creating cross-stock features for each stock...")

def create_cross_features(target_ticker, target_data, all_returns, split):
    """
    Add features from other stocks:
    - Other stocks' same-day returns
    - Sector average return
    - Sector breadth (% of stocks up)
    - Relative strength vs sector
    """
    target_dates = pd.to_datetime(target_data[split]['dates'])
    X_original = target_data[split]['X']

    # Create DataFrame with target dates
    df = pd.DataFrame({'date': target_dates})
    df = df.set_index('date')

    # Add other stocks' returns
    other_tickers = [t for t in all_returns.keys() if t != target_ticker]

    for other_ticker in other_tickers:
        # Merge returns (forward fill for missing dates)
        other_df = all_returns[other_ticker][split]
        df = df.join(other_df, how='left')

    # Forward fill missing values (weekends/holidays)
    df = df.fillna(method='ffill').fillna(0)

    # Calculate sector features
    return_columns = [col for col in df.columns if col.endswith('_return')]

    # Sector average return
    df['sector_avg_return'] = df[return_columns].mean(axis=1)

    # Sector breadth (% up)
    df['sector_breadth'] = (df[return_columns] > 0).sum(axis=1) / len(return_columns)

    # Sector volatility (std of returns)
    df['sector_volatility'] = df[return_columns].std(axis=1)

    # Target stock's relative strength
    target_return = target_data[split]['y']
    df['relative_strength'] = target_return - df['sector_avg_return'].values

    # Correlation signal: how many stocks moving same direction as sector
    sector_direction = (df['sector_avg_return'] > 0).astype(int)
    same_direction = sum((df[col] > 0).astype(int) == sector_direction for col in return_columns)
    df['sector_consensus'] = same_direction / len(return_columns)

    # Convert to features (exclude date index)
    cross_features = df.values  # Shape: (n_samples, n_cross_features)

    # Expand to match timesteps
    # Repeat across timesteps (same cross-stock feature for all timesteps in sequence)
    n_samples, n_timesteps, n_features = X_original.shape
    cross_expanded = np.repeat(
        cross_features[:, np.newaxis, :],
        n_timesteps,
        axis=1
    )

    # Concatenate to original features
    X_enhanced = np.concatenate([X_original, cross_expanded], axis=2)

    return X_enhanced, cross_features.shape[1]

# Process each stock
os.makedirs('data/cross_stock_ready', exist_ok=True)

for ticker in all_data.keys():
    print(f"\n   Processing {ticker}...")

    enhanced_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    n_cross_features = None

    for split in ['train', 'val', 'test']:
        X_enhanced, n_cross = create_cross_features(
            ticker, all_data[ticker], stock_returns, split
        )

        enhanced_data[split]['X'] = X_enhanced
        enhanced_data[split]['y'] = all_data[ticker][split]['y']
        enhanced_data[split]['base_prices'] = all_data[ticker][split]['base_prices']
        enhanced_data[split]['dates'] = all_data[ticker][split]['dates']

        n_cross_features = n_cross

    # Save enhanced data
    with open(f'data/cross_stock_ready/{ticker}_cross_stock.pkl', 'wb') as f:
        pickle.dump(enhanced_data, f)

    original_features = all_data[ticker]['train']['X'].shape[2]
    new_features = enhanced_data['train']['X'].shape[2]

    print(f"   ✓ {ticker}: {original_features} → {new_features} features (+{n_cross_features} cross-stock)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-STOCK FEATURES SUMMARY".center(80))
print("=" * 80)

print(f"\n   Cross-stock features added per stock:")
print(f"   - {len(STOCKS) - 1} other stocks' returns (same-day)")
print(f"   - sector_avg_return (average of all banks)")
print(f"   - sector_breadth (% of banks with positive returns)")
print(f"   - sector_volatility (std of returns)")
print(f"   - relative_strength (stock vs sector)")
print(f"   - sector_consensus (directional agreement)")
print(f"\n   Total new features: ~{n_cross_features}")

print(f"\n   Output: data/cross_stock_ready/{'{TICKER}'}_cross_stock.pkl")

print("\n   USAGE:")
print("   To train V8 with cross-stock features, modify train_v7_final_push.py:")
print("   Change: 'data/enhanced_model_ready' → 'data/cross_stock_ready'")
print("   Change: '{TICKER}_enhanced.pkl' → '{TICKER}_cross_stock.pkl'")

print("\n   EXPECTED IMPROVEMENT:")
print("   Cross-stock features capture sector momentum and correlation")
print("   Expected gain: +1-2% directional accuracy")
print("   Best for: Correlated assets (banks, tech stocks, etc.)")

print("\n" + "=" * 80)
print("✓ CROSS-STOCK FEATURES CREATED!".center(80))
print("=" * 80)
