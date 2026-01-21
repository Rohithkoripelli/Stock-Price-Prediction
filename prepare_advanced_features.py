"""
Prepare Features with Advanced Signals
Integrates: Technical Indicators (35) + FinBERT (4) + Advanced Signals (11) = 50 features
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED FEATURE PREPARATION (50 Features)".center(80))
print("=" * 80)

# Configuration
LOOKBACK_WINDOW = 60
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

STOCKS = {
    'Private Banks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK'],
    'PSU Banks': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK']
}

os.makedirs('data/advanced_model_ready', exist_ok=True)

def load_technical_indicators(ticker, sector):
    """Load technical indicators (35 features)"""
    sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
    file_path = f"data/stocks_with_indicators/{sector_dir}/{ticker}_with_indicators.csv"

    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1], index_col=0)
        df.index = pd.to_datetime(df.index)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.sort_index()
    except Exception as e:
        print(f"      ✗ Error loading technical data: {e}")
        return None

def load_finbert_sentiment(ticker):
    """Load FinBERT daily sentiment (4 features)"""
    file_path = f"data/finbert_daily_sentiment/{ticker}_daily_sentiment.csv"

    if not os.path.exists(file_path):
        print(f"      ⚠ No FinBERT sentiment found")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        print(f"      ✗ Error loading FinBERT data: {e}")
        return None

def load_advanced_signals(ticker):
    """Load advanced market signals (11 features)"""
    file_path = f"data/advanced_signals/{ticker}_advanced_signals.csv"

    if not os.path.exists(file_path):
        print(f"      ⚠ No advanced signals found")
        return None

    try:
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)

        # Convert boolean columns to numeric
        bool_cols = ['analyst_rating_present', 'earnings_event_present']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df.sort_index()
    except Exception as e:
        print(f"      ✗ Error loading advanced signals: {e}")
        return None

def create_sequences(data, lookback=60):
    """Create sequences for LSTM/Transformer"""
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # Predict close price

    return np.array(X), np.array(y)

def prepare_stock_data(ticker, sector):
    """Prepare data for one stock with all 50 features"""

    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    # Load technical indicators (35 features)
    print("  Loading technical indicators...")
    tech_df = load_technical_indicators(ticker, sector)

    if tech_df is None:
        return None

    print(f"    ✓ Loaded {len(tech_df)} days of technical data")
    print(f"    ✓ Technical features: {len(tech_df.columns)}")

    # Load FinBERT sentiment (4 features)
    print("  Loading FinBERT sentiment...")
    sentiment_df = load_finbert_sentiment(ticker)

    if sentiment_df is not None:
        print(f"    ✓ Loaded {len(sentiment_df)} days of sentiment data")
        combined = tech_df.join(sentiment_df, how='left')

        # Forward fill sentiment for days without news
        sentiment_cols = ['sentiment_polarity', 'sentiment_score', 'news_volume', 'earnings_event']
        for col in sentiment_cols:
            if col in combined.columns:
                combined[col] = combined[col].fillna(method='ffill').fillna(0)
    else:
        combined = tech_df
        print(f"    ⚠ No FinBERT features")

    # Load advanced signals (11 features)
    print("  Loading advanced signals...")
    advanced_df = load_advanced_signals(ticker)

    if advanced_df is not None:
        print(f"    ✓ Loaded {len(advanced_df)} days of advanced signals")
        combined = combined.join(advanced_df, how='left')

        # Forward fill advanced signals for days without news
        advanced_cols = [
            'technical_signal_score', 'technical_bullish_mentions', 'technical_bearish_mentions',
            'analyst_rating_score', 'analyst_rating_present',
            'macro_signal_score', 'macro_mentions',
            'risk_score', 'high_risk_mentions',
            'leadership_signal_score',
            'earnings_signal_score', 'earnings_event_present'
        ]
        for col in advanced_cols:
            if col in combined.columns:
                combined[col] = combined[col].fillna(method='ffill').fillna(0)

        print(f"    ✓ Total features: {len(combined.columns)}")
    else:
        print(f"    ⚠ No advanced signals, using {len(combined.columns)} features")

    # Drop rows with NaN
    initial_len = len(combined)
    combined = combined.dropna()
    print(f"    ✓ Cleaned data: {len(combined)} days ({initial_len - len(combined)} dropped)")

    if len(combined) < LOOKBACK_WINDOW + 100:
        print(f"    ✗ Insufficient data (need at least {LOOKBACK_WINDOW + 100})")
        return None

    # Normalize features
    print("  Normalizing features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined.values)

    # Create sequences
    print(f"  Creating sequences (lookback={LOOKBACK_WINDOW})...")
    X, y = create_sequences(scaled_data, LOOKBACK_WINDOW)

    print(f"    ✓ Created {len(X)} sequences")
    print(f"    ✓ Input shape: {X.shape}")
    print(f"    ✓ Output shape: {y.shape}")

    # Split data
    train_size = int(len(X) * TRAIN_SPLIT)
    val_size = int(len(X) * VAL_SPLIT)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\n  Data Split:")
    print(f"    Train: {len(X_train)} sequences")
    print(f"    Val:   {len(X_val)} sequences")
    print(f"    Test:  {len(X_test)} sequences")

    # Save
    output_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': list(combined.columns),
        'num_features': X.shape[2],
        'lookback': LOOKBACK_WINDOW
    }

    output_file = f"data/advanced_model_ready/{ticker}_advanced.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n  ✓ Saved to {output_file}")

    return {
        'ticker': ticker,
        'num_features': X.shape[2],
        'num_sequences': len(X),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }

# Main execution
if __name__ == "__main__":

    all_results = []

    for sector, tickers in STOCKS.items():
        print(f"\n\n{'#'*80}")
        print(f"# {sector.upper()}")
        print(f"{'#'*80}")

        for ticker in tickers:
            result = prepare_stock_data(ticker, sector)
            if result:
                all_results.append(result)

    # Summary
    print("\n\n" + "="*80)
    print("ADVANCED FEATURE PREPARATION COMPLETE".center(80))
    print("="*80)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + summary_df.to_string(index=False))

        print(f"\n✓ Prepared {len(all_results)} stocks")
        print(f"✓ Features per stock: {all_results[0]['num_features']}")
        print(f"✓ Feature Breakdown:")
        print(f"    - Technical Indicators: 35")
        print(f"    - FinBERT Sentiment: 4")
        print(f"    - Advanced Signals: 11")
        print(f"    - Total: {all_results[0]['num_features']}")
        print(f"✓ Total sequences: {sum(r['num_sequences'] for r in all_results)}")

        # Save summary
        summary_df.to_csv('data/advanced_model_ready/preparation_summary.csv', index=False)
    else:
        print("\n✗ No stocks prepared successfully")
