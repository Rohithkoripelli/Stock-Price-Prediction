import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE PREPARATION V2 - PERCENTAGE CHANGE APPROACH".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

LOOKBACK_WINDOW = 60  # Number of days to look backward
TRAIN_SPLIT = 0.70  # 70% for training
VAL_SPLIT = 0.15  # 15% for validation
TEST_SPLIT = 0.15  # 15% for testing

BANKING_STOCKS = {
    'Private Banks': {
        'HDFC Bank': 'HDFCBANK',
        'ICICI Bank': 'ICICIBANK',
        'Kotak Mahindra Bank': 'KOTAKBANK',
        'Axis Bank': 'AXISBANK'
    },
    'PSU Banks': {
        'State Bank of India': 'SBIN',
        'Punjab National Bank': 'PNB',
        'Bank of Baroda': 'BANKBARODA',
        'Canara Bank': 'CANBK'
    }
}

# Technical features to use
TECHNICAL_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_12', 'EMA_26',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'RSI_14',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Percent',
    'ATR_14',
    'ADX', 'DI_Plus', 'DI_Minus',
    'Stoch_K', 'Stoch_D',
    'OBV',
    'CCI',
    'MFI',
    'ROC',
    'Williams_R'
]

# Sentiment features
SENTIMENT_FEATURES = [
    'sentiment_compound',
    'sentiment_positive',
    'sentiment_neutral',
    'sentiment_negative',
    'article_count'
]

os.makedirs('data/model_ready_v2', exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_stock_technical_data(ticker, sector):
    """Load stock data with technical indicators"""
    sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
    file_path = f"data/stocks_with_indicators/{sector_dir}/{ticker}_with_indicators.csv"

    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1], index_col=0)
        df.index = pd.to_datetime(df.index)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        df = df.sort_index()
        return df

    except Exception as e:
        print(f"  ✗ Error loading {ticker}: {e}")
        return None

def load_news_sentiment_data(ticker, sector):
    """Load and aggregate daily news sentiment"""
    sentiment_data = []

    # Load NewsAPI sentiment
    try:
        newsapi_file = f"data/news/daily_sentiment/{ticker}_daily_sentiment.csv"
        if os.path.exists(newsapi_file):
            df_newsapi = pd.read_csv(newsapi_file)
            df_newsapi['date'] = pd.to_datetime(df_newsapi['date'])
            df_newsapi['source'] = 'newsapi'
            sentiment_data.append(df_newsapi)
    except:
        pass

    # Load GNews sentiment
    try:
        sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
        gnews_file = f"data/news/gnews/{sector_dir}/{ticker}_gnews.csv"
        if os.path.exists(gnews_file):
            df_gnews = pd.read_csv(gnews_file)
            df_gnews['published_date'] = pd.to_datetime(df_gnews['published_date'], errors='coerce')

            daily_gnews = df_gnews.groupby(df_gnews['published_date'].dt.date).agg({
                'sentiment_compound': 'mean',
                'sentiment_positive': 'mean',
                'sentiment_neutral': 'mean',
                'sentiment_negative': 'mean',
                'title': 'count'
            }).reset_index()

            daily_gnews.columns = ['date', 'sentiment_compound', 'sentiment_positive',
                                   'sentiment_neutral', 'sentiment_negative', 'article_count']
            daily_gnews['date'] = pd.to_datetime(daily_gnews['date'])
            daily_gnews['source'] = 'gnews'
            sentiment_data.append(daily_gnews)
    except:
        pass

    if sentiment_data:
        combined = pd.concat(sentiment_data, ignore_index=True)
        daily_sentiment = combined.groupby('date').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_neutral': 'mean',
            'sentiment_negative': 'mean',
            'article_count': 'sum'
        }).reset_index()

        daily_sentiment = daily_sentiment.set_index('date')
        return daily_sentiment

    return None

def merge_features(stock_df, sentiment_df):
    """Merge stock and sentiment data by date"""
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)

    date_range = pd.date_range(start=stock_df.index.min(), end=stock_df.index.max(), freq='D')
    stock_df = stock_df.reindex(date_range)
    stock_df = stock_df.fillna(method='ffill')

    if sentiment_df is not None:
        sentiment_df = sentiment_df.reindex(stock_df.index)
        for col in SENTIMENT_FEATURES:
            if col in sentiment_df.columns:
                sentiment_df[col] = sentiment_df[col].fillna(method='ffill').fillna(0)

        merged = stock_df.join(sentiment_df, how='left')
    else:
        merged = stock_df.copy()
        for col in SENTIMENT_FEATURES:
            merged[col] = 0

    merged = merged.dropna()
    return merged

def create_percentage_change_features(data, technical_cols):
    """
    ✅ KEY IMPROVEMENT: Convert to percentage changes instead of absolute values
    This makes the problem stationary and much easier to predict
    """
    data_pct = data.copy()

    # Calculate percentage change for price-based features
    price_features = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'SMA_200',
                     'EMA_12', 'EMA_26', 'BB_Middle', 'BB_Upper', 'BB_Lower']

    for col in price_features:
        if col in data_pct.columns:
            data_pct[f'{col}_pct'] = data_pct[col].pct_change() * 100  # Percentage change

    # Keep volume as log-transform
    if 'Volume' in data_pct.columns:
        data_pct['Volume_log'] = np.log1p(data_pct['Volume'])

    # Other technical indicators can stay as-is (they're already normalized)

    # Drop first row (has NaN from pct_change)
    data_pct = data_pct.iloc[1:].copy()

    return data_pct

def create_sequences_v2(data, lookback=60):
    """
    ✅ IMPROVED: Create sequences with percentage changes for technical features
    Predict next day's price percentage change instead of absolute price
    """
    sequences = []
    targets = []
    target_base_prices = []  # Store base price for converting back
    dates = []

    if len(data) < lookback + 2:
        return None, None, None, None

    # Technical features (percentage changes)
    tech_pct_features = [f'{col}_pct' for col in ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50',
                                                    'SMA_200', 'EMA_12', 'EMA_26', 'BB_Middle',
                                                    'BB_Upper', 'BB_Lower']]
    tech_pct_features = [f for f in tech_pct_features if f in data.columns]

    # Add other technical indicators
    other_tech = ['Volume_log', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI_14',
                  'BB_Percent', 'ATR_14', 'ADX', 'DI_Plus', 'DI_Minus',
                  'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'MFI', 'ROC', 'Williams_R']
    other_tech = [f for f in other_tech if f in data.columns]

    technical_features = tech_pct_features + other_tech

    # Sentiment features
    sentiment_features = [f for f in SENTIMENT_FEATURES if f in data.columns]

    for i in range(lookback, len(data) - 1):  # -1 because we need next day
        # Extract sequence
        seq_technical = data[technical_features].iloc[i-lookback:i].values
        seq_sentiment = data[sentiment_features].iloc[i-lookback:i].values

        # ✅ KEY: Target is PERCENTAGE CHANGE to next day's close
        current_close = data['Close'].iloc[i]
        next_close = data['Close'].iloc[i + 1]
        target_pct_change = ((next_close - current_close) / current_close) * 100

        date = data.index[i + 1]

        sequence = {
            'technical': seq_technical,
            'sentiment': seq_sentiment
        }

        sequences.append(sequence)
        targets.append(target_pct_change)
        target_base_prices.append(current_close)  # Store for converting back
        dates.append(date)

    return sequences, np.array(targets), np.array(target_base_prices), dates

def normalize_features_v2(sequences_train, sequences_val, sequences_test):
    """
    ✅ IMPROVED: Only normalize features, not targets (targets are already percentage changes)
    """
    scaler_tech = StandardScaler()
    scaler_sent = StandardScaler()

    # Flatten training sequences for fitting
    train_tech_flat = np.vstack([seq['technical'] for seq in sequences_train])
    train_sent_flat = np.vstack([seq['sentiment'] for seq in sequences_train])

    # Fit scalers
    scaler_tech.fit(train_tech_flat)
    scaler_sent.fit(train_sent_flat)

    # Transform function
    def transform_sequences(sequences, scaler_tech, scaler_sent):
        transformed = []
        for seq in sequences:
            tech_scaled = scaler_tech.transform(seq['technical'])
            sent_scaled = scaler_sent.transform(seq['sentiment'])
            transformed.append({
                'technical': tech_scaled,
                'sentiment': sent_scaled
            })
        return transformed

    # Transform all sets
    sequences_train_scaled = transform_sequences(sequences_train, scaler_tech, scaler_sent)
    sequences_val_scaled = transform_sequences(sequences_val, scaler_tech, scaler_sent)
    sequences_test_scaled = transform_sequences(sequences_test, scaler_tech, scaler_sent)

    scalers = {
        'technical': scaler_tech,
        'sentiment': scaler_sent
    }

    return sequences_train_scaled, sequences_val_scaled, sequences_test_scaled, scalers

def prepare_for_model(sequences, targets):
    """Convert sequences to numpy arrays ready for model"""
    X_technical = np.array([seq['technical'] for seq in sequences])
    X_sentiment = np.array([seq['sentiment'] for seq in sequences])
    y = np.array(targets)

    return X_technical, X_sentiment, y

# =============================================================================
# MAIN PROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("PROCESSING ALL BANKING STOCKS (V2 - PERCENTAGE CHANGE)".center(80))
print("=" * 80)

feature_metadata = {
    'lookback_window': LOOKBACK_WINDOW,
    'train_split': TRAIN_SPLIT,
    'val_split': VAL_SPLIT,
    'test_split': TEST_SPLIT,
    'approach': 'percentage_change',
    'stocks_processed': []
}

for sector, banks in BANKING_STOCKS.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)

    for bank_name, ticker in banks.items():
        print(f"\n📊 Processing: {bank_name} ({ticker})")
        print("-" * 60)

        try:
            # Step 1: Load data
            print(f"   1. Loading stock data...")
            stock_df = load_stock_technical_data(ticker, sector)

            if stock_df is None:
                continue

            print(f"      ✓ Loaded {len(stock_df)} records")

            # Step 2: Load sentiment
            print(f"   2. Loading sentiment data...")
            sentiment_df = load_news_sentiment_data(ticker, sector)

            if sentiment_df is not None:
                print(f"      ✓ Loaded {len(sentiment_df)} sentiment records")
            else:
                print(f"      ⚠ No sentiment data, using zeros")

            # Step 3: Merge features
            print(f"   3. Merging features...")
            merged_df = merge_features(stock_df, sentiment_df)
            print(f"      ✓ Merged data: {len(merged_df)} records")

            # Step 4: Create percentage change features
            print(f"   4. Creating percentage change features...")
            pct_df = create_percentage_change_features(merged_df, TECHNICAL_FEATURES)
            print(f"      ✓ Percentage change features created")

            # Step 5: Create sequences
            print(f"   5. Creating sequences (lookback={LOOKBACK_WINDOW})...")
            sequences, targets, base_prices, dates = create_sequences_v2(
                pct_df,
                lookback=LOOKBACK_WINDOW
            )

            if sequences is None:
                print(f"      ✗ Not enough data for sequences")
                continue

            print(f"      ✓ Created {len(sequences)} sequences")
            print(f"      ✓ Target: percentage change to next day's close")

            # Step 6: Split data
            n_samples = len(sequences)
            train_end = int(n_samples * TRAIN_SPLIT)
            val_end = int(n_samples * (TRAIN_SPLIT + VAL_SPLIT))

            sequences_train = sequences[:train_end]
            sequences_val = sequences[train_end:val_end]
            sequences_test = sequences[val_end:]

            targets_train = targets[:train_end]
            targets_val = targets[train_end:val_end]
            targets_test = targets[val_end:]

            base_prices_train = base_prices[:train_end]
            base_prices_val = base_prices[train_end:val_end]
            base_prices_test = base_prices[val_end:]

            dates_train = dates[:train_end]
            dates_val = dates[train_end:val_end]
            dates_test = dates[val_end:]

            print(f"   6. Data split:")
            print(f"      Train: {len(sequences_train)} samples")
            print(f"      Val:   {len(sequences_val)} samples")
            print(f"      Test:  {len(sequences_test)} samples")

            # Step 7: Normalize (only features, not targets!)
            print(f"   7. Normalizing features...")
            (seq_train_norm, seq_val_norm, seq_test_norm, scalers) = normalize_features_v2(
                sequences_train, sequences_val, sequences_test
            )

            # Step 8: Prepare arrays
            X_tech_train, X_sent_train, y_train = prepare_for_model(seq_train_norm, targets_train)
            X_tech_val, X_sent_val, y_val = prepare_for_model(seq_val_norm, targets_val)
            X_tech_test, X_sent_test, y_test = prepare_for_model(seq_test_norm, targets_test)

            print(f"      ✓ Technical input shape: {X_tech_train.shape}")
            print(f"      ✓ Sentiment input shape: {X_sent_train.shape}")
            print(f"      ✓ Target shape: {y_train.shape} (percentage changes)")

            # Step 9: Save processed data
            output_data = {
                'train': {
                    'X_technical': X_tech_train,
                    'X_sentiment': X_sent_train,
                    'y': y_train,  # Percentage changes
                    'base_prices': base_prices_train,
                    'dates': dates_train
                },
                'val': {
                    'X_technical': X_tech_val,
                    'X_sentiment': X_sent_val,
                    'y': y_val,
                    'base_prices': base_prices_val,
                    'dates': dates_val
                },
                'test': {
                    'X_technical': X_tech_test,
                    'X_sentiment': X_sent_test,
                    'y': y_test,
                    'base_prices': base_prices_test,
                    'dates': dates_test
                },
                'scalers': scalers,
                'metadata': {
                    'bank_name': bank_name,
                    'ticker': ticker,
                    'sector': sector,
                    'lookback_window': LOOKBACK_WINDOW,
                    'target_type': 'percentage_change',
                    'n_features': X_tech_train.shape[2] + X_sent_train.shape[2]
                }
            }

            output_file = f"data/model_ready_v2/{ticker}_features.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)

            print(f"   8. ✓ Saved to: {output_file}")

            feature_metadata['stocks_processed'].append({
                'bank_name': bank_name,
                'ticker': ticker,
                'sector': sector,
                'n_samples': n_samples,
                'train_samples': len(sequences_train),
                'val_samples': len(sequences_val),
                'test_samples': len(sequences_test)
            })

        except Exception as e:
            print(f"   ✗ Error processing {bank_name}: {e}")
            import traceback
            traceback.print_exc()

# Save metadata
metadata_file = 'data/model_ready_v2/feature_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(feature_metadata, f, indent=2, default=str)

print("\n\n" + "=" * 80)
print("FEATURE PREPARATION V2 COMPLETE!".center(80))
print("=" * 80)

print(f"\n✓ Processed {len(feature_metadata['stocks_processed'])} stocks")
print(f"✓ Using percentage change approach (stationary data)")
print(f"✓ Metadata saved: {metadata_file}")

print("\n" + "=" * 80)
print("Ready for model training (V2)!")
print("=" * 80)
