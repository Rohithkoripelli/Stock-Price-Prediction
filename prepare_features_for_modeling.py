import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FEATURE PREPARATION FOR HIERARCHICAL ATTENTION MODEL".center(80))
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

# Technical features to use (all numeric columns from technical indicators)
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

# Sentiment features to extract
SENTIMENT_FEATURES = [
    'sentiment_compound',
    'sentiment_positive',
    'sentiment_neutral',
    'sentiment_negative',
    'article_count'  # Number of articles that day
]

# Create output directory
os.makedirs('data/model_ready', exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_stock_technical_data(ticker, sector):
    """Load stock data with technical indicators"""
    sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
    file_path = f"data/stocks_with_indicators/{sector_dir}/{ticker}_with_indicators.csv"
    
    try:
        # Read CSV with proper header handling
        df = pd.read_csv(file_path, header=0, skiprows=[1], index_col=0)
        df.index = pd.to_datetime(df.index)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Sort by date
        df = df.sort_index()
        
        return df
    
    except Exception as e:
        print(f"  ✗ Error loading {ticker}: {e}")
        return None

def load_news_sentiment_data(ticker, sector):
    """Load and aggregate daily news sentiment"""
    # Try both NewsAPI and GNews sources
    sentiment_data = []
    
    # Load NewsAPI sentiment
    try:
        sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
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
            
            # Aggregate by date
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
    
    # Combine all sentiment sources
    if sentiment_data:
        combined = pd.concat(sentiment_data, ignore_index=True)
        
        # Aggregate by date (average if multiple sources for same day)
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
    # Ensure stock_df index is datetime
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)
    
    # Create a complete date range
    date_range = pd.date_range(start=stock_df.index.min(), end=stock_df.index.max(), freq='D')
    stock_df = stock_df.reindex(date_range)
    
    # Forward fill for weekends/holidays
    stock_df = stock_df.fillna(method='ffill')
    
    # Merge sentiment data
    if sentiment_df is not None:
        # Reindex sentiment to match stock dates
        sentiment_df = sentiment_df.reindex(stock_df.index)

        # ✅ IMPROVED: Forward-fill sentiment (carry forward last sentiment)
        # Then fill remaining NaNs at the beginning with 0
        for col in SENTIMENT_FEATURES:
            if col in sentiment_df.columns:
                sentiment_df[col] = sentiment_df[col].fillna(method='ffill').fillna(0)

        # Combine
        merged = stock_df.join(sentiment_df, how='left')
    else:
        # No sentiment data, create zero columns
        merged = stock_df.copy()
        for col in SENTIMENT_FEATURES:
            merged[col] = 0
    
    # Drop any remaining NaN rows
    merged = merged.dropna()
    
    return merged

def create_sequences(data, technical_cols, sentiment_cols, lookback=60):
    """Create time-series sequences for LSTM"""
    sequences = []
    targets = []
    dates = []
    
    # Ensure we have enough data
    if len(data) < lookback + 1:
        return None, None, None
    
    for i in range(lookback, len(data)):
        # Extract sequence
        seq_technical = data[technical_cols].iloc[i-lookback:i].values
        seq_sentiment = data[sentiment_cols].iloc[i-lookback:i].values
        
        # Target is next day's closing price
        target = data['Close'].iloc[i]
        date = data.index[i]
        
        sequence = {
            'technical': seq_technical,
            'sentiment': seq_sentiment
        }
        
        sequences.append(sequence)
        targets.append(target)
        dates.append(date)
    
    return sequences, np.array(targets), dates

def normalize_features(sequences_train, sequences_val, sequences_test, targets_train, targets_val, targets_test):
    """Normalize features using training data statistics"""
    # ✅ IMPROVED: Use StandardScaler instead of MinMaxScaler
    # StandardScaler is better for neural networks and handles outliers better
    # RobustScaler for target is less sensitive to extreme values
    scaler_tech = StandardScaler()
    scaler_sent = StandardScaler()
    scaler_target = RobustScaler()  # ✅ CRITICAL: Better for stock prices

    # Flatten training sequences for fitting
    train_tech_flat = np.vstack([seq['technical'] for seq in sequences_train])
    train_sent_flat = np.vstack([seq['sentiment'] for seq in sequences_train])

    # Fit scalers
    scaler_tech.fit(train_tech_flat)
    scaler_sent.fit(train_sent_flat)
    scaler_target.fit(targets_train.reshape(-1, 1))
    
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
    
    # Transform targets
    targets_train_scaled = scaler_target.transform(targets_train.reshape(-1, 1)).flatten()
    targets_val_scaled = scaler_target.transform(targets_val.reshape(-1, 1)).flatten()
    targets_test_scaled = scaler_target.transform(targets_test.reshape(-1, 1)).flatten()
    
    scalers = {
        'technical': scaler_tech,
        'sentiment': scaler_sent,
        'target': scaler_target
    }
    
    return (sequences_train_scaled, sequences_val_scaled, sequences_test_scaled,
            targets_train_scaled, targets_val_scaled, targets_test_scaled, scalers)

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
print("PROCESSING ALL BANKING STOCKS".center(80))
print("=" * 80)

feature_metadata = {
    'lookback_window': LOOKBACK_WINDOW,
    'train_split': TRAIN_SPLIT,
    'val_split': VAL_SPLIT,
    'test_split': TEST_SPLIT,
    'technical_features': TECHNICAL_FEATURES,
    'sentiment_features': SENTIMENT_FEATURES,
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
                print(f"   ✗ Failed to load stock data")
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
            
            # Step 4: Get feature columns
            available_tech_features = [f for f in TECHNICAL_FEATURES if f in merged_df.columns]
            available_sent_features = [f for f in SENTIMENT_FEATURES if f in merged_df.columns]
            
            print(f"      ✓ Technical features: {len(available_tech_features)}")
            print(f"      ✓ Sentiment features: {len(available_sent_features)}")
            
            # Step 5: Create sequences
            print(f"   4. Creating sequences (lookback={LOOKBACK_WINDOW})...")
            sequences, targets, dates = create_sequences(
                merged_df, 
                available_tech_features, 
                available_sent_features, 
                lookback=LOOKBACK_WINDOW
            )
            
            if sequences is None:
                print(f"      ✗ Not enough data for sequences")
                continue
            
            print(f"      ✓ Created {len(sequences)} sequences")
            
            # Step 6: Split data (temporal split - no shuffling!)
            n_samples = len(sequences)
            train_end = int(n_samples * TRAIN_SPLIT)
            val_end = int(n_samples * (TRAIN_SPLIT + VAL_SPLIT))
            
            sequences_train = sequences[:train_end]
            sequences_val = sequences[train_end:val_end]
            sequences_test = sequences[val_end:]
            
            targets_train = targets[:train_end]
            targets_val = targets[train_end:val_end]
            targets_test = targets[val_end:]
            
            dates_train = dates[:train_end]
            dates_val = dates[train_end:val_end]
            dates_test = dates[val_end:]
            
            print(f"   5. Data split:")
            print(f"      Train: {len(sequences_train)} samples ({dates_train[0]} to {dates_train[-1]})")
            print(f"      Val:   {len(sequences_val)} samples ({dates_val[0]} to {dates_val[-1]})")
            print(f"      Test:  {len(sequences_test)} samples ({dates_test[0]} to {dates_test[-1]})")
            
            # Step 7: Normalize
            print(f"   6. Normalizing features...")
            (seq_train_norm, seq_val_norm, seq_test_norm,
             tgt_train_norm, tgt_val_norm, tgt_test_norm, scalers) = normalize_features(
                sequences_train, sequences_val, sequences_test,
                targets_train, targets_val, targets_test
            )
            
            # Step 8: Prepare arrays
            X_tech_train, X_sent_train, y_train = prepare_for_model(seq_train_norm, tgt_train_norm)
            X_tech_val, X_sent_val, y_val = prepare_for_model(seq_val_norm, tgt_val_norm)
            X_tech_test, X_sent_test, y_test = prepare_for_model(seq_test_norm, tgt_test_norm)
            
            print(f"      ✓ Technical input shape: {X_tech_train.shape}")
            print(f"      ✓ Sentiment input shape: {X_sent_train.shape}")
            print(f"      ✓ Target shape: {y_train.shape}")
            
            # Step 9: Save processed data
            output_data = {
                'train': {
                    'X_technical': X_tech_train,
                    'X_sentiment': X_sent_train,
                    'y': y_train,
                    'dates': dates_train,
                    'y_original': targets_train
                },
                'val': {
                    'X_technical': X_tech_val,
                    'X_sentiment': X_sent_val,
                    'y': y_val,
                    'dates': dates_val,
                    'y_original': targets_val
                },
                'test': {
                    'X_technical': X_tech_test,
                    'X_sentiment': X_sent_test,
                    'y': y_test,
                    'dates': dates_test,
                    'y_original': targets_test
                },
                'scalers': scalers,
                'feature_names': {
                    'technical': available_tech_features,
                    'sentiment': available_sent_features
                },
                'metadata': {
                    'bank_name': bank_name,
                    'ticker': ticker,
                    'sector': sector,
                    'lookback_window': LOOKBACK_WINDOW,
                    'n_technical_features': len(available_tech_features),
                    'n_sentiment_features': len(available_sent_features)
                }
            }
            

            output_file = f"data/model_ready/{ticker}_features.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)
            
            print(f"   7. ✓ Saved to: {output_file}")
            
            # Update metadata
            feature_metadata['stocks_processed'].append({
                'bank_name': bank_name,
                'ticker': ticker,
                'sector': sector,
                'n_samples': n_samples,
                'train_samples': len(sequences_train),
                'val_samples': len(sequences_val),
                'test_samples': len(sequences_test),
                'date_range': f"{dates[0]} to {dates[-1]}",
                'n_technical_features': len(available_tech_features),
                'n_sentiment_features': len(available_sent_features)
            })
            
        except Exception as e:
            print(f"   ✗ Error processing {bank_name}: {e}")
            import traceback
            traceback.print_exc()

# Save metadata
metadata_file = 'data/model_ready/feature_metadata.json'
with open(metadata_file, 'w') as f:
    # Convert dates to strings for JSON serialization
    for stock in feature_metadata['stocks_processed']:
        pass  # dates already strings
    json.dump(feature_metadata, f, indent=2, default=str)

print("\n\n" + "=" *80)
print("FEATURE PREPARATION COMPLETE!".center(80))
print("=" * 80)

print(f"\n✓ Processed {len(feature_metadata['stocks_processed'])} stocks")
print(f"✓ Metadata saved: {metadata_file}")

print("\nProcessed Stocks:")
for stock in feature_metadata['stocks_processed']:
    print(f"  • {stock['bank_name']}: {stock['train_samples']} train, {stock['val_samples']} val, {stock['test_samples']} test")

print("\n" + "=" * 80)
print("Ready for model training!")
print("=" * 80)
