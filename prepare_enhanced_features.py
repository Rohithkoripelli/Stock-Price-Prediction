"""
Prepare Enhanced Features for Modeling

Integrates:
1. Technical indicators (existing)
2. Enhanced news sentiment (more sources)
3. Company fundamentals (P/E, EPS, etc.)
4. Macroeconomic indicators (Nifty, Bank Nifty, USD/INR)
5. Sector performance (relative strength)

Creates feature-rich dataset for improved predictions
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENHANCED FEATURE PREPARATION".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

LOOKBACK_WINDOW = 60
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

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

os.makedirs('data/enhanced_model_ready', exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_technical_data(ticker, sector):
    """Load existing technical indicators"""
    sector_dir = 'private_banks' if sector == 'Private Banks' else 'psu_banks'
    file_path = f"data/stocks_with_indicators/{sector_dir}/{ticker}_with_indicators.csv"

    try:
        df = pd.read_csv(file_path, header=0, skiprows=[1], index_col=0)
        df.index = pd.to_datetime(df.index)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df.sort_index()
    except Exception as e:
        print(f"      ✗ Error loading technical data: {e}")
        return None

def load_enhanced_sentiment(ticker):
    """Load enhanced sentiment from multiple sources"""
    sentiment_dfs = []

    # GNews
    gnews_file = f"data/enhanced/news/{ticker}_gnews_news.csv"
    if os.path.exists(gnews_file):
        df = pd.read_csv(gnews_file)
        df['date'] = pd.to_datetime(df['date'])
        sentiment_dfs.append(df)

    # NewsAPI
    newsapi_file = f"data/enhanced/news/{ticker}_newsapi_news.csv"
    if os.path.exists(newsapi_file):
        df = pd.read_csv(newsapi_file)
        df['date'] = pd.to_datetime(df['date'])
        sentiment_dfs.append(df)

    # Original sentiment
    original_file = f"data/news/daily_sentiment/{ticker}_daily_sentiment.csv"
    if os.path.exists(original_file):
        df = pd.read_csv(original_file)
        df['date'] = pd.to_datetime(df['date'])
        # Rename columns to match
        if 'sentiment_compound' in df.columns:
            df['sentiment_polarity'] = df['sentiment_compound']
        sentiment_dfs.append(df)

    if not sentiment_dfs:
        print(f"      ⚠ No sentiment data found")
        return None

    # Combine all sources
    combined = pd.concat(sentiment_dfs, ignore_index=True)

    # Aggregate by date
    daily_sentiment = combined.groupby('date').agg({
        'sentiment_polarity': 'mean',
        'sentiment_subjectivity': 'mean',
        'title': 'count'
    }).reset_index()

    daily_sentiment.columns = ['date', 'sentiment_score', 'sentiment_subjectivity', 'news_count']
    daily_sentiment = daily_sentiment.set_index('date')

    print(f"      ✓ Loaded {len(combined)} articles, aggregated to {len(daily_sentiment)} days")
    return daily_sentiment

def load_fundamentals(ticker):
    """Load company fundamentals"""
    file_path = f"data/enhanced/fundamentals/{ticker}_fundamentals.json"

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            fundamentals = json.load(f)

        # Extract key metrics
        return {
            'pe_ratio': fundamentals.get('pe_ratio', np.nan),
            'price_to_book': fundamentals.get('price_to_book', np.nan),
            'roe': fundamentals.get('roe', np.nan),
            'debt_to_equity': fundamentals.get('debt_to_equity', np.nan),
            'profit_margin': fundamentals.get('profit_margin', np.nan),
            'dividend_yield': fundamentals.get('dividend_yield', np.nan)
        }
    else:
        print(f"      ⚠ No fundamentals found")
        return None

def load_macro_data():
    """Load macroeconomic indicators"""
    file_path = "data/enhanced/macro/macro_indicators.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        print(f"      ✓ Loaded macro data: {len(df)} days")
        return df
    else:
        print(f"      ⚠ No macro data found")
        return None

def load_sector_data():
    """Load banking sector data"""
    file_path = "data/enhanced/market/banking_sector.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        print(f"      ✓ Loaded sector data: {len(df)} days")
        return df
    else:
        print(f"      ⚠ No sector data found")
        return None

def merge_all_features(stock_df, sentiment_df, fundamentals, macro_df, sector_df):
    """Merge all features into one dataframe"""

    # Ensure stock_df index is datetime
    if not isinstance(stock_df.index, pd.DatetimeIndex):
        stock_df.index = pd.to_datetime(stock_df.index)

    # Create complete date range
    date_range = pd.date_range(start=stock_df.index.min(), end=stock_df.index.max(), freq='D')
    stock_df = stock_df.reindex(date_range)
    stock_df = stock_df.fillna(method='ffill')

    # Merge sentiment
    if sentiment_df is not None:
        sentiment_df = sentiment_df.reindex(stock_df.index)
        for col in ['sentiment_score', 'sentiment_subjectivity', 'news_count']:
            if col in sentiment_df.columns:
                sentiment_df[col] = sentiment_df[col].fillna(method='ffill').fillna(0)
        stock_df = stock_df.join(sentiment_df, how='left')
    else:
        stock_df['sentiment_score'] = 0
        stock_df['sentiment_subjectivity'] = 0
        stock_df['news_count'] = 0

    # Add fundamentals (constant for the period)
    if fundamentals:
        for key, value in fundamentals.items():
            stock_df[f'fund_{key}'] = value if not pd.isna(value) else 0

    # Merge macro data
    if macro_df is not None:
        macro_df = macro_df.reindex(stock_df.index)
        macro_df = macro_df.fillna(method='ffill')
        stock_df = stock_df.join(macro_df, how='left', rsuffix='_macro')

    # Merge sector data
    if sector_df is not None:
        sector_df = sector_df.reindex(stock_df.index)
        sector_df = sector_df.fillna(method='ffill')
        stock_df = stock_df.join(sector_df, how='left', rsuffix='_sector')

    # Calculate relative strength vs sector
    if 'sector_return' in stock_df.columns and 'Close' in stock_df.columns:
        stock_df['relative_strength'] = stock_df['Close'].pct_change() - stock_df['sector_return'] / 100

    stock_df = stock_df.dropna()
    return stock_df

def create_percentage_change_features(data):
    """Convert to percentage changes"""
    data_pct = data.copy()

    # Price features
    price_features = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'SMA_200',
                     'EMA_12', 'EMA_26', 'BB_Middle', 'BB_Upper', 'BB_Lower']

    for col in price_features:
        if col in data_pct.columns:
            data_pct[f'{col}_pct'] = data_pct[col].pct_change() * 100

    # Volume log transform
    if 'Volume' in data_pct.columns:
        data_pct['Volume_log'] = np.log1p(data_pct['Volume'])

    data_pct = data_pct.iloc[1:].copy()
    return data_pct

def create_sequences_enhanced(data, lookback=60):
    """Create sequences with enhanced features"""

    if len(data) < lookback + 2:
        return None, None, None, None

    # Define feature groups
    tech_pct_features = [f'{col}_pct' for col in ['Open', 'High', 'Low', 'Close', 'SMA_20',
                                                    'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                                                    'BB_Middle', 'BB_Upper', 'BB_Lower']]
    tech_pct_features = [f for f in tech_pct_features if f in data.columns]

    other_tech = ['Volume_log', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI_14',
                  'BB_Percent', 'ATR_14', 'ADX', 'DI_Plus', 'DI_Minus',
                  'Stoch_K', 'Stoch_D', 'OBV', 'CCI', 'MFI', 'ROC', 'Williams_R']
    other_tech = [f for f in other_tech if f in data.columns]

    # Sentiment features
    sentiment_features = ['sentiment_score', 'sentiment_subjectivity', 'news_count']
    sentiment_features = [f for f in sentiment_features if f in data.columns]

    # Fundamental features
    fundamental_features = [col for col in data.columns if col.startswith('fund_')]

    # Macro features
    macro_features = ['nifty_return', 'banknifty_return', 'usdinr_change']
    macro_features = [f for f in macro_features if f in data.columns]

    # Sector features
    sector_features = ['sector_return', 'sector_volatility', 'relative_strength']
    sector_features = [f for f in sector_features if f in data.columns]

    # Combine all
    all_features = (tech_pct_features + other_tech + sentiment_features +
                   fundamental_features + macro_features + sector_features)

    sequences = []
    targets = []
    base_prices = []
    dates = []

    for i in range(lookback, len(data) - 1):
        seq = data[all_features].iloc[i-lookback:i].values

        current_close = data['Close'].iloc[i]
        next_close = data['Close'].iloc[i + 1]
        target_pct_change = ((next_close - current_close) / current_close) * 100

        sequences.append(seq)
        targets.append(target_pct_change)
        base_prices.append(current_close)
        dates.append(data.index[i + 1])

    print(f"      ✓ Created sequences with {len(all_features)} features")
    print(f"        - Technical: {len(tech_pct_features + other_tech)}")
    print(f"        - Sentiment: {len(sentiment_features)}")
    print(f"        - Fundamental: {len(fundamental_features)}")
    print(f"        - Macro: {len(macro_features)}")
    print(f"        - Sector: {len(sector_features)}")

    return sequences, np.array(targets), np.array(base_prices), dates

# =============================================================================
# MAIN PROCESSING
# =============================================================================

print("\n" + "=" * 80)
print("LOADING MACRO AND SECTOR DATA".center(80))
print("=" * 80)

macro_df = load_macro_data()
sector_df = load_sector_data()

print("\n" + "=" * 80)
print("PROCESSING STOCKS".center(80))
print("=" * 80)

for sector, banks in BANKING_STOCKS.items():
    print(f"\n{sector}:")
    print("-" * 80)

    for bank_name, ticker in banks.items():
        print(f"\n📊 {bank_name} ({ticker})")

        try:
            # Load all data
            print("   1. Loading data...")
            stock_df = load_technical_data(ticker, sector)
            if stock_df is None:
                continue

            sentiment_df = load_enhanced_sentiment(ticker)
            fundamentals = load_fundamentals(ticker)

            # Merge all features
            print("   2. Merging features...")
            merged_df = merge_all_features(stock_df, sentiment_df, fundamentals,
                                          macro_df, sector_df)
            print(f"      ✓ Merged data: {len(merged_df)} records")
            print(f"      ✓ Total features: {len(merged_df.columns)}")

            # Create percentage change features
            print("   3. Creating percentage change features...")
            pct_df = create_percentage_change_features(merged_df)

            # Create sequences
            print("   4. Creating sequences...")
            sequences, targets, base_prices, dates = create_sequences_enhanced(
                pct_df, lookback=LOOKBACK_WINDOW
            )

            if sequences is None:
                print(f"      ✗ Not enough data")
                continue

            print(f"      ✓ Created {len(sequences)} sequences")

            # Split data
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

            # Normalize
            print("   5. Normalizing...")
            scaler = StandardScaler()
            train_flat = np.vstack(sequences_train)
            scaler.fit(train_flat)

            sequences_train_scaled = [scaler.transform(seq) for seq in sequences_train]
            sequences_val_scaled = [scaler.transform(seq) for seq in sequences_val]
            sequences_test_scaled = [scaler.transform(seq) for seq in sequences_test]

            # Prepare arrays
            X_train = np.array(sequences_train_scaled)
            X_val = np.array(sequences_val_scaled)
            X_test = np.array(sequences_test_scaled)

            # Save
            output_data = {
                'train': {
                    'X': X_train,
                    'y': targets_train,
                    'base_prices': base_prices_train,
                    'dates': dates_train
                },
                'val': {
                    'X': X_val,
                    'y': targets_val,
                    'base_prices': base_prices_val,
                    'dates': dates_val
                },
                'test': {
                    'X': X_test,
                    'y': targets_test,
                    'base_prices': base_prices_test,
                    'dates': dates_test
                },
                'scaler': scaler,
                'metadata': {
                    'bank_name': bank_name,
                    'ticker': ticker,
                    'n_features': X_train.shape[2],
                    'lookback': LOOKBACK_WINDOW
                }
            }

            output_file = f"data/enhanced_model_ready/{ticker}_enhanced.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(output_data, f)

            print(f"   6. ✓ Saved: {output_file}")
            print(f"      Features: {X_train.shape[2]} (was 20-25)")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

print("\n" + "=" * 80)
print("✓ ENHANCED FEATURE PREPARATION COMPLETE!".center(80))
print("=" * 80)
