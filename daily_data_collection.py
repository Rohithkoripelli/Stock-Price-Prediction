"""
Daily Data Collection Script
Fetches latest stock data, technical indicators, and news for all bank stocks
Run daily to keep data current
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import json
from pathlib import Path

# Stock tickers
STOCKS = [
    ('HDFCBANK.NS', 'HDFCBANK', 'HDFC Bank'),
    ('ICICIBANK.NS', 'ICICIBANK', 'ICICI Bank'),
    ('KOTAKBANK.NS', 'KOTAKBANK', 'Kotak Mahindra Bank'),
    ('AXISBANK.NS', 'AXISBANK', 'Axis Bank'),
    ('SBIN.NS', 'SBIN', 'State Bank of India'),
    ('PNB.NS', 'PNB', 'Punjab National Bank'),
    ('BANKBARODA.NS', 'BANKBARODA', 'Bank of Baroda'),
    ('CANBK.NS', 'CANBK', 'Canara Bank')
]

print("=" * 80)
print("DAILY DATA COLLECTION - Bank Stocks".center(80))
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
print("=" * 80)

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def fetch_stock_data(ticker, ticker_short, name):
    """Fetch latest stock data with technical indicators"""
    print(f"\nFetching data for {name} ({ticker_short})...")

    try:
        # Fetch last 100 days to have enough data for technical indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            print(f"  ✗ No data available for {ticker_short}")
            return None

        # Calculate technical indicators
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df)

        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()

        # Price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100

        # Save to CSV
        output_dir = Path('data/stocks/daily_updates')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{ticker_short}_latest.csv'
        df.to_csv(output_file)

        print(f"  ✓ Saved {len(df)} rows to {output_file}")
        print(f"  Latest close: ₹{df['Close'].iloc[-1]:.2f}")

        return df

    except Exception as e:
        print(f"  ✗ Error fetching data for {ticker_short}: {e}")
        return None

def fetch_news_data(ticker_short, name):
    """Fetch latest news from GNews API"""
    print(f"\nFetching news for {name}...")

    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        print("  ⚠ GNEWS_API_KEY not set, skipping news collection")
        return None

    try:
        # Fetch news from last 7 days
        query = f"{name} OR {ticker_short}"
        url = f"https://gnews.io/api/v4/search"
        params = {
            'q': query,
            'token': api_key,
            'lang': 'en',
            'country': 'in',
            'max': 10,
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])

            if articles:
                # Save news data
                output_dir = Path('data/news/daily_updates')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f'{ticker_short}_latest_news.json'

                with open(output_file, 'w') as f:
                    json.dump(articles, f, indent=2)

                print(f"  ✓ Saved {len(articles)} articles to {output_file}")
                return articles
            else:
                print(f"  ⚠ No news articles found")
                return None
        else:
            print(f"  ✗ API Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"  ✗ Error fetching news: {e}")
        return None

# Main execution
print("\n" + "=" * 80)
print("COLLECTING DATA FOR ALL STOCKS")
print("=" * 80)

results = {}
for ticker, ticker_short, name in STOCKS:
    stock_data = fetch_stock_data(ticker, ticker_short, name)
    news_data = fetch_news_data(ticker_short, name)

    results[ticker_short] = {
        'stock_data': stock_data is not None,
        'news_data': news_data is not None,
        'latest_price': stock_data['Close'].iloc[-1] if stock_data is not None else None
    }

# Summary
print("\n" + "=" * 80)
print("COLLECTION SUMMARY")
print("=" * 80)

success_count = sum(1 for r in results.values() if r['stock_data'])
print(f"\n✓ Successfully collected data for {success_count}/{len(STOCKS)} stocks")

for ticker_short, result in results.items():
    status = "✓" if result['stock_data'] else "✗"
    price = f"₹{result['latest_price']:.2f}" if result['latest_price'] else "N/A"
    print(f"{status} {ticker_short}: {price}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'stocks_updated': success_count,
    'total_stocks': len(STOCKS),
    'results': {k: {'success': v['stock_data'], 'price': float(v['latest_price']) if v['latest_price'] else None}
                for k, v in results.items()}
}

summary_file = Path('data/collection_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary saved to {summary_file}")
print("\n" + "=" * 80)
print("DATA COLLECTION COMPLETE!")
print("=" * 80)
