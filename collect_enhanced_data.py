"""
Enhanced Data Collection: News, Fundamentals, Macro, Market Data

This script collects comprehensive data from multiple sources:
1. More news from GNews and NewsAPI
2. Company fundamentals from Yahoo Finance
3. Macroeconomic indicators
4. Market indices and sector data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json
import os
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BANKING_STOCKS = {
    'Private Banks': {
        'HDFC Bank': 'HDFCBANK.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Axis Bank': 'AXISBANK.NS'
    },
    'PSU Banks': {
        'State Bank of India': 'SBIN.NS',
        'Punjab National Bank': 'PNB.NS',
        'Bank of Baroda': 'BANKBARODA.NS',
        'Canara Bank': 'CANBK.NS'
    }
}

# API Keys (you'll need to get these)
NEWSAPI_KEY = "85368198a0b44776a0562a593c413723"  # Get from https://newsapi.org
GNEWS_API_KEY = "d2135a1a96fabb6b66d5dd66516a1871"  # Get from https://gnews.io

# Date range
START_DATE = '2019-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Output directory
os.makedirs('data/enhanced', exist_ok=True)
os.makedirs('data/enhanced/fundamentals', exist_ok=True)
os.makedirs('data/enhanced/news', exist_ok=True)
os.makedirs('data/enhanced/macro', exist_ok=True)
os.makedirs('data/enhanced/market', exist_ok=True)

# =============================================================================
# 1. COMPANY FUNDAMENTALS (Yahoo Finance)
# =============================================================================

def collect_fundamentals(ticker, stock_name):
    """
    Collect company fundamentals from Yahoo Finance

    Returns:
    - P/E ratio, EPS, Revenue, Profit Margin
    - Debt/Equity, ROE, ROA
    - Book Value, Market Cap
    - Dividend Yield
    """
    print(f"   Collecting fundamentals for {stock_name}...")

    try:
        stock = yf.Ticker(ticker)

        # Get info
        info = stock.info

        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials

        # Get quarterly earnings
        quarterly_earnings = stock.quarterly_earnings

        # Extract key metrics
        fundamentals = {
            'ticker': ticker,
            'stock_name': stock_name,
            'last_updated': datetime.now().isoformat(),

            # Valuation metrics
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'price_to_book': info.get('priceToBook', np.nan),
            'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),

            # Profitability
            'profit_margin': info.get('profitMargins', np.nan),
            'operating_margin': info.get('operatingMargins', np.nan),
            'roe': info.get('returnOnEquity', np.nan),
            'roa': info.get('returnOnAssets', np.nan),

            # Financial health
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),

            # Growth
            'revenue_growth': info.get('revenueGrowth', np.nan),
            'earnings_growth': info.get('earningsGrowth', np.nan),

            # Size
            'market_cap': info.get('marketCap', np.nan),
            'book_value': info.get('bookValue', np.nan),

            # Dividend
            'dividend_yield': info.get('dividendYield', np.nan),

            # Analyst ratings
            'recommendation': info.get('recommendationKey', 'none'),
            'target_mean_price': info.get('targetMeanPrice', np.nan)
        }

        # Get historical quarterly data
        if quarterly_earnings is not None and not quarterly_earnings.empty:
            fundamentals['quarterly_earnings'] = quarterly_earnings.to_dict()

        # Save
        output_file = f"data/enhanced/fundamentals/{ticker.replace('.NS', '')}_fundamentals.json"
        with open(output_file, 'w') as f:
            json.dump(fundamentals, f, indent=2, default=str)

        print(f"      ✓ Fundamentals saved: {output_file}")
        return fundamentals

    except Exception as e:
        print(f"      ✗ Error collecting fundamentals: {e}")
        return None

# =============================================================================
# 2. ENHANCED NEWS COLLECTION
# =============================================================================

def collect_enhanced_news(stock_name, ticker, api_type='gnews'):
    """
    Collect more news from multiple sources

    Sources:
    - GNews API: More comprehensive, recent news
    - NewsAPI: Business news, financial articles
    """
    print(f"   Collecting enhanced news for {stock_name} using {api_type}...")

    articles = []

    try:
        if api_type == 'gnews':
            # GNews API
            url = "https://gnews.io/api/v4/search"
            params = {
                'q': f'"{stock_name}" OR "{ticker.replace(".NS", "")}"',
                'lang': 'en',
                'country': 'in',
                'max': 100,
                'apikey': GNEWS_API_KEY,
                'from': START_DATE,
                'to': END_DATE
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                for article in data.get('articles', []):
                    # Sentiment analysis
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = TextBlob(text).sentiment

                    articles.append({
                        'date': article.get('publishedAt', '')[:10],
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity
                    })

        elif api_type == 'newsapi':
            # NewsAPI.org
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{stock_name}" OR "{ticker.replace(".NS", "")}"',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100,
                'apiKey': NEWSAPI_KEY,
                'from': START_DATE,
                'to': END_DATE
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                for article in data.get('articles', []):
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = TextBlob(text).sentiment

                    articles.append({
                        'date': article.get('publishedAt', '')[:10],
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity
                    })

        if articles:
            df = pd.DataFrame(articles)
            df['date'] = pd.to_datetime(df['date'])

            # Save
            ticker_clean = ticker.replace('.NS', '')
            output_file = f"data/enhanced/news/{ticker_clean}_{api_type}_news.csv"
            df.to_csv(output_file, index=False)

            print(f"      ✓ Collected {len(articles)} articles")
            print(f"      ✓ Saved to: {output_file}")
            return df
        else:
            print(f"      ⚠ No articles found")
            return None

    except Exception as e:
        print(f"      ✗ Error collecting news: {e}")
        return None

# =============================================================================
# 3. MACROECONOMIC INDICATORS
# =============================================================================

def collect_macro_indicators():
    """
    Collect macroeconomic indicators for India

    Sources:
    - Yahoo Finance (indices, currency)
    - Manual data entry (RBI rates, inflation)
    """
    print("\n   Collecting macroeconomic indicators...")

    try:
        # Get market indices
        nifty = yf.download('^NSEI', start=START_DATE, end=END_DATE, progress=False)
        banknifty = yf.download('^NSEBANK', start=START_DATE, end=END_DATE, progress=False)
        usdinr = yf.download('INR=X', start=START_DATE, end=END_DATE, progress=False)

        # Combine
        macro_data = pd.DataFrame({
            'date': nifty.index,
            'nifty_close': nifty['Close'].values,
            'nifty_return': nifty['Close'].pct_change().values * 100,
            'banknifty_close': banknifty['Close'].values,
            'banknifty_return': banknifty['Close'].pct_change().values * 100,
            'usdinr': usdinr['Close'].values,
            'usdinr_change': usdinr['Close'].pct_change().values * 100
        })

        # Save
        output_file = "data/enhanced/macro/macro_indicators.csv"
        macro_data.to_csv(output_file, index=False)

        print(f"      ✓ Collected macro indicators")
        print(f"      ✓ Saved to: {output_file}")

        return macro_data

    except Exception as e:
        print(f"      ✗ Error collecting macro indicators: {e}")
        return None

# =============================================================================
# 4. SECTOR & MARKET DATA
# =============================================================================

def collect_sector_data():
    """
    Collect banking sector performance data
    """
    print("\n   Collecting sector data...")

    try:
        # Bank Nifty is the banking sector index
        banknifty = yf.download('^NSEBANK', start=START_DATE, end=END_DATE, progress=False)

        # Calculate sector metrics
        sector_data = pd.DataFrame({
            'date': banknifty.index,
            'sector_close': banknifty['Close'].values,
            'sector_volume': banknifty['Volume'].values,
            'sector_return': banknifty['Close'].pct_change().values * 100,
            'sector_volatility': banknifty['Close'].pct_change().rolling(20).std().values * 100
        })

        # Save
        output_file = "data/enhanced/market/banking_sector.csv"
        sector_data.to_csv(output_file, index=False)

        print(f"      ✓ Collected sector data")
        print(f"      ✓ Saved to: {output_file}")

        return sector_data

    except Exception as e:
        print(f"      ✗ Error collecting sector data: {e}")
        return None

# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def main():
    print("=" * 80)
    print("ENHANCED DATA COLLECTION".center(80))
    print("=" * 80)

    print("\n" + "=" * 80)
    print("1. COLLECTING MACROECONOMIC INDICATORS".center(80))
    print("=" * 80)
    collect_macro_indicators()

    print("\n" + "=" * 80)
    print("2. COLLECTING SECTOR DATA".center(80))
    print("=" * 80)
    collect_sector_data()

    print("\n" + "=" * 80)
    print("3. COLLECTING STOCK-SPECIFIC DATA".center(80))
    print("=" * 80)

    for sector, stocks in BANKING_STOCKS.items():
        print(f"\n{sector}:")
        print("-" * 80)

        for stock_name, ticker in stocks.items():
            print(f"\n📊 {stock_name} ({ticker})")

            # Fundamentals
            collect_fundamentals(ticker, stock_name)

            # Enhanced news (GNews)
            if GNEWS_API_KEY != "YOUR_GNEWS_KEY_HERE":
                collect_enhanced_news(stock_name, ticker, api_type='gnews')
                time.sleep(1)  # Rate limiting
            else:
                print("      ⚠ GNews API key not configured")

            # Enhanced news (NewsAPI)
            if NEWSAPI_KEY != "YOUR_NEWSAPI_KEY_HERE":
                collect_enhanced_news(stock_name, ticker, api_type='newsapi')
                time.sleep(1)  # Rate limiting
            else:
                print("      ⚠ NewsAPI key not configured")

            print()

    print("\n" + "=" * 80)
    print("✓ DATA COLLECTION COMPLETE!".center(80))
    print("=" * 80)

    print("\nNext steps:")
    print("1. Add API keys for GNews and NewsAPI")
    print("2. Run: python collect_enhanced_data.py")
    print("3. Run: python prepare_enhanced_features.py")
    print("4. Retrain model with enhanced features")

if __name__ == "__main__":
    main()
