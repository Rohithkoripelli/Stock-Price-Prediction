"""
Collect 30 days of news for all 8 banking stocks
Quick collection for FinBERT training
"""

from gnews import GNews
import pandas as pd
from datetime import datetime
import os
import time
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('Collecting 30 days news for all 8 stocks')
print('='*70)

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

os.makedirs('data/news_historical', exist_ok=True)

# Initialize GNews
google_news = GNews(language='en', country='IN', period='30d', max_results=100)

all_results = []

for ticker, name in STOCKS.items():
    print(f'\n{ticker} ({name})')
    print('-'*60)

    all_articles = []

    # Search with stock name
    print(f'  Searching for: {name}...')
    try:
        articles = google_news.get_news(name)
        print(f'  Found: {len(articles)} articles')

        for article in articles:
            all_articles.append({
                'ticker': ticker,
                'date': article.get('published date', datetime.now().strftime('%Y-%m-%d')),
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('publisher', {}).get('title', 'Unknown')
            })
    except Exception as e:
        print(f'  Error: {e}')

    # Also search with ticker if different
    if ticker != name:
        print(f'  Searching for: {ticker}...')
        try:
            articles2 = google_news.get_news(ticker)
            print(f'  Found: {len(articles2)} additional articles')

            for article in articles2:
                all_articles.append({
                    'ticker': ticker,
                    'date': article.get('published date', datetime.now().strftime('%Y-%m-%d')),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('publisher', {}).get('title', 'Unknown')
                })
        except Exception as e:
            print(f'  Error: {e}')

    # Remove duplicates
    df = pd.DataFrame(all_articles)
    if len(df) > 0:
        df = df.drop_duplicates(subset=['title'], keep='first')

        # Save
        output_file = f'data/news_historical/{ticker}_news_30d.csv'
        df.to_csv(output_file, index=False)

        print(f'  ✓ Saved {len(df)} unique articles')
        all_results.append({'ticker': ticker, 'articles': len(df)})
    else:
        print(f'  ⚠ No articles found')

    # Rate limiting
    time.sleep(3)

# Summary
print('\n' + '='*70)
print('COLLECTION SUMMARY')
print('='*70)
if all_results:
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    print(f'\nTotal articles: {summary_df["articles"].sum()}')

    # Save summary
    summary_df.to_csv('data/news_historical/collection_summary.csv', index=False)
else:
    print('No articles collected')
