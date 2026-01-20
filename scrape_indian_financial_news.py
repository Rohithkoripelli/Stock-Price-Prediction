"""
Real-time Indian Financial News Scraper

Sources:
1. MoneyControl - Stock-specific news
2. Economic Times - Market news
3. LiveMint - Banking sector news
4. BSE/NSE announcements (corporate actions, results)

Focuses on:
- Quarterly results announcements
- Corporate actions (dividends, bonus, splits)
- Regulatory news (RBI policy, SEBI directives)
- Management changes
- NPA updates
- Merger & acquisition news
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INDIAN FINANCIAL NEWS SCRAPER".center(80))
print("Real-time news from MoneyControl, ET, LiveMint".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Stock name mappings for news search
STOCK_MAPPINGS = {
    'HDFCBANK': {
        'name': 'HDFC Bank',
        'search_terms': ['HDFC Bank', 'HDFCBANK', 'HDFC'],
        'moneycontrol_code': 'HB'  # MoneyControl specific
    },
    'ICICIBANK': {
        'name': 'ICICI Bank',
        'search_terms': ['ICICI Bank', 'ICICIBANK', 'ICICI'],
        'moneycontrol_code': 'ICI'
    },
    'KOTAKBANK': {
        'name': 'Kotak Mahindra Bank',
        'search_terms': ['Kotak Bank', 'Kotak Mahindra', 'KOTAKBANK'],
        'moneycontrol_code': 'KMB'
    },
    'AXISBANK': {
        'name': 'Axis Bank',
        'search_terms': ['Axis Bank', 'AXISBANK'],
        'moneycontrol_code': 'AB'
    },
    'SBIN': {
        'name': 'State Bank of India',
        'search_terms': ['SBI', 'State Bank of India', 'State Bank'],
        'moneycontrol_code': 'SBI'
    },
    'PNB': {
        'name': 'Punjab National Bank',
        'search_terms': ['PNB', 'Punjab National Bank'],
        'moneycontrol_code': 'PNB'
    },
    'BANKBARODA': {
        'name': 'Bank of Baroda',
        'search_terms': ['Bank of Baroda', 'BOB', 'BANKBARODA'],
        'moneycontrol_code': 'BOB'
    },
    'CANBK': {
        'name': 'Canara Bank',
        'search_terms': ['Canara Bank', 'CANBK'],
        'moneycontrol_code': 'CB'
    }
}

os.makedirs('data/news_scraped', exist_ok=True)

# =============================================================================
# SCRAPING FUNCTIONS
# =============================================================================

def scrape_moneycontrol_stock_news(ticker, days_back=7):
    """
    Scrape MoneyControl for stock-specific news

    Note: MoneyControl has anti-scraping measures.
    For production, consider using their API or RSS feeds if available.
    """
    news_items = []
    stock_info = STOCK_MAPPINGS.get(ticker, {})
    stock_name = stock_info.get('name', ticker)

    print(f"\n   📰 MoneyControl - {stock_name}")

    try:
        # MoneyControl stock news URL pattern (example - may need adjustment)
        # Alternative: Use MoneyControl RSS feeds
        search_url = f"https://www.moneycontrol.com/news/tags/{stock_name.replace(' ', '-').lower()}.html"

        response = requests.get(search_url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find news articles (structure may vary - this is a template)
            articles = soup.find_all('div', class_='news_item')  # Adjust class name

            for article in articles[:20]:  # Get latest 20
                try:
                    title_tag = article.find('h2') or article.find('a')
                    title = title_tag.get_text(strip=True) if title_tag else ''

                    link_tag = article.find('a')
                    link = link_tag['href'] if link_tag and link_tag.has_attr('href') else ''

                    date_tag = article.find('span', class_='date')  # Adjust class
                    date_str = date_tag.get_text(strip=True) if date_tag else datetime.now().strftime('%Y-%m-%d')

                    desc_tag = article.find('p')
                    description = desc_tag.get_text(strip=True) if desc_tag else ''

                    if title and len(title) > 10:
                        news_items.append({
                            'date': date_str,
                            'title': title,
                            'description': description,
                            'url': link if link.startswith('http') else f"https://www.moneycontrol.com{link}",
                            'source': 'MoneyControl'
                        })
                except Exception as e:
                    continue

            print(f"      ✓ Found {len(news_items)} articles")
        else:
            print(f"      ⚠ HTTP {response.status_code}")

    except Exception as e:
        print(f"      ✗ Error: {e}")

    return news_items

def scrape_economictimes_stock_news(ticker, days_back=7):
    """Scrape Economic Times for stock news"""
    news_items = []
    stock_info = STOCK_MAPPINGS.get(ticker, {})
    stock_name = stock_info.get('name', ticker)

    print(f"\n   📰 Economic Times - {stock_name}")

    try:
        # ET topic/tag page
        search_term = stock_name.replace(' ', '-').lower()
        search_url = f"https://economictimes.indiatimes.com/topic/{search_term}"

        response = requests.get(search_url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Parse ET news structure
            articles = soup.find_all('div', class_='eachStory')  # Adjust class

            for article in articles[:20]:
                try:
                    title_tag = article.find('h3') or article.find('a')
                    title = title_tag.get_text(strip=True) if title_tag else ''

                    link_tag = article.find('a')
                    link = link_tag['href'] if link_tag and link_tag.has_attr('href') else ''

                    date_tag = article.find('time')
                    date_str = date_tag.get_text(strip=True) if date_tag else datetime.now().strftime('%Y-%m-%d')

                    desc_tag = article.find('p')
                    description = desc_tag.get_text(strip=True) if desc_tag else ''

                    if title and len(title) > 10:
                        news_items.append({
                            'date': date_str,
                            'title': title,
                            'description': description,
                            'url': link if link.startswith('http') else f"https://economictimes.indiatimes.com{link}",
                            'source': 'EconomicTimes'
                        })
                except Exception as e:
                    continue

            print(f"      ✓ Found {len(news_items)} articles")
        else:
            print(f"      ⚠ HTTP {response.status_code}")

    except Exception as e:
        print(f"      ✗ Error: {e}")

    return news_items

def scrape_livemint_stock_news(ticker, days_back=7):
    """Scrape LiveMint for stock news"""
    news_items = []
    stock_info = STOCK_MAPPINGS.get(ticker, {})
    stock_name = stock_info.get('name', ticker)

    print(f"\n   📰 LiveMint - {stock_name}")

    try:
        # LiveMint search/topic page
        search_term = stock_name.replace(' ', '%20')
        search_url = f"https://www.livemint.com/Search/Link/Keyword/{search_term}"

        response = requests.get(search_url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            articles = soup.find_all('div', class_='listView')  # Adjust class

            for article in articles[:20]:
                try:
                    title_tag = article.find('h2') or article.find('a')
                    title = title_tag.get_text(strip=True) if title_tag else ''

                    link_tag = article.find('a')
                    link = link_tag['href'] if link_tag and link_tag.has_attr('href') else ''

                    date_tag = article.find('span', class_='date')
                    date_str = date_tag.get_text(strip=True) if date_tag else datetime.now().strftime('%Y-%m-%d')

                    desc_tag = article.find('p')
                    description = desc_tag.get_text(strip=True) if desc_tag else ''

                    if title and len(title) > 10:
                        news_items.append({
                            'date': date_str,
                            'title': title,
                            'description': description,
                            'url': link if link.startswith('http') else f"https://www.livemint.com{link}",
                            'source': 'LiveMint'
                        })
                except Exception as e:
                    continue

            print(f"      ✓ Found {len(news_items)} articles")
        else:
            print(f"      ⚠ HTTP {response.status_code}")

    except Exception as e:
        print(f"      ✗ Error: {e}")

    return news_items

def get_gnews_fallback(ticker, days_back=7):
    """Use GNews as fallback/supplementary source"""
    from gnews import GNews

    news_items = []
    stock_info = STOCK_MAPPINGS.get(ticker, {})
    stock_name = stock_info.get('name', ticker)

    print(f"\n   📰 GNews (Fallback) - {stock_name}")

    try:
        google_news = GNews(language='en', country='IN', period=f'{days_back}d', max_results=30)

        # Try multiple search terms
        for search_term in stock_info.get('search_terms', [stock_name]):
            articles = google_news.get_news(search_term)

            for article in articles:
                news_items.append({
                    'date': article.get('published date', datetime.now().strftime('%Y-%m-%d')),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': f"GNews-{article.get('publisher', {}).get('title', 'Unknown')}"
                })

            time.sleep(1)  # Rate limiting

        print(f"      ✓ Found {len(news_items)} articles")

    except Exception as e:
        print(f"      ✗ Error: {e}")

    return news_items

# =============================================================================
# AGGREGATE NEWS FOR EACH STOCK
# =============================================================================

def collect_all_news_for_stock(ticker, days_back=7):
    """Collect news from all sources for a single stock"""

    print(f"\n{'='*60}")
    print(f"🔍 Collecting news for {ticker}")
    print(f"{'='*60}")

    all_news = []

    # Try each source
    sources = [
        ('MoneyControl', lambda: scrape_moneycontrol_stock_news(ticker, days_back)),
        ('EconomicTimes', lambda: scrape_economictimes_stock_news(ticker, days_back)),
        ('LiveMint', lambda: scrape_livemint_stock_news(ticker, days_back)),
        ('GNews', lambda: get_gnews_fallback(ticker, days_back))
    ]

    for source_name, scraper_func in sources:
        try:
            news = scraper_func()
            all_news.extend(news)
            time.sleep(2)  # Rate limiting between sources
        except Exception as e:
            print(f"   ⚠ {source_name} failed: {e}")

    # Remove duplicates based on title similarity
    df = pd.DataFrame(all_news)

    if len(df) > 0:
        df = df.drop_duplicates(subset=['title'], keep='first')
        df['ticker'] = ticker
        df['collected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save to CSV
        output_file = f"data/news_scraped/{ticker}_raw_news.csv"
        df.to_csv(output_file, index=False)

        print(f"\n   ✓ Total unique articles: {len(df)}")
        print(f"   ✓ Saved to: {output_file}")

        return df
    else:
        print(f"\n   ⚠ No news found for {ticker}")
        return None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    all_stocks_data = []

    for ticker in STOCK_MAPPINGS.keys():
        try:
            df = collect_all_news_for_stock(ticker, days_back=7)
            if df is not None:
                all_stocks_data.append(df)
        except Exception as e:
            print(f"\n✗ Failed for {ticker}: {e}")

        # Rate limiting between stocks
        time.sleep(3)

    # Combine all data
    if all_stocks_data:
        combined_df = pd.concat(all_stocks_data, ignore_index=True)
        combined_file = 'data/news_scraped/all_stocks_news.csv'
        combined_df.to_csv(combined_file, index=False)

        print("\n\n" + "="*80)
        print("✓ NEWS COLLECTION COMPLETE".center(80))
        print("="*80)
        print(f"\nTotal articles collected: {len(combined_df)}")
        print(f"Combined file: {combined_file}")

        # Summary by source
        print("\n📊 Articles by source:")
        print(combined_df['source'].value_counts())
    else:
        print("\n⚠ No data collected")
