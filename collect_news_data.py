import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os

# =============================================================================
# NEWS DATA COLLECTION FOR INDIAN BANKING STOCKS
# Using NewsAPI.org - Optimized for Free Plan
# =============================================================================

# Create directories
os.makedirs('data/news/private_banks', exist_ok=True)
os.makedirs('data/news/psu_banks', exist_ok=True)
os.makedirs('data/news/combined', exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

# YOUR API KEY HERE
NEWS_API_KEY = '85368198a0b44776a0562a593c413723'  # Replace with your actual API key

# Banking stocks
banking_stocks = {
    'Private Banks': {
        'HDFC Bank': ['HDFC Bank', 'HDFCBANK', 'HDFC'],
        'ICICI Bank': ['ICICI Bank', 'ICICIBANK', 'ICICI'],
        'Kotak Mahindra Bank': ['Kotak Bank', 'Kotak Mahindra', 'KOTAKBANK'],
        'Axis Bank': ['Axis Bank', 'AXISBANK', 'Axis']
    },
    'PSU Banks': {
        'State Bank of India': ['SBI', 'State Bank', 'State Bank of India'],
        'Punjab National Bank': ['PNB', 'Punjab National Bank'],
        'Bank of Baroda': ['Bank of Baroda', 'BoB Bank', 'BANKBARODA'],
        'Canara Bank': ['Canara Bank', 'CANBK']
    }
}

# NewsAPI parameters
NEWS_API_BASE_URL = 'https://newsapi.org/v2/everything'

# Free plan limitation: Only last 30 days
# We'll collect in chunks to maximize coverage
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=29)  # 29 days to stay within limit

print("=" * 80)
print("NEWS DATA COLLECTION FOR INDIAN BANKING STOCKS".center(80))
print(f"Date Range: {START_DATE.date()} to {END_DATE.date()}".center(80))
print("Using NewsAPI.org (Free Plan)".center(80))
print("=" * 80)

# =============================================================================
# OPTIMIZED SEARCH STRATEGY FOR FREE PLAN
# =============================================================================

print("\n📋 OPTIMIZATION STRATEGY:")
print("-" * 80)
print("Free Plan Limits:")
print("  • 100 requests/day")
print("  • Last 30 days of news only")
print("  • Max 100 articles per request")
print("\nStrategy:")
print("  • Use broad sector-level queries instead of individual banks")
print("  • Filter results locally by bank names")
print("  • Minimize API calls by grouping similar queries")
print("  • Save raw responses for reprocessing without additional API calls")

# =============================================================================
# FUNCTION: SEARCH NEWS
# =============================================================================

def search_news(query, from_date, to_date, api_key, page=1, language='en'):
    """
    Search news using NewsAPI
    """
    params = {
        'q': query,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'language': language,
        'sortBy': 'publishedAt',
        'pageSize': 100,  # Max allowed
        'page': page,
        'apiKey': api_key
    }
    
    try:
        response = requests.get(NEWS_API_BASE_URL, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data
        elif response.status_code == 426:
            print(f"   ⚠ Rate limit reached or upgrade required")
            return None
        elif response.status_code == 429:
            print(f"   ⚠ Too many requests. Please wait.")
            return None
        else:
            print(f"   ✗ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"   ✗ Exception: {e}")
        return None

# =============================================================================
# FUNCTION: SAVE RAW NEWS DATA
# =============================================================================

def save_raw_news(data, filename):
    """
    Save raw JSON response for later reprocessing
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =============================================================================
# FUNCTION: FILTER ARTICLES BY BANK
# =============================================================================

def filter_articles_by_bank(articles, bank_keywords):
    """
    Filter articles that mention specific bank
    """
    filtered = []
    
    for article in articles:
        title = article.get('title', '').lower()
        description = article.get('description', '').lower() if article.get('description') else ''
        content = article.get('content', '').lower() if article.get('content') else ''
        
        # Check if any bank keyword appears
        text = f"{title} {description} {content}"
        
        for keyword in bank_keywords:
            if keyword.lower() in text:
                filtered.append(article)
                break  # Avoid duplicates
    
    return filtered

# =============================================================================
# STRATEGY: OPTIMIZED QUERIES FOR FREE PLAN
# =============================================================================

# Instead of 8 separate queries (one per bank), use 2-3 broad queries
# Then filter locally

optimized_queries = [
    {
        'query': 'Indian banks OR banking sector India OR private banks India',
        'category': 'Indian Banking General',
        'filename': 'indian_banking_general.json'
    },
    {
        'query': 'HDFC Bank OR ICICI Bank OR Kotak Bank OR Axis Bank',
        'category': 'Private Banks',
        'filename': 'private_banks.json'
    },
    {
        'query': 'SBI OR State Bank India OR PNB OR Bank of Baroda OR Canara Bank',
        'category': 'PSU Banks',
        'filename': 'psu_banks.json'
    }
]

print("\n\n" + "=" * 80)
print("EXECUTING OPTIMIZED NEWS QUERIES".center(80))
print("=" * 80)

all_raw_responses = {}
total_api_calls = 0

for query_config in optimized_queries:
    query = query_config['query']
    category = query_config['category']
    filename = query_config['filename']
    
    print(f"\n📰 Query: {category}")
    print(f"   Search: {query}")
    print("-" * 60)
    
    # Make API call
    print(f"   ⏳ Fetching news...")
    news_data = search_news(query, START_DATE, END_DATE, NEWS_API_KEY)
    total_api_calls += 1
    
    if news_data and news_data.get('status') == 'ok':
        total_results = news_data.get('totalResults', 0)
        articles = news_data.get('articles', [])
        
        print(f"   ✓ Found: {total_results} total results")
        print(f"   ✓ Retrieved: {len(articles)} articles")
        
        # Save raw response
        raw_file = f"data/news/combined/{filename}"
        save_raw_news(news_data, raw_file)
        print(f"   ✓ Saved raw data: {raw_file}")
        
        all_raw_responses[category] = news_data
        
    else:
        print(f"   ✗ Failed to fetch news")
    
    # Respect rate limits
    time.sleep(2)

print(f"\n\n{'=' * 80}")
print(f"API Calls Made: {total_api_calls} / 100 daily limit".center(80))
print("=" * 80)

# =============================================================================
# PROCESS AND FILTER NEWS FOR EACH BANK
# =============================================================================

print("\n\n" + "=" * 80)
print("FILTERING NEWS FOR INDIVIDUAL BANKS".center(80))
print("=" * 80)

bank_news_summary = []

for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    save_dir = 'data/news/private_banks' if sector == 'Private Banks' else 'data/news/psu_banks'
    
    for bank_name, keywords in banks.items():
        print(f"\n📊 Processing: {bank_name}")
        print(f"   Keywords: {', '.join(keywords)}")
        print("-" * 60)
        
        bank_articles = []
        
        # Filter from all raw responses
        for category, news_data in all_raw_responses.items():
            articles = news_data.get('articles', [])
            filtered = filter_articles_by_bank(articles, keywords)
            bank_articles.extend(filtered)
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in bank_articles:
            url = article.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        print(f"   ✓ Found {len(unique_articles)} unique articles")
        
        if len(unique_articles) > 0:
            # Convert to DataFrame
            articles_df = pd.DataFrame([{
                'bank': bank_name,
                'title': article.get('title'),
                'description': article.get('description'),
                'content': article.get('content'),
                'url': article.get('url'),
                'source': article.get('source', {}).get('name'),
                'author': article.get('author'),
                'publishedAt': article.get('publishedAt'),
                'urlToImage': article.get('urlToImage')
            } for article in unique_articles])
            
            # Convert publishedAt to datetime
            articles_df['publishedAt'] = pd.to_datetime(articles_df['publishedAt'])
            
            # Sort by date
            articles_df = articles_df.sort_values('publishedAt', ascending=False)
            
            # Save to CSV
            bank_clean = bank_name.replace(' ', '_').replace('&', 'and')
            csv_file = f"{save_dir}/{bank_clean}_news.csv"
            articles_df.to_csv(csv_file, index=False)
            print(f"   ✓ Saved to: {csv_file}")
            
            # Statistics
            date_range = f"{articles_df['publishedAt'].min().date()} to {articles_df['publishedAt'].max().date()}"
            sources = articles_df['source'].nunique()
            
            print(f"   ✓ Date Range: {date_range}")
            print(f"   ✓ Unique Sources: {sources}")
            
            # Top sources
            top_sources = articles_df['source'].value_counts().head(3)
            print(f"   ✓ Top Sources: {', '.join([f'{s} ({c})' for s, c in top_sources.items()])}")
            
            bank_news_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': len(unique_articles),
                'Date Range': date_range,
                'Sources': sources,
                'File': csv_file
            })
        else:
            print(f"   ⚠ No articles found")
            bank_news_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': 0,
                'Date Range': 'N/A',
                'Sources': 0,
                'File': 'N/A'
            })

# =============================================================================
# GENERATE SUMMARY REPORT
# =============================================================================

print("\n\n" + "=" * 80)
print("GENERATING NEWS COLLECTION SUMMARY".center(80))
print("=" * 80)

summary_df = pd.DataFrame(bank_news_summary)

# Save summary
summary_file = 'data/news/combined/news_collection_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved to: {summary_file}")

# Display summary
print("\n" + "=" * 80)
print("NEWS COLLECTION SUMMARY".center(80))
print("=" * 80)
print(summary_df.to_string(index=False))

# Overall statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS".center(80))
print("=" * 80)

total_articles = summary_df['Articles'].sum()
avg_articles = summary_df['Articles'].mean()

print(f"\nTotal Articles Collected: {total_articles}")
print(f"Average Articles per Bank: {avg_articles:.1f}")
print(f"API Calls Used: {total_api_calls} / 100")

private_articles = summary_df[summary_df['Sector'] == 'Private Banks']['Articles'].sum()
psu_articles = summary_df[summary_df['Sector'] == 'PSU Banks']['Articles'].sum()

print(f"\nPrivate Banks: {private_articles} articles")
print(f"PSU Banks: {psu_articles} articles")

# =============================================================================
# SAVE COMPLETE COMBINED DATASET
# =============================================================================

print("\n\n" + "=" * 80)
print("CREATING COMBINED DATASET".center(80))
print("=" * 80)

combined_articles = []

for sector, banks in banking_stocks.items():
    save_dir = 'data/news/private_banks' if sector == 'Private Banks' else 'data/news/psu_banks'
    
    for bank_name in banks.keys():
        bank_clean = bank_name.replace(' ', '_').replace('&', 'and')
        csv_file = f"{save_dir}/{bank_clean}_news.csv"
        
        try:
            df = pd.read_csv(csv_file)
            combined_articles.append(df)
        except:
            pass

if combined_articles:
    combined_df = pd.concat(combined_articles, ignore_index=True)
    combined_df = combined_df.sort_values('publishedAt', ascending=False)
    
    combined_file = 'data/news/combined/all_banking_news.csv'
    combined_df.to_csv(combined_file, index=False)
    
    print(f"✓ Combined dataset saved: {combined_file}")
    print(f"✓ Total records: {len(combined_df)}")

# =============================================================================
# RECOMMENDATIONS FOR FREE PLAN
# =============================================================================

print("\n\n" + "=" * 80)
print("RECOMMENDATIONS & NEXT STEPS".center(80))
print("=" * 80)

print("\n📌 For Free Plan Users:")
print("-" * 60)
print("✓ Run this script daily to collect fresh news (stays within 100 req/day limit)")
print("✓ Build historical dataset by running daily over 3-4 weeks")
print("✓ Alternative: Use web scraping for older news (Google News, Economic Times)")
print("✓ Alternative: Use RSS feeds (free, no limits)")

print("\n📌 Coverage:")
print("-" * 60)
print(f"✓ Current coverage: Last {(END_DATE - START_DATE).days} days")
print("✓ For 3-5 years historical: Consider web scraping or paid plan")

print("\n📌 Next Steps:")
print("-" * 60)
print("1. Review news data quality and coverage")
print("2. Implement sentiment analysis on collected news")
print("3. Set up daily cron job to collect fresh news")
print("4. Consider supplementing with RSS feeds or web scraping")

print("\n" + "=" * 80)
print("NEWS COLLECTION COMPLETE!".center(80))
print("=" * 80)