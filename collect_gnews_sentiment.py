from gnews import GNews
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GOOGLE NEWS COLLECTION - INDIAN BANKING STOCKS".center(80))
print("GNews API + VADER Sentiment Analysis".center(80))
print("=" * 80)

# Create directories
os.makedirs('data/news/gnews/private_banks', exist_ok=True)
os.makedirs('data/news/gnews/psu_banks', exist_ok=True)
os.makedirs('data/news/gnews/combined', exist_ok=True)

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize GNews (1 year period, max 100 results per query)
google_news = GNews(language='en', country='IN', period='1y', max_results=100)

# =============================================================================
# BANKING STOCKS CONFIGURATION
# =============================================================================

banking_stocks = {
    'Private Banks': {
        'HDFC Bank': {'keywords': ['HDFC Bank', 'HDFCBANK'], 'ticker': 'HDFCBANK'},
        'ICICI Bank': {'keywords': ['ICICI Bank', 'ICICIBANK'], 'ticker': 'ICICIBANK'},
        'Kotak Mahindra Bank': {'keywords': ['Kotak Bank', 'Kotak Mahindra'], 'ticker': 'KOTAKBANK'},
        'Axis Bank': {'keywords': ['Axis Bank', 'AXISBANK'], 'ticker': 'AXISBANK'}
    },
    'PSU Banks': {
        'State Bank of India': {'keywords': ['SBI', 'State Bank of India'], 'ticker': 'SBIN'},
        'Punjab National Bank': {'keywords': ['PNB', 'Punjab National Bank'], 'ticker': 'PNB'},
        'Bank of Baroda': {'keywords': ['Bank of Baroda', 'BOB Bank'], 'ticker': 'BANKBARODA'},
        'Canara Bank': {'keywords': ['Canara Bank'], 'ticker': 'CANBK'}
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    if not text or pd.isna(text) or str(text).strip() == '':
        return {
            'sentiment_compound': 0.0,
            'sentiment_positive': 0.0,
            'sentiment_neutral': 1.0,
            'sentiment_negative': 0.0,
            'sentiment_label': 'neutral'
        }
    
    scores = sentiment_analyzer.polarity_scores(str(text))
    
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'sentiment_compound': round(scores['compound'], 4),
        'sentiment_positive': round(scores['pos'], 4),
        'sentiment_neutral': round(scores['neu'], 4),
        'sentiment_negative': round(scores['neg'], 4),
        'sentiment_label': label
    }

# =============================================================================
# COLLECT GOOGLE NEWS FOR ALL BANKS
# =============================================================================

print("\n" + "=" * 80)
print("COLLECTING GOOGLE NEWS".center(80))
print("=" * 80)

gnews_summary = []
all_articles_collected = []

for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    save_dir = f'data/news/gnews/{"private_banks" if sector == "Private Banks" else "psu_banks"}'
    
    for bank_name, config in banks.items():
        print(f"\n📰 Collecting Google News for: {bank_name}")
        print("-" * 60)
        
        all_articles = []
        
        # Search for each keyword
        for keyword in config['keywords']:
            print(f"   ⏳ Searching: '{keyword}'")
            
            try:
                news = google_news.get_news(keyword)
                
                for article in news:
                    all_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_date': article.get('published date', ''),
                        'publisher': article.get('publisher', {}).get('title', '') if isinstance(article.get('publisher'), dict) else str(article.get('publisher', '')),
                        'keyword': keyword
                    })
                
                print(f"      ✓ Found: {len(news)} articles")
                time.sleep(2)  # Be respectful
                
            except Exception as e:
                print(f"      ✗ Error: {str(e)[:100]}")
        
        # Remove duplicates based on title
        unique_articles = {article['title']: article for article in all_articles if article['title']}.values()
        unique_articles = list(unique_articles)
        
        if len(unique_articles) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(unique_articles)
            df['bank'] = bank_name
            df['sector'] = sector
            
            # Analyze sentiment on title + description
            print(f"   📊 Analyzing sentiment for {len(df)} articles...")
            df['text_for_sentiment'] = df.apply(
                lambda row: f"{row['title']} {row['description']}", axis=1
            )
            
            sentiment_results = df['text_for_sentiment'].apply(analyze_sentiment)
            sentiment_df = pd.DataFrame(sentiment_results.tolist())
            df = pd.concat([df, sentiment_df], axis=1)
            
            # Add collection metadata
            df['source'] = 'GNews'
            df['collected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate statistics
            positive = len(df[df['sentiment_label'] == 'positive'])
            negative = len(df[df['sentiment_label'] == 'negative'])
            neutral = len(df[df['sentiment_label'] == 'neutral'])
            avg_sentiment = df['sentiment_compound'].mean()
            
            # Save individual bank data
            csv_file = f"{save_dir}/{config['ticker']}_gnews.csv"
            df.to_csv(csv_file, index=False)
            
            # Add to combined list
            all_articles_collected.append(df)
            
            print(f"   ✓ Unique articles: {len(unique_articles)}")
            print(f"   ✓ Positive: {positive} ({positive/len(df)*100:.1f}%)")
            print(f"   ✓ Negative: {negative} ({negative/len(df)*100:.1f}%)")
            print(f"   ✓ Neutral: {neutral} ({neutral/len(df)*100:.1f}%)")
            print(f"   ✓ Avg Sentiment: {avg_sentiment:.4f}")
            print(f"   ✓ Saved: {csv_file}")
            
            gnews_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': len(unique_articles),
                'Positive': positive,
                'Negative': negative,
                'Neutral': neutral,
                'Positive %': f"{(positive/len(df)*100):.1f}",
                'Avg Sentiment': f"{avg_sentiment:.4f}"
            })
        else:
            print(f"   ⚠ No articles found")
            gnews_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': 0,
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0,
                'Positive %': '0',
                'Avg Sentiment': '0'
            })

# =============================================================================
# SAVE GNEWS SUMMARY
# =============================================================================

print("\n\n" + "=" * 80)
print("GNEWS COLLECTION SUMMARY".center(80))
print("=" * 80)

gnews_summary_df = pd.DataFrame(gnews_summary)
gnews_summary_df.to_csv('data/news/gnews/combined/gnews_summary.csv', index=False)
print(f"\n✓ Summary saved: data/news/gnews/combined/gnews_summary.csv")

print("\n" + gnews_summary_df.to_string(index=False))

# Save combined GNews articles
if all_articles_collected:
    combined_gnews = pd.concat(all_articles_collected, ignore_index=True)
    combined_gnews.to_csv('data/news/gnews/combined/all_gnews_articles.csv', index=False)
    print(f"\n✓ Combined GNews articles saved: data/news/gnews/combined/all_gnews_articles.csv")
    print(f"   Total GNews articles: {len(combined_gnews)}")

# =============================================================================
# COMBINE WITH NEWSAPI DATA
# =============================================================================

print("\n\n" + "=" * 80)
print("COMBINING NEWSAPI + GNEWS DATA".center(80))
print("=" * 80)

# Load existing NewsAPI summary
try:
    newsapi_summary = pd.read_csv('data/news/sentiment_analysis/sentiment_summary.csv')
    print("\n✓ Loaded NewsAPI summary")
    print(f"   NewsAPI articles: {newsapi_summary['Total Articles'].sum()}")
except:
    print("\n⚠ Could not load NewsAPI summary")
    newsapi_summary = None

# Combine summaries
if newsapi_summary is not None:
    # Merge on Bank name
    combined_summary = newsapi_summary.merge(
        gnews_summary_df[['Bank', 'Articles', 'Avg Sentiment']],
        on='Bank',
        how='outer',
        suffixes=('_newsapi', '_gnews')
    )
    
    # Calculate totals
    combined_summary['Total_Articles'] = (
        combined_summary['Total Articles'].fillna(0) + 
        combined_summary['Articles'].fillna(0)
    )
    
    combined_summary['Combined_Avg_Sentiment'] = combined_summary.apply(
        lambda row: (
            float(row.get('Average Sentiment', 0)) * float(row.get('Total Articles', 0)) +
            float(row.get('Avg Sentiment', 0)) * float(row.get('Articles', 0))
        ) / max(float(row.get('Total_Articles', 1)), 1) if row.get('Total_Articles', 0) > 0 else 0,
        axis=1
    )
    
    combined_summary.to_csv('data/news/combined/comprehensive_news_summary.csv', index=False)
    print(f"\n✓ Combined summary saved: data/news/combined/comprehensive_news_summary.csv")
    
    # Display combined summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE NEWS SENTIMENT SUMMARY".center(80))
    print("=" * 80)
    
    display_cols = ['Bank', 'Total Articles', 'Articles', 'Total_Articles', 'Average Sentiment', 'Avg Sentiment', 'Combined_Avg_Sentiment']
    available_cols = [col for col in display_cols if col in combined_summary.columns]
    print("\n" + combined_summary[available_cols].to_string(index=False))

# =============================================================================
# SECTOR COMPARISON
# =============================================================================

print("\n\n" + "=" * 80)
print("SECTOR-LEVEL COMPARISON".center(80))
print("=" * 80)

for sector in ['Private Banks', 'PSU Banks']:
    gnews_sector = gnews_summary_df[gnews_summary_df['Sector'] == sector]
    
    total_articles = gnews_sector['Articles'].sum()
    avg_sentiment = gnews_sector['Avg Sentiment'].apply(lambda x: float(x) if x != '0' else 0).mean()
    positive_count = gnews_sector['Positive'].sum()
    
    print(f"\n{sector} (GNews):")
    print(f"   Articles: {total_articles}")
    print(f"   Avg Sentiment: {avg_sentiment:.4f}")
    print(f"   Positive Articles: {positive_count}")

# =============================================================================
# FINAL STATISTICS
# =============================================================================

print("\n\n" + "=" * 80)
print("FINAL STATISTICS".center(80))
print("=" * 80)

total_gnews = gnews_summary_df['Articles'].sum()
print(f"\n📰 GNews Articles Collected: {total_gnews}")
print(f"📰 Banks with GNews Data: {len(gnews_summary_df[gnews_summary_df['Articles'] > 0])}/8")

if newsapi_summary is not None:
    total_newsapi = newsapi_summary['Total Articles'].sum()
    print(f"\n📰 NewsAPI Articles (existing): {total_newsapi}")
    print(f"📊 Combined Total: {total_gnews + total_newsapi}")
    print(f"📈 Coverage Increase: {((total_gnews + total_newsapi) / max(total_newsapi, 1) - 1) * 100:.1f}%")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS".center(80))
print("=" * 80)

if total_gnews > 0:
    # Most positive bank
    banks_with_data = gnews_summary_df[gnews_summary_df['Articles'] > 0]
    if not banks_with_data.empty:
        most_positive = banks_with_data.loc[banks_with_data['Avg Sentiment'].apply(float).idxmax()]
        print(f"\n✨ Most Positive (GNews): {most_positive['Bank']}")
        print(f"   Sentiment Score: {most_positive['Avg Sentiment']}")
        print(f"   {most_positive['Positive']} positive / {most_positive['Articles']} total")
        
        # Most covered
        most_covered = banks_with_data.loc[banks_with_data['Articles'].idxmax()]
        print(f"\n📊 Most News Coverage: {most_covered['Bank']}")
        print(f"   Articles: {most_covered['Articles']}")

print("\n" + "=" * 80)
print("GOOGLE NEWS COLLECTION COMPLETE!".center(80))
print("=" * 80)

print("\nFiles Created:")
print("  📰 Individual GNews: data/news/gnews/{sector}/{TICKER}_gnews.csv")
print("  📊 GNews Summary: data/news/gnews/combined/gnews_summary.csv")
print("  📊 Combined Summary: data/news/combined/comprehensive_news_summary.csv")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Analyze combined news sentiment trends")
print("  2. Correlate with stock price movements")
print("  3. Use in predictive models")
print("=" * 80)
