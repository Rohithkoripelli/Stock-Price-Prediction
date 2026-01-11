import snscrape.modules.twitter as sntwitter
from gnews import GNews
import pandas as pd
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE SOCIAL MEDIA & NEWS DATA COLLECTION".center(80))
print("Twitter (SNScrape) + Google News (GNews) + Sentiment Analysis".center(80))
print("=" * 80)

# Create directories
os.makedirs('data/social_media/twitter/private_banks', exist_ok=True)
os.makedirs('data/social_media/twitter/psu_banks', exist_ok=True)
os.makedirs('data/social_media/google_news/private_banks', exist_ok=True)
os.makedirs('data/social_media/google_news/psu_banks', exist_ok=True)
os.makedirs('data/social_media/combined', exist_ok=True)

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# =============================================================================
# BANKING STOCKS CONFIGURATION
# =============================================================================

banking_stocks = {
    'Private Banks': {
        'HDFC Bank': {
            'twitter_keywords': ['#HDFCBank', 'HDFCBANK', '$HDFCBANK'],
            'news_keywords': ['HDFC Bank', 'HDFCBANK'],
            'ticker': 'HDFCBANK'
        },
        'ICICI Bank': {
            'twitter_keywords': ['#ICICIBank', 'ICICIBANK', '$ICICIBANK'],
            'news_keywords': ['ICICI Bank', 'ICICIBANK'],
            'ticker': 'ICICIBANK'
        },
        'Kotak Mahindra Bank': {
            'twitter_keywords': ['#KotakBank', 'KOTAKBANK', '$KOTAKBANK'],
            'news_keywords': ['Kotak Bank', 'Kotak Mahindra'],
            'ticker': 'KOTAKBANK'
        },
        'Axis Bank': {
            'twitter_keywords': ['#AxisBank', 'AXISBANK', '$AXISBANK'],
            'news_keywords': ['Axis Bank', 'AXISBANK'],
            'ticker': 'AXISBANK'
        }
    },
    'PSU Banks': {
        'State Bank of India': {
            'twitter_keywords': ['#SBI', 'SBIN', '$SBIN'],
            'news_keywords': ['SBI', 'State Bank of India'],
            'ticker': 'SBIN'
        },
        'Punjab National Bank': {
            'twitter_keywords': ['#PNB', '$PNB', 'PNB Bank'],
            'news_keywords': ['PNB', 'Punjab National Bank'],
            'ticker': 'PNB'
        },
        'Bank of Baroda': {
            'twitter_keywords': ['#BankOfBaroda', 'BANKBARODA', '$BANKBARODA'],
            'news_keywords': ['Bank of Baroda', 'BOB Bank'],
            'ticker': 'BANKBARODA'
        },
        'Canara Bank': {
            'twitter_keywords': ['#CanaraBank', 'CANBK', '$CANBK'],
            'news_keywords': ['Canara Bank', 'CANBK'],
            'ticker': 'CANBK'
        }
    }
}

# Date configuration
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)  # Last 3 months

print(f"\nDate Range: {START_DATE.date()} to {END_DATE.date()}")

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

def scrape_tweets(query, since_date, until_date, max_tweets=200):
    """Scrape tweets for a given query"""
    tweets_list = []
    since_str = since_date.strftime('%Y-%m-%d')
    until_str = until_date.strftime('%Y-%m-%d')
    search_query = f"{query} lang:en since:{since_str} until:{until_str}"
    
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
            if i >= max_tweets:
                break
            
            tweets_list.append({
                'date': tweet.date,
                'id': tweet.id,
                'content': tweet.rawContent,
                'user': tweet.user.username,
                'reply_count': tweet.replyCount,
                'retweet_count': tweet.retweetCount,
                'like_count': tweet.likeCount,
                'engagement': tweet.likeCount + tweet.retweetCount
            })
        
        return tweets_list
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return []

# =============================================================================
# 1. TWITTER DATA COLLECTION
# =============================================================================

print("\n\n" + "=" * 80)
print("1. COLLECTING TWITTER DATA".center(80))
print("=" * 80)

twitter_summary = []

for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    save_dir = f'data/social_media/twitter/{"private_banks" if sector == "Private Banks" else "psu_banks"}'
    
    for bank_name, config in banks.items():
        print(f"\n📱 Collecting tweets for: {bank_name}")
        print("-" * 60)
        
        all_tweets = []
        
        # Search for each keyword (limited to avoid redundancy)
        for keyword in config['twitter_keywords'][:2]:
            print(f"   ⏳ Searching: {keyword}")
            
            tweets = scrape_tweets(
                query=keyword,
                since_date=START_DATE,
                until_date=END_DATE,
                max_tweets=150
            )
            
            all_tweets.extend(tweets)
            print(f"      ✓ Found: {len(tweets)} tweets")
            time.sleep(2)
        
        # Remove duplicates
        unique_tweets = {tweet['id']: tweet for tweet in all_tweets}.values()
        unique_tweets = list(unique_tweets)
        
        if len(unique_tweets) > 0:
            # Convert to DataFrame
            tweets_df = pd.DataFrame(unique_tweets)
            tweets_df['bank'] = bank_name
            tweets_df['sector'] = sector
            
            # Analyze sentiment
            print(f"   📊 Analyzing sentiment for {len(tweets_df)} tweets...")
            sentiment_results = tweets_df['content'].apply(analyze_sentiment)
            sentiment_df = pd.DataFrame(sentiment_results.tolist())
            tweets_df = pd.concat([tweets_df, sentiment_df], axis=1)
            
            # Sort by date
            tweets_df['date'] = pd.to_datetime(tweets_df['date'])
            tweets_df = tweets_df.sort_values('date', ascending=False)
            
            # Calculate statistics
            positive = len(tweets_df[tweets_df['sentiment_label'] == 'positive'])
            negative = len(tweets_df[tweets_df['sentiment_label'] == 'negative'])
            neutral = len(tweets_df[tweets_df['sentiment_label'] == 'neutral'])
            avg_sentiment = tweets_df['sentiment_compound'].mean()
            total_engagement = tweets_df['engagement'].sum()
            
            # Save to CSV
            csv_file = f"{save_dir}/{config['ticker']}_tweets.csv"
            tweets_df.to_csv(csv_file, index=False)
            
            print(f"   ✓ Unique tweets: {len(unique_tweets)}")
            print(f"   ✓ Positive: {positive} ({positive/len(tweets_df)*100:.1f}%)")
            print(f"   ✓ Negative: {negative} ({negative/len(tweets_df)*100:.1f}%)")
            print(f"   ✓ Avg Sentiment: {avg_sentiment:.4f}")
            print(f"   ✓ Total Engagement: {total_engagement:,}")
            print(f"   ✓ Saved: {csv_file}")
            
            twitter_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Tweets': len(unique_tweets),
                'Positive': positive,
                'Negative': negative,
                'Neutral': neutral,
                'Avg Sentiment': f"{avg_sentiment:.4f}",
                'Total Engagement': total_engagement
            })
        else:
            print(f"   ⚠ No tweets found")
            twitter_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Tweets': 0,
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0,
                'Avg Sentiment': '0',
                'Total Engagement': 0
            })

# Save Twitter summary
twitter_summary_df = pd.DataFrame(twitter_summary)
twitter_summary_df.to_csv('data/social_media/combined/twitter_summary.csv', index=False)
print(f"\n✓ Twitter summary saved: data/social_media/combined/twitter_summary.csv")

# =============================================================================
# 2. GOOGLE NEWS COLLECTION
# =============================================================================

print("\n\n" + "=" * 80)
print("2. COLLECTING GOOGLE NEWS DATA".center(80))
print("=" * 80)

# Initialize GNews
google_news = GNews(language='en', country='IN', period='3m', max_results=50)

news_summary = []

for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    save_dir = f'data/social_media/google_news/{"private_banks" if sector == "Private Banks" else "psu_banks"}'
    
    for bank_name, config in banks.items():
        print(f"\n📰 Collecting Google News for: {bank_name}")
        print("-" * 60)
        
        all_articles = []
        
        # Search for each keyword
        for keyword in config['news_keywords']:
            print(f"   ⏳ Searching: {keyword}")
            
            try:
                news = google_news.get_news(keyword)
                
                for article in news:
                    all_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_date': article.get('published date', ''),
                        'publisher': article.get('publisher', {}).get('title', '') if isinstance(article.get('publisher'), dict) else str(article.get('publisher', ''))
                    })
                
                print(f"      ✓ Found: {len(news)} articles")
                time.sleep(2)
                
            except Exception as e:
                print(f"      ✗ Error: {e}")
        
        # Remove duplicates based on title
        unique_articles = {article['title']: article for article in all_articles if article['title']}.values()
        unique_articles = list(unique_articles)
        
        if len(unique_articles) > 0:
            # Convert to DataFrame
            news_df = pd.DataFrame(unique_articles)
            news_df['bank'] = bank_name
            news_df['sector'] = sector
            
            # Analyze sentiment on title + description
            print(f"   📊 Analyzing sentiment for {len(news_df)} articles...")
            news_df['text_for_sentiment'] = news_df.apply(
                lambda row: f"{row['title']} {row['description']}", axis=1
            )
            
            sentiment_results = news_df['text_for_sentiment'].apply(analyze_sentiment)
            sentiment_df = pd.DataFrame(sentiment_results.tolist())
            news_df = pd.concat([news_df, sentiment_df], axis=1)
            
            # Calculate statistics
            positive = len(news_df[news_df['sentiment_label'] == 'positive'])
            negative = len(news_df[news_df['sentiment_label'] == 'negative'])
            neutral = len(news_df[news_df['sentiment_label'] == 'neutral'])
            avg_sentiment = news_df['sentiment_compound'].mean()
            
            # Save to CSV
            csv_file = f"{save_dir}/{config['ticker']}_gnews.csv"
            news_df.to_csv(csv_file, index=False)
            
            print(f"   ✓ Unique articles: {len(unique_articles)}")
            print(f"   ✓ Positive: {positive} ({positive/len(news_df)*100:.1f}%)")
            print(f"   ✓ Negative: {negative} ({negative/len(news_df)*100:.1f}%)")
            print(f"   ✓ Avg Sentiment: {avg_sentiment:.4f}")
            print(f"   ✓ Saved: {csv_file}")
            
            news_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': len(unique_articles),
                'Positive': positive,
                'Negative': negative,
                'Neutral': neutral,
                'Avg Sentiment': f"{avg_sentiment:.4f}"
            })
        else:
            print(f"   ⚠ No articles found")
            news_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Articles': 0,
                'Positive': 0,
                'Negative': 0,
                'Neutral': 0,
                'Avg Sentiment': '0'
            })

# Save Google News summary
news_summary_df = pd.DataFrame(news_summary)
news_summary_df.to_csv('data/social_media/combined/google_news_summary.csv', index=False)
print(f"\n✓ Google News summary saved: data/social_media/combined/google_news_summary.csv")

# =============================================================================
# 3. COMBINED ANALYSIS & INSIGHTS
# =============================================================================

print("\n\n" + "=" * 80)
print("COMBINED SOCIAL MEDIA & NEWS INSIGHTS".center(80))
print("=" * 80)

# Merge summaries
combined_summary = twitter_summary_df.merge(
    news_summary_df[['Bank', 'Articles', 'Avg Sentiment']],
    on='Bank',
    how='outer',
    suffixes=('_twitter', '_news')
)

print("\n" + "=" * 80)
print("TWITTER SENTIMENT SUMMARY".center(80))
print("=" * 80)
print(twitter_summary_df.to_string(index=False))

print("\n\n" + "=" * 80)
print("GOOGLE NEWS SENTIMENT SUMMARY".center(80))
print("=" * 80)
print(news_summary_df.to_string(index=False))

# Save combined summary
combined_summary.to_csv('data/social_media/combined/comprehensive_social_summary.csv', index=False)
print(f"\n✓ Combined summary saved: data/social_media/combined/comprehensive_social_summary.csv")

# =============================================================================
# FINAL STATISTICS
# =============================================================================

print("\n\n" + "=" * 80)
print("OVERALL STATISTICS".center(80))
print("=" * 80)

total_tweets = twitter_summary_df['Tweets'].sum()
total_articles = news_summary_df['Articles'].sum()
total_twitter_engagement = twitter_summary_df['Total Engagement'].sum()

print(f"\n📱 Twitter:")
print(f"   Total Tweets: {total_tweets:,}")
print(f"   Total Engagement: {total_twitter_engagement:,}")
print(f"   Average per Bank: {total_tweets/8:.1f} tweets")

print(f"\n📰 Google News:")
print(f"   Total Articles: {total_articles:,}")
print(f"   Average per Bank: {total_articles/8:.1f} articles")

print(f"\n📊 Combined Data Points: {total_tweets + total_articles:,}")

# Sector comparisons
print("\n" + "=" * 80)
print("SECTOR-LEVEL COMPARISON".center(80))
print("=" * 80)

for sector in ['Private Banks', 'PSU Banks']:
    twitter_sector = twitter_summary_df[twitter_summary_df['Sector'] == sector]
    news_sector = news_summary_df[news_summary_df['Sector'] == sector]
    
    twitter_sent = twitter_sector['Avg Sentiment'].apply(lambda x: float(x) if x != '0' else 0).mean()
    news_sent = news_sector['Avg Sentiment'].apply(lambda x: float(x) if x != '0' else 0).mean()
    
    print(f"\n{sector}:")
    print(f"   Twitter Sentiment: {twitter_sent:.4f}")
    print(f"   News Sentiment: {news_sent:.4f}")
    print(f"   Tweets: {twitter_sector['Tweets'].sum():,}")
    print(f"   Articles: {news_sector['Articles'].sum():,}")

# Top insights
print("\n" + "=" * 80)
print("KEY INSIGHTS".center(80))
print("=" * 80)

# Most positive on Twitter
twitter_positive = twitter_summary_df[twitter_summary_df['Tweets'] > 0]
if not twitter_positive.empty:
    most_positive_twitter = twitter_positive.loc[twitter_positive['Avg Sentiment'].apply(float).idxmax()]
    print(f"\n📱 Most Positive on Twitter: {most_positive_twitter['Bank']}")
    print(f"   Sentiment: {most_positive_twitter['Avg Sentiment']}")
    print(f"   {most_positive_twitter['Positive']} positive / {most_positive_twitter['Tweets']} total tweets")

# Most positive in News
news_positive = news_summary_df[news_summary_df['Articles'] > 0]
if not news_positive.empty:
    most_positive_news = news_positive.loc[news_positive['Avg Sentiment'].apply(float).idxmax()]
    print(f"\n📰 Most Positive in News: {most_positive_news['Bank']}")
    print(f"   Sentiment: {most_positive_news['Avg Sentiment']}")
    print(f"   {most_positive_news['Positive']} positive / {most_positive_news['Articles']} total articles")

# Most engaging on Twitter
if not twitter_positive.empty:
    most_engaging = twitter_positive.loc[twitter_positive['Total Engagement'].idxmax()]
    print(f"\n💬 Most Twitter Engagement: {most_engaging['Bank']}")
    print(f"   Total Engagement: {most_engaging['Total Engagement']:,}")

print("\n" + "=" * 80)
print("DATA COLLECTION COMPLETE!".center(80))
print("=" * 80)

print("\nFiles Created:")
print("  📱 Twitter data: data/social_media/twitter/{sector}/{TICKER}_tweets.csv")
print("  📰 Google News: data/social_media/google_news/{sector}/{TICKER}_gnews.csv")
print("  📊 Summaries: data/social_media/combined/")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Correlate social sentiment with stock price movements")
print("  2. Identify sentiment-driven trading opportunities")
print("  3. Combine with news sentiment for comprehensive analysis")
print("  4. Track sentiment trends over time")
print("=" * 80)
