import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import os
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SOCIAL MEDIA SENTIMENT COLLECTION".center(80))
print("StockTwits + Google Trends Analysis".center(80))
print("=" * 80)

# Create output directories
os.makedirs('data/social_media/stocktwits', exist_ok=True)
os.makedirs('data/social_media/google_trends', exist_ok=True)
os.makedirs('data/social_media/combined', exist_ok=True)

# =============================================================================
# BANKING STOCKS CONFIGURATION
# =============================================================================

banking_stocks = {
    'Private Banks': {
        'HDFC Bank': {'ticker': 'HDFCBANK', 'symbol': 'HDFCBANK.NS', 'keywords': ['HDFC Bank', 'HDFC']},
        'ICICI Bank': {'ticker': 'ICICIBANK', 'symbol': 'ICICIBANK.NS', 'keywords': ['ICICI Bank', 'ICICI']},
        'Kotak Mahindra Bank': {'ticker': 'KOTAKBANK', 'symbol': 'KOTAKBANK.NS', 'keywords': ['Kotak Bank', 'Kotak Mahindra']},
        'Axis Bank': {'ticker': 'AXISBANK', 'symbol': 'AXISBANK.NS', 'keywords': ['Axis Bank']}
    },
    'PSU Banks': {
        'State Bank of India': {'ticker': 'SBIN', 'symbol': 'SBIN.NS', 'keywords': ['SBI', 'State Bank of India']},
        'Punjab National Bank': {'ticker': 'PNB', 'symbol': 'PNB.NS', 'keywords': ['PNB', 'Punjab National Bank']},
        'Bank of Baroda': {'ticker': 'BANKBARODA', 'symbol': 'BANKBARODA.NS', 'keywords': ['Bank of Baroda', 'BOB']},
        'Canara Bank': {'ticker': 'CANBK', 'symbol': 'CANBK.NS', 'keywords': ['Canara Bank']}
    }
}

# =============================================================================
# 1. STOCKTWITS DATA COLLECTION
# =============================================================================

print("\n" + "=" * 80)
print("1. COLLECTING STOCKTWITS SENTIMENT".center(80))
print("=" * 80)

def get_stocktwits_data(symbol, max_messages=30):
    """
    Collect StockTwits messages for a given symbol
    StockTwits API is public and doesn't require authentication for basic access
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'messages' in data and len(data['messages']) > 0:
                messages = []
                
                for msg in data['messages'][:max_messages]:
                    message_data = {
                        'id': msg.get('id'),
                        'created_at': msg.get('created_at'),
                        'body': msg.get('body', ''),
                        'user': msg.get('user', {}).get('username', 'Unknown'),
                        'sentiment': msg.get('entities', {}).get('sentiment', {}).get('basic', 'None')
                    }
                    messages.append(message_data)
                
                return messages
            else:
                return []
        else:
            print(f"   ⚠ API returned status {response.status_code}")
            return []
    
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return []

stocktwits_data = []
stocktwits_summary = []

for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    for bank_name, info in banks.items():
        print(f"\n📱 Collecting StockTwits data for: {bank_name}")
        print("-" * 60)
        
        # Try different symbol formats
        symbols_to_try = [
            info['ticker'],  # NSE ticker
            info['symbol'],  # With .NS
            info['ticker'] + '.NS'  # Alternative format
        ]
        
        messages = []
        for symbol in symbols_to_try:
            print(f"   Trying symbol: {symbol}")
            messages = get_stocktwits_data(symbol)
            if messages:
                print(f"   ✓ Found {len(messages)} messages")
                break
            time.sleep(1)  # Be respectful with API calls
        
        if messages:
            # Calculate sentiment distribution
            bullish = sum(1 for m in messages if m['sentiment'] == 'Bullish')
            bearish = sum(1 for m in messages if m['sentiment'] == 'Bearish')
            neutral = len(messages) - bullish - bearish
            
            # Create DataFrame
            df = pd.DataFrame(messages)
            df['bank'] = bank_name
            df['sector'] = sector
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Save individual bank data
            output_dir = 'data/social_media/stocktwits/private_banks' if sector == 'Private Banks' else 'data/social_media/stocktwits/psu_banks'
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/{info['ticker']}_stocktwits.csv"
            df.to_csv(output_file, index=False)
            
            stocktwits_data.append(df)
            
            # Summary
            stocktwits_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Total Messages': len(messages),
                'Bullish': bullish,
                'Bearish': bearish,
                'Neutral': neutral,
                'Bullish %': f"{(bullish/len(messages)*100):.1f}" if messages else "0",
                'Sentiment Score': f"{((bullish - bearish) / len(messages)):.2f}" if messages else "0"
            })
            
            print(f"   ✓ Bullish: {bullish}, Bearish: {bearish}, Neutral: {neutral}")
            print(f"   ✓ Saved: {output_file}")
        else:
            print(f"   ⚠ No StockTwits messages found")
            stocktwits_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Total Messages': 0,
                'Bullish': 0,
                'Bearish': 0,
                'Neutral': 0,
                'Bullish %': "0",
                'Sentiment Score': "0"
            })
        
        time.sleep(2)  # Rate limiting

# Save StockTwits summary
if stocktwits_data:
    combined_df = pd.concat(stocktwits_data, ignore_index=True)
    combined_df.to_csv('data/social_media/stocktwits/all_stocktwits_messages.csv', index=False)
    print(f"\n✓ Saved combined data: data/social_media/stocktwits/all_stocktwits_messages.csv")

stocktwits_summary_df = pd.DataFrame(stocktwits_summary)
stocktwits_summary_df.to_csv('data/social_media/combined/stocktwits_summary.csv', index=False)
print(f"✓ Saved summary: data/social_media/combined/stocktwits_summary.csv")

# =============================================================================
# 2. GOOGLE TRENDS DATA COLLECTION
# =============================================================================

print("\n\n" + "=" * 80)
print("2. COLLECTING GOOGLE TRENDS DATA".center(80))
print("=" * 80)

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

trends_data = []
trends_summary = []

# Collect data for each bank
for sector, banks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    for bank_name, info in banks.items():
        print(f"\n📈 Collecting Google Trends for: {bank_name}")
        print("-" * 60)
        
        try:
            # Use primary keyword
            keyword = info['keywords'][0]
            
            # Build payload for last 90 days
            pytrends.build_payload(
                [keyword],
                cat=0,
                timeframe='today 3-m',  # Last 3 months
                geo='IN',  # India
                gprop=''
            )
            
            # Get interest over time
            interest_df = pytrends.interest_over_time()
            
            if not interest_df.empty and keyword in interest_df.columns:
                # Remove 'isPartial' column if it exists
                if 'isPartial' in interest_df.columns:
                    interest_df = interest_df.drop(columns=['isPartial'])
                
                # Add metadata
                interest_df['bank'] = bank_name
                interest_df['sector'] = sector
                interest_df['keyword'] = keyword
                
                # Calculate statistics
                avg_interest = interest_df[keyword].mean()
                max_interest = interest_df[keyword].max()
                trend_direction = "Increasing" if interest_df[keyword].iloc[-7:].mean() > interest_df[keyword].iloc[-30:-7].mean() else "Decreasing"
                
                # Save individual bank data
                output_dir = 'data/social_media/google_trends/private_banks' if sector == 'Private Banks' else 'data/social_media/google_trends/psu_banks'
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/{info['ticker']}_trends.csv"
                interest_df.to_csv(output_file)
                
                trends_data.append(interest_df)
                
                trends_summary.append({
                    'Sector': sector,
                    'Bank': bank_name,
                    'Keyword': keyword,
                    'Avg Interest': f"{avg_interest:.1f}",
                    'Max Interest': int(max_interest),
                    'Current Trend': trend_direction,
                    'Data Points': len(interest_df)
                })
                
                print(f"   ✓ Avg Interest: {avg_interest:.1f}, Max: {max_interest}, Trend: {trend_direction}")
                print(f"   ✓ Saved: {output_file}")
            else:
                print(f"   ⚠ No trend data available")
                trends_summary.append({
                    'Sector': sector,
                    'Bank': bank_name,
                    'Keyword': keyword,
                    'Avg Interest': "0",
                    'Max Interest': 0,
                    'Current Trend': 'N/A',
                    'Data Points': 0
                })
            
            time.sleep(2)  # Rate limiting for Google
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            trends_summary.append({
                'Sector': sector,
                'Bank': bank_name,
                'Keyword': info['keywords'][0],
                'Avg Interest': "0",
                'Max Interest': 0,
                'Current Trend': 'Error',
                'Data Points': 0
            })
            time.sleep(5)  # Wait longer after error

# Save trends summary
trends_summary_df = pd.DataFrame(trends_summary)
trends_summary_df.to_csv('data/social_media/combined/google_trends_summary.csv', index=False)
print(f"\n✓ Saved summary: data/social_media/combined/google_trends_summary.csv")

# =============================================================================
# 3. COMBINED SUMMARY & INSIGHTS
# =============================================================================

print("\n\n" + "=" * 80)
print("COMBINED SOCIAL MEDIA INSIGHTS".center(80))
print("=" * 80)

# Merge summaries
combined_summary = stocktwits_summary_df.merge(
    trends_summary_df[['Bank', 'Avg Interest', 'Current Trend']],
    on='Bank',
    how='outer'
)

print("\n" + "=" * 80)
print("StockTwits Sentiment Summary".center(80))
print("=" * 80)
print(stocktwits_summary_df.to_string(index=False))

print("\n\n" + "=" * 80)
print("Google Trends Summary".center(80))
print("=" * 80)
print(trends_summary_df.to_string(index=False))

# Save combined summary
combined_summary.to_csv('data/social_media/combined/social_media_summary.csv', index=False)
print(f"\n✓ Saved combined summary: data/social_media/combined/social_media_summary.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n\n" + "=" * 80)
print("SOCIAL MEDIA DATA COLLECTION COMPLETE!".center(80))
print("=" * 80)

total_stocktwits = stocktwits_summary_df['Total Messages'].sum()
total_trends_data = trends_summary_df['Data Points'].sum()

print(f"\n✓ StockTwits Messages Collected: {total_stocktwits}")
print(f"✓ Google Trends Data Points: {total_trends_data}")

print("\nFiles Created:")
print("  1. Individual StockTwits data: data/social_media/stocktwits/{sector}/{TICKER}_stocktwits.csv")
print("  2. Individual Google Trends: data/social_media/google_trends/{sector}/{TICKER}_trends.csv")
print("  3. Combined summaries: data/social_media/combined/")

print("\n" + "=" * 80)
print("KEY INSIGHTS:")
print("=" * 80)

if not stocktwits_summary_df.empty:
    # Most bullish bank
    bullish_data = stocktwits_summary_df[stocktwits_summary_df['Total Messages'] > 0]
    if not bullish_data.empty:
        most_bullish = bullish_data.loc[bullish_data['Bullish %'].apply(float).idxmax()]
        print(f"\n📱 StockTwits - Most Bullish: {most_bullish['Bank']} ({most_bullish['Bullish %']}% bullish)")

if not trends_summary_df.empty:
    # Highest interest bank
    trends_data_avail = trends_summary_df[trends_summary_df['Data Points'] > 0]
    if not trends_data_avail.empty:
        highest_interest = trends_data_avail.loc[trends_data_avail['Avg Interest'].apply(float).idxmax()]
        print(f"📈 Google Trends - Highest Interest: {highest_interest['Bank']} (Avg: {highest_interest['Avg Interest']})")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Correlate social sentiment with stock prices")
print("  2. Identify sentiment-driven price movements") 
print("  3. Integrate with news sentiment for comprehensive analysis")
print("=" * 80)
