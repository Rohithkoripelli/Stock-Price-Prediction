"""
Analyze collected news with FinBERT for all 8 stocks
Creates daily sentiment scores for model training
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('Analyzing news with FinBERT for all 8 stocks')
print('='*70)

STOCKS = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK',
          'SBIN', 'PNB', 'BANKBARODA', 'CANBK']

# Load FinBERT model once
print('\nLoading FinBERT model...')
model_name = 'yiyanghkust/finbert-tone'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=-1, framework='pt')
print('✓ Model loaded\n')

os.makedirs('data/finbert_daily_sentiment', exist_ok=True)

all_results = []

for ticker in STOCKS:
    print(f'\n{ticker}')
    print('-'*60)

    # Load news
    news_file = f'data/news_historical/{ticker}_news_30d.csv'

    if not os.path.exists(news_file):
        print(f'  ⚠ No news file found')
        continue

    df = pd.read_csv(news_file)
    print(f'  Loaded {len(df)} articles')

    # Analyze with FinBERT
    results = []
    for idx, row in df.iterrows():
        text = f"{row['title']} {row.get('description', '')}"
        text = text[:512]  # Truncate

        sentiment = finbert(text)[0]

        # Convert label to directional score (-1 to +1)
        # Weight by confidence to preserve signal strength
        label = sentiment['label']
        confidence = sentiment['score']

        if label == 'Positive':
            sentiment_score = confidence  # 0 to +1
        elif label == 'Negative':
            sentiment_score = -confidence  # 0 to -1
        else:  # Neutral
            sentiment_score = 0.0

        results.append({
            'date': row['date'],
            'title': row['title'],
            'sentiment_label': label,
            'sentiment_score': sentiment_score,
            'confidence': confidence
        })

        if (idx + 1) % 20 == 0:
            print(f'    Analyzed {idx + 1}/{len(df)}...')

    result_df = pd.DataFrame(results)
    result_df['date'] = pd.to_datetime(result_df['date'])

    # Create daily aggregated sentiment
    daily = result_df.groupby('date').agg({
        'sentiment_score': 'mean',
        'title': 'count'  # news volume
    }).rename(columns={'title': 'news_volume'})

    # Calculate sentiment polarity (-1 to +1)
    def calc_polarity(date):
        day_articles = result_df[result_df['date'] == date]
        pos = len(day_articles[day_articles['sentiment_label'] == 'Positive'])
        neg = len(day_articles[day_articles['sentiment_label'] == 'Negative'])
        neu = len(day_articles[day_articles['sentiment_label'] == 'Neutral'])
        total = len(day_articles)

        if total == 0:
            return 0.0

        # Weighted polarity
        return (pos - neg) / total

    daily['sentiment_polarity'] = [calc_polarity(date) for date in daily.index]

    # Add event flags
    def has_earnings(date):
        day_articles = result_df[result_df['date'] == date]
        earnings_keywords = ['quarterly', 'results', 'earnings', 'Q1', 'Q2', 'Q3', 'Q4', 'profit']
        return int(any(any(kw.lower() in str(title).lower() for kw in earnings_keywords)
                      for title in day_articles['title']))

    daily['earnings_event'] = [has_earnings(date) for date in daily.index]

    # Save
    output_file = f'data/finbert_daily_sentiment/{ticker}_daily_sentiment.csv'
    daily.to_csv(output_file)

    print(f'  ✓ Saved daily sentiment to {output_file}')
    avg_polarity = daily['sentiment_polarity'].mean()
    earnings_count = daily['earnings_event'].sum()
    print(f'  📊 Avg sentiment polarity: {avg_polarity:.3f}')
    print(f'  📰 Total trading days with news: {len(daily)}')
    print(f'  📅 Earnings events detected: {earnings_count}')

    all_results.append({
        'ticker': ticker,
        'articles': len(df),
        'days_with_news': len(daily),
        'avg_polarity': daily['sentiment_polarity'].mean(),
        'earnings_events': daily['earnings_event'].sum()
    })

# Summary
print('\n' + '='*70)
print('FINBERT ANALYSIS SUMMARY')
print('='*70)
if all_results:
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('data/finbert_daily_sentiment/analysis_summary.csv', index=False)
    print(f'\n✓ Analysis complete for {len(all_results)} stocks')
else:
    print('No stocks analyzed')
