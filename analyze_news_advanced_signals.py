#!/usr/bin/env python3
"""
Advanced Market Signal Extraction from News

Extracts sophisticated trading signals that professional brokers use:
- Technical signals (bullish/bearish patterns)
- Analyst ratings (buy/hold/sell recommendations)
- Macroeconomic indicators (policy, liquidity, growth)
- Risk signals (regulatory, operational, competitive)
- Leadership signals (CEO statements, governance)
- Market context (sector performance, peer comparison)

This goes beyond basic sentiment to capture actionable trading insights.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from collections import Counter

# =============================================================================
# ADVANCED SIGNAL KEYWORDS
# =============================================================================

# Technical Signals - Price & Chart Patterns
BULLISH_TECHNICAL = [
    # Price Action
    'rally', 'rallied', 'surge', 'surged', 'breakout', 'broke out',
    '52-week high', '52 week high', 'all-time high', 'record high',
    'momentum', 'uptrend', 'rising', 'gaining', 'advancing',
    'outperform', 'outperformed', 'strong gain', 'robust gain',

    # Chart Patterns
    'golden cross', 'bullish pattern', 'reversal', 'support level',
    'accumulation', 'buying pressure', 'demand surge',

    # Volume & Activity
    'high volume', 'heavy buying', 'institutional buying',
    'call option activity', 'bullish bet'
]

BEARISH_TECHNICAL = [
    # Price Action
    'crash', 'crashed', 'plunge', 'plunged', 'tumble', 'tumbled',
    'sell-off', 'selloff', 'decline', 'declined', 'fall', 'fell',
    'drag', 'dragged', 'weigh', 'weighed', 'slump', 'slumped',
    '52-week low', '52 week low', 'multi-year low',

    # Chart Patterns
    'death cross', 'bearish pattern', 'breakdown', 'broke down',
    'resistance level', 'distribution', 'selling pressure',
    'technicals turn', 'technical deterioration',

    # Market Context
    'underperform', 'underperformed', 'lag', 'lagged',
    'heavy selling', 'panic selling', 'put option'
]

NEUTRAL_TECHNICAL = [
    'consolidation', 'consolidate', 'sideways', 'range-bound',
    'flat', 'unchanged', 'stable', 'steady', 'hold',
    'wait-and-see', 'cautious', 'mixed signals'
]

# Analyst Ratings & Recommendations
RATING_BUY = [
    'buy', 'strong buy', 'accumulate', 'overweight',
    'upgrade', 'upgraded', 'rating increase', 'outperform',
    'positive outlook', 'bullish call', 'recommends buying'
]

RATING_SELL = [
    'sell', 'strong sell', 'reduce', 'underweight',
    'downgrade', 'downgraded', 'rating cut', 'rating decrease',
    'negative outlook', 'bearish call', 'recommends selling'
]

RATING_HOLD = [
    'hold', 'neutral', 'maintain', 'equal weight',
    'market perform', 'in-line', 'rated hold'
]

# Macroeconomic & Policy Signals
MACRO_POSITIVE = [
    'rbi support', 'liquidity injection', 'rate cut', 'stimulus',
    'gdp growth', 'economic expansion', 'policy support',
    'government aid', 'favorable regulation', 'tax benefit',
    'deposit growth', 'credit expansion'
]

MACRO_NEGATIVE = [
    'rate hike', 'tightening', 'liquidity crunch', 'liquidity stress',
    'deposit stress', 'credit squeeze', 'regulatory scrutiny',
    'policy headwind', 'slowdown', 'recession concern',
    'inflation worry', 'fiscal concern'
]

# Risk Signals
RISK_HIGH = [
    'npa', 'non-performing asset', 'bad loan', 'stressed asset',
    'default', 'defaulted', 'fraud', 'scam', 'investigation',
    'penalty', 'fine', 'violation', 'breach',
    'asset quality concern', 'provisioning increase',
    'competitive pressure', 'margin compression'
]

RISK_MEDIUM = [
    'challenge', 'concern', 'uncertainty', 'volatility',
    'competition', 'headwind', 'pressure',
    'slower growth', 'margin pressure'
]

# Leadership & Governance
LEADERSHIP_POSITIVE = [
    'ceo bullish', 'management confident', 'strategic plan',
    'expansion plan', 'growth strategy', 'innovation',
    'digital transformation', 'new initiative',
    'board approval', 'shareholder approval'
]

LEADERSHIP_NEGATIVE = [
    'ceo exit', 'management shake-up', 'board reshuffle',
    'leadership concern', 'governance issue',
    'strategy unclear', 'management warning'
]

# Earnings & Financial Performance
EARNINGS_BEAT = [
    'earnings beat', 'beat estimate', 'surpass expectation',
    'profit surge', 'revenue growth', 'strong quarter',
    'record earnings', 'above consensus'
]

EARNINGS_MISS = [
    'earnings miss', 'missed estimate', 'below expectation',
    'profit decline', 'revenue fall', 'weak quarter',
    'disappointing results', 'below consensus'
]

# =============================================================================
# SIGNAL EXTRACTION FUNCTIONS
# =============================================================================

def extract_technical_signals(text):
    """Extract bullish/bearish technical signals from text"""
    text_lower = text.lower()

    bullish_count = sum(1 for keyword in BULLISH_TECHNICAL if keyword in text_lower)
    bearish_count = sum(1 for keyword in BEARISH_TECHNICAL if keyword in text_lower)
    neutral_count = sum(1 for keyword in NEUTRAL_TECHNICAL if keyword in text_lower)

    # Calculate technical signal score (-1 to +1)
    total = bullish_count + bearish_count + neutral_count
    if total == 0:
        signal_score = 0.0
    else:
        signal_score = (bullish_count - bearish_count) / total

    # Determine dominant signal
    if bullish_count > bearish_count and bullish_count > neutral_count:
        signal_type = 'bullish'
    elif bearish_count > bullish_count and bearish_count > neutral_count:
        signal_type = 'bearish'
    else:
        signal_type = 'neutral'

    return {
        'technical_signal_score': signal_score,
        'technical_signal_type': signal_type,
        'technical_bullish_mentions': bullish_count,
        'technical_bearish_mentions': bearish_count
    }

def extract_analyst_rating(text):
    """Extract analyst rating signals"""
    text_lower = text.lower()

    buy_count = sum(1 for keyword in RATING_BUY if keyword in text_lower)
    sell_count = sum(1 for keyword in RATING_SELL if keyword in text_lower)
    hold_count = sum(1 for keyword in RATING_HOLD if keyword in text_lower)

    # Rating score (-1=sell, 0=hold, +1=buy)
    total = buy_count + sell_count + hold_count
    if total == 0:
        rating_score = 0.0
        rating_present = False
    else:
        rating_score = (buy_count - sell_count) / total
        rating_present = True

    return {
        'analyst_rating_score': rating_score,
        'analyst_rating_present': rating_present
    }

def extract_macro_signals(text):
    """Extract macroeconomic policy signals"""
    text_lower = text.lower()

    positive_count = sum(1 for keyword in MACRO_POSITIVE if keyword in text_lower)
    negative_count = sum(1 for keyword in MACRO_NEGATIVE if keyword in text_lower)

    # Macro signal score
    total = positive_count + negative_count
    if total == 0:
        macro_score = 0.0
    else:
        macro_score = (positive_count - negative_count) / total

    return {
        'macro_signal_score': macro_score,
        'macro_mentions': total
    }

def extract_risk_signals(text):
    """Extract risk and concern signals"""
    text_lower = text.lower()

    high_risk_count = sum(1 for keyword in RISK_HIGH if keyword in text_lower)
    medium_risk_count = sum(1 for keyword in RISK_MEDIUM if keyword in text_lower)

    # Risk score (0=low, 1=high)
    # High risk keywords weighted 2x
    total_risk = (high_risk_count * 2) + medium_risk_count
    risk_score = min(total_risk / 10.0, 1.0)  # Cap at 1.0

    return {
        'risk_score': risk_score,
        'high_risk_mentions': high_risk_count
    }

def extract_leadership_signals(text):
    """Extract leadership and governance signals"""
    text_lower = text.lower()

    positive_count = sum(1 for keyword in LEADERSHIP_POSITIVE if keyword in text_lower)
    negative_count = sum(1 for keyword in LEADERSHIP_NEGATIVE if keyword in text_lower)

    total = positive_count + negative_count
    if total == 0:
        leadership_score = 0.0
    else:
        leadership_score = (positive_count - negative_count) / total

    return {
        'leadership_signal_score': leadership_score
    }

def extract_earnings_signals(text):
    """Extract earnings beat/miss signals"""
    text_lower = text.lower()

    beat_count = sum(1 for keyword in EARNINGS_BEAT if keyword in text_lower)
    miss_count = sum(1 for keyword in EARNINGS_MISS if keyword in text_lower)

    total = beat_count + miss_count
    if total == 0:
        earnings_score = 0.0
        earnings_present = False
    else:
        earnings_score = (beat_count - miss_count) / total
        earnings_present = True

    return {
        'earnings_signal_score': earnings_score,
        'earnings_event_present': earnings_present
    }

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_article_advanced(row):
    """Extract all advanced signals from a single article"""
    # Combine title and description for analysis
    text = f"{row.get('title', '')} {row.get('description', '')}"

    signals = {}

    # Extract all signal types
    signals.update(extract_technical_signals(text))
    signals.update(extract_analyst_rating(text))
    signals.update(extract_macro_signals(text))
    signals.update(extract_risk_signals(text))
    signals.update(extract_leadership_signals(text))
    signals.update(extract_earnings_signals(text))

    return signals

def process_stock_news(ticker, stock_name):
    """Process all news for a single stock and aggregate signals"""

    print(f"\n{'='*80}")
    print(f"Processing: {stock_name} ({ticker})".center(80))
    print("="*80)

    news_file = f'data/news_historical/{ticker}_news_30d.csv'

    if not os.path.exists(news_file):
        print(f"   ✗ News file not found: {news_file}")
        return None

    # Load news
    df = pd.read_csv(news_file)
    print(f"   ✓ Loaded {len(df)} news articles")

    # Extract signals for each article
    print(f"   ⚙️  Extracting advanced market signals...")

    signals_list = []
    for idx, row in df.iterrows():
        signals = analyze_article_advanced(row)
        signals['date'] = row['date']
        signals_list.append(signals)

    signals_df = pd.DataFrame(signals_list)

    # Aggregate by date
    daily_signals = signals_df.groupby(pd.to_datetime(signals_df['date']).dt.date).agg({
        # Technical signals
        'technical_signal_score': 'mean',
        'technical_bullish_mentions': 'sum',
        'technical_bearish_mentions': 'sum',

        # Analyst ratings
        'analyst_rating_score': 'mean',
        'analyst_rating_present': 'max',  # 1 if any rating present

        # Macro signals
        'macro_signal_score': 'mean',
        'macro_mentions': 'sum',

        # Risk signals
        'risk_score': 'mean',
        'high_risk_mentions': 'sum',

        # Leadership
        'leadership_signal_score': 'mean',

        # Earnings
        'earnings_signal_score': 'mean',
        'earnings_event_present': 'max'
    }).reset_index()

    daily_signals.columns = ['date'] + list(daily_signals.columns[1:])

    # Save to CSV
    output_file = f'data/advanced_signals/{ticker}_advanced_signals.csv'
    os.makedirs('data/advanced_signals', exist_ok=True)
    daily_signals.to_csv(output_file, index=False)

    print(f"   ✓ Extracted {len(daily_signals.columns)-1} advanced signal features")
    print(f"   ✓ Saved to: {output_file}")

    # Print summary statistics
    print(f"\n   📊 Signal Summary:")
    print(f"      Technical Bullish Days: {(daily_signals['technical_signal_score'] > 0.2).sum()}")
    print(f"      Technical Bearish Days: {(daily_signals['technical_signal_score'] < -0.2).sum()}")
    print(f"      Analyst Ratings Found: {daily_signals['analyst_rating_present'].sum()}")
    print(f"      High Risk Mentions: {daily_signals['high_risk_mentions'].sum()}")
    print(f"      Earnings Events: {daily_signals['earnings_event_present'].sum()}")

    return daily_signals

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    STOCKS = [
        ('HDFCBANK', 'HDFC Bank'),
        ('ICICIBANK', 'ICICI Bank'),
        ('KOTAKBANK', 'Kotak Mahindra Bank'),
        ('AXISBANK', 'Axis Bank'),
        ('SBIN', 'State Bank of India'),
        ('PNB', 'Punjab National Bank'),
        ('BANKBARODA', 'Bank of Baroda'),
        ('CANBK', 'Canara Bank')
    ]

    print("=" * 80)
    print("ADVANCED MARKET SIGNAL EXTRACTION".center(80))
    print("Professional-Grade Signal Analysis for Stock Trading".center(80))
    print("=" * 80)

    all_summaries = []

    for ticker, stock_name in STOCKS:
        result = process_stock_news(ticker, stock_name)
        if result is not None:
            summary = {
                'ticker': ticker,
                'stock': stock_name,
                'total_days': len(result),
                'avg_technical_score': result['technical_signal_score'].mean(),
                'avg_risk_score': result['risk_score'].mean(),
                'total_analyst_ratings': result['analyst_rating_present'].sum(),
                'total_earnings_events': result['earnings_event_present'].sum()
            }
            all_summaries.append(summary)

    # Save overall summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('data/advanced_signals/extraction_summary.csv', index=False)

    print("\n\n" + "=" * 80)
    print("EXTRACTION COMPLETE - SUMMARY".center(80))
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("✓ Advanced signal extraction complete!".center(80))
    print("=" * 80)
    print("\nNew Features Extracted:")
    print("  • Technical Signals: bullish/bearish score, mention counts")
    print("  • Analyst Ratings: buy/hold/sell scores")
    print("  • Macro Signals: policy/liquidity indicators")
    print("  • Risk Signals: NPA, regulatory, competition")
    print("  • Leadership: CEO statements, governance")
    print("  • Earnings: beat/miss indicators")
    print(f"\nTotal: 11 new advanced features per stock")
    print(f"Combined with 4 FinBERT features = 15 sentiment features")
    print(f"Total model features: 35 technical + 15 sentiment = 50 features")
