"""
Future Predictions with Advanced Signals - Next Trading Day

Uses the last available data sequence with Advanced Signals sentiment features
to predict the next trading day for all 8 stocks using trained V5
Transformer models with Advanced Signals.

Includes Market Mood Override system that combines:
- Bank Nifty index movement (macro market direction)
- Per-stock sentiment from FinBERT + Advanced Signals (stock-level news)
- USD/INR forex weakness (FII selling pressure)
to produce a combined signal that overrides raw model output when
market conditions clearly contradict the model's prediction.
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model

print("=" * 80)
print("ADVANCED PREDICTIONS - NEXT TRADING DAY".center(80))
print("with Market Mood Override (Bank Nifty + Sentiment + INR)".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

STOCKS = [
    ('HDFCBANK', 'HDFC Bank', 'private_banks'),
    ('ICICIBANK', 'ICICI Bank', 'private_banks'),
    ('KOTAKBANK', 'Kotak Mahindra Bank', 'private_banks'),
    ('AXISBANK', 'Axis Bank', 'private_banks'),
    ('SBIN', 'State Bank of India', 'psu_banks'),
    ('PNB', 'Punjab National Bank', 'psu_banks'),
    ('BANKBARODA', 'Bank of Baroda', 'psu_banks'),
    ('CANBK', 'Canara Bank', 'psu_banks')
]

# =============================================================================
# MARKET MOOD OVERRIDE SYSTEM
# =============================================================================

# --- Bank Nifty Thresholds ---
# These define what constitutes a significant index move
NIFTY_STRONG_BEARISH = -1.0    # Index down >1%: strong bearish
NIFTY_BEARISH = -0.5           # Index down >0.5%: bearish
NIFTY_MILD_BEARISH = -0.25     # Index down >0.25%: mild bearish
NIFTY_MILD_BULLISH = 0.25      # Index up >0.25%: mild bullish
NIFTY_BULLISH = 0.5            # Index up >0.5%: bullish
NIFTY_STRONG_BULLISH = 1.0     # Index up >1%: strong bullish

# --- INR Weakness Threshold ---
INR_WEAKNESS_THRESHOLD = 0.003  # 0.3% weakness triggers override

# --- Sentiment score thresholds ---
SENTIMENT_STRONG_NEGATIVE = -0.3
SENTIMENT_NEGATIVE = -0.1
SENTIMENT_POSITIVE = 0.1
SENTIMENT_STRONG_POSITIVE = 0.3

# --- How many recent days of sentiment to consider ---
SENTIMENT_LOOKBACK_DAYS = 5


def get_bank_nifty_signal():
    """
    Read Bank Nifty index data and compute a market signal.

    Returns:
        dict with:
            - nifty_return_1d: latest 1-day % return
            - nifty_return_5d: latest 5-day % return
            - nifty_signal: float score (-1.0 to +1.0)
            - nifty_label: human-readable label
    """
    try:
        nifty_df = pd.read_csv(
            'data/market_index/NIFTY_BANK_index.csv',
            skiprows=[1, 2], index_col=0, parse_dates=True
        )
        nifty_df = nifty_df.sort_index()

        if len(nifty_df) < 21:
            print("   ⚠ Insufficient Bank Nifty data")
            return None

        # Calculate returns
        close = nifty_df['Close']
        ret_1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100
        ret_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100
        ret_20d = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100

        # Weighted composite: 1-day is most important for next-day prediction,
        # but 5-day trend adds context
        # 60% weight to 1-day, 30% to 5-day trend (normalized to daily), 10% to 20-day
        composite = (0.60 * ret_1d) + (0.30 * (ret_5d / 5)) + (0.10 * (ret_20d / 20))

        # Convert to signal score (-1 to +1)
        # Using tanh-like scaling: ±2% composite maps to ±1.0
        signal = max(-1.0, min(1.0, composite / 2.0))

        # Label
        if ret_1d <= NIFTY_STRONG_BEARISH:
            label = "STRONG BEARISH"
        elif ret_1d <= NIFTY_BEARISH:
            label = "BEARISH"
        elif ret_1d <= NIFTY_MILD_BEARISH:
            label = "MILD BEARISH"
        elif ret_1d >= NIFTY_STRONG_BULLISH:
            label = "STRONG BULLISH"
        elif ret_1d >= NIFTY_BULLISH:
            label = "BULLISH"
        elif ret_1d >= NIFTY_MILD_BULLISH:
            label = "MILD BULLISH"
        else:
            label = "NEUTRAL"

        return {
            'nifty_return_1d': float(ret_1d),
            'nifty_return_5d': float(ret_5d),
            'nifty_return_20d': float(ret_20d),
            'nifty_composite': float(composite),
            'nifty_signal': float(signal),
            'nifty_label': label,
            'nifty_level': float(close.iloc[-1])
        }

    except Exception as e:
        print(f"   ⚠ Could not read Bank Nifty data: {e}")
        return None


def get_inr_signal():
    """
    Read USD/INR forex data and compute INR weakness signal.

    Returns:
        dict with inr_weakness, usd_inr_rate, inr_signal, inr_label
    """
    try:
        usd_inr_data = pd.read_csv('data/forex/USD_INR_rates.csv')
        latest_weakness = float(usd_inr_data['inr_weakness_score'].iloc[-1])
        latest_rate = float(usd_inr_data['usd_inr_rate'].iloc[-1])

        # Signal: positive weakness = bearish for stocks
        if latest_weakness > 0.005:
            label = "STRONG BEARISH"
            signal = -0.8
        elif latest_weakness > INR_WEAKNESS_THRESHOLD:
            label = "BEARISH"
            signal = -0.5
        elif latest_weakness > 0.001:
            label = "MILD BEARISH"
            signal = -0.2
        elif latest_weakness < -0.003:
            label = "BULLISH"
            signal = 0.4
        elif latest_weakness < -0.001:
            label = "MILD BULLISH"
            signal = 0.2
        else:
            label = "NEUTRAL"
            signal = 0.0

        return {
            'inr_weakness': float(latest_weakness),
            'usd_inr_rate': float(latest_rate),
            'inr_signal': float(signal),
            'inr_label': label
        }

    except Exception as e:
        print(f"   ⚠ Could not read USD/INR data: {e}")
        return None


def get_stock_sentiment(ticker):
    """
    Read FinBERT daily sentiment + Advanced Signals for a specific stock.
    Aggregate the last SENTIMENT_LOOKBACK_DAYS days into a single score.

    Returns:
        dict with:
            - sentiment_score: weighted average sentiment (-1 to +1)
            - news_volume: total recent articles
            - advanced_signal_score: from advanced signals analysis
            - combined_sentiment: blended score
            - sentiment_label: human-readable
    """
    finbert_score = 0.0
    finbert_volume = 0
    advanced_score = 0.0
    has_finbert = False
    has_advanced = False

    # --- FinBERT Sentiment ---
    finbert_path = f"data/finbert_daily_sentiment/{ticker}_daily_sentiment.csv"
    if os.path.exists(finbert_path):
        try:
            df = pd.read_csv(finbert_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Get the last N days of sentiment
            cutoff = datetime.now() - timedelta(days=SENTIMENT_LOOKBACK_DAYS)
            recent = df[df['date'] >= cutoff]

            if len(recent) > 0:
                # Volume-weighted sentiment: articles with more news_volume
                # carry more weight
                total_volume = recent['news_volume'].sum()
                if total_volume > 0:
                    finbert_score = (recent['sentiment_score'] * recent['news_volume']).sum() / total_volume
                else:
                    finbert_score = recent['sentiment_score'].mean()
                finbert_volume = int(total_volume)
                has_finbert = True
            elif len(df) > 0:
                # Fallback: use last 3 entries
                last_entries = df.tail(3)
                finbert_score = last_entries['sentiment_score'].mean()
                finbert_volume = int(last_entries['news_volume'].sum())
                has_finbert = True
        except Exception as e:
            print(f"      ⚠ Error reading FinBERT for {ticker}: {e}")

    # --- Advanced Signals ---
    advanced_path = f"data/advanced_signals/{ticker}_advanced_signals.csv"
    if os.path.exists(advanced_path):
        try:
            df = pd.read_csv(advanced_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            cutoff = datetime.now() - timedelta(days=SENTIMENT_LOOKBACK_DAYS)
            recent = df[df['date'] >= cutoff]

            if len(recent) == 0 and len(df) > 0:
                recent = df.tail(3)

            if len(recent) > 0:
                # Combine advanced signal dimensions:
                # technical_signal_score: bullish vs bearish technical mentions
                # analyst_rating_score: buy/sell ratings
                # macro_signal_score: macro environment
                # risk_score: risk/NPA mentions (negative = bad)
                # earnings_signal_score: earnings beats/misses

                tech = recent['technical_signal_score'].mean()
                analyst = recent['analyst_rating_score'].mean()
                macro = recent['macro_signal_score'].mean()
                risk = -recent['risk_score'].mean()  # Invert: higher risk = more negative
                earnings = recent['earnings_signal_score'].mean()

                # Also look at raw bullish vs bearish mention counts
                total_bullish = recent['technical_bullish_mentions'].sum()
                total_bearish = recent['technical_bearish_mentions'].sum()

                # Bullish/bearish ratio as a signal
                if (total_bullish + total_bearish) > 0:
                    bull_bear_ratio = (total_bullish - total_bearish) / (total_bullish + total_bearish)
                else:
                    bull_bear_ratio = 0.0

                # Weighted combination
                advanced_score = (
                    0.30 * tech +          # Technical signals
                    0.25 * analyst +       # Analyst ratings
                    0.15 * macro +         # Macro environment
                    0.15 * risk +          # Risk factors (inverted)
                    0.10 * earnings +      # Earnings signals
                    0.05 * bull_bear_ratio  # Bull/bear ratio
                )
                has_advanced = True

        except Exception as e:
            print(f"      ⚠ Error reading Advanced Signals for {ticker}: {e}")

    # --- Combine FinBERT + Advanced Signals ---
    if has_finbert and has_advanced:
        # 50/50 blend when both available
        combined = 0.50 * finbert_score + 0.50 * advanced_score
    elif has_finbert:
        combined = finbert_score
    elif has_advanced:
        combined = advanced_score
    else:
        combined = 0.0

    # Clamp to [-1, +1]
    combined = max(-1.0, min(1.0, combined))

    # Label
    if combined <= SENTIMENT_STRONG_NEGATIVE:
        label = "STRONG NEGATIVE"
    elif combined <= SENTIMENT_NEGATIVE:
        label = "NEGATIVE"
    elif combined >= SENTIMENT_STRONG_POSITIVE:
        label = "STRONG POSITIVE"
    elif combined >= SENTIMENT_POSITIVE:
        label = "POSITIVE"
    else:
        label = "NEUTRAL"

    return {
        'finbert_score': float(finbert_score),
        'finbert_volume': finbert_volume,
        'advanced_score': float(advanced_score),
        'combined_sentiment': float(combined),
        'sentiment_label': label,
        'has_finbert': has_finbert,
        'has_advanced': has_advanced
    }


def compute_market_mood(nifty_info, inr_info, sentiment_info):
    """
    Combine Bank Nifty signal, INR signal, and stock-level sentiment
    into a single Market Mood score and override decision.

    The decision matrix:
    ┌─────────────────────┬──────────────┬──────────────┬──────────────┐
    │ Bank Nifty ↓ / Sent │  Negative    │   Neutral    │  Positive    │
    ├─────────────────────┼──────────────┼──────────────┼──────────────┤
    │ Strong Bearish      │ FORCE DOWN   │ FORCE DOWN   │ BIAS DOWN    │
    │ Bearish             │ FORCE DOWN   │ BIAS DOWN    │ REDUCE CONF  │
    │ Mild Bearish        │ BIAS DOWN    │ REDUCE CONF  │ NO OVERRIDE  │
    │ Neutral             │ BIAS DOWN    │ NO OVERRIDE  │ NO OVERRIDE  │
    │ Mild Bullish        │ NO OVERRIDE  │ NO OVERRIDE  │ BIAS UP      │
    │ Bullish             │ REDUCE CONF  │ BIAS UP      │ BIAS UP      │
    │ Strong Bullish      │ REDUCE CONF  │ BIAS UP      │ FORCE UP     │
    └─────────────────────┴──────────────┴──────────────┴──────────────┘

    Returns:
        dict with mood_score, override_action, override_reason, etc.
    """
    nifty_signal = nifty_info['nifty_signal'] if nifty_info else 0.0
    nifty_1d = nifty_info['nifty_return_1d'] if nifty_info else 0.0
    nifty_label = nifty_info['nifty_label'] if nifty_info else "UNKNOWN"

    inr_signal = inr_info['inr_signal'] if inr_info else 0.0
    inr_label = inr_info['inr_label'] if inr_info else "UNKNOWN"

    sentiment = sentiment_info['combined_sentiment']
    sent_label = sentiment_info['sentiment_label']

    # Compute overall market mood score:
    # Bank Nifty is the primary driver (50% weight) because all stocks are banking
    # INR is secondary macro (25% weight) - affects FII flows
    # Stock sentiment is the per-stock qualifier (25% weight)
    mood_score = (0.50 * nifty_signal) + (0.25 * inr_signal) + (0.25 * sentiment)
    mood_score = max(-1.0, min(1.0, mood_score))

    # Determine override action based on the decision matrix
    override_action = "NO_OVERRIDE"
    override_reason = ""
    mood_confidence_adjustment = 0.0  # Additive adjustment to model confidence
    mood_direction_override = None     # None = don't override, "UP"/"DOWN" = force

    # --- FORCE DOWN scenarios ---
    # Bank Nifty strongly bearish: override regardless of sentiment
    if nifty_1d <= NIFTY_STRONG_BEARISH:
        if sentiment <= SENTIMENT_POSITIVE:
            # Strong bearish index + non-positive sentiment → Force DOWN
            override_action = "FORCE_DOWN"
            mood_direction_override = "DOWN"
            mood_confidence_adjustment = 0.15 + abs(nifty_1d / 100)
            override_reason = (
                f"Bank Nifty fell {nifty_1d:+.2f}% ({nifty_label}) with "
                f"{sent_label.lower()} stock sentiment ({sentiment:+.3f})"
            )
        else:
            # Strong bearish index but positive sentiment → Bias down but less harsh
            override_action = "BIAS_DOWN"
            mood_confidence_adjustment = 0.05
            override_reason = (
                f"Bank Nifty fell {nifty_1d:+.2f}% ({nifty_label}) but "
                f"stock has positive sentiment ({sentiment:+.3f}) - partial override"
            )

    # Bank Nifty bearish
    elif nifty_1d <= NIFTY_BEARISH:
        if sentiment <= SENTIMENT_NEGATIVE:
            override_action = "FORCE_DOWN"
            mood_direction_override = "DOWN"
            mood_confidence_adjustment = 0.10 + abs(nifty_1d / 100)
            override_reason = (
                f"Bank Nifty fell {nifty_1d:+.2f}% ({nifty_label}) confirmed by "
                f"negative stock sentiment ({sentiment:+.3f})"
            )
        elif sentiment <= SENTIMENT_POSITIVE:
            override_action = "BIAS_DOWN"
            mood_confidence_adjustment = 0.05
            override_reason = (
                f"Bank Nifty fell {nifty_1d:+.2f}% ({nifty_label}) with "
                f"neutral sentiment ({sentiment:+.3f}) - bearish bias"
            )
        else:
            override_action = "REDUCE_CONFIDENCE"
            mood_confidence_adjustment = -0.10
            override_reason = (
                f"Bank Nifty fell {nifty_1d:+.2f}% but positive sentiment "
                f"({sentiment:+.3f}) provides counter-signal - reducing confidence"
            )

    # Bank Nifty mild bearish
    elif nifty_1d <= NIFTY_MILD_BEARISH:
        if sentiment <= SENTIMENT_NEGATIVE:
            override_action = "BIAS_DOWN"
            mood_confidence_adjustment = 0.05
            override_reason = (
                f"Bank Nifty slightly down {nifty_1d:+.2f}% confirmed by "
                f"negative sentiment ({sentiment:+.3f})"
            )
        elif sentiment <= SENTIMENT_POSITIVE:
            override_action = "REDUCE_CONFIDENCE"
            mood_confidence_adjustment = -0.05
            override_reason = (
                f"Bank Nifty slightly down {nifty_1d:+.2f}% with "
                f"neutral sentiment - mild uncertainty"
            )
        # else: positive sentiment overrides mild bearish → no override

    # --- FORCE UP scenarios ---
    elif nifty_1d >= NIFTY_STRONG_BULLISH:
        if sentiment >= SENTIMENT_NEGATIVE:
            if sentiment >= SENTIMENT_STRONG_POSITIVE:
                override_action = "FORCE_UP"
                mood_direction_override = "UP"
                mood_confidence_adjustment = 0.15 + (nifty_1d / 100)
                override_reason = (
                    f"Bank Nifty surged {nifty_1d:+.2f}% ({nifty_label}) with "
                    f"strong positive sentiment ({sentiment:+.3f})"
                )
            else:
                override_action = "BIAS_UP"
                mood_confidence_adjustment = 0.05
                override_reason = (
                    f"Bank Nifty surged {nifty_1d:+.2f}% ({nifty_label}) - bullish bias"
                )
        else:
            override_action = "REDUCE_CONFIDENCE"
            mood_confidence_adjustment = -0.05
            override_reason = (
                f"Bank Nifty surged {nifty_1d:+.2f}% but stock has "
                f"negative sentiment ({sentiment:+.3f}) - conflicting signals"
            )

    elif nifty_1d >= NIFTY_BULLISH:
        if sentiment >= SENTIMENT_POSITIVE:
            override_action = "BIAS_UP"
            mood_confidence_adjustment = 0.05
            override_reason = (
                f"Bank Nifty up {nifty_1d:+.2f}% ({nifty_label}) with "
                f"positive sentiment ({sentiment:+.3f})"
            )
        elif sentiment <= SENTIMENT_NEGATIVE:
            override_action = "REDUCE_CONFIDENCE"
            mood_confidence_adjustment = -0.05
            override_reason = (
                f"Bank Nifty up {nifty_1d:+.2f}% but stock has "
                f"negative sentiment ({sentiment:+.3f})"
            )

    # --- INR weakness as additional bearish pressure ---
    # If INR is significantly weak, add extra bearish pressure on top of Nifty signal
    if inr_info and inr_info['inr_weakness'] > INR_WEAKNESS_THRESHOLD:
        if override_action in ("NO_OVERRIDE", "REDUCE_CONFIDENCE"):
            override_action = "BIAS_DOWN"
            mood_confidence_adjustment = max(mood_confidence_adjustment, 0.05)
            inr_reason = f" + INR weakness {inr_info['inr_weakness']*100:.2f}% (FII selling)"
            override_reason = (override_reason + inr_reason) if override_reason else inr_reason
        elif override_action in ("BIAS_DOWN", "FORCE_DOWN"):
            # Amplify the bearish signal
            mood_confidence_adjustment += 0.05
            override_reason += f" + INR weakness amplifier ({inr_info['inr_weakness']*100:.2f}%)"
        elif override_action in ("BIAS_UP", "FORCE_UP"):
            # INR weakness contradicts bullish Nifty → reduce to no override
            override_action = "REDUCE_CONFIDENCE"
            mood_confidence_adjustment = -0.10
            override_reason = (
                f"Bullish Nifty ({nifty_1d:+.2f}%) contradicted by INR weakness "
                f"({inr_info['inr_weakness']*100:.2f}%) - conflicting macro signals"
            )

    # Mood label
    if mood_score <= -0.4:
        mood_label = "VERY BEARISH"
    elif mood_score <= -0.15:
        mood_label = "BEARISH"
    elif mood_score <= -0.05:
        mood_label = "SLIGHTLY BEARISH"
    elif mood_score >= 0.4:
        mood_label = "VERY BULLISH"
    elif mood_score >= 0.15:
        mood_label = "BULLISH"
    elif mood_score >= 0.05:
        mood_label = "SLIGHTLY BULLISH"
    else:
        mood_label = "NEUTRAL"

    return {
        'mood_score': float(mood_score),
        'mood_label': mood_label,
        'override_action': override_action,
        'override_reason': override_reason,
        'mood_direction_override': mood_direction_override,
        'mood_confidence_adjustment': float(mood_confidence_adjustment),
        'nifty_signal': float(nifty_signal),
        'inr_signal': float(inr_signal),
        'sentiment_signal': float(sentiment)
    }


def apply_market_mood_override(direction, confidence, predicted_pct_change, magnitude, mood):
    """
    Apply the Market Mood Override to the model's raw prediction.

    Returns:
        (direction, confidence, predicted_pct_change) — adjusted values
    """
    action = mood['override_action']
    adj = mood['mood_confidence_adjustment']

    original_direction = direction
    original_confidence = confidence
    original_change = predicted_pct_change

    if action == "FORCE_DOWN":
        direction = "DOWN"
        # Use model magnitude but ensure negative
        predicted_pct_change = -abs(magnitude) * (1 + abs(mood['mood_score']))
        # Boost confidence for the forced direction
        confidence = min(0.95, max(confidence, 0.65) + adj)

    elif action == "FORCE_UP":
        direction = "UP"
        predicted_pct_change = abs(magnitude) * (1 + abs(mood['mood_score']))
        confidence = min(0.95, max(confidence, 0.65) + adj)

    elif action == "BIAS_DOWN":
        if direction == "UP":
            # Flip to DOWN with reduced confidence
            direction = "DOWN"
            predicted_pct_change = -abs(magnitude) * 0.5  # Halved magnitude since we're less sure
            confidence = min(0.75, 0.55 + adj)
        else:
            # Already DOWN, boost confidence
            confidence = min(0.95, confidence + adj)
            predicted_pct_change = predicted_pct_change * (1 + abs(mood['mood_score']) * 0.5)

    elif action == "BIAS_UP":
        if direction == "DOWN":
            # Flip to UP with reduced confidence
            direction = "UP"
            predicted_pct_change = abs(magnitude) * 0.5
            confidence = min(0.75, 0.55 + adj)
        else:
            # Already UP, boost confidence
            confidence = min(0.95, confidence + adj)
            predicted_pct_change = predicted_pct_change * (1 + abs(mood['mood_score']) * 0.5)

    elif action == "REDUCE_CONFIDENCE":
        # Keep direction but reduce confidence (cap at 65%)
        confidence = max(0.50, min(0.65, confidence + adj))

    # NO_OVERRIDE: keep everything as-is

    return direction, confidence, predicted_pct_change


# =============================================================================
# STEP 1: COMPUTE MACRO SIGNALS (once, shared across all stocks)
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: MACRO MARKET SIGNALS".center(80))
print("=" * 80)

nifty_info = get_bank_nifty_signal()
if nifty_info:
    print(f"\n   📊 BANK NIFTY INDEX")
    print(f"   Level: {nifty_info['nifty_level']:,.2f}")
    print(f"   1-Day Return: {nifty_info['nifty_return_1d']:+.2f}%")
    print(f"   5-Day Return: {nifty_info['nifty_return_5d']:+.2f}%")
    print(f"   20-Day Return: {nifty_info['nifty_return_20d']:+.2f}%")
    print(f"   Composite Score: {nifty_info['nifty_composite']:+.4f}")
    print(f"   Signal: {nifty_info['nifty_signal']:+.3f} → {nifty_info['nifty_label']}")
else:
    print("\n   ⚠ Bank Nifty data unavailable - macro signal disabled")

inr_info = get_inr_signal()
if inr_info:
    print(f"\n   💱 USD/INR FOREX")
    print(f"   Rate: ₹{inr_info['usd_inr_rate']:.4f}")
    print(f"   INR Weakness: {inr_info['inr_weakness']*100:+.3f}%")
    print(f"   Signal: {inr_info['inr_signal']:+.3f} → {inr_info['inr_label']}")
else:
    print("\n   ⚠ USD/INR data unavailable - INR signal disabled")


# =============================================================================
# STEP 2: GENERATE PREDICTIONS FOR EACH STOCK
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: STOCK PREDICTIONS WITH MARKET MOOD OVERRIDE".center(80))
print("=" * 80)

all_predictions = []

for ticker, stock_name, sector in STOCKS:
    print(f"\n{'='*80}")
    print(f"PREDICTING: {stock_name} ({ticker})".center(80))
    print("="*80)

    try:
        # 1. Load Advanced Signals-prepared data
        print(f"\n1. Loading Advanced Signals-prepared data...")
        with open(f'data/advanced_model_ready/{ticker}_advanced.pkl', 'rb') as f:
            data = pickle.load(f)

        # Get the last sequence from test set (most recent data)
        X_last = data['X_test'][-1:]

        # Get base price from the raw data
        stock_data = pd.read_csv(
            f'data/stocks/{sector}/{ticker}_data.csv',
            skiprows=[1, 2],
            index_col=0,
            parse_dates=[0]
        )
        stock_data = stock_data.sort_index()
        base_price_last = stock_data['Close'].iloc[-1]

        print(f"   ✓ Loaded last sequence with Advanced Signals features")
        print(f"   ✓ Base price: ₹{base_price_last:.2f}")
        print(f"   ✓ Sequence shape: {X_last.shape}")
        print(f"   ✓ Features: {data['num_features']} (includes Advanced Signals sentiment)")

        # 2. Load Advanced Signals model
        print(f"\n2. Loading V5 Transformer + Advanced Signals model...")
        model = load_model(
            f'models/saved_v5_advanced/{ticker}/best_model.keras',
            compile=False
        )
        print(f"   ✓ Advanced Signals-enhanced model loaded")

        # 3. Make raw model prediction
        print(f"\n3. Making raw model prediction...")
        predictions = model.predict(X_last, verbose=0)

        direction_prob = predictions[0][0][0]
        magnitude = predictions[1][0][0]

        if direction_prob > 0.5:
            direction = "UP"
            predicted_pct_change = abs(magnitude)
            confidence = direction_prob
        else:
            direction = "DOWN"
            predicted_pct_change = -abs(magnitude)
            confidence = 1 - direction_prob

        print(f"   Raw Model Output: {direction} ({confidence*100:.1f}% conf) Change: {predicted_pct_change:+.2f}%")

        # 4. Get per-stock sentiment
        print(f"\n4. Reading stock-level sentiment...")
        sentiment_info = get_stock_sentiment(ticker)
        print(f"   FinBERT Score: {sentiment_info['finbert_score']:+.3f} (volume: {sentiment_info['finbert_volume']})")
        print(f"   Advanced Signals Score: {sentiment_info['advanced_score']:+.3f}")
        print(f"   Combined Sentiment: {sentiment_info['combined_sentiment']:+.3f} → {sentiment_info['sentiment_label']}")

        # 5. Compute Market Mood and apply override
        print(f"\n5. Computing Market Mood Override...")
        mood = compute_market_mood(nifty_info, inr_info, sentiment_info)

        print(f"   Mood Score: {mood['mood_score']:+.3f} → {mood['mood_label']}")
        print(f"   Components: Nifty={mood['nifty_signal']:+.3f} | INR={mood['inr_signal']:+.3f} | Sentiment={mood['sentiment_signal']:+.3f}")
        print(f"   Override Action: {mood['override_action']}")
        if mood['override_reason']:
            print(f"   Reason: {mood['override_reason']}")

        # Save pre-override values for logging
        raw_direction = direction
        raw_confidence = confidence
        raw_change = predicted_pct_change

        # Apply the override
        direction, confidence, predicted_pct_change = apply_market_mood_override(
            direction, confidence, predicted_pct_change, magnitude, mood
        )

        # Log if anything changed
        if direction != raw_direction or abs(confidence - raw_confidence) > 0.001:
            print(f"\n   🔄 OVERRIDE APPLIED:")
            print(f"   Before: {raw_direction} ({raw_confidence*100:.1f}%) {raw_change:+.2f}%")
            print(f"   After:  {direction} ({confidence*100:.1f}%) {predicted_pct_change:+.2f}%")
        else:
            print(f"\n   ✓ No override needed - model prediction stands")

        # 6. Calculate price range
        uncertainty_factor = (1 - confidence) * 2
        range_pct = 0.5 + (uncertainty_factor * 0.75)

        predicted_price_mid = base_price_last * (1 + predicted_pct_change / 100)
        predicted_price_low = base_price_last * (1 + (predicted_pct_change - range_pct) / 100)
        predicted_price_high = base_price_last * (1 + (predicted_pct_change + range_pct) / 100)

        print(f"\n   FINAL PREDICTION:")
        print(f"   Current Price: ₹{base_price_last:.2f}")
        print(f"   Predicted Direction: {direction} ({confidence*100:.1f}% confidence)")
        print(f"   Predicted Change: {predicted_pct_change:+.2f}%")
        print(f"   Predicted Price Range: ₹{predicted_price_low:.2f} - ₹{predicted_price_high:.2f}")
        print(f"   Mid-Point: ₹{predicted_price_mid:.2f}")

        # Store result with override metadata
        prediction_entry = {
            'Stock': stock_name,
            'Ticker': ticker,
            'Current_Price': float(base_price_last),
            'Predicted_Direction': direction,
            'Direction_Confidence': float(confidence * 100),
            'Predicted_Change_Pct': float(predicted_pct_change),
            'Predicted_Price_Low': float(predicted_price_low),
            'Predicted_Price_Mid': float(predicted_price_mid),
            'Predicted_Price_High': float(predicted_price_high),
            'Range_Pct': float(range_pct),
            'Potential_Gain_Loss_Mid': float(predicted_price_mid - base_price_last),
            'Model_Type': 'V5_Transformer_Advanced Signals',
            'Features': int(data['num_features']),
            # Market Mood Override metadata
            'Market_Mood_Score': float(mood['mood_score']),
            'Market_Mood_Label': mood['mood_label'],
            'Override_Action': mood['override_action'],
            'Override_Reason': mood['override_reason'] if mood['override_reason'] else 'N/A',
            'Raw_Model_Direction': raw_direction,
            'Raw_Model_Confidence': float(raw_confidence * 100),
            'Raw_Model_Change_Pct': float(raw_change),
            'Bank_Nifty_Return_1d': float(nifty_info['nifty_return_1d']) if nifty_info else None,
            'Stock_Sentiment': float(sentiment_info['combined_sentiment']),
            'Stock_Sentiment_Label': sentiment_info['sentiment_label']
        }
        all_predictions.append(prediction_entry)

    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# SAVE AND DISPLAY PREDICTIONS
# =============================================================================

print("\n\n" + "=" * 80)
print("PREDICTIONS SUMMARY - NEXT TRADING DAY".center(80))
print("(with Market Mood Override: Bank Nifty + Sentiment + INR)".center(80))
print("=" * 80)

if all_predictions:
    predictions_df = pd.DataFrame(all_predictions)

    # Sort by predicted change (highest gains first)
    predictions_df = predictions_df.sort_values('Predicted_Change_Pct', ascending=False)

    # Display market mood context
    if nifty_info:
        nifty_emoji = "🔴" if nifty_info['nifty_return_1d'] < -0.5 else ("🟢" if nifty_info['nifty_return_1d'] > 0.5 else "🟡")
        print(f"\n{nifty_emoji} Bank Nifty: {nifty_info['nifty_return_1d']:+.2f}% ({nifty_info['nifty_label']})")
    if inr_info:
        inr_emoji = "🔴" if inr_info['inr_weakness'] > INR_WEAKNESS_THRESHOLD else ("🟢" if inr_info['inr_weakness'] < -0.001 else "🟡")
        print(f"{inr_emoji} INR: weakness {inr_info['inr_weakness']*100:+.3f}% ({inr_info['inr_label']})")

    # Show overrides summary
    overridden = predictions_df[predictions_df['Override_Action'] != 'NO_OVERRIDE']
    if len(overridden) > 0:
        print(f"\n⚡ Market Mood overrode {len(overridden)}/{len(predictions_df)} stock predictions:")
        for _, row in overridden.iterrows():
            print(f"   {row['Ticker']}: {row['Raw_Model_Direction']}→{row['Predicted_Direction']} ({row['Override_Action']})")

    # Display with price ranges and midpoint
    print(f"\n{'Stock':<20} {'Dir':<6} {'Conf%':<7} {'Change%':<8} {'Mood':<16} {'Override':<16} {'Raw':<6}")
    print("=" * 100)
    for _, row in predictions_df.iterrows():
        raw_dir_marker = "" if row['Predicted_Direction'] == row['Raw_Model_Direction'] else f"({row['Raw_Model_Direction']})"
        print(f"{row['Stock']:<20} {row['Predicted_Direction']:<6} "
              f"{row['Direction_Confidence']:<6.1f}% {row['Predicted_Change_Pct']:<+7.2f}% "
              f"{row['Market_Mood_Label']:<16} {row['Override_Action']:<16} {raw_dir_marker}")

    # Save to CSV
    output_file = 'future_predictions_next_day.csv'
    predictions_df.to_csv(output_file, index=False)

    # Save to JSON (main prediction file)
    output_json = 'future_predictions_next_day.json'
    with open(output_json, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Also save for web (same data, different location)
    web_json = 'web/future_predictions_next_day.json'
    with open(web_json, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'model_type': 'V5_Transformer_Advanced Signals + Market Mood Override',
        'features_per_stock': all_predictions[0]['Features'] if all_predictions else 0,
        'total_stocks': len(all_predictions),
        'avg_confidence': float(predictions_df['Direction_Confidence'].mean()),
        'high_confidence_count': int((predictions_df['Direction_Confidence'] > 70).sum()),
        'overrides_applied': int(len(overridden)) if len(overridden) > 0 else 0,
        'bank_nifty_return_1d': float(nifty_info['nifty_return_1d']) if nifty_info else None,
        'bank_nifty_label': nifty_info['nifty_label'] if nifty_info else None,
        'inr_weakness': float(inr_info['inr_weakness']) if inr_info else None,
        'market_mood_system': 'v1.0 (Bank Nifty 50% + INR 25% + Sentiment 25%)'
    }

    with open('prediction_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Predictions saved to:")
    print(f"  - {output_file}")
    print(f"  - {output_json}")
    print(f"  - {web_json}")
    print(f"  - prediction_metadata.json")

    # Trading recommendations
    print("\n" + "=" * 80)
    print("TRADING RECOMMENDATIONS WITH ENTRY/EXIT POINTS".center(80))
    print("(Market Mood Override Applied)".center(80))
    print("=" * 80)

    # Categorize stocks
    strong_buys = predictions_df[
        (predictions_df['Predicted_Direction'] == 'UP') &
        (predictions_df['Direction_Confidence'] > 60)
    ]

    moderate_buys = predictions_df[
        (predictions_df['Predicted_Direction'] == 'UP') &
        (predictions_df['Direction_Confidence'] >= 50) &
        (predictions_df['Direction_Confidence'] <= 60)
    ]

    strong_sells = predictions_df[
        (predictions_df['Predicted_Direction'] == 'DOWN') &
        (predictions_df['Direction_Confidence'] > 60)
    ]

    moderate_sells = predictions_df[
        (predictions_df['Predicted_Direction'] == 'DOWN') &
        (predictions_df['Direction_Confidence'] >= 50) &
        (predictions_df['Direction_Confidence'] <= 60)
    ]

    holds = predictions_df[
        (predictions_df['Direction_Confidence'] < 50)
    ]

    # Strong BUY signals
    if len(strong_buys) > 0:
        print("\n🟢 STRONG BUY (UP with >60% confidence):")
        print("-" * 80)
        for _, row in strong_buys.iterrows():
            override_note = f" [Override: {row['Override_Action']}]" if row['Override_Action'] != 'NO_OVERRIDE' else ""
            print(f"\n📈 {row['Stock']}{override_note}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Market Mood: {row['Market_Mood_Label']} | Sentiment: {row['Stock_Sentiment_Label']}")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 ENTRY POINT: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Current_Price']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f} (Gain: {((row['Predicted_Price_Mid']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🎯 TARGET (High): ₹{row['Predicted_Price_High']:.2f} (Gain: {((row['Predicted_Price_High']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_Low']*0.98:.2f} (2% below low)")

    # Moderate BUY signals
    if len(moderate_buys) > 0:
        print("\n🟡 MODERATE BUY (UP with 50-60% confidence):")
        print("-" * 80)
        for _, row in moderate_buys.iterrows():
            override_note = f" [Override: {row['Override_Action']}]" if row['Override_Action'] != 'NO_OVERRIDE' else ""
            print(f"\n📊 {row['Stock']}{override_note}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Market Mood: {row['Market_Mood_Label']} | Sentiment: {row['Stock_Sentiment_Label']}")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 ENTRY POINT: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Current_Price']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f}")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_Low']*0.98:.2f}")
            print(f"   ⚠️  Moderate confidence - use smaller position size")

    # Strong SELL signals
    if len(strong_sells) > 0:
        print("\n🔴 STRONG SELL (DOWN with >60% confidence):")
        print("-" * 80)
        for _, row in strong_sells.iterrows():
            override_note = f" [Override: {row['Override_Action']}]" if row['Override_Action'] != 'NO_OVERRIDE' else ""
            print(f"\n📉 {row['Stock']}{override_note}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Market Mood: {row['Market_Mood_Label']} | Sentiment: {row['Stock_Sentiment_Label']}")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 EXIT POINT: ₹{row['Current_Price']:.2f} - ₹{row['Predicted_Price_High']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f} (Loss: {((row['Predicted_Price_Mid']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🎯 TARGET (Low): ₹{row['Predicted_Price_Low']:.2f} (Loss: {((row['Predicted_Price_Low']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_High']*1.02:.2f} (2% above high)")
            print(f"   💡 ACTION: Avoid buying, exit long positions, consider shorting")

    # Moderate SELL signals
    if len(moderate_sells) > 0:
        print("\n🟠 MODERATE SELL (DOWN with 50-60% confidence):")
        print("-" * 80)
        for _, row in moderate_sells.iterrows():
            override_note = f" [Override: {row['Override_Action']}]" if row['Override_Action'] != 'NO_OVERRIDE' else ""
            print(f"\n📊 {row['Stock']}{override_note}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Market Mood: {row['Market_Mood_Label']} | Sentiment: {row['Stock_Sentiment_Label']}")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 EXIT POINT: ₹{row['Current_Price']:.2f} - ₹{row['Predicted_Price_High']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f}")
            print(f"   ⚠️  Moderate downward signal - reduce exposure")

    # HOLD signals
    if len(holds) > 0:
        print("\n⚪ HOLD / UNCERTAIN (<50% confidence):")
        print("-" * 80)
        for _, row in holds.iterrows():
            override_note = f" [Override: {row['Override_Action']}]" if row['Override_Action'] != 'NO_OVERRIDE' else ""
            print(f"\n⏸️  {row['Stock']}{override_note}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Market Mood: {row['Market_Mood_Label']} | Sentiment: {row['Stock_Sentiment_Label']}")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   ⚠️  Low confidence - model is uncertain")
            print(f"   💡 ACTION: Wait for clearer signal, avoid new positions")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY".center(80))
    print("=" * 80)
    print(f"\n🟢 Strong Buy:     {len(strong_buys)} stocks")
    print(f"🟡 Moderate Buy:   {len(moderate_buys)} stocks")
    print(f"🔴 Strong Sell:    {len(strong_sells)} stocks")
    print(f"🟠 Moderate Sell:  {len(moderate_sells)} stocks")
    print(f"⚪ Hold/Uncertain: {len(holds)} stocks")
    print(f"\n📊 Total Stocks:   {len(predictions_df)}")
    print(f"📈 Avg Confidence: {metadata['avg_confidence']:.1f}%")
    print(f"🎯 High Conf (>70%): {metadata['high_confidence_count']} stocks")
    print(f"⚡ Overrides Applied: {metadata['overrides_applied']}")

else:
    print("\n⚠️  No predictions generated")

print("\n" + "=" * 80)
print("✓ PREDICTIONS COMPLETE!".center(80))
print("=" * 80)

print("\nNote: These predictions use V5 Transformer models enhanced with Advanced Signals")
print("sentiment analysis, with a Market Mood Override system that combines:")
print("  • Bank Nifty index movement (50% weight - macro market direction)")
print("  • USD/INR forex weakness (25% weight - FII selling pressure)")
print("  • Stock-level FinBERT + Advanced Signals sentiment (25% weight)")
print("The override system prevents all-UP or all-DOWN predictions when macro")
print("conditions clearly contradict the model's output.")
print("Use these predictions as ONE input in your trading decisions.")
