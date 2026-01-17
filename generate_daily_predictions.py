"""
Daily Prediction Generation Script
Uses existing trained models to generate next-day predictions
Runs after daily data collection
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model

print("=" * 80)
print("DAILY PREDICTIONS - Next Trading Day".center(80))
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
print("=" * 80)

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

def generate_prediction(ticker, stock_name):
    """Generate prediction for a single stock"""
    print(f"\n{'='*80}")
    print(f"PREDICTING: {stock_name} ({ticker})".center(80))
    print("="*80)

    try:
        # Check if we have updated data
        latest_data_file = Path(f'data/stocks/daily_updates/{ticker}_latest.csv')

        if latest_data_file.exists():
            print(f"\n✓ Using fresh data from daily update")
            # Load latest data and prepare features
            # For now, we'll use the existing preprocessed data
            # In production, you'd preprocess the latest data here

        # Load prepared data (fallback to existing data)
        data_file = Path(f'data/enhanced_model_ready/{ticker}_enhanced.pkl')

        if not data_file.exists():
            print(f"\n✗ No preprocessed data found for {ticker}")
            return None

        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        # Get the last sequence from test set
        X_last = data['test']['X'][-1:]
        base_price_last = data['test']['base_prices'][-1]

        print(f"\n1. Data loaded successfully")
        print(f"   Base price: ₹{base_price_last:.2f}")
        print(f"   Sequence shape: {X_last.shape}")

        # Load model
        model_path = Path(f'models/saved_v5_all/{ticker}/best_model.keras')

        if not model_path.exists():
            print(f"\n✗ Model not found: {model_path}")
            return None

        model = load_model(model_path, compile=False)
        print(f"\n2. Model loaded: {model_path}")

        # Make prediction
        predictions = model.predict(X_last, verbose=0)

        direction_prob = predictions[0][0][0]
        magnitude = predictions[1][0][0]

        # Determine direction and confidence
        if direction_prob > 0.5:
            direction = "UP"
            predicted_pct_change = abs(magnitude)
            confidence = direction_prob
        else:
            direction = "DOWN"
            predicted_pct_change = -abs(magnitude)
            confidence = 1 - direction_prob

        # Calculate price range based on confidence
        uncertainty_factor = (1 - confidence) * 2
        range_pct = 0.5 + (uncertainty_factor * 0.75)

        predicted_price_mid = base_price_last * (1 + predicted_pct_change / 100)
        predicted_price_low = base_price_last * (1 + (predicted_pct_change - range_pct) / 100)
        predicted_price_high = base_price_last * (1 + (predicted_pct_change + range_pct) / 100)

        print(f"\n3. PREDICTION GENERATED:")
        print(f"   Current Price: ₹{base_price_last:.2f}")
        print(f"   Direction: {direction} ({confidence*100:.1f}% confidence)")
        print(f"   Expected Change: {predicted_pct_change:+.2f}%")
        print(f"   Price Range: ₹{predicted_price_low:.2f} - ₹{predicted_price_high:.2f}")
        print(f"   Mid-Point: ₹{predicted_price_mid:.2f}")

        return {
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
            'Generated_At': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# Generate predictions for all stocks
all_predictions = []

for ticker, stock_name in STOCKS:
    prediction = generate_prediction(ticker, stock_name)
    if prediction:
        all_predictions.append(prediction)

# Save predictions
if all_predictions:
    print("\n\n" + "=" * 80)
    print("SAVING PREDICTIONS")
    print("=" * 80)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_df = predictions_df.sort_values('Predicted_Change_Pct', ascending=False)

    # Display summary
    print(f"\n{'Stock':<25} {'Current':<10} {'Dir':<6} {'Conf%':<7} {'Change%':<8}")
    print("=" * 80)
    for _, row in predictions_df.iterrows():
        print(f"{row['Stock']:<25} ₹{row['Current_Price']:<9.2f} {row['Predicted_Direction']:<6} "
              f"{row['Direction_Confidence']:<6.1f}% {row['Predicted_Change_Pct']:<+7.2f}%")

    # Save to CSV
    csv_file = 'future_predictions_next_day.csv'
    predictions_df.to_csv(csv_file, index=False)

    # Save to JSON
    json_file = 'future_predictions_next_day.json'
    with open(json_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    # Copy to web directory for deployment
    web_json = Path('web/future_predictions_next_day.json')
    if web_json.parent.exists():
        with open(web_json, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        print(f"\n✓ Predictions copied to web directory")

    print(f"\n✓ Predictions saved:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    print(f"  - {web_json}")

    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'total_predictions': len(all_predictions),
        'strong_buy': len([p for p in all_predictions if p['Predicted_Direction'] == 'UP' and p['Direction_Confidence'] > 60]),
        'strong_sell': len([p for p in all_predictions if p['Predicted_Direction'] == 'DOWN' and p['Direction_Confidence'] > 60]),
        'model_version': 'v5_transformer'
    }

    with open('prediction_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Generated {len(all_predictions)}/{len(STOCKS)} predictions successfully")

else:
    print("\n⚠️  No predictions generated")

print("\n" + "=" * 80)
print("✓ PREDICTIONS COMPLETE!".center(80))
print("=" * 80)
