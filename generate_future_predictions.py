"""
Simple Future Predictions - Next Trading Day

Uses the last available data sequence to predict the next trading day
for all 8 stocks using trained V5 Transformer models.
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model

print("=" * 80)
print("FUTURE PREDICTIONS - NEXT TRADING DAY".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

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

# =============================================================================
# GENERATE PREDICTIONS
# =============================================================================

all_predictions = []

for ticker, stock_name in STOCKS:
    print(f"\n{'='*80}")
    print(f"PREDICTING: {stock_name} ({ticker})".center(80))
    print("="*80)
    
    try:
        # 1. Load prepared data
        print(f"\n1. Loading prepared data...")
        with open(f'data/enhanced_model_ready/{ticker}_enhanced.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Get the last sequence from test set (most recent data)
        X_last = data['test']['X'][-1:]  # Shape: (1, 60, 35)
        base_price_last = data['test']['base_prices'][-1]
        
        print(f"   ✓ Loaded last sequence")
        print(f"   ✓ Base price: ₹{base_price_last:.2f}")
        print(f"   ✓ Sequence shape: {X_last.shape}")
        
        # 2. Load model
        print(f"\n2. Loading V5 Transformer model...")
        model = load_model(
            f'models/saved_v5_all/{ticker}/best_model.keras',
            compile=False
        )
        print(f"   ✓ Model loaded")
        
        # 3. Make prediction
        print(f"\n3. Making prediction...")
        predictions = model.predict(X_last, verbose=0)
        
        direction_prob = predictions[0][0][0]
        magnitude = predictions[1][0][0]
        
        # Direction is UP if probability > 0.5, otherwise DOWN
        # The magnitude should be positive for UP, negative for DOWN
        if direction_prob > 0.5:
            direction = "UP"
            predicted_pct_change = abs(magnitude)  # Ensure positive for UP
            confidence = direction_prob
        else:
            direction = "DOWN"
            predicted_pct_change = -abs(magnitude)  # Ensure negative for DOWN
            confidence = 1 - direction_prob  # Confidence in DOWN prediction
        
        # Calculate uncertainty based on confidence
        # Lower confidence = wider range
        uncertainty_factor = (1 - confidence) * 2  # 0 to 2
        
        # Calculate price range
        # Base range: ±0.5% for high confidence, up to ±2% for low confidence
        range_pct = 0.5 + (uncertainty_factor * 0.75)  # 0.5% to 2%
        
        predicted_price_mid = base_price_last * (1 + predicted_pct_change / 100)
        predicted_price_low = base_price_last * (1 + (predicted_pct_change - range_pct) / 100)
        predicted_price_high = base_price_last * (1 + (predicted_pct_change + range_pct) / 100)
        
        print(f"\n   PREDICTION:")
        print(f"   Current Price: ₹{base_price_last:.2f}")
        print(f"   Predicted Direction: {direction} ({confidence*100:.1f}% confidence)")
        print(f"   Predicted Change: {predicted_pct_change:+.2f}%")
        print(f"   Predicted Price Range: ₹{predicted_price_low:.2f} - ₹{predicted_price_high:.2f}")
        print(f"   Mid-Point: ₹{predicted_price_mid:.2f}")
        print(f"   Range Width: ±{range_pct:.2f}%")
        
        # Store result
        all_predictions.append({
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
            'Potential_Gain_Loss_Mid': float(predicted_price_mid - base_price_last)
        })
        
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# SAVE AND DISPLAY PREDICTIONS
# =============================================================================

print("\n\n" + "=" * 80)
print("FUTURE PREDICTIONS SUMMARY - NEXT TRADING DAY".center(80))
print("=" * 80)

if all_predictions:
    predictions_df = pd.DataFrame(all_predictions)
    
    # Sort by predicted change (highest gains first)
    predictions_df = predictions_df.sort_values('Predicted_Change_Pct', ascending=False)
    
    # Display with price ranges and midpoint
    print(f"\n{'Stock':<25} {'Current':<10} {'Dir':<6} {'Conf%':<7} {'Change%':<8} {'Low':<10} {'Mid':<10} {'High':<10}")
    print("=" * 105)
    for _, row in predictions_df.iterrows():
        print(f"{row['Stock']:<25} ₹{row['Current_Price']:<9.2f} {row['Predicted_Direction']:<6} "
              f"{row['Direction_Confidence']:<6.1f}% {row['Predicted_Change_Pct']:<+7.2f}% "
              f"₹{row['Predicted_Price_Low']:<9.2f} ₹{row['Predicted_Price_Mid']:<9.2f} ₹{row['Predicted_Price_High']:<9.2f}")
    
    # Save to CSV
    output_file = 'future_predictions_next_day.csv'
    predictions_df.to_csv(output_file, index=False)
    
    # Save to JSON
    output_json = 'future_predictions_next_day.json'
    with open(output_json, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\n✓ Predictions saved to:")
    print(f"  - {output_file}")
    print(f"  - {output_json}")
    
    # Trading recommendations
    print("\n" + "=" * 80)
    print("TRADING RECOMMENDATIONS WITH ENTRY/EXIT POINTS".center(80))
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
            print(f"\n📈 {row['Stock']}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 ENTRY POINT: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Current_Price']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f} (Gain: {((row['Predicted_Price_Mid']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🎯 TARGET (High): ₹{row['Predicted_Price_High']:.2f} (Gain: {((row['Predicted_Price_High']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_Low']*0.98:.2f} (2% below low)")
            print(f"   ✅ JUSTIFICATION: High confidence ({row['Direction_Confidence']:.1f}%) upward prediction")
    
    # Moderate BUY signals
    if len(moderate_buys) > 0:
        print("\n🟡 MODERATE BUY (UP with 50-60% confidence):")
        print("-" * 80)
        for _, row in moderate_buys.iterrows():
            print(f"\n📊 {row['Stock']}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 ENTRY POINT: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Current_Price']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f}")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_Low']*0.98:.2f}")
            print(f"   ⚠️  JUSTIFICATION: Moderate confidence - use smaller position size")
    
    # Strong SELL signals
    if len(strong_sells) > 0:
        print("\n🔴 STRONG SELL (DOWN with >60% confidence):")
        print("-" * 80)
        for _, row in strong_sells.iterrows():
            print(f"\n📉 {row['Stock']}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 EXIT POINT: ₹{row['Current_Price']:.2f} - ₹{row['Predicted_Price_High']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f} (Loss: {((row['Predicted_Price_Mid']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🎯 TARGET (Low): ₹{row['Predicted_Price_Low']:.2f} (Loss: {((row['Predicted_Price_Low']/row['Current_Price'])-1)*100:+.2f}%)")
            print(f"   🛑 STOP LOSS: ₹{row['Predicted_Price_High']*1.02:.2f} (2% above high)")
            print(f"   ✅ JUSTIFICATION: High confidence ({row['Direction_Confidence']:.1f}%) downward prediction")
            print(f"   💡 ACTION: Avoid buying, exit long positions, consider shorting")
    
    # Moderate SELL signals
    if len(moderate_sells) > 0:
        print("\n🟠 MODERATE SELL (DOWN with 50-60% confidence):")
        print("-" * 80)
        for _, row in moderate_sells.iterrows():
            print(f"\n📊 {row['Stock']}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   📍 EXIT POINT: ₹{row['Current_Price']:.2f} - ₹{row['Predicted_Price_High']:.2f}")
            print(f"   🎯 TARGET (Mid): ₹{row['Predicted_Price_Mid']:.2f}")
            print(f"   ⚠️  JUSTIFICATION: Moderate downward signal - reduce exposure")
    
    # HOLD signals
    if len(holds) > 0:
        print("\n⚪ HOLD / UNCERTAIN (<50% confidence):")
        print("-" * 80)
        for _, row in holds.iterrows():
            print(f"\n⏸️  {row['Stock']}")
            print(f"   Current Price: ₹{row['Current_Price']:.2f}")
            print(f"   Prediction: {row['Predicted_Direction']} {row['Predicted_Change_Pct']:+.2f}% ({row['Direction_Confidence']:.1f}% confidence)")
            print(f"   Target Range: ₹{row['Predicted_Price_Low']:.2f} - ₹{row['Predicted_Price_High']:.2f} (Mid: ₹{row['Predicted_Price_Mid']:.2f})")
            print(f"   ⚠️  JUSTIFICATION: Low confidence - model is uncertain")
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

else:
    print("\n⚠️  No predictions generated")

print("\n" + "=" * 80)
print("✓ PREDICTIONS COMPLETE!".center(80))
print("=" * 80)

print("\nNote: These predictions are based on the V5 Transformer models trained")
print("on historical data. Past performance does not guarantee future results.")
print("Use these predictions as ONE input in your trading decisions.")
