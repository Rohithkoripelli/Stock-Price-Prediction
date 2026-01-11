"""
Analyze Class Balance in Training Data

Check if there's an imbalance between UP and DOWN days
that could explain the model's bearish bias.
"""

import pickle
import numpy as np
import pandas as pd

print("=" * 80)
print("CLASS BALANCE ANALYSIS - TRAINING DATA".center(80))
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

all_stats = []

for ticker, stock_name in STOCKS:
    print(f"\n{'='*80}")
    print(f"{stock_name} ({ticker})".center(80))
    print("="*80)
    
    # Load data
    with open(f'data/enhanced_model_ready/{ticker}_enhanced.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Analyze each split
    for split_name in ['train', 'val', 'test']:
        y = data[split_name]['y']
        
        # Count UP and DOWN days
        up_days = np.sum(y > 0)
        down_days = np.sum(y < 0)
        neutral_days = np.sum(y == 0)
        total = len(y)
        
        up_pct = (up_days / total) * 100
        down_pct = (down_days / total) * 100
        neutral_pct = (neutral_days / total) * 100
        
        # Calculate imbalance ratio
        if down_days > 0:
            imbalance_ratio = up_days / down_days
        else:
            imbalance_ratio = float('inf')
        
        print(f"\n{split_name.upper()} SET ({total} samples):")
        print(f"  UP days:      {up_days:4d} ({up_pct:5.2f}%)")
        print(f"  DOWN days:    {down_days:4d} ({down_pct:5.2f}%)")
        print(f"  Neutral days: {neutral_days:4d} ({neutral_pct:5.2f}%)")
        print(f"  UP/DOWN ratio: {imbalance_ratio:.3f}")
        
        if split_name == 'train':
            all_stats.append({
                'Stock': stock_name,
                'Ticker': ticker,
                'Total_Samples': total,
                'UP_Days': up_days,
                'DOWN_Days': down_days,
                'Neutral_Days': neutral_days,
                'UP_Pct': up_pct,
                'DOWN_Pct': down_pct,
                'UP_DOWN_Ratio': imbalance_ratio
            })
        
        # Check for significant imbalance
        if abs(up_pct - down_pct) > 10:
            if up_pct > down_pct:
                print(f"  ⚠️  IMBALANCE: More UP days (+{up_pct - down_pct:.1f}%)")
            else:
                print(f"  ⚠️  IMBALANCE: More DOWN days (+{down_pct - up_pct:.1f}%)")

# Summary
print("\n\n" + "=" * 80)
print("TRAINING SET SUMMARY - ALL STOCKS".center(80))
print("=" * 80)

stats_df = pd.DataFrame(all_stats)

print(f"\n{stats_df[['Stock', 'Total_Samples', 'UP_Days', 'DOWN_Days', 'UP_Pct', 'DOWN_Pct', 'UP_DOWN_Ratio']].to_string(index=False)}")

# Overall statistics
total_up = stats_df['UP_Days'].sum()
total_down = stats_df['DOWN_Days'].sum()
total_samples = stats_df['Total_Samples'].sum()

overall_up_pct = (total_up / total_samples) * 100
overall_down_pct = (total_down / total_samples) * 100
overall_ratio = total_up / total_down if total_down > 0 else float('inf')

print("\n" + "=" * 80)
print("OVERALL STATISTICS (All Stocks Combined)".center(80))
print("=" * 80)

print(f"\nTotal Training Samples: {total_samples}")
print(f"Total UP days:          {total_up} ({overall_up_pct:.2f}%)")
print(f"Total DOWN days:        {total_down} ({overall_down_pct:.2f}%)")
print(f"Overall UP/DOWN ratio:  {overall_ratio:.3f}")

# Diagnosis
print("\n" + "=" * 80)
print("DIAGNOSIS".center(80))
print("=" * 80)

if abs(overall_up_pct - overall_down_pct) < 5:
    print("\n✅ BALANCED: UP and DOWN days are well balanced (within 5%)")
    print("   The model's bearish bias is NOT due to class imbalance.")
    print("\n   Possible causes:")
    print("   1. Model architecture bias")
    print("   2. Loss function weighting")
    print("   3. Recent market trends in test data")
    print("   4. Feature scaling issues")
elif overall_up_pct > overall_down_pct:
    imbalance = overall_up_pct - overall_down_pct
    print(f"\n⚠️  BULLISH BIAS in training data: {imbalance:.1f}% more UP days")
    print("   But model predicts DOWN - this is OPPOSITE of training data!")
    print("\n   Possible causes:")
    print("   1. Model learned to predict AGAINST the trend (contrarian)")
    print("   2. Overfitting to validation set which may be more bearish")
    print("   3. Test set has different characteristics than training")
else:
    imbalance = overall_down_pct - overall_up_pct
    print(f"\n⚠️  BEARISH BIAS in training data: {imbalance:.1f}% more DOWN days")
    print("   This EXPLAINS why the model predicts DOWN for all stocks!")
    print("\n   Solutions:")
    print("   1. Use class weights to balance training")
    print("   2. Oversample minority class (UP days)")
    print("   3. Undersample majority class (DOWN days)")
    print("   4. Use focal loss to focus on hard examples")

# Check individual stock imbalances
print("\n" + "=" * 80)
print("STOCKS WITH SIGNIFICANT IMBALANCE (>10%)".center(80))
print("=" * 80)

imbalanced_stocks = stats_df[abs(stats_df['UP_Pct'] - stats_df['DOWN_Pct']) > 10]

if len(imbalanced_stocks) > 0:
    print("\n")
    for _, row in imbalanced_stocks.iterrows():
        diff = row['UP_Pct'] - row['DOWN_Pct']
        if diff > 0:
            print(f"{row['Stock']}: {diff:+.1f}% more UP days")
        else:
            print(f"{row['Stock']}: {abs(diff):+.1f}% more DOWN days")
else:
    print("\n✅ No stocks have significant imbalance (all within 10%)")

# Save analysis
stats_df.to_csv('class_balance_analysis.csv', index=False)
print(f"\n✓ Analysis saved to: class_balance_analysis.csv")

print("\n" + "=" * 80)
print("✓ ANALYSIS COMPLETE!".center(80))
print("=" * 80)
