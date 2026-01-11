"""
Check Test Period Market Performance

Analyze if the test period is actually bearish to see if
the model's DOWN predictions are justified.
"""

import pickle
import numpy as np
import pandas as pd

print("=" * 80)
print("TEST PERIOD MARKET ANALYSIS".center(80))
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

all_test_stats = []

for ticker, stock_name in STOCKS:
    # Load data
    with open(f'data/enhanced_model_ready/{ticker}_enhanced.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Test set analysis
    y_test = data['test']['y']
    base_prices = data['test']['base_prices']
    dates = data['test']['dates']
    
    # Calculate statistics
    up_days = np.sum(y_test > 0)
    down_days = np.sum(y_test < 0)
    neutral_days = np.sum(y_test == 0)
    total = len(y_test)
    
    up_pct = (up_days / total) * 100
    down_pct = (down_days / total) * 100
    
    # Overall performance
    start_price = base_prices[0]
    end_price = base_prices[-1] * (1 + y_test[-1] / 100)
    total_return = ((end_price - start_price) / start_price) * 100
    
    # Average daily return
    avg_daily_return = np.mean(y_test)
    
    # Volatility
    volatility = np.std(y_test)
    
    print(f"\n{stock_name} ({ticker}):")
    print(f"  Test Period: {dates[0]} to {dates[-1]}")
    print(f"  UP days:     {up_days} ({up_pct:.1f}%)")
    print(f"  DOWN days:   {down_days} ({down_pct:.1f}%)")
    print(f"  Start Price: ₹{start_price:.2f}")
    print(f"  End Price:   ₹{end_price:.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Avg Daily Return: {avg_daily_return:+.3f}%")
    print(f"  Volatility: {volatility:.3f}%")
    
    if total_return < -5:
        print(f"  📉 BEARISH period (>5% loss)")
    elif total_return > 5:
        print(f"  📈 BULLISH period (>5% gain)")
    else:
        print(f"  ➡️  SIDEWAYS period")
    
    all_test_stats.append({
        'Stock': stock_name,
        'Ticker': ticker,
        'UP_Days_Pct': up_pct,
        'DOWN_Days_Pct': down_pct,
        'Total_Return': total_return,
        'Avg_Daily_Return': avg_daily_return,
        'Volatility': volatility,
        'Period_Start': dates[0],
        'Period_End': dates[-1]
    })

# Overall summary
print("\n" + "=" * 80)
print("OVERALL TEST PERIOD SUMMARY".center(80))
print("=" * 80)

stats_df = pd.DataFrame(all_test_stats)

avg_up_pct = stats_df['UP_Days_Pct'].mean()
avg_down_pct = stats_df['DOWN_Days_Pct'].mean()
avg_return = stats_df['Total_Return'].mean()
avg_daily_return = stats_df['Avg_Daily_Return'].mean()

print(f"\nAverage UP days:        {avg_up_pct:.1f}%")
print(f"Average DOWN days:      {avg_down_pct:.1f}%")
print(f"Average Total Return:   {avg_return:+.2f}%")
print(f"Average Daily Return:   {avg_daily_return:+.3f}%")

# Diagnosis
print("\n" + "=" * 80)
print("DIAGNOSIS".center(80))
print("=" * 80)

if avg_return < -2:
    print(f"\n📉 TEST PERIOD WAS BEARISH: Average return {avg_return:.2f}%")
    print("   The model's DOWN predictions may be JUSTIFIED!")
    print("   The model learned to predict the bearish trend.")
elif avg_return > 2:
    print(f"\n📈 TEST PERIOD WAS BULLISH: Average return {avg_return:.2f}%")
    print("   The model's DOWN predictions are WRONG!")
    print("   The model has a bearish bias despite bullish test period.")
else:
    print(f"\n➡️  TEST PERIOD WAS SIDEWAYS: Average return {avg_return:.2f}%")
    print("   Market was range-bound with no clear trend.")
    
if avg_up_pct > avg_down_pct:
    print(f"\n   More UP days ({avg_up_pct:.1f}%) than DOWN days ({avg_down_pct:.1f}%)")
    print("   But model predicts DOWN - this suggests:")
    print("   1. Model is too conservative")
    print("   2. Model learned to predict against short-term noise")
    print("   3. Model needs retraining with better loss function")
else:
    print(f"\n   More DOWN days ({avg_down_pct:.1f}%) than UP days ({avg_up_pct:.1f}%)")
    print("   Model's DOWN predictions align with test period characteristics")

# Save analysis
stats_df.to_csv('test_period_analysis.csv', index=False)
print(f"\n✓ Analysis saved to: test_period_analysis.csv")

print("\n" + "=" * 80)
print("✓ ANALYSIS COMPLETE!".center(80))
print("=" * 80)
