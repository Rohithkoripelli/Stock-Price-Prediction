"""
Train V7 on all 8 stocks

Runs V7 (stratified + focal + temporal) on all bank stocks
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd

STOCKS = {
    'Private Banks': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK'],
    'PSU Banks': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK']
}

ALL_TICKERS = [ticker for sector in STOCKS.values() for ticker in sector]

print("=" * 80)
print("TRAINING V7 ON ALL 8 BANK STOCKS".center(80))
print("=" * 80)

print(f"\nStocks to train: {', '.join(ALL_TICKERS)}")
print(f"Total: {len(ALL_TICKERS)} stocks")

# =============================================================================
# TRAIN EACH STOCK
# =============================================================================

results_summary = []
overall_start = datetime.now()

for i, ticker in enumerate(ALL_TICKERS, 1):
    print(f"\n{'=' * 80}")
    print(f"TRAINING {i}/{len(ALL_TICKERS)}: {ticker}".center(80))
    print(f"{'=' * 80}\n")

    # Modify train_v7_final_push.py for this ticker
    with open('train_v7_final_push.py', 'r') as f:
        script = f.read()

    # Replace TICKER and STOCK_NAME
    stock_names = {
        'HDFCBANK': 'HDFC Bank',
        'ICICIBANK': 'ICICI Bank',
        'KOTAKBANK': 'Kotak Mahindra Bank',
        'AXISBANK': 'Axis Bank',
        'SBIN': 'State Bank of India',
        'PNB': 'Punjab National Bank',
        'BANKBARODA': 'Bank of Baroda',
        'CANBK': 'Canara Bank'
    }

    script_modified = script.replace("TICKER = 'HDFCBANK'", f"TICKER = '{ticker}'")
    script_modified = script_modified.replace("STOCK_NAME = 'HDFC Bank'", f"STOCK_NAME = '{stock_names[ticker]}'")

    # Save temporary script
    temp_script = f'temp_train_{ticker}.py'
    with open(temp_script, 'w') as f:
        f.write(script_modified)

    # Run training
    try:
        start_time = datetime.now()
        result = subprocess.run(
            ['python', temp_script],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes max per stock
        )

        train_time = (datetime.now() - start_time).total_seconds() / 60

        if result.returncode == 0:
            print(f"\n✓ {ticker} training completed in {train_time:.1f} minutes")

            # Load results
            results_file = f'models/saved_v7/{ticker}/results.json'
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    results_summary.append({
                        'ticker': ticker,
                        'stock': stock_names[ticker],
                        'overall_acc': results['test_metrics']['Directional_Accuracy_All'],
                        'significant_acc': results['test_metrics']['Directional_Accuracy_Significant'],
                        'large_acc': results['test_metrics'].get('Directional_Accuracy_Large', 0),
                        'rmse': results['test_metrics']['RMSE'],
                        'r2': results['test_metrics']['R2'],
                        'train_time_min': train_time
                    })
        else:
            print(f"\n✗ {ticker} training failed!")
            print(result.stderr[-500:])  # Last 500 chars of error

        # Clean up temp script
        os.remove(temp_script)

    except subprocess.TimeoutExpired:
        print(f"\n✗ {ticker} training timed out (>30 min)")
        os.remove(temp_script)
    except Exception as e:
        print(f"\n✗ {ticker} training error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)

total_time = (datetime.now() - overall_start).total_seconds() / 60

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING SUMMARY - ALL STOCKS".center(80))
print("=" * 80)

if results_summary:
    df = pd.DataFrame(results_summary)

    print(f"\n{'Stock':<25} {'Overall':<12} {'Significant':<12} {'Large':<12} {'RMSE':<10}")
    print("=" * 71)

    for sector, tickers in STOCKS.items():
        print(f"\n{sector}:")
        sector_results = df[df['ticker'].isin(tickers)]

        for _, row in sector_results.iterrows():
            print(f"{row['stock']:<25} {row['overall_acc']:>6.2f}%     "
                  f"{row['significant_acc']:>6.2f}%     "
                  f"{row['large_acc']:>6.2f}%     "
                  f"₹{row['rmse']:>7.2f}")

    print("\n" + "=" * 71)

    # Averages
    avg_overall = df['overall_acc'].mean()
    avg_significant = df['significant_acc'].mean()
    avg_large = df['large_acc'].mean()

    print(f"\n{'AVERAGE':<25} {avg_overall:>6.2f}%     "
          f"{avg_significant:>6.2f}%     "
          f"{avg_large:>6.2f}%")

    print(f"\n   Total training time: {total_time:.1f} minutes")
    print(f"   Average per stock: {total_time/len(results_summary):.1f} minutes")

    # Achievement check
    print("\n" + "=" * 80)
    if avg_overall >= 70:
        print("🎯 TARGET ACHIEVED! Average accuracy >= 70%".center(80))
    elif avg_overall >= 68:
        print("⭐ EXCELLENT! Very close to 70% target".center(80))
    elif avg_overall >= 65:
        print("✅ GOOD PROGRESS! Strong improvement from V5/V6".center(80))

    # Significant moves improvement
    if avg_significant >= 60:
        print("✅ SIGNIFICANT MOVES: Major improvement (>60%)!".center(80))
    elif avg_significant >= 55:
        print("📈 SIGNIFICANT MOVES: Good improvement!".center(80))

    print("=" * 80)

    # Save summary
    df.to_csv('models/saved_v7/all_stocks_summary.csv', index=False)

    # Save detailed JSON
    summary_json = {
        'training_date': datetime.now().isoformat(),
        'version': 'V7 - Stratified + Focal + Temporal',
        'total_stocks': len(results_summary),
        'total_time_minutes': total_time,
        'averages': {
            'overall_accuracy': float(avg_overall),
            'significant_moves_accuracy': float(avg_significant),
            'large_moves_accuracy': float(avg_large),
            'r2': float(df['r2'].mean())
        },
        'individual_stocks': results_summary,
        'improvements_vs_v5': {
            'v5_avg': 64.0,
            'v7_avg': float(avg_overall),
            'gain': float(avg_overall - 64.0)
        }
    }

    with open('models/saved_v7/all_stocks_detailed_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n✓ Summary saved to models/saved_v7/")

else:
    print("\n✗ No successful trainings to summarize")

print("\n" + "=" * 80)
print("✓ ALL STOCKS TRAINING COMPLETE!".center(80))
print("=" * 80)
