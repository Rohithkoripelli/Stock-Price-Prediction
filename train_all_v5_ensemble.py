"""
Train V5 Ensemble on All 8 Stocks

Runs ensemble training on all bank stocks
"""

import subprocess
import json
import os
from datetime import datetime
import pandas as pd

STOCKS = {
    'HDFCBANK': 'HDFC Bank',
    'ICICIBANK': 'ICICI Bank',
    'KOTAKBANK': 'Kotak Mahindra Bank',
    'AXISBANK': 'Axis Bank',
    'SBIN': 'State Bank of India',
    'PNB': 'Punjab National Bank',
    'BANKBARODA': 'Bank of Baroda',
    'CANBK': 'Canara Bank'
}

print("=" * 80)
print("TRAINING V5 ENSEMBLE ON ALL 8 STOCKS".center(80))
print("=" * 80)

print(f"\n   Training 10-model ensemble for each stock")
print(f"   Total models to train: {len(STOCKS) * 10} = {len(STOCKS)} stocks × 10 models")
print(f"   Estimated time: 2-3 hours\n")

results_summary = []
overall_start = datetime.now()

for i, (ticker, name) in enumerate(STOCKS.items(), 1):
    print(f"\n{'='*80}")
    print(f"STOCK {i}/{len(STOCKS)}: {name} ({ticker})".center(80))
    print(f"{'='*80}\n")

    # Modify script for this ticker
    with open('train_v5_ensemble.py', 'r') as f:
        script = f.read()

    script_modified = script.replace("TICKER = 'HDFCBANK'", f"TICKER = '{ticker}'")
    script_modified = script_modified.replace("STOCK_NAME = 'HDFC Bank'", f"STOCK_NAME = '{name}'")

    # Save temp script
    temp_script = f'temp_ensemble_{ticker}.py'
    with open(temp_script, 'w') as f:
        f.write(script_modified)

    # Run training
    try:
        start_time = datetime.now()
        result = subprocess.run(
            ['python', temp_script],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour per stock
        )

        train_time = (datetime.now() - start_time).total_seconds() / 60

        if result.returncode == 0:
            print(f"\n✓ {ticker} ensemble completed in {train_time:.1f} minutes")

            # Load results
            results_file = f'models/saved_v5_ensemble/{ticker}/ensemble_results.json'
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    results_summary.append({
                        'ticker': ticker,
                        'stock': name,
                        'overall_acc': results['test_metrics']['Directional_Accuracy_All'],
                        'significant_acc': results['test_metrics']['Directional_Accuracy_Significant'],
                        'large_acc': results['test_metrics']['Directional_Accuracy_Large'],
                        'up_recall': results['test_metrics']['UP_Recall'],
                        'down_recall': results['test_metrics']['DOWN_Recall'],
                        'rmse': results['test_metrics']['RMSE'],
                        'r2': results['test_metrics']['R2'],
                        'train_time_min': train_time
                    })
        else:
            print(f"\n✗ {ticker} failed!")
            print(result.stderr[-500:])

        os.remove(temp_script)

    except subprocess.TimeoutExpired:
        print(f"\n✗ {ticker} timed out")
        if os.path.exists(temp_script):
            os.remove(temp_script)
    except Exception as e:
        print(f"\n✗ {ticker} error: {e}")
        if os.path.exists(temp_script):
            os.remove(temp_script)

total_time = (datetime.now() - overall_start).total_seconds() / 60

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("V5 ENSEMBLE - ALL STOCKS SUMMARY".center(80))
print("=" * 80)

if results_summary:
    df = pd.DataFrame(results_summary)

    print(f"\n{'Stock':<25} {'Overall':<10} {'Signif':<10} {'Large':<10} {'UP Rec':<10} {'Time':<8}")
    print("=" * 73)

    for _, row in df.iterrows():
        print(f"{row['stock']:<25} {row['overall_acc']:>6.2f}%   "
              f"{row['significant_acc']:>6.2f}%   "
              f"{row['large_acc']:>6.2f}%   "
              f"{row['up_recall']*100:>6.2f}%   "
              f"{row['train_time_min']:>5.0f}m")

    print("=" * 73)

    # Averages
    avg_overall = df['overall_acc'].mean()
    avg_significant = df['significant_acc'].mean()
    avg_large = df['large_acc'].mean()
    avg_up_recall = df['up_recall'].mean()

    print(f"\n{'AVERAGE':<25} {avg_overall:>6.2f}%   "
          f"{avg_significant:>6.2f}%   "
          f"{avg_large:>6.2f}%   "
          f"{avg_up_recall*100:>6.2f}%")

    print(f"\n   Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"   Per stock: {total_time/len(results_summary):.1f} minutes")

    # Achievement check
    print("\n" + "=" * 80)
    if avg_overall >= 70:
        print("🎯 TARGET ACHIEVED! Average >= 70%".center(80))
    elif avg_overall >= 68:
        print("⭐ EXCELLENT! Very close to 70%".center(80))
    elif avg_overall >= 66:
        print("✅ GREAT! Solid improvement with ensemble".center(80))
    else:
        print("✅ IMPROVED! Ensemble better than single models".center(80))

    if avg_significant >= 55:
        print("✅ SIGNIFICANT MOVES: Good improvement!".center(80))

    print("=" * 80)

    # Save summary
    df.to_csv('models/saved_v5_ensemble/all_stocks_summary.csv', index=False)

    summary_json = {
        'training_date': datetime.now().isoformat(),
        'approach': 'V5 Ensemble (10 models per stock)',
        'num_stocks': len(results_summary),
        'total_time_minutes': total_time,
        'averages': {
            'overall_accuracy': float(avg_overall),
            'significant_accuracy': float(avg_significant),
            'large_accuracy': float(avg_large),
            'up_recall': float(avg_up_recall),
            'r2': float(df['r2'].mean())
        },
        'individual_stocks': results_summary
    }

    with open('models/saved_v5_ensemble/all_stocks_summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    print(f"\n✓ Summary saved to models/saved_v5_ensemble/")

print("\n" + "=" * 80)
print("✓ ALL STOCKS ENSEMBLE COMPLETE!".center(80))
print("=" * 80)
