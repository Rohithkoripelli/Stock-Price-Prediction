"""
Compare VADER-based predictions vs FinBERT-enhanced predictions

This script compares the old system (VADER sentiment) with the new system
(FinBERT sentiment) to demonstrate improvements in confidence and accuracy.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("COMPARING VADER vs FINBERT PREDICTIONS".center(80))
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

# Load old VADER results
print("\n1. Loading VADER-based results...")
vader_results = []

if os.path.exists('models/saved_v5_all/all_stocks_summary.json'):
    with open('models/saved_v5_all/all_stocks_summary.json', 'r') as f:
        vader_summary = json.load(f)
        vader_results = vader_summary.get('individual_results', [])
    print(f"   ✓ Loaded VADER results for {len(vader_results)} stocks")
else:
    print("   ⚠ VADER results not found (models/saved_v5_all/all_stocks_summary.json)")

# Load new FinBERT results
print("\n2. Loading FinBERT-enhanced results...")
finbert_results = []

if os.path.exists('models/saved_v5_finbert/all_stocks_summary.json'):
    with open('models/saved_v5_finbert/all_stocks_summary.json', 'r') as f:
        finbert_summary = json.load(f)
        finbert_results = finbert_summary.get('individual_results', [])
    print(f"   ✓ Loaded FinBERT results for {len(finbert_results)} stocks")
else:
    print("   ⚠ FinBERT results not found (models/saved_v5_finbert/all_stocks_summary.json)")

# Compare results
print("\n3. Comparing results...")

comparison_data = []

for ticker, stock_name in STOCKS:
    # Find VADER result
    vader_result = next((r for r in vader_results if r['ticker'] == ticker), None)

    # Find FinBERT result
    finbert_result = next((r for r in finbert_results if r['ticker'] == ticker), None)

    if vader_result and finbert_result:
        vader_dir_acc = vader_result['test_metrics'].get('Directional_Accuracy', 0)

        finbert_dir_acc = finbert_result['test_metrics'].get('Directional_Accuracy', 0)
        finbert_conf = finbert_result['test_metrics'].get('Average_Confidence', 0)

        # Calculate improvement
        dir_acc_improvement = finbert_dir_acc - vader_dir_acc

        comparison_data.append({
            'Stock': stock_name,
            'Ticker': ticker,
            'VADER_Dir_Acc': vader_dir_acc,
            'FinBERT_Dir_Acc': finbert_dir_acc,
            'Dir_Acc_Improvement': dir_acc_improvement,
            'FinBERT_Avg_Confidence': finbert_conf,
            'VADER_Features': vader_result.get('n_features', 0),
            'FinBERT_Features': finbert_result.get('n_features', 0)
        })

if comparison_data:
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Display results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS".center(80))
    print("=" * 80)

    print(f"\n{'Stock':<25} {'VADER':<10} {'FinBERT':<10} {'Improve':<10} {'Confidence':<12}")
    print("=" * 80)

    for _, row in comparison_df.iterrows():
        improvement_sign = "+" if row['Dir_Acc_Improvement'] >= 0 else ""
        print(f"{row['Stock']:<25} "
              f"{row['VADER_Dir_Acc']:>8.2f}% "
              f"{row['FinBERT_Dir_Acc']:>8.2f}% "
              f"{improvement_sign}{row['Dir_Acc_Improvement']:>8.2f}% "
              f"{row['FinBERT_Avg_Confidence']:>10.2f}%")

    print("=" * 80)

    # Calculate summary statistics
    avg_vader_acc = comparison_df['VADER_Dir_Acc'].mean()
    avg_finbert_acc = comparison_df['FinBERT_Dir_Acc'].mean()
    avg_improvement = comparison_df['Dir_Acc_Improvement'].mean()
    avg_confidence = comparison_df['FinBERT_Avg_Confidence'].mean()

    stocks_improved = len(comparison_df[comparison_df['Dir_Acc_Improvement'] > 0])
    stocks_degraded = len(comparison_df[comparison_df['Dir_Acc_Improvement'] < 0])

    print(f"\n{'AVERAGE':<25} {avg_vader_acc:>8.2f}% {avg_finbert_acc:>8.2f}% "
          f"{'+' if avg_improvement >= 0 else ''}{avg_improvement:>8.2f}% {avg_confidence:>10.2f}%")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS".center(80))
    print("=" * 80)

    print(f"\nStocks with improved accuracy: {stocks_improved}/{len(comparison_df)}")
    print(f"Stocks with degraded accuracy: {stocks_degraded}/{len(comparison_df)}")
    print(f"\nAverage directional accuracy improvement: {'+' if avg_improvement >= 0 else ''}{avg_improvement:.2f}%")
    print(f"Average confidence score: {avg_confidence:.2f}%")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS".center(80))
    print("=" * 80)

    best_improvement = comparison_df.loc[comparison_df['Dir_Acc_Improvement'].idxmax()]
    worst_improvement = comparison_df.loc[comparison_df['Dir_Acc_Improvement'].idxmin()]

    print(f"\nBest improvement: {best_improvement['Stock']}")
    print(f"  VADER: {best_improvement['VADER_Dir_Acc']:.2f}% → FinBERT: {best_improvement['FinBERT_Dir_Acc']:.2f}%")
    print(f"  Improvement: +{best_improvement['Dir_Acc_Improvement']:.2f}%")

    if worst_improvement['Dir_Acc_Improvement'] < 0:
        print(f"\nWorst performance: {worst_improvement['Stock']}")
        print(f"  VADER: {worst_improvement['VADER_Dir_Acc']:.2f}% → FinBERT: {worst_improvement['FinBERT_Dir_Acc']:.2f}%")
        print(f"  Degradation: {worst_improvement['Dir_Acc_Improvement']:.2f}%")

    highest_conf = comparison_df.loc[comparison_df['FinBERT_Avg_Confidence'].idxmax()]
    print(f"\nHighest confidence: {highest_conf['Stock']}")
    print(f"  Average confidence: {highest_conf['FinBERT_Avg_Confidence']:.2f}%")
    print(f"  Directional accuracy: {highest_conf['FinBERT_Dir_Acc']:.2f}%")

    # Feature comparison
    print(f"\nFeature count:")
    print(f"  VADER models: {comparison_df['VADER_Features'].iloc[0]} features (technical only)")
    print(f"  FinBERT models: {comparison_df['FinBERT_Features'].iloc[0]} features (technical + sentiment)")

    # Save comparison
    output_file = 'prediction_comparison.csv'
    comparison_df.to_csv(output_file, index=False)
    print(f"\n✓ Comparison saved to: {output_file}")

    # Save summary JSON
    summary = {
        'comparison_date': datetime.now().isoformat(),
        'stocks_compared': len(comparison_df),
        'vader_system': {
            'average_directional_accuracy': float(avg_vader_acc),
            'sentiment_method': 'VADER (general-purpose)'
        },
        'finbert_system': {
            'average_directional_accuracy': float(avg_finbert_acc),
            'average_confidence': float(avg_confidence),
            'sentiment_method': 'FinBERT (financial domain-specific)'
        },
        'improvements': {
            'average_accuracy_improvement': float(avg_improvement),
            'stocks_improved': int(stocks_improved),
            'stocks_degraded': int(stocks_degraded)
        },
        'best_improvement': {
            'stock': best_improvement['Stock'],
            'ticker': best_improvement['Ticker'],
            'improvement_pct': float(best_improvement['Dir_Acc_Improvement'])
        }
    }

    with open('prediction_comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: prediction_comparison_summary.json")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION".center(80))
    print("=" * 80)

    if avg_improvement > 2 and avg_confidence > 65:
        print("\n✓ FinBERT models show significant improvement!")
        print(f"  - Directional accuracy improved by {avg_improvement:.2f}%")
        print(f"  - Average confidence is {avg_confidence:.2f}%")
        print("\nRecommended actions:")
        print("  1. Upload FinBERT models to HuggingFace")
        print("  2. Update GitHub Actions to use FinBERT models")
        print("  3. Deploy to production")
    elif avg_improvement > 0:
        print("\n~ FinBERT models show modest improvement")
        print(f"  - Directional accuracy improved by {avg_improvement:.2f}%")
        print(f"  - Average confidence is {avg_confidence:.2f}%")
        print("\nConsider:")
        print("  - Collecting more news data (extend beyond 30 days)")
        print("  - Fine-tuning FinBERT on Indian banking news")
        print("  - Increasing training epochs")
    else:
        print("\n⚠ FinBERT models did not improve performance")
        print(f"  - Directional accuracy changed by {avg_improvement:.2f}%")
        print("\nPossible causes:")
        print("  - Insufficient news data")
        print("  - Need longer training")
        print("  - VADER may be sufficient for this use case")

else:
    print("\n✗ No comparison data available")
    print("Make sure both VADER and FinBERT models have been trained")

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE".center(80))
print("=" * 80)
