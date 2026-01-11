import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

print("=" * 80)
print("MULTI-STOCK ANALYSIS & FINAL RESULTS".center(80))
print("Hierarchical Attention Model - Stock Price Prediction".center(80))
print("=" * 80)

# Set style
sns.set_style('whitegrid')

# Load evaluation summary
with open('results/evaluation/evaluation_summary.json', 'r') as f:
    eval_data = json.load(f)

# Load training summary
with open('models/saved/training_summary_all_stocks.json', 'r') as f:
    train_data = json.load(f)

# Combine data
results = []
for eval_stock in eval_data['results']:
    ticker = eval_stock['ticker']
    
    # Find matching training data
    train_stock = next((s for s in train_data['stocks'] if s['ticker'] == ticker), None)
    
    result = {
        'Stock': eval_stock['stock'],
        'Ticker': ticker,
        'Sector': eval_stock['sector']
    }
    if train_stock:
        result.update(train_stock['best_metrics'])
    result.update(eval_stock)
    results.append(result)

df = pd.DataFrame(results)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. MAPE Comparison
ax = fig.add_subplot(gs[0, :2])
stocks = df['Stock'].str.replace(' ', '\n', 1)  # Wrap long names
x_pos = np.arange(len(df))
colors = ['#1f77b4' if s == 'Private Banks' else '#ff7f0e' for s in df['Sector']]
bars = ax.bar(x_pos, df['MAPE'], color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10% MAPE Threshold', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(stocks, rotation=0, ha='center', fontsize=9)
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance by Stock (Test Set MAPE)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Directional Accuracy
ax = fig.add_subplot(gs[0, 2])
ax.barh(df['Stock'], df['Directional_Accuracy'], color=colors, alpha=0.8, edgecolor='black')
ax.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Directional Accuracy', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 3. R² Score
ax = fig.add_subplot(gs[1, 0])
ax.barh(df['Stock'], df['R2'], color=colors, alpha=0.8, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax.set_title('Model Fit (R²)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 4. Training Epochs
ax = fig.add_subplot(gs[1, 1])
train_df = pd.DataFrame(train_data['stocks'])
ax.bar(range(len(train_df)), train_df['training_config'].apply(lambda x: x['epochs_run']), 
       color=colors, alpha=0.8, edgecolor='black')
ax.set_xticks(range(len(train_df)))
ax.set_xticklabels([s.replace(' ', '\n', 1) for s in train_df['stock']], rotation=0, ha='center', fontsize=9)
ax.set_ylabel('Epochs', fontsize=11, fontweight='bold')
ax.set_title('Training Epochs (Early Stopping)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 5. Validation vs Test Performance
ax = fig.add_subplot(gs[1, 2])
val_mape = train_df['best_metrics'].apply(lambda x: x['best_val_mape'])
test_mape = df['MAPE']
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, val_mape, width, label='Validation MAPE', alpha=0.8, color='skyblue', edgecolor='black')
ax.bar(x + width/2, test_mape, width, label='Test MAPE', alpha=0.8, color='coral', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels([s.replace(' ', '\n', 1) for s in df['Stock']], rotation=0, ha='center', fontsize=9)
ax.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
ax.set_title('Validation vs Test MAPE', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Sector Comparison
ax = fig.add_subplot(gs[2, 0])
sector_stats = df.groupby('Sector').agg({
    'MAPE': 'mean',
    'R2': 'mean',
    'Directional_Accuracy': 'mean'
}).round(2)

x_pos = np.arange(len(sector_stats))
width = 0.25
ax.bar(x_pos - width, sector_stats['MAPE'], width, label='MAPE (%)', alpha=0.8, color='#1f77b4')
ax.bar(x_pos, sector_stats['Directional_Accuracy'], width, label='Dir. Acc (%)', alpha=0.8, color='#ff7f0e')
ax.bar(x_pos + width, sector_stats['R2']*10, width, label='R² (×10)', alpha=0.8, color='#2ca02c')
ax.set_xticks(x_pos)
ax.set_xticklabels(sector_stats.index, fontsize=11)
ax.set_ylabel('Value', fontsize=11, fontweight='bold')
ax.set_title('Sector-Level Performance', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 7. Best vs Worst Performers
ax = fig.add_subplot(gs[2, 1])
top_3 = df.nsmallest(3, 'MAPE')[['Stock', 'MAPE']]
bottom_3 = df.nlargest(3, 'MAPE')[['Stock', 'MAPE']]

combined = pd.concat([top_3, bottom_3])
colors_perf = ['green']*3 + ['red']*3
ax.barh(combined['Stock'], combined['MAPE'], color=colors_perf, alpha=0.6, edgecolor='black')
ax.set_xlabel('MAPE (%)', fontsize=11, fontweight='bold')
ax.set_title('Best & Worst Performers', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 8. Summary Statistics
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = f"""
MODEL PERFORMANCE SUMMARY
═══════════════════════════════════

Overall Statistics:
  • Stocks Evaluated:  {len(df)}
  • Avg MAPE:          {df['MAPE'].mean():.2f}%
  • Avg R²:            {df['R2'].mean():.3f}
  • Avg Dir. Acc:      {df['Directional_Accuracy'].mean():.2f}%

Best Performers (MAPE):
  1. {df.nsmallest(1, 'MAPE').iloc[0]['Stock']}: {df.nsmallest(1, 'MAPE').iloc[0]['MAPE']:.2f}%
  2. {df.nsmallest(2, 'MAPE').iloc[1]['Stock']}: {df.nsmallest(2, 'MAPE').iloc[1]['MAPE']:.2f}%
  3. {df.nsmallest(3, 'MAPE').iloc[2]['Stock']}: {df.nsmallest(3, 'MAPE').iloc[2]['MAPE']:.2f}%

Sector Performance:
  • Private Banks:     {df[df['Sector']=='Private Banks']['MAPE'].mean():.2f}% MAPE
  • PSU Banks:         {df[df['Sector']=='PSU Banks']['MAPE'].mean():.2f}% MAPE

Model Architecture:
  • Total Parameters:  165,553
  • LSTM Units:        128, 64, 32
  • Attention Layers:  3
  • Training Time:     ~10 min/stock
"""

ax.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.suptitle('Hierarchical Attention Model - Comprehensive Results\nIndian Banking Stocks Price Prediction', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('results/final_results_summary.png', dpi=200, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved: results/final_results_summary.png")
plt.close()

# Create detailed results table
print("\n" + "=" * 80)
print("DETAILED RESULTS TABLE".center(80))
print("=" * 80)

print(f"\n{'Stock':<25} {'MAPE':<8} {'R²':<8} {'Dir.Acc':<10} {'RMSE':<10} {'Sector':<15}")
print("=" * 90)
for _, row in df.sort_values('MAPE').iterrows():
    print(f"{row['Stock']:<25} {row['MAPE']:<8.2f} {row['R2']:<8.3f} {row['Directional_Accuracy']:<10.2f} ₹{row['RMSE']:<9.2f} {row['Sector']:<15}")

print("\n" + "=" * 80)
print("FINAL SUMMARY".center(80))
print("=" * 80)

print(f"\n✅ Successfully trained and evaluated hierarchical attention models for 8 Indian banking stocks")
print(f"\n📊 Key Achievements:")
print(f"   • Average MAPE: {df['MAPE'].mean():.2f}% (Target: <10%)")
print(f"   • {len(df[df['MAPE'] < 10])} out of 8 stocks achieved MAPE < 10%")
print(f"   • Average Directional Accuracy: {df['Directional_Accuracy'].mean():.2f}%")
print(f"   • Best model: {df.nsmallest(1, 'MAPE').iloc[0]['Stock']} (MAPE: {df.nsmallest(1, 'MAPE').iloc[0]['MAPE']:.2f}%)")

print(f"\n📁 All Results Saved:")
print(f"   • Final summary: results/final_results_summary.png")
print(f"   • Trained models: models/saved/{{TICKER}}/best_model.keras")
print(f"   • Predictions: results/evaluation/{{TICKER}}_predictions.csv")
print(f"   • Attention visualizations: results/evaluation/attention/{{TICKER}}_attention.png")
print(f"   • Backtests: results/backtest/{{TICKER}}_backtest.png")

print("\n" + "=" * 80)
print("✓ COMPLETE: Hierarchical Attention Model Development".center(80))
print("=" * 80)
