import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# Create data directories
os.makedirs('data/stocks/private_banks', exist_ok=True)
os.makedirs('data/stocks/psu_banks', exist_ok=True)
os.makedirs('data/summary', exist_ok=True)

# =============================================================================
# INDIAN BANKING STOCKS - PRIVATE & PSU
# =============================================================================

banking_stocks = {
    'Private Banks': {
        'HDFC Bank': 'HDFCBANK.NS',
        'ICICI Bank': 'ICICIBANK.NS',
        'Kotak Mahindra Bank': 'KOTAKBANK.NS',
        'Axis Bank': 'AXISBANK.NS'
    },
    
    'PSU Banks': {
        'State Bank of India': 'SBIN.NS',
        'Punjab National Bank': 'PNB.NS',
        'Bank of Baroda': 'BANKBARODA.NS',
        'Canara Bank': 'CANBK.NS'
    }
}

# Download parameters
START_DATE = '2019-01-01'  # 5+ years of data
# Auto-fetch until yesterday (T-1)
# Note: yfinance end date is EXCLUSIVE, so we add 1 day to include yesterday
from datetime import timedelta
END_DATE = datetime.now().strftime('%Y-%m-%d')  # Today's date (to include yesterday's data)

print("=" * 80)
print("COLLECTING INDIAN BANKING STOCK DATA".center(80))
print("Private Banks: HDFC, ICICI, Kotak, Axis".center(80))
print("PSU Banks: SBI, PNB, BOB, Canara".center(80))
print("=" * 80)

# Store all data for summary
all_data = {}
summary_stats = []

# Download data for each sector
for sector, stocks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    # Determine save directory
    save_dir = 'data/stocks/private_banks' if sector == 'Private Banks' else 'data/stocks/psu_banks'
    
    for name, ticker in stocks.items():
        print(f"\n📊 Downloading: {name} ({ticker})")
        print("-" * 60)
        
        try:
            # Download data
            stock_data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            
            if stock_data.empty:
                print(f"   ✗ No data available for {ticker}")
                continue
            
            # Basic statistics
            total_records = len(stock_data)
            date_start = stock_data.index[0].date()
            date_end = stock_data.index[-1].date()
            price_min = float(stock_data['Close'].min())
            price_max = float(stock_data['Close'].max())
            price_current = float(stock_data['Close'].iloc[-1])
            avg_volume = float(stock_data['Volume'].mean())
            
            print(f"   ✓ Total Records: {total_records}")
            print(f"   ✓ Date Range: {date_start} to {date_end}")
            print(f"   ✓ Current Price: ₹{price_current:.2f}")
            print(f"   ✓ Price Range: ₹{price_min:.2f} - ₹{price_max:.2f}")
            print(f"   ✓ Average Volume: {avg_volume:,.0f}")
            
            # Calculate returns
            total_return = float(((price_current - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100)
            print(f"   ✓ Total Return: {total_return:.2f}%")
            
            # Save to CSV
            ticker_clean = ticker.replace('.NS', '')
            filename = f"{save_dir}/{ticker_clean}_data.csv"
            stock_data.to_csv(filename)
            print(f"   ✓ Saved to: {filename}")
            
            # Store for summary
            all_data[name] = stock_data
            summary_stats.append({
                'Sector': sector,
                'Bank': name,
                'Ticker': ticker,
                'Records': total_records,
                'Start Date': date_start,
                'End Date': date_end,
                'Current Price': f"₹{price_current:.2f}",
                'Min Price': f"₹{price_min:.2f}",
                'Max Price': f"₹{price_max:.2f}",
                'Avg Volume': f"{avg_volume:,.0f}",
                'Total Return %': f"{total_return:.2f}%",
                'File': filename
            })
            
            # Be nice to the API (avoid rate limiting)
            time.sleep(1)
            
        except Exception as e:
            print(f"   ✗ Error downloading {ticker}: {e}")
            continue

# =============================================================================
# GENERATE SUMMARY REPORT
# =============================================================================

print("\n\n" + "=" * 80)
print("GENERATING SUMMARY REPORT".center(80))
print("=" * 80)

# Create summary DataFrame
summary_df = pd.DataFrame(summary_stats)

# Save summary to CSV
summary_file = 'data/summary/data_collection_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved to: {summary_file}")

# Display summary
print("\n" + "=" * 80)
print("DATA COLLECTION SUMMARY".center(80))
print("=" * 80)
print(summary_df.to_string(index=False))

# =============================================================================
# SECTOR-WISE COMPARISON
# =============================================================================

print("\n\n" + "=" * 80)
print("SECTOR-WISE COMPARISON".center(80))
print("=" * 80)

for sector in ['Private Banks', 'PSU Banks']:
    sector_data = summary_df[summary_df['Sector'] == sector]
    print(f"\n{sector}:")
    print("-" * 60)
    print(f"  • Number of Stocks: {len(sector_data)}")
    print(f"  • Banks: {', '.join(sector_data['Bank'].tolist())}")
    print(f"  • Total Records: {sector_data['Records'].sum():,}")

# =============================================================================
# DATA QUALITY CHECK
# =============================================================================

print("\n\n" + "=" * 80)
print("DATA QUALITY CHECK".center(80))
print("=" * 80)

quality_report = []

for name, data in all_data.items():
    # Check for missing values
    missing_values = data.isnull().sum().sum()
    
    # Check for duplicate dates
    duplicate_dates = data.index.duplicated().sum()
    
    # Check data continuity
    date_diff = data.index.to_series().diff().dt.days
    large_gaps = len(date_diff[date_diff > 5])  # Gaps > 5 days
    
    quality_report.append({
        'Bank': name,
        'Missing Values': missing_values,
        'Duplicate Dates': duplicate_dates,
        'Large Gaps (>5 days)': large_gaps,
        'Quality': '✓ Good' if (missing_values == 0 and duplicate_dates == 0) else '⚠ Check'
    })

quality_df = pd.DataFrame(quality_report)
print("\n" + quality_df.to_string(index=False))

# Save quality report
quality_file = 'data/summary/data_quality_report.csv'
quality_df.to_csv(quality_file, index=False)
print(f"\n✓ Quality report saved to: {quality_file}")

# =============================================================================
# VISUALIZE PRICE COMPARISON
# =============================================================================

print("\n\n" + "=" * 80)
print("GENERATING VISUALIZATION".center(80))
print("=" * 80)

try:
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot Private Banks
    ax1 = axes[0]
    for name, ticker in banking_stocks['Private Banks'].items():
        if name in all_data:
            # Normalize to 100 for comparison
            normalized = (all_data[name]['Close'] / all_data[name]['Close'].iloc[0]) * 100
            ax1.plot(normalized.index, normalized, label=name, linewidth=2)
    
    ax1.set_title('Private Banks - Normalized Price Performance (Base = 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Normalized Price', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot PSU Banks
    ax2 = axes[1]
    for name, ticker in banking_stocks['PSU Banks'].items():
        if name in all_data:
            # Normalize to 100 for comparison
            normalized = (all_data[name]['Close'] / all_data[name]['Close'].iloc[0]) * 100
            ax2.plot(normalized.index, normalized, label=name, linewidth=2)
    
    ax2.set_title('PSU Banks - Normalized Price Performance (Base = 100)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Normalized Price', fontsize=11)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = 'data/summary/banking_stocks_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    
    # Also create individual sector plots
    
    # Private Banks detailed plot
    fig2, ax = plt.subplots(figsize=(14, 6))
    for name, ticker in banking_stocks['Private Banks'].items():
        if name in all_data:
            ax.plot(all_data[name].index, all_data[name]['Close'], label=name, linewidth=2)
    
    ax.set_title('Private Banks - Actual Price Movement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (₹)', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    private_plot = 'data/summary/private_banks_price.png'
    plt.savefig(private_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Private banks plot saved to: {private_plot}")
    
    # PSU Banks detailed plot
    fig3, ax = plt.subplots(figsize=(14, 6))
    for name, ticker in banking_stocks['PSU Banks'].items():
        if name in all_data:
            ax.plot(all_data[name].index, all_data[name]['Close'], label=name, linewidth=2)
    
    ax.set_title('PSU Banks - Actual Price Movement', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price (₹)', fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    psu_plot = 'data/summary/psu_banks_price.png'
    plt.savefig(psu_plot, dpi=300, bbox_inches='tight')
    print(f"✓ PSU banks plot saved to: {psu_plot}")
    
    plt.close('all')
    
except ImportError:
    print("⚠ matplotlib not installed. Skipping visualization.")
    print("  Install with: pip install matplotlib")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n\n" + "=" * 80)
print("DATA COLLECTION COMPLETE!".center(80))
print("=" * 80)

print(f"\n✓ Total Banks Downloaded: {len(all_data)}")
print(f"✓ Private Banks: {len([s for s in summary_stats if s['Sector'] == 'Private Banks'])}")
print(f"✓ PSU Banks: {len([s for s in summary_stats if s['Sector'] == 'PSU Banks'])}")

print("\nFiles Created:")
print(f"  1. Individual stock CSVs in data/stocks/private_banks/ and data/stocks/psu_banks/")
print(f"  2. Summary report: {summary_file}")
print(f"  3. Quality report: {quality_file}")
print(f"  4. Comparison plots in data/summary/")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review the summary and quality reports")
print("  2. Check the visualization plots")
print("  3. Proceed to calculate technical indicators")
print("=" * 80)