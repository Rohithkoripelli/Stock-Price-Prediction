import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GENERATING TECHNICAL INDICATOR VISUALIZATIONS".center(80))
print("For all 8 banking stocks".center(80))
print("=" * 80)

# Define all stocks
all_stocks = [
    ('Private Banks', 'HDFC Bank', 'HDFCBANK'),
    ('Private Banks', 'ICICI Bank', 'ICICIBANK'),
    ('Private Banks', 'Kotak Mahindra Bank', 'KOTAKBANK'),
    ('Private Banks', 'Axis Bank', 'AXISBANK'),
    ('PSU Banks', 'State Bank of India', 'SBIN'),
    ('PSU Banks', 'Punjab National Bank', 'PNB'),
    ('PSU Banks', 'Bank of Baroda', 'BANKBARODA'),
    ('PSU Banks', 'Canara Bank', 'CANBK')
]

for sector, name, ticker in all_stocks:
    try:
        print(f"\n📊 Creating visualization for: {name}")
        
        input_dir = 'data/stocks_with_indicators/private_banks' if sector == 'Private Banks' else 'data/stocks_with_indicators/psu_banks'
        df = pd.read_csv(f"{input_dir}/{ticker}_with_indicators.csv", index_col=0, parse_dates=True)
        
        # Use last 6 months of data for clearer visualization
        df_recent = df.tail(120)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot 1: Price with Bollinger Bands and Moving Averages
        ax1 = axes[0]
        ax1.plot(df_recent.index, df_recent['Close'], label='Close Price', linewidth=2, color='black')
        ax1.plot(df_recent.index, df_recent['BB_High'], label='BB High', linestyle='--', alpha=0.7, color='red')
        ax1.plot(df_recent.index, df_recent['BB_Mid'], label='BB Mid', linestyle='--', alpha=0.7, color='blue')
        ax1.plot(df_recent.index, df_recent['BB_Low'], label='BB Low', linestyle='--', alpha=0.7, color='green')
        ax1.plot(df_recent.index, df_recent['SMA_20'], label='SMA 20', alpha=0.6, color='orange')
        ax1.plot(df_recent.index, df_recent['SMA_50'], label='SMA 50', alpha=0.6, color='purple')
        ax1.fill_between(df_recent.index, df_recent['BB_Low'], df_recent['BB_High'], alpha=0.1)
        ax1.set_title(f'{name} - Price with Bollinger Bands & Moving Averages', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Price (₹)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2 = axes[1]
        ax2.plot(df_recent.index, df_recent['RSI_14'], label='RSI(14)', linewidth=2, color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(df_recent.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_title('Relative Strength Index (RSI)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('RSI')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Plot 3: MACD
        ax3 = axes[2]
        ax3.plot(df_recent.index, df_recent['MACD'], label='MACD', linewidth=2, color='blue')
        ax3.plot(df_recent.index, df_recent['MACD_Signal'], label='Signal', linewidth=2, color='red')
        ax3.bar(df_recent.index, df_recent['MACD_Diff'], label='Histogram', alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('MACD (Moving Average Convergence Divergence)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('MACD')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Volume with SMA
        ax4 = axes[3]
        ax4.bar(df_recent.index, df_recent['Volume'], label='Volume', alpha=0.6, color='steelblue')
        ax4.plot(df_recent.index, df_recent['Volume_SMA_20'], label='Volume SMA 20', 
                linewidth=2, color='red')
        ax4.set_title('Volume Analysis', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Volume')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = f'data/indicators_summary/{ticker}_technical_indicators.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Visualization saved to: {plot_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"   ✗ Error creating visualization: {e}")

print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!".center(80))
print("=" * 80)
print("\n✓ Created 8 technical indicator visualizations")
print("✓ All files saved to: data/indicators_summary/")
print("\nFiles created:")
for sector, name, ticker in all_stocks:
    print(f"  • {ticker}_technical_indicators.png")
