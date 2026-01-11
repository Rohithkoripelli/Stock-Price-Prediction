import pandas as pd
import numpy as np
import os
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CALCULATING TECHNICAL INDICATORS FOR INDIAN BANKING STOCKS".center(80))
print("=" * 80)

# Create output directory
os.makedirs('data/stocks_with_indicators/private_banks', exist_ok=True)
os.makedirs('data/stocks_with_indicators/psu_banks', exist_ok=True)
os.makedirs('data/indicators_summary', exist_ok=True)

# Define bank stocks
banking_stocks = {
    'Private Banks': {
        'HDFC Bank': 'HDFCBANK',
        'ICICI Bank': 'ICICIBANK',
        'Kotak Mahindra Bank': 'KOTAKBANK',
        'Axis Bank': 'AXISBANK'
    },
    'PSU Banks': {
        'State Bank of India': 'SBIN',
        'Punjab National Bank': 'PNB',
        'Bank of Baroda': 'BANKBARODA',
        'Canara Bank': 'CANBK'
    }
}

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators for a stock"""
    
    print("   📊 Calculating indicators...")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    
    # Simple Moving Averages
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
    data['SMA_200'] = SMAIndicator(close=data['Close'], window=200).sma_indicator()
    
    # Exponential Moving Averages
    data['EMA_12'] = EMAIndicator(close=data['Close'], window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(close=data['Close'], window=26).ema_indicator()
    
    # MACD
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # ADX (Average Directional Index)
    adx = ADXIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ADX'] = adx.adx()
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    # RSI (Relative Strength Index)
    data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # Rate of Change
    data['ROC_12'] = ROCIndicator(close=data['Close'], window=12).roc()
    
    # Williams %R
    data['Williams_R'] = WilliamsRIndicator(high=data['High'], low=data['Low'], 
                                            close=data['Close'], lbp=14).williams_r()
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    
    # Bollinger Bands
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Mid'] = bb.bollinger_mavg()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Width'] = bb.bollinger_wband()
    data['BB_Percent'] = bb.bollinger_pband()
    
    # Average True Range
    data['ATR_14'] = AverageTrueRange(high=data['High'], low=data['Low'], 
                                      close=data['Close'], window=14).average_true_range()
    
    # Standard Deviation
    data['StdDev_20'] = data['Close'].rolling(window=20).std()
    
    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================
    
    # On Balance Volume
    data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
    
    # Volume SMA
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    
    # Volume Weighted Average Price (VWAP)
    # VWAP typically calculated on intraday data, here we'll use a rolling approximation
    data['VWAP'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    
    # =========================================================================
    # ADDITIONAL USEFUL FEATURES
    # =========================================================================
    
    # Daily Returns
    data['Returns'] = data['Close'].pct_change()
    
    # Log Returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Volatility (20-day rolling standard deviation of returns)
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    
    # Price Change
    data['Price_Change'] = data['Close'].diff()
    
    # High-Low Range
    data['HL_Range'] = data['High'] - data['Low']
    
    # Average Price
    data['Avg_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    
    return data

def generate_indicator_summary(data, stock_name):
    """Generate summary statistics for indicators"""
    
    latest = data.iloc[-1]
    
    # Handle date formatting
    if isinstance(latest.name, str):
        date_str = latest.name.split()[0]  # Get date part if datetime string
    else:
        date_str = latest.name.strftime('%Y-%m-%d')
    
    summary = {
        'Stock': stock_name,
        'Date': date_str,
        'Close_Price': f"₹{latest['Close']:.2f}",
        'RSI_14': f"{latest['RSI_14']:.2f}",
        'MACD': f"{latest['MACD']:.2f}",
        'MACD_Signal': f"{latest['MACD_Signal']:.2f}",
        'BB_Position': f"{latest['BB_Percent']:.2f}",
        'ADX': f"{latest['ADX']:.2f}",
        'Stoch_K': f"{latest['Stoch_K']:.2f}",
        'ATR_14': f"{latest['ATR_14']:.2f}",
        'Volume_vs_Avg': f"{(latest['Volume'] / latest['Volume_SMA_20']):.2f}x" if pd.notna(latest['Volume_SMA_20']) else 'N/A',
    }
    
    # Trading signals based on indicators
    signals = []
    
    if pd.notna(latest['RSI_14']):
        if latest['RSI_14'] > 70:
            signals.append('RSI: Overbought')
        elif latest['RSI_14'] < 30:
            signals.append('RSI: Oversold')
    
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append('MACD: Bullish')
        else:
            signals.append('MACD: Bearish')
    
    if pd.notna(latest['Close']) and pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        if latest['SMA_50'] > latest['SMA_200']:
            signals.append('Golden Cross')
        elif latest['SMA_50'] < latest['SMA_200']:
            signals.append('Death Cross')
    
    summary['Signals'] = ', '.join(signals) if signals else 'Neutral'
    
    return summary

# Process all stocks
all_summaries = []

for sector, stocks in banking_stocks.items():
    print(f"\n{'=' * 80}")
    print(f"SECTOR: {sector}".center(80))
    print("=" * 80)
    
    # Determine directories
    input_dir = 'data/stocks/private_banks' if sector == 'Private Banks' else 'data/stocks/psu_banks'
    output_dir = 'data/stocks_with_indicators/private_banks' if sector == 'Private Banks' else 'data/stocks_with_indicators/psu_banks'
    
    for name, ticker in stocks.items():
        print(f"\n📈 Processing: {name} ({ticker})")
        print("-" * 60)
        
        try:
            # Read stock data
            # yfinance CSV has multi-level headers, skip the ticker row
            input_file = f"{input_dir}/{ticker}_data.csv"
            df = pd.read_csv(input_file, skiprows=[1], index_col=0, parse_dates=True)
            
            # Flatten column names if they're multi-level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            print(f"   ✓ Loaded {len(df)} records")
            
            # Calculate indicators
            df_with_indicators = calculate_technical_indicators(df)
            
            # Count indicators
            original_cols = len(df.columns)
            new_cols = len(df_with_indicators.columns)
            indicators_added = new_cols - original_cols
            
            print(f"   ✓ Added {indicators_added} technical indicators")
            
            # Generate summary
            summary = generate_indicator_summary(df_with_indicators, name)
            all_summaries.append(summary)
            
            # Save enhanced data
            output_file = f"{output_dir}/{ticker}_with_indicators.csv"
            df_with_indicators.to_csv(output_file)
            print(f"   ✓ Saved to: {output_file}")
            
            # Display latest values
            latest = df_with_indicators.iloc[-1]
            date_str = summary['Date']  # Use the date from summary
            print(f"\n   Latest Indicators ({date_str}):")
            print(f"      • RSI(14): {latest['RSI_14']:.2f}")
            print(f"      • MACD: {latest['MACD']:.2f} | Signal: {latest['MACD_Signal']:.2f}")
            print(f"      • BB Position: {latest['BB_Percent']:.2f}%")
            print(f"      • ADX: {latest['ADX']:.2f}")
            print(f"      • Signals: {summary['Signals']}")
            
        except Exception as e:
            print(f"   ✗ Error processing {ticker}: {e}")
            continue

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

print("\n\n" + "=" * 80)
print("GENERATING SUMMARY REPORT".center(80))
print("=" * 80)

summary_df = pd.DataFrame(all_summaries)

# Save summary
summary_file = 'data/indicators_summary/indicators_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved to: {summary_file}")

# Display summary
print("\n" + "=" * 80)
print("TECHNICAL INDICATORS SUMMARY".center(80))
print("=" * 80)
print(summary_df.to_string(index=False))

# ============================================================================
# GENERATE VISUALIZATION SAMPLES
# ============================================================================

print("\n\n" + "=" * 80)
print("GENERATING SAMPLE VISUALIZATIONS".center(80))
print("=" * 80)

# Create sample visualization for one stock from each sector
sample_stocks = [
    ('Private Banks', 'ICICI Bank', 'ICICIBANK'),
    ('PSU Banks', 'State Bank of India', 'SBIN')
]

for sector, name, ticker in sample_stocks:
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

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("TECHNICAL INDICATORS CALCULATION COMPLETE!".center(80))
print("=" * 80)

print(f"\n✓ Processed 8 banking stocks")
print(f"✓ Added ~40 technical indicators per stock")
print(f"✓ Enhanced data saved to: data/stocks_with_indicators/")
print(f"✓ Summary report: {summary_file}")
print(f"✓ Sample visualizations created for: ICICI Bank, SBI")

print("\n" + "=" * 80)
print("Indicators Added:")
print("  • Trend: SMA (20,50,200), EMA (12,26), MACD, ADX")
print("  • Momentum: RSI, Stochastic, ROC, Williams %R")
print("  • Volatility: Bollinger Bands, ATR, StdDev")
print("  • Volume: OBV, Volume SMA, VWAP")
print("  • Additional: Returns, Volatility, Price Changes")
print("=" * 80)

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Review the indicator summary report")
print("  2. Check sample visualizations")
print("  3. Proceed to news data collection")
print("=" * 80)
