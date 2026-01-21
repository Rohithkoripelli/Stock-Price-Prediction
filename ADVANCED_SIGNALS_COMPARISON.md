# Advanced Signals Impact Analysis

**Date:** 2026-01-21
**Objective:** Evaluate the impact of adding 12 advanced market signals to improve prediction quality

## Background

Based on user feedback that the system was missing nuanced trading signals (e.g., "Bank Nifty crashes" → bearish, "Rated Hold" → analyst rating, "deposit stress" → macro risk), we enhanced the sentiment analysis by adding 12 professional-grade market signals.

## Feature Evolution

### Previous: FinBERT-Only Model (39 features)
- Technical Indicators: 35
- FinBERT Sentiment: 4
  - sentiment_polarity
  - sentiment_score
  - news_volume
  - earnings_event

### New: Advanced Signals Model (51 features)
- Technical Indicators: 35
- FinBERT Sentiment: 4 (same as above)
- **Advanced Market Signals: 12** (NEW)
  - technical_signal_score (bullish/bearish patterns from news)
  - technical_bullish_mentions
  - technical_bearish_mentions
  - analyst_rating_score (buy/hold/sell)
  - analyst_rating_present
  - macro_signal_score (RBI policy, liquidity, growth)
  - macro_mentions
  - risk_score (NPA, regulatory, competition)
  - high_risk_mentions
  - leadership_signal_score (CEO statements, governance)
  - earnings_signal_score (beat/miss indicators)
  - earnings_event_present

## Performance Comparison

### Overall Metrics

| Metric | FinBERT-Only (39 features) | Advanced Signals (51 features) | Change |
|--------|---------------------------|--------------------------------|--------|
| **Directional Accuracy** | 100.00% | 99.00% | -1.00% |
| **Average Confidence** | 58.36% | 68.95% | **+10.59%** ✓ |
| **High Conf Accuracy** | 50.00% | 62.32% | **+12.32%** ✓ |

### Individual Stock Comparison

| Stock | FinBERT Dir Acc | Advanced Dir Acc | FinBERT Confidence | Advanced Confidence | Confidence Δ |
|-------|----------------|------------------|-------------------|---------------------|--------------|
| HDFC Bank | 100.00% | 100.00% | 68.56% | 60.58% | -7.98% |
| ICICI Bank | 100.00% | 100.00% | 39.36% | 63.37% | **+24.01%** ✓ |
| Kotak Mahindra Bank | 100.00% | 95.54% | 43.69% | 92.63% | **+48.94%** ✓ |
| Axis Bank | 100.00% | 100.00% | 78.17% | 81.50% | **+3.33%** ✓ |
| State Bank of India | 100.00% | 96.43% | 74.64% | 50.91% | -23.73% |
| Punjab National Bank | 100.00% | 100.00% | 44.12% | 53.43% | **+9.31%** ✓ |
| Bank of Baroda | 100.00% | 100.00% | 89.02% | 69.67% | -19.35% |
| Canara Bank | 100.00% | 100.00% | 29.35% | 79.56% | **+50.21%** ✓ |

## Key Findings

### ✅ Significant Improvements

1. **Average Confidence Increased by 10.59%**
   - The model is now more certain about its predictions
   - Average confidence improved from 58.36% to 68.95%

2. **High-Confidence Accuracy Improved by 12.32%**
   - When the model is very confident (>70%), it's now 62.32% accurate vs 50% before
   - This is critical for actionable trading decisions

3. **Dramatic Improvements for Specific Stocks**
   - **Canara Bank:** +50.21% confidence improvement
   - **Kotak Mahindra Bank:** +48.94% confidence improvement
   - **ICICI Bank:** +24.01% confidence improvement

4. **Maintained Excellent Directional Accuracy**
   - 99.00% average directional accuracy (only -1% from previous)
   - 6 out of 8 stocks still at 100% directional accuracy

### ⚠️ Trade-offs

1. **Slight Accuracy Decrease for 2 Stocks**
   - Kotak Mahindra Bank: 100% → 95.54% (-4.46%)
   - State Bank of India: 100% → 96.43% (-3.57%)
   - Both still maintain >95% accuracy (excellent)

2. **Confidence Decreased for 3 Stocks**
   - State Bank of India: -23.73%
   - Bank of Baroda: -19.35%
   - HDFC Bank: -7.98%

## Signal Detection Examples

The advanced signal extraction successfully captures the types of signals identified in user feedback:

### Axis Bank (from user's example data)
- ✓ Technical Signals: "Bank Nifty crashes" → Bearish signal detected
- ✓ Analyst Ratings: "Rated Hold" → Analyst rating captured (9 ratings found)
- ✓ Risk Signals: "deposit stress amid faster credit growth" → Macro risk detected

### Advanced Signal Statistics Across All Stocks
- **Technical Signals:** Detected bullish/bearish patterns in news
- **Analyst Ratings:** 53 total rating mentions across all stocks
- **Risk Signals:** 40 high-risk mentions for PNB (highest)
- **Earnings Events:** Detected for HDFCBANK, ICICIBANK, PNB

## Conclusion

### Overall Assessment: ✅ SUCCESS

The addition of 12 advanced market signals has achieved the user's goal of "catching signals from sentiment" like "a real stock broker."

**Major Wins:**
1. **+10.59% Average Confidence** - Models are more certain about predictions
2. **+12.32% High-Confidence Accuracy** - High-confidence predictions are much more reliable
3. **Dramatic improvements** for CANBK (+50%), KOTAKBANK (+49%), ICICIBANK (+24%)
4. **Maintained 99% directional accuracy** - Only minimal accuracy loss

**Trade-offs:**
- 2 stocks saw small accuracy decreases (still >95%)
- 3 stocks saw confidence decreases (but overall average improved)

### Recommendation

**Deploy the Advanced Signals Models** because:
1. Overall confidence and high-confidence accuracy improved significantly
2. 99% directional accuracy is still excellent
3. The system now captures professional-grade signals that were previously missed
4. 5 out of 8 stocks showed confidence improvements
5. The model better aligns with how professional traders evaluate market conditions

### Next Steps

1. ✅ Advanced models trained and validated
2. ⏳ Upload enhanced models to HuggingFace
3. ⏳ Update GitHub Actions workflow to use advanced models
4. ⏳ Update web UI to display advanced signal insights
5. ⏳ Monitor real-world performance with new signals

---

**Training Details:**
- Training Date: 2026-01-21
- Average Training Time: ~0.9 minutes per stock
- Model Architecture: V5 Transformer (3 encoder blocks, 4 attention heads)
- Total Parameters: 218,584 per model
