# FinBERT Upgrade Plan: Improving Prediction Confidence

## Current Problem

**Low Confidence Predictions (60-65%)**
- Using VADER sentiment analysis (general-purpose, not finance-specific)
- Missing critical financial events (quarterly results, earnings calls)
- No real-time news integration
- Generic sentiment doesn't understand financial context

**Example Issue:**
```
Stock: HDFC Bank
Technical Indicators: Bullish (RSI 45, MACD positive)
VADER Sentiment: Neutral (0.02)  ← Misses nuance!
Reality: Quarterly results beat expectations (should be strong positive!)
Current Prediction: UP with 62% confidence ← Too low!
```

---

## Solution: FinBERT + Real-time News

### What is FinBERT?

**FinBERT** = BERT fine-tuned specifically on financial news and earnings reports

**Key Advantages over VADER:**

| Feature | VADER | FinBERT |
|---------|-------|---------|
| Domain | General text | Financial news |
| Understanding | Keyword-based | Contextual (transformer) |
| Accuracy on finance | ~60% | ~85-90% |
| Earnings detection | ❌ Poor | ✅ Excellent |
| Regulatory news | ❌ Misses | ✅ Detects |
| Nuanced context | ❌ No | ✅ Yes |

**Example:**
```
Text: "Bank reports 15% YoY profit growth but NPA rises to 3.2%"

VADER: +0.25 (slightly positive - just sees "growth")
FinBERT: -0.42 (negative - understands NPA rise is concerning)
```

---

## Implementation Architecture

### Phase 1: Data Collection (New Scripts Created ✓)

**1. Real-time News Scraper** (`scrape_indian_financial_news.py`)
```
Sources:
├── MoneyControl (stock-specific news)
├── Economic Times (market news)
├── LiveMint (banking sector)
├── GNews (fallback/supplementary)
└── Future: BSE/NSE announcements
```

**2. FinBERT Sentiment Analyzer** (`collect_finbert_sentiment.py`)
```
Features:
├── Financial domain-specific sentiment
├── Event detection (earnings, policy, risk)
├── Source credibility weighting
├── Multi-text aggregation
└── Daily sentiment scores
```

**Output Structure:**
```
data/finbert_sentiment/
├── private_banks/
│   ├── HDFCBANK_news_detailed.csv   # All articles with sentiment
│   └── HDFCBANK_daily_sentiment.csv # Aggregated daily scores
├── psu_banks/
│   └── ... (same structure)
└── sentiment_summary.csv  # Overall summary
```

---

### Phase 2: Enhanced Feature Preparation

**Modified:** `prepare_enhanced_features.py`

**New Features Added:**

```python
# Existing: 40 technical indicators
# NEW: 8 FinBERT sentiment features

1. finbert_sentiment_polarity   # -1 to +1 (weighted by source credibility)
2. finbert_sentiment_score       # 0 to 1 (confidence)
3. news_volume                   # Number of articles per day
4. earnings_event_flag           # 1 if quarterly results mentioned
5. policy_event_flag             # 1 if RBI/regulatory news
6. risk_event_flag               # 1 if NPA/fraud/penalty news
7. sentiment_momentum_3d         # 3-day rolling sentiment trend
8. sentiment_volatility_7d       # 7-day sentiment volatility
```

**Total Features:** 40 (technical) + 8 (FinBERT) = **48 features**

---

### Phase 3: Model Architecture (No Change Needed!)

**Good News:** V5 Transformer already handles variable features

```python
# Current model architecture:
Input: (batch, 60 timesteps, 35 features)  ← Will become 48 features
      ↓
Transformer Encoder (4 heads, 128 dim)
      ↓
Dense layers (128 → 64 → 32)
      ↓
Output: Price prediction

# FinBERT features automatically integrated!
```

**Why it works:**
- Transformer attention mechanism learns feature importance
- FinBERT sentiment gets weighted naturally
- No architectural changes needed

---

### Phase 4: Training with FinBERT Data

**Steps:**

1. **Collect Historical FinBERT Sentiment** (one-time)
   ```bash
   # Scrape last 2 years of news
   python scrape_indian_financial_news.py --days 730

   # Analyze with FinBERT
   python collect_finbert_sentiment.py
   ```

2. **Prepare Enhanced Features**
   ```bash
   # Merge technical + FinBERT features
   python prepare_enhanced_features.py --use-finbert
   ```

3. **Retrain Models**
   ```bash
   # Train with 48 features instead of 35
   CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 \
   ./venv/bin/python train_all_v5_transformer.py
   ```

4. **Upload to Hugging Face**
   ```bash
   ./venv/bin/python upload_models_to_hf.py
   ```

**Expected Training Time:**
- Data collection: 2-3 hours (one-time)
- FinBERT analysis: 1-2 hours (GPU recommended)
- Model training: 2-3 hours (all 8 stocks)
- **Total: ~6-8 hours** (one-time setup)

---

### Phase 5: Daily Automation

**Modified:** `.github/workflows/daily-predictions.yml`

```yaml
steps:
  - name: Download models from HF
  - name: Collect stock data          # Existing
  - name: Calculate technical indicators  # Existing

  # NEW: FinBERT sentiment collection
  - name: Scrape latest financial news
    run: python scrape_indian_financial_news.py --days 7

  - name: Analyze with FinBERT
    run: python collect_finbert_sentiment.py

  - name: Prepare enhanced features   # Modified to include FinBERT
    run: python prepare_enhanced_features.py --use-finbert

  - name: Generate predictions        # Existing
  - name: Commit and deploy           # Existing
```

**Daily Execution Time:**
- News scraping: ~3-5 minutes
- FinBERT analysis: ~2-3 minutes
- Feature prep: ~1 minute
- Predictions: ~2 minutes
- **Total: ~10 minutes** (vs current 3 minutes)

---

## Expected Improvements

### Confidence Boost

**Current:**
```json
{
  "Stock": "HDFC Bank",
  "Predicted_Direction": "UP",
  "Direction_Confidence": 62.68,  ← Low!
  "Predicted_Change_Pct": 0.45
}
```

**After FinBERT:**
```json
{
  "Stock": "HDFC Bank",
  "Predicted_Direction": "UP",
  "Direction_Confidence": 78.32,  ← Better!
  "Predicted_Change_Pct": 0.62,
  "Sentiment_Contribution": "Strong positive (Q3 results beat estimates)",
  "News_Events": ["earnings_positive", "expansion_announced"]
}
```

### Accuracy Improvements (Estimated)

| Metric | Current (VADER) | With FinBERT | Improvement |
|--------|----------------|--------------|-------------|
| Direction Accuracy | 55-60% | 70-75% | +15-20% |
| Confidence (avg) | 60-65% | 75-80% | +15% |
| Earnings Day Accuracy | 50% | 80-85% | +30-35% |
| Policy Event Detection | 0% | 90%+ | New capability |
| False Positives | 40% | 25% | -15% |

---

## Real-World Example: Earnings Day

**Scenario:** HDFC Bank announces Q3 results on Jan 15, 2026

**Current System (VADER):**
```
Technical Indicators: Mixed
  - RSI: 52 (neutral)
  - MACD: Slightly positive
  - Bollinger: Mid-range

VADER Sentiment: 0.12 (weak positive)
  - Just sees keywords like "profit", "growth"
  - Doesn't understand magnitude

Prediction: UP, 58% confidence ← Useless for traders!
```

**With FinBERT:**
```
Technical Indicators: Same (mixed)

FinBERT Sentiment: 0.78 (strong positive) ✓
  - Detects: "EPS beat by 12%"
  - Detects: "NII growth 18% YoY"
  - Detects: "Asset quality improves"
  - Flags: earnings_event = 1

Weighted Analysis:
  - Technical: Neutral (40% weight)
  - FinBERT: Strong Positive (60% weight on earnings day)

Prediction: UP, 82% confidence ✓
Price Target: +2.8% (vs current +0.5%)

→ Actionable for day traders! ✓
```

---

## Dependencies to Add

```txt
# requirements.txt additions:

# FinBERT and transformers
transformers>=4.35.0
torch>=2.0.0  # CPU version sufficient for inference
sentencepiece>=0.1.99
accelerate>=0.24.0

# Web scraping
beautifulsoup4>=4.12.0
requests>=2.31.0
lxml>=4.9.0

# Existing (no change)
gnews>=0.3.0
```

**Installation:**
```bash
pip install transformers torch beautifulsoup4 lxml
```

**Size Impact:**
- FinBERT model: ~500 MB (downloaded once, cached)
- Dependencies: ~2 GB
- **Total: ~2.5 GB** (one-time download)

---

## GitHub Actions Considerations

### Model Download Strategy

**Option 1: Download FinBERT on each run**
- Pro: No storage needed
- Con: +30 seconds per run
- Verdict: ✅ Acceptable (models cached by HF)

**Option 2: Upload FinBERT to your HF repo**
- Pro: Faster downloads
- Con: Repo size +500 MB
- Verdict: ❌ Unnecessary

### Compute Resources

**GitHub Actions Limits:**
- 6 hours per workflow
- 2-core CPU (no GPU)
- 7 GB RAM

**FinBERT CPU Inference:**
- 50 articles: ~2-3 minutes
- Memory: <2 GB
- **Verdict: ✅ Fits comfortably**

---

## Migration Path

### Week 1: Setup and Testing
```bash
# Day 1-2: Install dependencies, test FinBERT locally
pip install transformers torch beautifulsoup4
python collect_finbert_sentiment.py --test

# Day 3-4: Collect historical news (2 years)
python scrape_indian_financial_news.py --days 730

# Day 5: Analyze with FinBERT
python collect_finbert_sentiment.py

# Day 6-7: Prepare enhanced features, verify data quality
python prepare_enhanced_features.py --use-finbert
```

### Week 2: Training
```bash
# Day 8-10: Retrain all 8 models with FinBERT features
./run_full_pipeline.sh --use-finbert

# Day 11: Validate predictions on holdout set
python validate_predictions.py

# Day 12-13: A/B test (VADER vs FinBERT)
python compare_models.py --baseline vader --new finbert

# Day 14: Upload to HF if results improve
./venv/bin/python upload_models_to_hf.py
```

### Week 3: Deployment
```bash
# Day 15-16: Update GitHub Actions workflow
# Add FinBERT steps, test with workflow_dispatch

# Day 17-18: Monitor first automated runs
# Check logs, verify sentiment data quality

# Day 19-20: Website updates
# Add sentiment indicators to UI
# Show news events driving predictions

# Day 21: Full production deployment
```

---

## Cost Analysis

**Completely Free!**

| Component | Cost |
|-----------|------|
| FinBERT model | Free (HuggingFace) |
| News scraping | Free (public sites) |
| GitHub Actions | Free (2000 min/month) |
| Hugging Face hosting | Free (public repos) |
| Vercel deployment | Free tier |
| **Total** | **$0/month** ✅ |

---

## Monitoring and Validation

### Metrics to Track

**Daily:**
```
1. News articles collected per stock (target: 5-10/day)
2. FinBERT sentiment distribution (check for anomalies)
3. Prediction confidence (should increase to 75%+)
4. Workflow execution time (target: <10 min)
```

**Weekly:**
```
1. Prediction accuracy on realized prices
2. Earnings day performance (critical test)
3. False positive rate
4. User feedback from day traders
```

### Validation Script

```python
# validate_finbert_predictions.py

# Compare actual next-day price vs predicted
# Calculate:
- Direction accuracy (up/down correct?)
- Price accuracy (within ±X%?)
- Confidence calibration (80% confident = 80% accurate?)
- Earnings day boost (accuracy improvement on results days?)
```

---

## Rollback Plan

**If FinBERT doesn't improve accuracy:**

1. GitHub Actions: Comment out FinBERT steps
2. Revert to previous models (keep backup on HF)
3. Root cause analysis:
   - Check news source quality
   - Verify FinBERT setup
   - Analyze feature importance

**Rollback Time:** <1 hour (just update workflow + redeploy)

---

## Next Steps (Action Items)

### Immediate (Today):
1. ✅ Created `collect_finbert_sentiment.py`
2. ✅ Created `scrape_indian_financial_news.py`
3. ✅ Created implementation plan (this document)

### Tomorrow:
1. Install dependencies:
   ```bash
   pip install transformers torch beautifulsoup4 lxml
   ```

2. Test FinBERT locally:
   ```bash
   python collect_finbert_sentiment.py --test --stock HDFCBANK
   ```

3. Collect 7 days of news for testing:
   ```bash
   python scrape_indian_financial_news.py --days 7
   ```

### This Week:
1. Modify `prepare_enhanced_features.py` to integrate FinBERT
2. Retrain one model (HDFC Bank) as proof of concept
3. Compare predictions: VADER vs FinBERT
4. If successful → retrain all 8 stocks

### Next Week:
1. Update GitHub Actions workflow
2. Deploy to production
3. Monitor results
4. Iterate based on accuracy metrics

---

## Summary

**Problem:** 60-65% confidence, missing financial events
**Solution:** FinBERT + real-time Indian financial news
**Timeline:** 2-3 weeks for full deployment
**Cost:** $0
**Expected Improvement:** 75-80% confidence, 70-75% accuracy
**Risk:** Low (easy rollback, no infrastructure changes)

**Bottom Line:** This upgrade transforms your system from "interesting ML project" to "useful tool for day traders" by understanding financial context that VADER completely misses.

---

**Ready to proceed?** Let's start with installing dependencies and testing FinBERT on a single stock!
