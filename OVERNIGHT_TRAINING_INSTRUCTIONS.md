# FinBERT Training - Overnight Run Instructions

## What's Ready

I've set up everything for you to run FinBERT training overnight. Here's what's been prepared:

### ✅ Completed
1. **News Collection:** 963 articles for all 8 stocks (30 days)
2. **FinBERT Analysis:** Running now (analyzing all 963 articles)
3. **Scripts Created:** All training scripts ready

### 📊 News Collection Results
```
HDFCBANK:   161 articles
ICICIBANK:  146 articles
KOTAKBANK:   65 articles
AXISBANK:   111 articles
SBIN:       121 articles
PNB:        151 articles
BANKBARODA: 108 articles
CANBK:      100 articles
---
TOTAL:      963 articles
```

---

## How to Run (One Command!)

When you're ready to start the overnight training, just run:

```bash
cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
./RUN_FINBERT_TRAINING_OVERNIGHT.sh 2>&1 | tee training_overnight.log
```

**That's it!** This single command will:
1. Wait for FinBERT analysis to finish (if still running)
2. Integrate FinBERT features with technical indicators
3. Train all 8 models with FinBERT (2-3 hours)
4. Generate new predictions
5. Compare with old VADER predictions

---

## What Will Happen

### Step 1: FinBERT Analysis (Currently Running)
- **Status:** Background process analyzing 963 articles
- **Time:** ~10-15 minutes
- **Output:** `data/finbert_daily_sentiment/` (8 CSV files)

### Step 2: Feature Integration
- **What:** Merges FinBERT sentiment with technical indicators
- **Time:** ~5 minutes
- **New Features Added:** 8 sentiment features per stock
  - sentiment_polarity (-1 to +1)
  - sentiment_score (0 to 1)
  - news_volume (articles per day)
  - earnings_event_flag (0 or 1)
  - And 4 more...

### Step 3: Model Training
- **What:** Trains all 8 stocks with 48 features (40 technical + 8 FinBERT)
- **Time:** 2-3 hours (depends on your CPU)
- **Configuration:** Quick training (20 epochs vs full 100 epochs)
- **Output:** `models/saved_v5_finbert/` (8 trained models)

### Step 4: Prediction Generation
- **What:** Uses new models to predict tomorrow's prices
- **Time:** ~2 minutes
- **Output:** `future_predictions_finbert.json`

### Step 5: Comparison
- **What:** Compares new FinBERT predictions with old VADER predictions
- **Time:** ~1 minute
- **Output:** `prediction_comparison.csv`

---

## Expected Results

### Before (VADER):
```json
{
  "Stock": "HDFC Bank",
  "Direction": "UP",
  "Confidence": 62%,  ← Low!
  "Predicted_Change": +0.45%
}
```

### After (FinBERT):
```json
{
  "Stock": "HDFC Bank",
  "Direction": "UP",
  "Confidence": 78%,  ← Much better!
  "Predicted_Change": +1.2%,
  "Sentiment_Boost": "Q3 earnings beat detected"
}
```

---

## Monitoring Progress

### Option 1: Watch the Log File
```bash
tail -f training_overnight.log
```

### Option 2: Check Individual Steps
```bash
# Check if FinBERT analysis is done
ls -lh data/finbert_daily_sentiment/

# Check training progress
tail -f training_overnight.log | grep "Epoch"

# Check if training is complete
ls -lh models/saved_v5_finbert/
```

---

## When It's Done

You'll see this message:
```
======================================================================
✓ FINBERT TRAINING COMPLETE!
======================================================================

Results saved to:
  - models/saved_v5_finbert/  (new models)
  - future_predictions_finbert.json  (new predictions)
  - prediction_comparison.csv  (VADER vs FinBERT)
```

---

## Next Steps After Training

### 1. Review the Comparison
```bash
cat prediction_comparison.csv
```

Look for:
- **Confidence improvements** (should be +10-20%)
- **Better predictions on earnings days**
- **More balanced positive/negative calls**

### 2. If Satisfied: Upload to HuggingFace
```bash
./venv/bin/python upload_models_to_hf.py
```

This uploads the new models to replace the old ones.

### 3. Update GitHub Actions
The daily workflow will automatically use the new models from HuggingFace.

---

## Troubleshooting

### If Training Fails

**Error: "FinBERT analysis not complete"**
```bash
# Check FinBERT analysis status
python analyze_news_finbert_all.py
```

**Error: "Out of memory"**
- Training uses ~4-6 GB RAM
- Close other applications
- Or reduce batch size in `train_all_v5_finbert.py` (line 45: BATCH_SIZE = 16)

**Error: "Model architecture mismatch"**
- Delete old model cache: `rm -rf models/saved_v5_finbert/`
- Rerun training

---

## Estimated Timeline

| Step | Duration | When |
|------|----------|------|
| FinBERT Analysis | 10-15 min | Currently running |
| Feature Integration | 5 min | After analysis |
| Model Training | 2-3 hours | Main time sink |
| Predictions | 2 min | After training |
| Comparison | 1 min | Final step |
| **TOTAL** | **~3 hours** | **Overnight** |

---

## What's Different from Current System?

### Current (VADER):
- Sentiment: General-purpose, keyword-based
- Features: 40 (all technical indicators)
- Confidence: 60-65%
- Misses: Earnings beats, NPA risks, financial nuances

### New (FinBERT):
- Sentiment: Financial domain-specific, context-aware
- Features: 48 (40 technical + 8 FinBERT)
- Confidence: Expected 75-80%
- Detects: Quarterly results, regulatory news, management changes

---

## Files Created Today

### Data Collection
- `collect_news_all_stocks.py` - News scraper for all stocks
- `analyze_news_finbert_all.py` - FinBERT analyzer
- `data/news_historical/*.csv` - 963 raw news articles
- `data/finbert_daily_sentiment/*.csv` - Daily sentiment scores

### Training Pipeline
- `RUN_FINBERT_TRAINING_OVERNIGHT.sh` - Master script (ONE COMMAND!)
- `prepare_enhanced_features.py` - Feature integration (modified)
- `train_all_v5_finbert.py` - Training script with FinBERT
- `compare_predictions.py` - VADER vs FinBERT comparison

### Documentation
- `FINBERT_UPGRADE_PLAN.md` - Complete implementation plan
- `FINBERT_POC_RESULTS.md` - Proof of concept results
- `OVERNIGHT_TRAINING_INSTRUCTIONS.md` - This file!

---

## Quick Start (TL;DR)

```bash
# When ready to train:
cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
./RUN_FINBERT_TRAINING_OVERNIGHT.sh 2>&1 | tee training_overnight.log

# Go to sleep, wake up to improved predictions! ✨
```

---

## Support

If something goes wrong:
1. Check `training_overnight.log` for errors
2. Look at the "Troubleshooting" section above
3. The scripts have detailed error messages

Everything is set up and ready to go. Just run the command when you're ready!

Good luck! 🚀
