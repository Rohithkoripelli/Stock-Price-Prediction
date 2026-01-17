# 🚀 Quick Run Guide - Simple One-Command Solutions

## TL;DR - Just Run This

### Weekly Update (5-10 minutes)
```bash
./run_predictions_only.sh
```

### Monthly Retrain (2-3 hours)
```bash
./run_full_pipeline.sh
```

---

## Why It's Manual (Not Automatic)

### The Problem
Your trained models (`.keras` files) are **too large for GitHub**:
- Each model: 100-300 MB
- Total: ~1-2 GB
- GitHub limit: 100 MB per file
- **Result:** GitHub Actions can't access models to generate predictions

### Current Setup
- ✅ GitHub Actions runs daily at 10 PM IST
- ✅ Collects latest data
- ✅ Calculates indicators
- ❌ **Stops there** (no models available to generate predictions)

### Solutions

| Approach | Time | Cost | Automation Level |
|----------|------|------|-----------------|
| **Manual (Current)** | 5-10 mins weekly | Free | Run script manually |
| **Cloud Storage** | Setup once | $1-5/month | Fully automatic |
| **Local Cron** | Setup once | Free | Semi-automatic |

**Recommended:** Manual (what you have now) - simplest and free!

---

## The Two Scripts Explained

### 1. `run_predictions_only.sh` - Quick Updates

**When to use:** Every week
**Duration:** 5-10 minutes
**What it does:**
- Collects latest stock data (until yesterday)
- Calculates technical indicators
- Prepares features
- **Generates predictions** (using existing trained models)
- Deploys to Vercel

**Prerequisites:** Models already trained

```bash
./run_predictions_only.sh
```

**Output:**
```
Step 1/5: Collecting latest stock data ✓
Step 2/5: Calculating technical indicators ✓
Step 3/5: Preparing enhanced features ✓
Step 4/5: Generating predictions ✓
Step 5/5: Deploying to Vercel ✓

✅ UPDATE COMPLETE!
⏱️  Total time: 7m 32s
🌐 Your website: https://...vercel.app
```

---

### 2. `run_full_pipeline.sh` - Full Retrain

**When to use:** Every month (or when market patterns change significantly)
**Duration:** 2-3 hours
**What it does:**
- Collects latest stock data
- Calculates technical indicators
- Prepares features
- **RETRAINS all 8 models** (the time-consuming part)
- Generates predictions
- Deploys to Vercel

**Why retrain?**
- Market patterns change over time
- New data improves model accuracy
- Recommended: Once per month

```bash
./run_full_pipeline.sh
```

**Output:**
```
Step 1/7: Collecting latest stock data ✓
Step 2/7: Calculating technical indicators ✓
Step 3/7: Preparing features ✓
Step 4/7: Preparing enhanced features ✓
Step 5/7: Training all 8 models ⏱️  2-3 hours...
Step 6/7: Generating predictions ✓
Step 7/7: Deploying to Vercel ✓

✅ FULL PIPELINE COMPLETE!
⏱️  Total time: 2h 34m 18s
```

---

## Recommended Workflow

### Weekly (Every Monday)
```bash
./run_predictions_only.sh
```
- Updates predictions with latest data
- Uses existing trained models
- 5-10 minutes

### Monthly (First Sunday)
```bash
./run_full_pipeline.sh
```
- Retrains all models with latest data
- Improves prediction accuracy
- 2-3 hours (run overnight or during lunch)

---

## Want Full Automation?

If you want GitHub Actions to automatically update predictions daily, you need to:

### Option 1: Hugging Face (Free, Easiest)

1. **Upload models to Hugging Face:**
```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload your-username/stock-models models/saved_v5_all/
```

2. **Update GitHub Actions** to download models before predictions

3. **Result:** Fully automatic daily updates!

### Option 2: AWS S3 / Google Cloud Storage

1. Upload models to cloud storage
2. Update GitHub Actions to download models
3. Small monthly cost (~$1-5)

### Option 3: Local Cron Job

1. Add to crontab:
```bash
crontab -e
# Add this line (runs daily at 10 PM):
0 22 * * * cd /path/to/project && ./run_predictions_only.sh
```

2. Keep your Mac running 24/7
3. Free, but requires always-on machine

---

## Troubleshooting

### "Permission denied"
```bash
chmod +x run_predictions_only.sh run_full_pipeline.sh
```

### "Models not found"
Run the full pipeline first:
```bash
./run_full_pipeline.sh
```

### "vercel command not found"
```bash
npm install -g vercel
vercel login
```

### Check if everything worked
```bash
# View predictions
cat future_predictions_next_day.csv

# View model performance
cat models/saved_v5_all/all_stocks_summary.json
```

---

## Summary

### ✅ What You Have Now

**Two simple scripts:**
1. `./run_predictions_only.sh` - Quick weekly updates (5-10 mins)
2. `./run_full_pipeline.sh` - Full monthly retrain (2-3 hours)

**Why it's manual:**
- Models too large for GitHub
- GitHub Actions can't access them
- Manual = Free and simple

**If you want automation:**
- Upload models to Hugging Face (free)
- Or AWS S3 / Google Cloud ($1-5/month)
- I can help set this up if needed

### 📅 Recommended Schedule

| Frequency | Command | Duration |
|-----------|---------|----------|
| **Weekly** | `./run_predictions_only.sh` | 5-10 mins |
| **Monthly** | `./run_full_pipeline.sh` | 2-3 hours |

---

## Need Help?

See `AUTOMATION_OPTIONS.md` for detailed automation strategies, or just ask!
