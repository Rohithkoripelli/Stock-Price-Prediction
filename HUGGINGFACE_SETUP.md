# ✅ Hugging Face Automation Setup - COMPLETE!

## Status: Models Uploaded Successfully

**Repository:** https://huggingface.co/RohithKoripelli/indian-bank-stock-models

All 8 trained V5 Transformer models have been uploaded to Hugging Face and are ready for automated GitHub Actions workflows.

---

## What Was Done

### 1. ✅ Created Hugging Face Repository
- **Name:** `indian-bank-stock-models`
- **Owner:** RohithKoripelli
- **Type:** Public model repository
- **URL:** https://huggingface.co/RohithKoripelli/indian-bank-stock-models

### 2. ✅ Uploaded All 8 Models
Each stock model includes:
- `best_model.keras` (~2 MB each)
- `metrics.json` (model performance)
- `training_history.json` (training logs)

**Models:**
- ✅ HDFC Bank (HDFCBANK)
- ✅ ICICI Bank (ICICIBANK)
- ✅ Kotak Mahindra Bank (KOTAKBANK)
- ✅ Axis Bank (AXISBANK)
- ✅ State Bank of India (SBIN)
- ✅ Punjab National Bank (PNB)
- ✅ Bank of Baroda (BANKBARODA)
- ✅ Canara Bank (CANBK)

### 3. ✅ Created Upload Script
**File:** `upload_models_to_hf.py`

**Usage:**
```bash
./venv/bin/python upload_models_to_hf.py
```

**When to use:** After retraining models (monthly)

### 4. ✅ Created Download Script
**File:** `download_models_from_hf.py`

**Usage:**
```bash
./venv/bin/python download_models_from_hf.py
```

**When used:** Automatically by GitHub Actions

### 5. ✅ Updated GitHub Actions Workflow
**File:** `.github/workflows/daily-predictions.yml`

**New workflow:**
1. Install dependencies (including huggingface_hub)
2. **Download models from Hugging Face** ← NEW!
3. Collect latest stock data
4. Calculate technical indicators
5. Prepare enhanced features
6. Generate predictions
7. Commit and push
8. Deploy to Vercel

---

## 🔑 Next Step: Add Hugging Face Token to GitHub

### Why Needed?
GitHub Actions needs your Hugging Face token to download models.

### How to Add:

#### Step 1: Get Your Hugging Face Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `github-actions-read`
4. Type: Select "Read"
5. Click "Generate token"
6. **Copy the token** (starts with `hf_...`)

#### Step 2: Add Token to GitHub Secrets
1. Go to your GitHub repo: https://github.com/Rohithkoripelli/Stock-Price-Prediction
2. Click **Settings** (top menu)
3. In left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Name: `HF_TOKEN`
6. Value: Paste your Hugging Face token
7. Click **Add secret**

---

## 🎉 Result: Fully Automated Daily Predictions!

Once the token is added, GitHub Actions will:

### Daily (10 PM IST):
1. ✅ Download models from Hugging Face (~16 MB total)
2. ✅ Collect latest stock data (until yesterday)
3. ✅ Calculate technical indicators
4. ✅ Prepare enhanced features
5. ✅ Generate predictions
6. ✅ Commit to GitHub
7. ✅ Deploy to Vercel

**No manual intervention needed!**

---

## 📋 Model Information

### Model Specs
- **Architecture:** V5 Transformer
- **Parameters:** ~154,808 per model
- **Features:** 35 (technical, sentiment, fundamental, macro, sector)
- **Lookback:** 60 days
- **Training Data:** Jan 2019 - Jan 2026 (~1,743 records per stock)

### Performance
- **Average MAPE:** 0.84%
- **Average R²:** 0.9771
- **Directional Accuracy:** 65.15%

---

## 🔄 Updating Models (Monthly)

When you retrain models with new data:

### Step 1: Retrain Locally
```bash
./run_full_pipeline.sh
```

### Step 2: Upload to Hugging Face
```bash
./venv/bin/python upload_models_to_hf.py
```

### Step 3: Done!
GitHub Actions will automatically use the new models starting next run.

---

## 🧪 Testing the Automation

### Option 1: Manual Trigger
1. Go to: https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. Click "Daily Stock Predictions"
3. Click "Run workflow" → "Run workflow"
4. Watch it run!

### Option 2: Wait for Scheduled Run
Next automatic run: **Daily at 10 PM IST (4:30 PM UTC)**

---

## 📁 Files Created

1. **`upload_models_to_hf.py`**
   - Uploads all 8 models to Hugging Face
   - Creates repository if needed
   - Adds README with model info

2. **`download_models_from_hf.py`**
   - Downloads models from Hugging Face
   - Used by GitHub Actions
   - Verifies all models downloaded successfully

3. **`.github/workflows/daily-predictions.yml`** (updated)
   - Added model download step
   - Added huggingface_hub dependency
   - Full pipeline automation

---

## 🔐 Security Notes

### Token Permissions
- **Read-only token** is sufficient for GitHub Actions
- Never commit tokens to code
- Store in GitHub Secrets only

### Public vs Private Repository
- Models are **public** on Hugging Face (free, no limits)
- Anyone can download and use them
- Fine for research/educational purposes
- If you want private models, upgrade to Hugging Face Pro ($9/month)

---

## 💡 Benefits of This Setup

### Before (Manual):
- ❌ Models too large for GitHub
- ❌ GitHub Actions couldn't generate predictions
- ❌ Manual local execution required

### After (Automated):
- ✅ Models hosted on Hugging Face (free)
- ✅ GitHub Actions downloads on-demand
- ✅ Fully automatic daily predictions
- ✅ Zero manual intervention
- ✅ Always up-to-date website

---

## 🆘 Troubleshooting

### "403 Forbidden" when uploading
**Solution:** Token needs write permissions
- Go to https://huggingface.co/settings/tokens
- Create new token with "Write" permission

### "404 Not Found" when downloading
**Solution:** Repository doesn't exist yet
- Run `upload_models_to_hf.py` first

### GitHub Actions fails at download step
**Solution:** HF_TOKEN not set
- Add token to GitHub Secrets (see Step 2 above)

---

## 📊 Monitoring

### Check Model Downloads
1. Go to https://huggingface.co/RohithKoripelli/indian-bank-stock-models
2. Click "Files and versions"
3. See download counts

### Check GitHub Actions
1. Go to https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. View workflow runs
3. Check logs

---

## ✅ Summary

**What's automated now:**
- ✅ Model storage (Hugging Face)
- ✅ Model downloads (GitHub Actions)
- ✅ Data collection (daily)
- ✅ Technical indicators (daily)
- ✅ Feature preparation (daily)
- ✅ Prediction generation (daily)
- ✅ Deployment (Vercel)

**What's still manual:**
- ⚠️ Model retraining (monthly) - run `./run_full_pipeline.sh`
- ⚠️ Model upload (after retrain) - run `upload_models_to_hf.py`

**Why retrain is manual:**
- Takes 2-3 hours
- Requires significant compute
- Only needed monthly
- GitHub Actions has 6-hour timeout

---

**Next:** Add your Hugging Face token to GitHub Secrets, then you're done! 🎉
