# Automation Options for Stock Price Predictions

## Current Limitation

**Problem:** Model files (`.keras`) are too large for GitHub
- Each model: 100-300 MB
- Total: ~1-2 GB for all 8 models
- GitHub limit: 100 MB per file
- Result: GitHub Actions can't access models to generate predictions

## Option 1: Manual Local Execution (Current - SIMPLEST)

**When to use:** Weekly/monthly retraining
**Complexity:** Low
**Cost:** Free

### Single Command:
```bash
./run_full_pipeline.sh
```

This runs:
1. Data collection (until T-1)
2. Technical indicators
3. Feature preparation
4. Model retraining (optional, only when needed)
5. Prediction generation
6. Deployment to Vercel

**Pros:**
- ✅ Free
- ✅ Simple
- ✅ Full control
- ✅ No cloud setup needed

**Cons:**
- ❌ Manual process
- ❌ Need to remember to run it

## Option 2: Cloud Model Storage (Fully Automated)

**When to use:** Daily automatic updates
**Complexity:** Medium
**Cost:** ~$1-5/month

### How it works:
1. Upload trained models to cloud storage (AWS S3, Google Cloud Storage, Hugging Face)
2. GitHub Actions downloads models before prediction
3. Generates predictions daily
4. Auto-deploys to Vercel

### Setup:
```bash
# Option A: AWS S3
aws s3 sync models/saved_v5_all/ s3://your-bucket/models/

# Option B: Hugging Face (Free for public models)
huggingface-cli upload your-username/stock-models models/saved_v5_all/

# Option C: Google Cloud Storage
gsutil -m cp -r models/saved_v5_all/* gs://your-bucket/models/
```

**Pros:**
- ✅ Fully automated
- ✅ Daily predictions
- ✅ No manual intervention

**Cons:**
- ❌ Requires cloud account setup
- ❌ Small monthly cost
- ❌ More complex

## Option 3: Local Cron Job (Semi-Automated)

**When to use:** Daily updates without cloud costs
**Complexity:** Low
**Cost:** Free (requires machine running 24/7)

### Setup:
```bash
# Add to crontab (runs daily at 10 PM)
0 22 * * * cd /path/to/project && ./run_predictions_only.sh
```

**Pros:**
- ✅ Free
- ✅ Automated (if machine always on)
- ✅ No cloud setup

**Cons:**
- ❌ Requires machine to be running 24/7
- ❌ Need Mac/server always on

## Option 4: Vercel Serverless Functions (Advanced)

**When to use:** Real-time predictions on demand
**Complexity:** High
**Cost:** Free tier available

### How it works:
- Store models in Vercel Blob Storage
- Load models in serverless function
- Generate predictions on-demand via API

**Pros:**
- ✅ No manual work
- ✅ Always available
- ✅ Serverless (no maintenance)

**Cons:**
- ❌ Complex setup
- ❌ Function timeout limits (10s-300s)
- ❌ Model loading overhead

## Recommended Approach

### For Your Use Case (Weekly/Monthly Updates):

**Use Option 1: Manual Local Execution**

Why?
- Models don't need daily retraining (weekly/monthly is enough)
- Predictions can be updated weekly
- Free and simple
- Full control over when to retrain

### Workflow:
1. **Weekly:** Run `./run_predictions_only.sh` (5 minutes)
   - Uses existing trained models
   - Generates fresh predictions
   - Deploys to Vercel

2. **Monthly:** Run `./run_full_pipeline.sh` (2-3 hours)
   - Retrains all models with latest data
   - Generates predictions
   - Deploys to Vercel

## If You Want Full Automation:

**Use Option 2: Cloud Model Storage**

### Quick Setup (Hugging Face - Free):
```bash
# 1. Install Hugging Face CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Upload models
huggingface-cli upload rohithkoripelli/stock-models models/saved_v5_all/

# 4. Update GitHub Actions to download models
# (I can help with this)
```

Then GitHub Actions will:
- Run daily at 10 PM IST ✅
- Download models from Hugging Face ✅
- Generate predictions ✅
- Auto-deploy to Vercel ✅

## Comparison Table

| Feature | Manual Local | Cloud Storage | Local Cron | Vercel Functions |
|---------|-------------|---------------|------------|------------------|
| **Cost** | Free | $1-5/month | Free | Free tier |
| **Complexity** | Low | Medium | Low | High |
| **Automation** | Manual | Full | Semi | Full |
| **Setup Time** | 5 mins | 1-2 hours | 15 mins | 4-6 hours |
| **Maintenance** | Low | Low | Medium | Medium |
| **Best For** | Weekly updates | Daily auto | Daily (local) | Real-time |

## My Recommendation

**Start with Manual Local (Option 1)**
- Use the single script I'm creating: `./run_full_pipeline.sh`
- Run weekly (5 minutes)
- Retrain monthly (2-3 hours)
- Zero cost, zero complexity

**Upgrade to Cloud Storage later** if you want:
- Daily automatic predictions
- No manual intervention
- I can help set this up when needed

---

**Next:** I'm creating the single-script solution for you now!
