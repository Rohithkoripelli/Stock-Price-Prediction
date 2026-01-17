# Automated Daily Predictions - Setup Guide

## Overview

This system automatically:
- **Daily (10 PM IST)**: Fetches latest data → Generates predictions → Updates website
- **Weekly (Sunday 2 AM IST)**: Full model retraining with latest data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions                            │
│  ┌────────────────────┐        ┌────────────────────┐       │
│  │  Daily @ 10 PM IST │        │ Weekly @ 2 AM IST  │       │
│  │  ─────────────────  │        │  ───────────────── │       │
│  │ 1. Fetch stock data│        │ 1. Collect full data│      │
│  │ 2. Fetch news (GNews)       │ 2. Prepare features │       │
│  │ 3. Generate predictions     │ 3. Retrain models   │       │
│  │ 4. Commit to GitHub│        │ 4. Generate preds   │       │
│  │ 5. Auto-deploy     │        │ 5. Commit & deploy  │       │
│  └────────────────────┘        └────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │ Vercel Deploy │
                    │ Auto-updates  │
                    └───────────────┘
```

## Setup Instructions

### Step 1: GitHub Secrets Configuration

You need to add these secrets to your GitHub repository:

1. Go to: https://github.com/Rohithkoripelli/Stock-Price-Prediction/settings/secrets/actions

2. Click "New repository secret" and add:

#### Required Secrets:

**GNEWS_API_KEY** (Optional but recommended)
- Get from: https://gnews.io/
- Free tier: 100 requests/day
- Used for fetching daily news sentiment

**VERCEL_TOKEN** (Optional - for manual deployment trigger)
- Get from: https://vercel.com/account/tokens
- Only needed if you want GitHub Actions to trigger Vercel deploys
- Otherwise, Vercel auto-deploys on GitHub push

**VERCEL_ORG_ID** (If using VERCEL_TOKEN)
- Run: `cd web && vercel --token=YOUR_TOKEN`
- Find in `.vercel/project.json`

**VERCEL_PROJECT_ID** (If using VERCEL_TOKEN)
- Run: `cd web && vercel --token=YOUR_TOKEN`
- Find in `.vercel/project.json`

### Step 2: Enable GitHub Actions

1. Go to: https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. Enable workflows if not already enabled
3. You should see two workflows:
   - "Daily Stock Predictions"
   - "Weekly Model Retraining"

### Step 3: Test the Automation

#### Test Daily Predictions Manually:
1. Go to Actions tab
2. Select "Daily Stock Predictions"
3. Click "Run workflow" → "Run workflow"
4. Wait ~2-5 minutes
5. Check the logs and results

#### Test Locally:
```bash
# Test data collection
python daily_data_collection.py

# Test prediction generation
python generate_daily_predictions.py

# Check output
cat future_predictions_next_day.json
```

## How It Works

### Daily Workflow (10 PM IST)

1. **Data Collection** (`daily_data_collection.py`)
   - Fetches last 100 days of stock data for all 8 banks
   - Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
   - Fetches news from GNews API (if configured)
   - Saves to `data/stocks/daily_updates/`

2. **Prediction Generation** (`generate_daily_predictions.py`)
   - Loads existing trained models from `models/saved_v5_all/`
   - Uses latest data to generate predictions
   - Creates predictions with confidence scores
   - Saves to `future_predictions_next_day.json`

3. **Deployment**
   - Commits changes to GitHub
   - Vercel automatically detects changes
   - Website updates with new predictions

### Weekly Workflow (Sunday 2 AM IST)

1. **Full Data Collection**
   - Runs complete data collection pipeline
   - Collects extended historical data

2. **Feature Preparation**
   - Prepares features for all stocks
   - Normalizes and scales data

3. **Model Retraining**
   - Retrains Transformer models on CPU
   - Updates model weights with latest patterns
   - Saves new models

4. **Prediction & Deployment**
   - Generates fresh predictions
   - Deploys to production

## Monitoring & Maintenance

### Check Automation Status

**GitHub Actions Dashboard:**
https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions

**View Latest Run:**
- Green checkmark ✓ = Success
- Red X ✗ = Failed (check logs)

**Vercel Deployments:**
https://vercel.com/rohith-koripellis-projects/web/deployments

### Troubleshooting

#### Predictions not updating?
1. Check GitHub Actions logs for errors
2. Verify GNews API key is set (if using)
3. Check if markets are open (no data on weekends/holidays)

#### Build failures?
1. Model files might be too large for GitHub
2. Training timeout (6 hour limit on GitHub Actions)
3. Solution: Disable weekly retraining, only use daily predictions

#### Manual deployment needed?
```bash
# Generate predictions manually
python generate_daily_predictions.py

# Deploy to Vercel
cd web
vercel --prod --yes
```

## Cost Considerations

### GitHub Actions
- Free tier: 2,000 minutes/month
- Daily prediction: ~5 minutes/day = 150 min/month
- Weekly retrain: ~60 minutes/week = 240 min/month
- **Total: ~390 minutes/month (well within free tier)**

### GNews API
- Free tier: 100 requests/day
- Our usage: 8 stocks/day = 8 requests/day
- **Well within free tier**

### Vercel
- Free tier includes:
  - Unlimited deployments
  - 100GB bandwidth/month
  - **Our usage: Minimal, well within free tier**

## Customization

### Change Schedule

Edit `.github/workflows/daily-predictions.yml`:

```yaml
schedule:
  - cron: '30 16 * * *'  # 10:00 PM IST
```

Cron format: `minute hour day month weekday`
- IST = UTC + 5:30
- Convert IST to UTC for cron

Examples:
- 9:00 PM IST = 15:30 UTC → `'30 15 * * *'`
- 6:00 AM IST = 00:30 UTC → `'30 0 * * *'`

### Disable Weekly Retraining

If retraining takes too long or fails:

1. Go to `.github/workflows/weekly-retrain.yml`
2. Comment out or delete the schedule
3. Run manually when needed using "Run workflow" button

### Add More Stocks

Edit the STOCKS list in:
- `daily_data_collection.py`
- `generate_daily_predictions.py`

Then train models for new stocks and redeploy.

## Manual Commands

```bash
# Run daily update manually
./update_predictions.sh

# Or step by step:
python daily_data_collection.py
python generate_daily_predictions.py
cp future_predictions_next_day.json web/
cd web && vercel --prod --yes

# Retrain models manually
python prepare_features_for_modeling.py
python train_all_v5_transformer.py
python generate_daily_predictions.py
```

## Support

For issues:
1. Check GitHub Actions logs
2. Test scripts locally
3. Review error messages
4. Check API quotas (GNews, etc.)

## Next Steps

1. Set up GitHub secrets (especially GNEWS_API_KEY)
2. Test the workflows manually
3. Monitor first few automated runs
4. Adjust schedule if needed
5. Enjoy automated daily predictions!
