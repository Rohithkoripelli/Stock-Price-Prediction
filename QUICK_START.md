# Quick Start - Automated Predictions

## What's Been Set Up

Your stock prediction system now runs **completely automatically**:

### 🕙 Daily (10 PM IST)
- Fetches latest stock prices for all 8 banks
- Calculates 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Collects news sentiment from GNews API
- Generates next-day predictions using your trained models
- Updates the website automatically

### 📅 Weekly (Sunday 2 AM IST)
- Performs full data collection
- Retrains all models with latest patterns
- Generates fresh predictions
- Deploys to production

### 🎯 Zero Manual Work Required!

## Setup Steps (5 minutes)

### 1. Add GNews API Key (Recommended)

**Get Free API Key:**
1. Go to https://gnews.io/
2. Sign up (free tier: 100 requests/day)
3. Copy your API key

**Add to GitHub:**
1. Go to https://github.com/Rohithkoripelli/Stock-Price-Prediction/settings/secrets/actions
2. Click "New repository secret"
3. Name: `GNEWS_API_KEY`
4. Value: Paste your API key
5. Click "Add secret"

### 2. Enable GitHub Actions

1. Go to https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. If prompted, click "I understand my workflows, go ahead and enable them"
3. You should see two workflows:
   - ✅ Daily Stock Predictions
   - ✅ Weekly Model Retraining

### 3. Test It Now (Optional)

**Manual Test:**
1. Go to: https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. Click "Daily Stock Predictions"
3. Click "Run workflow" dropdown → "Run workflow"
4. Wait ~3-5 minutes
5. Check if it succeeds (green checkmark ✓)

**Local Test:**
```bash
cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
./test_automation.sh
```

## What Happens Daily

```
10:00 PM IST - GitHub Actions triggers
   ↓
📊 Fetch stock data (yfinance)
   ↓
📈 Calculate technical indicators
   ↓
📰 Fetch news sentiment (GNews)
   ↓
🤖 Generate predictions (ML models)
   ↓
💾 Save to JSON
   ↓
📤 Commit to GitHub
   ↓
🚀 Vercel auto-deploys
   ↓
✅ Website updated with fresh predictions!
```

## Monitoring

### Check Automation Status

**GitHub Actions:**
https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions

- Green ✓ = Success
- Red ✗ = Failed (click to see logs)
- Yellow ● = Running

**Vercel Deployments:**
https://vercel.com/rohith-koripellis-projects/web/deployments

**Latest Predictions:**
- Web: https://web-67uaqkjac-rohith-koripellis-projects.vercel.app
- JSON: https://github.com/Rohithkoripelli/Stock-Price-Prediction/blob/main/future_predictions_next_day.json

## Files Created

```
.github/workflows/
├── daily-predictions.yml      # Daily automation (10 PM IST)
└── weekly-retrain.yml         # Weekly retraining (Sunday 2 AM)

daily_data_collection.py       # Fetches daily stock + news data
generate_daily_predictions.py  # Generates predictions
test_automation.sh            # Test locally
requirements.txt              # Python dependencies
AUTOMATION_GUIDE.md          # Detailed guide
```

## Costs

Everything runs on **FREE tiers**:

- ✅ GitHub Actions: 2,000 min/month (using ~390 min/month)
- ✅ GNews API: 100 requests/day (using ~8/day)
- ✅ Vercel: Unlimited deployments
- ✅ yfinance: Free stock data

**Total cost: $0/month** 🎉

## Customization

### Change Schedule

Edit `.github/workflows/daily-predictions.yml`:

```yaml
schedule:
  - cron: '30 16 * * *'  # Currently 10 PM IST
```

**Examples:**
- 9:00 PM IST = `'30 15 * * *'`
- 6:00 AM IST = `'30 0 * * *'`
- Every 12 hours = `'0 */12 * * *'`

### Disable Automatic Retraining

If weekly retraining is too slow/fails:

1. Go to `.github/workflows/weekly-retrain.yml`
2. Comment out the schedule section
3. Run manually when needed

## Troubleshooting

### Predictions not updating?

1. Check GitHub Actions: https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
2. Look for red ✗ marks
3. Click to view error logs

### Common Issues:

**"No data available"**
- Markets might be closed (weekends/holidays)
- Wait for next market day

**"API rate limit"**
- GNews free tier exceeded
- Will work again next day

**Build timeout**
- Weekly retraining takes too long
- Disable it, only use daily predictions

### Manual Override

If automation fails, run manually:

```bash
# Generate predictions
python generate_daily_predictions.py

# Deploy
cd web
vercel --prod --yes
```

## What You Get

### Every Day at 10 PM IST:
- ✅ Fresh predictions for all 8 stocks
- ✅ Updated confidence scores
- ✅ Latest technical indicators
- ✅ News sentiment analysis
- ✅ Auto-deployed website

### Every Sunday at 2 AM:
- ✅ Models retrained with latest data
- ✅ Improved prediction accuracy
- ✅ Updated with new market patterns

## Next Steps

1. ✅ Set up GNews API key (5 min)
2. ✅ Enable GitHub Actions (1 min)
3. ✅ Test manual run (3 min)
4. ✅ Wait for first automated run (today 10 PM)
5. ✅ Enjoy automatic daily predictions! 🎉

## Support

- **Automation Guide**: [AUTOMATION_GUIDE.md](AUTOMATION_GUIDE.md)
- **Chat Setup**: [CHAT_SETUP.md](CHAT_SETUP.md)
- **GitHub Actions**: https://github.com/Rohithkoripelli/Stock-Price-Prediction/actions
- **Vercel Dashboard**: https://vercel.com/rohith-koripellis-projects/web

---

**That's it! Your prediction system is now fully automated.** 🚀

No more manual data collection, feature engineering, or deployments. Everything happens automatically while you sleep!
