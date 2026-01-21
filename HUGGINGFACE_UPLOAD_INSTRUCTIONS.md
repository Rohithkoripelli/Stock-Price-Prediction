# HuggingFace Upload Instructions

## How to Upload FinBERT Models to HuggingFace

The FinBERT models are ready to be uploaded! Follow these steps:

### Option 1: Manual Upload (Recommended for First Time)

1. **Get your HuggingFace Token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it: "Stock Prediction Models"
   - Type: "Write"
   - Copy the token

2. **Set the token as environment variable:**
   ```bash
   export HF_TOKEN='your_token_here'
   ```

3. **Run the upload script:**
   ```bash
   cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
   ./venv/bin/python upload_finbert_models_to_hf.py
   ```

4. **Verify the upload:**
   - Visit: https://huggingface.co/RohithKoripelli/indian-bank-stock-models-finbert
   - Check that all 8 model folders are present
   - README should show the performance stats

###Option 2: Interactive Upload

If you don't want to set environment variable:

```bash
cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
./venv/bin/python upload_finbert_models_to_hf.py
```

The script will prompt you for your token.

---

## What Gets Uploaded

### Repository Structure:
```
RohithKoripelli/indian-bank-stock-models-finbert/
├── README.md (auto-generated with performance stats)
├── all_stocks_summary.json
├── HDFCBANK/
│   ├── best_model.keras
│   ├── results.json
│   └── scaler.pkl
├── ICICIBANK/
│   ├── best_model.keras
│   ├── results.json
│   └── scaler.pkl
... (6 more stocks)
```

### File Sizes:
- Each model: ~170KB
- Total upload: ~1.4MB (all 8 models)
- Upload time: ~2-3 minutes

---

## After Upload: Update GitHub Actions

Once models are uploaded, update GitHub Actions to use the new FinBERT models:

### 1. Add HF_TOKEN to GitHub Secrets

1. Go to: https://github.com/Rohithkoripelli/Stock-Price-Prediction/settings/secrets/actions
2. Click "New repository secret"
3. Name: `HF_TOKEN`
4. Value: Paste your HuggingFace token
5. Click "Add secret"

### 2. Update Workflow File

The GitHub Actions workflow needs to be updated to:
- Use new repo: `RohithKoripelli/indian-bank-stock-models-finbert`
- Load 39 features instead of 35
- Include FinBERT sentiment features

This will be done automatically in the next step.

---

## Troubleshooting

### Error: "Login failed"
- Make sure your token has "Write" permissions
- Check that HF_TOKEN environment variable is set correctly
- Try interactive mode instead

### Error: "Repository already exists"
- This is OK! The script will update the existing repository
- Models will be overwritten with newer versions

### Error: "Upload failed for [STOCK]"
- Check internet connection
- Verify model files exist: `ls -lh models/saved_v5_finbert/[STOCK]/`
- Try uploading just that stock again

### Slow upload
- Normal for first upload (~2-3 minutes)
- Subsequent uploads are faster (only changed files)

---

## Verification Checklist

After upload completes, verify:

- [ ] Repository exists: https://huggingface.co/RohithKoripelli/indian-bank-stock-models-finbert
- [ ] README shows 100% accuracy for all stocks
- [ ] All 8 stock folders are present
- [ ] Each folder contains: best_model.keras, results.json, scaler.pkl
- [ ] all_stocks_summary.json is present
- [ ] HF_TOKEN added to GitHub Secrets

---

## Next Step

After upload and verification, the GitHub Actions workflow will be automatically updated to use these new models for daily predictions.
