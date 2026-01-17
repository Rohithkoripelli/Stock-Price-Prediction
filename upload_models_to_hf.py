#!/usr/bin/env python3
"""
Upload trained stock prediction models to Hugging Face Hub

This script uploads all 8 trained V5 Transformer models to Hugging Face
for use in automated GitHub Actions workflows.
"""

import os
from huggingface_hub import HfApi, create_repo, login
from pathlib import Path

# Configuration
REPO_ID = "RohithKoripelli/indian-bank-stock-models"
MODEL_DIR = "models/saved_v5_all"
STOCKS = [
    "HDFCBANK",
    "ICICIBANK",
    "KOTAKBANK",
    "AXISBANK",
    "SBIN",
    "PNB",
    "BANKBARODA",
    "CANBK"
]

def main():
    print("=" * 80)
    print("UPLOADING STOCK PREDICTION MODELS TO HUGGING FACE".center(80))
    print("=" * 80)
    print()

    # Check if models exist
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model directory not found: {MODEL_DIR}")
        print("Please train models first using: ./run_full_pipeline.sh")
        return

    # Check if all model directories exist
    missing = []
    for stock in STOCKS:
        model_path = os.path.join(MODEL_DIR, stock, "best_model.keras")
        if not os.path.exists(model_path):
            missing.append(stock)

    if missing:
        print(f"❌ Error: Models missing for: {', '.join(missing)}")
        print("Please train all models first using: ./run_full_pipeline.sh")
        return

    print("✅ All 8 models found")
    print()

    # Login to Hugging Face
    print("🔐 Logging in to Hugging Face...")
    print("Please enter your Hugging Face token when prompted.")
    print("(Get your token from: https://huggingface.co/settings/tokens)")
    print()

    try:
        login()
        print("✅ Successfully logged in!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return

    print()

    # Create repository
    print(f"📦 Creating/accessing repository: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository ready")
    except Exception as e:
        print(f"❌ Failed to create repository: {e}")
        return

    print()

    # Upload models
    api = HfApi()

    print("🚀 Uploading models...")
    print("-" * 80)

    for i, stock in enumerate(STOCKS, 1):
        stock_dir = os.path.join(MODEL_DIR, stock)
        print(f"[{i}/8] Uploading {stock}...")

        try:
            # Upload the entire stock directory
            api.upload_folder(
                folder_path=stock_dir,
                path_in_repo=stock,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"     ✅ {stock} uploaded successfully")
        except Exception as e:
            print(f"     ❌ Failed to upload {stock}: {e}")

    print("-" * 80)
    print()

    # Create README
    readme_content = f"""---
tags:
- stock-prediction
- transformer
- finance
- indian-banks
license: mit
---

# Indian Bank Stock Price Prediction Models

This repository contains 8 trained V5 Transformer models for predicting next-day price movements of major Indian banking stocks.

## Models

- **HDFC Bank** (HDFCBANK.NS)
- **ICICI Bank** (ICICIBANK.NS)
- **Kotak Mahindra Bank** (KOTAKBANK.NS)
- **Axis Bank** (AXISBANK.NS)
- **State Bank of India** (SBIN.NS)
- **Punjab National Bank** (PNB.NS)
- **Bank of Baroda** (BANKBARODA.NS)
- **Canara Bank** (CANBK.NS)

## Model Architecture

- **Type:** V5 Transformer
- **Features:** 35 (technical, sentiment, fundamental, macro, sector)
- **Lookback:** 60 days
- **Parameters:** ~154,808 per model

## Performance

Average metrics across all 8 models:
- **MAPE:** 0.84%
- **R²:** 0.9771
- **Directional Accuracy:** 65.15%

## Usage

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download a specific model
model_path = hf_hub_download(
    repo_id="Rohithkoripelli/indian-bank-stock-models",
    filename="HDFCBANK/best_model.keras"
)

# Load the model
model = tf.keras.models.load_model(model_path)
```

## Training Data

- **Date Range:** January 2019 - January 2026
- **Records:** ~1,743 per stock
- **Features:** Technical indicators, sentiment scores, fundamental metrics

## Automation

These models are used in an automated GitHub Actions workflow that:
1. Collects latest stock data daily
2. Downloads models from Hugging Face
3. Generates predictions
4. Deploys to Vercel

## License

MIT License - Free to use for research and educational purposes.

## Repository

Full code and documentation: [Stock-Price-Prediction](https://github.com/Rohithkoripelli/Stock-Price-Prediction)
"""

    print("📝 Creating README...")
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅ README created")
    except Exception as e:
        print(f"⚠️  Warning: Failed to create README: {e}")

    print()
    print("=" * 80)
    print("✅ UPLOAD COMPLETE!".center(80))
    print("=" * 80)
    print()
    print(f"🌐 Models available at: https://huggingface.co/{REPO_ID}")
    print()
    print("Next steps:")
    print("1. Verify models at the Hugging Face URL above")
    print("2. Copy your Hugging Face token for GitHub Secrets")
    print("3. Add token to GitHub: Settings > Secrets > HF_TOKEN")
    print()

if __name__ == "__main__":
    main()
