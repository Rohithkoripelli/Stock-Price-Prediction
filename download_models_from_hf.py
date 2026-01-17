#!/usr/bin/env python3
"""
Download trained models from Hugging Face Hub

This script is used by GitHub Actions to download models before generating predictions.
"""

import os
from huggingface_hub import hf_hub_download
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

FILES_PER_STOCK = [
    "best_model.keras",
    "metrics.json",
    "training_history.json"
]

def main():
    print("=" * 80)
    print("DOWNLOADING MODELS FROM HUGGING FACE".center(80))
    print("=" * 80)
    print()

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    total_files = len(STOCKS) * len(FILES_PER_STOCK)
    downloaded = 0

    print(f"📦 Downloading {total_files} files from {REPO_ID}")
    print("-" * 80)

    for i, stock in enumerate(STOCKS, 1):
        stock_dir = os.path.join(MODEL_DIR, stock)
        os.makedirs(stock_dir, exist_ok=True)

        print(f"[{i}/8] {stock}:")

        for file in FILES_PER_STOCK:
            try:
                # Download file from Hugging Face
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{stock}/{file}",
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                downloaded += 1
                print(f"     ✅ {file}")
            except Exception as e:
                # Only the model file is critical
                if file == "best_model.keras":
                    print(f"     ❌ {file} (CRITICAL): {e}")
                    raise
                else:
                    print(f"     ⚠️  {file} (optional): {e}")

    print("-" * 80)
    print()
    print(f"✅ Successfully downloaded {downloaded}/{total_files} files")
    print()

    # Verify all critical models exist
    print("🔍 Verifying models...")
    missing = []
    for stock in STOCKS:
        model_path = os.path.join(MODEL_DIR, stock, "best_model.keras")
        if not os.path.exists(model_path):
            missing.append(stock)
        else:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   ✅ {stock}: {size_mb:.1f} MB")

    if missing:
        print()
        print(f"❌ Error: Models missing for: {', '.join(missing)}")
        exit(1)

    print()
    print("=" * 80)
    print("✅ ALL MODELS READY!".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()
