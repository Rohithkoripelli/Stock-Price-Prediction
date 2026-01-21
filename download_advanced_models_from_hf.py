#!/usr/bin/env python3
"""
Download Advanced Signals models from Hugging Face Hub

This script downloads all 8 trained V5 Transformer models with Advanced Signals
from Hugging Face for use in automated GitHub Actions workflows.
"""

import os
from huggingface_hub import hf_hub_download
from pathlib import Path

# Configuration
REPO_ID = "RohithKoripelli/indian-bank-stock-models-advanced"
MODEL_DIR = "models/saved_v5_advanced"
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
    print("DOWNLOADING ADVANCED MODELS FROM HUGGING FACE".center(80))
    print("=" * 80)
    print()
    print(f"Repository: {REPO_ID}")
    print(f"Target: {MODEL_DIR}")
    print()

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download models
    print("📥 Downloading Advanced models...")
    print("-" * 80)

    for i, stock in enumerate(STOCKS, 1):
        print(f"[{i}/8] Downloading {stock}...")
        stock_dir = os.path.join(MODEL_DIR, stock)
        os.makedirs(stock_dir, exist_ok=True)

        files_to_download = [
            "best_model.keras",
            "scaler.pkl",
            "results.json"
        ]

        for file in files_to_download:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{stock}/{file}",
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                print(f"     ✅ {file}")
            except Exception as e:
                print(f"     ⚠️  {file} - {str(e)[:50]}")

    print("-" * 80)
    print()
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!".center(80))
    print("=" * 80)
    print()
    print(f"📁 Models saved to: {MODEL_DIR}")
    print()
    print("✓ Ready to generate predictions with advanced signals")
    print()

if __name__ == "__main__":
    main()
