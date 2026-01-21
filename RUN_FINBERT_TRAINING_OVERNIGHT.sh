#!/bin/bash

# ===================================================================
# FINBERT TRAINING - OVERNIGHT RUN
# ===================================================================
# This script will:
# 1. Wait for FinBERT analysis to complete (if still running)
# 2. Integrate FinBERT features with technical indicators
# 3. Train all 8 stock models with FinBERT features
# 4. Generate predictions with new models
# 5. Compare with VADER-based predictions
#
# Estimated time: 2-3 hours
# ===================================================================

set -e  # Exit on error

echo "======================================================================"
echo "FINBERT MODEL TRAINING - OVERNIGHT RUN"
echo "======================================================================"
echo ""
echo "Start time: $(date)"
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Check if FinBERT analysis is complete
echo "Step 1: Checking FinBERT analysis status..."
if [ ! -f "data/finbert_daily_sentiment/analysis_summary.csv" ]; then
    echo "  Waiting for FinBERT analysis to complete..."
    echo "  This analyzes 963 news articles (~10-15 min)"

    # Wait for analysis to finish (check every 30 seconds)
    while [ ! -f "data/finbert_daily_sentiment/analysis_summary.csv" ]; do
        sleep 30
        echo "  Still analyzing..."
    done
    echo "  ✓ FinBERT analysis complete!"
else
    echo "  ✓ FinBERT analysis already complete"
fi

echo ""

# Step 2: Integrate FinBERT with technical indicators
echo "Step 2: Integrating FinBERT features with technical indicators..."
python prepare_finbert_features.py
echo "  ✓ Feature integration complete"
echo ""

# Step 3: Train all 8 models (quick run: 20 epochs)
echo "Step 3: Training all 8 models with FinBERT features..."
echo "  This will take 2-3 hours depending on your CPU"
echo ""

CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 python train_all_v5_finbert.py

echo ""
echo "  ✓ Training complete!"
echo ""

# Step 4: Generate predictions
echo "Step 4: Generating predictions with new models..."
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 python generate_daily_predictions.py
echo "  ✓ Predictions generated"
echo ""

# Step 5: Compare with old predictions
echo "Step 5: Comparing FinBERT vs VADER predictions..."
python compare_predictions.py
echo "  ✓ Comparison complete"
echo ""

echo "======================================================================"
echo "✓ FINBERT TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - models/saved_v5_finbert/  (new models)"
echo "  - future_predictions_finbert.json  (new predictions)"
echo "  - prediction_comparison.csv  (VADER vs FinBERT)"
echo ""
echo "Next steps:"
echo "  1. Review prediction_comparison.csv"
echo "  2. If satisfied, upload to HuggingFace: ./venv/bin/python upload_models_to_hf.py"
echo "  3. Update GitHub Actions to use new models"
echo ""
