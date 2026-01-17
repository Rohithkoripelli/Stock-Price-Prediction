#!/bin/bash

################################################################################
# Stock Price Predictions - Full Pipeline (Retrain Models + Predictions)
#
# What it does:
# 1. Collects latest stock data (until T-1)
# 2. Calculates technical indicators
# 3. Prepares features for modeling
# 4. RETRAINS all 8 models with latest data
# 5. Generates predictions
# 6. Deploys to Vercel
#
# When to use: Monthly retraining (2-3 hours)
# Use case: Improve model accuracy with latest market patterns
################################################################################

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                      STOCK PREDICTIONS - FULL PIPELINE WITH MODEL RETRAINING                  ${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}⏱️  Estimated time: 2-3 hours${NC}"
echo -e "${YELLOW}📊 Retrains all 8 models + generates predictions${NC}"
echo -e "${RED}⚠️  WARNING: This will take significant time. Use './run_predictions_only.sh' for quick updates.${NC}"
echo ""
read -p "Continue with full retraining? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cancelled. Use './run_predictions_only.sh' for quick updates.${NC}"
    exit 0
fi

START_TIME=$(date +%s)

# Step 1: Collect latest stock data
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 1/7: Collecting latest stock data (Jan 2019 to yesterday)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python collect_stock_data.py

# Step 2: Calculate technical indicators
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 2/7: Calculating technical indicators (~40 indicators)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python calculate_technical_indicators.py

# Step 3: Prepare features for modeling
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 3/7: Preparing features for modeling${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python prepare_features_for_modeling.py

# Step 4: Prepare enhanced features
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 4/7: Preparing enhanced features (35 features)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python prepare_enhanced_features.py

# Step 5: Train all models (THIS IS THE LONG STEP - 2-3 hours)
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 5/7: Training all 8 V5 Transformer models${NC}"
echo -e "${YELLOW}⏱️  This will take 2-3 hours... Go grab a coffee! ☕${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 ./venv/bin/python train_all_v5_transformer.py

# Step 6: Generate predictions
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 6/7: Generating predictions (using newly trained models)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 ./venv/bin/python generate_daily_predictions.py

# Step 7: Deploy to Vercel
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 7/7: Deploying to Vercel${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
cd web && vercel --prod --yes

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Success message
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}                              ✅ FULL PIPELINE COMPLETE!                                      ${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}✅ Latest stock data collected${NC}"
echo -e "${GREEN}✅ Technical indicators calculated${NC}"
echo -e "${GREEN}✅ Features prepared${NC}"
echo -e "${GREEN}✅ All 8 models retrained${NC}"
echo -e "${GREEN}✅ Predictions generated${NC}"
echo -e "${GREEN}✅ Deployed to Vercel${NC}"
echo ""
echo -e "${YELLOW}⏱️  Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${BLUE}🌐 Your website: https://web-p1tce1b68-rohith-koripellis-projects.vercel.app${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "   • Check model results: cat models/saved_v5_all/all_stocks_summary.json"
echo -e "   • Check predictions: cat future_predictions_next_day.csv"
echo -e "   • Weekly updates: Use './run_predictions_only.sh' (5-10 mins)"
echo -e "   • Monthly retrain: Run this script again"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
