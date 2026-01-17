#!/bin/bash

################################################################################
# Stock Price Predictions - Quick Update (Uses Existing Models)
#
# What it does:
# 1. Collects latest stock data (until T-1)
# 2. Calculates technical indicators
# 3. Prepares enhanced features
# 4. Generates predictions (using existing trained models)
# 5. Deploys to Vercel
#
# When to use: Weekly updates (5-10 minutes)
# Prerequisites: Models already trained (models/saved_v5_all/)
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
echo -e "${BLUE}                    STOCK PREDICTIONS UPDATE - USING EXISTING MODELS                           ${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}⏱️  Estimated time: 5-10 minutes${NC}"
echo -e "${YELLOW}📊 Updates predictions without retraining models${NC}"
echo ""

# Check if models exist
if [ ! -d "models/saved_v5_all/HDFCBANK" ]; then
    echo -e "${RED}❌ ERROR: Models not found!${NC}"
    echo -e "${YELLOW}Please run './run_full_pipeline.sh' first to train models${NC}"
    exit 1
fi

START_TIME=$(date +%s)

# Step 1: Collect latest stock data
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 1/5: Collecting latest stock data (Jan 2019 to yesterday)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python collect_stock_data.py

# Step 2: Calculate technical indicators
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 2/5: Calculating technical indicators (~40 indicators)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python calculate_technical_indicators.py

# Step 3: Prepare enhanced features
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 3/5: Preparing enhanced features (35 features)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
./venv/bin/python prepare_enhanced_features.py

# Step 4: Generate predictions
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 4/5: Generating predictions (using existing models)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=2 ./venv/bin/python generate_daily_predictions.py

# Step 5: Deploy to Vercel
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 5/5: Deploying to Vercel${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
cd web && vercel --prod --yes

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Success message
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}                                   ✅ UPDATE COMPLETE!                                         ${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}✅ Latest stock data collected${NC}"
echo -e "${GREEN}✅ Technical indicators calculated${NC}"
echo -e "${GREEN}✅ Enhanced features prepared${NC}"
echo -e "${GREEN}✅ Predictions generated${NC}"
echo -e "${GREEN}✅ Deployed to Vercel${NC}"
echo ""
echo -e "${YELLOW}⏱️  Total time: ${MINUTES}m ${SECONDS}s${NC}"
echo ""
echo -e "${BLUE}🌐 Your website: https://web-p1tce1b68-rohith-koripellis-projects.vercel.app${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "   • Check predictions: cat future_predictions_next_day.csv"
echo -e "   • Run this script weekly to update predictions"
echo -e "   • Run './run_full_pipeline.sh' monthly to retrain models"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
