#!/bin/bash

# Update Stock Price Predictions and Deploy
echo "================================================"
echo "Stock Price Prediction Updater"
echo "================================================"

# Step 1: Generate new predictions
echo ""
echo "Step 1: Generating new predictions..."
python generate_future_predictions.py

# Check if generation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate predictions"
    exit 1
fi

# Step 2: Copy predictions to web directory
echo ""
echo "Step 2: Copying predictions to web directory..."
cp future_predictions_next_day.json web/

# Step 3: Commit and push changes
echo ""
echo "Step 3: Committing changes to git..."
git add future_predictions_next_day.json web/future_predictions_next_day.json
git commit -m "Update predictions - $(date '+%Y-%m-%d %H:%M:%S')"
git push

# Step 4: Deploy to Vercel
echo ""
echo "Step 4: Deploying to Vercel..."
cd web
vercel --prod --yes

echo ""
echo "================================================"
echo "✓ Predictions updated and deployed successfully!"
echo "================================================"
