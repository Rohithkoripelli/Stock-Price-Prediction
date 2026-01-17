#!/bin/bash

# Test Automation Pipeline Locally
echo "================================================"
echo "Testing Automated Prediction Pipeline"
echo "================================================"

# Step 1: Test data collection
echo ""
echo "Step 1: Testing data collection..."
echo "--------------------"
python daily_data_collection.py

if [ $? -ne 0 ]; then
    echo "❌ Data collection failed!"
    exit 1
fi

echo "✅ Data collection successful"

# Step 2: Test prediction generation
echo ""
echo "Step 2: Testing prediction generation..."
echo "--------------------"
python generate_daily_predictions.py

if [ $? -ne 0 ]; then
    echo "❌ Prediction generation failed!"
    exit 1
fi

echo "✅ Prediction generation successful"

# Step 3: Verify outputs
echo ""
echo "Step 3: Verifying outputs..."
echo "--------------------"

if [ -f "future_predictions_next_day.json" ]; then
    echo "✅ Predictions JSON created"
    echo "Preview:"
    head -20 future_predictions_next_day.json
else
    echo "❌ Predictions JSON not found!"
    exit 1
fi

if [ -f "web/future_predictions_next_day.json" ]; then
    echo "✅ Web predictions updated"
else
    echo "⚠️  Web predictions not copied (web directory may not exist locally)"
fi

# Step 4: Show summary
echo ""
echo "================================================"
echo "Test Complete! Summary:"
echo "================================================"

if [ -f "prediction_metadata.json" ]; then
    cat prediction_metadata.json
fi

echo ""
echo "✅ All tests passed!"
echo ""
echo "To deploy to Vercel:"
echo "  cd web && vercel --prod --yes"
echo ""
echo "Or commit and push to trigger GitHub Actions:"
echo "  git add ."
echo "  git commit -m 'Test automation'"
echo "  git push"
