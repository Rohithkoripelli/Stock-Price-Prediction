# Stock Price Prediction Model - Improvements Summary

## Date: December 10, 2025

## Critical Issues Identified

### 1. **Negative R² Scores** (Model Worse Than Baseline)
- HDFC Bank: R² = **-3.31** (TERRIBLE)
- ICICI Bank: R² = **-12.31** (WORSE)
- SBIN: R² = **-5.97** (TERRIBLE)

**Problem**: Negative R² means the model was performing worse than just predicting the average price!

### 2. **Flat Predictions** (Severe Underfitting)
- Actual HDFC prices: 800-860 (varying)
- Predicted prices: 780-806 (almost flat line)
- The model was barely adjusting predictions

### 3. **Over-Regularization**
The model had excessive regularization:
- 6x Layer Normalizations
- 6x Dropout layers (rate=0.2)
- 5x L2 Regularization (0.001)
- 3x Recurrent Dropout (0.1)
- Gradient Clipping (1.0 - too restrictive)

This caused severe underfitting!

### 4. **Scaling Problems**
- Used MinMaxScaler for stock prices (problematic when test prices outside training range)
- Scaling not robust to outliers
- Target scaling too sensitive

### 5. **Sentiment Feature Issues**
- Missing sentiment filled with 0 (not realistic)
- Should forward-fill (sentiment persists until new news)

### 6. **Loss Function**
- Used Huber loss (designed for outliers)
- Too conservative for normal stock price variations

---

## Improvements Implemented

### 1. **Preprocessing** (`prepare_features_for_modeling.py`)

#### Changed Scalers:
```python
# OLD (PROBLEMATIC):
scaler_tech = MinMaxScaler()
scaler_sent = MinMaxScaler()
scaler_target = MinMaxScaler()  # ❌ CRITICAL ISSUE

# NEW (IMPROVED):
scaler_tech = StandardScaler()      # ✅ Better for neural networks
scaler_sent = StandardScaler()      # ✅ Better for neural networks
scaler_target = RobustScaler()      # ✅ CRITICAL: Better for stock prices, handles outliers
```

**Why?**
- StandardScaler: Mean=0, std=1 (better for neural networks)
- RobustScaler: Uses median and IQR (less sensitive to extreme values)
- MinMaxScaler is sensitive to outliers and test data outside training range

#### Improved Sentiment Handling:
```python
# OLD:
sentiment_df[col] = sentiment_df[col].fillna(0)  # ❌ Unrealistic

# NEW:
sentiment_df[col] = sentiment_df[col].fillna(method='ffill').fillna(0)  # ✅ Forward-fill
```

**Why?** Sentiment persists until new news arrives, not reset to neutral.

---

### 2. **Model Architecture** (`models/attention_stock_predictor.py`)

#### Reduced Over-Regularization:

##### Layer Normalization:
```python
# OLD: 6 Layer Normalizations (suppressed signal)
# NEW: 0 Layer Normalizations (removed all) ✅
```

##### Dropout:
```python
# OLD:
x_tech = layers.Dropout(0.2)(x_tech)  # Too high

# NEW:
x_tech = layers.Dropout(self.dropout_rate * 0.5)(x_tech)  # ✅ Reduced to 0.1
```
**Reduction**: 50% less dropout (from 0.2 to 0.1)

##### L2 Regularization:
```python
# OLD:
kernel_regularizer=keras.regularizers.l2(0.001)  # Too strong

# NEW:
kernel_regularizer=keras.regularizers.l2(0.0001)  # ✅ Reduced by 90%
```

##### Recurrent Dropout:
```python
# OLD:
recurrent_dropout=0.1  # In all LSTM layers

# NEW:
recurrent_dropout=0.05  # ✅ Reduced in first two LSTMs
recurrent_dropout=0.0   # ✅ Removed from final LSTM
```

#### Changed Loss Function:
```python
# OLD:
loss='huber',  # Too conservative
clipnorm=1.0   # Too restrictive

# NEW:
loss='mse',    # ✅ Better for regression
clipnorm=5.0   # ✅ Allows larger updates
```

**Why?**
- MSE is standard for regression tasks
- Huber loss is for outlier-heavy data (not needed here)
- Increased clipnorm allows faster learning

---

## Expected Improvements

Based on these changes, you should see:

1. **Positive R² scores** (model beats baseline)
2. **Better MAPE** (lower percentage error)
3. **More dynamic predictions** (not flat lines)
4. **Higher directional accuracy** (better trend prediction)

---

## How to Retrain

### Step 1: Regenerate Features (ALREADY DONE)
```bash
python prepare_features_for_modeling.py
```
✅ **COMPLETED** - Features regenerated with new scalers

### Step 2: Train Models

#### Option A: Train Single Stock (Quick Test)
```bash
./venv/bin/python train_single_stock_improved.py
```

#### Option B: Train All 8 Stocks
```bash
./venv/bin/python train_attention_model.py
```

This will automatically use the improved architecture because the model file has been updated.

---

## Files Modified

1. ✅ `prepare_features_for_modeling.py`
   - Changed scalers (StandardScaler, RobustScaler)
   - Improved sentiment forward-filling

2. ✅ `models/attention_stock_predictor.py`
   - Removed layer normalizations
   - Reduced dropout (50% reduction)
   - Reduced L2 regularization (90% reduction)
   - Reduced recurrent dropout
   - Changed loss from Huber to MSE
   - Increased gradient clipping

3. ✅ `train_single_stock_improved.py` (NEW)
   - Test script for single stock training

---

## Comparison Table

| Metric | Old Model | Expected New |
|--------|-----------|--------------|
| **HDFC R²** | -3.31 | > 0.0 |
| **HDFC MAPE** | 12.58% | < 10% |
| **HDFC Dir. Acc** | 48.77% | > 50% |
| **Prediction Pattern** | Flat line | Dynamic |
| **Regularization** | EXTREME | Balanced |
| **Loss Function** | Huber | MSE |
| **Scaling** | MinMax | Standard/Robust |

---

## Next Steps

1. **Retrain in a fresh terminal** (to avoid TensorFlow session issues):
   ```bash
   cd "/Users/rohithkoripelli/M.Tech/Final Project/Stock_Price_Prediction"
   ./venv/bin/python train_single_stock_improved.py
   ```

2. **Check results**:
   - Training plots: `models/saved_improved/HDFCBANK/training_history.png`
   - Test metrics: `models/saved_improved/HDFCBANK/test_results.json`

3. **If improvements confirmed**, train all 8 stocks:
   ```bash
   ./venv/bin/python train_attention_model.py
   ```

4. **Evaluate all models**:
   ```bash
   ./venv/bin/python evaluate_model.py
   ```

---

## Key Takeaways

### What Was Wrong:
- **Over-regularization killed the model's ability to learn**
- **Wrong scalers** made test predictions unreliable
- **Huber loss** was too conservative

### What We Fixed:
- **Reduced regularization by 50-90%** across the board
- **Better scalers** (StandardScaler + RobustScaler)
- **MSE loss** for better regression performance
- **Forward-fill sentiment** for realistic feature engineering

### Expected Result:
- **Positive R²** (model actually learns)
- **Lower MAPE** (more accurate predictions)
- **Dynamic predictions** (follows price movements)

---

## Technical Details

### Regularization Comparison:

| Component | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Layer Norm | 6 instances | 0 instances | -100% |
| Dropout Rate | 0.2 | 0.1 | -50% |
| L2 Regularization | 0.001 | 0.0001 | -90% |
| Recurrent Dropout | 0.1 | 0.05/0.0 | -50%/-100% |
| Gradient Clipping | 1.0 | 5.0 | +400% |

### Scaling Comparison:

| Feature Type | Old Scaler | New Scaler | Benefit |
|--------------|-----------|------------|---------|
| Technical Features | MinMaxScaler | StandardScaler | Better for NN |
| Sentiment Features | MinMaxScaler | StandardScaler | Better for NN |
| Target (Price) | MinMaxScaler | RobustScaler | Handles outliers |

---

**Author**: Claude (AI Assistant)
**Date**: December 10, 2025
**Status**: Ready for retraining
