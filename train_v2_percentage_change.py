import pickle
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras import callbacks
import sys
sys.path.append('models')
from attention_stock_predictor import create_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING V2 - PERCENTAGE CHANGE APPROACH".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

TICKER = 'HDFCBANK'
STOCK_NAME = 'HDFC Bank'

EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
LEARNING_RATE = 0.001

MODEL_CONFIG = {
    'lstm_units': [128, 64, 32],
    'dense_units': [32, 16],
    'attention_units': 64,
    'dropout_rate': 0.2,
    'learning_rate': LEARNING_RATE
}

# =============================================================================
# LOAD DATA
# =============================================================================

print(f"\n1. Loading V2 data (percentage change approach)...")
with open(f'data/model_ready_v2/{TICKER}_features.pkl', 'rb') as f:
    data = pickle.load(f)

X_tech_train = data['train']['X_technical']
X_sent_train = data['train']['X_sentiment']
y_train = data['train']['y']  # Percentage changes
base_prices_train = data['train']['base_prices']

X_tech_val = data['val']['X_technical']
X_sent_val = data['val']['X_sentiment']
y_val = data['val']['y']
base_prices_val = data['val']['base_prices']

X_tech_test = data['test']['X_technical']
X_sent_test = data['test']['X_sentiment']
y_test = data['test']['y']
base_prices_test = data['test']['base_prices']
dates_test = data['test']['dates']

print(f"   ✓ Train: {len(y_train)} samples")
print(f"   ✓ Val: {len(y_val)} samples")
print(f"   ✓ Test: {len(y_test)} samples")
print(f"   ✓ Target: Percentage change to next day's close")
print(f"   ✓ Technical features: {X_tech_train.shape[2]}")
print(f"   ✓ Sentiment features: {X_sent_train.shape[2]}")

# =============================================================================
# CREATE MODEL
# =============================================================================

print(f"\n2. Building model...")
model, predictor = create_model(
    n_timesteps=X_tech_train.shape[1],
    n_technical_features=X_tech_train.shape[2],
    n_sentiment_features=X_sent_train.shape[2],
    **MODEL_CONFIG
)

print(f"   ✓ Model created with {model.count_params():,} parameters")

# =============================================================================
# SETUP CALLBACKS
# =============================================================================

print(f"\n3. Setting up callbacks...")

os.makedirs(f'models/saved_v2/{TICKER}', exist_ok=True)

checkpoint = callbacks.ModelCheckpoint(
    f'models/saved_v2/{TICKER}/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

print(f"   ✓ Callbacks configured")

# =============================================================================
# TRAIN MODEL
# =============================================================================

print(f"\n4. Training model (V2 - percentage change prediction)...")
print(f"   Max epochs: {EPOCHS}, Patience: {PATIENCE}")

start_time = datetime.now()

history = model.fit(
    [X_tech_train, X_sent_train],
    y_train,
    validation_data=([X_tech_val, X_sent_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=2
)

train_time = (datetime.now() - start_time).total_seconds()

print(f"\n   ✓ Training complete: {train_time/60:.1f} minutes")
print(f"   ✓ Epochs: {len(history.history['loss'])}")
print(f"   ✓ Best val_loss: {min(history.history['val_loss']):.6f}")
print(f"   ✓ Best val_mae: {min(history.history['val_mae']):.6f}")
print(f"   ✓ Best val_mape: {min(history.history['val_mape']):.2f}%")

# =============================================================================
# EVALUATE ON TEST SET
# =============================================================================

print(f"\n5. Evaluating on test set...")

# Predict percentage changes
y_pred_pct = model.predict([X_tech_test, X_sent_test], verbose=0).flatten()

# Convert percentage changes back to absolute prices
y_pred_prices = base_prices_test * (1 + y_pred_pct / 100)
y_true_prices = base_prices_test * (1 + y_test / 100)

# Calculate metrics on absolute prices
rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
mae = mean_absolute_error(y_true_prices, y_pred_prices)
mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
r2 = r2_score(y_true_prices, y_pred_prices)

# Directional accuracy
direction_true = y_test > 0  # True price went up
direction_pred = y_pred_pct > 0  # Predicted price goes up
directional_accuracy = np.mean(direction_true == direction_pred) * 100

print(f"\n   TEST SET METRICS (Absolute Prices):")
print(f"   ===================================")
print(f"   RMSE: ₹{rmse:.2f}")
print(f"   MAE:  ₹{mae:.2f}")
print(f"   MAPE: {mape:.2f}%")
print(f"   R²:   {r2:.4f}")
print(f"   Directional Accuracy: {directional_accuracy:.2f}%")

# Metrics on percentage changes
pct_mse = mean_squared_error(y_test, y_pred_pct)
pct_mae = mean_absolute_error(y_test, y_pred_pct)

print(f"\n   PERCENTAGE CHANGE METRICS:")
print(f"   =========================")
print(f"   MSE:  {pct_mse:.4f}")
print(f"   MAE:  {pct_mae:.4f}%")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results = {
    'stock': STOCK_NAME,
    'ticker': TICKER,
    'approach': 'percentage_change_v2',
    'test_metrics': {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape),
        'R2': float(r2),
        'Directional_Accuracy': float(directional_accuracy)
    },
    'percentage_metrics': {
        'MSE': float(pct_mse),
        'MAE': float(pct_mae)
    },
    'training_info': {
        'epochs_run': len(history.history['loss']),
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_mae': float(min(history.history['val_mae'])),
        'best_val_mape': float(min(history.history['val_mape'])),
        'train_time_seconds': train_time
    }
}

with open(f'models/saved_v2/{TICKER}/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save predictions
import pandas as pd
predictions_df = pd.DataFrame({
    'Date': dates_test,
    'Base_Price': base_prices_test,
    'Actual_Price': y_true_prices,
    'Predicted_Price': y_pred_prices,
    'Actual_Pct_Change': y_test,
    'Predicted_Pct_Change': y_pred_pct,
    'Error': y_true_prices - y_pred_prices,
    'Abs_Error': np.abs(y_true_prices - y_pred_prices)
})
predictions_df.to_csv(f'models/saved_v2/{TICKER}/predictions.csv', index=False)

# =============================================================================
# PLOT RESULTS
# =============================================================================

print(f"\n6. Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Price predictions
ax = axes[0, 0]
ax.plot(dates_test, y_true_prices, label='Actual', color='blue', linewidth=2, alpha=0.7)
ax.plot(dates_test, y_pred_prices, label='Predicted', color='red', linewidth=2, alpha=0.7)
ax.set_title(f'{STOCK_NAME} - Price Prediction (V2)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Price (₹)')
ax.legend()
ax.grid(alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# 2. Scatter plot
ax = axes[0, 1]
ax.scatter(y_true_prices, y_pred_prices, alpha=0.5, s=20)
ax.plot([y_true_prices.min(), y_true_prices.max()],
        [y_true_prices.min(), y_true_prices.max()],
        'r--', lw=2, label='Perfect')
ax.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontweight='bold', fontsize=14)
ax.set_xlabel('Actual Price (₹)')
ax.set_ylabel('Predicted Price (₹)')
ax.legend()
ax.grid(alpha=0.3)

# 3. Percentage change predictions
ax = axes[1, 0]
ax.scatter(y_test, y_pred_pct, alpha=0.5, s=20, color='green')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect')
ax.set_title('Percentage Change Prediction', fontweight='bold', fontsize=14)
ax.set_xlabel('Actual % Change')
ax.set_ylabel('Predicted % Change')
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

# 4. Training history
ax = axes[1, 1]
ax.plot(history.history['loss'], label='Train Loss', color='blue')
ax.plot(history.history['val_loss'], label='Val Loss', color='orange')
ax.set_title('Training History', fontweight='bold', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'models/saved_v2/{TICKER}/evaluation.png', dpi=200, bbox_inches='tight')
print(f"   ✓ Visualizations saved")

# =============================================================================
# COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON WITH PREVIOUS APPROACHES".center(80))
print("=" * 80)

print(f"\n{'Metric':<25} {'Old V1':<15} {'V2 (Pct Change)':<15}")
print("=" * 80)
print(f"{'R²':<25} {'-3.31':<15} {f'{r2:.4f}':<15}")
print(f"{'MAPE':<25} {'12.58%':<15} {f'{mape:.2f}%':<15}")
print(f"{'Directional Accuracy':<25} {'48.77%':<15} {f'{directional_accuracy:.2f}%':<15}")
print(f"{'Prediction Pattern':<25} {'Flat line':<15} {'Dynamic':<15}")
print("=" * 80)

if r2 > 0:
    print(f"\n✓✓✓ SUCCESS! R² is POSITIVE - model actually learns! ✓✓✓")
    improvement = ((r2 - (-3.31)) / abs(-3.31)) * 100
    print(f"✓✓✓ Improvement: {improvement:.1f}% better than baseline! ✓✓✓")
else:
    print(f"\n⚠ R² still negative, but might be better than {-3.31:.2f}")

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETE!".center(80))
print("=" * 80)

print(f"\nResults saved to:")
print(f"  • models/saved_v2/{TICKER}/results.json")
print(f"  • models/saved_v2/{TICKER}/predictions.csv")
print(f"  • models/saved_v2/{TICKER}/evaluation.png")
