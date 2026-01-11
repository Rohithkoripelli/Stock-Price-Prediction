"""
Train V5 Transformer Models for All 8 Banking Stocks

Uses the advanced Transformer architecture with multi-task learning
that achieved 64% directional accuracy for HDFC Bank
"""

import pickle
import json
import os
import numpy as np
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras import callbacks, layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING V5 TRANSFORMER - ALL 8 STOCKS".center(80))
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

STOCKS = [
    ('HDFCBANK', 'HDFC Bank'),
    ('ICICIBANK', 'ICICI Bank'),
    ('KOTAKBANK', 'Kotak Mahindra Bank'),
    ('AXISBANK', 'Axis Bank'),
    ('SBIN', 'State Bank of India'),
    ('PNB', 'Punjab National Bank'),
    ('BANKBARODA', 'Bank of Baroda'),
    ('CANBK', 'Canara Bank')
]

EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 30
LEARNING_RATE = 0.0001

# =============================================================================
# TRANSFORMER ARCHITECTURE
# =============================================================================

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Transformer encoder block"""
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def create_multitask_transformer(input_shape):
    """Multi-task model: direction classification + magnitude regression"""
    inputs = layers.Input(shape=input_shape)
    
    # Transformer blocks
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.2)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Shared dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Task 1: Direction classification
    direction_output = layers.Dense(32, activation='relu', name='direction_dense')(x)
    direction_output = layers.Dropout(0.2)(direction_output)
    direction_output = layers.Dense(1, activation='sigmoid', name='direction')(direction_output)
    
    # Task 2: Magnitude regression
    magnitude_output = layers.Dense(32, activation='relu', name='magnitude_dense')(x)
    magnitude_output = layers.Dropout(0.2)(magnitude_output)
    magnitude_output = layers.Dense(1, activation='linear', name='magnitude')(magnitude_output)
    
    model = models.Model(inputs=inputs, outputs=[direction_output, magnitude_output])
    
    return model

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_transformer_model(ticker, stock_name):
    """Train V5 Transformer model for a single stock"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {stock_name} ({ticker})".center(80))
    print("="*80)
    
    try:
        # Load data
        print(f"\n1. Loading enhanced data...")
        with open(f'data/enhanced_model_ready/{ticker}_enhanced.pkl', 'rb') as f:
            data = pickle.load(f)
        
        X_train = data['train']['X']
        y_train = data['train']['y']
        base_prices_train = data['train']['base_prices']
        
        X_val = data['val']['X']
        y_val = data['val']['y']
        base_prices_val = data['val']['base_prices']
        
        X_test = data['test']['X']
        y_test = data['test']['y']
        base_prices_test = data['test']['base_prices']
        dates_test = data['test']['dates']
        
        n_features = X_train.shape[2]
        n_timesteps = X_train.shape[1]
        
        print(f"   ✓ Features: {n_features}, Timesteps: {n_timesteps}")
        print(f"   ✓ Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        # Create direction labels
        y_train_direction = (y_train > 0).astype(int)
        y_val_direction = (y_val > 0).astype(int)
        y_test_direction = (y_test > 0).astype(int)
        
        # Build model
        print(f"\n2. Building Transformer model...")
        
        model = create_multitask_transformer((n_timesteps, n_features))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss={
                'direction': 'binary_crossentropy',
                'magnitude': 'huber'
            },
            loss_weights={
                'direction': 0.7,  # 70% weight on direction
                'magnitude': 0.3
            },
            metrics={
                'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
                'magnitude': ['mae']
            }
        )
        
        print(f"   ✓ Model with {model.count_params():,} parameters")
        
        # Setup callbacks
        print(f"\n3. Training model...")
        
        os.makedirs(f'models/saved_v5_all/{ticker}', exist_ok=True)
        
        checkpoint = callbacks.ModelCheckpoint(
            f'models/saved_v5_all/{ticker}/best_model.keras',
            monitor='val_direction_accuracy',
            save_best_only=True,
            mode='max',
            verbose=0
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_direction_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=0
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_direction_accuracy',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            mode='max',
            verbose=0
        )
        
        start_time = datetime.now()
        
        history = model.fit(
            X_train,
            {'direction': y_train_direction, 'magnitude': y_train},
            validation_data=(X_val, {'direction': y_val_direction, 'magnitude': y_val}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=0
        )
        
        train_time = (datetime.now() - start_time).total_seconds()
        
        best_val_dir_acc = max(history.history['val_direction_accuracy']) * 100
        
        print(f"   ✓ Training complete: {train_time/60:.1f} minutes")
        print(f"   ✓ Epochs: {len(history.history['loss'])}")
        print(f"   ✓ Best val direction accuracy: {best_val_dir_acc:.2f}%")
        
        # Evaluate
        print(f"\n4. Evaluating on test set...")
        
        predictions = model.predict(X_test, verbose=0)
        y_pred_direction_prob = predictions[0].flatten()
        y_pred_magnitude = predictions[1].flatten()
        
        y_pred_direction = (y_pred_direction_prob > 0.5).astype(int)
        
        # Directional accuracy
        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        
        # Reconstruct percentage change
        y_pred_pct = np.where(y_pred_direction == 1, 
                              np.abs(y_pred_magnitude),
                              -np.abs(y_pred_magnitude))
        
        # Price predictions
        y_pred_prices = base_prices_test * (1 + y_pred_pct / 100)
        y_true_prices = base_prices_test * (1 + y_test / 100)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
        mae = mean_absolute_error(y_true_prices, y_pred_prices)
        mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
        r2 = r2_score(y_true_prices, y_pred_prices)
        
        # Significant moves
        significant_moves = np.abs(y_test) > 0.5
        if np.sum(significant_moves) > 0:
            dir_acc_significant = np.mean(
                y_test_direction[significant_moves] == y_pred_direction[significant_moves]
            ) * 100
        else:
            dir_acc_significant = 0.0
        
        print(f"\n   TEST SET METRICS:")
        print(f"   RMSE: ₹{rmse:.2f}")
        print(f"   MAE:  ₹{mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²:   {r2:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   Dir. Acc (>0.5%): {dir_acc_significant:.2f}%")
        
        # Save results
        results = {
            'stock': stock_name,
            'ticker': ticker,
            'approach': 'v5_transformer_multitask',
            'n_features': int(n_features),
            'test_metrics': {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'MAPE': float(mape),
                'R2': float(r2),
                'Directional_Accuracy': float(directional_accuracy),
                'Directional_Accuracy_Significant': float(dir_acc_significant)
            },
            'training': {
                'epochs': len(history.history['loss']),
                'time_minutes': float(train_time/60),
                'best_val_dir_acc': float(best_val_dir_acc)
            }
        }
        
        with open(f'models/saved_v5_all/{ticker}/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save predictions
        import pandas as pd
        predictions_df = pd.DataFrame({
            'Date': dates_test,
            'Actual_Direction': y_test_direction,
            'Predicted_Direction': y_pred_direction,
            'Direction_Probability': y_pred_direction_prob,
            'Actual_Pct_Change': y_test,
            'Predicted_Pct_Change': y_pred_pct,
            'Correct_Direction': y_test_direction == y_pred_direction,
            'Actual_Price': y_true_prices,
            'Predicted_Price': y_pred_prices
        })
        predictions_df.to_csv(f'models/saved_v5_all/{ticker}/predictions.csv', index=False)
        
        print(f"   ✓ Results saved to models/saved_v5_all/{ticker}/")
        
        return results
        
    except Exception as e:
        print(f"\n   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# TRAIN ALL STOCKS
# =============================================================================

all_results = []

for ticker, stock_name in STOCKS:
    result = train_transformer_model(ticker, stock_name)
    if result:
        all_results.append(result)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n\n" + "=" * 80)
print("V5 TRANSFORMER - ALL STOCKS SUMMARY".center(80))
print("=" * 80)

print(f"\n{'Stock':<25} {'MAPE':<10} {'R²':<10} {'Dir. Acc':<12} {'Epochs':<10}")
print("=" * 70)

for result in all_results:
    print(f"{result['stock']:<25} "
          f"{result['test_metrics']['MAPE']:<10.2f} "
          f"{result['test_metrics']['R2']:<10.4f} "
          f"{result['test_metrics']['Directional_Accuracy']:<12.2f} "
          f"{result['training']['epochs']:<10}")

print("=" * 70)

# Calculate averages
avg_mape = np.mean([r['test_metrics']['MAPE'] for r in all_results])
avg_r2 = np.mean([r['test_metrics']['R2'] for r in all_results])
avg_dir_acc = np.mean([r['test_metrics']['Directional_Accuracy'] for r in all_results])

print(f"\n{'AVERAGE':<25} {avg_mape:<10.2f} {avg_r2:<10.4f} {avg_dir_acc:<12.2f}")

# Save summary
summary = {
    'training_date': datetime.now().isoformat(),
    'approach': 'v5_transformer_multitask_all_stocks',
    'architecture': 'Transformer with multi-head attention + multi-task learning',
    'stocks_trained': len(all_results),
    'average_metrics': {
        'MAPE': float(avg_mape),
        'R2': float(avg_r2),
        'Directional_Accuracy': float(avg_dir_acc)
    },
    'individual_results': all_results
}

with open('models/saved_v5_all/all_stocks_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Summary saved to: models/saved_v5_all/all_stocks_summary.json")

print("\n" + "=" * 80)
print("✓ V5 TRANSFORMER TRAINING COMPLETE!".center(80))
print("=" * 80)
