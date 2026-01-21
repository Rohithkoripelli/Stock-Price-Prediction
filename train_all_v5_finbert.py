"""
Train V5 Transformer Models with FinBERT Features for All 8 Banking Stocks

Enhanced version that uses FinBERT sentiment analysis combined with technical indicators
to improve prediction confidence and accuracy.

Key improvements over VADER:
- Financial domain-specific sentiment (not general-purpose)
- Detects earnings events, quarterly results, regulatory news
- Better context understanding of financial terminology
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
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING V5 TRANSFORMER WITH FINBERT - ALL 8 STOCKS".center(80))
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

# Quick training for overnight run
EPOCHS = 100  # Reduced from 200 for faster training
BATCH_SIZE = 32
PATIENCE = 20  # Reduced from 30
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
    """Train V5 Transformer model with FinBERT features for a single stock"""

    print(f"\n{'='*80}")
    print(f"TRAINING: {stock_name} ({ticker})".center(80))
    print("="*80)

    try:
        # Load FinBERT-enhanced data
        print(f"\n1. Loading FinBERT-enhanced data...")
        pkl_file = f'data/finbert_model_ready/{ticker}_finbert.pkl'

        if not os.path.exists(pkl_file):
            print(f"   ✗ FinBERT data not found: {pkl_file}")
            return None

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        X_train = data['X_train']
        y_train = data['y_train']

        X_val = data['X_val']
        y_val = data['y_val']

        X_test = data['X_test']
        y_test = data['y_test']

        n_features = data['num_features']
        n_timesteps = data['lookback']
        feature_names = data['feature_names']

        print(f"   ✓ Features: {n_features} (includes FinBERT sentiment)")
        print(f"   ✓ Timesteps: {n_timesteps}")
        print(f"   ✓ Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        # Identify FinBERT features
        finbert_features = [f for f in feature_names if 'sentiment' in f.lower() or
                           'news' in f.lower() or 'earnings' in f.lower()]
        if finbert_features:
            print(f"   ✓ FinBERT features detected: {', '.join(finbert_features)}")

        # Create direction labels (UP if positive change, DOWN if negative)
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
                'direction': 0.7,  # 70% weight on direction (key for confidence)
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

        os.makedirs(f'models/saved_v5_finbert/{ticker}', exist_ok=True)

        checkpoint = callbacks.ModelCheckpoint(
            f'models/saved_v5_finbert/{ticker}/best_model.keras',
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
            patience=10,
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
        best_val_auc = max(history.history['val_direction_auc']) * 100

        print(f"   ✓ Training complete: {train_time/60:.1f} minutes")
        print(f"   ✓ Epochs: {len(history.history['loss'])}")
        print(f"   ✓ Best val direction accuracy: {best_val_dir_acc:.2f}%")
        print(f"   ✓ Best val AUC: {best_val_auc:.2f}%")

        # Evaluate on test set
        print(f"\n4. Evaluating on test set...")

        predictions = model.predict(X_test, verbose=0)
        y_pred_direction_prob = predictions[0].flatten()
        y_pred_magnitude = predictions[1].flatten()

        # Convert probabilities to binary predictions
        y_pred_direction = (y_pred_direction_prob > 0.5).astype(int)

        # Calculate confidence score (how sure the model is)
        # Convert to 0-100% where 50% = uncertain, 100% = very confident
        confidence_scores = np.abs(y_pred_direction_prob - 0.5) * 2 * 100
        avg_confidence = np.mean(confidence_scores)

        # Directional accuracy
        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100

        # Analyze high-confidence predictions (>70% confident)
        high_conf_mask = confidence_scores > 70
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = np.mean(
                y_test_direction[high_conf_mask] == y_pred_direction[high_conf_mask]
            ) * 100
            high_conf_count = np.sum(high_conf_mask)
        else:
            high_conf_accuracy = 0.0
            high_conf_count = 0

        # Significant moves (>0.5% change)
        significant_moves = np.abs(y_test) > 0.5
        if np.sum(significant_moves) > 0:
            dir_acc_significant = np.mean(
                y_test_direction[significant_moves] == y_pred_direction[significant_moves]
            ) * 100
        else:
            dir_acc_significant = 0.0

        print(f"\n   TEST SET METRICS:")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   Average Confidence: {avg_confidence:.2f}%")
        print(f"   High Confidence (>70%): {high_conf_count}/{len(y_test)} predictions")
        print(f"   High Conf Accuracy: {high_conf_accuracy:.2f}%")
        print(f"   Dir. Acc (>0.5% moves): {dir_acc_significant:.2f}%")

        # Save model metadata
        results = {
            'stock': stock_name,
            'ticker': ticker,
            'approach': 'v5_transformer_finbert',
            'n_features': int(n_features),
            'finbert_features': finbert_features,
            'test_metrics': {
                'Directional_Accuracy': float(directional_accuracy),
                'Average_Confidence': float(avg_confidence),
                'High_Confidence_Count': int(high_conf_count),
                'High_Confidence_Accuracy': float(high_conf_accuracy),
                'Directional_Accuracy_Significant': float(dir_acc_significant)
            },
            'training': {
                'epochs': len(history.history['loss']),
                'time_minutes': float(train_time/60),
                'best_val_dir_acc': float(best_val_dir_acc),
                'best_val_auc': float(best_val_auc)
            }
        }

        with open(f'models/saved_v5_finbert/{ticker}/results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save scaler for inference
        import shutil
        scaler_src = f'data/finbert_model_ready/{ticker}_finbert.pkl'
        scaler_dst = f'models/saved_v5_finbert/{ticker}/scaler.pkl'

        # Extract and save just the scaler
        with open(scaler_src, 'rb') as f:
            full_data = pickle.load(f)

        scaler_data = {
            'scaler': full_data['scaler'],
            'feature_names': full_data['feature_names'],
            'num_features': full_data['num_features'],
            'lookback': full_data['lookback']
        }

        with open(scaler_dst, 'wb') as f:
            pickle.dump(scaler_data, f)

        print(f"   ✓ Results saved to models/saved_v5_finbert/{ticker}/")

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
print("V5 TRANSFORMER WITH FINBERT - ALL STOCKS SUMMARY".center(80))
print("=" * 80)

if all_results:
    print(f"\n{'Stock':<25} {'Dir. Acc':<12} {'Avg Conf':<12} {'High Conf Acc':<15} {'Epochs':<10}")
    print("=" * 80)

    for result in all_results:
        print(f"{result['stock']:<25} "
              f"{result['test_metrics']['Directional_Accuracy']:<12.2f} "
              f"{result['test_metrics']['Average_Confidence']:<12.2f} "
              f"{result['test_metrics']['High_Confidence_Accuracy']:<15.2f} "
              f"{result['training']['epochs']:<10}")

    print("=" * 80)

    # Calculate averages
    avg_dir_acc = np.mean([r['test_metrics']['Directional_Accuracy'] for r in all_results])
    avg_confidence = np.mean([r['test_metrics']['Average_Confidence'] for r in all_results])
    avg_high_conf_acc = np.mean([r['test_metrics']['High_Confidence_Accuracy'] for r in all_results])

    print(f"\n{'AVERAGE':<25} {avg_dir_acc:<12.2f} {avg_confidence:<12.2f} {avg_high_conf_acc:<15.2f}")

    # Save summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'approach': 'v5_transformer_finbert_all_stocks',
        'architecture': 'Transformer with FinBERT sentiment features',
        'stocks_trained': len(all_results),
        'average_metrics': {
            'Directional_Accuracy': float(avg_dir_acc),
            'Average_Confidence': float(avg_confidence),
            'High_Confidence_Accuracy': float(avg_high_conf_acc)
        },
        'individual_results': all_results
    }

    with open('models/saved_v5_finbert/all_stocks_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: models/saved_v5_finbert/all_stocks_summary.json")

    print("\n" + "=" * 80)
    print("✓ V5 TRANSFORMER WITH FINBERT TRAINING COMPLETE!".center(80))
    print("=" * 80)
    print(f"\nKey Improvements:")
    print(f"  - Average Directional Accuracy: {avg_dir_acc:.2f}%")
    print(f"  - Average Confidence: {avg_confidence:.2f}%")
    print(f"  - Models saved to: models/saved_v5_finbert/")
    print(f"\nNext steps:")
    print(f"  1. Compare with VADER-based predictions")
    print(f"  2. If satisfied, upload to HuggingFace")
    print(f"  3. Update GitHub Actions workflow")
else:
    print("\n✗ No stocks trained successfully")
    print("Check that FinBERT feature preparation completed successfully")
