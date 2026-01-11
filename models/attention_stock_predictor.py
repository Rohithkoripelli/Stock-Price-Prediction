"""
Hierarchical Attention-Based Neural Network for Stock Price Prediction
CORRECTED VERSION - Fixes sentiment input shape and loss function
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np

class AttentionLayer(layers.Layer):
    """Attention mechanism layer"""
    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W_q = self.add_weight(
            name='W_q',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_k = self.add_weight(
            name='W_k',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='W_v',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(attention_weights, V)
        
        return attended, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class HierarchicalAttentionStockPredictor:
    """Hierarchical Attention-Based Stock Price Prediction Model"""
    
    def __init__(self, 
                 n_timesteps=60,
                 n_technical_features=20,
                 n_sentiment_features=5,
                 lstm_units=[128, 64, 32],
                 dense_units=[32, 16],
                 attention_units=64,
                 dropout_rate=0.2,
                 learning_rate=0.001):
        
        self.n_timesteps = n_timesteps
        self.n_technical_features = n_technical_features
        self.n_sentiment_features = n_sentiment_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.attention_model = None
    
    def build_model(self):
        """Build the hierarchical attention-based model"""
        
        # =====================================================================
        # INPUT LAYERS
        # =====================================================================
        
        # Technical features input (batch_size, timesteps, n_technical_features)
        input_technical = layers.Input(
            shape=(self.n_timesteps, self.n_technical_features),
            name='technical_input'
        )
        
        # ✅ FIXED: Sentiment as 2D instead of 3D
        # Sentiment features input (batch_size, timesteps, n_sentiment_features)
        input_sentiment = layers.Input(
            shape=(self.n_timesteps, self.n_sentiment_features),
            name='sentiment_input'
        )
        
        # =====================================================================
        # TECHNICAL FEATURES BRANCH (LSTM + Attention)
        # =====================================================================

        # ✅ IMPROVED: Removed input layer normalization (let data speak for itself)
        x_tech = input_technical

        # First LSTM layer - ✅ REDUCED recurrent_dropout from 0.1 to 0.05
        x_tech = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            recurrent_dropout=0.05,  # ✅ REDUCED
            name='technical_lstm_1'
        )(x_tech)
        # ✅ REMOVED: Layer normalization after LSTM (can suppress signal)
        x_tech = layers.Dropout(self.dropout_rate * 0.5, name='technical_dropout_1')(x_tech)  # ✅ REDUCED dropout

        # Second LSTM layer
        x_tech = layers.LSTM(
            self.lstm_units[1],
            return_sequences=True,
            recurrent_dropout=0.05,  # ✅ REDUCED
            name='technical_lstm_2'
        )(x_tech)
        # ✅ REMOVED: Layer normalization
        x_tech = layers.Dropout(self.dropout_rate * 0.5, name='technical_dropout_2')(x_tech)  # ✅ REDUCED dropout
        
        # Temporal Attention on LSTM outputs
        x_tech_attended, tech_attention_weights = AttentionLayer(
            units=self.attention_units,
            name='technical_attention'
        )(x_tech)
        
        # Third LSTM layer (process attended features)
        x_tech = layers.LSTM(
            self.lstm_units[2],
            return_sequences=False,
            recurrent_dropout=0.0,  # ✅ REMOVED: No recurrent dropout in final LSTM
            name='technical_lstm_3'
        )(x_tech_attended)

        # Dense layer for technical branch - ✅ REDUCED regularization
        x_tech = layers.Dense(
            self.dense_units[0],
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.0001),  # ✅ REDUCED from 0.001 to 0.0001
            name='technical_dense'
        )(x_tech)
        x_tech = layers.Dropout(self.dropout_rate * 0.5, name='technical_dropout_3')(x_tech)  # ✅ REDUCED dropout
        
        # =====================================================================
        # SENTIMENT FEATURES BRANCH (Dense + Attention)
        # =====================================================================

        # ✅ IMPROVED: Removed input layer normalization
        x_sent = input_sentiment

        # First dense layer for temporal processing - ✅ REDUCED regularization
        x_sent = layers.TimeDistributed(
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            name='sentiment_temporal_dense_1'
        )(x_sent)
        x_sent = layers.Dropout(self.dropout_rate * 0.5, name='sentiment_dropout_1')(x_sent)  # ✅ REDUCED

        # Second temporal dense layer - ✅ REDUCED regularization
        x_sent = layers.TimeDistributed(
            layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
            name='sentiment_temporal_dense_2'
        )(x_sent)
        
        # Temporal Attention on sentiment features
        x_sent_attended, sent_attention_weights = AttentionLayer(
            units=self.attention_units,
            name='sentiment_attention'
        )(x_sent)
        
        # Flatten temporal dimension
        x_sent = layers.GlobalAveragePooling1D(name='sentiment_pool')(x_sent_attended)

        # Dense layer for sentiment branch - ✅ REDUCED regularization
        x_sent = layers.Dense(
            self.dense_units[0],
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.0001),  # ✅ REDUCED
            name='sentiment_dense'
        )(x_sent)
        x_sent = layers.Dropout(self.dropout_rate * 0.5, name='sentiment_dropout_2')(x_sent)  # ✅ REDUCED
        
        # =====================================================================
        # FEATURE FUSION WITH ATTENTION
        # =====================================================================
        
        # Concatenate both branches
        x_fused = layers.Concatenate(name='concat_features')([x_tech, x_sent])
        
        # Reshape for attention (add sequence dimension)
        x_fused_reshaped = layers.Reshape((2, self.dense_units[0]), name='reshape_for_fusion')(x_fused)
        
        # Cross-branch attention
        x_fused_attended, fusion_attention_weights = AttentionLayer(
            units=self.attention_units,
            name='fusion_attention'
        )(x_fused_reshaped)
        
        # Flatten after attention
        x_fused = layers.Flatten(name='fusion_flatten')(x_fused_attended)

        # Final dense layers - ✅ REDUCED regularization
        x_final = layers.Dense(
            self.dense_units[1],
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.0001),  # ✅ REDUCED
            name='final_dense_1'
        )(x_fused)
        x_final = layers.Dropout(self.dropout_rate * 0.5, name='final_dropout')(x_final)  # ✅ REDUCED
        
        # =====================================================================
        # OUTPUT LAYER (Price Prediction)
        # =====================================================================
        
        output = layers.Dense(1, activation='linear', name='price_output')(x_final)
        
        # =====================================================================
        # CREATE MODEL
        # =====================================================================
        
        self.model = models.Model(
            inputs=[input_technical, input_sentiment],
            outputs=output,
            name='Hierarchical_Attention_Stock_Predictor'
        )
        
        # ✅ IMPROVED: Changed loss function to MSE for better price prediction
        # Reduced gradient clipping to allow larger updates
        self.model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=5.0  # ✅ INCREASED: Allow larger gradients (was 1.0, too restrictive)
            ),
            loss='mse',  # ✅ CRITICAL: MSE is better for regression tasks, Huber was too conservative
            metrics=['mae', 'mape']
        )
        
        # Create attention visualization model
        self.attention_model = models.Model(
            inputs=[input_technical, input_sentiment],
            outputs=[
                output,
                tech_attention_weights,
                sent_attention_weights,
                fusion_attention_weights
            ],
            name='Attention_Visualization_Model'
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model.summary()
    
    def get_attention_weights(self, X_technical, X_sentiment):
        """Get attention weights for visualization"""
        if self.attention_model is None:
            raise ValueError("Attention model not built yet. Call build_model() first.")
        
        return self.attention_model.predict([X_technical, X_sentiment], verbose=0)
    
    def save_model(self, filepath):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not built yet.")
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model"""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"✓ Model loaded from: {filepath}")
        return self.model


def create_model(n_timesteps=60, n_technical_features=20, n_sentiment_features=5, **kwargs):
    """Factory function to create and build the model"""
    predictor = HierarchicalAttentionStockPredictor(
        n_timesteps=n_timesteps,
        n_technical_features=n_technical_features,
        n_sentiment_features=n_sentiment_features,
        **kwargs
    )
    
    model = predictor.build_model()
    return model, predictor


if __name__ == "__main__":
    print("=" * 80)
    print("HIERARCHICAL ATTENTION MODEL - ARCHITECTURE TEST".center(80))
    print("=" * 80)
    
    # Create model
    print("\nBuilding model...")
    model, predictor = create_model(
        n_timesteps=60,
        n_technical_features=20,
        n_sentiment_features=5,
        lstm_units=[128, 64, 32],
        dense_units=[32, 16],
        attention_units=64,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE".center(80))
    print("=" * 80)
    predictor.get_model_summary()
    
    # Test with dummy data
    print("\n" + "=" * 80)
    print("TESTING WITH DUMMY DATA".center(80))
    print("=" * 80)
    
    # Create dummy inputs
    X_tech_dummy = np.random.randn(10, 60, 20)
    X_sent_dummy = np.random.randn(10, 60, 5)  # ✅ FIXED: 3D shape
    y_dummy = np.random.randn(10, 1)
    
    print(f"\nInput shapes:")
    print(f"  Technical: {X_tech_dummy.shape}")
    print(f"  Sentiment: {X_sent_dummy.shape}")
    print(f"  Target: {y_dummy.shape}")
    
    # Make prediction
    print(f"\nMaking test prediction...")
    predictions = model.predict([X_tech_dummy, X_sent_dummy], verbose=0)
    print(f"  Output shape: {predictions.shape}")
    print(f"  ✓ Model working correctly!")
    
    print("\n" + "=" * 80)
    print("✓ MODEL ARCHITECTURE TEST COMPLETE!".center(80))
    print("=" * 80)
