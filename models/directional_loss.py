"""
Custom loss functions for stock price prediction that penalize directional errors
"""

import tensorflow as tf
from tensorflow import keras

def directional_mse_loss(alpha=0.5):
    """
    Combined MSE + Directional penalty loss

    Args:
        alpha: Weight for directional penalty (0 = pure MSE, 1 = pure directional)

    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # Standard MSE component
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Directional penalty: penalize wrong direction predictions
        # sign(y_true) != sign(y_pred) gets penalized
        direction_penalty = tf.where(
            tf.multiply(y_true, y_pred) < 0,  # Different signs
            tf.square(y_true - y_pred) * 2.0,  # Double the penalty
            tf.square(y_true - y_pred)          # Normal penalty
        )

        # Combine
        return (1 - alpha) * mse + alpha * tf.reduce_mean(direction_penalty)

    return loss


def focal_mse_loss(gamma=2.0):
    """
    Focal MSE: Focus more on hard-to-predict samples (large moves)

    Args:
        gamma: Focusing parameter (higher = more focus on large errors)

    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # Calculate squared error
        squared_error = tf.square(y_true - y_pred)

        # Weight by magnitude: larger actual changes get more weight
        magnitude_weight = tf.pow(tf.abs(y_true) + 1.0, gamma)

        # Weighted loss
        weighted_loss = squared_error * magnitude_weight

        return tf.reduce_mean(weighted_loss)

    return loss


def asymmetric_mse_loss(overpredict_penalty=1.5):
    """
    Asymmetric MSE: Different penalties for over/under prediction

    Useful when you want to penalize missing upward moves more
    (or vice versa for short strategies)

    Args:
        overpredict_penalty: Multiplier for overprediction errors

    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        error = y_true - y_pred

        # Penalize overprediction more (predicted higher than actual)
        loss_val = tf.where(
            error < 0,  # We overpredicted
            tf.square(error) * overpredict_penalty,
            tf.square(error)
        )

        return tf.reduce_mean(loss_val)

    return loss


def combined_regression_classification_loss(threshold=0.5, alpha=0.3):
    """
    Combines regression (predict exact value) with classification (predict direction)

    This helps the model learn both magnitude AND direction

    Args:
        threshold: Percentage change threshold for up/down (default 0.5%)
        alpha: Weight for classification component (0-1)

    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        # Regression component: MSE
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Classification component: Direction
        # Convert to binary: 1 if > threshold, 0 if < -threshold
        y_true_direction = tf.cast(y_true > threshold, tf.float32) - tf.cast(y_true < -threshold, tf.float32)
        y_pred_direction = tf.cast(y_pred > threshold, tf.float32) - tf.cast(y_pred < -threshold, tf.float32)

        # Binary cross-entropy on direction
        direction_loss = tf.reduce_mean(tf.square(y_true_direction - y_pred_direction))

        # Combine
        return (1 - alpha) * mse + alpha * direction_loss

    return loss


# For use in model compilation
CUSTOM_LOSSES = {
    'directional_mse': directional_mse_loss(alpha=0.3),
    'focal_mse': focal_mse_loss(gamma=2.0),
    'asymmetric_mse': asymmetric_mse_loss(overpredict_penalty=1.5),
    'combined': combined_regression_classification_loss(threshold=0.5, alpha=0.3)
}
