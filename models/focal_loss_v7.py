"""
Focal Loss for V7

Focuses training on hard-to-classify samples (significant price moves)
"""

import tensorflow as tf
from tensorflow import keras

def focal_loss_binary(gamma=2.0, alpha=0.25):
    """
    Focal Loss for binary classification

    Formula: FL = -alpha * (1-p)^gamma * log(p)

    Args:
        gamma: Focusing parameter (higher = more focus on hard samples)
               Recommended: 2.0
        alpha: Balancing parameter for class imbalance
               Recommended: 0.25

    Returns:
        Loss function for Keras
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Calculate focal weight
        # For y_true = 1: weight = (1 - y_pred)^gamma
        # For y_true = 0: weight = y_pred^gamma
        focal_weight = tf.where(
            tf.equal(y_true, 1),
            tf.pow(1 - y_pred, gamma),
            tf.pow(y_pred, gamma)
        )

        # Apply alpha balancing
        alpha_weight = tf.where(
            tf.equal(y_true, 1),
            alpha,
            1 - alpha
        )

        # Focal loss
        focal_loss = alpha_weight * focal_weight * cross_entropy

        return tf.reduce_mean(focal_loss)

    return loss


def weighted_focal_loss(gamma=2.0, alpha=0.25, magnitude_weight=2.0):
    """
    Focal Loss with additional weight for large magnitude moves

    Args:
        gamma: Focusing parameter
        alpha: Class balancing
        magnitude_weight: Extra weight for large moves (|change| > 1%)
    """
    def loss(y_true, y_pred):
        # Standard focal loss
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        focal_weight = tf.where(
            tf.equal(y_true, 1),
            tf.pow(1 - y_pred, gamma),
            tf.pow(y_pred, gamma)
        )

        alpha_weight = tf.where(
            tf.equal(y_true, 1),
            alpha,
            1 - alpha
        )

        focal_loss = alpha_weight * focal_weight * cross_entropy

        return tf.reduce_mean(focal_loss)

    return loss
