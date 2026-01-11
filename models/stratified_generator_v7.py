"""
Stratified Data Generator for V7

Creates balanced batches by magnitude to improve learning on significant moves
"""

import numpy as np
import tensorflow as tf

class StratifiedBatchGenerator(tf.keras.utils.Sequence):
    """
    Generates batches with balanced representation of small/medium/large moves

    Instead of natural distribution (80% small, 15% medium, 5% large),
    creates balanced batches (40% small, 40% medium, 20% large)
    """

    def __init__(self, X, y_direction, y_magnitude, batch_size=32,
                 small_threshold=0.5, large_threshold=1.5, shuffle=True):
        """
        Args:
            X: Input features (n_samples, timesteps, features)
            y_direction: Direction labels (n_samples,)
            y_magnitude: Magnitude values (n_samples,) - percentage changes
            batch_size: Batch size
            small_threshold: Threshold for small moves (default 0.5%)
            large_threshold: Threshold for large moves (default 1.5%)
            shuffle: Shuffle within each magnitude bin
        """
        self.X = X
        self.y_direction = y_direction
        self.y_magnitude = y_magnitude
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Categorize samples by magnitude
        abs_magnitude = np.abs(y_magnitude)

        self.small_indices = np.where(abs_magnitude < small_threshold)[0]
        self.medium_indices = np.where((abs_magnitude >= small_threshold) &
                                       (abs_magnitude < large_threshold))[0]
        self.large_indices = np.where(abs_magnitude >= large_threshold)[0]

        # Calculate samples per batch for each category
        self.n_small_per_batch = int(batch_size * 0.4)
        self.n_medium_per_batch = int(batch_size * 0.4)
        self.n_large_per_batch = batch_size - self.n_small_per_batch - self.n_medium_per_batch

        # Calculate number of batches
        # Limited by the smallest oversampled category
        self.n_batches = min(
            len(self.small_indices) // self.n_small_per_batch,
            len(self.medium_indices) // self.n_medium_per_batch,
            len(self.large_indices) // self.n_large_per_batch
        )

        print(f"\n   Stratified Batch Generator:")
        print(f"   Total samples: {len(X)}")
        print(f"   Small moves (<{small_threshold}%): {len(self.small_indices)} ({len(self.small_indices)/len(X)*100:.1f}%)")
        print(f"   Medium moves ({small_threshold}-{large_threshold}%): {len(self.medium_indices)} ({len(self.medium_indices)/len(X)*100:.1f}%)")
        print(f"   Large moves (>{large_threshold}%): {len(self.large_indices)} ({len(self.large_indices)/len(X)*100:.1f}%)")
        print(f"   Batches per epoch: {self.n_batches}")
        print(f"   Samples per batch: {self.n_small_per_batch} small + {self.n_medium_per_batch} medium + {self.n_large_per_batch} large")

        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return self.n_batches

    def __getitem__(self, index):
        """Generate one batch"""
        # Get indices for this batch
        start_small = index * self.n_small_per_batch
        start_medium = index * self.n_medium_per_batch
        start_large = index * self.n_large_per_batch

        batch_small_indices = self.shuffled_small[start_small:start_small + self.n_small_per_batch]
        batch_medium_indices = self.shuffled_medium[start_medium:start_medium + self.n_medium_per_batch]
        batch_large_indices = self.shuffled_large[start_large:start_large + self.n_large_per_batch]

        # Combine indices
        batch_indices = np.concatenate([batch_small_indices, batch_medium_indices, batch_large_indices])

        # Shuffle within batch
        np.random.shuffle(batch_indices)

        # Get batch data
        X_batch = self.X[batch_indices]
        y_direction_batch = self.y_direction[batch_indices]
        y_magnitude_batch = self.y_magnitude[batch_indices]

        return X_batch, {'direction': y_direction_batch, 'magnitude': y_magnitude_batch}

    def on_epoch_end(self):
        """Shuffle indices at end of epoch"""
        if self.shuffle:
            self.shuffled_small = np.random.permutation(self.small_indices)
            self.shuffled_medium = np.random.permutation(self.medium_indices)
            self.shuffled_large = np.random.permutation(self.large_indices)
        else:
            self.shuffled_small = self.small_indices.copy()
            self.shuffled_medium = self.medium_indices.copy()
            self.shuffled_large = self.large_indices.copy()


def create_stratified_generators(X_train, y_train_direction, y_train_magnitude,
                                 X_val, y_val_direction, y_val_magnitude,
                                 batch_size=32):
    """
    Create stratified generators for training and validation

    Returns:
        train_gen, val_gen
    """
    train_gen = StratifiedBatchGenerator(
        X_train, y_train_direction, y_train_magnitude,
        batch_size=batch_size, shuffle=True
    )

    # Validation uses regular batching (no stratification needed)
    # But we'll create a simple generator for consistency
    class SimpleGenerator(tf.keras.utils.Sequence):
        def __init__(self, X, y_direction, y_magnitude, batch_size):
            self.X = X
            self.y_direction = y_direction
            self.y_magnitude = y_magnitude
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, len(self.X))

            return (self.X[start:end],
                   {'direction': self.y_direction[start:end],
                    'magnitude': self.y_magnitude[start:end]})

    val_gen = SimpleGenerator(X_val, y_val_direction, y_val_magnitude, batch_size)

    return train_gen, val_gen
