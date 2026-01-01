"""
Training pipeline for chess stylometry model using GameAggregateViT.

This script trains the GameAggregateViT model on prepared training data
to classify chess players based on their move patterns.
"""

import os
import argparse
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT
from stylometry.ViTOneHot.pgn_to_training_data import PlayerIndexMapper


def create_dataset_from_files(
    data_prefix: str,
    num_players: int,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    indices: Optional[np.ndarray] = None
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset using memory-mapped arrays to avoid loading all data.
    
    Args:
        data_prefix: Prefix for data files
        num_players: Number of players for one-hot encoding
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        indices: Optional array of indices to use (for train/val split)
        
    Returns:
        tf.data.Dataset yielding ((sequences, masks), labels) tuples
    """
    # Load as memory-mapped arrays (doesn't load into RAM)
    sequences_mmap = np.load(f"{data_prefix}_sequences.npy", mmap_mode='r')
    labels_mmap = np.load(f"{data_prefix}_labels.npy", mmap_mode='r')
    masks_mmap = np.load(f"{data_prefix}_masks.npy", mmap_mode='r')
    
    # Use only specified indices if provided
    if indices is not None:
        num_samples = len(indices)
    else:
        num_samples = len(sequences_mmap)
        indices = np.arange(num_samples)
    
    # Create generator that reads from disk on-demand
    def data_generator():
        idx_order = indices.copy()
        if shuffle:
            np.random.shuffle(idx_order)
        
        for idx in idx_order:
            seq = np.array(sequences_mmap[idx])  # Load single sample
            label = labels_mmap[idx]
            mask = np.array(masks_mmap[idx])
            
            # Convert label to one-hot
            label_onehot = np.eye(num_players, dtype=np.float32)[label]
            
            yield ({'input': seq, 'mask': mask}, label_onehot)
    
    # Create dataset from generator
    output_signature = (
        {
            'input': tf.TensorSpec(shape=(None, 112, 8, 8), dtype=tf.float32),
            'mask': tf.TensorSpec(shape=(None,), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(num_players,), dtype=tf.float32)
    )
    
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_dataset(
    sequences: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from numpy arrays (legacy method).
    
    Args:
        sequences: Shape (num_samples, max_moves, 112, 8, 8)
        labels: Shape (num_samples, num_players) - one-hot encoded
        masks: Shape (num_samples, max_moves)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        
    Returns:
        tf.data.Dataset yielding ((sequences, masks), labels) tuples
    """
    dataset = tf.data.Dataset.from_tensor_slices((
        {'input': sequences, 'mask': masks},
        labels
    ))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def split_train_val(
    sequences: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Args:
        sequences: All sequences
        labels: All labels
        masks: All masks
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_sequences, train_labels, train_masks, 
                 val_sequences, val_labels, val_masks)
    """
    np.random.seed(seed)
    num_samples = len(sequences)
    indices = np.random.permutation(num_samples)
    
    val_size = int(num_samples * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    return (
        sequences[train_indices],
        labels[train_indices],
        masks[train_indices],
        sequences[val_indices],
        labels[val_indices],
        masks[val_indices]
    )


class MaskedGameAggregateViT(tf.keras.Model):
    """Wrapper around GameAggregateViT that handles mask input."""
    
    def __init__(self, base_model: GameAggregateViT, num_players: int, **kwargs):
        super(MaskedGameAggregateViT, self).__init__(**kwargs)
        self.base_model = base_model
        self.classifier = tf.keras.layers.Dense(num_players, activation='softmax')
    
    def call(self, inputs, training=None):
        if isinstance(inputs, dict):
            sequences = inputs['input']
            mask = inputs.get('mask', None)
        else:
            sequences = inputs
            mask = None
        
        # Get aggregated features from base model
        features = self.base_model(sequences, training=training, mask=mask)
        
        # Classify
        return self.classifier(features, training=training)


def train_model(
    data_prefix: str,
    player_map_path: str,
    output_dir: str = "models",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    val_split: float = 0.2,
    checkpoint_freq: int = 5,
    filters: int = 12,
    num_layers: int = 4,
    num_heads: int = 4,
    hidden_dim: int = 768,  # 128 * 8 * 8
    mlp_dim: int = 768,
    max_moves: int = 100
):
    """
    Train the stylometry model.
    
    Args:
        data_prefix: Prefix for data files (without _sequences.npy suffix)
        player_map_path: Path to player mapping file
        output_dir: Directory to save model checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        val_split: Validation split fraction
        checkpoint_freq: Save checkpoint every N epochs
        filters: Number of filters in residual body
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension (must equal filters * 8 * 8)
        mlp_dim: MLP dimension in transformer
        max_moves: Maximum moves per sequence
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load player mapping
    player_mapper = PlayerIndexMapper()
    player_mapper.load(player_map_path)
    num_players = player_mapper.num_players()
    
    print(f"Preparing data from {data_prefix} (using memory-mapped files)...")
    
    # Load only metadata to determine split (much lighter than loading all data)
    labels_mmap = np.load(f"{data_prefix}_labels.npy", mmap_mode='r')
    num_samples = len(labels_mmap)
    
    print(f"Total samples: {num_samples}")
    print(f"Number of players: {num_players}")
    
    # Create train/val split indices
    print(f"\nSplitting data (val_split={val_split})...")
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    # Create datasets using memory-mapped files (avoids loading all data)
    train_dataset = create_dataset_from_files(
        data_prefix, num_players, batch_size, shuffle=True, indices=train_indices
    )
    val_dataset = create_dataset_from_files(
        data_prefix, num_players, batch_size, shuffle=False, indices=val_indices
    )
    
    # Create model
    print(f"\nCreating model...")
    print(f"  filters={filters}, hidden_dim={hidden_dim}")
    print(f"  num_layers={num_layers}, num_heads={num_heads}")
    print(f"  mlp_dim={mlp_dim}, max_moves={max_moves}")
    
    base_model = GameAggregateViT(
        move_feature_dim=112 * 8 * 8,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        max_moves=max_moves
    )
    
    model = MaskedGameAggregateViT(base_model, num_players)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'checkpoint_epoch_{epoch:02d}.keras'),
            save_freq='epoch',
            period=checkpoint_freq,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = os.path.join(output_dir, 'final_model.keras')
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"  Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Top-5 Accuracy: {history.history['top5_accuracy'][-1]:.4f}")
    
    print("\nFinal Validation Metrics:")
    print(f"  Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"  Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Top-5 Accuracy: {history.history['val_top5_accuracy'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train chess stylometry model"
    )
    parser.add_argument("data_prefix", 
                       help="Prefix for data files (e.g., 'output_data' for output_data_sequences.npy)")
    parser.add_argument("--player-map", type=str, default="stylometry/player_mapping.txt",
                       help="Player mapping file (default: stylometry/player_mapping.txt)")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Output directory for model checkpoints (default: models)")
    parser.add_argument("--epochs", type=int, default=500,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split fraction (default: 0.1)")
    parser.add_argument("--checkpoint-freq", type=int, default=5,
                       help="Save checkpoint every N epochs (default: 5)")
    parser.add_argument("--filters", type=int, default=1,
                       help="Number of filters in residual body (default: 12)")
    parser.add_argument("--num-layers", type=int, default=6,
                       help="Number of transformer layers (default: 12)")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of attention heads (default: 4)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                       help="Hidden dimension, must equal filters*64 (default: 8192)")
    parser.add_argument("--mlp-dim", type=int, default=128,
                       help="MLP dimension in transformer (default: 3072)")
    parser.add_argument("--max-moves", type=int, default=100,
                       help="Maximum moves per sequence (default: 100)")
    
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
    else:
        print("No GPU found, using CPU")
    
    # Train model
    train_model(
        data_prefix=args.data_prefix,
        player_map_path=args.player_map,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        checkpoint_freq=args.checkpoint_freq,
        filters=args.filters,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        mlp_dim=args.mlp_dim,
        max_moves=args.max_moves
    )
