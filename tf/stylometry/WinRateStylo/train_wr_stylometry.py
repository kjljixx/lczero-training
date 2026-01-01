import os
import glob
import argparse
import random
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, List

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT
from stylometry.WinRateStylo.win_rate_stylo import WinRateStyloModel
from stylometry.WinRateStylo.pgn_to_training_data import PlayerIndexMapper

MAX_MOVES = 100
SEQ_PLANES = 21
POS_PLANES = 112

def pad_sequence(
  sequence: List[np.ndarray],
  max_moves: int = 100,
  padding_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
  padded = np.full(
    (max_moves, SEQ_PLANES, 8, 8),
    padding_value,
    dtype=np.int8
  )
  mask = np.zeros((max_moves,), dtype=np.int8)

  num_moves_in_seq = min(len(sequence), max_moves)
  padded[:num_moves_in_seq] = sequence[:num_moves_in_seq]
  mask[:num_moves_in_seq] = 1.0

  return padded, mask

def parse_position_example(serialized_example):
  feature_description = {
    'stm_player_seq': tf.io.FixedLenFeature([], tf.string),
    'opp_player_seq': tf.io.FixedLenFeature([], tf.string),
    'full_board_planes': tf.io.FixedLenFeature([], tf.string),
    'wdl': tf.io.FixedLenFeature([3], tf.float32),
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  stm_seq = tf.io.decode_raw(example['stm_player_seq'], tf.int8)
  if tf.size(stm_seq) == 0:
    stm_seq, stm_mask = tf.zeros((MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), tf.zeros((MAX_MOVES,), dtype=tf.int8)
  else:
    stm_seq, stm_mask = pad_sequence(tf.reshape(stm_seq, [-1, SEQ_PLANES, 8, 8]).numpy().tolist())
  opp_seq = tf.io.decode_raw(example['opp_player_seq'], tf.int8)
  if tf.size(opp_seq) == 0:
    opp_seq, opp_mask = tf.zeros((MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), tf.zeros((MAX_MOVES,), dtype=tf.int8)
  else:
    opp_seq, opp_mask = pad_sequence(tf.reshape(opp_seq, [-1, SEQ_PLANES, 8, 8]).numpy().tolist())
  planes = tf.io.decode_raw(example['full_board_planes'], tf.int8)
  planes = tf.reshape(planes, [POS_PLANES * 8 * 8])
  wdl = example['wdl']
  return stm_seq, stm_mask, opp_seq, opp_mask, planes, wdl

def create_position_dataset(
  pos_shard_paths: List[str],
  batch_size: int = 32,
  shuffle: bool = True,
  skip_rate: float = 0.0,
  repeat: bool = True
) -> tf.data.Dataset:
  def generator():
    while True:
      for shard_path in pos_shard_paths:
        try:
          dataset = tf.data.TFRecordDataset(shard_path)
          for raw_record in dataset:
            try:
              stm_seq, stm_mask, opp_seq, opp_mask, planes, wdl = parse_position_example(raw_record)
              if random.random() < skip_rate:
                continue
              yield (
                {
                  'input1': stm_seq,
                  'input2': opp_seq,
                  'pos': planes.numpy(),
                  'mask1': stm_mask,
                  'mask2': opp_mask
                },
                wdl.numpy()
              )
            except tf.errors.DataLossError as e:
              print(f"Skipping corrupted record in {shard_path}: {e}")
              continue
        except tf.errors.DataLossError as e:
          print(f"Skipping corrupted shard {shard_path}: {e}")
          continue
      if not repeat:
        break

  output_signature = (
    {
      'input1': tf.TensorSpec(shape=(MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), # type: ignore
      'input2': tf.TensorSpec(shape=(MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), # type: ignore
      'pos': tf.TensorSpec(shape=(POS_PLANES * 8 * 8,), dtype=tf.int8), # type: ignore
      'mask1': tf.TensorSpec(shape=(MAX_MOVES,), dtype=tf.int8), # type: ignore
      'mask2': tf.TensorSpec(shape=(MAX_MOVES,), dtype=tf.int8) # type: ignore
    },
    tf.TensorSpec(shape=(3,), dtype=tf.float32) # type: ignore
  )

  dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
  if shuffle:
    dataset = dataset.shuffle(buffer_size=1000)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(10)
  return dataset

def get_shard_paths(data_dir: str, shard_type: str) -> List[str]:
  shard_dir = os.path.join(data_dir, shard_type)
  paths = sorted(glob.glob(os.path.join(shard_dir, "*.tfrecord")))
  print(f"Found {len(paths)} {shard_type} shards")
  return paths

def split_shards(shard_paths: List[str], val_split: float) -> Tuple[List[str], List[str]]:
  num_shards = len(shard_paths)
  val_count = max(1, int(num_shards * val_split))
  random.shuffle(shard_paths)
  val_shards = shard_paths[:val_count]
  train_shards = shard_paths[val_count:]
  return train_shards, val_shards

@tf.keras.utils.register_keras_serializable()
class ScaffoldedViTAndWinRate(tf.keras.Model):
  """Wrapper around GameAggregateViT that handles mask input."""

  def __init__(self, vit: GameAggregateViT, wr_pred: WinRateStyloModel, **kwargs):
    super(ScaffoldedViTAndWinRate, self).__init__(**kwargs)
    self.vit = vit
    self.wr_pred = wr_pred

  def get_config(self):
    config = super().get_config()
    config['vit'] = tf.keras.utils.serialize_keras_object(self.vit)
    config['wr_pred'] = tf.keras.utils.serialize_keras_object(self.wr_pred)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    vit_config = config.pop('vit')
    vit = tf.keras.utils.deserialize_keras_object(vit_config)
    wr_pred_config = config.pop('wr_pred')
    wr_pred = tf.keras.utils.deserialize_keras_object(wr_pred_config)
    return cls(vit=vit, wr_pred=wr_pred, **config)

  def call(self, inputs, training=None, mask=None):
    if isinstance(inputs, dict):
      seq1 = inputs['input1']
      seq2 = inputs['input2']
      mask1 = inputs.get('mask1', None)
      mask2 = inputs.get('mask2', None)
      pos = inputs.get('pos', None)
    else:
      seq1, seq2 = inputs
      mask1 = mask2 = None
      pos = None
    
    features1 = self.vit(seq1, training=training, mask=mask1)
    features2 = self.vit(seq2, training=training, mask=mask2)

    # assert(tf.reduce_any(tf.math.is_nan(features1)) == False)
    # assert(tf.reduce_any(tf.math.is_nan(features2)) == False)
    # assert(tf.reduce_any(tf.math.is_nan(pos)) == False)
    
    combined = tf.concat([features1, features2, tf.cast(pos, tf.float32)], axis=-1)
    
    win_rate = self.wr_pred(combined, training=training)
    
    return win_rate

def train_model(
  data_dir: str,
  output_dir: str,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  val_split: float,
  num_layers: int,
  num_heads: int,
  mlp_dim: int,
  hidden_dim: int,
  wr_ffn_layers: int,
  wr_ffn_hidden_dim: int
):
  os.makedirs(output_dir, exist_ok=True)

  print(f"Loading data from {data_dir}...")

  pos_shard_paths = get_shard_paths(data_dir, "pos_shards")

  train_pos_shards, val_pos_shards = split_shards(pos_shard_paths, val_split)

  print(f"Train pos shards: {len(train_pos_shards)}, Val pos shards: {len(val_pos_shards)}")

  train_dataset = create_position_dataset(
    train_pos_shards, batch_size, shuffle=True, repeat=True
  )
  val_dataset = create_position_dataset(
    val_pos_shards, batch_size, shuffle=False, repeat=False
  )

  print("Creating model...")

  stylo_model = GameAggregateViT(
    move_feature_dim=SEQ_PLANES * 8 * 8,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    mlp_dim=mlp_dim,
    max_moves=MAX_MOVES
  )

  wr_model = WinRateStyloModel(
    style_vec_size=hidden_dim,
    num_hidden_layers=wr_ffn_layers,
    hidden_dim=wr_ffn_hidden_dim
  )

  model = ScaffoldedViTAndWinRate(stylo_model, wr_model)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
  )

  model.build(input_shape={
    'input1': (None, MAX_MOVES, SEQ_PLANES, 8, 8),
    'input2': (None, MAX_MOVES, SEQ_PLANES, 8, 8),
    'pos': (None, POS_PLANES * 8 * 8),
    'mask1': (None, MAX_MOVES),
    'mask2': (None, MAX_MOVES)
  })

  print(stylo_model.summary())
  print(wr_model.summary())

  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(output_dir, 'checkpoint_epoch_{epoch:02d}.keras'),
      save_freq='epoch',
      verbose=1
    ),
    tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(output_dir, 'logs'),
      histogram_freq=0
    ),
    tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=3,
      min_delta=0,
      mode='auto',
      restore_best_weights=True
    )
  ]

  print(f"Starting training for {epochs} epochs...")
  history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    steps_per_epoch=5000,
    callbacks=callbacks,
    verbose=1 # type: ignore
  )

  final_path = os.path.join(output_dir, 'final_model.keras')
  model.save(final_path)
  print(f"Final model saved to {final_path}")

  assert(history is not None)
  print("Final Training Metrics:")
  print(f"  Loss: {history.history['loss'][-1]:.4f}")
  print(f"  Accuracy: {history.history['accuracy'][-1]:.4f}")
  print(f"  Top-3 Accuracy: {history.history['top3_accuracy'][-1]:.4f}")

  print("Final Validation Metrics:")
  print(f"  Loss: {history.history['val_loss'][-1]:.4f}")
  print(f"  Accuracy: {history.history['val_accuracy'][-1]:.4f}")
  print(f"  Top-3 Accuracy: {history.history['val_top3_accuracy'][-1]:.4f}")

  return model, history

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train chess stylometry model")
  parser.add_argument("data_dir", help="Directory containing seq_shards and pos_shards")
  parser.add_argument("--output-dir", type=str, default="stylometry/WinRateStylo/models")
  parser.add_argument("--epochs", type=int, default=500)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--val-split", type=float, default=0.05)
  parser.add_argument("--hidden-dim", type=int, default=128)
  parser.add_argument("--wr-ffn-layers", type=int, default=2)
  parser.add_argument("--wr-ffn-hidden-dim", type=int, default=128)

  random.seed(42)

  args = parser.parse_args()

  print(args)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Found {len(gpus)} GPU(s)")
  else:
    print("No GPU found, using CPU")

  train_model(
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    val_split=args.val_split,
    num_layers=6,
    num_heads=4,
    mlp_dim=128,
    hidden_dim=args.hidden_dim,
    wr_ffn_layers=args.wr_ffn_layers,
    wr_ffn_hidden_dim=args.wr_ffn_hidden_dim
  )
# python3 -m stylometry.WinRateStylo.train_wr_stylometry stylometry/WinRateStylo/data/run2025-12-10 --output-dir stylometry/WinRateStylo/models/run2025-12-10
