import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import argparse
import random
import tensorflow as tf
import numpy as np
from typing import Tuple, List

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT


MAX_MOVES = 100
SEQ_PLANES = 21

def chessboard_struct_to_lc0_planes(structs, short=False):
  num_positions = 1 if short else 8
  planes = np.zeros((13*num_positions+8, 8, 8), dtype=np.int8)
  clocks = np.zeros((2,), dtype=np.int32)
  stm = (int(structs[num_positions * 4]) >> 24) & 0x1
  
  #ensure structs are uint64
  structs = np.asarray(structs, dtype=np.uint64)

  pcs_1_idx = 0
  pcs_2_idx = 0
  for pos_idx in range(num_positions):
    occ = structs[pos_idx * 4]
    pcs_1 = structs[pos_idx * 4 + 1]
    pcs_2 = structs[pos_idx * 4 + 2]
    
    for sq in range(64):
      chess_rank = sq // 8
      chess_file = sq % 8
      
      if int(occ) & (1 << sq):
        if pcs_1_idx < 16:
          val = (int(pcs_1) >> (4 * pcs_1_idx)) & 0xF
          pcs_1_idx += 1
        else:
          val = (int(pcs_2) >> (4 * pcs_2_idx)) & 0xF
          pcs_2_idx += 1
        
        piece_type = (val & 0x7) + 1
        piece_color = 0 if (val & 0x8) == 0 else 1

        plane_idx = piece_type - 1 + (0 if piece_color == 0 else 6)
        planes[13*pos_idx + plane_idx, chess_rank, chess_file] = 1.0
    rep_count = (int(structs[num_positions * 4]) >> (2 * pos_idx)) & 0x3
    if rep_count >= 1:
      planes[13*pos_idx + 12, :, :] = 1.0
    if pos_idx == 0:
      clocks[0] = (int(structs[pos_idx * 4 + 3]) >> 32)
      clocks[1] = (int(structs[pos_idx * 4 + 3]) & 0xFFFFFFFF)
      if stm == 1:
        clocks[0], clocks[1] = clocks[1], clocks[0]
      
  us_qs = (int(structs[num_positions * 4]) >> 24) & 0x2
  us_ks = (int(structs[num_positions * 4]) >> 24) & 0x4
  them_qs = (int(structs[num_positions * 4]) >> 24) & 0x8
  them_ks = (int(structs[num_positions * 4]) >> 24) & 0x10
  halfmove_clock = (int(structs[num_positions * 4]) >> 16) & 0xFF

  if us_qs:
    planes[13*num_positions + 0, :, :] = 1.0
  if us_ks:
    planes[13*num_positions + 1, :, :] = 1.0
  if them_qs:
    planes[13*num_positions + 2, :, :] = 1.0
  if them_ks:
    planes[13*num_positions + 3, :, :] = 1.0
  planes[13*num_positions + 4, :, :] = 1.0 if stm == 1 else 0.0
  planes[13*num_positions + 5, :, :] = halfmove_clock

  planes[13*num_positions + 6, :, :] = 0.0
  planes[13*num_positions + 7, :, :] = 1.0

  return planes, clocks

def parse_seq_example(serialized_example):
  """Parse a sequence record produced by pgn_to_training_data.py."""
  feature_description = {
    'stm_player_seq': tf.io.FixedLenFeature([], tf.string),
    'opp_player_seq': tf.io.FixedLenFeature([], tf.string),
    'wdl': tf.io.FixedLenFeature([3], tf.float32),
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  # seq is stored as (MAX_MOVES, 5) uint64 flattened to bytes
  # decode_raw doesn't support uint64, so decode as int64 and reinterpret later
  white_seq = tf.io.decode_raw(example['stm_player_seq'], tf.int64)
  white_seq = tf.reshape(white_seq, [MAX_MOVES, 5])
  black_seq = tf.io.decode_raw(example['opp_player_seq'], tf.int64)
  black_seq = tf.reshape(black_seq, [MAX_MOVES, 5])
  return white_seq, black_seq, example['wdl']


def create_seq_dataset(
  shard_paths: List[str],
  batch_size: int = 32,
  shuffle: bool = True,
  repeat: bool = True,
  skip_rate: float = 0.0
) -> tf.data.Dataset:
  """Dataset that yields (inputs_dict, elo) from seq_shards."""
  def generator():
    for shard_path in shard_paths:
      try:
        raw_ds = tf.data.TFRecordDataset(shard_path)
        for raw_record in raw_ds:
          try:
            white_seq, black_seq, wdl_tensor = parse_seq_example(raw_record)
            white_seq_np = white_seq.numpy().view(np.uint64)   # (MAX_MOVES, 5)
            black_seq_np = black_seq.numpy().view(np.uint64)   # (MAX_MOVES, 5)
            wdl_val = wdl_tensor.numpy()

            white_planes = np.zeros((MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
            black_planes = np.zeros((MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
            white_mask = np.zeros((MAX_MOVES,), dtype=np.float32)
            black_mask = np.zeros((MAX_MOVES,), dtype=np.float32)

            for m_idx in range(MAX_MOVES):
              struct = white_seq_np[m_idx]
              if np.any(struct > 0):
                p, _ = chessboard_struct_to_lc0_planes(struct, short=True)
                white_planes[m_idx] = p.astype(np.float32)
                white_mask[m_idx] = 1.0
            for m_idx in range(MAX_MOVES):
              struct = black_seq_np[m_idx]
              if np.any(struct > 0):
                p, _ = chessboard_struct_to_lc0_planes(struct, short=True)
                black_planes[m_idx] = p.astype(np.float32)
                black_mask[m_idx] = 1.0

            yield {'seq0': white_planes, 'seq1': black_planes, 'mask0': white_mask, 'mask1': black_mask}, wdl_val.astype(np.float32)
          except Exception as e:
            print(f"Skipping corrupted record in {shard_path}: {e}")
            continue
      except Exception as e:
        print(f"Skipping corrupted shard {shard_path}: {e}")
        continue

  output_signature = (
    {
      'seq0': tf.TensorSpec(shape=(MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.float32),  # type: ignore
      'seq1': tf.TensorSpec(shape=(MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.float32),  # type: ignore
      'mask0': tf.TensorSpec(shape=(MAX_MOVES,), dtype=tf.float32),                  # type: ignore
      'mask1': tf.TensorSpec(shape=(MAX_MOVES,), dtype=tf.float32),                  # type: ignore
    },
    tf.TensorSpec(shape=(3,), dtype=tf.float32)  # type: ignore
  )

  dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
  if repeat:
    dataset = dataset.repeat()
  if shuffle:
    dataset = dataset.shuffle(buffer_size=2000)
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
class EloPredictor(tf.keras.Model):
  """GameAggregateViT encoder + regression head that predicts player elo."""

  def __init__(self, vit: GameAggregateViT, hidden_dim: int,
               ffn_hidden_dim: int = 256, ffn_layers: int = 2, **kwargs):
    super(EloPredictor, self).__init__(**kwargs)
    self.vit = vit
    self.hidden_dim = hidden_dim
    self.ffn_hidden_dim = ffn_hidden_dim
    self.ffn_layers = ffn_layers
    self.regression_head = tf.keras.Sequential(
      [tf.keras.layers.Dense(ffn_hidden_dim, activation='relu') for _ in range(ffn_layers)]
      + [tf.keras.layers.Dense(1)]
    )

  def get_config(self):
    config = super().get_config()
    config['vit'] = tf.keras.utils.serialize_keras_object(self.vit)
    config['hidden_dim'] = self.hidden_dim
    config['ffn_hidden_dim'] = self.ffn_hidden_dim
    config['ffn_layers'] = self.ffn_layers
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    vit_config = config.pop('vit')
    vit = tf.keras.utils.deserialize_keras_object(vit_config)
    return cls(vit=vit, **config)

  def call(self, inputs, training=None, mask=None):
    if isinstance(inputs, dict):
      seq = inputs['seq']           # (batch, MAX_MOVES, SEQ_PLANES, 8, 8)
      m = inputs.get('mask', None)  # (batch, MAX_MOVES)
    else:
      seq = inputs
      m = None

    embedding = self.vit(seq, training=training, mask=m)  # (batch, hidden_dim)
    elo = self.regression_head(embedding, training=training)  # (batch, 1)
    return tf.squeeze(elo, axis=-1)  # (batch,)

class GameOutcomePredictor(tf.keras.Model):
  def __init__(self, elo_predictor: EloPredictor, **kwargs):
    super(GameOutcomePredictor, self).__init__(**kwargs)
    self.elo_predictor = elo_predictor
  
  def get_config(self):
    config = super().get_config()
    config['elo_predictor'] = tf.keras.utils.serialize_keras_object(self.elo_predictor)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    elo_predictor_config = config.pop('elo_predictor')
    elo_predictor = tf.keras.utils.deserialize_keras_object(elo_predictor_config)
    return cls(elo_predictor=elo_predictor, **config)
  
  def call(self, inputs, training=None, mask=None):
    seq0 = inputs['seq0']
    seq1 = inputs['seq1']
    mask0 = inputs.get('mask0', None)
    mask1 = inputs.get('mask1', None)

    elo0 = self.elo_predictor({'seq': seq0, 'mask': mask0}, training=training)
    elo1 = self.elo_predictor({'seq': seq1, 'mask': mask1}, training=training)

    elo = elo0 - elo1
    # Convert elo to win/draw/loss probabilities using logistic function
    p_win = 1 / (1 + tf.exp(-elo / 400))
    p_loss = 1 - p_win
    p_draw = tf.zeros_like(p_win)  # Placeholder for draw probability
    return tf.stack([p_win, p_draw, p_loss], axis=-1)  # (batch, 3)

def train_model(
  data_dir: str,
  output_dir: str,
  start_checkpoint: str,
  epochs: int,
  batch_size: int,
  learning_rate: float,
  val_split: float,
  num_layers: int,
  num_heads: int,
  mlp_dim: int,
  hidden_dim: int,
  ffn_layers: int,
  ffn_hidden_dim: int
):
  os.makedirs(output_dir, exist_ok=True)

  print(f"Loading data from {data_dir}...")

  seq_shard_paths = get_shard_paths(data_dir, "seq_shards")

  train_shards, val_shards = split_shards(seq_shard_paths, val_split)
  print(f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")

  train_dataset = create_seq_dataset(train_shards, batch_size, shuffle=True, repeat=True)
  val_dataset = create_seq_dataset(val_shards, batch_size, shuffle=False, repeat=False, skip_rate=0.95)

  print("Creating model...")

  stylo_model = GameAggregateViT(
    move_feature_dim=SEQ_PLANES * 8 * 8,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    mlp_dim=mlp_dim,
    max_moves=MAX_MOVES
  )

  if start_checkpoint != "":
    model = tf.keras.models.load_model(start_checkpoint)
  else:
    model = GameOutcomePredictor(
      elo_predictor=EloPredictor(
        vit=stylo_model,
        hidden_dim=hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_layers=ffn_layers
      )
    )
  assert isinstance(model, GameOutcomePredictor)

  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
  )

  model.build(input_shape={  # type: ignore
    'seq0': (None, MAX_MOVES, SEQ_PLANES, 8, 8),
    'seq1': (None, MAX_MOVES, SEQ_PLANES, 8, 8),
    'mask0': (None, MAX_MOVES),
    'mask1': (None, MAX_MOVES),
  })

  class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
      super().__init__()
      self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
      if logs is None or "learning_rate" in logs:
        return
      logs["learning_rate"] = self.model.optimizer.lr

  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(output_dir, 'checkpoint_epoch_{epoch:02d}.keras'),
      save_freq='epoch',
      verbose=1
    ),
    LearningRateLogger(),
    tf.keras.callbacks.TensorBoard(
      log_dir=os.path.join(output_dir, 'logs'),
      histogram_freq=0
    ),
  ]

  print(f"Starting training for {epochs} epochs...")
  history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    steps_per_epoch=5000,
    callbacks=callbacks,
    verbose=1  # type: ignore
  )

  final_path = os.path.join(output_dir, 'final_model.keras')
  model.save(final_path)
  print(f"Final model saved to {final_path}")

  assert history is not None
  print("Final Training Metrics:")
  print(f"  Loss (MSE): {history.history['loss'][-1]:.2f}")
  print(f"  MAE:        {history.history['mae'][-1]:.2f}")
  print("Final Validation Metrics:")
  print(f"  Loss (MSE): {history.history['val_loss'][-1]:.2f}")
  print(f"  MAE:        {history.history['val_mae'][-1]:.2f}")

  return model, history

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train elo prediction model on player sequences")
  parser.add_argument("data_dir", help="Directory containing seq_shards/ subfolder")
  parser.add_argument("--output-dir", type=str, default="stylometry/ViTOneHot/models")
  parser.add_argument("--start-checkpoint", type=str, default="")
  parser.add_argument("--epochs", type=int, default=500)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--val-split", type=float, default=0.003)
  parser.add_argument("--hidden-dim", type=int, default=256)
  parser.add_argument("--num-layers", type=int, default=6)
  parser.add_argument("--num-heads", type=int, default=4)
  parser.add_argument("--mlp-dim", type=int, default=256)
  parser.add_argument("--ffn-layers", type=int, default=2)
  parser.add_argument("--ffn-hidden-dim", type=int, default=256)

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
    start_checkpoint=args.start_checkpoint,
    epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    val_split=args.val_split,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    mlp_dim=args.mlp_dim,
    hidden_dim=args.hidden_dim,
    ffn_layers=args.ffn_layers,
    ffn_hidden_dim=args.ffn_hidden_dim
  )
# python3 -m stylometry.ViTOneHot.train_stylometry <data_dir>
