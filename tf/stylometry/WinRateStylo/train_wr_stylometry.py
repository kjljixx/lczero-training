import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
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
  if num_moves_in_seq == 0:
    return padded, mask
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
  
  stm_seq = tf.io.decode_raw(example['stm_player_seq'], tf.int64)
  opp_seq = tf.io.decode_raw(example['opp_player_seq'], tf.int64)
  full_board = tf.io.decode_raw(example['full_board_planes'], tf.int64)
  
  # Shapes: (5, 100, 5), (5, 100, 5), (25,)
  stm_seq = tf.reshape(stm_seq, [5, 100, 5])
  opp_seq = tf.reshape(opp_seq, [5, 100, 5])
  full_board = tf.reshape(full_board, [33])
  
  return stm_seq, opp_seq, full_board, example['wdl']

def create_position_dataset(
  pos_shard_paths: List[str],
  batch_size: int = 32,
  shuffle: bool = True,
  skip_rate: float = 0.0,
  repeat: bool = True
) -> tf.data.Dataset:
  def generator():
    for shard_path in pos_shard_paths:
      try:
        dataset = tf.data.TFRecordDataset(shard_path)
        for raw_record in dataset:
          try:
            if random.random() < skip_rate:
              continue
            
            stm_seq_tensor, opp_seq_tensor, full_board_tensor, wdl = parse_position_example(raw_record)
            
            stm_seq_np = stm_seq_tensor.numpy().astype(np.uint64)
            opp_seq_np = opp_seq_tensor.numpy().astype(np.uint64)
            full_board_np = full_board_tensor.numpy().astype(np.uint64)
            
            pos_planes, pos_clocks = chessboard_struct_to_lc0_planes(full_board_np, short=False)
            
            def process_seq(seq_np):
              planes_all_games = np.zeros((5, 100, SEQ_PLANES, 8, 8), dtype=np.int8)
              clocks_all_games = np.zeros((5, 100, 2), dtype=np.int32)
              mask_all_games = np.zeros((5, 100), dtype=np.int8)
              for g_idx in range(5):
                for m_idx in range(100):
                  struct = seq_np[g_idx, m_idx]
                  if np.any(struct > 0):
                    planes_all_games[g_idx, m_idx], clocks_all_games[g_idx, m_idx] = chessboard_struct_to_lc0_planes(struct, short=True)
                    mask_all_games[g_idx, m_idx] = 1
              return planes_all_games, mask_all_games, clocks_all_games

            stm_planes, stm_mask, stm_clocks = process_seq(stm_seq_np)
            opp_planes, opp_mask, opp_clocks = process_seq(opp_seq_np)
            
            yield (
              {
                'input1': stm_planes,
                'input1_clocks': stm_clocks,
                'input2': opp_planes,
                'input2_clocks': opp_clocks,
                'pos': pos_planes.flatten(),
                'pos_clocks': pos_clocks,
                'mask1': stm_mask,
                'mask2': opp_mask
              },
              wdl.numpy()
            )
          except Exception as e:
            print(f"Skipping corrupted record in {shard_path}: {e}")
            continue
      except Exception as e:
        print(f"Skipping corrupted shard {shard_path}: {e}")
        continue

  output_signature = (
    {
      'input1': tf.TensorSpec(shape=(5, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), # type: ignore
      'input1_clocks': tf.TensorSpec(shape=(5, MAX_MOVES, 2), dtype=tf.int32), # type: ignore
      'input2': tf.TensorSpec(shape=(5, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.int8), # type: ignore
      'input2_clocks': tf.TensorSpec(shape=(5, MAX_MOVES, 2), dtype=tf.int32), # type: ignore
      'pos': tf.TensorSpec(shape=(POS_PLANES * 8 * 8,), dtype=tf.int8), # type: ignore
      'pos_clocks': tf.TensorSpec(shape=(2,), dtype=tf.int32), # type: ignore
      'mask1': tf.TensorSpec(shape=(5, MAX_MOVES,), dtype=tf.int8), # type: ignore
      'mask2': tf.TensorSpec(shape=(5, MAX_MOVES,), dtype=tf.int8) # type: ignore
    },
    tf.TensorSpec(shape=(3,), dtype=tf.float32) # type: ignore
  )

  if repeat:
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature).repeat()
  else:
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
      pos_clocks = inputs.get('pos_clocks', None)
    else:
      seq1, seq2 = inputs
      mask1 = mask2 = None
      pos = None
      pos_clocks = None
    
    def process_player_seq(seq, mask):
      seq_unstacked = tf.unstack(seq, axis=1)
      mask_unstacked = tf.unstack(mask, axis=1)
      
      all_game_embeddings = []
      for g_seq, g_mask in zip(seq_unstacked, mask_unstacked):
        game_emb = self.vit(g_seq, training=training, mask=g_mask)
        all_game_embeddings.append(game_emb)
        
      game_embeddings = tf.stack(all_game_embeddings, axis=1)
      
      game_mask = tf.reduce_any(mask > 0, axis=-1) # (batch, num_games)
      game_mask_float = tf.expand_dims(tf.cast(game_mask, tf.float32), axis=-1)
      
      sum_embeddings = tf.reduce_sum(game_embeddings * game_mask_float, axis=1)
      count_embeddings = tf.reduce_sum(game_mask_float, axis=1)
      avg_embeddings = tf.math.divide_no_nan(sum_embeddings, count_embeddings)
      
      return avg_embeddings

    features1 = process_player_seq(seq1, mask1)
    features2 = process_player_seq(seq2, mask2)

    pos = tf.cast(pos, tf.float32)

    if pos_clocks is None:
      pos_clocks = tf.fill((tf.shape(pos)[0], 2), 600)

    combined = tf.concat([features1, features2, tf.cast(pos_clocks, tf.float32), pos], axis=-1)
    
    win_rate = self.wr_pred(combined, training=training)
    
    return win_rate

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
    val_pos_shards, batch_size, shuffle=False, repeat=False, skip_rate=0.8
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
    style_vec_size=hidden_dim+1,
    num_hidden_layers=wr_ffn_layers,
    hidden_dim=wr_ffn_hidden_dim
  )

  model = ScaffoldedViTAndWinRate(stylo_model, wr_model)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
  )

  if start_checkpoint != "":
    model = tf.keras.models.load_model(start_checkpoint)

  model.build(input_shape={ # type: ignore
    'input1': (None, 5, MAX_MOVES, SEQ_PLANES, 8, 8),
    'input2': (None, 5, MAX_MOVES, SEQ_PLANES, 8, 8),
    'pos': (None, POS_PLANES * 8 * 8),
    'mask1': (None, 5, MAX_MOVES),
    'mask2': (None, 5, MAX_MOVES)
  })

  for x, y in train_dataset.take(1):
    print("Sample batch shapes:")
    for key in x:
      print(f"  {key}: {x[key].shape}")
    print(f"  labels: {y.shape}")
    print(x["pos_clocks"])
    print(model(x, training=False))

  if start_checkpoint == "":
    print(stylo_model.summary())
    print(wr_model.summary())

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
    # tf.keras.callbacks.ReduceLROnPlateau(
    #   patience=5,
    #   factor=0.5,
    # )
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
  parser.add_argument("--start-checkpoint", type=str, default="")
  parser.add_argument("--epochs", type=int, default=500)
  parser.add_argument("--batch-size", type=int, default=7)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--val-split", type=float, default=0.01)
  parser.add_argument("--hidden-dim", type=int, default=256)
  parser.add_argument("--wr-ffn-layers", type=int, default=6)
  parser.add_argument("--wr-ffn-hidden-dim", type=int, default=256)

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
    num_layers=6,
    num_heads=4,
    mlp_dim=256,
    hidden_dim=args.hidden_dim,
    wr_ffn_layers=args.wr_ffn_layers,
    wr_ffn_hidden_dim=args.wr_ffn_hidden_dim
  )
# python3 -m stylometry.WinRateStylo.train_wr_stylometry stylometry/WinRateStylo/data/run2025-12-10 --output-dir stylometry/WinRateStylo/models/run2025-12-10
