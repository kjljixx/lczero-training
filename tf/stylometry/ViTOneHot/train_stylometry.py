import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import argparse
import random
import json
import hashlib
from datetime import datetime
import tensorflow as tf
import numpy as np
from typing import Tuple, List

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT


def _training_flag(training) -> bool:
  return bool(training) if training is not None else False


MAX_MOVES = 100
NUM_GAMES = 5
SEQ_PLANES = 21

PIECE_SYMBOLS = {
  1: 'p',
  2: 'n',
  3: 'b',
  4: 'r',
  5: 'q',
  6: 'k',
}


def _flip_vertical_square(square: int) -> int:
  rank = square // 8
  file = square % 8
  return (7 - rank) * 8 + file


def chessboard_struct_to_fen_like(struct: np.ndarray) -> str:
  values = np.asarray(struct, dtype=np.uint64)
  if values.shape[0] < 5 or not np.any(values > 0):
    return ""

  occ = int(values[0])
  pcs_1 = int(values[1])
  pcs_2 = int(values[2])
  metadata = int(values[4])

  stm_is_black = ((metadata >> 24) & 0x1) == 1
  us_qs = ((metadata >> 24) & 0x2) != 0
  us_ks = ((metadata >> 24) & 0x4) != 0
  them_qs = ((metadata >> 24) & 0x8) != 0
  them_ks = ((metadata >> 24) & 0x10) != 0
  halfmove_clock = (metadata >> 16) & 0xFF

  board = [''] * 64
  pcs_1_idx = 0
  pcs_2_idx = 0
  for sq in range(64):
    if occ & (1 << sq):
      if pcs_1_idx < 16:
        val = (pcs_1 >> (4 * pcs_1_idx)) & 0xF
        pcs_1_idx += 1
      else:
        val = (pcs_2 >> (4 * pcs_2_idx)) & 0xF
        pcs_2_idx += 1

      piece_type = (val & 0x7) + 1
      is_stm_piece = (val & 0x8) == 0
      symbol = PIECE_SYMBOLS.get(piece_type, '?')

      if stm_is_black:
        absolute_sq = _flip_vertical_square(sq)
        is_white_piece = not is_stm_piece
      else:
        absolute_sq = sq
        is_white_piece = is_stm_piece

      board[absolute_sq] = symbol.upper() if is_white_piece else symbol

  ranks = []
  for rank in range(7, -1, -1):
    empty = 0
    rank_str = []
    for file in range(8):
      piece = board[rank * 8 + file]
      if piece:
        if empty > 0:
          rank_str.append(str(empty))
          empty = 0
        rank_str.append(piece)
      else:
        empty += 1
    if empty > 0:
      rank_str.append(str(empty))
    ranks.append(''.join(rank_str))
  placement = '/'.join(ranks)

  stm_side = 'b' if stm_is_black else 'w'
  if stm_is_black:
    white_ks, white_qs, black_ks, black_qs = them_ks, them_qs, us_ks, us_qs
  else:
    white_ks, white_qs, black_ks, black_qs = us_ks, us_qs, them_ks, them_qs

  castling = (
    ('K' if white_ks else '')
    + ('Q' if white_qs else '')
    + ('k' if black_ks else '')
    + ('q' if black_qs else '')
  )
  if castling == '':
    castling = '-'

  return f"{placement} {stm_side} {castling} - {halfmove_clock} 1"


def _decode_name(name_value) -> str:
  if isinstance(name_value, bytes):
    return name_value.decode('utf-8', errors='replace')
  if isinstance(name_value, np.bytes_):
    return bytes(name_value).decode('utf-8', errors='replace')
  return str(name_value)

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
    'stm_player_name': tf.io.FixedLenFeature([], tf.string),
    'stm_player_elo': tf.io.FixedLenFeature([], tf.int64),
    'opp_player_name': tf.io.FixedLenFeature([], tf.string),
    'opp_player_elo': tf.io.FixedLenFeature([], tf.int64),
    'wdl': tf.io.FixedLenFeature([3], tf.float32),
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  # seq is stored as (NUM_GAMES, MAX_MOVES, 5) uint64 flattened to bytes
  # decode_raw doesn't support uint64, so decode as int64 and reinterpret later
  white_seq = tf.io.decode_raw(example['stm_player_seq'], tf.int64)
  white_seq = tf.reshape(white_seq, [NUM_GAMES, MAX_MOVES, 5])
  black_seq = tf.io.decode_raw(example['opp_player_seq'], tf.int64)
  black_seq = tf.reshape(black_seq, [NUM_GAMES, MAX_MOVES, 5])
  return (
    white_seq,
    black_seq,
    example['stm_player_name'],
    example['stm_player_elo'],
    example['opp_player_name'],
    example['opp_player_elo'],
    example['wdl'],
  )


def create_seq_dataset(
  shard_paths: List[str],
  batch_size: int = 32,
  shuffle: bool = True,
  repeat: bool = True,
  skip_rate: float = 0.0
) -> tf.data.Dataset:
  """Dataset that yields model inputs plus metadata and WDL labels from seq_shards."""
  def generator():
    for shard_path in shard_paths:
      try:
        raw_ds = tf.data.TFRecordDataset(shard_path)
        for record_idx, raw_record in enumerate(raw_ds):
          if skip_rate > 0.0:
            # Deterministic skip decision keeps validation subset fixed across epochs.
            key = f"{shard_path}:{record_idx}".encode('utf-8')
            hashed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big')
            keep_prob = hashed / float(2**64)
            if keep_prob < skip_rate:
              continue
          try:
            (
              white_seq,
              black_seq,
              stm_name,
              stm_elo,
              opp_name,
              opp_elo,
              wdl_tensor,
            ) = parse_seq_example(raw_record)
            white_seq_np = white_seq.numpy().view(np.uint64)   # (NUM_GAMES, MAX_MOVES, 5)
            black_seq_np = black_seq.numpy().view(np.uint64)   # (NUM_GAMES, MAX_MOVES, 5)
            wdl_val = wdl_tensor.numpy()

            white_has_games = bool(np.any(np.any(white_seq_np > 0, axis=(1, 2))))
            black_has_games = bool(np.any(np.any(black_seq_np > 0, axis=(1, 2))))
            if not white_has_games or not black_has_games:
              continue

            white_planes = np.zeros((NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
            black_planes = np.zeros((NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
            white_mask = np.zeros((NUM_GAMES, MAX_MOVES), dtype=np.float32)
            black_mask = np.zeros((NUM_GAMES, MAX_MOVES), dtype=np.float32)

            for g_idx in range(NUM_GAMES):
              for m_idx in range(MAX_MOVES):
                struct = white_seq_np[g_idx, m_idx]
                if np.any(struct > 0):
                  p, _ = chessboard_struct_to_lc0_planes(struct, short=True)
                  white_planes[g_idx, m_idx] = p.astype(np.float32)
                  white_mask[g_idx, m_idx] = 1.0
            for g_idx in range(NUM_GAMES):
              for m_idx in range(MAX_MOVES):
                struct = black_seq_np[g_idx, m_idx]
                if np.any(struct > 0):
                  p, _ = chessboard_struct_to_lc0_planes(struct, short=True)
                  black_planes[g_idx, m_idx] = p.astype(np.float32)
                  black_mask[g_idx, m_idx] = 1.0

            yield {
              'seq0': white_planes,
              'seq1': black_planes,
              'mask0': white_mask,
              'mask1': black_mask,
              'raw_seq0': white_seq.numpy().astype(np.int64),
              'raw_seq1': black_seq.numpy().astype(np.int64),
              'stm_player_name': stm_name.numpy(),
              'opp_player_name': opp_name.numpy(),
              'stm_player_elo': np.int64(stm_elo.numpy()),
              'opp_player_elo': np.int64(opp_elo.numpy()),
            }, {
              'w': wdl_val.astype(np.float32),
              'e0': np.float32(stm_elo.numpy()),
              'e1': np.float32(opp_elo.numpy()),
            }
          except Exception as e:
            print(f"Skipping corrupted record in {shard_path}: {e}")
            continue
      except Exception as e:
        print(f"Skipping corrupted shard {shard_path}: {e}")
        continue

  output_signature = (
    {
      'seq0': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.float32),  # type: ignore
      'seq1': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=tf.float32),  # type: ignore
      'mask0': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES), dtype=tf.float32),                     # type: ignore
      'mask1': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES), dtype=tf.float32),                     # type: ignore
      'raw_seq0': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES, 5), dtype=tf.int64),                 # type: ignore
      'raw_seq1': tf.TensorSpec(shape=(NUM_GAMES, MAX_MOVES, 5), dtype=tf.int64),                 # type: ignore
      'stm_player_name': tf.TensorSpec(shape=(), dtype=tf.string),                    # type: ignore
      'opp_player_name': tf.TensorSpec(shape=(), dtype=tf.string),                    # type: ignore
      'stm_player_elo': tf.TensorSpec(shape=(), dtype=tf.int64),                      # type: ignore
      'opp_player_elo': tf.TensorSpec(shape=(), dtype=tf.int64),                      # type: ignore
    },
    {
      'w': tf.TensorSpec(shape=(3,), dtype=tf.float32),     # type: ignore
      'e0': tf.TensorSpec(shape=(), dtype=tf.float32),      # type: ignore
      'e1': tf.TensorSpec(shape=(), dtype=tf.float32),      # type: ignore
    }
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

    is_training = _training_flag(training)
    embedding = self.vit(seq, training=is_training, mask=m)  # (batch, hidden_dim)
    elo = self.regression_head(embedding, training=is_training)  # (batch, 1)
    return tf.squeeze(elo, axis=-1)  # (batch,)

@tf.keras.utils.register_keras_serializable()
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

    is_training = _training_flag(training)

    # Predict Elo per game, then average over valid games for each player.
    batch_size = tf.shape(seq0)[0]
    flat_seq0 = tf.reshape(seq0, [batch_size * NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8])
    flat_seq1 = tf.reshape(seq1, [batch_size * NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8])

    flat_mask0 = None
    flat_mask1 = None
    if mask0 is not None:
      flat_mask0 = tf.reshape(mask0, [batch_size * NUM_GAMES, MAX_MOVES])
    if mask1 is not None:
      flat_mask1 = tf.reshape(mask1, [batch_size * NUM_GAMES, MAX_MOVES])

    game_elo0 = self.elo_predictor({'seq': flat_seq0, 'mask': flat_mask0}, training=is_training)
    game_elo1 = self.elo_predictor({'seq': flat_seq1, 'mask': flat_mask1}, training=is_training)

    game_elo0 = tf.reshape(game_elo0, [batch_size, NUM_GAMES])
    game_elo1 = tf.reshape(game_elo1, [batch_size, NUM_GAMES])

    if mask0 is not None:
      game_valid0 = tf.cast(tf.reduce_any(mask0 > 0.0, axis=-1), dtype=game_elo0.dtype)
      elo0 = tf.math.divide_no_nan(
        tf.reduce_sum(game_elo0 * game_valid0, axis=-1),
        tf.reduce_sum(game_valid0, axis=-1),
      )
    else:
      elo0 = tf.reduce_mean(game_elo0, axis=-1)

    if mask1 is not None:
      game_valid1 = tf.cast(tf.reduce_any(mask1 > 0.0, axis=-1), dtype=game_elo1.dtype)
      elo1 = tf.math.divide_no_nan(
        tf.reduce_sum(game_elo1 * game_valid1, axis=-1),
        tf.reduce_sum(game_valid1, axis=-1),
      )
    else:
      elo1 = tf.reduce_mean(game_elo1, axis=-1)

    elo_diff = elo0 - elo1
    draw_margin = 21.57 # represent approx 0.062 win prob as found empirically
    # Convert elo difference to win/draw/loss probabilities using glicko 2. 0.9 represents typical rating deviation dampening
    p_win = 1 / (1 + tf.pow(10.0, 0.9 * (-elo_diff + draw_margin) / 400))
    p_loss = 1 / (1 + tf.pow(10.0, 0.9 * (elo_diff + draw_margin) / 400))
    p_draw = 1 - p_win - p_loss
    return {
      'w': tf.stack([p_win, p_draw, p_loss], axis=-1),  # (batch, 3)
      'e0': elo0,  # (batch,)
      'e1': elo1,  # (batch,)
    }


class PeriodicSampleLogger(tf.keras.callbacks.Callback):
  def __init__(
    self,
    val_dataset: tf.data.Dataset,
    output_dir: str,
    interval_batches: int,
    sample_count: int,
    log_filename: str,
  ):
    super().__init__()
    self.val_dataset = val_dataset
    self.interval_batches = max(1, int(interval_batches))
    self.sample_count = max(1, int(sample_count))
    self.log_path = os.path.join(output_dir, log_filename)
    self._val_iter = iter(self.val_dataset)
    with open(self.log_path, 'a', encoding='utf-8') as handle:
      handle.write(json.dumps({
        'event': 'run_start',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'interval_batches': self.interval_batches,
        'sample_count': self.sample_count,
      }) + '\n')

  def _next_batch(self):
    try:
      return next(self._val_iter)
    except StopIteration:
      self._val_iter = iter(self.val_dataset)
      return next(self._val_iter)

  @staticmethod
  def _model_inputs(inputs):
    return {
      'seq0': inputs['seq0'],
      'seq1': inputs['seq1'],
      'mask0': inputs['mask0'],
      'mask1': inputs['mask1'],
    }

  def _seq_to_fens(self, raw_seq: np.ndarray, mask: np.ndarray) -> List[List[str]]:
    all_games_fens = []
    max_games = min(raw_seq.shape[0], mask.shape[0])
    for g_idx in range(max_games):
      game_fens = []
      for m_idx in range(min(raw_seq[g_idx].shape[0], mask[g_idx].shape[0])):
        if mask[g_idx, m_idx] <= 0.0:
          continue
        struct = raw_seq[g_idx, m_idx].astype(np.uint64)
        fen = chessboard_struct_to_fen_like(struct)
        if fen:
          game_fens.append(fen)
      all_games_fens.append(game_fens)
    return all_games_fens

  def on_train_batch_end(self, batch, logs=None):
    if (batch + 1) % self.interval_batches != 0:
      return

    try:
      if self.model is None:
        return
      inputs, labels = self._next_batch()
      preds_out = self.model(self._model_inputs(inputs), training=False)

      if isinstance(preds_out, dict):
        preds_w = preds_out['w'].numpy()
        preds_e0 = preds_out['e0'].numpy()
        preds_e1 = preds_out['e1'].numpy()
      else:
        preds_w = preds_out.numpy()
        preds_e0 = np.full((preds_w.shape[0],), np.nan, dtype=np.float32)
        preds_e1 = np.full((preds_w.shape[0],), np.nan, dtype=np.float32)

      if isinstance(labels, dict):
        labels_np = labels['w'].numpy()
      else:
        labels_np = labels.numpy()

      raw_seq0 = inputs['raw_seq0'].numpy()
      raw_seq1 = inputs['raw_seq1'].numpy()
      mask0 = inputs['mask0'].numpy()
      mask1 = inputs['mask1'].numpy()
      stm_names = inputs['stm_player_name'].numpy()
      opp_names = inputs['opp_player_name'].numpy()
      stm_elos = inputs['stm_player_elo'].numpy()
      opp_elos = inputs['opp_player_elo'].numpy()

      total = min(self.sample_count, preds_w.shape[0])
      step = int(self.model.optimizer.iterations.numpy()) if self.model.optimizer is not None else -1
      with open(self.log_path, 'a', encoding='utf-8') as handle:
        for idx in range(total):
          pred = np.asarray(preds_w[idx], dtype=np.float64)
          true = np.asarray(labels_np[idx], dtype=np.float64)

          record = {
            'event': 'sample',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'batch': int(batch),
            'step': step,
            'sample_index': int(idx),
            'stm_player_name': _decode_name(stm_names[idx]),
            'opp_player_name': _decode_name(opp_names[idx]),
            'stm_player_elo': int(stm_elos[idx]),
            'opp_player_elo': int(opp_elos[idx]),
            'pred_stm_player_elo': float(preds_e0[idx]),
            'pred_opp_player_elo': float(preds_e1[idx]),
            'pred_wdl': np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).tolist(),
            'true_wdl': np.nan_to_num(true, nan=0.0, posinf=1.0, neginf=0.0).tolist(),
            'seq0_fens_by_game': self._seq_to_fens(raw_seq0[idx], mask0[idx]),
            'seq1_fens_by_game': self._seq_to_fens(raw_seq1[idx], mask1[idx]),
          }
          handle.write(json.dumps(record) + '\n')
    except Exception as e:
      print(f"Sample logging failed at batch {batch}: {e}")

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
  ffn_hidden_dim: int,
  sample_log_interval_batches: int,
  sample_log_count: int,
  sample_log_file: str,
):
  os.makedirs(output_dir, exist_ok=True)

  print(f"Loading data from {data_dir}...")

  seq_shard_paths = get_shard_paths(data_dir, "seq_shards")

  train_shards, val_shards = split_shards(seq_shard_paths, val_split)
  print(f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")

  train_dataset = create_seq_dataset(train_shards, batch_size, shuffle=True, repeat=True)
  val_dataset = create_seq_dataset(val_shards, batch_size, shuffle=False, repeat=False, skip_rate=0.99)

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
    # elo_predictor_model = tf.keras.models.load_model(start_checkpoint)
    # model = GameOutcomePredictor(elo_predictor=elo_predictor_model)
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
    loss={
      'e0': tf.keras.losses.MeanSquaredError(),
      'e1': tf.keras.losses.MeanSquaredError(),
    },
    metrics={
      'e0': [tf.keras.metrics.MeanAbsoluteError(name='e'), tf.keras.metrics.MeanSquaredError(name='m')],
      'e1': [tf.keras.metrics.MeanAbsoluteError(name='e'), tf.keras.metrics.MeanSquaredError(name='m')],
    }
  )

  model.build(input_shape={  # type: ignore
    'seq0': (None, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8),
    'seq1': (None, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8),
    'mask0': (None, NUM_GAMES, MAX_MOVES),
    'mask1': (None, NUM_GAMES, MAX_MOVES),
  })

  print("Running pre-training validation...")
  try:
    pre_val_metrics = model.evaluate(
      val_dataset,
      verbose='auto',
      return_dict=True,
    )
    print("Pre-Training Validation Metrics:")
    if isinstance(pre_val_metrics, dict):
      for metric_name, metric_value in sorted(pre_val_metrics.items()):
        print(f"  {metric_name}: {float(metric_value):.4f}")
    else:
      for idx, metric_value in enumerate(pre_val_metrics):
        print(f"  metric_{idx}: {float(metric_value):.4f}")
  except Exception as e:
    print(f"Pre-training validation failed: {e}")

  class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
      super().__init__()
      self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
      if logs is None or "learning_rate" in logs or self.model is None or self.model.optimizer is None:
        return
      if isinstance(logs, dict):
        logs["learning_rate"] = self.model.optimizer.learning_rate

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
    PeriodicSampleLogger(
      val_dataset=val_dataset,
      output_dir=output_dir,
      interval_batches=sample_log_interval_batches,
      sample_count=sample_log_count,
      log_filename=sample_log_file,
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
  def _last_metric(*names: str):
    for name in names:
      if name in history.history and len(history.history[name]) > 0:
        return history.history[name][-1]
    return None

  print("Final Training Metrics:")
  train_loss = _last_metric('loss')
  train_w_a = _last_metric('w_a', 'wdl_accuracy', 'wdl_acc')
  train_w_m = _last_metric('w_m', 'wdl_mse')
  train_w_e = _last_metric('w_e', 'wdl_mae')
  train_e0_e = _last_metric('e0_e', 'e0_mae', 'elo0_mae')
  train_e0_m = _last_metric('e0_m', 'e0_mse', 'elo0_mse')
  train_e1_e = _last_metric('e1_e', 'e1_mae', 'elo1_mae')
  train_e1_m = _last_metric('e1_m', 'e1_mse', 'elo1_mse')
  if train_loss is not None:
    print(f"  Loss:       {train_loss:.2f}")
  if train_w_a is not None:
    print(f"  W A:        {train_w_a:.4f}")
  if train_w_m is not None:
    print(f"  W M:        {train_w_m:.2f}")
  if train_w_e is not None:
    print(f"  W E:        {train_w_e:.2f}")
  if train_e0_e is not None:
    print(f"  E0 E:       {train_e0_e:.2f}")
  if train_e0_m is not None:
    print(f"  E0 M:       {train_e0_m:.2f}")
  if train_e1_e is not None:
    print(f"  E1 E:       {train_e1_e:.2f}")
  if train_e1_m is not None:
    print(f"  E1 M:       {train_e1_m:.2f}")

  print("Final Validation Metrics:")
  val_loss = _last_metric('val_loss')
  val_w_a = _last_metric('val_w_a', 'val_wdl_accuracy', 'val_wdl_acc')
  val_w_m = _last_metric('val_w_m', 'val_wdl_mse')
  val_w_e = _last_metric('val_w_e', 'val_wdl_mae')
  val_e0_e = _last_metric('val_e0_e', 'val_e0_mae', 'val_elo0_mae')
  val_e0_m = _last_metric('val_e0_m', 'val_e0_mse', 'val_elo0_mse')
  val_e1_e = _last_metric('val_e1_e', 'val_e1_mae', 'val_elo1_mae')
  val_e1_m = _last_metric('val_e1_m', 'val_e1_mse', 'val_elo1_mse')
  if val_loss is not None:
    print(f"  Loss:       {val_loss:.2f}")
  if val_w_a is not None:
    print(f"  W A:        {val_w_a:.4f}")
  if val_w_m is not None:
    print(f"  W M:        {val_w_m:.2f}")
  if val_w_e is not None:
    print(f"  W E:        {val_w_e:.2f}")
  if val_e0_e is not None:
    print(f"  E0 E:       {val_e0_e:.2f}")
  if val_e0_m is not None:
    print(f"  E0 M:       {val_e0_m:.2f}")
  if val_e1_e is not None:
    print(f"  E1 E:       {val_e1_e:.2f}")
  if val_e1_m is not None:
    print(f"  E1 M:       {val_e1_m:.2f}")

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
  parser.add_argument("--sample-log-interval-batches", type=int, default=500)
  parser.add_argument("--sample-log-count", type=int, default=10)
  parser.add_argument("--sample-log-file", type=str, default="prediction_samples.jsonl")

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
    ffn_hidden_dim=args.ffn_hidden_dim,
    sample_log_interval_batches=args.sample_log_interval_batches,
    sample_log_count=args.sample_log_count,
    sample_log_file=args.sample_log_file,
  )
# python3 -m stylometry.ViTOneHot.train_stylometry stylometry/ViTOneHot/data/run2026-03-16/ --output-dir stylometry/ViTOneHot/models/run2026-03-21 --start-checkpoint stylometry/ViTOneHot/models/run2026-03-02-lr-0.00005/checkpoint_epoch_36.keras
