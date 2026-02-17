import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from stylometry.WinRateStylo.train_wr_stylometry import (
  ScaffoldedViTAndWinRate,
  chessboard_struct_to_lc0_planes,
  parse_position_example,
  MAX_MOVES,
  SEQ_PLANES,
  POS_PLANES,
)
import stylometry.WinRateStylo.pgn_to_training_data as pgn_data
import chess
import chess.engine
import random
from typing import List
import logging

logger = logging.getLogger(__name__)

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def flip_square(sq):
  rank = sq // 8
  file_idx = sq % 8
  return (7 - rank) * 8 + file_idx

def struct_to_board(structs):
  structs = np.asarray(structs, dtype=np.uint64)
  metadata = int(structs[32])
  castle_and_stm = (metadata >> 24) & 0xFF
  is_black_stm = bool(castle_and_stm & 1)
  halfmove_clock = (metadata >> 16) & 0xFF

  occ = int(structs[0])
  pcs_1 = int(structs[1])
  pcs_2 = int(structs[2])

  board = chess.Board.empty()
  board.turn = chess.BLACK if is_black_stm else chess.WHITE

  pcs_idx = 0
  temp_occ = occ
  while temp_occ:
    sq = (temp_occ & -temp_occ).bit_length() - 1
    if pcs_idx < 16:
      val = (pcs_1 >> (4 * pcs_idx)) & 0xF
    else:
      val = (pcs_2 >> (4 * (pcs_idx - 16))) & 0xF
    pcs_idx += 1

    piece_type = PIECE_TYPES[val & 0x7]
    is_opponent = bool(val & 0x8)
    real_sq = flip_square(sq) if is_black_stm else sq

    if is_opponent:
      color = chess.WHITE if is_black_stm else chess.BLACK
    else:
      color = chess.BLACK if is_black_stm else chess.WHITE

    board.set_piece_at(real_sq, chess.Piece(piece_type, color))
    temp_occ &= temp_occ - 1

  board.castling_rights = chess.BB_EMPTY
  stm = board.turn
  opp = not stm
  if castle_and_stm & 0x2:
    board.castling_rights |= (chess.BB_A1 if stm == chess.WHITE else chess.BB_A8)
  if castle_and_stm & 0x4:
    board.castling_rights |= (chess.BB_H1 if stm == chess.WHITE else chess.BB_H8)
  if castle_and_stm & 0x8:
    board.castling_rights |= (chess.BB_A1 if opp == chess.WHITE else chess.BB_A8)
  if castle_and_stm & 0x10:
    board.castling_rights |= (chess.BB_H1 if opp == chess.WHITE else chess.BB_H8)

  board.halfmove_clock = halfmove_clock
  return board

def engine_wdl_to_stm_probs(wdl_result, board_turn):
  if isinstance(wdl_result, chess.engine.Wdl):
    w = wdl_result.wins / 1000.0
    d = wdl_result.draws / 1000.0
    l = wdl_result.losses / 1000.0
  else:
    w, d, l = float(wdl_result[0]), float(wdl_result[1]), float(wdl_result[2])
    total = w + d + l
    if total > 0:
      w, d, l = w / total, d / total, l / total
    else:
      w, d, l = 0.0, 1.0, 0.0
  if board_turn == chess.BLACK:
    w, l = l, w
  probs = np.array([w, d, l])
  return np.clip(probs / np.sum(probs), 1e-7, 1.0 - 1e-7)

def get_tfrecord_paths(paths: List[str]) -> List[str]:
  result = []
  for p in paths:
    if os.path.isfile(p) and p.endswith(".tfrecord"):
      result.append(p)
    elif os.path.isdir(p):
      found = sorted(glob.glob(os.path.join(p, "**", "*.tfrecord"), recursive=True))
      result.extend(found)
  logger.info(f"Found {len(result)} tfrecord files")
  return result

def process_tfrecords(
  tfrecord_paths: List[str],
  model_path: str = "",
  skip_rate: float = 0.0,
  batch_size: int = 32,
):
  model = keras.saving.load_model(
    model_path,
    custom_objects={"ScaffoldedViTAndWinRate": ScaffoldedViTAndWinRate}
  )

  total = 0
  m_correct = 0
  m_confusion = np.zeros((3, 3), dtype=np.int64)
  sf_correct = 0
  sf_confusion = np.zeros((3, 3), dtype=np.int64)
  m_correct_sf_wrong = 0

  for tfrecord_idx, tfrecord_path in enumerate(tfrecord_paths):
    logger.info(f"Processing {tfrecord_path} ({tfrecord_idx + 1}/{len(tfrecord_paths)})...")
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    pos_batch = []
    clocks_batch = []
    wdl_batch = []
    structs_batch = []

    for raw_record in dataset:
      if random.random() < skip_rate:
        continue

      try:
        stm_seq, opp_seq, full_board, wdl = parse_position_example(raw_record)
      except Exception as e:
        logger.warning(f"Skipping corrupted record: {e}")
        continue

      full_board_np = full_board.numpy().astype(np.uint64)
      wdl_np = wdl.numpy()
      pos_planes, pos_clocks = chessboard_struct_to_lc0_planes(full_board_np, short=False)

      pos_batch.append(pos_planes.flatten())
      clocks_batch.append(pos_clocks)
      wdl_batch.append(wdl_np)
      structs_batch.append(full_board_np)

      if len(pos_batch) >= batch_size:
        results = run_batch(model, pos_batch, clocks_batch, wdl_batch, structs_batch)
        total += results["count"]
        m_correct += results["m_correct"]
        m_confusion += results["m_confusion"]
        sf_correct += results["sf_correct"]
        sf_confusion += results["sf_confusion"]
        m_correct_sf_wrong += results["m_correct_sf_wrong"]
        pos_batch = []
        clocks_batch = []
        wdl_batch = []
        structs_batch = []

    if len(pos_batch) > 0:
      results = run_batch(model, pos_batch, clocks_batch, wdl_batch, structs_batch)
      total += results["count"]
      m_correct += results["m_correct"]
      m_confusion += results["m_confusion"]
      sf_correct += results["sf_correct"]
      sf_confusion += results["sf_confusion"]
      m_correct_sf_wrong += results["m_correct_sf_wrong"]
      pos_batch = []
      clocks_batch = []
      wdl_batch = []
      structs_batch = []

    if total > 0:
      logger.info(
        f"Running - n={total}, "
        f"model_acc={m_correct / total:.4f}, "
        f"sf_acc={sf_correct / total:.4f}, "
        f"model_right_sf_wrong={m_correct_sf_wrong}"
      )

  logger.info("=== Final Results ===")
  logger.info(f"Total positions: {total}")
  if total > 0:
    logger.info(f"Model - correct={m_correct}/{total} ({m_correct / total:.4f})")
    logger.info(f"Model confusion [pred][actual] (W/D/L):\n{m_confusion}")
    logger.info(f"SF    - correct={sf_correct}/{total} ({sf_correct / total:.4f})")
    logger.info(f"SF confusion [pred][actual] (W/D/L):\n{sf_confusion}")
    logger.info(f"Model correct & SF wrong: {m_correct_sf_wrong}")


def run_batch(model, pos_batch, clocks_batch, wdl_batch, structs_batch):
  pos_tensor = tf.cast(np.array(pos_batch), tf.float32)
  clocks_tensor = tf.zeros((len(clocks_batch), 2), dtype=tf.float32)
  wdl_np = np.array(wdl_batch)
  n = len(pos_batch)

  inputs = {
    'input1': tf.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
    'input2': tf.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
    'mask1': tf.zeros((n, 5, MAX_MOVES)),
    'mask2': tf.zeros((n, 5, MAX_MOVES)),
    'pos': pos_tensor,
    'pos_clocks': clocks_tensor,
  }
  logits = model(inputs, training=False).numpy()

  m_pred = np.argmax(logits, axis=-1)
  actual = np.argmax(wdl_np, axis=-1)

  m_confusion = np.zeros((3, 3), dtype=np.int64)
  sf_confusion = np.zeros((3, 3), dtype=np.int64)
  sf_preds = np.zeros(n, dtype=np.int64)

  for i in range(n):
    board = struct_to_board(structs_batch[i])
    sf_wdl_raw = pgn_data.engine_eval(board)
    sf_probs = engine_wdl_to_stm_probs(sf_wdl_raw, board.turn)
    sf_preds[i] = int(np.argmax(sf_probs))

  for p, a in zip(m_pred, actual):
    m_confusion[p, a] += 1
  for p, a in zip(sf_preds, actual):
    sf_confusion[p, a] += 1

  m_correct_mask = m_pred == actual
  sf_wrong_mask = sf_preds != actual
  m_correct_sf_wrong = int(np.sum(m_correct_mask & sf_wrong_mask))

  if m_correct_sf_wrong > 0:
    wdl_names = ["W", "D", "L"]
    for i in range(n):
      if m_correct_mask[i] and sf_wrong_mask[i]:
        board = struct_to_board(structs_batch[i])
        logger.info(
          f"Model right, SF wrong: actual={wdl_names[actual[i]]}, "
          f"model={wdl_names[m_pred[i]]}, sf={wdl_names[sf_preds[i]]}\n"
          f"{board.fen()}"
        )

  return {
    "count": n,
    "m_correct": int(np.sum(m_correct_mask)),
    "m_confusion": m_confusion,
    "sf_correct": int(np.sum(sf_preds == actual)),
    "sf_confusion": sf_confusion,
    "m_correct_sf_wrong": m_correct_sf_wrong,
  }


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  np.random.seed(42)
  random.seed(42)

  parser = argparse.ArgumentParser(description="Evaluate model vs Stockfish on position tfrecords")
  parser.add_argument("inputs", nargs="+", help="Input tfrecord file(s) or folder(s)")
  parser.add_argument("model", help="Path to the trained model")
  parser.add_argument("--engine-path", type=str, default="./stylometry/WinRateStylo/engines/stockfish/stockfish-ubuntu-x86-64-avx2")
  parser.add_argument("--skip-rate", type=float, default=0.0)
  parser.add_argument("--batch-size", type=int, default=32)

  args = parser.parse_args()

  pgn_data.DO_ENGINE_EVAL = True
  pgn_data.engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
  pgn_data.engine.configure({"UCI_ShowWDL": True})

  tfrecord_files = get_tfrecord_paths(args.inputs)

  try:
    process_tfrecords(
      tfrecord_paths=tfrecord_files,
      model_path=args.model,
      skip_rate=args.skip_rate,
      batch_size=args.batch_size,
    )
  finally:
    pgn_data.engine.quit()