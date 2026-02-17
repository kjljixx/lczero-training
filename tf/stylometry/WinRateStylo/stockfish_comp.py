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
  MAX_MOVES,
  SEQ_PLANES,
  POS_PLANES,
)
import stylometry.WinRateStylo.pgn_to_training_data as pgn_data
from stylometry.WinRateStylo.pgn_to_training_data import (
  get_pgns,
  board_to_chessboard_struct,
  PlayerIndexMapper,
)
import chess
import chess.pgn
import chess.polyglot
import chess.engine
import random
from typing import List
import logging

logger = logging.getLogger(__name__)

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

def process_pgns(
  pgn_paths: List[str],
  model_path: str = "",
  skip_rate: float = 0.0,
  batch_size: int = 32,
  min_moves: int = 3,
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
  game_count = 0

  pos_batch = []
  clocks_batch = []
  wdl_batch = []
  board_batch = []
  meta_batch = []  # (white_name, black_name, white_elo, black_elo, stm_color)

  for pgn_file_idx, pgn_file_path in enumerate(pgn_paths):
    logger.info(f"Processing {pgn_file_path} ({pgn_file_idx + 1}/{len(pgn_paths)})...")
    pgn_file = open(pgn_file_path, 'r')

    while True:
      game = chess.pgn.read_game(pgn_file)
      if game is None:
        break

      white_name = game.headers.get("White", "Unknown_White")
      black_name = game.headers.get("Black", "Unknown_Black")
      white_elo = game.headers.get("WhiteElo", "?")
      black_elo = game.headers.get("BlackElo", "?")
      result = game.headers.get("Result", "*")

      if result not in ("1-0", "0-1", "1/2-1/2"):
        continue

      board_history = []
      clock_history = []
      white_clock = 600
      black_clock = 600
      position_hashes = []
      repetition_counts = []

      board = game.board()

      for move_num, node in enumerate(game.mainline()):
        board_copy = board.copy(stack=False)
        board_history.insert(0, board_copy)

        pos_hash = chess.polyglot.zobrist_hash(board)
        position_hashes.insert(0, pos_hash)

        rep_count = sum(1 for h in position_hashes if h == pos_hash) - 1
        repetition_counts.insert(0, rep_count)

        if len(board_history) > 8:
          board_history = board_history[:8]
          clock_history = clock_history[:8]
          position_hashes = position_hashes[:8]
          repetition_counts = repetition_counts[:8]

        if move_num >= min_moves:
          if random.random() < skip_rate:
            pass  # skip this position
          else:
            board_planes_struct = pgn_data.board_to_chessboard_struct(
              board_history, clock_history, repetition_counts
            )
            pos_planes, pos_clocks = chessboard_struct_to_lc0_planes(
              board_planes_struct, short=False
            )

            # WDL from STM perspective
            if board.turn == chess.WHITE:
              wdl = [1, 0, 0] if result == "1-0" else [0, 0, 1] if result == "0-1" else [0, 1, 0]
            else:
              wdl = [1, 0, 0] if result == "0-1" else [0, 0, 1] if result == "1-0" else [0, 1, 0]

            pos_batch.append(pos_planes.flatten())
            clocks_batch.append(pos_clocks)
            wdl_batch.append(wdl)
            board_batch.append(board.copy())
            meta_batch.append((white_name, black_name, white_elo, black_elo, board.turn))

        curr_clock = node.clock()
        if board.turn == chess.BLACK and curr_clock is not None:
          black_clock = int(curr_clock)
        elif board.turn == chess.WHITE and curr_clock is not None:
          white_clock = int(curr_clock)
        if len(clock_history) == 0:
          clock_history.insert(0, (white_clock << 32) | black_clock)
        clock_history.insert(0, (white_clock << 32) | black_clock)

        board.push(node.move)

      game_count += 1

      if len(pos_batch) >= batch_size:
        results = run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch)
        total += results["count"]
        m_correct += results["m_correct"]
        m_confusion += results["m_confusion"]
        sf_correct += results["sf_correct"]
        sf_confusion += results["sf_confusion"]
        m_correct_sf_wrong += results["m_correct_sf_wrong"]
        pos_batch = []
        clocks_batch = []
        wdl_batch = []
        board_batch = []
        meta_batch = []

      if game_count % 100 == 0 and total > 0:
        logger.info(
          f"Running - games={game_count}, n={total}, "
          f"model_acc={m_correct / total:.4f}, "
          f"sf_acc={sf_correct / total:.4f}, "
          f"model_right_sf_wrong={m_correct_sf_wrong}"
        )

    pgn_file.close()

  # flush remaining
  if len(pos_batch) > 0:
    results = run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch)
    total += results["count"]
    m_correct += results["m_correct"]
    m_confusion += results["m_confusion"]
    sf_correct += results["sf_correct"]
    sf_confusion += results["sf_confusion"]
    m_correct_sf_wrong += results["m_correct_sf_wrong"]

  logger.info("=== Final Results ===")
  logger.info(f"Total games: {game_count}")
  logger.info(f"Total positions: {total}")
  if total > 0:
    logger.info(f"Model - correct={m_correct}/{total} ({m_correct / total:.4f})")
    logger.info(f"Model confusion [pred][actual] (W/D/L):\n{m_confusion}")
    logger.info(f"SF    - correct={sf_correct}/{total} ({sf_correct / total:.4f})")
    logger.info(f"SF confusion [pred][actual] (W/D/L):\n{sf_confusion}")
    logger.info(f"Model correct & SF wrong: {m_correct_sf_wrong}")


def run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch):
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
    sf_wdl_raw = pgn_data.engine_eval(board_batch[i])
    sf_probs = engine_wdl_to_stm_probs(sf_wdl_raw, board_batch[i].turn)
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
        white_name, black_name, white_elo, black_elo, stm = meta_batch[i]
        logger.info(
          f"Model right, SF wrong: actual={wdl_names[actual[i]]}, "
          f"model={wdl_names[m_pred[i]]}, sf={wdl_names[sf_preds[i]]}, "
          f"logits={logits[i]}\n"
          f"  White: {white_name} ({white_elo}), Black: {black_name} ({black_elo}), "
          f"STM: {'Black' if stm == chess.BLACK else 'White'}\n"
          f"  {board_batch[i].fen()}"
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

  parser = argparse.ArgumentParser(description="Evaluate model vs Stockfish on PGN games")
  parser.add_argument("inputs", nargs="+", help="Input PGN file(s) or folder(s)")
  parser.add_argument("model", help="Path to the trained model")
  parser.add_argument("--engine-path", type=str, default="./stylometry/WinRateStylo/engines/stockfish/stockfish-ubuntu-x86-64-avx2")
  parser.add_argument("--skip-rate", type=float, default=0.0, help="Probability of skipping each position")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--min-moves", type=int, default=3, help="Skip positions before this move number")

  args = parser.parse_args()

  pgn_data.DO_ENGINE_EVAL = True
  pgn_data.engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)
  pgn_data.engine.configure({"UCI_ShowWDL": True})

  pgn_files = get_pgns(args.inputs)

  try:
    process_pgns(
      pgn_paths=pgn_files,
      model_path=args.model,
      skip_rate=args.skip_rate,
      batch_size=args.batch_size,
      min_moves=args.min_moves,
    )
  finally:
    pgn_data.engine.quit()