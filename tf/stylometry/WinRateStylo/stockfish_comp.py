import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from stylometry.WinRateStylo.pgn_to_training_data import board_to_chessboard_struct, engine_eval, DO_ENGINE_EVAL, get_pgns
from stylometry.WinRateStylo.train_wr_stylometry import ScaffoldedViTAndWinRate, chessboard_struct_to_lc0_planes, MAX_MOVES, SEQ_PLANES
import chess
import chess.pgn
import chess.engine
import chess.polyglot
import random
from typing import List
import logging

logger = logging.getLogger(__name__)

DO_ENGINE_EVAL = True
engine = chess.engine.SimpleEngine.popen_uci("./stylometry/WinRateStylo/engines/stockfish/stockfish-ubuntu-x86-64-avx2")
engine.configure({"UCI_ShowWDL": True})

def process_pgns(
    pgn_paths: List[str],
    skip_rate: float = 0.95,
    model_path: str = ""
):
  model = keras.saving.load_model(
    model_path,
    custom_objects={"ScaffoldedViTAndWinRate": ScaffoldedViTAndWinRate}
  )

  for pgn_file_idx, pgn_file_path in enumerate(pgn_paths):
    logger.info(f"Processing {pgn_file_path}...")
    pgn_file = open(pgn_file_path, 'r')
    while True:
      game = chess.pgn.read_game(pgn_file)
      if game is None:
        break
      if game.headers["White"] == "?" or game.headers["Black"] == "?":
        continue

      board_history = []
      clock_history = []
      white_clock = 600 #assume 10 minutes if not specified
      black_clock = 600
      position_hashes = []
      repetition_counts = []

      board = game.board()
      result = game.headers.get("Result", "*")

      for move_num, pos in enumerate(game.mainline()):
        board_copy = board.copy(stack=False)
        board_history.insert(0, board_copy)

        pos_hash = chess.polyglot.zobrist_hash(board)
        position_hashes.insert(0, pos_hash)

        rep_count = sum(1 for h in position_hashes if h == pos_hash) - 1
        repetition_counts.insert(0, rep_count)
        if move_num >= 2 and random.random() > skip_rate:
          board_planes = board_to_chessboard_struct(board_history, clock_history, repetition_counts)
          planes, clocks = chessboard_struct_to_lc0_planes(board_planes)
          pos_flat = tf.reshape(tf.cast(planes, tf.float32), (1, -1))
          clocks_tensor = tf.reshape(tf.cast(clocks, tf.float32), (1, 2))
          inputs = {
            'input1': tf.zeros((1, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
            'input2': tf.zeros((1, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
            'mask1': tf.zeros((1, 5, MAX_MOVES)),
            'mask2': tf.zeros((1, 5, MAX_MOVES)),
            'pos': pos_flat,
            'pos_clocks': clocks_tensor,
          }
          prediction = model(inputs, training=False)
          logger.info(f"Model prediction: {prediction}")
          stockfish_wdl = engine_eval(board)
          logger.info(f"Stockfish WDL: {stockfish_wdl}")
          actual_result = 0.5 if result == "1/2-1/2" else (1.0 if (result == "1-0" and board.turn == chess.WHITE) or (result == "0-1" and board.turn == chess.BLACK) else 0.0)
          logger.info(f"Actual result from this position: {actual_result}")

        curr_clock = pos.clock()
        if board.turn == chess.BLACK and curr_clock is not None:
          black_clock = int(curr_clock)
        elif board.turn == chess.WHITE and curr_clock is not None:
          white_clock = int(curr_clock)
        if len(clock_history) == 0:
          clock_history.insert(0, (white_clock << 32) | black_clock)
        clock_history.insert(0, (white_clock << 32) | black_clock)

        board.push(pos.move)

if __name__ == "__main__":
  import argparse
  import glob
  import os

  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  np.random.seed(42)
  random.seed(42)

  parser = argparse.ArgumentParser(
    description="Convert PGN files to training data"
  )
  parser.add_argument("inputs", nargs="+",
                     help="Input PGN file(s) or folder(s) containing PGN files")
  parser.add_argument("model", help="Path to the trained model to use for evaluation")

  args = parser.parse_args()

  pgn_files = get_pgns(args.inputs)

  process_pgns(pgn_paths=pgn_files, model_path=args.model)