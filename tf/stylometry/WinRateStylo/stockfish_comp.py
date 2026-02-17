import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
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
  extract_game_data,
  PlayerIndexMapper,
)
import chess
import chess.pgn
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

def process_seq_to_planes(seq_np):
  """Convert a (5, max_moves, 5) uint64 sequence array into planes and masks."""
  num_games = seq_np.shape[0]
  max_moves = seq_np.shape[1]
  planes = np.zeros((num_games, max_moves, SEQ_PLANES, 8, 8), dtype=np.int8)
  masks = np.zeros((num_games, max_moves), dtype=np.int8)
  for g_idx in range(num_games):
    for m_idx in range(max_moves):
      struct = seq_np[g_idx, m_idx]
      if np.any(struct > 0):
        planes[g_idx, m_idx], _ = chessboard_struct_to_lc0_planes(struct, short=True)
        masks[g_idx, m_idx] = 1
  return planes, masks


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
  player_mapper = PlayerIndexMapper()

  total = 0
  m_correct = 0
  m_confusion = np.zeros((3, 3), dtype=np.int64)
  sf_correct = 0
  sf_confusion = np.zeros((3, 3), dtype=np.int64)
  m_correct_sf_wrong = 0
  game_count = 0

  # Per-player sequence accumulation (player_idx -> list of game move sequences)
  player_sequences = {}  # {player_idx: [game_seq, ...]}

  pos_batch = []
  clocks_batch = []
  wdl_batch = []
  board_batch = []
  meta_batch = []  # (white_name, black_name, white_elo, black_elo, stm_color)
  seq_batch = []   # (stm_games, opp_games) each is list of game move lists

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

      # Use extract_game_data to get sequences and positions
      game_data = extract_game_data(game, player_mapper, max_moves=MAX_MOVES)
      sequences_data, positions_data = game_data

      # sequences_data: [(white_moves, white_idx), (black_moves, black_idx)]
      for seq_moves, player_idx in sequences_data:
        if player_idx not in player_sequences:
          player_sequences[player_idx] = []
        player_sequences[player_idx].append(seq_moves)

      # Replay the game to capture chess.Board objects at each position
      # extract_game_data yields positions starting at move_num >= 2
      replay_boards = []
      replay_board = game.board()
      for move_num, node in enumerate(game.mainline()):
        if move_num >= 2:
          replay_boards.append(replay_board.copy())
        replay_board.push(node.move)

      # positions_data: [((opp_idx, stm_idx, board_struct), wdl_label), ...]
      for pos_i, ((opp_idx, stm_idx, board_struct), wdl_label) in enumerate(positions_data):
        if random.random() < skip_rate:
          continue

        pos_planes, pos_clocks = chessboard_struct_to_lc0_planes(
          board_struct, short=False
        )

        # Determine STM color from the struct metadata
        stm_color_bit = (int(board_struct[32]) >> 24) & 0x1
        stm_color = chess.BLACK if stm_color_bit else chess.WHITE

        # Sample up to 5 game sequences per player
        def sample_games(player_idx, max_games=5):
          if player_idx in player_sequences and len(player_sequences[player_idx]) > 0:
            n_avail = min(max_games, len(player_sequences[player_idx]))
            return [g for g in random.sample(player_sequences[player_idx], n_avail)]
          return []

        stm_games = sample_games(stm_idx)
        opp_games = sample_games(opp_idx)

        pos_batch.append(pos_planes.flatten())
        clocks_batch.append(pos_clocks)
        wdl_batch.append(wdl_label)
        board_batch.append(replay_boards[pos_i] if pos_i < len(replay_boards) else chess.Board())
        meta_batch.append((white_name, black_name, white_elo, black_elo, stm_color))
        seq_batch.append((stm_games, opp_games))

      game_count += 1

      if len(pos_batch) >= batch_size:
        results = run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch, seq_batch)
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
        seq_batch = []

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
    results = run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch, seq_batch)
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


def build_seq_tensor(seq_batch_side, n):
  """Build (n, 5, MAX_MOVES, 5) uint64 array from a list of game-move-lists."""
  arr = np.zeros((n, 5, MAX_MOVES, 5), dtype=np.uint64)
  for i, games in enumerate(seq_batch_side):
    for g_idx, game_moves in enumerate(games[:5]):
      num_moves = min(len(game_moves), MAX_MOVES)
      if num_moves > 0:
        arr[i, g_idx, :num_moves, :] = np.array(game_moves[:num_moves], dtype=np.uint64)
  return arr


def run_batch(model, pos_batch, clocks_batch, wdl_batch, board_batch, meta_batch, seq_batch):
  pos_tensor = tf.cast(np.array(pos_batch), tf.float32)
  clocks_tensor = tf.zeros((len(clocks_batch), 2), dtype=tf.float32)
  wdl_np = np.array(wdl_batch)
  n = len(pos_batch)

  # Build sequence inputs from per-position game lists
  stm_games_list = [sb[0] for sb in seq_batch]
  opp_games_list = [sb[1] for sb in seq_batch]

  stm_seq_np = build_seq_tensor(stm_games_list, n)
  opp_seq_np = build_seq_tensor(opp_games_list, n)

  # Convert uint64 structs to planes + masks
  stm_planes = np.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.int8)
  stm_masks = np.zeros((n, 5, MAX_MOVES), dtype=np.int8)
  opp_planes = np.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.int8)
  opp_masks = np.zeros((n, 5, MAX_MOVES), dtype=np.int8)

  for i in range(n):
    sp, sm = process_seq_to_planes(stm_seq_np[i])
    stm_planes[i] = sp
    stm_masks[i] = sm
    op, om = process_seq_to_planes(opp_seq_np[i])
    opp_planes[i] = op
    opp_masks[i] = om

  inputs = {
    'input1': tf.constant(stm_planes),
    'input2': tf.constant(opp_planes),
    'mask1': tf.constant(stm_masks),
    'mask2': tf.constant(opp_masks),
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
        stm_num_games = len(seq_batch[i][0])
        opp_num_games = len(seq_batch[i][1])
        logger.info(
          f"Model right, SF wrong: actual={wdl_names[actual[i]]}, "
          f"model={wdl_names[m_pred[i]]}, sf={wdl_names[sf_preds[i]]}, "
          f"logits={logits[i]}\n"
          f"  White: {white_name} ({white_elo}), Black: {black_name} ({black_elo}), "
          f"STM: {'Black' if stm == chess.BLACK else 'White'}, "
          f"STM games: {stm_num_games}, OPP games: {opp_num_games}\n"
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