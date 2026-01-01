import chess
import chess.engine
import chess.pgn
import numpy as np
import random
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PlayerIndexMapper:
  def __init__(self):
    self.player_to_idx: Dict[str, int] = {}
    self.idx_to_player: Dict[int, str] = {}
    self.next_idx = 0

  def get_or_create_index(self, player_name: str) -> int:
    if player_name not in self.player_to_idx:
      self.player_to_idx[player_name] = self.next_idx
      self.idx_to_player[self.next_idx] = player_name
      self.next_idx += 1
    return self.player_to_idx[player_name]

  def get_index(self, player_name: str) -> Optional[int]:
    return self.player_to_idx.get(player_name)

  def num_players(self) -> int:
    return self.next_idx

  def save(self, filepath: str):
    with open(filepath, 'w') as f:
      for player, idx in sorted(self.player_to_idx.items(), key=lambda x: x[1]):
        f.write(f"{idx}\t{player}\n")

  def load(self, filepath: str):
    self.player_to_idx.clear()
    self.idx_to_player.clear()
    self.next_idx = 0

    with open(filepath, 'r') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        idx_str, player = line.split('\t', 1)
        idx = int(idx_str)
        self.player_to_idx[player] = idx
        self.idx_to_player[idx] = player
        self.next_idx = max(self.next_idx, idx + 1)

def get_pgns(
  pgn_paths: List[str]
):
  pgn_files = []
  for pgn_path in pgn_paths:
    if os.path.isfile(pgn_path):
      if pgn_path.endswith('.pgn'):
        pgn_files.append(pgn_path)
      else:
        logger.warning(f"Skipping non-PGN file: {pgn_path}")
    elif os.path.isdir(pgn_path):
      logger.info(f"Searching dir: {pgn_path}")
      folder_pgns = glob.glob(os.path.join(pgn_path, "*.pgn"))
      if folder_pgns:
        pgn_files.extend(folder_pgns)
        logger.info(f"Found: {len(folder_pgns)} PGNs in {pgn_path}")
      else:
        logger.warning(f"No PGNs in: {pgn_path}")
    else:
      logger.warning(f"Path not found: {pgn_path}")

  logger.info(f"# PGNs found: {len(pgn_files)}")

  return pgn_files

def board_history_to_lc0_planes(board_history: List[chess.Board], repetition_counts: List[int]) -> np.ndarray:
  planes = np.zeros((112, 8, 8), dtype=np.int8)

  history_len = len(board_history)
  if history_len > 8:
    board_history = board_history[:8]
    repetition_counts = repetition_counts[:8]
    history_len = 8

  # (8 positions × 13 planes each)
  for hist_idx in range(8):
    plane_offset = hist_idx * 13

    if hist_idx < history_len:
      board = board_history[hist_idx]
      rep_count = repetition_counts[hist_idx] if hist_idx < len(repetition_counts) else 0

      us_color = board_history[0].turn  # stm
      them_color = not us_color

      for piece_type in range(1, 7):  # PAWN=1 to KING=6
        piece_plane = plane_offset + piece_type - 1
        piece_bb = board.pieces(piece_type, us_color)
        for square in piece_bb:
          chess_rank = chess.square_rank(square)
          if us_color == chess.BLACK:
            chess_rank = 7 - chess_rank
          chess_file = chess.square_file(square)
          planes[piece_plane, chess_rank, chess_file] = 1.0

      for piece_type in range(1, 7):
        piece_plane = plane_offset + 6 + piece_type - 1
        piece_bb = board.pieces(piece_type, them_color)
        for square in piece_bb:
          chess_rank = chess.square_rank(square)
          if them_color == chess.BLACK:
            chess_rank = 7 - chess_rank
          chess_file = chess.square_file(square)
          planes[piece_plane, chess_rank, chess_file] = 1.0

      if rep_count >= 1:
        planes[plane_offset + 12, :, :] = 1.0

  current_board = board_history[0] if history_len > 0 else chess.Board()
  us_color = current_board.turn
  them_color = not us_color

  if current_board.has_queenside_castling_rights(us_color): # our queenside castling
    planes[104, :, :] = 1.0  
  if current_board.has_kingside_castling_rights(us_color): # our kingside castling
    planes[105, :, :] = 1.0
  if current_board.has_queenside_castling_rights(them_color): # their queenside castling
    planes[106, :, :] = 1.0
  if current_board.has_kingside_castling_rights(them_color): # their kingside castling
    planes[107, :, :] = 1.0 

  planes[108, :, :] = 1.0 if current_board.turn == chess.BLACK else 0.0 # stm
  planes[109, :, :] = current_board.halfmove_clock

  planes[110, :, :] = 0.0
  planes[111, :, :] = 1.0

  return planes

def extract_game_data(
  game: chess.pgn.Game,
  player_mapper: PlayerIndexMapper,
  max_moves: int = 100
):
  white_name = game.headers.get("White", "Unknown_White")
  black_name = game.headers.get("Black", "Unknown_Black")

  white_idx = player_mapper.get_or_create_index(white_name)
  black_idx = player_mapper.get_or_create_index(black_name)

  board_history = []
  position_hashes = []
  repetition_counts = []

  board = game.board()
  result = game.headers.get("Result", "*")

  sequences = [[], []]
  sequences_labels = [white_idx, black_idx]

  positions = []
  positions_labels = []

  for move_num, move in enumerate(game.mainline_moves()):
    board_copy = board.copy()
    board_history.insert(0, board_copy)

    pos_hash = hash(board.fen().split(' ')[0])
    position_hashes.insert(0, pos_hash)

    rep_count = sum(1 for h in position_hashes if h == pos_hash) - 1
    repetition_counts.insert(0, rep_count)

    if len(board_history) > 8:
      board_history = board_history[:8]
      position_hashes = position_hashes[:8]
      repetition_counts = repetition_counts[:8]

    board_planes = board_history_to_lc0_planes(board_history, repetition_counts)

    if move_num != 0:
      if board.turn == chess.WHITE:
        #append to BLACK
        sequences[1].append(np.concatenate((board_planes[:13], board_planes[-8:])))
        positions.append((white_idx, black_idx, board_planes))
        positions_labels.append(
          [1, 0, 0] if result == "1-0" else [0, 0, 1] if result == "0-1" else [0, 1, 0]
        )
      else:
        #append to WHITE
        sequences[0].append(np.concatenate((board_planes[:13], board_planes[-8:])))
        positions.append((black_idx, white_idx, board_planes))
        positions_labels.append(
          [1, 0, 0] if result == "0-1" else [0, 0, 1] if result == "1-0" else [0, 1, 0]
        )

    board.push(move)

    if len(sequences[0]) >= max_moves and len(sequences[1]) >= max_moves:
      break

  return (list(zip(sequences, sequences_labels)), list(zip(positions, positions_labels)))

engine = chess.engine.SimpleEngine.popen_uci("./stylometry/WinRateStylo/engines/stockfish-17-1")
engine.configure({"UCI_ShowWDL": True})

DO_ENGINE_EVAL = False

def engine_eval(board):
  if not DO_ENGINE_EVAL:
    return [1, 0, 0]
  info = engine.analyse(board, chess.engine.Limit(depth=10))
  if "wdl" not in info:
    assert "score" in info
    return [1, 0, 0] if info["score"].white().score(mate_score=10000) > 100 else [0, 0, 1] if info["score"].white().score(mate_score=10000) < -100 else [0, 1, 0]
  score = info["wdl"].white()
  return score

def process_pgns(
  pgn_paths: List[str],
  output_prefix: str,
  player_mapper: PlayerIndexMapper,
  max_moves: int = 100,
  min_moves: int = 1,
  pos_skip_rate: float = 0.95
):
  curr_sequences = {}
  curr_positions = []

  curr_pos_shard_idx = 0
  POS_SHARD_SIZE = 10000/(1 - pos_skip_rate) if pos_skip_rate < 1.0 else 10000
  if not os.path.exists(output_prefix):
    os.mkdir(output_prefix)
  if not os.path.exists(f"{output_prefix}/pos_shards"):
    os.mkdir(f"{output_prefix}/pos_shards")

  for pgn_file_idx, pgn_file_path in enumerate(pgn_paths):
    logger.info(f"Processing {pgn_file_path}...")
    pgn_file = open(pgn_file_path, 'r')
    game_count = 0

    seq_counts = {}

    has_seq_pos_count = 0
    total_seq_pos_count = 0

    white_wdl_counts = [0, 0, 0]
    elo_wdl_counts = [0, 0, 0, 0]
    engine_wdl_counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    engine_wdl_counts_no_draws = [[0, 0, 0], [0, 0, 0]]
    elo_counts = {}
    for i in range(1000, 3000, 100):
      elo_counts[i] = 0
    while True:
      game = chess.pgn.read_game(pgn_file)
      if game is None:
        break
      if game.headers["White"] == "?" or game.headers["Black"] == "?":
        continue

      game_data = extract_game_data(
        game, player_mapper, max_moves
      )

      #make sure both players played at least 1 move
      if len(game_data[0][0][0]) < min_moves or len(game_data[0][1][0]) < min_moves:
        continue
      
      white = player_mapper.get_index(game.headers["White"])
      black = player_mapper.get_index(game.headers["Black"])
      rand = random.random()
      if (seq_counts.get(white, 0) < 50 or seq_counts.get(black, 0) < 50) and rand < 0.4:
        if white in curr_sequences:
          curr_sequences[white].append(game_data[0][0])
        else:
          curr_sequences[white] = [game_data[0][0]]
        if black in curr_sequences:
          curr_sequences[black].append(game_data[0][1])
        else:
          curr_sequences[black] = [game_data[0][1]]
        seq_counts[white] = seq_counts.get(white, 0) + 1
        seq_counts[black] = seq_counts.get(black, 0) + 1
      else:
        curr_positions.extend(game_data[1])

      if len(curr_positions) > POS_SHARD_SIZE:
        curr_positions = random.sample(curr_positions, k=int(len(curr_positions)*(1.0 - pos_skip_rate)) if pos_skip_rate < 1.0 else 0)
        def to_paired(pos):
          nonlocal has_seq_pos_count, total_seq_pos_count
          if pos[0][0] in curr_sequences:
            has_seq_pos_count += 1
          if pos[0][1] in curr_sequences:
            has_seq_pos_count += 1
          total_seq_pos_count += 2
          return (random.choice(curr_sequences[pos[0][0]])[0] if pos[0][0] in curr_sequences else [], random.choice(curr_sequences[pos[0][1]])[0] if pos[0][1] in curr_sequences else [], pos)
        curr_paired_positions = map(to_paired, curr_positions)
        save_shard(f"{output_prefix}/pos_shards/{curr_pos_shard_idx:04d}.tfrecord", curr_paired_positions, serialize_position)
        curr_pos_shard_idx += 1
        curr_positions = []
        curr_sequences = {}
        seq_counts = {}

      result = game.headers.get("Result", "*")

      white_wdl_counts[0] += len(game_data[1]) if result == "1-0" else 0
      white_wdl_counts[1] += len(game_data[1]) if result == "1/2-1/2" else 0
      white_wdl_counts[2] += len(game_data[1]) if result == "0-1" else 0

      try:
        white_elo = int(game.headers.get("WhiteElo", "0"))
        black_elo = int(game.headers.get("BlackElo", "0"))
        elo_diff = white_elo - black_elo
        result_to_val = {"1-0": 1, "1/2-1/2": 0.5, "0-1": 0}
        adjusted_result = 1-result_to_val[result] if elo_diff < 0 else result_to_val[result]
        elo_wdl_counts[0] += len(game_data[1]) if adjusted_result == 1 else 0
        elo_wdl_counts[1] += len(game_data[1]) if adjusted_result == 0.5 else 0
        elo_wdl_counts[2] += len(game_data[1]) if adjusted_result == 0 else 0
      except:
        elo_wdl_counts[3] += len(game_data[1])

      if DO_ENGINE_EVAL:
        for pos in game.mainline():
          board = pos.board()
          eval_wdl = engine_eval(board)
          prediction = max(enumerate(eval_wdl), key=lambda x: x[1])[0]

          result_to_val = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}

          engine_wdl_counts[prediction][result_to_val[result]] += 1
          
          if isinstance(eval_wdl, chess.engine.Wdl):
            engine_wdl_counts_no_draws[0 if eval_wdl.expectation() > 0 else 1][result_to_val[result]] += 1
          else:
            engine_wdl_counts_no_draws[0 if eval_wdl[0] > eval_wdl[2] else 1][result_to_val[result]] += 1
      
      white_elo = int(game.headers.get("WhiteElo", "0"))
      black_elo = int(game.headers.get("BlackElo", "0"))
      white_elo_bucket = (white_elo // 100) * 100
      black_elo_bucket = (black_elo // 100) * 100
      if white_elo_bucket in elo_counts:
        elo_counts[white_elo_bucket] += len(game_data[1])
      if black_elo_bucket in elo_counts:
        elo_counts[black_elo_bucket] += len(game_data[1])

      game_count += 1
      if game_count % 20 == 0:
        logger.info(f"Processed: {game_count} games in PGN")
        logger.info(f"In Memory: {len(curr_sequences)} sequences, {len(curr_positions)} positions")
        logger.info(f"WDL dist (from white POV) (not accounting for skipping): {white_wdl_counts}, As Pct: {[f'{100*count/sum(white_wdl_counts):.2f}' for count in white_wdl_counts]}")
        logger.info(f"Higher elo WDL dist (not accounting for skipping): {elo_wdl_counts}, As Pct: {[f'{100*count/sum(elo_wdl_counts):.2f}' for count in elo_wdl_counts]}")
        logger.info(f"Engine predicted WDL dist matrix (not accounting for skipping, [engine_pred][actual]): {engine_wdl_counts}, " +
                    (f"As Pct: {[f'{100*sum(engine_wdl_counts[i])/sum(sum(engine_wdl_counts, [])):.2f}' for i in range(3)]}, " +
                    f"Accuracy: {100*(engine_wdl_counts[0][0] + engine_wdl_counts[1][1] + engine_wdl_counts[2][2])/sum(sum(engine_wdl_counts, [])):.2f}%, " +
                    f"Accuracy w/o draws: {100*(engine_wdl_counts_no_draws[0][0] + engine_wdl_counts_no_draws[1][2])/(sum(engine_wdl_counts_no_draws[0]) + sum(engine_wdl_counts_no_draws[1])):.2f}%")
                    if DO_ENGINE_EVAL else "")
        logger.info(f"Elo counts (not accounting for skipping): {elo_counts}")
        logger.info(f"Player count: {player_mapper.num_players()}")
        logger.info(f"Has sequences in positions: {has_seq_pos_count}/{total_seq_pos_count} ({(has_seq_pos_count/total_seq_pos_count)*100 if total_seq_pos_count > 0 else 0:.2f}%)")
      player_mapper.save(f"{args.output_prefix}/player_map_{game_count % 2}.txt")
    
    logger.info(f"Finished: {pgn_file_path}")
    logger.info(f"File Idx: {pgn_file_idx}")
    logger.info(f"Processed: {game_count} games in PGN")
    logger.info(f"In Memory: {len(curr_sequences)} sequences, {len(curr_positions)} positions")

def save_shard(shard_path, items, serialize_function):
  with tf.io.TFRecordWriter(shard_path) as writer:
    for item in items:
      example = serialize_function(item)
      writer.write(example)
  logger.info(f"Saved shard to {shard_path}")

def serialize_position(paired_position):
  feature = {
    'stm_player_seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(paired_position[0], dtype=np.int8).tobytes()])),
    'opp_player_seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(paired_position[1], dtype=np.int8).tobytes()])),
    'full_board_planes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(paired_position[2][0][2], dtype=np.int8).tobytes()])),
    'wdl': tf.train.Feature(float_list=tf.train.FloatList(value=paired_position[2][1]))
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

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
  parser.add_argument("output_prefix", help="Output file prefix")
  parser.add_argument("--max-moves", type=int, default=100,
                     help="Maximum moves per sequence (default: 100)")

  args = parser.parse_args()

  pgn_files = get_pgns(args.inputs)

  player_mapper = PlayerIndexMapper()

  process_pgns(pgn_files, args.output_prefix, player_mapper, max_moves=args.max_moves)

  player_mapper.save(f"{args.output_prefix}/player_map.txt")
  logger.info(f"Player mapping path: {args.output_prefix}/player_map.txt")
  logger.info(f"Total players: {player_mapper.num_players()}")

#  python3 stylometry/WinRateStylo/pgn_to_training_data.py ../../maia-individual/data-lichess/chess/unzipped stylometry/WinRateStylo/data/run2025-12-22