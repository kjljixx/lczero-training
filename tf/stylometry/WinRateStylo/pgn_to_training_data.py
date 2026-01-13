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

def board_to_chessboard_struct(board_history, repetition_counts, short=False):
  structs = []

  history_len = len(board_history)
  max_len = 1 if short else 8
  if history_len > max_len:
    board_history = board_history[:max_len]
    repetition_counts = repetition_counts[:max_len]
    history_len = max_len

  for hist_idx in range(history_len):
    board = board_history[hist_idx]
    stm = board.turn
    occ = 0
    pcs_1 = 0
    pcs_2 = 0
    
    sq_map = lambda s: s if stm == chess.WHITE else s ^ 56
    
    occupied = board.occupied
    if stm == chess.BLACK:
      occupied = chess.flip_vertical(occupied)
    
    occ = occupied
    
    pcs_1_idx = 0
    pcs_2_idx = 0
    for sq in range(64):
      if not (occ & (1 << sq)):
        continue
      
      orig_sq = sq_map(sq)
      piece = board.piece_at(orig_sq)
      
      pc_type = piece.piece_type - 1
      pc_color = 0 if piece.color == stm else 8
      val = pc_color | pc_type
      
      if pcs_1_idx < 16:
        pcs_1 |= (val << (4 * (pcs_1_idx)))
        pcs_1_idx += 1
      else:
        pcs_2 |= (val << (4 * (pcs_2_idx)))
        pcs_2_idx += 1
    structs.extend([occ, pcs_1, pcs_2])
  for _ in range(max_len - history_len):
    structs.extend([0, 0, 0])

  #metadata: (5 bits for castling + STM color + 3 bit padding) + (8 bit hm clock) + (8 positions * 2 bits each for repetition count = 16 bits)
  board = board_history[0]
  stm = board.turn
  mtd = 0
  castle_and_stm = 0
  if board.has_queenside_castling_rights(stm):
    castle_and_stm |= 2
  if board.has_kingside_castling_rights(stm):
    castle_and_stm |= 4
  if board.has_queenside_castling_rights(not stm):
    castle_and_stm |= 8
  if board.has_kingside_castling_rights(not stm):
    castle_and_stm |= 16
  
  if stm == chess.BLACK:
    castle_and_stm |= 1

  mtd |= castle_and_stm << 24

  mtd |= (board.halfmove_clock & 0xFF) << 16

  for idx, rep_count in enumerate(repetition_counts):
    mtd |= min(rep_count, 3) << (2 * idx)
  structs.append(mtd)

  return np.array(structs, dtype=np.uint64)

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

    board_planes = board_to_chessboard_struct(board_history, repetition_counts)

    if move_num != 0:
      if board.turn == chess.WHITE:
        #append to BLACK
        sequences[1].append(np.concatenate((board_planes[:3], board_planes[-1:])))
        positions.append((white_idx, black_idx, board_planes))
        positions_labels.append(
          [1, 0, 0] if result == "1-0" else [0, 0, 1] if result == "0-1" else [0, 1, 0]
        )
      else:
        #append to WHITE
        sequences[0].append(np.concatenate((board_planes[:3], board_planes[-1:])))
        positions.append((black_idx, white_idx, board_planes))
        positions_labels.append(
          [1, 0, 0] if result == "0-1" else [0, 0, 1] if result == "1-0" else [0, 1, 0]
        )

    board.push(move)

    if len(sequences[0]) >= max_moves and len(sequences[1]) >= max_moves:
      break

  return (list(zip(sequences, sequences_labels)), list(zip(positions, positions_labels)))

DO_ENGINE_EVAL = False

if DO_ENGINE_EVAL:
  engine = chess.engine.SimpleEngine.popen_uci("./stylometry/WinRateStylo/engines/stockfish-17-1")
  engine.configure({"UCI_ShowWDL": True})

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
          num_in_0 = 0
          num_in_1 = 0
          if pos[0][0] in curr_sequences:
            num_in_0 = min(5, len(curr_sequences[pos[0][0]]))
            has_seq_pos_count += num_in_0
          if pos[0][1] in curr_sequences:
            num_in_1 = min(5, len(curr_sequences[pos[0][1]]))
            has_seq_pos_count += num_in_1
          total_seq_pos_count += 10
          return (list(map(lambda x: x[0], random.sample(curr_sequences[pos[0][0]], num_in_0))) if pos[0][0] in curr_sequences else [], list(map(lambda x: x[0], random.sample(curr_sequences[pos[0][1]], num_in_1))) if pos[0][1] in curr_sequences else [], pos)
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
  def pad_and_flatten(games, max_games=5, max_moves=100):
    all_game_moves = np.zeros((max_games, max_moves, 4), dtype=np.uint64)
    for i, game in enumerate(games[:max_games]):
      if not game:
        continue
      num_moves = min(len(game), max_moves)
      if num_moves > 0:
        all_game_moves[i, :num_moves, :] = np.array(game[:num_moves], dtype=np.uint64)
    return all_game_moves

  stm_padded = pad_and_flatten(paired_position[0])
  opp_padded = pad_and_flatten(paired_position[1])

  feature = {
    'stm_player_seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stm_padded.tobytes()])),
    'opp_player_seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[opp_padded.tobytes()])),
    'full_board_planes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[paired_position[2][0][2].tobytes()])),
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