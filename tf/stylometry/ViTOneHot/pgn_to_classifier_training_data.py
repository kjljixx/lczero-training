import chess
import chess.engine
import chess.pgn
import chess.polyglot
import numpy as np
import random
import logging
import argparse
import glob
import os
import io
from collections import deque
import concurrent.futures
import multiprocessing
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

MIN_STARTING_TIME = 180

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

def get_pgns(pgn_paths: List[str]):
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

def _flip_vertical(bb: int) -> int:
  bb = ((bb >> 8) & 0x00FF00FF00FF00FF) | ((bb & 0x00FF00FF00FF00FF) << 8)
  bb = ((bb >> 16) & 0x0000FFFF0000FFFF) | ((bb & 0x0000FFFF0000FFFF) << 16)
  bb = (bb >> 32) | ((bb & 0xFFFFFFFF) << 32)
  return bb

def board_to_struct_short_player_perspective(board: chess.Board, clock: int, rep_count: int, player_color: chess.Color) -> np.ndarray:
  """
  Optimized version of board_to_chessboard_struct_player_perspective with short=True.
  Bypasses Python object allocations by querying board state and bitboards directly.
  """
  is_black = (player_color == chess.BLACK)
  occupied = int(board.occupied)
  occ = _flip_vertical(occupied) if is_black else occupied

  pcs_1 = 0
  pcs_2 = 0
  
  player_occupied = int(board.occupied_co[player_color])
  
  for i, sq in enumerate(chess.scan_forward(occ)):
    orig_sq = sq ^ 56 if is_black else sq
    
    pt = board.piece_type_at(orig_sq)
    val = pt - 1
    
    if not (player_occupied & (1 << orig_sq)):
      val |= 8
    
    if i < 16:
      pcs_1 |= (val << (4 * i))
    else:
      pcs_2 |= (val << (4 * (i - 16)))

  # Castling and STM metadata
  castle_and_stm = 0
  if board.has_queenside_castling_rights(player_color):
    castle_and_stm |= 2
  if board.has_kingside_castling_rights(player_color):
    castle_and_stm |= 4
  if board.has_queenside_castling_rights(not player_color):
    castle_and_stm |= 8
  if board.has_kingside_castling_rights(not player_color):
    castle_and_stm |= 16

  if board.turn != player_color:
    castle_and_stm |= 1

  mtd = (castle_and_stm << 24) | ((board.halfmove_clock & 0xFF) << 16) | min(rep_count, 3)

  return np.array([occ, pcs_1, pcs_2, clock, mtd], dtype=np.uint64)

def extract_game_data_optimized(game: chess.pgn.Game, max_moves: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """
  In-place board updates, avoiding board.copy() calls and redundant parsing.
  """
  white_clock = 600
  black_clock = 600
  position_hashes = deque(maxlen=8)

  board = game.board()
  sequences = [[], []]

  for node in game.mainline():
    moving_color = board.turn
    
    # 1. Before step
    before_hash = chess.polyglot.zobrist_hash(board)
    before_rep_count = position_hashes.count(before_hash)
    
    curr_clock = node.clock()
    if moving_color == chess.BLACK and curr_clock is not None:
      black_clock = int(curr_clock)
    elif moving_color == chess.WHITE and curr_clock is not None:
      white_clock = int(curr_clock)

    before_struct = board_to_struct_short_player_perspective(
      board,
      (white_clock << 32) | black_clock,
      before_rep_count,
      player_color=moving_color,
    )

    # 2. Update board state
    board.push(node.move)

    # 3. After step
    after_hash = chess.polyglot.zobrist_hash(board)
    after_rep_count = position_hashes.count(after_hash)

    after_struct = board_to_struct_short_player_perspective(
      board,
      (white_clock << 32) | black_clock,
      after_rep_count,
      player_color=moving_color,
    )

    target_seq = sequences[0] if moving_color == chess.WHITE else sequences[1]

    if len(target_seq) + 2 <= max_moves:
      target_seq.append(before_struct)
      target_seq.append(after_struct)

    position_hashes.appendleft(before_hash)
    position_hashes.appendleft(after_hash)

    if len(sequences[0]) >= max_moves and len(sequences[1]) >= max_moves:
      break

  if len(sequences[0]) % 2 == 1:
    sequences[0] = sequences[0][:-1]
  if len(sequences[1]) % 2 == 1:
    sequences[1] = sequences[1][:-1]

  return sequences[0], sequences[1]

def pgn_game_stripper(file_path: str):
  """
  Extremely fast file-system generator that yields raw game text segments from the PGN file.
  """
  with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    current_game = []
    for line in f:
      cleaned = line.lstrip()
      if cleaned.startswith('[Event '):
        if current_game:
          yield ''.join(current_game)
          current_game = []
      current_game.append(line)
    if current_game:
      yield ''.join(current_game)

def process_game_batch(game_texts: List[str], max_moves: int, min_moves: int) -> Tuple[List, set, int]:
  """
  Target worker method to run inside separate CPU processes. 
  Constructs representations and tracks player names. Returns parsed valid results,
  the set of player names, and the original number of attempted games in the batch.
  """
  results = []
  player_names = set()

  for game_text in game_texts:
    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None:
      continue
    
    white_name = game.headers.get("White", "?")
    black_name = game.headers.get("Black", "?")
    if white_name == "?" or black_name == "?":
      continue
    
    time_control = game.headers.get("TimeControl", "600+0")
    if time_control == "-":
      continue
    try:
      starting_time = int(time_control.split('+')[0])
      if starting_time < MIN_STARTING_TIME:
        continue
    except ValueError:
      continue

    # Extract sequences
    white_seq, black_seq = extract_game_data_optimized(game, max_moves)
    if not white_seq or not black_seq:
      continue

    if len(white_seq) < min_moves * 2 or len(black_seq) < min_moves * 2:
      continue

    result = game.headers.get("Result", "*")
    white_elo = 0
    black_elo = 0
    try:
      white_elo = int(game.headers.get("WhiteElo", "0"))
    except ValueError:
      pass
    try:
      black_elo = int(game.headers.get("BlackElo", "0"))
    except ValueError:
      pass

    wdl = [1.0, 0.0, 0.0] if result == "1-0" else [0.0, 0.0, 1.0] if result == "0-1" else [0.0, 1.0, 0.0]

    results.append((
      [white_seq],
      [black_seq],
      wdl,
      white_name,
      black_name,
      white_elo,
      black_elo
    ))
    player_names.add(white_name)
    player_names.add(black_name)

  return results, player_names, len(game_texts)

def process_pgns(
  pgn_paths: List[str],
  output_prefix: str,
  player_mapper: PlayerIndexMapper,
  max_moves: int = 100,
  min_moves: int = 1
):
  SHARD_SIZE = 100000
  curr_results = []
  curr_pos_shard_idx = 0

  # Counter variables to track progress in the main thread
  games_attempted = 0
  games_valid = 0

  if not os.path.exists(output_prefix):
    os.mkdir(output_prefix)
  if not os.path.exists(f"{output_prefix}/seq_shards"):
    os.mkdir(f"{output_prefix}/seq_shards")

  num_workers = 4
  logger.info(f"Using ProcessPoolExecutor with {num_workers} workers.")

  BATCH_SIZE = 2000  # Number of games to chunk and submit to a worker
  futures = []

  with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for pgn_file_path in pgn_paths:
      logger.info(f"Streaming games from {pgn_file_path}...")
      
      game_text_batch = []
      for game_text in pgn_game_stripper(pgn_file_path):
        game_text_batch.append(game_text)
        
        if len(game_text_batch) >= BATCH_SIZE:
          futures.append(executor.submit(process_game_batch, game_text_batch, max_moves, min_moves))
          game_text_batch = []
          
          # Throttle queue submission to keep memory consumption in check
          if len(futures) >= num_workers * 2:
            done, not_done = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
              results, player_names, batch_attempted_count = f.result()
              for name in player_names:
                player_mapper.get_or_create_index(name)
              
              prev_valid = games_valid
              games_valid += len(results)
              games_attempted += batch_attempted_count
              curr_results.extend(results)
              
              # Logging threshold triggers every 2,000 valid games processed
              if (games_valid // 2000) > (prev_valid // 2000):
                logger.info(
                  f"Progress: Converted {games_valid} valid games (from {games_attempted} attempted). "
                  f"Shard buffer size: {len(curr_results)}. Player count: {player_mapper.num_players()}."
                )
                # Periodically save the player map
                player_mapper.save(f"{output_prefix}/player_map_checkpoint.txt")
              
              while len(curr_results) >= SHARD_SIZE:
                shard_items = curr_results[:SHARD_SIZE]
                save_shard(f"{output_prefix}/seq_shards/{curr_pos_shard_idx:04d}.tfrecord", shard_items, serialize_position)
                curr_pos_shard_idx += 1
                curr_results = curr_results[SHARD_SIZE:]
            futures = list(not_done)
            
      if game_text_batch:
        futures.append(executor.submit(process_game_batch, game_text_batch, max_moves, min_moves))

    # Resolve pending tasks
    logger.info("Finishing remaining background tasks...")
    for f in concurrent.futures.as_completed(futures):
      results, player_names, batch_attempted_count = f.result()
      for name in player_names:
        player_mapper.get_or_create_index(name)
      
      prev_valid = games_valid
      games_valid += len(results)
      games_attempted += batch_attempted_count
      curr_results.extend(results)
      
      if (games_valid // 2000) > (prev_valid // 2000):
        logger.info(
          f"Progress: Converted {games_valid} valid games (from {games_attempted} attempted). "
          f"Shard buffer size: {len(curr_results)}. Player count: {player_mapper.num_players()}."
        )
        player_mapper.save(f"{output_prefix}/player_map_checkpoint.txt")
      
      while len(curr_results) >= SHARD_SIZE:
        shard_items = curr_results[:SHARD_SIZE]
        save_shard(f"{output_prefix}/seq_shards/{curr_pos_shard_idx:04d}.tfrecord", shard_items, serialize_position)
        curr_pos_shard_idx += 1
        curr_results = curr_results[SHARD_SIZE:]

  # Flush remainder
  if curr_results:
    save_shard(f"{output_prefix}/seq_shards/{curr_pos_shard_idx:04d}.tfrecord", curr_results, serialize_position)

def save_shard(shard_path, items, serialize_function):
  import tensorflow as tf  # Lazy load to avoid loading overhead in worker processes
  options = tf.io.TFRecordOptions(compression_type='GZIP')
  with tf.io.TFRecordWriter(shard_path, options=options) as writer:
    for item in items:
      example = serialize_function(item)
      writer.write(example)
  logger.info(f"Saved shard to {shard_path}")

def serialize_position(paired_position):
  import tensorflow as tf  # Lazy load inside serialization step
  def pad_and_flatten(games, max_games=1, max_moves=100):
    all_game_moves = np.zeros((max_games, max_moves, 5), dtype=np.uint64)
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
    'stm_player_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[paired_position[3].encode('utf-8')])),
    'stm_player_elo': tf.train.Feature(int64_list=tf.train.Int64List(value=[paired_position[5]])),
    'opp_player_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[paired_position[4].encode('utf-8')])),
    'opp_player_elo': tf.train.Feature(int64_list=tf.train.Int64List(value=[paired_position[6]])),
    'wdl': tf.train.Feature(float_list=tf.train.FloatList(value=paired_position[2]))
  }

  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

if __name__ == "__main__":
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