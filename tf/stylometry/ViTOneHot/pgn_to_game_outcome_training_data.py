import chess
import chess.engine
import chess.pgn
import chess.polyglot
import numpy as np
import random
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
import argparse
import glob
import os
import io
import concurrent.futures

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

def _flip_vertical(bb):
  bb = ((bb >> 8) & 0x00FF00FF00FF00FF) | ((bb & 0x00FF00FF00FF00FF) << 8)
  bb = ((bb >> 16) & 0x0000FFFF0000FFFF) | ((bb & 0x0000FFFF0000FFFF) << 16)
  bb = (bb >> 32) | ((bb & 0xFFFFFFFF) << 32)
  return bb

def board_to_chessboard_struct_player_perspective(
  board_history,
  clock_history,
  repetition_counts,
  player_color,
  short=False
):
  structs = []
  history_len = len(board_history)
  max_len = 1 if short else 8
  if history_len > max_len:
    board_history = board_history[:max_len]
    clock_history = clock_history[:max_len]
    repetition_counts = repetition_counts[:max_len]
    history_len = max_len

  for hist_idx in range(history_len):
    board = board_history[hist_idx]
    perspective = player_color
    pcs_1 = 0
    pcs_2 = 0
    is_black_perspective = perspective == chess.BLACK

    occupied = board.occupied
    if is_black_perspective:
      occupied = _flip_vertical(occupied)
    occ = occupied

    stm_mask = board.occupied_co[perspective]
    opp_mask = board.occupied_co[not perspective]

    piece_bbs = [board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings]

    piece_vals = [0] * 64
    for pt_idx, pbb in enumerate(piece_bbs):
      bb = pbb & stm_mask
      if is_black_perspective:
        bb = _flip_vertical(bb)
      while bb:
        sq = (bb & -bb).bit_length() - 1
        piece_vals[sq] = pt_idx
        bb &= bb - 1

      bb = pbb & opp_mask
      if is_black_perspective:
        bb = _flip_vertical(bb)
      while bb:
        sq = (bb & -bb).bit_length() - 1
        piece_vals[sq] = pt_idx | 8
        bb &= bb - 1

    pcs_1_idx = 0
    pcs_2_idx = 0
    temp_occ = occ
    while temp_occ:
      sq = (temp_occ & -temp_occ).bit_length() - 1
      val = piece_vals[sq]
      if pcs_1_idx < 16:
        pcs_1 |= (val << (4 * pcs_1_idx))
        pcs_1_idx += 1
      else:
        pcs_2 |= (val << (4 * pcs_2_idx))
        pcs_2_idx += 1
      temp_occ &= temp_occ - 1
    structs.extend([occ, pcs_1, pcs_2, clock_history[hist_idx]])
  for _ in range(max_len - history_len):
    structs.extend([0, 0, 0, 0])

  board = board_history[0]
  mtd = 0
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

  mtd |= castle_and_stm << 24
  mtd |= (board.halfmove_clock & 0xFF) << 16

  for idx, rep_count in enumerate(repetition_counts):
    mtd |= min(rep_count, 3) << (2 * idx)
  structs.append(mtd)

  return np.array(structs, dtype=np.uint64)

def extract_game_data(
  game: chess.pgn.Game,
  max_moves: int = 100
):
  white_name = game.headers.get("White", "Unknown_White")
  black_name = game.headers.get("Black", "Unknown_Black")

  white_clock = 600
  black_clock = 600
  position_hashes = []

  board = game.board()
  result = game.headers.get("Result", "*")

  sequences = [[], []]
  sequences_labels = [white_name, black_name]

  positions = []
  positions_labels = []

  for _, node in enumerate(game.mainline()):
    moving_color = board.turn
    before_board = board.copy(stack=False)

    before_hash = chess.polyglot.zobrist_hash(before_board)
    before_rep_count = sum(1 for h in position_hashes if h == before_hash)

    curr_clock = node.clock()
    if moving_color == chess.BLACK and curr_clock is not None:
      black_clock = int(curr_clock)
    elif moving_color == chess.WHITE and curr_clock is not None:
      white_clock = int(curr_clock)

    board.push(node.move)
    after_board = board.copy(stack=False)
    after_hash = chess.polyglot.zobrist_hash(after_board)
    after_rep_count = sum(1 for h in position_hashes if h == after_hash)

    if moving_color == chess.WHITE:
      target_seq = sequences[0]
      player_idx = white_name
      opp_idx = black_name
    else:
      target_seq = sequences[1]
      player_idx = black_name
      opp_idx = white_name

    if len(target_seq) + 2 <= max_moves:
      before_struct = board_to_chessboard_struct_player_perspective(
        [before_board],
        [(white_clock << 32) | black_clock],
        [before_rep_count],
        player_color=moving_color,
        short=True,
      )
      after_struct = board_to_chessboard_struct_player_perspective(
        [after_board],
        [(white_clock << 32) | black_clock],
        [after_rep_count],
        player_color=moving_color,
        short=True,
      )

      target_seq.append(np.concatenate((before_struct[:4], before_struct[-1:])))
      target_seq.append(np.concatenate((after_struct[:4], after_struct[-1:])))

      positions.append((player_idx, opp_idx, before_struct))
      positions_labels.append(
        [0, 0, 1] if result == "1-0" else [1, 0, 0] if result == "0-1" else [0, 1, 0]
      )
      positions.append((player_idx, opp_idx, after_struct))
      positions_labels.append(
        [0, 0, 1] if result == "1-0" else [1, 0, 0] if result == "0-1" else [0, 1, 0]
      )

    position_hashes.insert(0, before_hash)
    position_hashes.insert(0, after_hash)
    if len(position_hashes) > 8:
      position_hashes = position_hashes[:8]

    if len(sequences[0]) >= max_moves and len(sequences[1]) >= max_moves:
      break

  if len(sequences[0]) % 2 == 1:
    sequences[0] = sequences[0][:-1]
  if len(sequences[1]) % 2 == 1:
    sequences[1] = sequences[1][:-1]

  seq_0_arr = np.array(sequences[0], dtype=np.uint64) if sequences[0] else np.zeros((0, 5), dtype=np.uint64)
  seq_1_arr = np.array(sequences[1], dtype=np.uint64) if sequences[1] else np.zeros((0, 5), dtype=np.uint64)

  return ([(seq_0_arr, white_name), (seq_1_arr, black_name)], list(zip(positions, positions_labels)))

def yield_game_strings(pgn_file_path: str):
  with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as f:
    current_game = []
    for line in f:
      if line.startswith("[Event "):
        if current_game:
          yield "".join(current_game)
          current_game = []
      current_game.append(line)
    if current_game:
      yield "".join(current_game)

def worker_process_game(game_text: str, max_moves: int, min_moves: int) -> Optional[Tuple[dict, tuple]]:
  game = chess.pgn.read_game(io.StringIO(game_text))
  if game is None:
    return None

  white_name = game.headers.get("White", "?")
  black_name = game.headers.get("Black", "?")
  if white_name == "?" or black_name == "?":
    return None

  time_control = game.headers.get("TimeControl", "600+0")
  if time_control == "-":
    return None
  try:
    starting_time = int(time_control.split('+')[0])
    if starting_time < MIN_STARTING_TIME:
      return None
  except (ValueError, IndexError):
    return None

  mainline_len = 0
  for _ in game.mainline():
    mainline_len += 1
    if mainline_len >= min_moves * 2:
      break
  if mainline_len < min_moves * 2:
    return None

  game_data = extract_game_data(game, max_moves)
  return (dict(game.headers), game_data)

def write_shard_background(output_prefix: str, shard_idx: int, results_snapshot: list, sequences_snapshot: dict):
  """Executes inside a background thread to prevent blocking game processing."""
  logger.info(f"Background worker starting write task for shard {shard_idx:04d}...")
  
  def to_paired(pos):
    num_in_0 = 0
    num_in_1 = 0
    NUM_GAMES = 20
    white_id = pos[0]
    black_id = pos[1]

    if white_id in sequences_snapshot:
      num_in_0 = min(NUM_GAMES, len(sequences_snapshot[white_id]))
    if black_id in sequences_snapshot:
      num_in_1 = min(NUM_GAMES, len(sequences_snapshot[black_id]))

    # Extract single 2D NumPy array matrices (massively lighter payload)
    white_sampled = [x[0] for x in random.sample(sequences_snapshot[white_id], num_in_0)] if white_id in sequences_snapshot else []
    black_sampled = [x[0] for x in random.sample(sequences_snapshot[black_id], num_in_1)] if black_id in sequences_snapshot else []

    return (
      white_sampled,
      black_sampled,
      pos[2], pos[3], pos[4], pos[5], pos[6]
    )

  curr_paired_positions = map(to_paired, results_snapshot)
  save_shard(f"{output_prefix}/seq_shards/{shard_idx:04d}.tfrecord", curr_paired_positions, serialize_position)
  logger.info(f"Background worker successfully finished writing shard {shard_idx:04d}.")

def process_pgns(
  pgn_paths: List[str],
  output_prefix: str,
  player_mapper: PlayerIndexMapper,
  max_moves: int = 100,
  min_moves: int = 1,
  num_workers: Optional[int] = None,
  reset_sequences_every_n_shards: int = 40
):
  SHARD_SIZE = 10000  # Highly optimized standard size
  curr_sequences = {}
  curr_results = []
  curr_pos_shard_idx = 0
  shards_written_since_reset = 0

  if not os.path.exists(output_prefix):
    os.mkdir(output_prefix)
  if not os.path.exists(f"{output_prefix}/seq_shards"):
    os.mkdir(f"{output_prefix}/seq_shards")

  seq_counts = {}
  game_count = 0

  # Background I/O thread pool
  io_writer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

  if num_workers is None:
    num_workers = os.cpu_count() or 4
  logger.info(f"Using {num_workers} processes for parallel games parsing.")

  with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    max_queue_size = num_workers * 4
    futures = {}

    for pgn_file_idx, pgn_file_path in enumerate(pgn_paths):
      logger.info(f"Processing {pgn_file_path}...")
      game_str_gen = yield_game_strings(pgn_file_path)
      has_more_games = True

      while has_more_games or futures:
        while has_more_games and len(futures) < max_queue_size:
          try:
            game_text = next(game_str_gen)
            future = executor.submit(worker_process_game, game_text, max_moves, min_moves)
            futures[future] = game_text
          except StopIteration:
            has_more_games = False

        if not futures:
          break

        done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
          result_data = future.result()
          del futures[future]

          if result_data is None:
            continue

          headers, game_data = result_data
          white_name = headers["White"]
          black_name = headers["Black"]
          result = headers.get("Result", "*")

          white = player_mapper.get_or_create_index(white_name)
          black = player_mapper.get_or_create_index(black_name)

          NUM_GAMES = 20
          seq_white = (game_data[0][0][0], white)
          seq_black = (game_data[0][1][0], black)

          if seq_counts.get(white, 0) < NUM_GAMES:
            if white in curr_sequences:
              curr_sequences[white].append(seq_white)
            else:
              curr_sequences[white] = [seq_white]
            seq_counts[white] = seq_counts.get(white, 0) + 1

          if seq_counts.get(black, 0) < NUM_GAMES:
            if black in curr_sequences:
              curr_sequences[black].append(seq_black)
            else:
              curr_sequences[black] = [seq_black]
            seq_counts[black] = seq_counts.get(black, 0) + 1

          if seq_counts.get(white, 0) >= NUM_GAMES and seq_counts.get(black, 0) >= NUM_GAMES:
            curr_results.append((
              white, black,
              [1, 0, 0] if result == "1-0" else [0, 0, 1] if result == "0-1" else [0, 1, 0],
              white_name, black_name,
              int(headers.get("WhiteElo", "0") or "0"),
              int(headers.get("BlackElo", "0") or "0")
            ))

          game_count += 1
          if game_count % 1000 == 0:
            logger.info(f"Processed: {game_count} games in PGN")
            logger.info(f"In Memory: {len(curr_sequences)} sequences, {len(curr_results)} positions")
            logger.info(f"Player count: {player_mapper.num_players()}")
            player_mapper.save(f"{output_prefix}/player_map_{game_count % 2}.txt")

          if len(curr_results) >= SHARD_SIZE:
            # Create shallow copies of structures so the main thread can instantly reset them
            results_snapshot = list(curr_results)
            sequences_snapshot = {k: list(v) for k, v in curr_sequences.items()}

            logger.info(f"Submitting shard {curr_pos_shard_idx:04d} to background writer thread.")
            io_writer_pool.submit(
              write_shard_background,
              output_prefix,
              curr_pos_shard_idx,
              results_snapshot,
              sequences_snapshot
            )

            curr_pos_shard_idx += 1
            shards_written_since_reset += 1

            # Reset logic to prevent memory bloat
            if shards_written_since_reset >= reset_sequences_every_n_shards:
              logger.info(f"Resetting sequence buffers after {shards_written_since_reset} shards to clear memory bloat.")
              curr_sequences = {}
              seq_counts = {}
              shards_written_since_reset = 0

            curr_results = []

      logger.info(f"Finished file: {pgn_file_path}")

  # Write any final remaining records
  if curr_results:
    results_snapshot = list(curr_results)
    sequences_snapshot = {k: list(v) for k, v in curr_sequences.items()}
    logger.info(f"Submitting final remaining shard {curr_pos_shard_idx:04d} to background writer thread.")
    io_writer_pool.submit(
      write_shard_background,
      output_prefix,
      curr_pos_shard_idx,
      results_snapshot,
      sequences_snapshot
    )

  # Shutdown writer and wait for all tasks to complete cleanly
  logger.info("Main pipeline complete. Waiting for background writes to finish...")
  io_writer_pool.shutdown(wait=True)
  logger.info("All shards successfully written.")

def save_shard(shard_path, items, serialize_function):
  options = tf.io.TFRecordOptions(compression_type='GZIP')
  with tf.io.TFRecordWriter(shard_path, options=options) as writer:
    for item in items:
      example = serialize_function(item)
      writer.write(example)
  logger.info(f"Saved shard to {shard_path}")

def serialize_position(paired_position):
  # OPTIMIZATION: pad_and_flatten now copies 2D NumPy slices directly (extremely fast)
  def pad_and_flatten(games, max_games=20, max_moves=100):
    all_game_moves = np.zeros((max_games, max_moves, 5), dtype=np.uint64)
    for i, game in enumerate(games[:max_games]):
      if game is None or game.shape[0] == 0:
        continue
      num_moves = min(game.shape[0], max_moves)
      if num_moves > 0:
        all_game_moves[i, :num_moves, :] = game[:num_moves, :]
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

  parser = argparse.ArgumentParser(description="Convert PGN files to training data")
  parser.add_argument("inputs", nargs="+", help="Input PGN file(s) or folder(s) containing PGN files")
  parser.add_argument("output_prefix", help="Output file prefix")
  parser.add_argument("--max-moves", type=int, default=100, help="Maximum moves per sequence (default: 100)")
  parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes")
  parser.add_argument("--reset-interval", type=int, default=40, help="Reset sequences dictionary every N shards")

  args = parser.parse_args()

  pgn_files = get_pgns(args.inputs)
  player_mapper = PlayerIndexMapper()

  process_pgns(
    pgn_files, 
    args.output_prefix, 
    player_mapper, 
    max_moves=args.max_moves, 
    num_workers=args.num_workers,
    reset_sequences_every_n_shards=args.reset_interval
  )

  player_mapper.save(f"{args.output_prefix}/player_map.txt")
  logger.info(f"Player mapping path: {args.output_prefix}/player_map.txt")
  logger.info(f"Total players: {player_mapper.num_players()}")