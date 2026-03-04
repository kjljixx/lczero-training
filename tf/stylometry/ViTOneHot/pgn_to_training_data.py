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

def _flip_vertical(bb):
  bb = ((bb >> 8) & 0x00FF00FF00FF00FF) | ((bb & 0x00FF00FF00FF00FF) << 8)
  bb = ((bb >> 16) & 0x0000FFFF0000FFFF) | ((bb & 0x0000FFFF0000FFFF) << 16)
  bb = (bb >> 32) | ((bb & 0xFFFFFFFF) << 32)
  return bb

def board_to_chessboard_struct(board_history, clock_history, repetition_counts, short=False):
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
    stm = board.turn
    pcs_1 = 0
    pcs_2 = 0
    is_black = stm == chess.BLACK
    
    occupied = board.occupied
    if is_black:
      occupied = _flip_vertical(occupied)
    occ = occupied
    
    stm_mask = board.occupied_co[stm]
    opp_mask = board.occupied_co[not stm]
    
    piece_bbs = [board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings]
    
    piece_vals = [0] * 64
    for pt_idx, pbb in enumerate(piece_bbs):
      bb = pbb & stm_mask
      if is_black:
        bb = _flip_vertical(bb)
      while bb:
        sq = (bb & -bb).bit_length() - 1
        piece_vals[sq] = pt_idx
        bb &= bb - 1
      
      bb = pbb & opp_mask
      if is_black:
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
  clock_history = []
  white_clock = 600 #assume 10 minutes if not specified
  black_clock = 600
  position_hashes = []
  repetition_counts = []

  board = game.board()
  result = game.headers.get("Result", "*")

  sequences = [[], []]
  sequences_labels = [white_idx, black_idx]

  positions = []
  positions_labels = []

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

    if move_num >= 2:
      board_planes = board_to_chessboard_struct(board_history, clock_history, repetition_counts)
      if board.turn == chess.BLACK:
        #append to BLACK
        sequences[1].append(np.concatenate((board_planes[:4], board_planes[-1:])))
        positions.append((white_idx, black_idx, board_planes))
        positions_labels.append(
          [0, 0, 1] if result == "1-0" else [1, 0, 0] if result == "0-1" else [0, 1, 0]
        )
      else:
        #append to WHITE
        sequences[0].append(np.concatenate((board_planes[:4], board_planes[-1:])))
        positions.append((black_idx, white_idx, board_planes))
        positions_labels.append(
          [0, 0, 1] if result == "0-1" else [1, 0, 0] if result == "1-0" else [0, 1, 0]
        )

    curr_clock = node.clock()
    if board.turn == chess.BLACK and curr_clock is not None:
      black_clock = int(curr_clock)
    elif board.turn == chess.WHITE and curr_clock is not None:
      white_clock = int(curr_clock)
    if len(clock_history) == 0:
      clock_history.insert(0, (white_clock << 32) | black_clock)
    clock_history.insert(0, (white_clock << 32) | black_clock)

    board.push(node.move)

    if len(sequences[0]) >= max_moves and len(sequences[1]) >= max_moves:
      break

  return (list(zip(sequences, sequences_labels)), list(zip(positions, positions_labels)))

def process_pgns(
  pgn_paths: List[str],
  output_prefix: str,
  player_mapper: PlayerIndexMapper,
  max_moves: int = 100,
  min_moves: int = 1
):
  SHARD_SIZE = 40000
  curr_shard_idx = 0

  curr_sequences = []

  if not os.path.exists(output_prefix):
    os.mkdir(output_prefix)
  if not os.path.exists(f"{output_prefix}/seq_shards"):
    os.mkdir(f"{output_prefix}/seq_shards")

  for pgn_file_idx, pgn_file_path in enumerate(pgn_paths):
    logger.info(f"Processing {pgn_file_path}...")
    pgn_file = open(pgn_file_path, 'r')
    game_count = 0

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
      
      white_elo = int(game.headers.get("WhiteElo", "0"))
      black_elo = int(game.headers.get("BlackElo", "0"))
      curr_sequences.append([game_data[0][0], white_elo])
      curr_sequences.append([game_data[0][1], black_elo])

      if len(curr_sequences) > SHARD_SIZE:
        save_shard(f"{output_prefix}/seq_shards/{curr_shard_idx:04d}.tfrecord", curr_sequences, serialize_sequence)
        curr_shard_idx += 1
        curr_sequences = []

      game_count += 1
      if game_count % 100 == 0:
        logger.info(f"Processed: {game_count} games in PGN")
        logger.info(f"In Memory: {len(curr_sequences)} sequences")
    
    logger.info(f"Finished: {pgn_file_path}")
    logger.info(f"File Idx: {pgn_file_idx}")
    logger.info(f"Processed: {game_count} games in PGN")
    logger.info(f"In Memory: {len(curr_sequences)} sequences")

def save_shard(shard_path, items, serialize_function):
  with tf.io.TFRecordWriter(shard_path) as writer:
    for item in items:
      example = serialize_function(item)
      writer.write(example)
  logger.info(f"Saved shard to {shard_path}")

def serialize_sequence(sequence):
  def pad_and_flatten(games, max_moves=100):
    all_game_moves = np.zeros((max_moves, 5), dtype=np.uint64)
    num_moves = min(len(games), max_moves)
    if num_moves > 0:
      all_game_moves[:num_moves, :] = np.array(games[:num_moves], dtype=np.uint64)
    return all_game_moves

  stm_padded = pad_and_flatten(sequence[0][0])

  feature = {
    'seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[stm_padded.tobytes()])),
    'elo': tf.train.Feature(float_list=tf.train.FloatList(value=[sequence[1]]))
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

#  python3 stylometry/WinRateStylo/pgn_to_training_data.py ../../maia-individual/data-lichess/chess/unzipped stylometry/WinRateStylo/data/run2025-12-22