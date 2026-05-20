import tensorflow as tf
import numpy as np
import chess
import chess.svg
import argparse
import random
import os
import sys
from typing import List, Tuple, Optional

# Constants matching the data generation script
MAX_MOVES = 100
NUM_GAMES = 5

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

def chessboard_struct_to_board(struct: np.ndarray) -> Optional[chess.Board]:
  """
  Converts a 5-element uint64 array to a chess.Board object.
  The encoding follows pgn_to_game_outcome_training_data.py player perspective.
  """
  values = np.asarray(struct, dtype=np.uint64)
  if values.shape[0] < 5 or not np.any(values[:3] > 0):
    return None

  occ = int(values[0])
  pcs_1 = int(values[1])
  pcs_2 = int(values[2])
  metadata = int(values[4])

  # Bit 0 of castle_and_stm (top byte) is 1 if it is NOT the player's turn (opponent's turn)
  # But the board_to_chessboard_struct_player_perspective helper says:
  # castle_and_stm |= 1 if board.turn != player_color
  # However, the pieces are encoded from the perspective of player_color.
  
  # Metadata bits:
  # stm_bit = 1 if it's NOT player's turn. 
  # castle_rights (from player's perspective): 2 (us qs), 4 (us ks), 8 (them qs), 16 (them ks)
  
  castle_and_stm = (metadata >> 24) & 0xFF
  is_opponent_turn = (castle_and_stm & 0x1) == 1
  us_qs = (castle_and_stm & 0x2) != 0
  us_ks = (castle_and_stm & 0x4) != 0
  them_qs = (castle_and_stm & 0x8) != 0
  them_ks = (castle_and_stm & 0x16) != 0
  
  # Note: The generation script uses _flip_vertical if is_black_perspective.
  # So sq=0 in occupied/pieces refers to A1 if white, A8 if black.
  # But we just need a valid FEN or board state.
  
  board = chess.Board(None) # Empty board
  
  # We don't actually know if player_color was White or Black unless we guess.
  # But the encoding is "player-centric". 
  # Let's assume White for the sake of GUI display, or use the turn bit.
  
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
      
      # piece_type = pt_idx + 1 (since pawns is index 0)
      piece_type = (val & 0x7) + 1
      is_stm_piece = (val & 0x8) == 0
      
      # Since we are showing "player perspective", let's map player pieces to White
      # and opponent pieces to Black for the visual board.
      color = chess.WHITE if is_stm_piece else chess.BLACK
      board.set_piece_at(sq, chess.Piece(piece_type, color))

  # Set turn based on is_opponent_turn
  board.turn = chess.BLACK if is_opponent_turn else chess.WHITE
  
  # Set castling rights (best effort as we don't know real absolute color)
  # For the GUI, we'll just treat player as white.
  board.castling_rights = 0
  if us_ks: board.castling_rights |= chess.BB_H1
  if us_qs: board.castling_rights |= chess.BB_A1
  if them_ks: board.castling_rights |= chess.BB_H8
  if them_qs: board.castling_rights |= chess.BB_A8
  
  return board

def parse_tfrecord(example_proto):
  feature_description = {
    'stm_player_seq': tf.io.FixedLenFeature([], tf.string),
    'opp_player_seq': tf.io.FixedLenFeature([], tf.string),
    'stm_player_name': tf.io.FixedLenFeature([], tf.string),
    'stm_player_elo': tf.io.FixedLenFeature([], tf.int64),
    'opp_player_name': tf.io.FixedLenFeature([], tf.string),
    'opp_player_elo': tf.io.FixedLenFeature([], tf.int64),
    'wdl': tf.io.FixedLenFeature([3], tf.float32),
  }
  return tf.io.parse_single_example(example_proto, feature_description)

def get_random_sequence(dataset_path: str):
  files = tf.io.gfile.glob(os.path.join(dataset_path, "*.tfrecord"))
  if not files:
    # Try seq_shards subfolder
    files = tf.io.gfile.glob(os.path.join(dataset_path, "seq_shards", "*.tfrecord"))
  
  if not files:
    print(f"No .tfrecord files found in {dataset_path}")
    return None

  raw_dataset = tf.data.TFRecordDataset(files)
  # Take a large shuffle buffer and pick one
  shuffled = raw_dataset.shuffle(1000).take(1)
  
  for record in shuffled:
    parsed = parse_tfrecord(record)
    
    # stm_player_seq is (NUM_GAMES, MAX_MOVES, 5) uint64
    stm_seq = np.frombuffer(parsed['stm_player_seq'].numpy(), dtype=np.uint64).reshape((NUM_GAMES, MAX_MOVES, 5))
    
    return {
      'name': parsed['stm_player_name'].numpy().decode('utf-8'),
      'elo': int(parsed['stm_player_elo'].numpy()),
      'seq': stm_seq
    }
  return None

def clear_screen():
  os.system('cls' if os.name == 'nt' else 'clear')

def main():
  parser = argparse.ArgumentParser(description="Visualize stylometry dataset")
  parser.add_argument("dataset_path", help="Path to the dataset directory containing tfrecords")
  args = parser.parse_args()

  while True:
    data = get_random_sequence(args.dataset_path)
    if not data:
      break
    
    games = data['seq']
    num_games_present = 0
    valid_games = []
    for g_idx in range(NUM_GAMES):
      # A game is valid if it has at least one non-zero move
      if np.any(games[g_idx]):
        valid_games.append(games[g_idx])
    
    if not valid_games:
      print("Found a sequence with no valid games, skipping...")
      continue

    curr_game_idx = 0
    curr_move_idx = 0
    
    while True:
      clear_screen()
      game_data = valid_games[curr_game_idx]
      num_moves = 0
      for i in range(MAX_MOVES):
        if not np.any(game_data[i]):
          break
        num_moves += 1
      
      board = chessboard_struct_to_board(game_data[curr_move_idx])
      
      print(f"Viewing Player Sequence (Game {curr_game_idx + 1}/{len(valid_games)}, Move {curr_move_idx + 1}/{num_moves})")
      if board:
        print(board)
      else:
        print("Error displaying board (empty struct)")
      
      print("\nCommands: [n]ext move, [p]rev move, [N]ext game, [P]rev game, [g]uess Elo, [q]uit")
      cmd = input("> ").strip().lower()
      
      if cmd == 'n':
        if curr_move_idx < num_moves - 1:
          curr_move_idx += 1
      elif cmd == 'p':
        if curr_move_idx > 0:
          curr_move_idx -= 1
      elif cmd == 'next': # Shift+N usually. Use 'j'/'k' maybe? Let's stick to N/P
        pass
      elif cmd == 'n' or cmd == 'N': # input().lower() makes it 'n'
        # Ambiguity with next move. Let's use 'j' for next game, 'k' for prev game as common shortcuts
        pass
      
      # Re-doing command logic for clarity
      if cmd == 'j': # Next game
        curr_game_idx = (curr_game_idx + 1) % len(valid_games)
        curr_move_idx = 0
      elif cmd == 'k': # Prev game
        curr_game_idx = (curr_game_idx - 1) % len(valid_games)
        curr_move_idx = 0
      elif cmd == 'g':
        guess = input("Enter your Elo guess: ")
        print(f"\nActual Player: {data['name']}")
        print(f"Actual Elo: {data['elo']}")
        input("\nPress Enter to continue to next sequence...")
        break
      elif cmd == 'q':
        sys.exit(0)
      
      # Handle 'n' and 'p' specifically
      if cmd == 'n':
        if curr_move_idx < num_moves - 1:
          curr_move_idx += 1
      elif cmd == 'p':
        if curr_move_idx > 0:
          curr_move_idx -= 1

if __name__ == "__main__":
  main()
