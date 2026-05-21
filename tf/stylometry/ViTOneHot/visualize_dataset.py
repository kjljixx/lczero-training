import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import numpy as np
import chess
import chess.svg
import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parents[1]) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parents[1]))

from stylometry.ViTOneHot.train_stylometry import (
    chessboard_struct_to_lc0_planes,
    SEQ_PLANES,
)
from stylometry.ViTOneHot.get_bin_accuracy import load_model, detect_model_kind

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

def get_char():
  if os.name == 'nt':
    import msvcrt
    return msvcrt.getch().decode('utf-8', 'ignore').lower()
  else:
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setraw(sys.stdin.fileno())
      ch = sys.stdin.read(1)
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()

def main():
  parser = argparse.ArgumentParser(description="Visualize stylometry dataset")
  parser.add_argument("dataset_path", help="Path to the dataset directory containing tfrecords")
  parser.add_argument("--model", type=str, help="Path to the keras model to evaluate Elo")
  parser.add_argument("--shift", type=float, default=0.0, help="Amount to shift the model's Elo prediction")
  args = parser.parse_args()

  model = None
  elo_predictor = None
  if args.model:
    model = load_model(args.model)
    _, elo_predictor = detect_model_kind(model)

  all_guesses = []
  all_actuals = []
  all_model_guesses = []

  while True:
    data = get_random_sequence(args.dataset_path)
    if not data:
      break
    
    games = data['seq']
    valid_games = []
    for g_idx in range(NUM_GAMES):
      # A game is valid if it has at least one non-zero move
      if np.any(games[g_idx]):
        valid_games.append(games[g_idx])
        break # Only take the first game
    
    if not valid_games:
      print("Found a sequence with no valid games, skipping...")
      continue

    # Prepare features for the model if needed
    model_elo_pred = None
    if elo_predictor is not None:
      planes = np.zeros((1, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
      mask = np.zeros((1, NUM_GAMES, MAX_MOVES), dtype=np.float32)
      # We evaluate the entire sequence of 1 game that we just picked (or whatever is in valid_games)
      # Or actually we can just pass the first game in index 0 of the model. 
      for m_idx in range(MAX_MOVES):
        struct = valid_games[0][m_idx]
        if np.any(struct > 0):
          p, _ = chessboard_struct_to_lc0_planes(struct, short=True)
          planes[0, 0, m_idx] = p.astype(np.float32)
          mask[0, 0, m_idx] = 1.0

      flat_seq = tf.reshape(tf.convert_to_tensor(planes), [1 * NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8])
      flat_mask = tf.reshape(tf.convert_to_tensor(mask), [1 * NUM_GAMES, MAX_MOVES])
      game_elo = elo_predictor({'seq': flat_seq, 'mask': flat_mask}, training=False)
      game_elo = tf.reshape(game_elo, [1, NUM_GAMES])
      # Since we only put it in index 0
      model_elo_pred = float(game_elo[0, 0]) + args.shift

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
        is_player_turn = board.turn == chess.WHITE
        turn_str = "PLAYER's turn (Guess THEIR Elo!)" if is_player_turn else "OPPONENT's turn"
        print(f"[{turn_str}]\n")
        
        # Make the board much prettier using unicode pieces mapping
        print(board.unicode(empty_square="·"))
      else:
        print("Error displaying board (empty struct)")
      
      print("\nCommands: [n]ext move, [b]ack, [j] next game, [k] prev game, [g]uess Elo, [q]uit")
      print("> ", end="", flush=True)
      cmd = get_char()
      print(cmd) # Echo the command
      
      if cmd == 'n':
        if curr_move_idx < num_moves - 1:
          curr_move_idx += 1
      elif cmd == 'b':
        if curr_move_idx > 0:
          curr_move_idx -= 1
      elif cmd == 'j':
        curr_game_idx = (curr_game_idx + 1) % len(valid_games)
        curr_move_idx = 0
      elif cmd == 'k':
        curr_game_idx = (curr_game_idx - 1) % len(valid_games)
        curr_move_idx = 0
      elif cmd == 'g':
        guess_str = input("Enter your Elo guess: ")
        try:
          guess_elo = float(guess_str)
        except ValueError:
          print("Invalid Elo. Skipping this sequence without recording guess...")
          input("\nPress Enter to continue to next sequence...")
          break
        
        actual_elo = float(data['elo'])
        all_guesses.append(guess_elo)
        all_actuals.append(actual_elo)
        
        guesses_np = np.array(all_guesses)
        actuals_np = np.array(all_actuals)
        mae = np.mean(np.abs(guesses_np - actuals_np))
        mse = np.mean(np.square(guesses_np - actuals_np))

        print(f"\nActual Player: {data['name']}")
        print(f"Actual Elo: {data['elo']}")
        
        if elo_predictor is not None and model_elo_pred is not None:
          all_model_guesses.append(model_elo_pred)
          model_guesses_np = np.array(all_model_guesses)
          model_mae = np.mean(np.abs(model_guesses_np - actuals_np))
          model_mse = np.mean(np.square(model_guesses_np - actuals_np))
          print(f"\nModel Guess: {model_elo_pred:.1f}")

        print(f"\n--- Stats over {len(all_guesses)} guesses ---")
        print(f"Your MAE: {mae:.2f}")
        print(f"Your MSE: {mse:.2f}")
        
        if elo_predictor is not None and model_elo_pred is not None:
          print(f"Model MAE: {model_mae:.2f}")
          print(f"Model MSE: {model_mse:.2f}")

        input("\nPress Enter to continue to next sequence...")
        break
      elif cmd == 'q':
        sys.exit(0)

if __name__ == "__main__":
  main()
