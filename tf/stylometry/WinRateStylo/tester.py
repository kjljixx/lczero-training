import argparse
import chess
import chess.pgn
import tensorflow as tf
import tensorflow.keras as keras  # type: ignore
from stylometry.WinRateStylo.train_wr_stylometry import ScaffoldedViTAndWinRate
from stylometry.WinRateStylo.pgn_to_training_data import board_history_to_lc0_planes, PlayerIndexMapper
import numpy as np
from typing import List, Tuple, Optional

MODEL_FILE = "stylometry/WinRateStylo/models/run4-stylo-disabled/checkpoint_run3_epoch_162.keras"
MAX_MOVES = 100


def lc0_planes_to_board(planes: np.ndarray) -> chess.Board:
  """
  Convert LC0 input planes back to a chess.Board.
  This is the inverse of board_history_to_lc0_planes (for the current position only).
  """
  board = chess.Board.empty()
  
  # Determine side to move from plane 108 (stm plane)
  # If plane 108 is all 1s, white is to move; otherwise black
  stm_plane = planes[108]
  white_to_move = stm_plane[0, 0] < 0.5
  board.turn = chess.WHITE if white_to_move else chess.BLACK
  
  # The first 13 planes (hist_idx=0) contain the current position
  # Planes 0-5: our pieces (P, N, B, R, Q, K)
  # Planes 6-11: their pieces (P, N, B, R, Q, K)
  
  us_color = board.turn
  them_color = not us_color
  
  piece_symbols = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
  
  # Our pieces (planes 0-5)
  for piece_idx, piece_type in enumerate(piece_symbols):
    plane = planes[piece_idx]
    for rank in range(8):
      for file in range(8):
        if plane[rank, file] > 0.5:
          square = chess.square(file, rank if us_color == chess.WHITE else 7-rank)
          board.set_piece_at(square, chess.Piece(piece_type, us_color))
  
  # Their pieces (planes 6-11)
  for piece_idx, piece_type in enumerate(piece_symbols):
    plane = planes[6 + piece_idx]
    for rank in range(8):
      for file in range(8):
        if plane[rank, file] > 0.5:
          square = chess.square(file, rank if them_color == chess.WHITE else 7-rank)
          board.set_piece_at(square, chess.Piece(piece_type, them_color))
  
  # Castling rights from planes 104-107
  castling_fen = ""
  if planes[105, 0, 0] > 0.5:  # our kingside
    castling_fen += "K" if us_color == chess.WHITE else "k"
  if planes[104, 0, 0] > 0.5:  # our queenside
    castling_fen += "Q" if us_color == chess.WHITE else "q"
  if planes[107, 0, 0] > 0.5:  # their kingside
    castling_fen += "K" if them_color == chess.WHITE else "k"
  if planes[106, 0, 0] > 0.5:  # their queenside
    castling_fen += "Q" if them_color == chess.WHITE else "q"
  
  # Sort castling string to standard order (KQkq)
  castling_order = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}
  castling_fen = ''.join(sorted(castling_fen, key=lambda c: castling_order.get(c, 4)))
  
  if castling_fen:
    board.set_castling_fen(castling_fen)
  else:
    board.set_castling_fen("-")
  
  # 50-move rule from plane 109
  halfmove_clock = int(planes[109, 0, 0] * 99.0)
  board.halfmove_clock = halfmove_clock
  
  return board


def pretty_print_board(board: chess.Board) -> str:
  """Pretty print a chess board with coordinates."""
  lines = []
  lines.append("  +-----------------+")
  
  material_diff = 0
  for rank in range(7, -1, -1):
    row = f"{rank + 1} | "
    for file in range(8):
      square = chess.square(file, rank)
      piece = board.piece_at(square)
      if piece:
        material_diff += (1 if piece.color == chess.WHITE else -1) * {
          chess.PAWN: 1,
          chess.KNIGHT: 3,
          chess.BISHOP: 3,
          chess.ROOK: 5,
          chess.QUEEN: 9,
          chess.KING: 0
        }[piece.piece_type]
        row += piece.symbol() + " "
      else:
        row += ". "
    row += "|"
    lines.append(row)
  
  lines.append("  +-----------------+")
  lines.append("    a b c d e f g h")
  lines.append(" STM: " + ("White" if board.turn == chess.WHITE else "Black"))
  lines.append(" Piece Diff: " + str(material_diff))
  lines.append(board.fen())
  
  return "\n".join(lines)

def extract_player_moves(
    pgn_path: str,
    color: chess.Color,
    max_moves: int = MAX_MOVES
) -> np.ndarray:
  """Extract moves from the first game in a PGN file for the specified color."""
  with open(pgn_path, 'r') as pgn_file:
    game = chess.pgn.read_game(pgn_file)
    if game is None:
      raise ValueError(f"No game found in {pgn_path}")
  
  moves = []
  board_history = []
  position_hashes = []
  repetition_counts = []
  board = game.board()
  
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
    
    # Collect moves for the specified color (after the first move)
    if move_num != 0:
      if board.turn != color:  # Previous move was by our color
        moves.append(board_planes)
    
    board.push(move)
    
    if len(moves) >= max_moves:
      break
  
  if not moves:
    raise ValueError(f"No moves found for {'white' if color == chess.WHITE else 'black'} in {pgn_path}")
  
  return np.array(moves[:max_moves], dtype=np.float32)


def fen_to_lc0_planes(fen: str) -> np.ndarray:
  """Convert a FEN string to LC0 input planes."""
  board = chess.Board(fen)
  board_history = [board]
  repetition_counts = [0]
  return board_history_to_lc0_planes(board_history, repetition_counts)


def pad_sequence(
    sequence: np.ndarray,
    max_moves: int = MAX_MOVES
) -> Tuple[np.ndarray, np.ndarray]:
  """Pad a sequence to max_moves and generate mask."""
  num_moves = len(sequence)
  padded = np.zeros((max_moves, 112, 8, 8), dtype=np.float32)
  mask = np.zeros(max_moves, dtype=np.float32)
  
  num_moves_to_copy = min(num_moves, max_moves)
  padded[:num_moves_to_copy] = sequence[:num_moves_to_copy]
  mask[:num_moves_to_copy] = 1.0
  
  return padded, mask


def parse_color(color_str: str) -> chess.Color:
  """Parse color string to chess.Color."""
  color_str = color_str.lower().strip()
  if color_str in ('w', 'white'):
    return chess.WHITE
  elif color_str in ('b', 'black'):
    return chess.BLACK
  else:
    raise ValueError(f"Invalid color: {color_str}. Use 'white'/'w' or 'black'/'b'")


def create_val_dataset_with_metadata(
    data_prefix: str,
    player_mapper: PlayerIndexMapper,
    batch_size: int = 1,
    shuffle: bool = True,
    indices: Optional[np.ndarray] = None,
    val_split: float = 0.1,
    is_val: bool = True
):
  """
  Create validation dataset that also yields player names and position planes.
  Yields: (inputs_dict, label, player1_name, player2_name, position_planes)
  """
  sequences_mmap = np.load(f"{data_prefix}_sequences.npy", mmap_mode='r')
  labels_mmap = np.load(f"{data_prefix}_labels.npy", mmap_mode='r')
  masks_mmap = np.load(f"{data_prefix}_masks.npy", mmap_mode='r')
  positions_players_mmap = np.load(f"{data_prefix}_position_players.npy", mmap_mode='r')
  positions_mmap = np.load(f"{data_prefix}_positions.npy", mmap_mode='r')
  positions_labels_mmap = np.load(f"{data_prefix}_positions_labels.npy", mmap_mode='r')

  label_to_indices = {}
  for idx in range(len(sequences_mmap)):
    label = int(labels_mmap[idx])
    if label not in label_to_indices:
      label_to_indices[label] = []
    label_to_indices[label].append(idx)
  for label in label_to_indices:
    if is_val:
      split_idx = int(len(label_to_indices[label]) * val_split)
      label_to_indices[label] = label_to_indices[label][:split_idx]
    else:
      split_idx = int(len(label_to_indices[label]) * val_split)
      label_to_indices[label] = label_to_indices[label][split_idx:]

  if indices is not None:
    num_samples = len(indices)
  else:
    num_samples = len(positions_mmap)
    indices = np.arange(num_samples)
  
  idx_order = indices.copy()
  if shuffle:
    np.random.shuffle(idx_order)
  
  for idx in idx_order:
    player1_idx = int(positions_players_mmap[idx][0])
    player2_idx = int(positions_players_mmap[idx][1])
    
    player1_name = player_mapper.idx_to_player.get(player1_idx, f"Player_{player1_idx}")
    player2_name = player_mapper.idx_to_player.get(player2_idx, f"Player_{player2_idx}")

    if int(positions_players_mmap[idx][0]) not in label_to_indices:
      continue
    if int(positions_players_mmap[idx][1]) not in label_to_indices:
      continue
    if len(label_to_indices[int(positions_players_mmap[idx][0])]) == 0:
      continue
    if len(label_to_indices[int(positions_players_mmap[idx][1])]) == 0:
      continue
    seq1idx = np.random.choice(label_to_indices[player1_idx])
    seq1 = sequences_mmap[seq1idx]
    mask1 = masks_mmap[seq1idx]
    seq2idx = np.random.choice(label_to_indices[player2_idx])
    seq2 = sequences_mmap[seq2idx]
    mask2 = masks_mmap[seq2idx]
    position_planes = positions_mmap[idx]
    position = position_planes.flatten()
    position_label = positions_labels_mmap[idx]
    
    inputs = {
      'input1': np.expand_dims(seq1, axis=0),
      'input2': np.expand_dims(seq2, axis=0),
      'pos': np.expand_dims(position, axis=0),
      'mask1': np.expand_dims(mask1, axis=0),
      'mask2': np.expand_dims(mask2, axis=0),
    }
    
    yield (inputs, position_label, player1_name, player2_name, position_planes)


def walk_validation(model, data_prefix: str, player_map_path: str, val_split: float = 0.1):
  """Walk through validation dataset one sample at a time."""
  positions_mmap = np.load(f"{data_prefix}_positions.npy", mmap_mode='r')
  num_samples = len(positions_mmap)
  
  player_mapper = PlayerIndexMapper()
  player_mapper.load(player_map_path)
  
  np.random.seed(42)
  indices = np.arange(num_samples)
  val_size = int(num_samples * val_split)
  val_indices = indices[:val_size]
  
  print(f"Validation samples: {len(val_indices)}")
  print(f"Total players: {player_mapper.num_players()}")
  print("-" * 50)
  
  label_names = ["Win", "Draw", "Loss"]
  
  val_generator = create_val_dataset_with_metadata(
    data_prefix, player_mapper, batch_size=1, shuffle=True, val_split=val_split, is_val=True, indices=val_indices
  )
  
  for i, (inputs, label, player1_name, player2_name, position_planes) in enumerate(val_generator):
    # inputs["input1"] = np.zeros((1, MAX_MOVES, 34, 8, 8), dtype=np.float32)
    # inputs["input2"] = np.zeros((1, MAX_MOVES, 34, 8, 8), dtype=np.float32)
    # inputs["mask1"] = np.ones((1, MAX_MOVES), dtype=np.float32)
    # inputs["mask2"] = np.ones((1, MAX_MOVES), dtype=np.float32)
    pred = model(inputs)
    pred_np = pred.numpy()[0]
    label_np = np.array(label)
    
    pred_idx = np.argmax(pred_np)
    label_idx = np.argmax(label_np)
    correct = pred_idx == label_idx
    
    # Reconstruct board from position planes
    board = lc0_planes_to_board(position_planes)
    
    print(f"\n{'=' * 50}")
    print(f"Sample {i + 1}/{len(val_indices)}")
    print(f"{'=' * 50}")
    print(f"\nPlayers:")
    print(f"  Player 1 (STM): {player1_name}")
    print(f"  Player 2:       {player2_name}")
    print(f"\nPosition:")
    print(pretty_print_board(board))
    print(f"\nPrediction: {label_names[pred_idx]:4s} (W:{pred_np[0]:.3f} D:{pred_np[1]:.3f} L:{pred_np[2]:.3f})")
    print(f"Actual:     {label_names[label_idx]:4s} (W:{label_np[0]:.3f} D:{label_np[1]:.3f} L:{label_np[2]:.3f})")
    print(f"Result:     {'CORRECT' if correct else 'INCORRECT'}")
    
    try:
      user_input = input("\nPress Enter for next, 'q' to quit: ").strip().lower()
      if user_input == 'q':
        print("Exiting validation walk...")
        break
    except KeyboardInterrupt:
      print("\nExiting validation walk...")
      break
  
  print("\nValidation walk complete.")


def main():
  parser = argparse.ArgumentParser(
    description="Test win rate prediction model with custom PGN inputs"
  )
  
  subparsers = parser.add_subparsers(dest="mode", help="Run mode")
  
  # PGN inference mode
  pgn_parser = subparsers.add_parser("pgn", help="Run inference with custom PGN inputs")
  pgn_parser.add_argument("--pgn1", type=str, required=True,
                          help="Path to first PGN file")
  pgn_parser.add_argument("--color1", type=str, required=True,
                          help="Color to extract from pgn1 (white/w or black/b)")
  pgn_parser.add_argument("--pgn2", type=str, required=True,
                          help="Path to second PGN file")
  pgn_parser.add_argument("--color2", type=str, required=True,
                          help="Color to extract from pgn2 (white/w or black/b)")
  pgn_parser.add_argument("--fen", type=str, required=True,
                          help="FEN string for position evaluation")
  pgn_parser.add_argument("--model", type=str, default=MODEL_FILE,
                          help=f"Path to model file (default: {MODEL_FILE})")
  
  # Validation walk mode
  val_parser = subparsers.add_parser("val", help="Walk through validation dataset")
  val_parser.add_argument("data_prefix", type=str,
                          help="Data prefix for validation dataset")
  val_parser.add_argument("--player-map", type=str, default="stylometry/player_mapping.txt",
                          help="Player mapping file (default: stylometry/player_mapping.txt)")
  val_parser.add_argument("--val-split", type=float, default=0.1,
                          help="Validation split ratio (default: 0.1)")
  val_parser.add_argument("--model", type=str, default=MODEL_FILE,
                          help=f"Path to model file (default: {MODEL_FILE})")
  
  args = parser.parse_args()
  
  if args.mode is None:
    parser.print_help()
    return
  
  model_path = args.model
  print(f"Loading model from {model_path}...")
  model = keras.saving.load_model(
    model_path,
    custom_objects={"ScaffoldedViTAndWinRate": ScaffoldedViTAndWinRate}
  )
  
  if args.mode == "pgn":
    # Parse colors
    color1 = parse_color(args.color1)
    color2 = parse_color(args.color2)
    
    print(f"\nExtracting moves from {args.pgn1} ({'white' if color1 == chess.WHITE else 'black'})...")
    seq1_raw = extract_player_moves(args.pgn1, color1, MAX_MOVES)
    seq1, mask1 = pad_sequence(seq1_raw, MAX_MOVES)
    print(f"  Extracted {len(seq1_raw)} moves")
    
    print(f"\nExtracting moves from {args.pgn2} ({'white' if color2 == chess.WHITE else 'black'})...")
    seq2_raw = extract_player_moves(args.pgn2, color2, MAX_MOVES)
    seq2, mask2 = pad_sequence(seq2_raw, MAX_MOVES)
    print(f"  Extracted {len(seq2_raw)} moves")
    
    print(f"\nConverting FEN to position planes...")
    print(f"  FEN: {args.fen}")
    pos = fen_to_lc0_planes(args.fen).flatten()
    
    # Build batched input
    batched_input = {
      'input1': np.expand_dims(seq1, axis=0),
      'input2': np.expand_dims(seq2, axis=0),
      'pos': np.expand_dims(pos, axis=0),
      'mask1': np.expand_dims(mask1, axis=0),
      'mask2': np.expand_dims(mask2, axis=0),
    }
    
    print("\nRunning inference...")
    pred = model(batched_input)
    pred_np = pred.numpy()[0]
    
    print("\n" + "=" * 50)
    print("Prediction (Win / Draw / Loss for player 1):")
    print(f"  Win:  {pred_np[0]:.4f} ({pred_np[0]*100:.1f}%)")
    print(f"  Draw: {pred_np[1]:.4f} ({pred_np[1]*100:.1f}%)")
    print(f"  Loss: {pred_np[2]:.4f} ({pred_np[2]*100:.1f}%)")
    print("=" * 50)
  
  elif args.mode == "val":
    walk_validation(model, args.data_prefix, args.player_map, args.val_split)


if __name__ == "__main__":
  main()
  #python3 -m stylometry.WinRateStylo.tester val stylometry/WinRateStylo/data/run3