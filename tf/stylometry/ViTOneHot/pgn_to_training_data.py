"""
Convert chess PGN files to TensorFlow-compatible training data for stylometry.

This script processes PGN files and creates training data where:
- Input: Sequences of board positions (moves) from a single player in a game
- Output: One-hot encoded player identity

The data format is compatible with GameAggregateViT model which expects:
- Input shape: (batch, num_moves, 112, 8, 8) - LC0 board representation
- Output shape: (batch, num_players) - one-hot encoded player identity
"""

import chess
import chess.pgn
import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import io


class PlayerIndexMapper:
    """Maps player names to integer indices for one-hot encoding."""
    
    def __init__(self):
        self.player_to_idx: Dict[str, int] = {}
        self.idx_to_player: Dict[int, str] = {}
        self.next_idx = 0
    
    def get_or_create_index(self, player_name: str) -> int:
        """Get existing index or create new one for player."""
        if player_name not in self.player_to_idx:
            self.player_to_idx[player_name] = self.next_idx
            self.idx_to_player[self.next_idx] = player_name
            self.next_idx += 1
        return self.player_to_idx[player_name]
    
    def get_index(self, player_name: str) -> Optional[int]:
        """Get index for player, or None if not found."""
        return self.player_to_idx.get(player_name)
    
    def num_players(self) -> int:
        """Get total number of registered players."""
        return self.next_idx
    
    def save(self, filepath: str):
        """Save player mapping to file."""
        with open(filepath, 'w') as f:
            for player, idx in sorted(self.player_to_idx.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{player}\n")
    
    def load(self, filepath: str):
        """Load player mapping from file."""
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


def board_history_to_lc0_planes(board_history: List[chess.Board], repetition_counts: List[int]) -> np.ndarray:
    """
    Convert chess board history to LC0's 112-plane representation.
    
    Implements the full LC0 encoding with 8 positions of history:
    - 8 positions × 13 planes = 104 planes (pieces + repetitions)
    - 8 constant planes (castling, side to move, rule50, all ones)
    
    Args:
        board_history: List of up to 8 board positions (most recent first)
        repetition_counts: List of repetition counts for each position
    
    Returns:
        np.ndarray: Shape (112, 8, 8) representing the board state
    """
    planes = np.zeros((112, 8, 8), dtype=np.float32)
    
    # Ensure we have exactly 8 history positions (pad with empty boards if needed)
    history_len = len(board_history)
    if history_len > 8:
        board_history = board_history[:8]
        repetition_counts = repetition_counts[:8]
        history_len = 8
    
    # Encode each historical position (8 positions × 13 planes each)
    for hist_idx in range(8):
        plane_offset = hist_idx * 13
        
        if hist_idx < history_len:
            board = board_history[hist_idx]
            rep_count = repetition_counts[hist_idx] if hist_idx < len(repetition_counts) else 0
            
            # Encode pieces from the perspective of side to move in the current position
            # "Us" = side to move, "Them" = opponent
            us_color = board_history[0].turn  # Current side to move
            them_color = not us_color
            
            # Planes 0-5: Our pieces (P, N, B, R, Q, K)
            for piece_type in range(1, 7):  # PAWN=1 to KING=6
                piece_plane = plane_offset + piece_type - 1
                piece_bb = board.pieces(piece_type, us_color)
                for square in piece_bb:
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    planes[piece_plane, rank, file] = 1.0
            
            # Planes 6-11: Their pieces (P, N, B, R, Q, K)
            for piece_type in range(1, 7):
                piece_plane = plane_offset + 6 + piece_type - 1
                piece_bb = board.pieces(piece_type, them_color)
                for square in piece_bb:
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    planes[piece_plane, rank, file] = 1.0
            
            # Plane 12: Repetition counter (1 if position has repeated)
            if rep_count >= 1:
                planes[plane_offset + 12, :, :] = 1.0
    
    # Constant planes (104-111)
    current_board = board_history[0] if history_len > 0 else chess.Board()
    us_color = current_board.turn
    them_color = not us_color
    
    # Plane 104: Our queenside castling
    if current_board.has_queenside_castling_rights(us_color):
        planes[104, :, :] = 1.0
    
    # Plane 105: Our kingside castling
    if current_board.has_kingside_castling_rights(us_color):
        planes[105, :, :] = 1.0
    
    # Plane 106: Their queenside castling
    if current_board.has_queenside_castling_rights(them_color):
        planes[106, :, :] = 1.0
    
    # Plane 107: Their kingside castling
    if current_board.has_kingside_castling_rights(them_color):
        planes[107, :, :] = 1.0
    
    # Plane 108: Side to move (constant 1.0)
    planes[108, :, :] = 1.0
    
    # Plane 109: Rule50 counter (normalized)
    planes[109, :, :] = current_board.halfmove_clock / 99.0
    
    # Plane 110: Zero plane (move count - set to 0)
    planes[110, :, :] = 0.0
    
    # Plane 111: All ones plane (helps NN detect board edges)
    planes[111, :, :] = 1.0
    
    return planes


def extract_player_moves_from_game(
    game: chess.pgn.Game,
    player_mapper: PlayerIndexMapper,
    max_moves: int = 100
) -> List[Tuple[np.ndarray, int]]:
    """
    Extract move sequences for each player from a game.
    
    Args:
        game: chess.pgn.Game object
        player_mapper: PlayerIndexMapper to assign player indices
        max_moves: Maximum number of moves to include per sequence
        
    Returns:
        List of (move_sequence, player_idx) tuples
        move_sequence shape: (num_moves, 112, 8, 8)
    """
    white_name = game.headers.get("White", "Unknown_White")
    black_name = game.headers.get("Black", "Unknown_Black")
    
    white_idx = player_mapper.get_or_create_index(white_name)
    black_idx = player_mapper.get_or_create_index(black_name)
    
    # Track moves for each player with history
    white_moves = []
    black_moves = []
    
    # Maintain board history (up to 8 positions)
    board_history = []
    position_hashes = []  # For tracking repetitions
    repetition_counts = []
    
    board = game.board()
    
    for move_num, move in enumerate(game.mainline_moves()):
        # Add current position to history
        board_copy = board.copy()
        board_history.insert(0, board_copy)
        
        # Track position hash for repetition detection (simple FEN-based hash)
        pos_hash = hash(board.fen().split(' ')[0])  # Hash only piece positions
        position_hashes.insert(0, pos_hash)
        
        # Calculate repetition count for current position
        rep_count = sum(1 for h in position_hashes if h == pos_hash) - 1
        repetition_counts.insert(0, rep_count)
        
        # Keep only last 8 positions
        if len(board_history) > 8:
            board_history = board_history[:8]
            position_hashes = position_hashes[:8]
            repetition_counts = repetition_counts[:8]
        
        # Encode with full history
        board_planes = board_history_to_lc0_planes(board_history, repetition_counts)
        
        # Assign to white or black. Assign based on who made the previous move.
        if move_num != 0:
          if board.turn == chess.WHITE:
              black_moves.append(board_planes)
          else:
              white_moves.append(board_planes)
        
        # Make the move
        board.push(move)
        
        # Stop if we've collected enough moves
        if len(white_moves) >= max_moves and len(black_moves) >= max_moves:
            break
    
    results = []
    
    # Add white's moves if there are any
    if white_moves:
        white_array = np.array(white_moves[:max_moves], dtype=np.float32)
        results.append((white_array, white_idx))
    
    # Add black's moves if there are any
    if black_moves:
        black_array = np.array(black_moves[:max_moves], dtype=np.float32)
        results.append((black_array, black_idx))
    
    return results


def process_pgn_file(
    pgn_path: str,
    player_mapper: PlayerIndexMapper,
    max_moves: int = 100,
    max_games: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Process a PGN file and extract training data.
    
    Args:
        pgn_path: Path to PGN file
        player_mapper: PlayerIndexMapper instance
        max_moves: Maximum moves per sequence
        max_games: Maximum number of games to process (None = all)
        
    Returns:
        Tuple of (sequences_list, labels_list)
        Each sequence has shape (num_moves, 112, 8, 8)
    """
    sequences = []
    labels = []
    
    with open(pgn_path, 'r') as pgn_file:
        game_count = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            # Extract moves for both players
            player_sequences = extract_player_moves_from_game(
                game, player_mapper, max_moves
            )
            
            for seq, label in player_sequences:
                sequences.append(seq)
                labels.append(label)
            
            game_count += 1
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(sequences)} sequences")
            
            if max_games is not None and game_count >= max_games:
                break
    
    print(f"Total: {game_count} games, {len(sequences)} sequences")
    return sequences, labels


def pad_sequences(
    sequences: List[np.ndarray],
    max_moves: int = 100,
    padding_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to uniform length and create attention masks.
    
    Args:
        sequences: List of arrays with shape (num_moves, 112, 8, 8)
        max_moves: Target length for padding
        padding_value: Value to use for padding
        
    Returns:
        Tuple of (padded_sequences, masks)
        padded_sequences shape: (num_sequences, max_moves, 112, 8, 8)
        masks shape: (num_sequences, max_moves) - 1 for real moves, 0 for padding
    """
    num_sequences = len(sequences)
    padded = np.full(
        (num_sequences, max_moves, 112, 8, 8),
        padding_value,
        dtype=np.float32
    )
    masks = np.zeros((num_sequences, max_moves), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        num_moves_in_seq = min(len(seq), max_moves)
        padded[i, :num_moves_in_seq] = seq[:num_moves_in_seq]
        masks[i, :num_moves_in_seq] = 1.0
    
    return padded, masks

def save_training_data(
    sequences: List[np.ndarray],
    labels: List[int],
    output_prefix: str,
    max_moves: int = 100
):
    """
    Save training data to numpy files.
    
    Args:
        sequences: List of move sequences
        labels: List of player indices
        output_prefix: Prefix for output files
        max_moves: Maximum moves per sequence
    """
    padded_sequences, masks = pad_sequences(sequences, max_moves)
    labels_array = np.array(labels, dtype=np.int32)
    
    np.save(f"{output_prefix}_sequences.npy", padded_sequences)
    np.save(f"{output_prefix}_labels.npy", labels_array)
    np.save(f"{output_prefix}_masks.npy", masks)
    
    print(f"Saved data to {output_prefix}_*.npy")
    print(f"  Sequences shape: {padded_sequences.shape}")
    print(f"  Labels shape: {labels_array.shape}")
    print(f"  Masks shape: {masks.shape}")

if __name__ == "__main__":
    import argparse
    import glob
    import os

    MAX_FILES = 100000
    
    parser = argparse.ArgumentParser(
        description="Convert PGN files to training data for chess stylometry"
    )
    parser.add_argument("inputs", nargs="+", 
                       help="Input PGN file(s) or folder(s) containing PGN files")
    parser.add_argument("output_prefix", help="Output file prefix")
    parser.add_argument("--max-moves", type=int, default=100,
                       help="Maximum moves per sequence (default: 100)")
    parser.add_argument("--max-games", type=int, default=None,
                       help="Maximum games to process per file (default: all)")
    parser.add_argument("--player-map", type=str, default="stylometry/player_mapping.txt",
                       help="Player mapping file (default: stylometry/player_mapping.txt)")
    
    args = parser.parse_args()
    
    # Collect all PGN files from inputs (files and folders)
    pgn_files = []
    for input_path in args.inputs:
        if os.path.isfile(input_path):
            # Direct file path
            if input_path.endswith('.pgn'):
                pgn_files.append(input_path)
            else:
                print(f"Warning: Skipping non-PGN file: {input_path}")
        elif os.path.isdir(input_path):
            print(f"Searching directory: {input_path}")
            # Directory - glob for all .pgn files
            folder_pgns = glob.glob(os.path.join(input_path, "*.pgn"))
            if folder_pgns:
                pgn_files.extend(folder_pgns[:MAX_FILES])
                print(f"Found {len(folder_pgns[:MAX_FILES])} PGN files in {input_path} (limited to {MAX_FILES})")
            else:
                print(f"Warning: No PGN files found in directory: {input_path}")
        else:
            print(f"Warning: Path not found: {input_path}")
    
    if not pgn_files:
        print("Error: No PGN files found to process!")
        exit(1)
    
    print(f"\nTotal PGN files to process: {len(pgn_files)}")
    print("-" * 60)
    
    # Create player mapper
    player_mapper = PlayerIndexMapper()
    
    # Process all PGN files
    all_sequences = []
    all_labels = []
    
    for pgn_file in pgn_files:
        print(f"\nProcessing {pgn_file}...")
        sequences, labels = process_pgn_file(
            pgn_file,
            player_mapper,
            max_moves=args.max_moves,
            max_games=args.max_games
        )
        all_sequences.extend(sequences)
        all_labels.extend(labels)
        print(f"  Extracted {len(sequences)} sequences")
    
    print(f"\n{'='*60}")
    print(f"Total sequences: {len(all_sequences)}")
    
    # Save player mapping
    player_mapper.save(args.player_map)
    print(f"Saved player mapping to {args.player_map}")
    print(f"Total players: {player_mapper.num_players()}")
    
    # Save training data
    save_training_data(all_sequences, all_labels, args.output_prefix, args.max_moves)
