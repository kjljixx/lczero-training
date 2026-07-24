import numpy as np
import argparse
import os
import array
from tqdm import tqdm

def calculate_metrics(elo_diffs):
    """Calculate mean and quartiles for the given list of absolute Elo differences."""
    if len(elo_diffs) == 0:
        return 0, 0, 0, 0
    return (
        np.mean(elo_diffs),
        np.percentile(elo_diffs, 25),
        np.percentile(elo_diffs, 50),
        np.percentile(elo_diffs, 75)
    )

def print_metrics(name, elo_diffs):
    mean, q25, q50, q75 = calculate_metrics(elo_diffs)
    print(f"[{name}]")
    print(f"  Games: {len(elo_diffs)}")
    print(f"  Mean:  {mean:.3f}")
    print(f"  Q1/Q2/Q3: {q25:.3f} / {q50:.3f} / {q75:.3f}\n")

def filter_by_truncation(games, max_drop_rate=0.20, increase_median=False):
    total_games = len(games)
    keep_count = int(total_games * (1 - max_drop_rate))
    
    if increase_median:
        sorted_indices = np.argsort(games['diff'])[::-1]
    else:
        sorted_indices = np.argsort(games['diff'])
        
    filtered_indices = sorted_indices[:keep_count]
    filtered_indices = np.sort(filtered_indices)  # Maintain original PGN order
    return games[filtered_indices]

def filter_by_distribution_matching(games, target_median=73.0, max_drop_rate=0.20, increase_median=False):
    total_games = len(games)
    drop_budget = int(total_games * max_drop_rate)
    
    diffs = games['diff']
    if increase_median:
        penalties = np.maximum(0, target_median - diffs)
    else:
        penalties = np.maximum(0, diffs - target_median)
    
    if penalties.sum() == 0:
        return games
        
    # Gumbel-style weighted sampling without replacement to avoid np.random.choice overhead.
    # We generate exponential random variables E_i ~ Exp(1) and divide by penalties.
    # The smallest keys correspond to the dropped elements.
    keys = np.full(total_games, np.inf)
    
    non_zero_mask = penalties > 0
    num_non_zero = np.count_nonzero(non_zero_mask)
    
    if num_non_zero > 0:
        exp_samples = np.random.exponential(1.0, size=num_non_zero)
        keys[non_zero_mask] = exp_samples / penalties[non_zero_mask]
        
    if drop_budget >= num_non_zero:
        drop_indices = np.where(non_zero_mask)[0]
    else:
        # np.argpartition finds the drop_budget smallest keys in O(N) time
        drop_indices = np.argpartition(keys, drop_budget)[:drop_budget]
    
    keep_mask = np.ones(total_games, dtype=bool)
    keep_mask[drop_indices] = False
    return games[keep_mask]

def parse_elos(header_bytes):
    """Fast, safe parsing of Elos from accumulated raw header lines using bytes."""
    white_elo = None
    black_elo = None
    for line_bytes in header_bytes:
        if line_bytes.startswith(b'[WhiteElo "'):
            try:
                parts = line_bytes.split(b'"')
                if len(parts) > 1:
                    white_elo = int(parts[1])
            except (IndexError, ValueError):
                pass
        elif line_bytes.startswith(b'[BlackElo "'):
            try:
                parts = line_bytes.split(b'"')
                if len(parts) > 1:
                    black_elo = int(parts[1])
            except (IndexError, ValueError):
                pass
    return white_elo, black_elo

def main():
    parser = argparse.ArgumentParser(description="Filter a PGN file to adjust Elo diff distribution.")
    parser.add_argument("input_pgn", help="Path to input PGN file")
    parser.add_argument("output_pgn", help="Path to output PGN file")
    parser.add_argument("--strategy", choices=["truncation", "probabilistic"], default="probabilistic")
    parser.add_argument("--drop-rate", type=float, default=0.20, help="Max fraction of games to drop (default: 0.20)")
    parser.add_argument("--target-median", type=float, default=73.0, help="Target median for probabilistic matching (default: 73.0)")
    parser.add_argument("--increase-median", action="store_true", help="Increase the median instead of decreasing it")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to read from input (default: all)")
    args = parser.parse_args()

    # Highly memory-efficient array storing 16-bit signed integers (2 bytes per entry)
    games_diff = array.array('h')
    
    with open(args.input_pgn, "rb") as f:
        header_bytes = []
        is_reading_headers = True
        
        pbar_read = tqdm(total=args.max_games, desc="Parsing headers")
        
        for line in f:
            if args.max_games is not None and len(games_diff) >= args.max_games:
                break
                
            if line.startswith(b"[Event "):
                if header_bytes:  # Save the completed previous game
                    white_elo, black_elo = parse_elos(header_bytes)
                    valid = (white_elo is not None and black_elo is not None)
                    diff = abs(white_elo - black_elo) if valid else 0
                    games_diff.append(diff if valid else -1)
                    pbar_read.update(1)
                
                header_bytes = []
                is_reading_headers = True
            
            if is_reading_headers:
                if line == b"\n" or line == b"\r\n":
                    is_reading_headers = False
                elif line.isspace():
                    is_reading_headers = False
                else:
                    header_bytes.append(line)
                    
        # Save the final game if active
        if header_bytes:
            white_elo, black_elo = parse_elos(header_bytes)
            valid = (white_elo is not None and black_elo is not None)
            diff = abs(white_elo - black_elo) if valid else 0
            games_diff.append(diff if valid else -1)
            pbar_read.update(1)
            
        pbar_read.close()

    # Convert the memory-efficient array to NumPy
    games_diff_np = np.frombuffer(games_diff, dtype=np.int16)
    
    valid_mask = games_diff_np >= 0
    valid_indices = np.where(valid_mask)[0].astype(np.uint32)
    valid_diffs = games_diff_np[valid_mask]
    
    # Store only the minimum structured data needed for the active selection
    valid_games = np.empty(len(valid_indices), dtype=[('idx', 'u4'), ('diff', 'i2')])
    valid_games['idx'] = valid_indices
    valid_games['diff'] = valid_diffs
    
    print_metrics("Original Dataset", valid_games["diff"])

    if args.strategy == "truncation":
        filtered_valid = filter_by_truncation(valid_games, max_drop_rate=args.drop_rate, increase_median=args.increase_median)
    else:
        filtered_valid = filter_by_distribution_matching(valid_games, target_median=args.target_median, max_drop_rate=args.drop_rate, increase_median=args.increase_median)

    print_metrics("Filtered Dataset", filtered_valid["diff"])

    # Create the keep mask (True if we want to write the game, False if we discard it)
    keep_mask = np.zeros(len(games_diff_np), dtype=bool)
    keep_mask[~valid_mask] = True  # Keep all invalid games
    keep_mask[filtered_valid['idx']] = True  # Keep selected valid games
    
    keep_mask_len = len(keep_mask)
    
    # Force immediate garbage collection of temporary structures to free RAM before the write step
    del games_diff
    del games_diff_np
    del valid_mask
    del valid_indices
    del valid_diffs
    del valid_games
    del filtered_valid
    import gc
    gc.collect()

    # Read and write sequentially (O(1) memory, avoids seeks)
    with open(args.input_pgn, "rb") as pgn_in, open(args.output_pgn, "wb") as pgn_out:
        current_idx = 0
        game_lines = []
        has_game = False
        
        pbar_write = tqdm(total=keep_mask_len, desc="Writing games")
        
        for line in pgn_in:
            if args.max_games is not None and current_idx >= args.max_games:
                break
                
            if line.startswith(b"[Event "):
                if has_game:  # We completed a game, decide whether to write it
                    if current_idx < keep_mask_len and keep_mask[current_idx]:
                        pgn_out.writelines(game_lines)
                    current_idx += 1
                    pbar_write.update(1)
                
                game_lines = [line]
                has_game = True
            else:
                if has_game:
                    game_lines.append(line)
                    
        # Write the final game of the file
        if has_game and current_idx < keep_mask_len:
            if keep_mask[current_idx]:
                pgn_out.writelines(game_lines)
            pbar_write.update(1)
            
        pbar_write.close()

if __name__ == "__main__":
    main()