import numpy as np
import argparse
import os
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
        
    drop_probabilities = penalties / penalties.sum()
    drop_indices = np.random.choice(total_games, size=drop_budget, replace=False, p=drop_probabilities)
    
    keep_mask = np.ones(total_games, dtype=bool)
    keep_mask[drop_indices] = False
    return games[keep_mask]

def parse_elos(header_bytes):
    """Fast, safe parsing of Elos from accumulated raw header lines."""
    white_elo = None
    black_elo = None
    for line_bytes in header_bytes:
        line = line_bytes.decode("utf-8", errors="ignore").strip()
        if line.startswith('[WhiteElo "'):
            try:
                white_elo = int(line.split('"')[1])
            except (IndexError, ValueError):
                pass
        elif line.startswith('[BlackElo "'):
            try:
                black_elo = int(line.split('"')[1])
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

    games_list = []
    
    # Parse games line-by-line in binary mode for perfect byte tracking
    with open(args.input_pgn, "rb") as f:
        idx = 0
        game_start_offset = 0
        current_offset = 0
        header_bytes = []
        is_reading_headers = True
        
        pbar_read = tqdm(total=args.max_games, desc="Parsing headers")
        
        while True:
            if args.max_games is not None and len(games_list) >= args.max_games:
                break
                
            line = f.readline()
            if not line:
                # End of file: Save the final game if there is one active
                if current_offset > game_start_offset and header_bytes:
                    white_elo, black_elo = parse_elos(header_bytes)
                    valid = (white_elo is not None and black_elo is not None)
                    diff = abs(white_elo - black_elo) if valid else 0
                    games_list.append((idx, game_start_offset, current_offset - game_start_offset, diff, valid))
                break
            
            # Detect a new game starting
            if line.startswith(b"[Event "):
                if header_bytes:  # Save the previous completed game
                    white_elo, black_elo = parse_elos(header_bytes)
                    valid = (white_elo is not None and black_elo is not None)
                    diff = abs(white_elo - black_elo) if valid else 0
                    games_list.append((idx, game_start_offset, current_offset - game_start_offset, diff, valid))
                    idx += 1
                    pbar_read.update(1)
                
                game_start_offset = current_offset
                header_bytes = []
                is_reading_headers = True
            
            # Collect header metadata lines
            if is_reading_headers:
                if line.strip() == b"":
                    is_reading_headers = False
                else:
                    header_bytes.append(line)
            
            current_offset += len(line)
            
        pbar_read.close()

    # Convert the Python list to a highly memory-efficient structured NumPy array
    dtype = [('idx', 'i4'), ('offset', 'i8'), ('length', 'i4'), ('diff', 'i2'), ('valid', '?')]
    games_info = np.array(games_list, dtype=dtype)
    del games_list  # Explicitly free memory

    valid_games = games_info[games_info["valid"]]
    print_metrics("Original Dataset", valid_games["diff"])

    if args.strategy == "truncation":
        filtered_valid = filter_by_truncation(valid_games, max_drop_rate=args.drop_rate, increase_median=args.increase_median)
    else:
        filtered_valid = filter_by_distribution_matching(valid_games, target_median=args.target_median, max_drop_rate=args.drop_rate, increase_median=args.increase_median)

    print_metrics("Filtered Dataset", filtered_valid["diff"])

    # Convert keep set for O(1) lookups
    valid_keep_set = set(filtered_valid["idx"])
    
    # Read and write in binary mode for guaranteed clean output
    with open(args.input_pgn, "rb") as pgn_in, open(args.output_pgn, "wb") as pgn_out:
        for g in tqdm(games_info, desc="Writing games"):
            if not g["valid"] or g["idx"] in valid_keep_set:
                pgn_in.seek(g["offset"])
                chunk = pgn_in.read(g["length"])
                pgn_out.write(chunk)

if __name__ == "__main__":
    main()