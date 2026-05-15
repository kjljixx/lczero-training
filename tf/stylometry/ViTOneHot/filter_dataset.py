import numpy as np
import chess.pgn
import argparse
import os
from tqdm import tqdm

def calculate_metrics(elo_diffs):
    """Calculate mean and quartiles for the given list of absolute Elo differences."""
    if not elo_diffs:
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
    
    sorted_games = sorted(games, key=lambda g: g['diff'], reverse=increase_median)
    filtered_games = sorted_games[:keep_count]
    
    # Sort back by original index to keep original PGN order
    filtered_games = sorted(filtered_games, key=lambda g: g['idx'])
    return filtered_games

def filter_by_distribution_matching(games, target_median=73.0, max_drop_rate=0.20, increase_median=False):
    total_games = len(games)
    drop_budget = int(total_games * max_drop_rate)
    
    diffs = np.array([g['diff'] for g in games])
    if increase_median:
        penalties = np.maximum(0, target_median - diffs)
    else:
        penalties = np.maximum(0, diffs - target_median)
    
    if penalties.sum() == 0:
        return games
        
    drop_probabilities = penalties / penalties.sum()
    drop_indices = set(np.random.choice(total_games, size=drop_budget, replace=False, p=drop_probabilities))
    
    return [g for i, g in enumerate(games) if i not in drop_indices]

def get_elo(headers, color):
    elo_str = headers.get(f"{color}Elo", "?")
    try:
        return int(elo_str)
    except ValueError:
        return None

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

    games_info = []
    
    # Parse games and record file offsets
    with open(args.input_pgn, "r") as pgn:
        idx = 0
        pbar_read = tqdm(total=args.max_games, desc="Parsing headers")
        while True:
            if args.max_games is not None and idx >= args.max_games:
                break
                
            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break
                
            next_offset = pgn.tell()
                
            white_elo = get_elo(headers, "White")
            black_elo = get_elo(headers, "Black")
            
            if white_elo is not None and black_elo is not None:
                diff = abs(white_elo - black_elo)
                games_info.append({"idx": idx, "offset": offset, "length": next_offset - offset, "diff": diff, "valid": True})
            else:
                # Keep games without Elo but don't count them towards diff filtering
                games_info.append({"idx": idx, "offset": offset, "length": next_offset - offset, "diff": 0, "valid": False})
            idx += 1
            pbar_read.update(1)
            
        pbar_read.close()

    valid_games = [g for g in games_info if g["valid"]]
    print_metrics("Original Dataset", [g["diff"] for g in valid_games])

    if args.strategy == "truncation":
        filtered_valid = filter_by_truncation(valid_games, max_drop_rate=args.drop_rate, increase_median=args.increase_median)
    else:
        filtered_valid = filter_by_distribution_matching(valid_games, target_median=args.target_median, max_drop_rate=args.drop_rate, increase_median=args.increase_median)

    print_metrics("Filtered Dataset", [g["diff"] for g in filtered_valid])

    # Reconstruct the list of games to write
    valid_keep_set = set(g["idx"] for g in filtered_valid)
    
    with open(args.input_pgn, "r") as pgn_in, open(args.output_pgn, "w") as pgn_out:
        for g in tqdm(games_info, desc="Writing games"):
            if not g["valid"] or g["idx"] in valid_keep_set:
                pgn_in.seek(g["offset"])
                chunk = pgn_in.read(g["length"])
                pgn_out.write(chunk)

if __name__ == "__main__":
    main()
# python3 stylometry/ViTOneHot/filter_dataset.py stylometry/WinRateStylo/data/lichess-raw/lichess_db_standard_rated_2017-06.pgn stylometry/ViTOneHot/data/chess/big-filtered.pgn --strategy probabilistic --drop-rate 0.20 --target-median 152 --increase-median