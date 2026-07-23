import tensorflow as tf
import os
import argparse
import logging
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only parse the name fields to avoid loading the heavy sequences into memory
NAME_FEATURE_DESCRIPTION = {
    'stm_player_name': tf.io.FixedLenFeature([], tf.string),
    'opp_player_name': tf.io.FixedLenFeature([], tf.string),
}

def count_single_shard(file_path):
    """Counts player games in a single shard using batched TF reading."""
    def _parse_batch(example_proto):
        return tf.io.parse_example(example_proto, NAME_FEATURE_DESCRIPTION)

    # Use batching and autotuning for faster parallel I/O and parsing
    dataset = (
        tf.data.TFRecordDataset(file_path, compression_type='GZIP')
        .batch(8192)
        .map(_parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    shard_counts = Counter()
    for batch in dataset:
        # batch['...'].numpy() returns an array of raw bytes.
        # Keeping them as bytes avoids the overhead of UTF-8 decoding.
        shard_counts.update(batch['stm_player_name'].numpy())
        shard_counts.update(batch['opp_player_name'].numpy())
    return shard_counts

def get_player_counts(input_files, max_workers):
    logger.info("Counting games per player...")
    player_counts = Counter()
    
    # Process shards in parallel. TensorFlow's C++ decompression/parsing 
    # releases the GIL, allowing threads to run efficiently in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(count_single_shard, input_files), 
            total=len(input_files), 
            desc="Counting Shards"
        ))
        
    for shard_counts in results:
        player_counts.update(shard_counts)
        
    return player_counts

def filter_single_shard(args):
    """Filters a single shard and writes the output."""
    file_path, output_dir, valid_players = args
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    
    def _parse_batch(raw_records):
        # Only extract the names to decide if the record is kept.
        parsed = tf.io.parse_example(raw_records, NAME_FEATURE_DESCRIPTION)
        return raw_records, parsed['stm_player_name'], parsed['opp_player_name']

    dataset = (
        tf.data.TFRecordDataset(file_path, compression_type='GZIP')
        .batch(8192)
        .map(_parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    options = tf.io.TFRecordOptions(compression_type='GZIP')
    kept = 0
    removed = 0
    
    with tf.io.TFRecordWriter(output_file, options=options) as writer:
        for raw_batch, stm_batch, opp_batch in dataset:
            raw_numpy = raw_batch.numpy()
            stm_numpy = stm_batch.numpy()
            opp_numpy = opp_batch.numpy()
            
            for raw_rec, stm, opp in zip(raw_numpy, stm_numpy, opp_numpy):
                # valid_players contains raw bytes, matching stm and opp types
                if stm in valid_players and opp in valid_players:
                    writer.write(raw_rec)
                    kept += 1
                else:
                    removed += 1
                    
    return kept, removed

def filter_dataset(input_files, output_dir, min_games=5, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First pass: Count appearances
    player_counts = get_player_counts(input_files, max_workers)
    
    # Store valid players as a set of bytes
    valid_players = {p for p, count in player_counts.items() if count >= min_games}
    logger.info(f"Found {len(player_counts)} players. {len(valid_players)} have >= {min_games} games.")

    logger.info(f"Filtering and saving to {output_dir}...")
    
    tasks = [(file_path, output_dir, valid_players) for file_path in input_files]
    total_kept = 0
    total_removed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(filter_single_shard, tasks), 
            total=len(input_files), 
            desc="Filtering Shards"
        ))
        
    for kept, removed in results:
        total_kept += kept
        total_removed += removed

    logger.info(f"Finished. Kept: {total_kept}, Removed: {total_removed}")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset to exclude players with < N non-sequence games.")
    parser.add_argument("input_dir", help="Directory containing TFRecord shards")
    parser.add_argument("output_dir", help="Directory for filtered shards")
    parser.add_argument("--min-games", type=int, default=5, help="Min games per player (default: 5)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2), 
                        help="Number of parallel worker threads for shard processing")
    
    args = parser.parse_args()

    input_files = []
    search_dirs = [args.input_dir, os.path.join(args.input_dir, "seq_shards")]
    for d in search_dirs:
        if os.path.isdir(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.tfrecord')]
            if files:
                input_files = sorted(files)
                logger.info(f"Using shards from: {d}")
                break
    
    if not input_files:
        logger.error("No .tfrecord files found.")
        return

    filter_dataset(input_files, args.output_dir, min_games=args.min_games, max_workers=args.workers)

if __name__ == "__main__":
    main()