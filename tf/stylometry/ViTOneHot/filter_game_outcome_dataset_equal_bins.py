import tensorflow as tf
import os
import argparse
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_tfrecord_feature_description():
    return {
        'stm_player_seq': tf.io.FixedLenFeature([], tf.string),
        'opp_player_seq': tf.io.FixedLenFeature([], tf.string),
        'stm_player_name': tf.io.FixedLenFeature([], tf.string),
        'stm_player_elo': tf.io.FixedLenFeature([], tf.int64),
        'opp_player_name': tf.io.FixedLenFeature([], tf.string),
        'opp_player_elo': tf.io.FixedLenFeature([], tf.int64),
        'wdl': tf.io.FixedLenFeature([3], tf.float32),
    }

def get_bin_index(elo):
    """Maps ELO rating to its respective bin index.
    Bins are defined in steps of 200 from 1000 up to 2200+.
    """
    if elo < 1000:
        return None
    if elo >= 2200:
        return 6  # Bin for 2200+
    
    # 1000-1199 -> 0, 1200-1399 -> 1, ..., 2000-2199 -> 5
    return int((elo - 1000) // 200)

def get_bin_name(bin_idx):
    """Helper to convert bin index back to a string representation."""
    if bin_idx == 6:
        return "2200+"
    lower = 1000 + bin_idx * 200
    upper = lower + 199
    return f"{lower}-{upper}"

def get_elo_bin_counts(input_files):
    bin_counts = {i: 0 for i in range(7)}
    feature_description = {
        'stm_player_elo': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    logger.info("Counting games per ELO bin...")
    for file_path in tqdm(input_files, desc="Counting Shards"):
        dataset = tf.data.TFRecordDataset(file_path).map(_parse_function)
        for features in dataset:
            elo = features['stm_player_elo'].numpy()
            bin_idx = get_bin_index(elo)
            if bin_idx is not None:
                bin_counts[bin_idx] += 1
    return bin_counts

def filter_dataset(input_files, output_dir, seed=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First pass: Count distribution
    bin_counts = get_elo_bin_counts(input_files)
    
    logger.info("Initial ELO Bin Distribution:")
    for b in range(7):
        logger.info(f"  Bin {get_bin_name(b)}: {bin_counts[b]} games")

    # Find the minimum count among active bins to establish balancing target
    active_counts = [count for count in bin_counts.values() if count > 0]
    if not active_counts:
        logger.error("No valid games found in defined ELO ranges.")
        return

    target_count = min(active_counts)
    logger.info(f"Targeting approximately {target_count} games per active ELO bin.")

    # Calculate retention probability for each bin
    retention_probs = {}
    for b, count in bin_counts.items():
        if count > 0:
            retention_probs[b] = target_count / count
        else:
            retention_probs[b] = 0.0

    # Ensure reproducibility for random downsampling
    random.seed(seed)

    feature_description = get_tfrecord_feature_description()
    total_kept = 0
    total_removed = 0
    written_counts = {i: 0 for i in range(7)}

    logger.info(f"Filtering and saving to {output_dir}...")
    for file_path in tqdm(input_files, desc="Filtering Shards"):
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        raw_dataset = tf.data.TFRecordDataset(file_path)
        
        with tf.io.TFRecordWriter(output_file) as writer:
            for raw_record in raw_dataset:
                features = tf.io.parse_single_example(raw_record, feature_description)
                elo = features['stm_player_elo'].numpy()
                bin_idx = get_bin_index(elo)
                
                if bin_idx is not None:
                    # Probabilistic determination based on the ELO bin's retention target
                    if random.random() < retention_probs[bin_idx]:
                        writer.write(raw_record.numpy())
                        written_counts[bin_idx] += 1
                        total_kept += 1
                    else:
                        total_removed += 1
                else:
                    total_removed += 1

    logger.info(f"Finished. Kept: {total_kept}, Removed: {total_removed}")
    logger.info("Final ELO Bin Distribution:")
    for b in range(7):
        logger.info(f"  Bin {get_bin_name(b)}: {written_counts[b]} games")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset to balance data across ELO bins.")
    parser.add_argument("input_dir", help="Directory containing TFRecord shards (or the seq_shards subfolder)")
    parser.add_argument("output_dir", help="Directory for balanced output shards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable sampling (default: 42)")
    
    args = parser.parse_args()

    # Collect files
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

    filter_dataset(input_files, args.output_dir, seed=args.seed)

if __name__ == "__main__":
    main()