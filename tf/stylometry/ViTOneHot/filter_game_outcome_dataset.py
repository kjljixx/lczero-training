import tensorflow as tf
import os
import argparse
import logging
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

def get_player_counts(input_files):
    player_counts = {}
    feature_description = {
        'stm_player_name': tf.io.FixedLenFeature([], tf.string),
        'opp_player_name': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    logger.info("Counting games per player...")
    for file_path in tqdm(input_files, desc="Shards"):
        dataset = tf.data.TFRecordDataset(file_path).map(_parse_function)
        for features in dataset:
            stm = features['stm_player_name'].numpy().decode('utf-8')
            opp = features['opp_player_name'].numpy().decode('utf-8')
            player_counts[stm] = player_counts.get(stm, 0) + 1
            player_counts[opp] = player_counts.get(opp, 0) + 1
    return player_counts

def filter_dataset(input_files, output_dir, min_games=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First pass: Count appearances
    player_counts = get_player_counts(input_files)
    
    # Optional: Iterative filtering to ensure all players in final set have >= min_games
    # But the prompt says "exclude all datapoints from all players who have played less than 5"
    # which usually means based on the initial dataset.
    # However, to be safe and thorough, let's consider if we should remove games where 
    # EITHER player has < 5 games.
    
    valid_players = {p for p, count in player_counts.items() if count >= min_games}
    logger.info(f"Found {len(player_counts)} players. {len(valid_players)} have >= {min_games} games.")

    feature_description = get_tfrecord_feature_description()
    total_kept = 0
    total_removed = 0

    logger.info(f"Filtering and saving to {output_dir}...")
    for file_path in tqdm(input_files, desc="Filtering Shards"):
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        raw_dataset = tf.data.TFRecordDataset(file_path)
        
        with tf.io.TFRecordWriter(output_file) as writer:
            for raw_record in raw_dataset:
                features = tf.io.parse_single_example(raw_record, feature_description)
                stm = features['stm_player_name'].numpy().decode('utf-8')
                opp = features['opp_player_name'].numpy().decode('utf-8')
                
                if stm in valid_players and opp in valid_players:
                    writer.write(raw_record.numpy())
                    total_kept += 1
                else:
                    total_removed += 1

    logger.info(f"Finished. Kept: {total_kept}, Removed: {total_removed}")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset to exclude players with < N non-sequence games.")
    parser.add_argument("input_dir", help="Directory containing TFRecord shards (or the seq_shards subfolder)")
    parser.add_argument("output_dir", help="Directory for filtered shards")
    parser.add_argument("--min-games", type=int, default=5, help="Min games per player (default: 5)")
    
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

    filter_dataset(input_files, args.output_dir, min_games=args.min_games)

if __name__ == "__main__":
    main()
