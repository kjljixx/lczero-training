import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import tensorflow as tf


MODULE_DIR = Path(__file__).resolve().parent
TF_ROOT = MODULE_DIR.parents[1]
if str(TF_ROOT) not in sys.path:
	sys.path.insert(0, str(TF_ROOT))

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT
from stylometry.ViTOneHot.train_stylometry import (
	EloPredictor,
	GameOutcomePredictor,
	MAX_MOVES,
	NUM_GAMES,
	SEQ_PLANES,
	create_seq_dataset,
)


BIN_LABELS = [
	'<1100',
	'1100-1200',
	'1200-1300',
	'1300-1400',
	'1400-1500',
	'1500-1600',
	'1600-1700',
	'1700-1800',
	'1800-1900',
	'>1900',
]
BIN_COUNT = len(BIN_LABELS)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Evaluate Elo bin accuracy from seq_shards TFRecords.'
	)
	parser.add_argument('model_path', help='Path to model .keras file')
	parser.add_argument('data_dir', help='Directory containing seq_shards/*.tfrecord')
	parser.add_argument('--batch-size', type=int, default=8)
	parser.add_argument('--max-batches', type=int, default=0,
											help='Optional batch limit for quick runs (0 means all).')
	parser.add_argument('--json-out', type=str, default='',
											help='Optional output path for JSON summary.')
	parser.add_argument('--progress-every', type=int, default=50,
											help='Print progress every N batches.')
	return parser.parse_args()


def find_shards(data_dir: str) -> List[str]:
	seq_dir = os.path.join(data_dir, 'seq_shards')
	seq_paths = sorted(glob.glob(os.path.join(seq_dir, '*.tfrecord')))
	if seq_paths:
		return seq_paths
	fallback_paths = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord')))
	if fallback_paths:
		return fallback_paths
	raise FileNotFoundError(
		f'No TFRecord files found in {seq_dir} or {data_dir}.'
	)


def load_model(model_path: str) -> tf.keras.Model:
	if not os.path.exists(model_path):
		raise FileNotFoundError(f'Model path does not exist: {model_path}')
	loaded = tf.keras.models.load_model(
		model_path,
		custom_objects={
			'GameOutcomePredictor': GameOutcomePredictor,
			'EloPredictor': EloPredictor,
			'GameAggregateViT': GameAggregateViT,
		},
	)
	if loaded is None:
		raise ValueError(f'Failed to load model from {model_path}.')
	return cast(tf.keras.Model, loaded)


def detect_model_kind(model: tf.keras.Model) -> Tuple[str, EloPredictor]:
	if hasattr(model, 'elo_predictor') and callable(getattr(model, 'elo_predictor')):
		elo_predictor = getattr(model, 'elo_predictor')
		if hasattr(elo_predictor, 'vit') and hasattr(elo_predictor, 'regression_head'):
			return 'game_outcome_predictor', cast(EloPredictor, elo_predictor)
	if hasattr(model, 'vit') and hasattr(model, 'regression_head'):
		return 'elo_predictor', cast(EloPredictor, model)
	raise ValueError(
		'Unsupported model type. Expected GameOutcomePredictor or EloPredictor-compatible model.'
	)


def _predict_player_elos(
	elo_predictor: EloPredictor,
	seq: tf.Tensor,
	mask: Optional[tf.Tensor],
) -> tf.Tensor:
	batch_size = tf.shape(seq)[0]
	flat_seq = tf.reshape(
		seq,
		[batch_size * NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8],
	)
	flat_mask = None
	if mask is not None:
		flat_mask = tf.reshape(mask, [-1, MAX_MOVES])

	game_elo = elo_predictor({'seq': flat_seq, 'mask': flat_mask}, training=False)
	game_elo = tf.reshape(game_elo, [batch_size, NUM_GAMES])

	if mask is None:
		return tf.reduce_mean(game_elo, axis=-1)

	mask_positive = tf.math.greater(mask, tf.zeros_like(mask))
	game_valid = tf.cast(tf.reduce_any(mask_positive, axis=-1), dtype=game_elo.dtype)
	return tf.math.divide_no_nan(
		tf.reduce_sum(game_elo * game_valid, axis=-1),
		tf.reduce_sum(game_valid, axis=-1),
	)


def elo_to_bin_index(elo: float) -> int:
	if elo < 1100.0:
		return 0
	if elo > 1900.0:
		return 9
	idx = int((elo - 1100.0) // 100.0) + 1
	return max(1, min(8, idx))


def bin_all(elos: Iterable[float]) -> np.ndarray:
	return np.asarray([elo_to_bin_index(float(v)) for v in elos], dtype=np.int32)


def compute_confusion(actual_bins: np.ndarray, pred_bins: np.ndarray) -> np.ndarray:
	confusion = np.zeros((BIN_COUNT, BIN_COUNT), dtype=np.int64)
	for a, p in zip(actual_bins, pred_bins):
		confusion[int(a), int(p)] += 1
	return confusion


def per_bin_accuracy(confusion: np.ndarray) -> np.ndarray:
	correct = np.diag(confusion).astype(np.float64)
	totals = confusion.sum(axis=1).astype(np.float64)
	with np.errstate(divide='ignore', invalid='ignore'):
		acc = np.divide(correct, totals, out=np.full_like(correct, np.nan), where=totals > 0)
	return acc


def format_confusion(confusion: np.ndarray) -> str:
	header_cells = ['actual\\pred'] + BIN_LABELS
	lines = ['\t'.join(header_cells)]
	for row_idx, row in enumerate(confusion):
		row_cells = [BIN_LABELS[row_idx]] + [str(int(v)) for v in row]
		lines.append('\t'.join(row_cells))
	return '\n'.join(lines)


def format_per_bin(acc_by_bin: np.ndarray, counts: np.ndarray) -> str:
	lines = ['bin\tcount\taccuracy']
	for idx, label in enumerate(BIN_LABELS):
		acc = acc_by_bin[idx]
		acc_str = 'nan' if np.isnan(acc) else f'{acc:.4f}'
		lines.append(f'{label}\t{int(counts[idx])}\t{acc_str}')
	return '\n'.join(lines)


def evaluate(
	elo_predictor: EloPredictor,
	shard_paths: Sequence[str],
	batch_size: int,
	max_batches: int,
	progress_every: int,
) -> Dict[str, object]:
	dataset = create_seq_dataset(
		list(shard_paths),
		batch_size=batch_size,
		shuffle=False,
		repeat=False,
		skip_rate=0.0,
	)

	predicted_elos: List[float] = []
	actual_elos: List[float] = []

	for batch_idx, (inputs, _labels) in enumerate(dataset):
		if max_batches > 0 and batch_idx >= max_batches:
			break

		model_inputs = {
			'seq0': inputs['seq0'],
			'seq1': inputs['seq1'],
			'mask0': inputs['mask0'],
			'mask1': inputs['mask1'],
		}

		pred_e0 = _predict_player_elos(elo_predictor, model_inputs['seq0'], model_inputs['mask0'])
		pred_e1 = _predict_player_elos(elo_predictor, model_inputs['seq1'], model_inputs['mask1'])

		actual_e0 = tf.cast(inputs['stm_player_elo'], tf.float32)
		actual_e1 = tf.cast(inputs['opp_player_elo'], tf.float32)

		pred_e0_np = np.asarray(pred_e0, dtype=np.float64).reshape(-1)
		pred_e1_np = np.asarray(pred_e1, dtype=np.float64).reshape(-1)
		actual_e0_np = np.asarray(actual_e0, dtype=np.float64).reshape(-1)
		actual_e1_np = np.asarray(actual_e1, dtype=np.float64).reshape(-1)

		predicted_elos.extend(pred_e0_np.tolist())
		predicted_elos.extend(pred_e1_np.tolist())
		actual_elos.extend(actual_e0_np.tolist())
		actual_elos.extend(actual_e1_np.tolist())

		if progress_every > 0 and (batch_idx + 1) % progress_every == 0:
			running_actual = np.asarray(actual_elos, dtype=np.float64)
			running_pred = np.asarray(predicted_elos, dtype=np.float64)

			running_shift = float(np.mean(running_actual) - np.mean(running_pred))
			running_pred_norm = running_pred + running_shift

			running_actual_bins = bin_all(running_actual)
			running_pred_bins = bin_all(running_pred_norm)

			running_confusion = compute_confusion(running_actual_bins, running_pred_bins)
			running_bin_acc = per_bin_accuracy(running_confusion)
			running_counts = running_confusion.sum(axis=1)
			running_overall = float(np.mean(running_actual_bins == running_pred_bins))

			print(
				f'Processed {batch_idx + 1} batches ({len(actual_elos)} player-samples). '
				f'shift={running_shift:.3f} overall_bin_acc={running_overall:.4f}'
			)
			print('Running per-bin accuracy (actual bin denominator):')
			print(format_per_bin(running_bin_acc, running_counts))
			print('Running confusion matrix (rows=actual, cols=predicted):')
			print(format_confusion(running_confusion))

	if not actual_elos:
		raise ValueError('No evaluation samples were produced from the provided TFRecords.')

	actual_arr = np.asarray(actual_elos, dtype=np.float64)
	pred_arr = np.asarray(predicted_elos, dtype=np.float64)

	actual_mean = float(np.mean(actual_arr))
	pred_mean = float(np.mean(pred_arr))
	mean_shift = actual_mean - pred_mean
	norm_pred_arr = pred_arr + mean_shift
	norm_pred_mean = float(np.mean(norm_pred_arr))

	actual_bins = bin_all(actual_arr)
	pred_bins = bin_all(norm_pred_arr)

	confusion = compute_confusion(actual_bins, pred_bins)
	bin_acc = per_bin_accuracy(confusion)
	counts = confusion.sum(axis=1)

	overall_accuracy = float(np.mean(actual_bins == pred_bins))

	return {
		'sample_count': int(actual_arr.shape[0]),
		'actual_mean_elo': actual_mean,
		'raw_pred_mean_elo': pred_mean,
		'applied_mean_shift': float(mean_shift),
		'normalized_pred_mean_elo': norm_pred_mean,
		'overall_bin_accuracy': overall_accuracy,
		'per_bin_accuracy': {
			BIN_LABELS[i]: None if np.isnan(bin_acc[i]) else float(bin_acc[i])
			for i in range(BIN_COUNT)
		},
		'per_bin_counts': {
			BIN_LABELS[i]: int(counts[i])
			for i in range(BIN_COUNT)
		},
		'confusion_matrix_labels': BIN_LABELS,
		'confusion_matrix': confusion.tolist(),
	}


def print_summary(model_kind: str, shard_count: int, result: Dict[str, object]) -> None:
	print('=== Elo Bin Accuracy Evaluation ===')
	print(f'Model kind: {model_kind}')
	print(f'Shard count: {shard_count}')
	print(f"Player samples: {result['sample_count']}")
	print(f"Actual mean Elo: {result['actual_mean_elo']:.3f}")
	print(f"Raw predicted mean Elo: {result['raw_pred_mean_elo']:.3f}")
	print(f"Applied mean shift: {result['applied_mean_shift']:.3f}")
	print(f"Normalized predicted mean Elo: {result['normalized_pred_mean_elo']:.3f}")
	print(f"Overall bin accuracy: {result['overall_bin_accuracy']:.4f}")

	confusion = np.asarray(result['confusion_matrix'], dtype=np.int64)
	acc_by_bin = per_bin_accuracy(confusion)
	counts = confusion.sum(axis=1)

	print('\nPer-bin accuracy (actual bin denominator):')
	print(format_per_bin(acc_by_bin, counts))

	print('\nConfusion matrix (rows=actual, cols=predicted):')
	print(format_confusion(confusion))


def save_json(path: str, payload: Dict[str, object]) -> None:
	with open(path, 'w', encoding='utf-8') as handle:
		json.dump(payload, handle, indent=2)
	print(f'Wrote JSON summary to {path}')


def main() -> None:
	args = parse_args()
	shard_paths = find_shards(args.data_dir)
	loaded_model = load_model(args.model_path)
	model_kind, elo_predictor = detect_model_kind(loaded_model)

	print(f'Loaded model from: {args.model_path}')
	print(f'Detected model type: {model_kind}')
	if model_kind == 'game_outcome_predictor':
		print('Using extracted inner Elo predictor for evaluation.')

	result = evaluate(
		elo_predictor=elo_predictor,
		shard_paths=shard_paths,
		batch_size=args.batch_size,
		max_batches=args.max_batches,
		progress_every=args.progress_every,
	)

	print_summary(model_kind, len(shard_paths), result)

	if args.json_out:
		save_json(args.json_out, {
			'model_path': args.model_path,
			'data_dir': args.data_dir,
			'model_kind': model_kind,
			**result,
		})


if __name__ == '__main__':
	main()
