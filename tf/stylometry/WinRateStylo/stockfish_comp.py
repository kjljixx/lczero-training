import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import glob
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from stylometry.WinRateStylo.train_wr_stylometry import (
  ScaffoldedViTAndWinRate,
  chessboard_struct_to_lc0_planes,
  parse_position_example,
  MAX_MOVES,
  SEQ_PLANES,
  POS_PLANES,
)
import random
from typing import List
import logging

logger = logging.getLogger(__name__)


def get_tfrecord_paths(paths: List[str]) -> List[str]:
  result = []
  for p in paths:
    if os.path.isfile(p) and p.endswith(".tfrecord"):
      result.append(p)
    elif os.path.isdir(p):
      found = sorted(glob.glob(os.path.join(p, "**", "*.tfrecord"), recursive=True))
      result.extend(found)
  logger.info(f"Found {len(result)} tfrecord files")
  return result


def process_tfrecords(
  tfrecord_paths: List[str],
  model_path: str = "",
  skip_rate: float = 0.0,
  batch_size: int = 32,
):
  model = keras.saving.load_model(
    model_path,
    custom_objects={"ScaffoldedViTAndWinRate": ScaffoldedViTAndWinRate}
  )

  total = 0
  correct_top1 = 0
  correct_top1_no_draw = 0
  total_no_draw = 0
  loss_sum = 0.0
  confusion = np.zeros((3, 3), dtype=np.int64)

  for tfrecord_idx, tfrecord_path in enumerate(tfrecord_paths):
    logger.info(f"Processing {tfrecord_path} ({tfrecord_idx + 1}/{len(tfrecord_paths)})...")
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    pos_batch = []
    clocks_batch = []
    wdl_batch = []

    for raw_record in dataset:
      if random.random() < skip_rate:
        continue

      try:
        stm_seq, opp_seq, full_board, wdl = parse_position_example(raw_record)
      except Exception as e:
        logger.warning(f"Skipping corrupted record: {e}")
        continue

      full_board_np = full_board.numpy().astype(np.uint64)
      pos_planes, pos_clocks = chessboard_struct_to_lc0_planes(full_board_np, short=False)

      pos_batch.append(pos_planes.flatten())
      clocks_batch.append(pos_clocks)
      wdl_batch.append(wdl.numpy())

      if len(pos_batch) >= batch_size:
        results = run_batch(model, pos_batch, clocks_batch, wdl_batch)
        total += results["count"]
        correct_top1 += results["correct_top1"]
        correct_top1_no_draw += results["correct_top1_no_draw"]
        total_no_draw += results["total_no_draw"]
        loss_sum += results["loss_sum"]
        confusion += results["confusion"]
        pos_batch = []
        clocks_batch = []
        wdl_batch = []

    if len(pos_batch) > 0:
      results = run_batch(model, pos_batch, clocks_batch, wdl_batch)
      total += results["count"]
      correct_top1 += results["correct_top1"]
      correct_top1_no_draw += results["correct_top1_no_draw"]
      total_no_draw += results["total_no_draw"]
      loss_sum += results["loss_sum"]
      confusion += results["confusion"]
      pos_batch = []
      clocks_batch = []
      wdl_batch = []

    if total > 0:
      logger.info(
        f"Running — n={total}, acc={correct_top1 / total:.4f}, "
        f"acc_no_draw={correct_top1_no_draw / total_no_draw:.4f if total_no_draw > 0 else 0:.4f}, "
        f"avg_loss={loss_sum / total:.4f}"
      )

  logger.info("=== Final Results ===")
  logger.info(f"Total positions: {total}")
  if total > 0:
    logger.info(f"Top-1 accuracy: {correct_top1 / total:.4f}")
    logger.info(f"Top-1 accuracy (no draw): {correct_top1_no_draw / total_no_draw:.4f}" if total_no_draw > 0 else "")
    logger.info(f"Average loss: {loss_sum / total:.4f}")
    logger.info(f"Confusion matrix [pred][actual] (W/D/L):\n{confusion}")


def run_batch(model, pos_batch, clocks_batch, wdl_batch):
  pos_tensor = tf.cast(np.array(pos_batch), tf.float32)
  clocks_tensor = tf.cast(np.array(clocks_batch), tf.float32)
  wdl_np = np.array(wdl_batch)
  n = len(pos_batch)

  inputs = {
    'input1': tf.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
    'input2': tf.zeros((n, 5, MAX_MOVES, SEQ_PLANES, 8, 8)),
    'mask1': tf.zeros((n, 5, MAX_MOVES)),
    'mask2': tf.zeros((n, 5, MAX_MOVES)),
    'pos': pos_tensor,
    'pos_clocks': clocks_tensor,
  }
  logits = model(inputs, training=False).numpy()

  cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
  losses = cce(wdl_np, logits).numpy()

  pred_idx = np.argmax(logits, axis=-1)
  actual_idx = np.argmax(wdl_np, axis=-1)

  correct_top1 = int(np.sum(pred_idx == actual_idx))

  not_draw = actual_idx != 1
  correct_top1_no_draw = int(np.sum((pred_idx == actual_idx) & not_draw))
  total_no_draw = int(np.sum(not_draw))

  confusion = np.zeros((3, 3), dtype=np.int64)
  for p, a in zip(pred_idx, actual_idx):
    confusion[p, a] += 1

  return {
    "count": n,
    "correct_top1": correct_top1,
    "correct_top1_no_draw": correct_top1_no_draw,
    "total_no_draw": total_no_draw,
    "loss_sum": float(np.sum(losses)),
    "confusion": confusion,
  }


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

  np.random.seed(42)
  random.seed(42)

  parser = argparse.ArgumentParser(description="Evaluate model on position tfrecords")
  parser.add_argument("inputs", nargs="+", help="Input tfrecord file(s) or folder(s)")
  parser.add_argument("model", help="Path to the trained model")
  parser.add_argument("--skip-rate", type=float, default=0.0)
  parser.add_argument("--batch-size", type=int, default=32)

  args = parser.parse_args()

  tfrecord_files = get_tfrecord_paths(args.inputs)

  process_tfrecords(
    tfrecord_paths=tfrecord_files,
    model_path=args.model,
    skip_rate=args.skip_rate,
    batch_size=args.batch_size,
  )