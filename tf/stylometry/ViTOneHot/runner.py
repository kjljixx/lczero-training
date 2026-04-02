import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess.pgn
import numpy as np

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import tensorflow as tf


MODULE_DIR = Path(__file__).resolve().parent
TF_ROOT = MODULE_DIR.parents[1]
if str(TF_ROOT) not in sys.path:
	sys.path.insert(0, str(TF_ROOT))

from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT
from stylometry.ViTOneHot.pgn_to_game_outcome_training_data import PlayerIndexMapper, extract_game_data
from stylometry.ViTOneHot.train_stylometry import (
	EloPredictor,
	GameOutcomePredictor,
	MAX_MOVES,
	NUM_GAMES,
	SEQ_PLANES,
	chessboard_struct_to_lc0_planes,
)


CHECKPOINT_PATH = Path("stylometry/ViTOneHot/models/run2026-03-27-det-val/checkpoint_epoch_48.keras")

PGN_TEXT = """
[Event "FIDE Candidates 2026"]
[Site "Pegeia CYP"]
[Date "2026.03.29"]
[Round "1.1"]
[White "Caruana, Fabiano"]
[Black "Nakamura, Hikaru"]
[Result "1-0"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2795"]
[BlackElo "2810"]
[ECO "A13"]
[Opening "English opening"]
[Variation "Agincourt variation"]
[WhiteFideId "2020009"]
[BlackFideId "2016192"]
[EventDate "2026.03.29"]

1. Nf3 d5 2. c4 e6 3. g3 d4 4. Bg2 Nc6 5. O-O Bc5 6. e3 Nge7 7. Nxd4 Nxd4 8. b4
Bxb4 9. exd4 O-O 10. Qb3 Ba5 11. Nc3 Nf5 12. Ba3 Re8 13. d5 Nd4 14. Qa4 b6 15.
Rae1 Bd7 16. Qd1 c5 17. Bb2 Rb8 18. a4 a6 19. dxe6 Bxe6 20. Nd5 Qd6 21. Bxd4
cxd4 22. Re4 Bxd5 23. Rxe8+ Rxe8 24. Bxd5 Bb4 25. h4 a5 26. d3 Qf6 27. Kg2 Qe5
28. Qf3 Qf6 29. Qg4 Bc5 30. h5 h6 31. Rh1 Qg5 32. Qd1 Qe7 33. Bc6 Rc8 34. Re1
Qc7 35. Bd5 Kf8 36. Qg4 Rd8 37. Qe4 Kg8 38. Qf5 Qd7 39. Qf3 Rf8 40. Re5 Bd6 41.
Rf5 Qe7 42. Qg4 Be5 43. Rf3 Bf6 44. Rf4 Qd8 45. Be4 Re8 46. Rf5 Qd7 47. Qf4 Bg5
48. Qf3 Qc7 49. Rxf7 Qxf7 50. Bd5 Re6 51. Qg4 Kf8 52. Bxe6 Qe8 53. Bd7 Qa8+ 54.
Kg1 Bf6 55. Qe6 Qd8 56. Bc6 Qe7 57. Qc8+ Qd8 58. Qb7 Be5 59. Bd5 Qc7 60. Qa8+
Ke7 61. Qg8 Kd6 62. Be4 Ke7 63. Bg6 Bf6 64. Qf7+ Kd6 65. Qd5+ Ke7 66. Bf5 Be5
67. f4 Bf6 68. Kg2 Qd6 69. Qb7+ Kf8 70. Kf3 Qe7 71. Qe4 Qxe4+ 72. dxe4 Be7 73.
e5 Bb4 74. Bd3 Be1 75. g4 Ke7 76. Ke4 Bg3 77. f5 Kd7 78. Kd5 Bh4 79. f6 gxf6 80.
e6+ Ke7 81. Kc6 Kxe6 82. Kxb6 Be1 83. c5 1-0
""".strip()


def parse_first_game_from_pgn(pgn_text: str) -> chess.pgn.Game:
	pgn_stream = io.StringIO(pgn_text)
	first_game = chess.pgn.read_game(pgn_stream)
	if first_game is None:
		raise ValueError('No game found in PGN_TEXT.')
	return first_game


def convert_sequence_to_planes_and_mask(
	sequence_structs: List[np.ndarray],
	max_moves: int,
) -> Tuple[np.ndarray, np.ndarray]:
	sequence_planes = np.zeros((max_moves, SEQ_PLANES, 8, 8), dtype=np.float32)
	sequence_mask = np.zeros((max_moves,), dtype=np.float32)

	usable_moves = min(len(sequence_structs), max_moves)
	for move_index in range(usable_moves):
		board_struct = np.asarray(sequence_structs[move_index], dtype=np.uint64).reshape(-1)
		if board_struct.size < 5:
			continue

		board_struct = board_struct[:5]
		if not np.any(board_struct > 0):
			continue

		planes, _ = chessboard_struct_to_lc0_planes(board_struct, short=True)
		sequence_planes[move_index] = planes.astype(np.float32)
		sequence_mask[move_index] = 1.0

	return sequence_planes, sequence_mask


def build_model_inputs_from_first_game(first_game: chess.pgn.Game) -> Dict[str, np.ndarray]:
	player_mapper = PlayerIndexMapper()
	game_sequences, _ = extract_game_data(first_game, player_mapper, max_moves=MAX_MOVES)

	if len(game_sequences) < 2:
		raise ValueError('First game did not produce both white and black sequence entries.')

	white_sequence_structs = game_sequences[0][0]
	black_sequence_structs = game_sequences[1][0]
	if len(white_sequence_structs) == 0:
		raise ValueError('First game has no usable white sequence moves for inference.')
	if len(black_sequence_structs) == 0:
		raise ValueError('First game has no usable black sequence moves for inference.')

	white_planes, white_mask = convert_sequence_to_planes_and_mask(white_sequence_structs, MAX_MOVES)
	black_planes, black_mask = convert_sequence_to_planes_and_mask(black_sequence_structs, MAX_MOVES)

	seq0 = np.zeros((1, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
	seq1 = np.zeros((1, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8), dtype=np.float32)
	mask0 = np.zeros((1, NUM_GAMES, MAX_MOVES), dtype=np.float32)
	mask1 = np.zeros((1, NUM_GAMES, MAX_MOVES), dtype=np.float32)

	# Slot 0 corresponds to the first game for each player.
	seq0[0, 0] = white_planes
	mask0[0, 0] = white_mask
	seq1[0, 0] = black_planes
	mask1[0, 0] = black_mask

	return {
		'seq0': seq0,
		'seq1': seq1,
		'mask0': mask0,
		'mask1': mask1,
	}


def load_game_outcome_predictor(checkpoint_path: Path) -> Any:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

	model = tf.keras.models.load_model(
		str(checkpoint_path),
		custom_objects={
			'GameOutcomePredictor': GameOutcomePredictor,
			'EloPredictor': EloPredictor,
			'GameAggregateViT': GameAggregateViT,
		},
	)
	return model


def to_python_value(value: Any) -> Any:
	if isinstance(value, dict):
		return {key: to_python_value(item) for key, item in value.items()}
	if isinstance(value, tf.Tensor):
		return to_python_value(np.asarray(value))
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def main() -> None:
	first_game = parse_first_game_from_pgn(PGN_TEXT)
	white_player = first_game.headers.get('White', 'Unknown_White')
	black_player = first_game.headers.get('Black', 'Unknown_Black')

	model_inputs = build_model_inputs_from_first_game(first_game)
	expected_seq_shape = (1, NUM_GAMES, MAX_MOVES, SEQ_PLANES, 8, 8)
	expected_mask_shape = (1, NUM_GAMES, MAX_MOVES)
	if model_inputs['seq0'].shape != expected_seq_shape or model_inputs['seq1'].shape != expected_seq_shape:
		raise ValueError(f'Unexpected sequence shape: seq0={model_inputs["seq0"].shape}, seq1={model_inputs["seq1"].shape}')
	if model_inputs['mask0'].shape != expected_mask_shape or model_inputs['mask1'].shape != expected_mask_shape:
		raise ValueError(f'Unexpected mask shape: mask0={model_inputs["mask0"].shape}, mask1={model_inputs["mask1"].shape}')

	predictor = load_game_outcome_predictor(CHECKPOINT_PATH)
	prediction_output = predictor(model_inputs, training=False)
	print(json.dumps({
		'white_player': white_player,
		'black_player': black_player,
		'output': to_python_value(prediction_output),
	}, indent=2))


if __name__ == '__main__':
	main()
