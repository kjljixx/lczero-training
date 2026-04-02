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
[Round "1.2"]
[White "Praggnanandhaa, R"]
[Black "Giri, Anish"]
[Result "1-0"]
[WhiteTitle "GM"]
[BlackTitle "GM"]
[WhiteElo "2741"]
[BlackElo "2753"]
[ECO "B23"]
[Opening "Sicilian"]
[Variation "closed"]
[WhiteFideId "25059530"]
[BlackFideId "24116068"]
[EventDate "2026.03.29"]

1. e4 c5 2. Nc3 d6 3. f4 g6 4. Nf3 Bg7 5. Bb5+ Bd7 6. a4 Nc6 7. O-O a6 8. Bc4 e6
9. d3 Nge7 10. f5 exf5 11. Qe1 Be6 12. exf5 gxf5 13. Bxe6 fxe6 14. Qxe6 Qd7 15.
Qc4 O-O-O 16. Bg5 Rde8 17. Rae1 Ng6 18. Nd5 Kb8 19. Nf6 Bxf6 20. Bxf6 Rhf8 21.
Bc3 f4 22. Rxe8+ Rxe8 23. Re1 Rf8 24. Qe6 Qxe6 25. Rxe6 Kc7 26. Ng5 Kd7 27. Re4
Rf5 28. Nxh7 d5 29. Re2 d4 30. Be1 Nce5 31. Re4 Ng4 32. h3 Ne3 33. c3 Nd5 34. h4
dxc3 35. bxc3 Rh5 36. Ng5 Nf6 37. Re6 Rxg5 38. Rxf6 Rg4 39. h5 Ne5 40. h6 Nf3+
41. Kf1 Nxe1 42. Kxe1 Rxg2 43. Rxf4 Rh2 44. Rf6 Rh3 45. Kd2 Rh2+ 46. Ke3 Rh3+
47. Ke4 c4 48. dxc4 Rxc3 49. Kf5 Ke7 50. Rb6 Rxc4 51. h7 1-0
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
