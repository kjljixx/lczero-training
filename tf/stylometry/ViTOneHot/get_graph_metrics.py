import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
import tensorflow as tf


PlayerId = int
EdgeKey = Tuple[PlayerId, PlayerId]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Compute scalable graph metrics from game-outcome TFRecords.'
	)
	parser.add_argument(
		'input_paths',
		nargs='+',
		help='TFRecord file(s), seq_shards dir(s), or output dir(s) containing TFRecords.',
	)
	parser.add_argument(
		'--progress-every',
		type=int,
		default=50000,
		help='Print progress every N records processed.',
	)
	parser.add_argument(
		'--diameter-exact-node-threshold',
		type=int,
		default=1200,
		help='Use exact directed diameter only when largest SCC has <= this many nodes.',
	)
	parser.add_argument(
		'--diameter-samples',
		type=int,
		default=64,
		help='Number of BFS roots for approximate diameter when exact mode is too expensive.',
	)
	parser.add_argument(
		'--top-k',
		type=int,
		default=10,
		help='Top-K players to print in degree/volume tables.',
	)
	parser.add_argument(
		'--flow-source',
		type=str,
		default='',
		help='Optional source player name for max-flow computation.',
	)
	parser.add_argument(
		'--flow-sink',
		type=str,
		default='',
		help='Optional sink player name for max-flow computation.',
	)
	parser.add_argument(
		'--flow-use-wins',
		action='store_true',
		help='Use winner->loser capacities instead of white->black game-count capacities.',
	)
	parser.add_argument(
		'--json-out',
		type=str,
		default='',
		help='Optional output JSON summary path.',
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42,
		help='Random seed for approximate metrics.',
	)
	return parser.parse_args()


def find_tfrecords(input_paths: Sequence[str]) -> List[str]:
	tfrecord_paths: List[str] = []
	for path in input_paths:
		if os.path.isfile(path):
			if path.endswith('.tfrecord'):
				tfrecord_paths.append(path)
			continue

		if not os.path.isdir(path):
			continue

		seq_dir = os.path.join(path, 'seq_shards')
		if os.path.isdir(seq_dir):
			tfrecord_paths.extend(sorted(glob.glob(os.path.join(seq_dir, '*.tfrecord'))))
			continue

		tfrecord_paths.extend(sorted(glob.glob(os.path.join(path, '*.tfrecord'))))

	unique_paths = sorted(set(tfrecord_paths))
	if not unique_paths:
		raise FileNotFoundError(
			'No TFRecord files found in the provided paths. '
			'Expected *.tfrecord files or directories containing seq_shards/*.tfrecord.'
		)
	return unique_paths


def _decode_text_feature(example: tf.train.Example, key: str) -> str:
	values = example.features.feature[key].bytes_list.value
	if not values:
		return ''
	return values[0].decode('utf-8', errors='replace')


def _decode_int_feature(example: tf.train.Example, key: str) -> int:
	values = example.features.feature[key].int64_list.value
	if not values:
		return 0
	return int(values[0])


def _decode_wdl_feature(example: tf.train.Example) -> Tuple[float, float, float]:
	values = example.features.feature['wdl'].float_list.value
	if len(values) != 3:
		return (0.0, 1.0, 0.0)
	return (float(values[0]), float(values[1]), float(values[2]))


def _result_from_wdl(wdl: Tuple[float, float, float]) -> int:
	max_index = int(np.argmax(np.asarray(wdl, dtype=np.float32)))
	if max_index == 0:
		return 1
	if max_index == 2:
		return -1
	return 0


@dataclass
class GraphBuildResult:
	name_to_id: Dict[str, PlayerId]
	id_to_name: List[str]
	edge_games: Dict[EdgeKey, int]
	undirected_games: Dict[Tuple[PlayerId, PlayerId], int]
	win_caps: Dict[EdgeKey, int]
	white_win_count: int
	draw_count: int
	black_win_count: int
	elo_sum: DefaultDict[PlayerId, float]
	elo_count: DefaultDict[PlayerId, int]
	total_records: int


def build_graph_from_tfrecords(
	tfrecord_paths: Sequence[str],
	progress_every: int,
) -> GraphBuildResult:
	name_to_id: Dict[str, PlayerId] = {}
	id_to_name: List[str] = []

	edge_games: DefaultDict[EdgeKey, int] = defaultdict(int)
	undirected_games: DefaultDict[Tuple[PlayerId, PlayerId], int] = defaultdict(int)
	win_caps: DefaultDict[EdgeKey, int] = defaultdict(int)

	white_win_count = 0
	draw_count = 0
	black_win_count = 0
	total_records = 0

	elo_sum: DefaultDict[PlayerId, float] = defaultdict(float)
	elo_count: DefaultDict[PlayerId, int] = defaultdict(int)

	def get_or_create_player_id(player_name: str) -> PlayerId:
		existing = name_to_id.get(player_name)
		if existing is not None:
			return existing
		player_id = len(id_to_name)
		name_to_id[player_name] = player_id
		id_to_name.append(player_name)
		return player_id

	dataset = tf.data.TFRecordDataset(list(tfrecord_paths), num_parallel_reads=tf.data.AUTOTUNE)
	example = tf.train.Example()

	for raw_record in dataset:
		total_records += 1
		example.ParseFromString(bytes(raw_record.numpy()))

		white_name = _decode_text_feature(example, 'stm_player_name')
		black_name = _decode_text_feature(example, 'opp_player_name')
		white_elo = _decode_int_feature(example, 'stm_player_elo')
		black_elo = _decode_int_feature(example, 'opp_player_elo')
		wdl = _decode_wdl_feature(example)
		game_result = _result_from_wdl(wdl)

		white_id = get_or_create_player_id(white_name)
		black_id = get_or_create_player_id(black_name)

		edge_games[(white_id, black_id)] += 1
		undirected_key = (white_id, black_id) if white_id <= black_id else (black_id, white_id)
		undirected_games[undirected_key] += 1

		if game_result > 0:
			white_win_count += 1
			win_caps[(white_id, black_id)] += 1
		elif game_result < 0:
			black_win_count += 1
			win_caps[(black_id, white_id)] += 1
		else:
			draw_count += 1

		if white_elo > 0:
			elo_sum[white_id] += float(white_elo)
			elo_count[white_id] += 1
		if black_elo > 0:
			elo_sum[black_id] += float(black_elo)
			elo_count[black_id] += 1


		if progress_every > 0 and total_records % progress_every == 0:
			# Compute and print all available metrics so far
			num_players = len(id_to_name)
			num_directed_edges = len(edge_games)
			num_undirected_edges = len(undirected_games)
			directed_possible_edges = max(1, num_players * max(0, num_players - 1))
			undirected_possible_edges = max(1, (num_players * max(0, num_players - 1)) // 2)
			edge_weights = np.asarray(list(edge_games.values()), dtype=np.int64)
			# Build graph views
			graph_views = build_graph_views(num_players, edge_games)
			connected_components = connected_components_undirected(graph_views.undirected_adj)
			sccs = strongly_connected_components(graph_views.directed_adj, graph_views.directed_rev)
			largest_component_size = len(connected_components[0]) if connected_components else 0
			largest_scc_nodes = sccs[0] if sccs else []
			largest_scc_size = len(largest_scc_nodes)
			# Diameter (approx, fast)
			diameter_info = {'directed_diameter': 0, 'method': 'skip', 'sample_count': 0, 'avg_shortest_path_estimate': 0.0}
			try:
				diameter_info = directed_diameter_metrics(
					graph_views.directed_adj,
					largest_scc_nodes,
					exact_node_threshold=300,  # Use a small threshold for speed in progress
					diameter_samples=8,
					seed=42,
				)
			except Exception:
				pass
			# Reciprocity
			def compute_reciprocity(edge_games):
				directed_edges = len(edge_games)
				if directed_edges == 0:
					return {'mutual_pairs': 0.0, 'edge_reciprocity': 0.0}
				mutual_pairs = 0
				for src, dst in edge_games:
					if src < dst and (dst, src) in edge_games:
						mutual_pairs += 1
				reciprocal_edges = 2 * mutual_pairs
				edge_reciprocity = reciprocal_edges / directed_edges
				return {'mutual_pairs': float(mutual_pairs), 'edge_reciprocity': float(edge_reciprocity)}
			reciprocity = compute_reciprocity(edge_games)
			# Degree/strength stats
			def summarize_distribution(values):
				if isinstance(values, list):
					values = np.asarray(values, dtype=np.float64)
				if values.size == 0:
					return {'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'max': 0.0}
				return {
					'min': float(np.min(values)),
					'mean': float(np.mean(values)),
					'median': float(np.median(values)),
					'p90': float(np.quantile(values, 0.90)),
					'max': float(np.max(values)),
				}
			# Elo
			known_elo_values = []
			for player_id in range(num_players):
				if elo_count[player_id] > 0:
					known_elo_values.append(elo_sum[player_id] / elo_count[player_id])
			known_elo = np.asarray(known_elo_values, dtype=np.float64)
			# Print summary
			print('\n=== Progress Metrics ===')
			print(f"Records: {total_records}")
			print(f"Players: {num_players}")
			print(f"Directed edges: {num_directed_edges}")
			print(f"Undirected edges: {num_undirected_edges}")
			print(f"Directed density: {num_directed_edges / directed_possible_edges:.8f}")
			print(f"Undirected density: {num_undirected_edges / undirected_possible_edges:.8f}")
			print(f"white_win={white_win_count}, draw={draw_count}, black_win={black_win_count}")
			print(f"Connected components (undirected): {len(connected_components)} | Largest: {largest_component_size}")
			print(f"Strongly connected components: {len(sccs)} | Largest SCC: {largest_scc_size}")
			print(f"Directed diameter (approx): {diameter_info['directed_diameter']} [{diameter_info['method']}, samples={diameter_info['sample_count']}] Avg shortest-path: {diameter_info['avg_shortest_path_estimate']:.4f}")
			print(f"Mutual pairs: {int(reciprocity['mutual_pairs'])} | Edge reciprocity: {reciprocity['edge_reciprocity']:.4f}")
			print(f"Out-degree stats: {json.dumps(summarize_distribution(graph_views.out_degree))}")
			print(f"In-degree stats: {json.dumps(summarize_distribution(graph_views.in_degree))}")
			print(f"Out-strength stats: {json.dumps(summarize_distribution(graph_views.out_strength))}")
			print(f"In-strength stats: {json.dumps(summarize_distribution(graph_views.in_strength))}")
			print(f"Player Elo stats: {json.dumps(summarize_distribution(known_elo))} (n={known_elo.size})")

	return GraphBuildResult(
		name_to_id=name_to_id,
		id_to_name=id_to_name,
		edge_games=dict(edge_games),
		undirected_games=dict(undirected_games),
		win_caps=dict(win_caps),
		white_win_count=white_win_count,
		draw_count=draw_count,
		black_win_count=black_win_count,
		elo_sum=elo_sum,
		elo_count=elo_count,
		total_records=total_records,
	)


@dataclass
class GraphViews:
	directed_adj: List[List[PlayerId]]
	directed_rev: List[List[PlayerId]]
	undirected_adj: List[List[PlayerId]]
	out_strength: np.ndarray
	in_strength: np.ndarray
	out_degree: np.ndarray
	in_degree: np.ndarray


def build_graph_views(num_players: int, edge_games: Dict[EdgeKey, int]) -> GraphViews:
	directed_adj_sets: List[Set[PlayerId]] = [set() for _ in range(num_players)]
	directed_rev_sets: List[Set[PlayerId]] = [set() for _ in range(num_players)]
	undirected_adj_sets: List[Set[PlayerId]] = [set() for _ in range(num_players)]

	out_strength = np.zeros(num_players, dtype=np.int64)
	in_strength = np.zeros(num_players, dtype=np.int64)

	for (src, dst), capacity in edge_games.items():
		directed_adj_sets[src].add(dst)
		directed_rev_sets[dst].add(src)
		undirected_adj_sets[src].add(dst)
		undirected_adj_sets[dst].add(src)
		out_strength[src] += int(capacity)
		in_strength[dst] += int(capacity)

	directed_adj = [sorted(list(neighbors)) for neighbors in directed_adj_sets]
	directed_rev = [sorted(list(neighbors)) for neighbors in directed_rev_sets]
	undirected_adj = [sorted(list(neighbors)) for neighbors in undirected_adj_sets]

	out_degree = np.asarray([len(neighbors) for neighbors in directed_adj], dtype=np.int64)
	in_degree = np.asarray([len(neighbors) for neighbors in directed_rev], dtype=np.int64)

	return GraphViews(
		directed_adj=directed_adj,
		directed_rev=directed_rev,
		undirected_adj=undirected_adj,
		out_strength=out_strength,
		in_strength=in_strength,
		out_degree=out_degree,
		in_degree=in_degree,
	)


def connected_components_undirected(undirected_adj: Sequence[Sequence[int]]) -> List[List[int]]:
	node_count = len(undirected_adj)
	visited = [False] * node_count
	components: List[List[int]] = []

	for start_node in range(node_count):
		if visited[start_node]:
			continue
		queue = deque([start_node])
		visited[start_node] = True
		component_nodes: List[int] = []

		while queue:
			node = queue.popleft()
			component_nodes.append(node)
			for neighbor in undirected_adj[node]:
				if not visited[neighbor]:
					visited[neighbor] = True
					queue.append(neighbor)

		components.append(component_nodes)

	components.sort(key=len, reverse=True)
	return components


def strongly_connected_components(
	directed_adj: Sequence[Sequence[int]],
	directed_rev: Sequence[Sequence[int]],
) -> List[List[int]]:
	node_count = len(directed_adj)
	visited = [False] * node_count
	finish_order: List[int] = []

	for start_node in range(node_count):
		if visited[start_node]:
			continue

		stack: List[Tuple[int, int]] = [(start_node, 0)]
		visited[start_node] = True

		while stack:
			node, next_index = stack[-1]
			if next_index < len(directed_adj[node]):
				neighbor = directed_adj[node][next_index]
				stack[-1] = (node, next_index + 1)
				if not visited[neighbor]:
					visited[neighbor] = True
					stack.append((neighbor, 0))
			else:
				finish_order.append(node)
				stack.pop()

	visited = [False] * node_count
	sccs: List[List[int]] = []

	for start_node in reversed(finish_order):
		if visited[start_node]:
			continue

		stack = [start_node]
		visited[start_node] = True
		component_nodes: List[int] = []

		while stack:
			node = stack.pop()
			component_nodes.append(node)
			for neighbor in directed_rev[node]:
				if not visited[neighbor]:
					visited[neighbor] = True
					stack.append(neighbor)

		sccs.append(component_nodes)

	sccs.sort(key=len, reverse=True)
	return sccs


def bfs_distances(
	start_node: int,
	adjacency: Sequence[Sequence[int]],
) -> List[int]:
	distances = [-1] * len(adjacency)
	distances[start_node] = 0
	queue = deque([start_node])

	while queue:
		node = queue.popleft()
		next_distance = distances[node] + 1
		for neighbor in adjacency[node]:
			if distances[neighbor] == -1:
				distances[neighbor] = next_distance
				queue.append(neighbor)

	return distances


def induced_subgraph(
	directed_adj: Sequence[Sequence[int]],
	nodes: Sequence[int],
) -> List[List[int]]:
	node_to_local = {node: idx for idx, node in enumerate(nodes)}
	local_adj: List[List[int]] = [[] for _ in range(len(nodes))]

	for local_src, global_src in enumerate(nodes):
		row = local_adj[local_src]
		for global_dst in directed_adj[global_src]:
			local_dst = node_to_local.get(global_dst)
			if local_dst is not None:
				row.append(local_dst)

	return local_adj


def directed_diameter_metrics(
	directed_adj: Sequence[Sequence[int]],
	largest_scc_nodes: Sequence[int],
	exact_node_threshold: int,
	diameter_samples: int,
	seed: int,
) -> Dict[str, object]:
	scc_size = len(largest_scc_nodes)
	if scc_size == 0:
		return {
			'directed_diameter': 0,
			'method': 'none',
			'sample_count': 0,
			'avg_shortest_path_estimate': 0.0,
		}
	if scc_size == 1:
		return {
			'directed_diameter': 0,
			'method': 'trivial_singleton',
			'sample_count': 1,
			'avg_shortest_path_estimate': 0.0,
		}

	local_adj = induced_subgraph(directed_adj, largest_scc_nodes)

	if scc_size <= exact_node_threshold:
		best_distance = 0
		distance_sum = 0
		reachable_pairs = 0

		for src in range(scc_size):
			distances = bfs_distances(src, local_adj)
			finite_distances = [d for d in distances if d >= 0]
			if finite_distances:
				best_distance = max(best_distance, max(finite_distances))
			for d in finite_distances:
				if d > 0:
					distance_sum += d
					reachable_pairs += 1

		avg_shortest_path = float(distance_sum / reachable_pairs) if reachable_pairs else 0.0
		return {
			'directed_diameter': int(best_distance),
			'method': 'exact',
			'sample_count': scc_size,
			'avg_shortest_path_estimate': avg_shortest_path,
		}

	rng = random.Random(seed)
	out_degrees = [(idx, len(local_adj[idx])) for idx in range(scc_size)]
	out_degrees.sort(key=lambda pair: pair[1], reverse=True)

	requested_samples = max(2, min(diameter_samples, scc_size))
	top_seed_count = min(requested_samples // 2, scc_size)
	seed_nodes: List[int] = [idx for idx, _ in out_degrees[:top_seed_count]]

	if len(seed_nodes) < requested_samples:
		all_nodes = list(range(scc_size))
		rng.shuffle(all_nodes)
		for node in all_nodes:
			if node not in seed_nodes:
				seed_nodes.append(node)
			if len(seed_nodes) >= requested_samples:
				break

	best_distance = 0
	distance_sum = 0
	reachable_pairs = 0

	for seed_node in seed_nodes:
		distances = bfs_distances(seed_node, local_adj)
		finite = [d for d in distances if d >= 0]
		if not finite:
			continue
		best_seed_distance = max(finite)
		farthest_node = int(np.argmax(np.asarray(distances, dtype=np.int64)))
		best_distance = max(best_distance, best_seed_distance)

		second_distances = bfs_distances(farthest_node, local_adj)
		second_finite = [d for d in second_distances if d >= 0]
		if second_finite:
			best_distance = max(best_distance, max(second_finite))
			for d in second_finite:
				if d > 0:
					distance_sum += d
					reachable_pairs += 1

	avg_shortest_path = float(distance_sum / reachable_pairs) if reachable_pairs else 0.0
	return {
		'directed_diameter': int(best_distance),
		'method': 'approx_double_sweep',
		'sample_count': len(seed_nodes),
		'avg_shortest_path_estimate': avg_shortest_path,
	}


class Dinic:
	def __init__(self, node_count: int):
		self.node_count = node_count
		self.to: List[int] = []
		self.cap: List[int] = []
		self.next_edge: List[int] = []
		self.head = [-1] * node_count

	def add_edge(self, src: int, dst: int, capacity: int) -> None:
		if capacity <= 0:
			return
		self.to.append(dst)
		self.cap.append(capacity)
		self.next_edge.append(self.head[src])
		self.head[src] = len(self.to) - 1

		self.to.append(src)
		self.cap.append(0)
		self.next_edge.append(self.head[dst])
		self.head[dst] = len(self.to) - 1

	def max_flow(self, source: int, sink: int) -> int:
		if source == sink:
			return 0

		total_flow = 0
		level = [-1] * self.node_count

		while self._bfs_level_graph(source, sink, level):
			next_iter = self.head.copy()
			while True:
				pushed = self._dfs_blocking_flow(source, sink, math.inf, next_iter, level)
				if pushed <= 0:
					break
				total_flow += int(pushed)

		return int(total_flow)

	def _bfs_level_graph(self, source: int, sink: int, level: List[int]) -> bool:
		for idx in range(self.node_count):
			level[idx] = -1
		level[source] = 0

		queue = deque([source])
		while queue:
			node = queue.popleft()
			edge_idx = self.head[node]
			while edge_idx != -1:
				neighbor = self.to[edge_idx]
				if self.cap[edge_idx] > 0 and level[neighbor] < 0:
					level[neighbor] = level[node] + 1
					queue.append(neighbor)
				edge_idx = self.next_edge[edge_idx]

		return level[sink] >= 0

	def _dfs_blocking_flow(
		self,
		node: int,
		sink: int,
		flow_limit: float,
		next_iter: List[int],
		level: Sequence[int],
	) -> int:
		if node == sink:
			return int(flow_limit)

		edge_idx = next_iter[node]
		while edge_idx != -1:
			next_iter[node] = self.next_edge[edge_idx]
			neighbor = self.to[edge_idx]

			if self.cap[edge_idx] > 0 and level[neighbor] == level[node] + 1:
				candidate_flow = min(flow_limit, self.cap[edge_idx])
				pushed = self._dfs_blocking_flow(neighbor, sink, candidate_flow, next_iter, level)
				if pushed > 0:
					self.cap[edge_idx] -= pushed
					self.cap[edge_idx ^ 1] += pushed
					return pushed

			edge_idx = next_iter[node]

		return 0


def resolve_player_name_to_id(
	raw_name: str,
	name_to_id: Dict[str, int],
) -> Optional[int]:
	if not raw_name:
		return None
	exact = name_to_id.get(raw_name)
	if exact is not None:
		return exact

	lower_raw_name = raw_name.lower()
	for player_name, player_id in name_to_id.items():
		if player_name.lower() == lower_raw_name:
			return player_id
	return None


def choose_flow_endpoints(
	args: argparse.Namespace,
	id_to_name: Sequence[str],
	name_to_id: Dict[str, int],
	out_strength: np.ndarray,
	in_strength: np.ndarray,
) -> Tuple[Optional[int], Optional[int], str]:
	requested_source = resolve_player_name_to_id(args.flow_source, name_to_id)
	requested_sink = resolve_player_name_to_id(args.flow_sink, name_to_id)

	if args.flow_source and requested_source is None:
		return None, None, f'flow source player not found: {args.flow_source}'
	if args.flow_sink and requested_sink is None:
		return None, None, f'flow sink player not found: {args.flow_sink}'

	if requested_source is not None and requested_sink is not None:
		return requested_source, requested_sink, 'user_provided'

	if len(id_to_name) < 2:
		return None, None, 'not_enough_players'

	if requested_source is None:
		requested_source = int(np.argmax(out_strength))

	if requested_sink is None:
		sorted_sink_candidates = np.argsort(-in_strength)
		sink_candidate = None
		for candidate in sorted_sink_candidates:
			if int(candidate) != requested_source:
				sink_candidate = int(candidate)
				break
		requested_sink = sink_candidate

	if requested_source is None or requested_sink is None:
		return None, None, 'could_not_choose_endpoints'

	return requested_source, requested_sink, 'auto_volume_extremes'


def compute_max_flow(
	node_count: int,
	capacities: Dict[Tuple[int, int], int],
	source: Optional[int],
	sink: Optional[int],
) -> Dict[str, object]:
	if source is None or sink is None:
		return {
			'max_flow': 0,
			'flow_computed': False,
			'reason': 'missing_endpoints',
		}

	dinic = Dinic(node_count)
	for (src, dst), capacity in capacities.items():
		dinic.add_edge(src, dst, int(capacity))

	flow_value = dinic.max_flow(source, sink)
	return {
		'max_flow': int(flow_value),
		'flow_computed': True,
		'reason': 'ok',
	}


def summarize_distribution(values: np.ndarray) -> Dict[str, float]:
	if values.size == 0:
		return {'min': 0.0, 'mean': 0.0, 'median': 0.0, 'p90': 0.0, 'max': 0.0}
	return {
		'min': float(np.min(values)),
		'mean': float(np.mean(values)),
		'median': float(np.median(values)),
		'p90': float(np.quantile(values, 0.90)),
		'max': float(np.max(values)),
	}


def top_players_table(
	metric: np.ndarray,
	id_to_name: Sequence[str],
	top_k: int,
) -> List[Dict[str, object]]:
	if metric.size == 0:
		return []
	top_indices = np.argsort(-metric)[:top_k]
	table: List[Dict[str, object]] = []
	for player_id in top_indices:
		table.append(
			{
				'player': id_to_name[int(player_id)],
				'value': float(metric[int(player_id)]),
			}
		)
	return table


def compute_reciprocity(edge_games: Dict[EdgeKey, int]) -> Dict[str, float]:
	directed_edges = len(edge_games)
	if directed_edges == 0:
		return {'mutual_pairs': 0.0, 'edge_reciprocity': 0.0}

	mutual_pairs = 0
	for src, dst in edge_games:
		if src < dst and (dst, src) in edge_games:
			mutual_pairs += 1

	reciprocal_edges = 2 * mutual_pairs
	edge_reciprocity = reciprocal_edges / directed_edges
	return {
		'mutual_pairs': float(mutual_pairs),
		'edge_reciprocity': float(edge_reciprocity),
	}


def print_summary(metrics: Dict[str, object]) -> None:
	print('=== Graph Summary ===')
	print(f"Records: {metrics['total_records']}")
	print(f"Players: {metrics['num_players']}")
	print(f"Directed edges: {metrics['num_directed_edges']}")
	print(f"Undirected edges: {metrics['num_undirected_edges']}")
	print(f"Directed density: {metrics['directed_density']:.8f}")
	print(f"Undirected density: {metrics['undirected_density']:.8f}")

	print('\n=== Outcome Distribution (white POV) ===')
	print(
		f"white_win={metrics['white_win_count']} ({metrics['white_win_pct']:.2f}%), "
		f"draw={metrics['draw_count']} ({metrics['draw_pct']:.2f}%), "
		f"black_win={metrics['black_win_count']} ({metrics['black_win_pct']:.2f}%)"
	)

	print('\n=== Connectivity ===')
	print(
		f"Connected components (undirected): {metrics['connected_components']} | "
		f"Largest size: {metrics['largest_component_size']} "
		f"({metrics['largest_component_pct']:.2f}% of players)"
	)
	print(
		f"Strongly connected components (directed): {metrics['strongly_connected_components']} | "
		f"Largest SCC size: {metrics['largest_scc_size']} "
		f"({metrics['largest_scc_pct']:.2f}% of players)"
	)
	print(
		f"Directed diameter (largest SCC): {metrics['directed_diameter']} "
		f"[{metrics['directed_diameter_method']}, samples={metrics['directed_diameter_samples']}]"
	)
	print(
		f"Average shortest-path estimate (largest SCC): "
		f"{metrics['avg_shortest_path_estimate']:.4f}"
	)

	print('\n=== Reciprocity ===')
	print(
		f"Mutual directed pairs: {int(metrics['mutual_pairs'])} | "
		f"Edge reciprocity: {metrics['edge_reciprocity']:.4f}"
	)

	print('\n=== Capacity / Flow ===')
	print(f"Flow capacity basis: {metrics['flow_capacity_basis']}")
	print(
		f"Flow endpoints: source='{metrics['flow_source']}', "
		f"sink='{metrics['flow_sink']}', selection={metrics['flow_endpoint_selection']}"
	)
	print(f"Max flow: {metrics['max_flow']}")

	print('\n=== Degree / Volume Stats ===')
	print(f"Out-degree stats: {json.dumps(metrics['out_degree_distribution'])}")
	print(f"In-degree stats: {json.dumps(metrics['in_degree_distribution'])}")
	print(f"Out-strength stats: {json.dumps(metrics['out_strength_distribution'])}")
	print(f"In-strength stats: {json.dumps(metrics['in_strength_distribution'])}")

	print('\n=== Top Players by Out-Strength ===')
	for row in metrics['top_out_strength']:
		print(f"{row['player']}: {row['value']:.0f}")

	print('\n=== Top Players by In-Strength ===')
	for row in metrics['top_in_strength']:
		print(f"{row['player']}: {row['value']:.0f}")


def main() -> None:
	args = parse_args()
	random.seed(args.seed)
	np.random.seed(args.seed)

	tfrecord_paths = find_tfrecords(args.input_paths)
	print(f'Found {len(tfrecord_paths)} TFRecord file(s).')

	build_result = build_graph_from_tfrecords(tfrecord_paths, progress_every=args.progress_every)
	num_players = len(build_result.id_to_name)
	if num_players == 0:
		raise ValueError('No players found in TFRecords. Cannot compute graph metrics.')

	graph_views = build_graph_views(num_players, build_result.edge_games)
	connected_components = connected_components_undirected(graph_views.undirected_adj)
	sccs = strongly_connected_components(graph_views.directed_adj, graph_views.directed_rev)

	largest_component_size = len(connected_components[0]) if connected_components else 0
	largest_scc_nodes = sccs[0] if sccs else []
	largest_scc_size = len(largest_scc_nodes)

	diameter_info = directed_diameter_metrics(
		graph_views.directed_adj,
		largest_scc_nodes,
		exact_node_threshold=args.diameter_exact_node_threshold,
		diameter_samples=args.diameter_samples,
		seed=args.seed,
	)

	flow_capacities = build_result.win_caps if args.flow_use_wins else build_result.edge_games
	flow_source_id, flow_sink_id, flow_selection = choose_flow_endpoints(
		args=args,
		id_to_name=build_result.id_to_name,
		name_to_id=build_result.name_to_id,
		out_strength=graph_views.out_strength,
		in_strength=graph_views.in_strength,
	)
	flow_info = compute_max_flow(
		node_count=num_players,
		capacities=flow_capacities,
		source=flow_source_id,
		sink=flow_sink_id,
	)

	total_records = build_result.total_records
	white_win_count = build_result.white_win_count
	draw_count = build_result.draw_count
	black_win_count = build_result.black_win_count

	num_directed_edges = len(build_result.edge_games)
	num_undirected_edges = len(build_result.undirected_games)

	directed_possible_edges = max(1, num_players * max(0, num_players - 1))
	undirected_possible_edges = max(1, (num_players * max(0, num_players - 1)) // 2)

	edge_weights = np.asarray(list(build_result.edge_games.values()), dtype=np.int64)

	reciprocity = compute_reciprocity(build_result.edge_games)

	flow_source_name = (
		build_result.id_to_name[flow_source_id] if flow_source_id is not None else ''
	)
	flow_sink_name = (
		build_result.id_to_name[flow_sink_id] if flow_sink_id is not None else ''
	)

	metrics: Dict[str, object] = {
		'total_records': int(total_records),
		'num_players': int(num_players),
		'num_directed_edges': int(num_directed_edges),
		'num_undirected_edges': int(num_undirected_edges),
		'directed_density': float(num_directed_edges / directed_possible_edges),
		'undirected_density': float(num_undirected_edges / undirected_possible_edges),
		'games_per_directed_edge': summarize_distribution(edge_weights.astype(np.float64)),
		'white_win_count': int(white_win_count),
		'draw_count': int(draw_count),
		'black_win_count': int(black_win_count),
		'white_win_pct': float((100.0 * white_win_count / total_records) if total_records else 0.0),
		'draw_pct': float((100.0 * draw_count / total_records) if total_records else 0.0),
		'black_win_pct': float((100.0 * black_win_count / total_records) if total_records else 0.0),
		'connected_components': int(len(connected_components)),
		'largest_component_size': int(largest_component_size),
		'largest_component_pct': float((100.0 * largest_component_size / num_players) if num_players else 0.0),
		'strongly_connected_components': int(len(sccs)),
		'largest_scc_size': int(largest_scc_size),
		'largest_scc_pct': float((100.0 * largest_scc_size / num_players) if num_players else 0.0),
		'directed_diameter': int(diameter_info['directed_diameter']),
		'directed_diameter_method': str(diameter_info['method']),
		'directed_diameter_samples': int(diameter_info['sample_count']),
		'avg_shortest_path_estimate': float(diameter_info['avg_shortest_path_estimate']),
		'mutual_pairs': float(reciprocity['mutual_pairs']),
		'edge_reciprocity': float(reciprocity['edge_reciprocity']),
		'out_degree_distribution': summarize_distribution(graph_views.out_degree.astype(np.float64)),
		'in_degree_distribution': summarize_distribution(graph_views.in_degree.astype(np.float64)),
		'out_strength_distribution': summarize_distribution(graph_views.out_strength.astype(np.float64)),
		'in_strength_distribution': summarize_distribution(graph_views.in_strength.astype(np.float64)),
		'top_out_strength': top_players_table(
			metric=graph_views.out_strength.astype(np.float64),
			id_to_name=build_result.id_to_name,
			top_k=args.top_k,
		),
		'top_in_strength': top_players_table(
			metric=graph_views.in_strength.astype(np.float64),
			id_to_name=build_result.id_to_name,
			top_k=args.top_k,
		),
		'flow_capacity_basis': 'winner_to_loser' if args.flow_use_wins else 'white_to_black_games',
		'flow_source': flow_source_name,
		'flow_sink': flow_sink_name,
		'flow_endpoint_selection': flow_selection,
		'max_flow': int(flow_info['max_flow']),
	}

	known_elo_values = []
	for player_id in range(num_players):
		if build_result.elo_count[player_id] > 0:
			known_elo_values.append(build_result.elo_sum[player_id] / build_result.elo_count[player_id])
	known_elo = np.asarray(known_elo_values, dtype=np.float64)
	metrics['player_elo_distribution'] = summarize_distribution(known_elo)
	metrics['players_with_known_elo'] = int(known_elo.size)

	print_summary(metrics)

	if args.json_out:
		output_dir = os.path.dirname(args.json_out)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)
		with open(args.json_out, 'w', encoding='utf-8') as out_file:
			json.dump(metrics, out_file, indent=2, sort_keys=True)
		print(f'Wrote JSON metrics to: {args.json_out}')


if __name__ == '__main__':
	main()
