"""Budgeted structural observations obtained from explicitly sampled walks.

One walk has eight transitions and nine observed positions. Its declared cost is
eight neighbor-sampling attempts and 36 positional-pair membership probes,
including repeated positions and edges already encountered by the walk. Neither
labels nor graph-wide node features enter this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

try:
    from numba import njit
except ImportError:  # A local RNG also keeps the fallback independent of numpy's global RNG.
    njit = None


WALK_STEPS = 8
WALK_POSITIONS = WALK_STEPS + 1
PAIR_I, PAIR_J = np.triu_indices(WALK_POSITIONS, 1)
PAIR_COUNT = len(PAIR_I)
COST_PER_WALK = WALK_STEPS + PAIR_COUNT
ACTION_NAMES = ("srw", "nbrw")
FOUR_NODE_PAIRS = tuple(zip(*np.triu_indices(4, 1)))


def _canonical_mask(mask):
    """Canonicalize an induced four-node graph while fixing root vertex zero."""
    adjacency = np.zeros((4, 4), dtype=np.bool_)
    for bit, (i, j) in enumerate(FOUR_NODE_PAIRS):
        adjacency[i, j] = adjacency[j, i] = bool(mask & (1 << bit))
    codes = []
    for other in permutations((1, 2, 3)):
        order = (0,) + other
        codes.append(sum(int(adjacency[order[i], order[j]]) << bit
                         for bit, (i, j) in enumerate(FOUR_NODE_PAIRS)))
    return min(codes)


ROOTED_GRAPHLET_CODES = tuple(sorted({_canonical_mask(mask) for mask in range(64)}))
GRAPHLET_SENTINEL = len(ROOTED_GRAPHLET_CODES)
NUM_GRAPHLET_CLASSES = GRAPHLET_SENTINEL + 1
_CODE_TO_CLASS = {code: index for index, code in enumerate(ROOTED_GRAPHLET_CODES)}
GRAPHLET_CLASS_LUT = np.asarray([_CODE_TO_CLASS[_canonical_mask(mask)]
                               for mask in range(64)], dtype=np.int64)
RELATION_DIM = 2 * PAIR_COUNT + NUM_GRAPHLET_CLASSES


def _csr_valid(indptr, indices):
    for node in range(len(indptr) - 1):
        previous = -1
        for offset in range(indptr[node], indptr[node + 1]):
            neighbor = indices[offset]
            if neighbor < 0 or neighbor >= len(indptr) - 1 or neighbor == node:
                return False
            if neighbor <= previous:
                return False
            previous = neighbor
    return True


def _find_index(indices, start, stop, target):
    """Lower bound in one sorted CSR row (also counts as one membership probe)."""
    while start < stop:
        middle = (start + stop) // 2
        if indices[middle] < target:
            start = middle + 1
        else:
            stop = middle
    return start


def _sample_kernel(indptr, indices, roots, actions, previous_nodes, seed):
    np.random.seed(seed)
    batch_size, walks = actions.shape
    paths = np.empty((batch_size, walks, WALK_POSITIONS), dtype=np.int32)
    adjacency = np.zeros((batch_size, walks, WALK_POSITIONS, WALK_POSITIONS), dtype=np.bool_)
    traversals = np.zeros((batch_size, walks), dtype=np.int64)
    for batch_index in range(batch_size):
        for walk_index in range(walks):
            node = roots[batch_index]
            previous = previous_nodes[batch_index, walk_index]
            paths[batch_index, walk_index, 0] = node
            for position in range(1, WALK_POSITIONS):
                start, stop = indptr[node], indptr[node + 1]
                count = stop - start
                next_node = node
                if count:
                    exclude = stop
                    if actions[batch_index, walk_index] == 1 and count > 1 and previous >= 0:
                        candidate = _find_index(indices, start, stop, previous)
                        if candidate < stop and indices[candidate] == previous:
                            exclude = candidate
                    if exclude < stop:
                        selected = np.random.randint(start, stop - 1)
                        if selected >= exclude:
                            selected += 1
                    else:
                        selected = np.random.randint(start, stop)
                    next_node = indices[selected]
                    traversals[batch_index, walk_index] += 1
                previous, node = node, next_node
                paths[batch_index, walk_index, position] = node
            for i in range(WALK_POSITIONS):
                node = paths[batch_index, walk_index, i]
                start, stop = indptr[node], indptr[node + 1]
                for j in range(i + 1, WALK_POSITIONS):
                    other = paths[batch_index, walk_index, j]
                    found = _find_index(indices, start, stop, other)
                    exists = found < stop and indices[found] == other
                    adjacency[batch_index, walk_index, i, j] = exists
                    adjacency[batch_index, walk_index, j, i] = exists
    return paths, adjacency, traversals


if njit is not None:
    _csr_valid = njit(cache=False)(_csr_valid)
    _find_index = njit(cache=False)(_find_index)
    _sample_kernel = njit(cache=False)(_sample_kernel)
else:
    # Replace the kernel's RNG namespace, without seeding numpy's process-wide RNG.
    import types
    _fallback_globals = dict(_sample_kernel.__globals__)
    _fallback_globals["np"] = types.SimpleNamespace(
        random=np.random.RandomState(), empty=np.empty, zeros=np.zeros,
        int32=np.int32, int64=np.int64, bool_=np.bool_)
    _sample_kernel = types.FunctionType(_sample_kernel.__code__, _fallback_globals)


class CSRGraph:
    """A simple loopless graph with sorted unique adjacency rows.

    CSR arrays remain private to the sampling implementation. In particular,
    node degrees are not returned as free observations to representation models.
    """

    def __init__(self, indptr, indices):
        raw_indptr, raw_indices = np.asarray(indptr), np.asarray(indices)
        if raw_indptr.ndim != 1 or raw_indices.ndim != 1:
            raise ValueError("CSR indptr and indices must be one-dimensional")
        if not np.issubdtype(raw_indptr.dtype, np.integer) or not np.issubdtype(raw_indices.dtype, np.integer):
            raise TypeError("CSR indptr and indices must contain integers")
        if len(raw_indptr) == 0 or raw_indptr[0] != 0 or raw_indptr[-1] != len(raw_indices):
            raise ValueError("Invalid CSR pointer endpoints")
        if np.any(raw_indptr[1:] < raw_indptr[:-1]):
            raise ValueError("CSR row pointers must be nondecreasing")
        if len(raw_indptr) - 1 > np.iinfo(np.int32).max:
            raise ValueError("Node IDs must fit int32")
        # Check range before narrowing indices to int32.
        if len(raw_indices) and (raw_indices.min() < 0 or raw_indices.max() >= len(raw_indptr) - 1):
            raise ValueError("CSR contains an out-of-range node ID")
        self._indptr = np.array(raw_indptr, dtype=np.int64, order="C", copy=True)
        self._indices = np.array(raw_indices, dtype=np.int32, order="C", copy=True)
        if not _csr_valid(self._indptr, self._indices):
            raise ValueError("CSR rows must be sorted, unique, and have no self loops")
        self._indptr.flags.writeable = False
        self._indices.flags.writeable = False

    @property
    def num_nodes(self):
        return len(self._indptr) - 1

    @classmethod
    def from_edge_index(cls, edge_index, num_nodes, undirected=True):
        from scipy.sparse import coo_matrix
        edge_index = np.asarray(edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if not np.issubdtype(edge_index.dtype, np.integer):
            raise TypeError("edge_index must contain integers")
        num_nodes = _nonnegative_integer(num_nodes, "num_nodes")
        if num_nodes > np.iinfo(np.int32).max:
            raise ValueError("Node IDs must fit int32")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise ValueError("edge_index contains an out-of-range node ID")
        matrix = coo_matrix((np.ones(edge_index.shape[1], dtype=np.bool_),
                             (edge_index[0], edge_index[1])),
                            shape=(num_nodes, num_nodes), dtype=np.bool_).tocsr()
        if undirected:
            matrix = matrix.maximum(matrix.T)
        matrix.setdiag(False)
        matrix.eliminate_zeros()
        matrix.sort_indices()
        return cls(matrix.indptr, matrix.indices)


@dataclass(frozen=True)
class QueryLedger:
    """Actual primitive accounting; graphlet extraction reuses paid probes."""

    per_root: np.ndarray
    neighbor_samples: int
    membership_probes: int
    attempted_walk_steps: int
    actual_traversals: int
    cost_per_walk: int = COST_PER_WALK

    @property
    def total_queries(self):
        return self.neighbor_samples + self.membership_probes


@dataclass(frozen=True)
class ObservationBatch:
    paths: np.ndarray
    adjacency: np.ndarray
    equality: np.ndarray
    anonymous_ids: np.ndarray
    graphlet_labels: np.ndarray
    relations: np.ndarray
    actions: np.ndarray
    costs: QueryLedger

    @property
    def targets(self):
        """The fixed, non-learned structural target (an alias, not a copy)."""
        return self.relations

    @property
    def induced_adjacency(self):
        return self.adjacency

    @property
    def cost_ledger(self):
        return self.costs

    @property
    def total_queries(self):
        return self.costs.total_queries

    @property
    def queries_per_root(self):
        return self.costs.per_root


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _walk_array(value, batch_size, walks, name):
    value = np.asarray(value)
    if value.ndim == 0:
        return np.broadcast_to(value, (batch_size, walks))
    if value.shape == (batch_size,):
        return np.broadcast_to(value[:, None], (batch_size, walks))
    if value.shape == (batch_size, walks):
        return value
    raise ValueError(f"{name} must be scalar, [B], or [B, K]")


def _descriptors(paths, adjacency):
    equality = paths[..., :, None] == paths[..., None, :]
    anonymous_ids = np.zeros(paths.shape, dtype=np.int8)
    first_positions = np.zeros(paths.shape[:2] + (4,), dtype=np.int64)
    distinct_count = np.ones(paths.shape[:2], dtype=np.int8)
    for position in range(1, WALK_POSITIONS):
        earlier = equality[..., position, :position]
        seen = earlier.any(-1)
        first_seen = earlier.argmax(-1)
        existing_id = np.take_along_axis(anonymous_ids, first_seen[..., None], axis=-1)[..., 0]
        anonymous_ids[..., position] = np.where(seen, existing_id, distinct_count)
        for vertex in range(1, 4):
            first_positions[..., vertex] = np.where(
                (~seen) & (distinct_count == vertex), position, first_positions[..., vertex])
        distinct_count += ~seen
    masks = np.zeros(paths.shape[:2], dtype=np.int64)
    for bit, (i, j) in enumerate(FOUR_NODE_PAIRS):
        edge = np.take_along_axis(adjacency, first_positions[..., i, None, None], axis=-2)[..., 0, :]
        edge = np.take_along_axis(edge, first_positions[..., j, None], axis=-1)[..., 0]
        masks |= edge.astype(np.int64) << bit
    labels = np.where(distinct_count >= 4, GRAPHLET_CLASS_LUT[masks], GRAPHLET_SENTINEL)
    relations = np.concatenate((equality[..., PAIR_I, PAIR_J],
                                adjacency[..., PAIR_I, PAIR_J],
                                np.eye(NUM_GRAPHLET_CLASSES, dtype=np.bool_)[labels]), axis=-1).astype(np.float32)
    return equality, anonymous_ids, labels, relations


def sample_observations(graph, roots, num_walks, action, seed, *,
                        previous_nodes=None, max_queries=None):
    """Sample [B, K, 9] walks and their 93-dimensional fixed relations.

    Relation order is 36 upper-triangle equality bits, 36 upper-triangle
    induced-edge bits, and 21 rooted-four-node class indicators. Root-preserving
    classes include all 20 isomorphism classes (11 connected) and a sentinel
    for walks visiting fewer than four distinct nodes.

    ``action`` accepts "srw"/"nbrw", IDs 0/1, or arrays shaped [B] or [B,K].
    ``previous_nodes`` optionally supplies a predecessor for NBRW continuation.
    Isolated vertices stay in place while still paying every declared query.
    A supplied query cap is checked before any graph sampling is performed.
    """
    if not isinstance(graph, CSRGraph):
        raise TypeError("graph must be a CSRGraph")
    roots = np.asarray(roots)
    if roots.ndim != 1 or not np.issubdtype(roots.dtype, np.integer):
        raise ValueError("roots must be a one-dimensional integer array")
    if roots.size and (roots.min() < 0 or roots.max() >= graph.num_nodes):
        raise ValueError("A root is outside the graph")
    roots = np.ascontiguousarray(roots, dtype=np.int32)
    num_walks = _nonnegative_integer(num_walks, "num_walks")
    seed = _nonnegative_integer(seed, "seed")
    count = len(roots) * num_walks
    total_cost = count * COST_PER_WALK
    if max_queries is not None and _nonnegative_integer(max_queries, "max_queries") < total_cost:
        raise ValueError(f"Observation requires {total_cost} query units")
    action_values = _walk_array(action, len(roots), num_walks, "action")
    if action_values.dtype.kind in "US":
        if np.any((action_values != "srw") & (action_values != "nbrw")):
            raise ValueError("Actions must be 'srw' or 'nbrw'")
        actions = (action_values == "nbrw").astype(np.int8)
    elif np.issubdtype(action_values.dtype, np.integer):
        if np.any((action_values != 0) & (action_values != 1)):
            raise ValueError("Action IDs must be 0 or 1")
        actions = np.array(action_values, dtype=np.int8, order="C", copy=True)
    else:
        raise TypeError("Actions must be strings or integer IDs")
    if previous_nodes is None:
        previous = np.full(actions.shape, -1, dtype=np.int32)
    else:
        previous = _walk_array(previous_nodes, len(roots), num_walks, "previous_nodes")
        if not np.issubdtype(previous.dtype, np.integer):
            raise TypeError("previous_nodes must contain integers")
        if previous.size and (previous.min() < -1 or previous.max() >= graph.num_nodes):
            raise ValueError("A previous node is outside the graph")
        previous = np.array(previous, dtype=np.int32, order="C", copy=True)
    derived_seed = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])
    paths, adjacency, traversals = _sample_kernel(
        graph._indptr, graph._indices, roots, actions, previous, derived_seed)
    equality, anonymous_ids, labels, relations = _descriptors(paths, adjacency)
    ledger = QueryLedger(np.full(len(roots), num_walks * COST_PER_WALK, dtype=np.int64),
                         count * WALK_STEPS, count * PAIR_COUNT,
                         count * WALK_STEPS, int(traversals.sum()))
    return ObservationBatch(paths, adjacency, equality, anonymous_ids,
                            labels, relations, actions, ledger)


def gather_visited_features(features, paths):
    """Gather only the node-feature rows identified by actual observations."""
    features, paths = np.asarray(features), np.asarray(paths)
    if features.ndim != 2 or not np.issubdtype(paths.dtype, np.integer):
        raise ValueError("Expected [N,F] features and integer visited-node IDs")
    if paths.size and (paths.min() < 0 or paths.max() >= len(features)):
        raise ValueError("A visited node is outside the feature matrix")
    return features[paths]
