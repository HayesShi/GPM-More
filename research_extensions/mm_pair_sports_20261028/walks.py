"""Train-CSR SRW observations with exact batch-positive edge exclusion.

The exclusion overlay is consulted inside both neighbor sampling and every
induced-edge membership probe, so the walk and its 93 structural inputs refer
to the same effective graph. Production GPM sampling is not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

_GPM_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_GPM_SRC) not in sys.path:
    sys.path.insert(0, str(_GPM_SRC))
from gpm_more.sampling import (
    CSRGraph, COST_PER_WALK, QueryLedger, _descriptors, njit,
)


def _lower(values, start, stop, target):
    while start < stop:
        middle = (start + stop) // 2
        if values[middle] < target:
            start = middle + 1
        else:
            stop = middle
    return start


def _sample_excluded(indptr, indices, roots, walks, excluded_keys, excluded_offsets, seed):
    np.random.seed(seed)
    nodes = len(indptr) - 1
    paths = np.empty((len(roots), walks, 9), dtype=np.int32)
    valid = np.zeros((len(roots), walks, 9), dtype=np.bool_)
    adjacency = np.zeros((len(roots), walks, 9, 9), dtype=np.bool_)
    traversals = 0
    for batch in range(len(roots)):
        for walk in range(walks):
            node = roots[batch]
            paths[batch, walk, 0] = node
            valid[batch, walk, 0] = True
            for t in range(1, 9):
                start, stop = indptr[node], indptr[node + 1]
                ex_start = _lower(excluded_keys, 0, len(excluded_keys), node * nodes)
                ex_stop = _lower(excluded_keys, ex_start, len(excluded_keys), (node + 1) * nodes)
                allowed = stop - start - (ex_stop - ex_start)
                if allowed:
                    # Sample an allowed rank exactly uniformly; skip removed
                    # CSR offsets, with no unbounded rejection loop.
                    selected = start + np.random.randint(allowed)
                    for ex in range(ex_start, ex_stop):
                        if selected >= excluded_offsets[ex]:
                            selected += 1
                    node = indices[selected]
                    valid[batch, walk, t] = True
                    traversals += 1
                paths[batch, walk, t] = node
            for i in range(9):
                node = paths[batch, walk, i]
                start, stop = indptr[node], indptr[node + 1]
                for j in range(i + 1, 9):
                    other = paths[batch, walk, j]
                    found = _lower(indices, start, stop, other)
                    exists = found < stop and indices[found] == other
                    if exists:
                        key = node * nodes + other
                        ex = _lower(excluded_keys, 0, len(excluded_keys), key)
                        exists = ex == len(excluded_keys) or excluded_keys[ex] != key
                    adjacency[batch, walk, i, j] = exists
                    adjacency[batch, walk, j, i] = exists
    return paths, valid, adjacency, traversals


if njit is not None:
    _lower = njit(cache=False)(_lower)
    _sample_excluded = njit(cache=False)(_sample_excluded)
else:
    import types
    _fallback_globals = dict(_sample_excluded.__globals__)
    _fallback_globals["np"] = types.SimpleNamespace(
        random=np.random.RandomState(), empty=np.empty, zeros=np.zeros,
        int32=np.int32, bool_=np.bool_)
    _sample_excluded = types.FunctionType(_sample_excluded.__code__, _fallback_globals)


@dataclass(frozen=True)
class PairObservations:
    paths: np.ndarray
    relations: np.ndarray
    occurrence_valid: np.ndarray
    adjacency: np.ndarray
    equality: np.ndarray
    graphlet_labels: np.ndarray
    costs: QueryLedger
    excluded_directed_edges: int


def exclusion_overlay(graph: CSRGraph, excluded_edges=None):
    """Return sorted existing directed edge keys and their CSR offsets.

    Input is [2,E] (preferred, including the ambiguous 2x2 case) or [E,2].
    Reverse edges are always included. Nonedges have no effect on the graph.
    """
    if excluded_edges is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    edges = np.asarray(excluded_edges)
    if edges.ndim != 2 or (edges.shape[0] != 2 and edges.shape[1] != 2):
        raise ValueError("excluded_edges must have shape [2,E] or [E,2]")
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("excluded_edges must contain integers")
    if edges.shape[0] != 2:
        edges = edges.T
    if edges.size and (edges.min() < 0 or edges.max() >= graph.num_nodes):
        raise ValueError("excluded_edges contain an invalid node")
    nodes = graph.num_nodes
    keys = np.unique(np.concatenate((edges[0].astype(np.int64) * nodes + edges[1],
                                    edges[1].astype(np.int64) * nodes + edges[0])))
    kept, offsets = [], []
    for key in keys:
        source, target = divmod(int(key), nodes)
        start, stop = graph._indptr[source:source + 2]
        offset = int(start + np.searchsorted(graph._indices[start:stop], target))
        if offset < stop and graph._indices[offset] == target:
            kept.append(key)
            offsets.append(offset)
    return np.asarray(kept, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def sample_walks(graph: CSRGraph, roots, num_walks: int = 16, seed: int = 0,
                 *, excluded_edges=None) -> PairObservations:
    """Sample SRW length 8; all batch positives must be excluded together.

    Roots should be unique within a training context. A fixed evaluation bank
    is sampled once per root on the unchanged train graph (excluded_edges=None)
    and then reused for every candidate. Repeated roots are permitted.
    """
    if not isinstance(graph, CSRGraph):
        raise TypeError("graph must be the train-only gpm_more CSRGraph")
    roots = np.asarray(roots)
    if roots.ndim != 1 or not np.issubdtype(roots.dtype, np.integer):
        raise ValueError("roots must be a one-dimensional integer array")
    if roots.size and (roots.min() < 0 or roots.max() >= graph.num_nodes):
        raise ValueError("root outside graph")
    if isinstance(num_walks, bool) or not isinstance(num_walks, (int, np.integer)) or num_walks < 1:
        raise ValueError("num_walks must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    excluded_keys, excluded_offsets = exclusion_overlay(graph, excluded_edges)
    derived_seed = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])
    paths, valid, adjacency, traversals = _sample_excluded(
        graph._indptr, graph._indices, np.ascontiguousarray(roots, dtype=np.int32),
        int(num_walks), excluded_keys, excluded_offsets, derived_seed)
    equality, _, labels, relations = _descriptors(paths, adjacency)
    count = len(roots) * num_walks
    costs = QueryLedger(np.full(len(roots), num_walks * COST_PER_WALK, dtype=np.int64),
                        count * 8, count * 36, count * 8, int(traversals))
    return PairObservations(paths, relations, valid, adjacency, equality, labels,
                            costs, len(excluded_keys))


# Explicit descriptive alias for runners.
sample_observations_excluding = sample_walks
