"""Score official Sports validation candidates with CPU-only graph heuristics.

No model training, test access, candidate resampling, or data preparation changes.
CN/RA use the exact (unsampled) loop-free training CSR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numba
from numba import njit, prange
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"
OUT = ROOT / "outputs/mm_pair_sports_20261028/structural_baselines"


@njit(cache=True, parallel=True)
def score_block(source, candidates, indptr, indices):
    rows, columns = candidates.shape
    edge = np.zeros((rows, columns), dtype=np.uint8)
    cn = np.zeros((rows, columns), dtype=np.int32)
    ra = np.zeros((rows, columns), dtype=np.float64)
    for i in prange(rows):
        u = source[i]
        u_start, u_end = indptr[u], indptr[u + 1]
        for j in range(columns):
            v = candidates[i, j]
            # Sorted-neighbor binary search for direct-edge presence.
            low, high = u_start, u_end
            while low < high:
                mid = (low + high) // 2
                if indices[mid] < v:
                    low = mid + 1
                else:
                    high = mid
            if low < u_end and indices[low] == v:
                edge[i, j] = 1
            # Exact merge intersection. Every common neighbor contributes 1/d(w).
            left, right = u_start, indptr[v]
            v_end = indptr[v + 1]
            count, resource = 0, 0.0
            while left < u_end and right < v_end:
                a, b = indices[left], indices[right]
                if a == b:
                    count += 1
                    resource += 1.0 / (indptr[a + 1] - indptr[a])
                    left += 1
                    right += 1
                elif a < b:
                    left += 1
                else:
                    right += 1
            cn[i, j] = count
            ra[i, j] = resource
    return edge, cn, ra


def average_tie_rank(scores):
    positive = scores[:, :1]
    negative = scores[:, 1:]
    return 1.0 + 0.5 * ((negative >= positive).sum(axis=1) + (negative > positive).sum(axis=1))


def metric_summary(rank, mask):
    selected = rank[mask]
    if not selected.size:
        return {"rows": 0, "mrr": None, "hits@1": None, "hits@3": None, "hits@10": None}
    return {"rows": int(selected.size), "mrr": float(np.mean(1.0 / selected)),
            "hits@1": float(np.mean(selected <= 1)), "hits@3": float(np.mean(selected <= 3)),
            "hits@10": float(np.mean(selected <= 10)), "mean_rank": float(selected.mean())}


def zero_summary(stat, mask):
    count = int(mask.sum())
    positive_zero = int(stat["positive_zero"][mask].sum())
    negative_zero = int(stat["negative_zero"][mask].sum())
    all_zero_rows = int(stat["all_zero"][mask].sum())
    return {
        "rows": count, "candidate_pairs": count * 151,
        "positive_score_zero": positive_zero, "positive_score_zero_rate": positive_zero / count,
        "negative_scores_zero": negative_zero, "negative_scores_zero_rate": negative_zero / (count * 150),
        "candidate_scores_zero": positive_zero + negative_zero,
        "candidate_scores_zero_rate": (positive_zero + negative_zero) / (count * 151),
        "all_151_candidate_scores_zero_rows": all_zero_rows,
        "all_151_candidate_scores_zero_rate": all_zero_rows / count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args()
    if args.threads < 1 or args.block_rows < 1:
        parser.error("threads and block-rows must be positive")
    numba.set_num_threads(args.threads)
    start = time.perf_counter()
    source = np.load(DATA / "valid_source.npy", mmap_mode="r")
    target = np.load(DATA / "valid_target.npy", mmap_mode="r")
    negative = np.load(DATA / "valid_neg.npy", mmap_mode="r")
    indptr = np.load(DATA / "train_csr_indptr.npy", mmap_mode="r")
    indices = np.load(DATA / "train_csr_indices.npy", mmap_mode="r")
    reverse = np.load(DATA / "valid_reverse_in_train.npy")
    self_loop = np.load(DATA / "valid_self_loop.npy")
    n = len(source)
    if negative.shape != (45557, 150) or n != 45557:
        raise ValueError("Unexpected official validation candidate shape")
    if np.any(reverse & self_loop):
        raise ValueError("Audit masks are not disjoint")
    groups = {"full_valid": np.ones(n, dtype=bool), "clean_valid": ~reverse & ~self_loop,
              "exposed_reverse_train": reverse, "self_loop": self_loop}
    names = ("edge", "cn", "ra")
    ranks = {name: np.empty(n, dtype=np.float64) for name in names}
    zeros = {name: {"positive_zero": np.empty(n, dtype=bool),
                    "negative_zero": np.empty(n, dtype=np.int32),
                    "all_zero": np.empty(n, dtype=bool)} for name in names}
    total_cn_sum = np.zeros(n, dtype=np.int64)
    positive_cn = np.zeros(n, dtype=np.int32)
    for left in range(0, n, args.block_rows):
        right = min(left + args.block_rows, n)
        candidates = np.empty((right - left, 151), dtype=np.int64)
        candidates[:, 0] = target[left:right]
        candidates[:, 1:] = negative[left:right]
        scores = score_block(source[left:right], candidates, indptr, indices)
        for name, values in zip(names, scores):
            ranks[name][left:right] = average_tie_rank(values)
            zeros[name]["positive_zero"][left:right] = values[:, 0] == 0
            zeros[name]["negative_zero"][left:right] = (values[:, 1:] == 0).sum(axis=1)
            zeros[name]["all_zero"][left:right] = (values == 0).all(axis=1)
        positive_cn[left:right] = scores[1][:, 0]
        total_cn_sum[left:right] = scores[1].sum(axis=1)
        if left == 0 or right == n or (left // args.block_rows) % 10 == 0:
            print(f"CPU graph heuristics: {right}/{n} validation rows, {time.perf_counter() - start:.2f}s", flush=True)
    score_seconds = time.perf_counter() - start
    # Independently check the merge implementation against Python sets on a small
    # deterministic validation-only sample, including direct-edge and self queries.
    check_indices = np.unique(np.concatenate([np.arange(16), np.flatnonzero(reverse)[:8], np.flatnonzero(self_loop)]))
    for i in check_indices:
        u = int(source[i])
        candidates = np.array([[int(target[i]), int(negative[i, 0]), u]], dtype=np.int64)
        actual = score_block(np.array([u], dtype=np.int64), candidates, indptr, indices)
        u_neighbors = set(indices[indptr[u]:indptr[u + 1]].tolist())
        for j, v in enumerate(candidates[0]):
            v_neighbors = set(indices[indptr[v]:indptr[v + 1]].tolist())
            intersection = u_neighbors & v_neighbors
            expected_ra = sum(1.0 / (int(indptr[w + 1]) - int(indptr[w])) for w in sorted(intersection))
            assert int(actual[0][0, j]) == int(v in u_neighbors)
            assert int(actual[1][0, j]) == len(intersection)
            assert np.isclose(actual[2][0, j], expected_ra, rtol=1e-13, atol=1e-15)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        np.save(OUT / f"{name}_rr.npy", 1.0 / ranks[name], allow_pickle=False)
    summaries = {name: {group: metric_summary(ranks[name], mask) for group, mask in groups.items()} for name in names}
    zero_stats = {name: {group: zero_summary(zeros[name], mask) for group, mask in groups.items()} for name in names}
    join = {}
    for group, mask in groups.items():
        stat = zero_stats["cn"][group]
        join[group] = {
            "positive_pairs_with_common_neighbor": int(mask.sum()) - stat["positive_score_zero"],
            "positive_pair_join_available_rate": 1.0 - stat["positive_score_zero_rate"],
            "all_candidate_pairs_with_common_neighbor": stat["candidate_pairs"] - stat["candidate_scores_zero"],
            "all_candidate_pair_join_available_rate": 1.0 - stat["candidate_scores_zero_rate"],
            "rows_with_any_candidate_join_available": int(mask.sum()) - stat["all_151_candidate_scores_zero_rows"],
            "rows_any_candidate_join_available_rate": 1.0 - stat["all_151_candidate_scores_zero_rate"],
            "positive_cn_mean": float(positive_cn[mask].mean()),
            "candidate_cn_mean": float(total_cn_sum[mask].sum() / stat["candidate_pairs"]),
        }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MM-GRAPH sports-copurchase official validation", "validation_rows": n,
        "candidates_per_row": 151, "candidate_pairs": int(n * 151),
        "graph": {"source": "processed official-train-only bidirected deduplicated loop-free CSR",
                  "nodes": int(len(indptr) - 1), "directed_nnz": int(len(indices)),
                  "heldout_reverse_edge_exposure_retained_as_published": True},
        "definitions": {
            "edge": "1[(u,v) in training CSR]",
            "cn": "|N_train(u) intersect N_train(v)|",
            "ra": "sum_{w in N_train(u) intersect N_train(v)} 1/degree_train(w)",
            "tie_rank": "1 + 0.5 * (count(negative >= positive) + count(negative > positive))",
            "clean_valid": "official validation slice excluding valid_reverse_in_train and valid_self_loop; not a replacement official split",
            "exposed_reverse_train": "official validation rows with their reverse edge present in the original train split; excludes self-loops",
            "join_availability": "exact unsampled training-graph common-neighbor availability; not finite random-walk collision availability",
        },
        "metrics": summaries, "zero_scores": zero_stats, "unsampled_join_availability": join,
        "runtime": {"cpu_threads": args.threads, "block_rows": args.block_rows,
                    "scoring_seconds_including_jit": score_seconds,
                    "total_seconds": time.perf_counter() - start, "gpu_used": False,
                    "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__},
        "verification": {"python_set_crosscheck_queries": int(len(check_indices) * 3),
                         "test_read_or_scored": False, "data_preparation_files_modified": False},
        "reproduce_command": 'python research_extensions/mm_pair_sports_20261028/structural_baselines.py --threads 2 --block-rows 1024',
        "artifacts": [f"{name}_rr.npy" for name in names],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": summaries, "unsampled_join_availability": join,
                      "runtime": report["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
