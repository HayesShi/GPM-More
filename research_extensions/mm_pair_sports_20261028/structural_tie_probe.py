"""Exploratory FINAL-score comparison on exact positive-RA structure ties.

The comparison is declared before the joint FINAL score becomes available.
It does not train, read test data, change a split, or select a tolerance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import numba
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file())
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"
CAMPAIGN = ROOT / "outputs/mm_pair_sports_20261028"
OUT = CAMPAIGN / "structural_tie_probe"
SCRIPT_DIR = ROOT / "research_extensions/mm_pair_sports_20261028"
sys.path.insert(0, str(SCRIPT_DIR))
# Reuse precisely the float64 RA implementation of the existing graph baseline:
# sorted CSR neighbor intersection, ascending common-node sum, no fastmath.
from structural_baselines import score_block


DECLARATION = {
    "analysis": "exploratory exact-RA-tied positive-vs-negative FINAL-score ordering",
    "cpu_threads": 2,
    "validation_rows": 45557,
    "negative_candidates_per_row": 150,
    "eligibility": "Within the same official valid query: positive exact RA>0 and negative exact RA == positive exact RA",
    "ra_computation": "float64 sum of 1/degree(w) over sorted ascending shared train-CSR neighbor IDs; reuse structural_baselines.score_block; no fastmath",
    "ra_equality": "exact float64 equality, no tolerance and no rounding",
    "learned_order_correctness": "1 if learned positive>negative, 0.5 if learned positive==negative, 0 otherwise",
    "scores": {
        "independent": "runs/pair_independent/valid_scores_final.npy",
        "joint": "runs/pair_joint/valid_scores_final.npy",
    },
    "score_layout": "column 0 positive, columns 1..150 official negative candidates in released order",
    "groups": {
        "full_valid": "all original valid query rows",
        "clean_valid": "existing diagnostic mask: not valid_reverse_in_train and not valid_self_loop",
    },
    "primary_summaries": ["eligible query count", "eligible candidate-position pair count",
                          "pair-weighted ordering correctness", "query-averaged ordering correctness"],
    "candidate_duplicates": "retain official repeated negative positions and count each position as a pair",
    "limits": "Exploratory validation diagnostic, not causal or test evidence. Exact float equality can miss mathematically equal RA sums with different floating-point accumulation histories; near-equal RA scores are intentionally excluded. Comparisons are within the same query and use FINAL checkpoint outputs only.",
    "test_read_or_scored": False,
    "training_or_plan_modified": False,
}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = OUT / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        existing = {k: v for k, v in config.items() if k != "declared_utc"}
        if existing != DECLARATION:
            raise ValueError("Existing declaration differs; refusing to change the declared comparison")
    else:
        config = {"declared_utc": datetime.now(timezone.utc).isoformat(), **DECLARATION}
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if Path(__file__).resolve() != (OUT / "source.py").resolve():
        shutil.copyfile(__file__, OUT / "source.py")
    return config


def structure_ties(source, target, negative, indptr, indices):
    cache_path = OUT / "exact_ra_ties.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            positive_ra = cache["positive_ra"]
            tied = np.unpackbits(cache["negative_tie_mask_packed"], axis=1,
                                 count=150, bitorder="little").astype(bool)
        if positive_ra.shape != (45557,) or tied.shape != (45557, 150):
            raise ValueError("Unexpected exact-RA tie cache dimensions")
        return positive_ra, tied, "reused compact exact-RA tie cache; no repeated RA computation"
    # No full RA score array was retained by structural_baselines. Compute once,
    # retain the positive score and packed eligible mask needed by this probe.
    positive_ra = np.empty(len(source), dtype=np.float64)
    tied = np.empty((len(source), 150), dtype=bool)
    for left in range(0, len(source), 1024):
        right = min(left + 1024, len(source))
        candidates = np.column_stack([target[left:right], negative[left:right]])
        _, _, ra = score_block(source[left:right], candidates, indptr, indices)
        positive_ra[left:right] = ra[:, 0]
        tied[left:right] = (ra[:, :1] > 0) & (ra[:, 1:] == ra[:, :1])
    np.savez_compressed(cache_path, positive_ra=positive_ra,
                        negative_tie_mask_packed=np.packbits(tied, axis=1, bitorder="little"))
    return positive_ra, tied, "computed exact full-candidate RA once; cached positive scores and packed eligible mask"


def summarize(aggregate, mask):
    eligible = mask & (aggregate["tie_count"] > 0)
    queries = int(eligible.sum())
    pairs = int(aggregate["tie_count"][eligible].sum())
    result = {"source_query_rows": int(mask.sum()), "eligible_queries": queries,
              "eligible_candidate_position_pairs": pairs}
    for name in ("independent", "joint"):
        wins = aggregate[f"{name}_win_count"][eligible]
        ties = aggregate[f"{name}_tie_count"][eligible]
        correct = wins.astype(np.float64) + 0.5 * ties
        result[name] = {
            "strict_positive_wins": int(wins.sum()), "learned_score_ties": int(ties.sum()),
            "strict_positive_losses": int(pairs - wins.sum() - ties.sum()),
            "pair_weighted_correctness": float(correct.sum() / pairs) if pairs else None,
            "query_averaged_correctness": float((correct / aggregate["tie_count"][eligible]).mean()) if queries else None,
        }
    if pairs:
        result["joint_minus_independent"] = {
            "pair_weighted_correctness": result["joint"]["pair_weighted_correctness"] - result["independent"]["pair_weighted_correctness"],
            "query_averaged_correctness": result["joint"]["query_averaged_correctness"] - result["independent"]["query_averaged_correctness"],
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = prepare()
    if args.prepare_only:
        print(json.dumps({"status": "comparison_declared", "config": str(OUT / "config.json")}), flush=True)
        return
    score_paths = {name: CAMPAIGN / path for name, path in DECLARATION["scores"].items()}
    missing = [str(path) for path in score_paths.values() if not path.exists()]
    if missing:
        print(json.dumps({"status": "ready_waiting_for_FINAL_scores", "missing": missing}), flush=True)
        return
    start = time.perf_counter()
    numba.set_num_threads(2)
    # Read scores before graph work. A partially written npy fails here instead
    # of silently using an interim checkpoint or an incomplete score array.
    scores = {name: np.load(path, mmap_mode="r") for name, path in score_paths.items()}
    for name, value in scores.items():
        if value.shape != (45557, 151) or not np.isfinite(value).all():
            raise ValueError(f"Invalid FINAL score array for {name}: {value.shape}")
    source = np.load(DATA / "valid_source.npy", mmap_mode="r")
    target = np.load(DATA / "valid_target.npy", mmap_mode="r")
    negative = np.load(DATA / "valid_neg.npy", mmap_mode="r")
    indptr = np.load(DATA / "train_csr_indptr.npy", mmap_mode="r")
    indices = np.load(DATA / "train_csr_indices.npy", mmap_mode="r")
    reverse = np.load(DATA / "valid_reverse_in_train.npy")
    self_loop = np.load(DATA / "valid_self_loop.npy")
    positive_ra, eligible, reuse = structure_ties(source, target, negative, indptr, indices)
    aggregate = {"query_row": np.arange(len(source), dtype=np.int32),
                 "positive_ra": positive_ra,
                 "tie_count": eligible.sum(axis=1).astype(np.int32),
                 "clean_valid": ~reverse & ~self_loop}
    for name, value in scores.items():
        aggregate[f"{name}_win_count"] = ((value[:, :1] > value[:, 1:]) & eligible).sum(axis=1).astype(np.int32)
        aggregate[f"{name}_tie_count"] = ((value[:, :1] == value[:, 1:]) & eligible).sum(axis=1).astype(np.int32)
    # Small independent tuple-set RA reconstruction verifies deterministic sums
    # and the exact-equality mask, including repeated candidate positions.
    check_rows = np.flatnonzero(aggregate["tie_count"] > 0)[:10]
    checks = []
    for row in check_rows:
        column = int(np.flatnonzero(eligible[row])[0])
        u, pos, neg = int(source[row]), int(target[row]), int(negative[row, column])
        left = set(indices[indptr[u]:indptr[u + 1]].tolist())
        def set_ra(v):
            right = set(indices[indptr[v]:indptr[v + 1]].tolist())
            return sum(1.0 / (int(indptr[w + 1]) - int(indptr[w])) for w in sorted(left & right))
        p, q = set_ra(pos), set_ra(neg)
        assert p > 0 and p == q and p == positive_ra[row]
        checks.append({"query_row": int(row), "negative_column": column,
                       "source_node": u, "positive_target": pos, "negative_target": neg,
                       "exact_positive_and_negative_ra": p, "python_set_sum_verified": True})
    np.savez_compressed(OUT / "query_aggregates.npz", **aggregate)
    groups = {"full_valid": np.ones(len(source), dtype=bool), "clean_valid": aggregate["clean_valid"]}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "declaration_utc": config["declared_utc"],
        "comparison": DECLARATION["eligibility"], "exact_ra_cache": reuse,
        "query_rows_with_positive_ra_nonzero": {name: int(((positive_ra > 0) & mask).sum()) for name, mask in groups.items()},
        "results": {name: summarize(aggregate, mask) for name, mask in groups.items()},
        "source_scores": {name: {"path": str(path.relative_to(ROOT)),
                                 "shape": list(scores[name].shape), "dtype": str(scores[name].dtype),
                                 "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                          for name, path in score_paths.items()},
        "verification_examples": checks, "runtime_seconds": time.perf_counter() - start,
        "cpu_threads": 2, "gpu_used": False, "test_read_or_scored": False,
        "interpretation": DECLARATION["limits"],
        "reproduce_command": 'python research_extensions/mm_pair_sports_20261028/structural_tie_probe.py',
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if Path(__file__).resolve() != (OUT / "source.py").resolve():
        shutil.copyfile(__file__, OUT / "source.py")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
