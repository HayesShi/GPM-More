"""Read-only original-directed-train overlap audit of official valid negatives."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file())
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"
OUT = ROOT / "outputs/mm_pair_sports_20261028"
N = 50250


def membership(sorted_keys, query):
    location = np.searchsorted(sorted_keys, query)
    return (location < len(sorted_keys)) & (sorted_keys[np.minimum(location, len(sorted_keys) - 1)] == query)


def main():
    start = time.perf_counter()
    train_source = np.load(DATA / "train_source.npy", mmap_mode="r")
    train_target = np.load(DATA / "train_target.npy", mmap_mode="r")
    source = np.load(DATA / "valid_source.npy", mmap_mode="r")
    negative = np.load(DATA / "valid_neg.npy", mmap_mode="r")
    sorted_keys = np.unique(train_source * N + train_target)
    negative_keys = source[:, None] * N + negative
    forward = membership(sorted_keys, negative_keys)
    reverse = membership(sorted_keys, negative * N + source[:, None])
    self_negative = source[:, None] == negative
    categories = {
        "forward_only": forward & ~reverse,
        "reverse_only": ~forward & reverse,
        "both_directions": forward & reverse,
        "neither_direction": ~forward & ~reverse,
        "forward_inclusive": forward,
        "reverse_inclusive": reverse,
        "either_direction": forward | reverse,
        "self_negative": self_negative,
        "bidirected_loop_free_train_adjacency": (forward | reverse) & ~self_negative,
    }
    total = int(negative.size)
    counts = {}
    for name, mask in categories.items():
        count = int(mask.sum())
        rows = int(mask.any(axis=1).sum())
        counts[name] = {"candidate_positions": count, "candidate_position_rate": count / total,
                        "query_rows": rows, "query_row_rate": rows / len(source),
                        "unique_directed_pairs": int(np.unique(negative_keys[mask]).size)}
    # These four categories partition candidate positions exactly.
    assert sum(counts[k]["candidate_positions"] for k in
               ("forward_only", "reverse_only", "both_directions", "neither_direction")) == total
    # Select concrete overlap cases first, then fill from all other candidates.
    locations = []
    for category in ("forward_only", "reverse_only", "both_directions"):
        for row, column in np.argwhere(categories[category])[:3]:
            locations.append((int(row), int(column), category))
    occupied = {(r, c) for r, c, _ in locations}
    for category in ("forward_only", "reverse_only", "both_directions", "neither_direction", "self_negative"):
        for row, column in np.argwhere(categories[category])[:10]:
            if len(locations) == 10:
                break
            pair = (int(row), int(column))
            if pair not in occupied:
                locations.append((*pair, category))
                occupied.add(pair)
        if len(locations) == 10:
            break
    # Independent tuple-set checks use original train rows, without integer keys.
    train_pairs = set(zip(train_source.tolist(), train_target.tolist()))
    examples = []
    for row, column, category in locations:
        u, v = int(source[row]), int(negative[row, column])
        check_forward, check_reverse = (u, v) in train_pairs, (v, u) in train_pairs
        assert check_forward == bool(forward[row, column])
        assert check_reverse == bool(reverse[row, column])
        examples.append({"valid_query_row": row, "negative_column": column,
                         "source_node": u, "negative_target_node": v,
                         "category": category, "u_to_v_in_original_train": check_forward,
                         "v_to_u_in_original_train": check_reverse,
                         "independent_python_tuple_set_verified": True})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Original directed official train rows versus official valid target-negative candidate positions only; no test data opened",
        "train_directed_rows": int(train_source.size), "train_unique_directed_keys": int(sorted_keys.size),
        "validation_query_rows": int(len(source)), "negative_candidates_per_query": int(negative.shape[1]),
        "negative_candidate_positions": total,
        "counting": "All original matrix positions are counted, including repeated negative candidates; unique directed-pair counts are also supplied",
        "definitions": {
            "forward_only": "(u,v) in original train and (v,u) absent",
            "reverse_only": "(u,v) absent from original train and (v,u) present",
            "both_directions": "both (u,v) and (v,u) present in original train",
            "neither_direction": "neither direction present in original train",
            "forward_inclusive": "all original-direction (u,v) matches, including both_directions",
            "reverse_inclusive": "all reverse-direction (v,u) matches, including both_directions",
            "self_negative": "u equals negative target v; separate flag that may overlap direction categories",
            "bidirected_loop_free_train_adjacency": "either direction in original train, excluding u=v, matching prepared CSR membership",
            "query_rows": "number of valid query rows containing at least one candidate position in the category; row counts across categories may overlap",
        },
        "counts": counts, "independent_set_check_examples": examples,
        "test_read_or_scored": False, "split_or_training_files_modified": False,
        "runtime_seconds": time.perf_counter() - start,
        "implementation": "NumPy sorted-key searchsorted, single-thread integer operations; no GPU",
        "reproduce_command": 'python research_extensions/mm_pair_sports_20261028/negative_overlap_audit.py',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "negative_overlap_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(__file__, OUT / "negative_overlap_audit_source.py")
    print(json.dumps({"counts": counts, "runtime_seconds": report["runtime_seconds"],
                      "independent_set_checked_examples": len(examples)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
