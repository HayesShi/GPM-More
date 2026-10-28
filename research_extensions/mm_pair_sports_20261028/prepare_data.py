"""Prepare the pinned official MM-GRAPH Sports train/validation pilot data.

The test tensors remain in the original release archive. Test positive edge
indices are consulted only for split-isolation checks, never for scoring.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
import scipy.sparse as sp
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/mm_graph/sports-copurchase"
OUT = ROOT / "outputs/mm_pair_sports_20261028"
REVISION = "c707a3203b77a153f11370a05d2a9be6571f7670"
N = 50250
FILES = {
    "clip_feat.pt": {
        "bytes": 205824753,
        "sha256": "b2cb0c0184c5de2b092d67cc44abca9f264223aa23ac4ef05816d661f9510f01",
    },
    "lp-edge-split.pt": {
        "bytes": 106593135,
        "sha256": "6f5ffeb99f1aca5ea05671c6a86e5ee01a33a3673b6c3f0a24c6d482f4fa710a",
    },
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def download(name: str) -> dict:
    meta = FILES[name]
    path = DATA / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://huggingface.co/datasets/mm-graph-org/mm-graph/resolve/"
           f"{REVISION}/sports-copurchase/{name}")
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        for attempt in range(3):
            try:
                start = time.monotonic()
                # A full single copy is retained. No HF cache or duplicate raw file.
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise RuntimeError(f"Unexpected full-download status {response.status_code}")
                    with partial.open("wb") as handle:
                        received = 0
                        for block in response.iter_content(4 << 20):
                            handle.write(block)
                            received += len(block)
                            print(f"download {name} {received}/{meta['bytes']}", flush=True)
                if partial.stat().st_size != meta["bytes"]:
                    raise RuntimeError(f"Wrong size for {name}: {partial.stat().st_size}")
                partial.replace(path)
                print(f"download complete {name} {time.monotonic() - start:.1f}s", flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    actual = digest(path)
    if path.stat().st_size != meta["bytes"] or actual != meta["sha256"]:
        raise RuntimeError(f"Official LFS content check failed: {path}")
    return {"file": name, "url": url, "revision": REVISION,
            "bytes": path.stat().st_size, "sha256": actual,
            "sha256_source": "official HF tree LFS oid", "path": str(path.relative_to(ROOT))}


def as_index(value) -> np.ndarray:
    value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"Noninteger edge indices: {value.dtype}")
    value = np.ascontiguousarray(value, dtype=np.int64)
    if value.size and (value.min() < 0 or value.max() >= N):
        raise ValueError("Out-of-range node id")
    return value


def edge_ids(source: np.ndarray, target: np.ndarray, undirected=False) -> np.ndarray:
    if undirected:
        source, target = np.minimum(source, target), np.maximum(source, target)
    return source * N + target


def count_overlap(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.intersect1d(left, right, assume_unique=True).size)


def prepare(manifest: list[dict]) -> dict:
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    x_tensor = torch.load(DATA / "raw/clip_feat.pt", map_location="cpu", weights_only=True)
    if not isinstance(x_tensor, torch.Tensor):
        raise TypeError(f"Expected feature tensor, got {type(x_tensor)}")
    x = x_tensor.detach().numpy()
    if x.shape != (N, 1024) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError(f"Unexpected features: {x.shape}, {x.dtype}, finite={np.isfinite(x).all()}")
    np.save(processed / "x.npy", x, allow_pickle=False)
    release = torch.load(DATA / "raw/lp-edge-split.pt", map_location="cpu", weights_only=True)
    split_stats, edges, unique_directed, unique_undirected = {}, {}, {}, {}
    for name in ("train", "valid", "test"):
        split = release[name]
        source, target = as_index(split["source_node"]), as_index(split["target_node"])
        if source.ndim != 1 or source.shape != target.shape:
            raise ValueError(f"Invalid {name} shape")
        ids = edge_ids(source, target)
        undirected = edge_ids(source, target, undirected=True)
        unique_directed[name], unique_undirected[name] = np.unique(ids), np.unique(undirected)
        edges[name] = (source, target)
        split_stats[name] = {
            "positive_edges": int(source.size), "source_shape": list(source.shape),
            "target_shape": list(target.shape), "dtype": str(source.dtype),
            "min_node_id": int(min(source.min(), target.min())),
            "max_node_id": int(max(source.max(), target.max())),
            "self_loops": int(np.count_nonzero(source == target)),
            "directed_duplicate_rows": int(source.size - unique_directed[name].size),
            "unique_directed_edges": int(unique_directed[name].size),
            "unique_undirected_edges": int(unique_undirected[name].size),
            "reverse_pair_redundancy": int(unique_directed[name].size - unique_undirected[name].size),
        }
        if name != "test":
            np.save(processed / f"{name}_source.npy", source, allow_pickle=False)
            np.save(processed / f"{name}_target.npy", target, allow_pickle=False)
        if "target_node_neg" in split:
            split_stats[name]["negative_shape"] = list(split["target_node_neg"].shape)
            if name == "valid":
                neg = as_index(split["target_node_neg"])
                if neg.shape != (45557, 150):
                    raise ValueError(f"Unexpected valid negative shape: {neg.shape}")
                np.save(processed / "valid_neg.npy", neg, allow_pickle=False)
                split_stats[name]["negative_equals_positive_target"] = int((neg == target[:, None]).sum())
                split_stats[name]["negative_self_loops"] = int((neg == source[:, None]).sum())
                # No candidate replacement: the official ranking list stays byte-order identical.
                split_stats[name]["rows_with_duplicate_negative_candidates"] = int(
                    np.any(np.diff(np.sort(neg, axis=1), axis=1) == 0, axis=1).sum())
    expected = {"train": 356202, "valid": 45557, "test": 37413}
    for name, count in expected.items():
        if split_stats[name]["positive_edges"] != count:
            raise ValueError(f"Unexpected official {name} count")
    overlap = {}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap[f"{left}_{right}"] = {
            "directed": count_overlap(unique_directed[left], unique_directed[right]),
            "undirected": count_overlap(unique_undirected[left], unique_undirected[right]),
        }
    train_source, train_target = edges["train"]
    valid_source, valid_target = edges["valid"]
    valid_self_loop = valid_source == valid_target
    valid_reverse_in_train = np.isin(edge_ids(valid_target, valid_source), unique_directed["train"])
    valid_reverse_in_train &= ~valid_self_loop
    np.save(processed / "valid_reverse_in_train.npy", valid_reverse_in_train, allow_pickle=False)
    np.save(processed / "valid_self_loop.npy", valid_self_loop, allow_pickle=False)
    keep = train_source != train_target
    row = np.concatenate([train_source[keep], train_target[keep]])
    col = np.concatenate([train_target[keep], train_source[keep]])
    graph = sp.csr_matrix((np.ones(row.size, dtype=np.uint8), (row, col)), shape=(N, N))
    graph.sum_duplicates()
    graph.data[:] = 1
    graph.sort_indices()
    graph.eliminate_zeros()
    if graph.diagonal().any() or (graph != graph.T).nnz:
        raise AssertionError("Train graph is not loop-free symmetric")
    heldout_memberships = {}
    for name in ("valid", "test"):
        source, target = edges[name]
        heldout_memberships[name] = int(np.asarray(graph[source, target]).reshape(-1).sum())
    np.save(processed / "train_csr_indices.npy", graph.indices.astype(np.int64), allow_pickle=False)
    np.save(processed / "train_csr_indptr.npy", graph.indptr.astype(np.int64), allow_pickle=False)
    # Exact roundtrip checks cover the release-to-experiment boundary.
    if not np.array_equal(x, np.load(processed / "x.npy", mmap_mode="r")):
        raise AssertionError("Feature roundtrip changed values")
    for name in ("train", "valid"):
        for value, suffix in zip(edges[name], ("source", "target")):
            if not np.array_equal(value, np.load(processed / f"{name}_{suffix}.npy", mmap_mode="r")):
                raise AssertionError(f"{name} {suffix} roundtrip changed values")
    if not np.array_equal(neg, np.load(processed / "valid_neg.npy", mmap_mode="r")):
        raise AssertionError("Validation negative roundtrip changed candidates")
    if np.any(valid_reverse_in_train & valid_self_loop):
        raise AssertionError("Reverse-in-train mask must exclude self-loops")
    if list(processed.glob("test*")):
        raise AssertionError("Unexpected exported test arrays")
    degree = np.diff(graph.indptr)
    feature_stats = {
        "shape": list(x.shape), "dtype": str(x.dtype), "finite": True,
        "min": float(x.min()), "max": float(x.max()),
        "mean": float(x.mean(dtype=np.float64)), "std": float(x.std(dtype=np.float64)),
        "zero_rows": int(np.count_nonzero(np.linalg.norm(x, axis=1) == 0)),
        "block_order": "official downstream loaders label first 512 text and second 512 visual; generation-level order not independently verified",
    }
    for label, block in (("columns_0_511", x[:, :512]), ("columns_512_1023", x[:, 512:])):
        norms = np.linalg.norm(block, axis=1)
        feature_stats[label] = {"norm_min": float(norms.min()), "norm_median": float(np.median(norms)),
                                "norm_max": float(norms.max()), "zero_rows": int((norms == 0).sum())}
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MM-GRAPH Amazon-Sports / sports-copurchase", "nodes": N,
        "official_hf_revision": REVISION, "downloads": manifest,
        "official_code_revision": "0769fa670c69a6dc0756a2d7fc697b980e494775",
        "graph_pipeline_evidence": {
            "loader": "lp/lp_dataset.py lines 37-41 builds directed graph from official train rows",
            "main": "lp/main.py lines 247-249 removes self-loops then calls to_bidirected_with_reverse_mapping",
            "utility": "lp/utils.py lines 18-19 applies dgl.to_simple(dgl.add_reverse_edges(g))",
            "local_sources": "data/mm_graph/sports-copurchase/raw/sources/",
        },
        "feature_block_evidence": {
            "consumer_convention": "lp/MMGCN/Dataset.py lines 14-16 and lp/MGAT/run.py lines 199-201 label first half text, second half visual",
            "generation_implementation": "not present in inspected official code/HF trees; generation-level column order and exact encoder checkpoint remain unverified",
        },
        "features": feature_stats, "splits": split_stats, "positive_split_overlap": overlap,
        "train_graph": {"source": "official train positive edges only", "undirected": True,
                        "self_loops_removed": int((train_source == train_target).sum()),
                        "directed_csr_nnz": int(graph.nnz), "undirected_edges": int(graph.nnz // 2),
                        "sorted_indices": bool(graph.has_sorted_indices),
                        "degree_min": int(degree.min()), "degree_max": int(degree.max()),
                        "degree_mean": float(degree.mean()), "isolated_nodes": int((degree == 0).sum()),
                        "heldout_positive_row_memberships": heldout_memberships,
                        "protocol": "official train targets preserved; bidirected deduplicated message-passing graph, with self-loops removed; no heldout edges used to edit training",
                        "is_strict_undirected_heldout": False,
                        "known_protocol_contamination": "Directed split rows are disjoint, but published bidirectional message passing exposes heldout pairs through reverse train edges."},
        "validation_audit_masks": {
            "valid_reverse_in_train.npy": {"true_rows": int(valid_reverse_in_train.sum()), "excludes_self_loops": True},
            "valid_self_loop.npy": {"true_rows": int(valid_self_loop.sum())},
            "full_official_rows": int(valid_source.size),
            "clean_slice_rows_excluding_reverse_train_and_self_loops": int((~valid_reverse_in_train & ~valid_self_loop).sum()),
            "usage": "Report full official validation and a diagnostic slice excluding either mask. The diagnostic slice is not a new official split."
        },
        "test_policy": "Test remains in original raw archive. Only test positive indices and tensor shapes used for isolation audit; no test scoring, selection, or exported test arrays.",
        "verification": {"roundtrip_feature_and_train_valid_arrays_exact": True,
                         "csr_symmetry_sorted_loop_free": True,
                         "validation_masks_disjoint_and_counts_checked": True,
                         "test_arrays_exported": False},
        "processed_files": {p.name: {"shape": list(np.load(p, mmap_mode="r").shape),
                                       "dtype": str(np.load(p, mmap_mode="r").dtype), "bytes": p.stat().st_size}
                            for p in sorted(processed.glob("*.npy"))},
        "reproduce_command": 'python research_extensions/mm_pair_sports_20261028/prepare_data.py',
    }
    (OUT / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "processed": str(processed),
                      "features": x.shape, "splits": expected, "train_csr_nnz": int(graph.nnz),
                      "positive_split_overlap": overlap}, indent=2), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest = list(pool.map(download, FILES))
    if not args.download_only:
        prepare(manifest)


if __name__ == "__main__":
    main()
