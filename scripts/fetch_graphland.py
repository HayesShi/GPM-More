"""Fetch one official GraphLand dataset and export train/validation-only arrays.

The pinned upstream Dataset performs the original CPU preprocessing. It reads
the complete raw target CSV; this exporter writes only train/validation labels
and never evaluates a test metric. Dataset files are cached outside source code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile


REVISION = "7246fe3f6b53cffaa76e3dc22726a6db3984bbe3"
UPSTREAM = "https://github.com/yandex-research/graphland"
RAW_BASE = f"https://raw.githubusercontent.com/yandex-research/graphland/{REVISION}"
RELEASE = "https://zenodo.org/records/16895532"
ARCHIVE_MD5 = {
    "artnet-views": "48e3eb3f45bb3ff50b4c1ba323135f5b",
    "tolokers-2": "f453141143f3a93a61502e461a03d222",
}
REQUIRED_FILES = ("info.yaml", "features.csv", "targets.csv", "edgelist.csv", "split_masks_RL.csv")


def report(phase, **values):
    print(json.dumps({"phase": phase, **values}, ensure_ascii=False), flush=True)


def archive_url(name):
    return f"https://zenodo.org/api/records/16895532/files/{name}.zip/content"


def published_md5(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    """Cache a complete transfer; no existing complete file is replaced."""
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "GPM-More-GraphLand/0.1"})
    report("download", url=url)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(target)


def ensure_upstream(root, offline):
    source = root / "upstream" / REVISION
    for name in ("dataset.py", "LICENSE"):
        target = source / name
        if not target.is_file():
            if offline:
                raise FileNotFoundError(f"Offline mode needs cached {target}")
            download(f"{RAW_BASE}/{name}", target)
    return source / "dataset.py"


def ensure_raw(name, root, raw_root, offline):
    folder = raw_root / "data" / name
    missing = [filename for filename in REQUIRED_FILES if not (folder / filename).is_file()]
    if not missing:
        return folder, "reused extracted required CSV/YAML files"
    archive = root / "downloads" / f"{name}.zip"
    if not archive.is_file():
        if offline:
            raise FileNotFoundError(f"Missing raw files {missing} and no cached archive {archive}")
        download(archive_url(name), archive)
    # These are archive checksums published by the fixed Zenodo record.
    if published_md5(archive) != ARCHIVE_MD5[name]:
        raise ValueError(f"Archive differs from the publisher's MD5: {archive}")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for filename in missing:
            members = [item for item in package.infolist() if not item.is_dir()
                       and PurePosixPath(item.filename).name == filename]
            if len(members) != 1:
                raise ValueError(f"Expected one {filename}, found {len(members)} in {archive}")
            # The destination is an explicit allowlisted basename. ZIP member
            # paths never determine an extraction target.
            target = folder / filename
            partial = folder / (filename + ".part")
            with package.open(members[0]) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            partial.replace(target)
    return folder, "required files extracted from publisher-MD5-checked archive"


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def canonical_edges(edge_index, num_nodes):
    import numpy as np

    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Expected PyG edge_index[2,E]")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("Edge endpoint out of range")
    edges = edges[:, edges[0] != edges[1]]
    if edges.shape[1]:
        order = np.lexsort((edges[1], edges[0]))
        edges = edges[:, order]
        unique = np.r_[True, np.any(edges[:, 1:] != edges[:, :-1], axis=0)]
        edges = edges[:, unique]
    return edges


def export(name, root, output, *, raw_root=None, offline=False):
    import numpy as np
    import torch

    # Fail before downloading a large archive if an optional loader dependency
    # is absent. No graphland packages are imported during --help.
    for module in ("pandas", "yaml", "sklearn", "torch_geometric"):
        if importlib.util.find_spec(module) is None:
            raise ImportError(f"Missing {module}; install GPM-More with pip install -e '.[graphland]'")
    root, output = Path(root).resolve(), Path(output).resolve()
    raw_root = root if raw_root is None else Path(raw_root).resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must be a new .npz file")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Choose a new output NPZ and metadata filename")
    source = ensure_upstream(root, offline)
    _, raw_origin = ensure_raw(name, root, raw_root, offline)
    spec = importlib.util.spec_from_file_location("_gpm_more_graphland_dataset", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    torch.set_num_threads(4)
    with working_directory(raw_root):
        dataset = official.Dataset(name, split="RL", use_pyg=True,
                                   add_self_loops=False, to_undirected=True, device="cpu")
    x = dataset.features.detach().cpu().numpy().astype(np.float32, copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Official preprocessing returned invalid features")
    train_mask = dataset.train_mask.detach().cpu().numpy().astype(bool)
    val_mask = dataset.val_mask.detach().cpu().numpy().astype(bool)
    if np.any(train_mask & val_mask) or not train_mask.any() or not val_mask.any():
        raise ValueError("Train/validation masks must be nonempty and disjoint")
    train_ids, val_ids = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    # Upstream has already decoded all targets to perform official transforms.
    # Only these selected rows enter the exported NPZ.
    train_y = dataset.targets[dataset.train_mask].detach().cpu().numpy().copy()
    val_y = dataset.targets[dataset.val_mask].detach().cpu().numpy().copy()
    if not np.isfinite(train_y).all() or not np.isfinite(val_y).all():
        raise ValueError("Official train/validation targets contain nonfinite values")
    edges = canonical_edges(dataset.graph.detach().cpu().numpy(), len(x))
    transform = None
    if dataset.task == "regression":
        if dataset.regression_targets_transform_name != "standard-scaler":
            raise ValueError("Expected official default train-fitted StandardScaler")
        fitted = dataset.regression_targets_transform
        transform = {"name": "standard-scaler", "mean": float(fitted.mean_[0]),
                     "scale": float(fitted.scale_[0]), "fit_split": "train_labeled_only"}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".npz.part")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, x=x, edge_index=edges, train_ids=train_ids, val_ids=val_ids,
                            train_y=train_y, val_y=val_y)
    partial.replace(output)
    metadata = {
        "schema": "gpm-more-graphland-export-v1", "name": name,
        "upstream": UPSTREAM, "revision": REVISION, "loader": "dataset.py, unmodified by this script",
        "release": RELEASE, "archive": archive_url(name), "publisher_archive_md5": ARCHIVE_MD5[name],
        "raw_files": list(REQUIRED_FILES), "raw_origin": raw_origin,
        "split": "RL", "task": dataset.task, "metric": dataset.metric_name,
        "nodes": len(x), "arcs": edges.shape[1], "features": x.shape[1],
        "train_roots": len(train_ids), "val_roots": len(val_ids),
        "target_transform": transform, "targets_already_standardized": transform is not None,
        "feature_fit_scope": "official transductive all-node features",
        "graph": {"undirected": True, "self_loops": False, "duplicate_arcs": "coalesced", "order": "row/column sorted"},
        "exported_arrays": ["x", "edge_index", "train_ids", "val_ids", "train_y", "val_y"],
        "upstream_reads_complete_target_csv": True, "test_targets_exported": False,
        "test_metrics_computed": False, "historical_walk_cache_reproduced": False,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "torch", "pandas", "scikit-learn", "PyYAML", "torch-geometric")},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    report("exported", dataset=name, output=str(output), metadata=str(metadata_path),
           nodes=len(x), train_roots=len(train_ids), val_roots=len(val_ids), task=dataset.task)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(ARCHIVE_MD5), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/graphland"),
                        help="Cache root: upstream/, downloads/, and data/<dataset>/")
    parser.add_argument("--output", type=Path, required=True, help="New train/validation-only graph NPZ")
    parser.add_argument("--raw-root", type=Path,
                        help="Reuse required CSV/YAML files under this root's data/<dataset>/")
    parser.add_argument("--offline", action="store_true", help="Require cached source and raw files/archive")
    args = parser.parse_args(argv)
    return export(args.dataset, args.root, args.output, raw_root=args.raw_root, offline=args.offline)


if __name__ == "__main__":
    main()
