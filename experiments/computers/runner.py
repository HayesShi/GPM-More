"""Local, label-safe adapter around the unmodified official GPM Model.

The official Computers data transforms, seed42 high split, model, optimizer and
scheduler are preserved. Training can accumulate microbatches on the 8GB GPU.
The only labels decoded here belong to the exported high train/validation IDs.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path.cwd() / "runs/computers"
WORKSPACE = Path.cwd()
OFFICIAL = WORKSPACE / "external/GPM/GPM"
PREPARED = ROOT / "prepared"
RUN = ROOT / "model"
GPU_LOCK = WORKSPACE / "runs/.gpu0.lock"
SOURCE_GRAPH = WORKSPACE / "data/computers/task.npz"
SAFE_FEATURES = SOURCE_GRAPH
SOURCE_REVISION = "2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784"



def _replace_file(temporary: Path, path: Path):
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.01 * 2 ** min(attempt, 5), 0.25))


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf8")
    _replace_file(temporary, path)


def _save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    _replace_file(temporary, path)


class QueueLock:
    """Same byte-zero advisory lock as the existing experiment queues."""
    def __enter__(self):
        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.stream = GPU_LOCK.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError(f"GPU lock is held: {GPU_LOCK}")
        _save_json(ROOT / "gpu_lease.json", {"pid": os.getpid(), "active": True, "time": time.time()})
        return self

    def __exit__(self, *_):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        _save_json(ROOT / "gpu_lease.json", {"pid": os.getpid(), "active": False, "time": time.time()})


def official_imports():
    official_path = str(OFFICIAL)
    if official_path not in sys.path:
        sys.path.insert(0, official_path)
    model_module = importlib.import_module("model.model")
    if Path(inspect.getfile(model_module.Model)).resolve() != OFFICIAL / "model/model.py":
        raise RuntimeError("A different top-level model module is already imported")
    return model_module.Model


def official_params():
    official_imports()
    from utils.args import get_args
    original_argv = sys.argv
    try:
        sys.argv = ["official_args", "--dataset", "computers", "--use_params", "--debug"]
        params = get_args()
    finally:
        sys.argv = original_argv
    with (OFFICIAL.parent / "config/main.yaml").open(encoding="utf8") as stream:
        params.update(yaml.safe_load(stream)["node"]["computers"])
    with (OFFICIAL.parent / "config/data.yaml").open(encoding="utf8") as stream:
        dataset_config = yaml.safe_load(stream)["computers"]
    params.update(task=dataset_config["task"], metric=dataset_config["metric"],
                  num_tasks=dataset_config.get("num_tasks"), device="cuda:0",
                  data_path=str(PREPARED), pattern_path=str(PREPARED),
                  save_path=str(RUN), input_dim=767, edge_dim=0, output_dim=10)
    assert params["split"] == "high" and params["split_repeat"] == 3
    assert params["node_pe"] == "lap" and params["node_pe_dim"] == 8
    return params


def _selected_labels(path: Path, ids: np.ndarray, node_count: int):
    """Decode requested y.npy rows only, without instantiating skipped labels."""
    ids = np.asarray(ids, dtype=np.int64)
    with zipfile.ZipFile(path) as archive, archive.open("y.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0 if version == (2, 0) else None)
        if reader is None:
            raise ValueError("Unsupported NPY header")
        shape, fortran, dtype = reader(stream)
        trailing = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if shape[0] != node_count or dtype.hasobject or (fortran and trailing != 1):
            raise ValueError("Unexpected label layout")
        start, row_bytes = stream.tell(), trailing * dtype.itemsize
        values = np.empty((len(ids), *shape[1:]), dtype=dtype)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(sorted_ids) != 1) + 1, len(ids)]
        for left, right in zip(cuts[:-1], cuts[1:]):
            if left == right:
                continue
            stream.seek(start + int(sorted_ids[left]) * row_bytes)
            raw = stream.read((right - left) * row_bytes)
            if len(raw) != (right - left) * row_bytes:
                raise IOError("Truncated labels")
            values[order[left:right]] = np.frombuffer(raw, dtype=dtype).reshape((right-left, *shape[1:]))
    return values.reshape(-1)


def prepare_graph():
    """Reproduce load_node_task without asking the Amazon loader for its y."""
    from torch_geometric.data import Data
    import torch_geometric.transforms as T
    official_imports()
    from data.pyg_data_loader import get_split
    from utils.utils import seed_everything
    PREPARED.mkdir(parents=True, exist_ok=True)
    marker = PREPARED / "graph_complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf8"))
    params = official_params()
    seed_everything(42)
    started = time.perf_counter()
    with np.load(SAFE_FEATURES, allow_pickle=False) as arrays:
        x = torch.from_numpy(arrays["x"].copy())
    with np.load(SOURCE_GRAPH, allow_pickle=False) as arrays:
        edge_index = torch.from_numpy(arrays["edge_index"].copy())
    graph = Data(x=x, edge_index=edge_index)
    transform = T.Compose([T.NormalizeFeatures(), T.RemoveSelfLoops(), T.ToUndirected(),
                           T.AddLaplacianEigenvectorPE(params["node_pe_dim"], "pe")])
    print(json.dumps({"stage": "official_transforms", "nodes": graph.num_nodes}), flush=True)
    graph = transform(graph)
    # Official load_node_task calls the transform once and then get_split 3 times.
    rng_before_split = torch.get_rng_state().clone()
    splits = [get_split(graph, params["split"]) for _ in range(params["split_repeat"])]
    rng_after_split = torch.get_rng_state().clone()
    masks = {key + "_mask": torch.stack([split[key] for split in splits], dim=1).numpy()
             for key in ("train", "val", "test")}
    ids = {key + "_ids": np.flatnonzero(masks[key + "_mask"][:, 0])
           for key in ("train", "val", "test")}
    # Verify the actual arrays against the exact pre-split RNG state and official cuts.
    generator = torch.Generator().set_state(rng_before_split)
    first_perm = torch.randperm(graph.num_nodes, generator=generator).numpy()
    cut_train = int(graph.num_nodes * 0.6)
    cut_val = int(graph.num_nodes * (0.6 + 0.2))
    assert np.array_equal(ids["train_ids"], np.sort(first_perm[:cut_train]))
    assert np.array_equal(ids["val_ids"], np.sort(first_perm[cut_train:cut_val]))
    assert np.array_equal(ids["test_ids"], np.sort(first_perm[cut_val:]))
    assert np.all(sum(masks.values()) == 1)
    np.savez_compressed(PREPARED / "shared_graph.npz", x=graph.x.numpy(), pe=graph.pe.numpy(),
                        edge_index=graph.edge_index.numpy())
    np.savez_compressed(PREPARED / "splits.npz", **masks, **ids)
    # No high-test labels are decoded. Test IDs are exported solely as masks/IDs.
    train_y = _selected_labels(SOURCE_GRAPH, ids["train_ids"], graph.num_nodes)
    val_y = _selected_labels(SOURCE_GRAPH, ids["val_ids"], graph.num_nodes)
    np.savez_compressed(PREPARED / "train_val.npz", train_ids=ids["train_ids"],
                        val_ids=ids["val_ids"], train_y=train_y, val_y=val_y)
    _save_checkpoint(PREPARED / "preprocessing_rng.pt", {
        "torch_before_split": rng_before_split, "torch_after_split": rng_after_split,
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(), "python": random.getstate()})
    metadata = {
        "complete": True, "official_source_revision": SOURCE_REVISION,
        "dataset": "computers", "mask_seed": 42, "split": "high", "split_column": 0,
        "split_repeat_exported": 3, "fractions": [0.6, 0.2, 0.2],
        "split_counts": [{key: int(split[key].sum()) for key in ("train", "val", "test")}
                         for split in splits],
        "transforms": ["NormalizeFeatures", "RemoveSelfLoops", "ToUndirected",
                       "AddLaplacianEigenvectorPE(k=8, attr_name=pe, is_undirected=False [official default])"],
        "feature_shape": list(graph.x.shape), "pe_shape": list(graph.pe.shape),
        "edge_shape": list(graph.edge_index.shape), "labels": "high_train_validation_only",
        "test_labels_decoded": False,
        "array_verification": {"official_get_split_called": True,
                               "first_split_matches_exact_presplit_rng_perm": True,
                               "all_masks_disjoint_and_exhaustive": True,
                               "split0_first20_sorted_ids": {k: v[:20].tolist() for k, v in ids.items()},
                               "first20_official_perm": first_perm[:20].tolist(),
                               "feature_first5_rows_sums": graph.x[:5].sum(dim=1).tolist(),
                               "feature_first5_rows_nonzeros": (graph.x[:5] != 0).sum(dim=1).tolist()},
        "graph_source": str(SOURCE_GRAPH), "safe_x_source": str(SAFE_FEATURES),
        "label_extraction": "numeric y.npy selected train/val row reads; skipped rows not instantiated",
        "seconds": time.perf_counter() - started,
        "shared_graph_keys": ["x", "pe", "edge_index"],
        "splits_keys": list(masks) + list(ids),
        "train_val_keys": ["train_ids", "val_ids", "train_y", "val_y"]}
    _save_json(marker, metadata)
    _save_json(ROOT / "status.json", {"stage": "graph_prepared", "pid": os.getpid(), "metadata": metadata})
    print(json.dumps({"stage": "graph_prepared", "seconds": metadata["seconds"],
                      "counts": metadata["split_counts"][0]}), flush=True)
    return metadata


def load_graph(device="cpu"):
    from torch_geometric.data import Data
    with np.load(PREPARED / "shared_graph.npz", allow_pickle=False) as arrays:
        graph = Data(**{key: torch.from_numpy(arrays[key].copy()) for key in ("x", "pe", "edge_index")})
    return graph.to(device)


def prepare_patterns():
    marker = PREPARED / "patterns_complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf8"))
    official_imports()
    from model.random_walk import get_patterns
    params = official_params()
    state = torch.load(PREPARED / "preprocessing_rng.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    started = time.perf_counter()
    graph = load_graph()
    print(json.dumps({"stage": "official_get_patterns", "pre_sample_batch_size": params["pre_sample_batch_size"]}), flush=True)
    patterns, eids = get_patterns(graph, params)
    assert patterns.shape == (128, 13752, 9) and eids.shape == (128, 13752, 8)
    expected_roots = torch.arange(graph.num_nodes).view(1, -1)
    assert torch.equal(patterns[:, :, 0], expected_roots.expand(128, -1))
    assert int(patterns.min()) >= 0 and int(patterns.max()) < graph.num_nodes
    _save_checkpoint(PREPARED / "patterns.pt", {"pattern": patterns, "eid": eids})
    np.save(PREPARED / "paths.npy", patterns.permute(1, 0, 2).numpy().astype(np.int32), allow_pickle=False)
    metadata = {"complete": True, "sampler": str(OFFICIAL / "model/random_walk.py") + "::get_patterns",
                "pattern_shape": list(patterns.shape), "eid_shape": list(eids.shape),
                "rawsum_paths_shape": [13752, 128, 9], "rawsum_paths_dtype": "int32",
                "p": 1, "q": 1, "walk_length": 8, "pre_sample_batch_size": params["pre_sample_batch_size"],
                "seed_sequence": "restore official seed42 transform + three high split RNG; torch_cluster CUDA sampling",
                "roots_exact": True, "node0_first4_walks": patterns[:4, 0].tolist(),
                "seconds": time.perf_counter() - started, "test_labels_decoded": False}
    _save_json(marker, metadata)
    print(json.dumps({"stage": "patterns_prepared", "seconds": metadata["seconds"]}), flush=True)
    del graph, patterns, eids
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def _runtime_params(device):
    params = official_params()
    params["device"] = device
    params["pattern_set"] = torch.load(PREPARED / "patterns.pt", map_location="cpu", weights_only=False)
    return params


def _make_model(params, seed=0):
    Model = official_imports()
    from utils.utils import seed_everything
    seed_everything(seed)
    return Model(params=params).to(params["device"])


def _optimizer_scheduler(model, params):
    from utils.scheduler import get_scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"],
                                  betas=(params["opt_beta1"], params["opt_beta2"]), eps=params["opt_eps"])
    return optimizer, get_scheduler(optimizer, params)


@torch.no_grad()
def _predict(model, graph, params, ids, micro_batch_size):
    model.eval()
    predictions = []
    for left in range(0, len(ids), micro_batch_size):
        cur_nodes = torch.as_tensor(ids[left:left+micro_batch_size], dtype=torch.long)
        pred, _, _, _ = model(graph, cur_nodes, params, mode="eval")
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        predictions.append(pred.detach().cpu())
    return torch.cat(predictions).numpy() if predictions else np.empty((0, params["output_dim"]), dtype=np.float32)


def predict_from_best(run_dir: Path, node_ids: np.ndarray, device="cuda:0", micro_batch_size=8,
                      *, model_factory=None) -> np.ndarray:
    """Frozen official Model inference; never loads labels or acquires the GPU lock."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    params = _runtime_params(device)
    params["save_path"] = str(Path(run_dir))
    if checkpoint["params"] != {key: value for key, value in params.items() if key != "pattern_set"}:
        # Device is an execution choice, while all model/preprocessing params are frozen.
        saved = dict(checkpoint["params"])
        current = {key: value for key, value in params.items() if key != "pattern_set"}
        saved.pop("device", None)
        current.pop("device", None)
        if saved != current:
            raise ValueError("Frozen checkpoint configuration differs from this runner")
    graph = load_graph(device)
    model = (model_factory or _make_model)(params, seed=checkpoint["seed"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    try:
        return _predict(model, graph, params, np.asarray(node_ids, dtype=np.int64), micro_batch_size)
    finally:
        del model, graph, params
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def profile():
    output = ROOT / "profile.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf8"))
    params = _runtime_params("cuda:0")
    graph = load_graph("cuda:0")
    with np.load(PREPARED / "train_val.npz", allow_pickle=False) as arrays:
        ids, labels = arrays["train_ids"], arrays["train_y"]
        val_ids = arrays["val_ids"]
    attempts = []
    chosen = None
    for size in (256, 128, 64, 32):
        model = _make_model(params)
        optimizer, scheduler = _optimizer_scheduler(model, params)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        pred = loss = commit_loss = outputs = None
        try:
            outputs = model(graph, torch.from_numpy(ids[:size]), params, mode="train")
            pred, commit_loss = outputs[0], outputs[3]
            loss = F.cross_entropy(pred, torch.as_tensor(labels[:size], device="cuda:0"),
                                   label_smoothing=params["label_smoothing"]) + commit_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            attempts.append({"train_microbatch": size, "ok": True, "seconds": elapsed,
                             "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20})
            chosen = size
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"train_microbatch": size, "ok": False, "error": str(error)})
        finally:
            pred = loss = commit_loss = outputs = None
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
        if chosen is not None:
            break
    if chosen is None:
        raise RuntimeError("Official model did not fit even with train microbatch32")
    model = _make_model(params)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = _predict(model, graph, params, val_ids[:64], 8)
    torch.cuda.synchronize()
    validation64_seconds = time.perf_counter() - started
    result = {"complete": True, "engineering_profile_only": True, "training_attempts": attempts,
              "train_microbatch": chosen, "logical_batch_size": params["batch_size"],
              "eval_microbatch": 8, "eval64_seconds": validation64_seconds,
              "eval128_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "parameters": sum(p.numel() for p in model.parameters()),
              "buffers": sum(b.numel() for b in model.buffers()),
              "gpu": torch.cuda.get_device_name(), "cuda_total_mib": torch.cuda.get_device_properties(0).total_memory/2**20,
              "estimated_epoch_train_seconds": attempts[-1]["seconds"] * math.ceil(len(ids)/256) * math.ceil(256/chosen),
              "estimated_validation_seconds": validation64_seconds * len(val_ids)/64,
              "official_model_class_file": inspect.getfile(type(model)),
              "official_pattern_encoder_class_file": inspect.getfile(type(model.pattern_encoder))}
    _save_json(output, result)
    print(json.dumps({"stage": "profile_complete", **result}), flush=True)
    del logits, model, graph, params
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def train_seed0(profile_result, *, model_factory=None, run_dir=None, provenance=None, params_override=None, checkpoint_control=None, progress_callback=None):
    """Shared official loop; custom factories must preserve the four-output API."""
    run = Path(run_dir) if run_dir is not None else RUN
    run.mkdir(parents=True, exist_ok=True)
    metrics_path = run / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text(encoding="utf8")).get("complete"):
        print("Frozen completed run already exists; training skipped.", flush=True)
        return
    params = _runtime_params("cuda:0")
    params["save_path"] = str(run)
    if set(params_override or {}) - {"lr"}:
        raise ValueError("Only the frozen LR override is permitted")
    params.update(params_override or {})
    provenance = dict(provenance or {"method": "official_gpm", "pretraining": "none",
                                    "initialization": "official_seed0_from_scratch"})
    from utils.early_stop import EarlyStopping
    model = (model_factory or _make_model)(params, seed=0)
    graph = load_graph("cuda:0")
    optimizer, scheduler = _optimizer_scheduler(model, params)
    stopper = EarlyStopping(patience=params["early_stop"])
    microbatch = profile_result["train_microbatch"]
    eval_microbatch = profile_result["eval_microbatch"]
    with np.load(PREPARED / "train_val.npz", allow_pickle=False) as arrays:
        train_ids = torch.from_numpy(arrays["train_ids"].copy())
        train_y = torch.from_numpy(arrays["train_y"].copy())
        val_ids, val_y = arrays["val_ids"].copy(), arrays["val_y"].copy()
    frozen_params = {key: value for key, value in params.items() if key != "pattern_set"}
    _save_json(run / "config.json", {"params": frozen_params, "seed": 0, "split_column": 0,
               "train_microbatch": microbatch, "eval_microbatch": eval_microbatch,
               "pretraining": provenance.get("pretraining"), "provenance": provenance,
               "method": provenance.get("method"), "model_config": getattr(model, "model_config", frozen_params),
               "labels": "train_validation_only", "test_labels_decoded": False,
               "source_revision": SOURCE_REVISION, "model_file": inspect.getfile(type(model)),
               "backend": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_tf32": torch.backends.cudnn.allow_tf32}})
    history, update, first_epoch = [], 0, 1
    best_epoch, best_update, best_accuracy = None, None, -1.0
    train_seconds, validation_seconds = 0.0, 0.0
    latest_path = run / "latest.pt"
    if latest_path.exists():
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        _restore_rng(saved["rng"])
        stopper.best_val, stopper.counter = saved["stopper_best_val"], saved["stopper_counter"]
        history, update, first_epoch = saved["history"], saved["update"], saved["epoch"] + 1
        best_epoch, best_update, best_accuracy = saved["best_epoch"], saved["best_update"], saved["best_accuracy"]
        train_seconds, validation_seconds = saved["train_seconds"], saved["validation_seconds"]
    started = time.perf_counter()
    resumed_finished = stopper.counter >= params["early_stop"] or first_epoch > params["epochs"]
    stop_reason = "official_early_stopping" if stopper.counter >= params["early_stop"] else "epoch_cap"
    epoch = first_epoch - 1
    batch_count = math.ceil(len(train_ids) / params["batch_size"])
    for epoch in (() if resumed_finished else range(first_epoch, params["epochs"] + 1)):
        model.train()
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        permutation = torch.randperm(len(train_ids))
        total_loss = 0.0
        batch_count = math.ceil(len(train_ids) / params["batch_size"])
        for batch in range(batch_count):
            positions = permutation[batch*params["batch_size"]:(batch+1)*params["batch_size"]]
            optimizer.zero_grad()
            logical_loss = 0.0
            for left in range(0, len(positions), microbatch):
                sub_positions = positions[left:left+microbatch]
                nodes = train_ids[sub_positions]
                labels = train_y[sub_positions].to(params["device"])
                pred, _, _, commit_loss = model(graph, nodes, params, mode="train")
                loss = F.cross_entropy(pred, labels, label_smoothing=params["label_smoothing"]) + commit_loss
                weight = len(sub_positions) / len(positions)
                (loss * weight).backward()
                logical_loss += loss.item() * weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            optimizer.step()
            scheduler.step()
            update += 1
            total_loss += logical_loss
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        train_seconds += epoch_seconds
        row = {"epoch": epoch, "update": update, "train_loss": total_loss/batch_count,
               "lr_after_update": scheduler.get_last_lr()[0], "train_seconds": epoch_seconds}
        if epoch % params["eval_every"] == 0:
            validation_started = time.perf_counter()
            logits = _predict(model, graph, params, val_ids, eval_microbatch)
            accuracy = float(np.mean(logits.argmax(axis=1) == val_y))
            torch.cuda.synchronize()
            eval_seconds = time.perf_counter() - validation_started
            validation_seconds += eval_seconds
            row.update(val_accuracy=accuracy, validation_seconds=eval_seconds)
            is_stop = stopper({"val": accuracy * 100.0})
            if accuracy > best_accuracy:
                best_epoch, best_update, best_accuracy = epoch, update, accuracy
                _save_checkpoint(run / "best.pt", {
                    "schema": "local-graph-training-v1", "model_state": model.state_dict(),
                    "params": frozen_params, "seed": 0, "epoch": epoch, "update": update,
                    "val_accuracy": accuracy, "preprocessing": str(PREPARED),
                    "source_revision": SOURCE_REVISION, "initialization": provenance.get("initialization"),
                    "method": provenance.get("method"), "provenance": provenance,
                    "model_config": getattr(model, "model_config", frozen_params)})
                np.savez_compressed(run / "best_val_predictions.npz", node_ids=val_ids, logits=logits)
            row.update(stopper_counter=stopper.counter, best_val_accuracy=best_accuracy)
        else:
            is_stop = False
        history.append(row)
        progress = {"complete": False, "pid": os.getpid(), "epoch": epoch, "update": update,
                    "best_val_accuracy": best_accuracy, "best_epoch": best_epoch, "best_update": best_update,
                    "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                    "wall_this_process_seconds": time.perf_counter()-started,
                    "train_microbatch": microbatch, "eval_microbatch": eval_microbatch}
        _save_json(run / "progress.json", progress)
        if progress_callback is not None:
            progress_callback(progress)
        if epoch % params["eval_every"] == 0:
            _save_json(run / "history.json", history)
            _save_checkpoint(latest_path, {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "rng": _rng_state(), "epoch": epoch, "update": update,
                    "history": history, "stopper_best_val": stopper.best_val, "stopper_counter": stopper.counter,
                    "best_epoch": best_epoch, "best_update": best_update, "best_accuracy": best_accuracy,
                    "train_seconds": train_seconds, "validation_seconds": validation_seconds})
            print(json.dumps({"stage": "validation", **row}), flush=True)
        if is_stop:
            stop_reason = "official_early_stopping"
            break
        if epoch % params["eval_every"] == 0 and epoch < params["epochs"] and checkpoint_control is not None:
            pause_reason = checkpoint_control(progress)
            if pause_reason:
                paused = dict(progress, complete=False, stop_reason=pause_reason, checkpoint=str(latest_path))
                _save_json(metrics_path, paused)
                _save_json(run / "status.json", paused)
                return paused
    metrics = {"complete": True, "dataset": "computers", "split": "high", "split_column": 0, "seed": 0,
               "epoch": epoch, "update": update, "best_epoch": best_epoch, "best_update": best_update,
               "val_accuracy": best_accuracy, "best_val_accuracy": best_accuracy, "stop_reason": stop_reason,
               "train_seconds": train_seconds, "validation_seconds": validation_seconds,
               "wall_this_process_seconds": time.perf_counter()-started,
               "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
               "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
               "train_microbatch": microbatch, "eval_microbatch": eval_microbatch,
               "optimizer_steps_per_epoch": batch_count, "max_epoch": params["epochs"],
               "best_checkpoint": str(run/"best.pt"), "test_labels_decoded": False,
               "source_revision": SOURCE_REVISION, "method": provenance.get("method"), "provenance": provenance}
    _save_json(metrics_path, metrics)
    _save_json(run / "history.json", history)
    _save_json(run / "progress.json", metrics)
    status_path = ROOT / "status.json" if run == RUN else run / "status.json"
    _save_json(status_path, {"stage": "complete", "pid": os.getpid(), "metrics": metrics})
    print(json.dumps({"stage": "complete", **metrics}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "smoke"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "smoke":
        folder = ROOT / "smoke"
        _save_json(folder / "atomic.json", {"ok": True})
        _save_checkpoint(folder / "atomic.pt", {"ok": torch.tensor([1])})
        assert torch.load(folder/"atomic.pt", weights_only=False)["ok"].item() == 1
        params = official_params()
        assert params["split"] == "high" and params["lr"] == .01
        print("SMOKE_OK official config/import and real atomic writes", flush=True)
        return
    metadata = prepare_graph()
    if args.command == "prepare":
        return
    gpu_processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                                   text=True, capture_output=True, check=True).stdout
    live_python = [line for line in gpu_processes.splitlines() if "python" in line.lower() and str(os.getpid()) not in line.split(",")[0]]
    if live_python:
        raise RuntimeError("Live GPU Python process exists before lock: " + repr(live_python))
    with QueueLock():
        prepare_patterns()
        profile_result = profile()
        train_seed0(profile_result)


if __name__ == "__main__":
    raise SystemExit("Implementation module: use python experiments/computers/run.py run --help")
