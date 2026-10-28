"""MM-GRAPH Sports baselines using pinned official model definitions.

CPU sampling uses DGL's official edge-prediction sampler.  The installed Windows
DGL has no CUDA graph backend, so CUDA SAGE executes the same mean operator with
torch and the official SAGEConv parameters.  This is an operator port, not a
claim of reproducing the paper's hyperparameter search or published result.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import time
from typing import Any

import dgl
from dgl.dataloading import (
    DataLoader, MultiLayerFullNeighborSampler, NeighborSampler,
    as_edge_prediction_sampler, negative_sampler,
)
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "external/mm_graph_sports_snapshot"
OFFICIAL_COMMIT = "0769fa670c69a6dc0756a2d7fc697b980e494775"


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SNAPSHOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class _StopRun(Exception):
    pass


def torch_sage_block(layer, block, x: torch.Tensor) -> torch.Tensor:
    """DGL SAGEConv(mean) operator using torch; no DGL CUDA graph is needed.

    The projection-before-aggregation branch matches DGL 1.1.2.  Index-add can
    differ from DGL's kernel in floating-point summation order on CUDA.
    """
    if layer._aggre_type != "mean":
        raise ValueError("Only the official mean SAGE operator is supported")
    x = layer.feat_drop(x)
    n_dst = block.num_dst_nodes()
    h_self = x[:n_dst]
    before = layer._in_src_feats > layer._out_feats
    messages = layer.fc_neigh(x) if before else x
    source, target = block.edges(order="eid")
    source, target = source.to(x.device), target.to(x.device)
    mean = messages.new_zeros((n_dst, messages.shape[-1]))
    mean.index_add_(0, target, messages[source])
    degree = block.in_degrees().to(device=x.device, dtype=x.dtype).clamp_min_(1)
    mean = mean / degree[:, None]
    if not before:
        mean = layer.fc_neigh(mean)
    result = layer.fc_self(h_self) + mean
    if layer.activation is not None:
        result = layer.activation(result)
    if layer.norm is not None:
        result = layer.norm(result)
    return result


def _layer(layer, block, x):
    return layer(block, x) if x.device.type == "cpu" else torch_sage_block(layer, block, x)


def _encode(model, method, blocks, x):
    h = x
    for index, layer in enumerate(model.layers):
        h = _layer(layer, blocks[index], h) if method == "SAGE" else layer(h)
        if index + 1 != len(model.layers):
            h = F.relu(h)
    return h


def _checkpoint(path: Path, model, optimizer, scheduler, epoch, batch, config, best, status, step):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch, "step": step, "batch_in_epoch": batch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "config": config, "best_valid": best, "status": status,
        "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "official_commit": OFFICIAL_COMMIT,
    }, temporary)
    temporary.replace(path)


@torch.no_grad()
def _inference(model, method, graph, features, device, batch_size, check_stop):
    model.eval()
    feat = features
    if method == "MLP":
        result = torch.empty((features.shape[0], model.hid_size), dtype=torch.float32)
        for start in range(0, features.shape[0], batch_size):
            check_stop()
            x = features[start:start + batch_size].to(device)
            result[start:start + batch_size] = _encode(model, method, None, x).cpu()
        return result
    loader = DataLoader(
        graph, torch.arange(graph.num_nodes()), MultiLayerFullNeighborSampler(1),
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0,
    )
    for index, layer in enumerate(model.layers):
        check_stop()
        result = torch.empty((graph.num_nodes(), model.hid_size), dtype=torch.float32)
        for input_nodes, output_nodes, blocks in loader:
            check_stop()
            h = _layer(layer, blocks[0], feat[input_nodes].to(device))
            if index + 1 != len(model.layers):
                h = F.relu(h)
            result[output_nodes] = h.cpu()
        feat = result
    return feat


@torch.no_grad()
def _validate(model, node_emb, source, target, negatives, device, batch_size, check_stop):
    """Evaluate every released validation row, with the official average tie rank."""
    node_emb = node_emb.to(device)
    rr = np.empty(source.numel(), dtype=np.float32)
    ranks = np.empty(source.numel(), dtype=np.float32)
    model.eval()
    for start in range(0, source.numel(), batch_size):
        check_stop()
        end = min(start + batch_size, source.numel())
        destinations = torch.cat((target[start:end, None], negatives[start:end]), dim=1).to(device)
        left = node_emb[source[start:end].to(device)][:, None, :]
        right = node_emb[destinations]
        scores = model.predictor(left * right).squeeze(-1)
        pos, neg = scores[:, :1], scores[:, 1:]
        rank = 1.0 + 0.5 * ((neg > pos).sum(1) + (neg >= pos).sum(1)).float()
        ranks[start:end] = rank.cpu().numpy()
        rr[start:end] = rank.reciprocal().cpu().numpy()
    metrics = {
        "mrr": float(rr.mean(dtype=np.float64)),
        "hits@1": float(np.mean(ranks <= 1)),
        "hits@3": float(np.mean(ranks <= 3)),
        "hits@10": float(np.mean(ranks <= 10)),
        "positive_edges": int(source.numel()), "negatives_per_positive": int(negatives.shape[1]),
    }
    return metrics, rr, ranks


def _slice_metrics(rr, ranks, mask):
    size = int(mask.sum())
    if size == 0:
        return {"positive_edges": 0}
    return {"positive_edges": size, "mrr": float(rr[mask].mean(dtype=np.float64)),
            "hits@1": float(np.mean(ranks[mask] <= 1)),
            "hits@3": float(np.mean(ranks[mask] <= 3)),
            "hits@10": float(np.mean(ranks[mask] <= 10))}


def run_baseline(data_dir, out_dir, config: dict[str, Any], deadline_epoch=None):
    """Run one SAGE/MLP pilot. The caller owns the GPU lock and wall deadline.

    data_dir may be the Sports dataset root or its processed/ folder. Only the
    prepared train/valid arrays are loaded. ``deadline_epoch`` is Unix seconds.
    No test evaluation or hyperparameter search occurs here.
    """
    config = dict(config)
    defaults = {
        "method": "SAGE", "seed": 0, "max_steps": 1000, "eval_every_steps": 250,
        "batch_size": 512, "hidden_dim": 256, "num_layers": 3,
        "lr": 5e-4, "num_of_neighbors": 15, "lr_scheduler_step_size": 5,
        "lr_scheduler_gamma": 0.1, "inference_batch_size": 1000,
        "eval_batch_size": 64, "device": "cuda", "num_threads": 8,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if "max_train_positives" in config:
        if int(config["max_train_positives"]) % int(config["batch_size"]):
            raise ValueError("Baseline positive-target budget must be divisible by batch_size")
        config["max_steps"] = int(config["max_train_positives"]) // int(config["batch_size"])
    if int(config["max_steps"]) <= 0 or int(config["eval_every_steps"]) <= 0:
        raise ValueError("Step budget and evaluation interval must be positive")
    # Accept the runner's common abbreviated spelling without silently ignoring it.
    if "hid" in config:
        config["hidden_dim"] = config["hid"]
    if "hid_dim" in config:
        config["hidden_dim"] = config["hid_dim"]
    method = str(config["method"]).upper()
    if method not in {"SAGE", "MLP"}:
        raise ValueError(f"Unsupported official baseline: {method}")
    config["method"] = method
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    data_dir = data_dir / "processed" if (data_dir / "processed").is_dir() else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(config.get("stop_file", out_dir.parent / "STOP_REQUESTED"))
    _json(out_dir / "config.json", config)
    started = time.time()
    state = {"status": "initializing", "complete": False, "method": method, "epoch": 0, "step": 0,
             "batch_in_epoch": 0, "best_valid": None, "started_unix": started,
             "deadline_unix": float(deadline_epoch) if deadline_epoch is not None else None}

    def progress(**updates):
        state.update(updates)
        state["updated_unix"] = time.time()
        state["elapsed_seconds"] = state["updated_unix"] - started
        _json(out_dir / "progress.json", state)

    def check_stop():
        if stop_file.exists():
            raise _StopRun("stop_requested")
        if deadline_epoch is not None and time.time() >= float(deadline_epoch):
            raise _StopRun("deadline_reached")

    progress()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config["num_threads"]))
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU substitution")
    features = torch.from_numpy(np.load(data_dir / "x.npy", allow_pickle=False))
    train_source = torch.from_numpy(np.load(data_dir / "train_source.npy", allow_pickle=False)).long()
    train_target = torch.from_numpy(np.load(data_dir / "train_target.npy", allow_pickle=False)).long()
    valid_source = torch.from_numpy(np.load(data_dir / "valid_source.npy", allow_pickle=False)).long()
    valid_target = torch.from_numpy(np.load(data_dir / "valid_target.npy", allow_pickle=False)).long()
    valid_neg = torch.from_numpy(np.load(data_dir / "valid_neg.npy", allow_pickle=False)).long()
    reverse_exposed = np.load(data_dir / "valid_reverse_in_train.npy", allow_pickle=False).astype(bool)
    self_loop = np.load(data_dir / "valid_self_loop.npy", allow_pickle=False).astype(bool)
    clean = ~reverse_exposed & ~self_loop
    if valid_source.numel() != 45557 or valid_neg.shape != (45557, 150):
        raise ValueError("Expected the complete released Sports validation set: 45557 x 150")
    graph = dgl.remove_self_loop(dgl.graph((train_source, train_target), num_nodes=features.shape[0]))
    graph, reverse_eids = _module("mm_graph_official_utils", "utils.py").to_bidirected_with_reverse_mapping(graph)
    official = _module("mm_graph_official_models", "models.py")
    model = getattr(official, method)(features.shape[1], int(config["hidden_dim"]), int(config["num_layers"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(config["lr_scheduler_step_size"]), gamma=float(config["lr_scheduler_gamma"]))
    # MLP is pointwise: zero-neighbor blocks preserve the same roots and official
    # Uniform(1) target corruption while avoiding unused neighbor expansion.
    fanout = int(config["num_of_neighbors"]) if method == "SAGE" else 0
    sampler = as_edge_prediction_sampler(
        NeighborSampler([fanout] * int(config["num_layers"])), exclude="reverse_id",
        reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1),
    )
    loader = DataLoader(
        graph, torch.arange(graph.num_edges()), sampler, batch_size=int(config["batch_size"]),
        shuffle=True, drop_last=False, num_workers=0,
    )
    provenance = {
        "official_repository": "https://github.com/mm-graph-benchmark/mm-graph-benchmark",
        "official_commit": OFFICIAL_COMMIT, "model_definition": "lp/models.py",
        "backend": "native_dgl_cpu" if device.type == "cpu" else "dgl_cpu_sampler_torch_cuda_mean_operator_port",
        "winning_paper_config_available": False,
        "result_scope": "single-seed local validation pilot; separate from published multi-run test results",
        "training_target_directed_edges": int(graph.num_edges()),
        "train_target_policy": "all edges of official simplified bidirected training graph",
        "target_exclusion": "DGL reverse_id excludes batch targets and reverse edges before neighborhood sampling",
        "negative_sampling": "official DGL Uniform(1), unfiltered target corruption",
        "validation_policy": "complete fixed official candidates; checkpoint selection by validation MRR only",
        "test_evaluated": False, "max_steps_requested": int(config["max_steps"]),
        "max_positive_targets_requested": int(config["max_steps"]) * int(config["batch_size"]),
        "official_defaults_epochs": 30, "official_defaults_layers": 3,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation_reverse_train_exposed_count": int(reverse_exposed.sum()),
        "validation_self_loop_count": int(self_loop.sum()), "validation_clean_count": int(clean.sum()),
        "split_limitation": "official bidirection makes some validation positives training edges; full validation is not strictly held out",
        "mlp_sampling_optimization": "zero-neighbor blocks; pointwise official encoder/predictor unchanged" if method == "MLP" else None,
    }
    _json(out_dir / "provenance.json", provenance)
    best = None
    epoch, batch, step = 0, 0, 0
    last_progress = 0.0
    status, complete = "training", False
    try:
        while step < int(config["max_steps"]):
            epoch += 1
            check_stop()
            model.train()
            loss_sum, examples, batch = 0.0, 0, 0
            epoch_start = time.time()
            progress(status="training", epoch=epoch, step=step, batch_in_epoch=0, batches_per_epoch=len(loader))
            # Same cap as official main.py: at most 512000 positive targets per epoch.
            # Sports's symmetrized graph can exceed this even though release train has 356202 rows.
            official_batch_cap = 1000 * 512 / int(config["batch_size"])
            for batch, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(loader, 1):
                check_stop()
                if method == "MLP":
                    root_nodes = blocks[-1].dstdata[dgl.NID]
                    x = features[root_nodes].to(device)
                else:
                    x = features[input_nodes].to(device)
                h = _encode(model, method, blocks, x)
                ps, pt = pair_graph.edges(order="eid")
                ns, nt = neg_pair_graph.edges(order="eid")
                positive = model.predictor(h[ps.to(device)] * h[pt.to(device)]).flatten()
                negative = model.predictor(h[ns.to(device)] * h[nt.to(device)]).flatten()
                scores = torch.cat((positive, negative))
                labels = torch.cat((torch.ones_like(positive), torch.zeros_like(negative)))
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                loss_sum += float(loss.detach())
                examples += positive.numel()
                if time.time() - last_progress >= 10:
                    progress(epoch=epoch, batch_in_epoch=batch, step=step, training_loss=loss_sum / batch,
                             positive_targets_seen_this_epoch=examples,
                             positive_targets_seen=step * int(config["batch_size"]))
                    last_progress = time.time()
                epoch_finished = batch == official_batch_cap or batch == len(loader)
                if epoch_finished:
                    # The scheduler unit remains one official capped epoch,
                    # never one 250-step validation interval. At total step
                    # 1000 this is scheduler epoch 1 and lr remains 5e-4.
                    scheduler.step()
                evaluate = step % int(config["eval_every_steps"]) == 0 or step == int(config["max_steps"])
                if evaluate:
                    row = {"epoch": epoch, "step": step, "training_loss": loss_sum / max(batch, 1),
                           "training_seconds_this_epoch": time.time() - epoch_start, "batches_this_epoch": batch,
                           "positive_targets_seen": step * int(config["batch_size"]),
                           "lr": scheduler.get_last_lr()[0]}
                    _checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, batch, config, best, "step_trained", step)
                    check_stop()
                    progress(status="validating", epoch=epoch, step=step, batch_in_epoch=batch)
                    valid_start = time.time()
                    embeddings = _inference(model, method, graph, features, device,
                                            int(config["inference_batch_size"]), check_stop)
                    valid, rr, ranks = _validate(model, embeddings, valid_source, valid_target, valid_neg,
                                                device, int(config["eval_batch_size"]), check_stop)
                    valid_clean = _slice_metrics(rr, ranks, clean)
                    row.update(valid=valid, valid_clean=valid_clean,
                               valid_reverse_exposed=_slice_metrics(rr, ranks, reverse_exposed),
                               valid_self_loop=_slice_metrics(rr, ranks, self_loop),
                               validation_seconds=time.time() - valid_start)
                    np.save(out_dir / f"valid_rr_step_{step:06d}.npy", rr, allow_pickle=False)
                    np.save(out_dir / f"valid_rank_step_{step:06d}.npy", ranks, allow_pickle=False)
                    np.save(out_dir / "valid_rr_final.npy", rr, allow_pickle=False)
                    np.save(out_dir / "valid_rank_final.npy", ranks, allow_pickle=False)
                    if best is None or valid["mrr"] > best["mrr"]:
                        best = {**valid, "epoch": epoch, "step": step, "clean": valid_clean}
                        np.save(out_dir / "valid_rr.npy", rr, allow_pickle=False)
                        np.save(out_dir / "valid_rank.npy", ranks, allow_pickle=False)
                        _checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, batch, config, best, "best_valid", step)
                    del embeddings
                    row["best_valid"] = best
                    with (out_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    progress(status="training", best_valid=best, final_valid=valid, final_valid_clean=valid_clean,
                             final_valid_step=step)
                    _checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, batch, config, best, "evaluation_complete", step)
                    model.train()
                if epoch_finished or step >= int(config["max_steps"]):
                    break
        status, complete = "completed", True
    except _StopRun as exc:
        status = str(exc)
    except BaseException:
        status = "failed"
        _checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, batch, config, best, status, step)
        progress(status=status, complete=False, epoch=epoch, step=step, batch_in_epoch=batch, best_valid=best)
        _json(out_dir / "metrics.json", {**state, "config": config, "provenance": provenance})
        raise
    _checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, batch, config, best, status, step)
    progress(status=status, complete=complete, epoch=epoch, step=step, batch_in_epoch=batch,
             best_valid=best, positive_targets_seen=step * int(config["batch_size"]))
    result = {**state, "config": config, "provenance": provenance}
    _json(out_dir / "metrics.json", result)
    return result
