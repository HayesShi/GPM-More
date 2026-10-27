"""Portable supervised training with forward-identical relation feedback.

This runner starts a new experiment or resumes its own complete checkpoint.
Loading historical raw_sum weights initializes a new optimizer; it does not
recreate the optimizer history of a published historical experiment.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import PreparedData
from .feedback import FeedbackSchedule
from .model import ModelConfig, RelationalPatternModel


DEFAULTS = {
    "steps": 1000, "seed": 0, "lr": 3e-4, "weight_decay": .01,
    "warmup": 100, "batch_size": 128, "micro_batch_size": 32,
    "train_budgets": [4, 8, 12, 16], "eval_budget": 12,
    "eval_every": 50, "grad_clip": 1., "clip_policy": "common_full",
    "feedback": {"mode": "full", "alpha": 1., "stop_after": 400},
    "late_lr": {"after": 400, "factor": 1.}, "model": {},
    "label_smoothing": .05,
}


def resolved_config(path_or_mapping):
    supplied = (dict(path_or_mapping) if isinstance(path_or_mapping, Mapping)
                else json.loads(Path(path_or_mapping).read_text(encoding="utf8")))
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training configuration keys: {sorted(unknown)}")
    config = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if key in {"feedback", "late_lr"}:
            config[key].update(value)
        else:
            config[key] = value
    for key in ("steps", "batch_size", "micro_batch_size", "eval_every"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["seed"], int) or not isinstance(config["warmup"], int):
        raise ValueError("seed and warmup must be integers")
    if config["warmup"] < 0 or config["lr"] <= 0 or config["grad_clip"] <= 0:
        raise ValueError("warmup must be nonnegative; lr/grad_clip must be positive")
    if config["weight_decay"] < 0 or not 0 <= config["label_smoothing"] <= 1:
        raise ValueError("Invalid weight decay or label smoothing")
    if config["clip_policy"] != "common_full":
        raise ValueError("This runner implements clip_policy='common_full'")
    if (not config["train_budgets"]
            or any(not isinstance(k, int) or not 1 <= k <= 16 for k in config["train_budgets"])
            or not isinstance(config["eval_budget"], int) or not 1 <= config["eval_budget"] <= 16):
        raise ValueError("Walk budgets must be in [1,16]")
    if (not isinstance(config["late_lr"]["after"], int)
            or config["late_lr"]["after"] < 0 or config["late_lr"]["factor"] <= 0):
        raise ValueError("Invalid late learning-rate control")
    FeedbackSchedule(**config["feedback"])
    # Normalize tuples supplied by Python callers to the on-disk JSON form.
    return json.loads(json.dumps(config, allow_nan=False))


def _rng_capture(rng, device):
    return {"numpy_generator": copy.deepcopy(rng.bit_generator.state),
            "python": random.getstate(), "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if device.type == "cuda" else []}


def _rng_restore(state, rng, device):
    rng.bit_generator.state = copy.deepcopy(state["numpy_generator"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _rng_equal(first, second):
    return (first["numpy_generator"] == second["numpy_generator"]
            and first["python"] == second["python"]
            and torch.equal(first["torch_cpu"], second["torch_cpu"])
            and len(first["torch_cuda"]) == len(second["torch_cuda"])
            and all(torch.equal(a, b) for a, b in zip(first["torch_cuda"], second["torch_cuda"])))


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_cpu_tree(payload), temporary)
    temporary.replace(path)


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _optimizer(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    warmup = min(config["warmup"], config["steps"])

    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1., (step - warmup) / max(1, config["steps"] - warmup))
        return .5 * (1. + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _loss(prediction, target, task, smoothing):
    if task == "multiclass_classification":
        return F.cross_entropy(prediction, target.long().reshape(-1), label_smoothing=smoothing)
    if task == "binary_classification":
        return F.binary_cross_entropy_with_logits(prediction.reshape(-1), target.float().reshape(-1))
    if task == "regression":
        return F.mse_loss(prediction.reshape(-1), target.float().reshape(-1))
    raise ValueError(f"Unsupported task: {task}")


def task_metrics(prediction, target, task):
    prediction, target = np.asarray(prediction), np.asarray(target).reshape(-1)
    if task == "multiclass_classification":
        return {"accuracy": float(np.mean(prediction.argmax(-1) == target))}
    prediction = prediction.reshape(-1)
    if task == "regression":
        error = prediction.astype(np.float64) - target.astype(np.float64)
        mse = float(np.mean(error ** 2))
        variance = float(np.mean((target - target.mean()) ** 2))
        r2 = 1. - mse / variance if variance > 0 else float(mse == 0)
        return {"R2": r2, "MSE": mse}
    if task == "binary_classification":
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as error:
            raise ImportError("Binary AP/AUC metrics require scikit-learn; install gpm-more[data]") from error
        metrics = {"AP": float(average_precision_score(target, prediction)),
                   "accuracy": float(np.mean((prediction >= 0) == target))}
        if len(np.unique(target)) == 2:
            metrics["ROC_AUC"] = float(roc_auc_score(target, prediction))
        return metrics
    raise ValueError(f"Unsupported task: {task}")


def _forward(model, data, x, nodes, budget, actions, alpha):
    paths, relations, action_ids = data.batch(nodes, budget, actions=actions)
    return model(x, torch.as_tensor(paths, dtype=torch.long, device=x.device),
                 torch.as_tensor(relations, dtype=torch.float32, device=x.device),
                 action_ids=torch.as_tensor(action_ids, dtype=torch.long, device=x.device),
                 feedback_alpha=alpha)["prediction"]


def _backward(model, data, x, nodes, budget, actions, alpha, config):
    model.train()
    total = 0.
    for offset in range(0, len(nodes), config["micro_batch_size"]):
        ids = nodes[offset:offset + config["micro_batch_size"]]
        chosen = None if actions is None else actions[offset:offset + len(ids)]
        prediction = _forward(model, data, x, ids, budget, chosen, alpha)
        target = torch.as_tensor(data.targets(ids), device=x.device)
        loss = _loss(prediction, target, data.task, config["label_smoothing"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        weight = len(ids) / len(nodes)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
    return total


@torch.no_grad()
def evaluate(model, data, x, config):
    model.eval()
    values = []
    for start in range(0, len(data.val_ids), config["micro_batch_size"]):
        nodes = data.val_ids[start:start + config["micro_batch_size"]]
        values.append(_forward(model, data, x, nodes, config["eval_budget"], None, 1.).cpu().numpy())
    prediction = np.concatenate(values)
    metrics = task_metrics(prediction, data.targets(data.val_ids), data.task)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite validation metrics")
    return metrics, prediction


def _identity(data):
    return {"metadata": data.metadata, "feature_shape": list(data.x.shape),
            "train_ids": data.train_ids.tolist(), "val_ids": data.val_ids.tolist(),
            "task": data.task, "num_outputs": data.num_outputs}


def _write_curves(path, history):
    rows = [{"update": row["update"], **row["validation"]} for row in history]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train(data_dir, output_dir, config_path, device="cpu", resume=None,
          init_weights=None, *, max_updates=None):
    """Train or resume one explicit recipe without reading test labels.

    ``max_updates`` is an optional absolute update boundary for interruption
    and resume checks. It does not change the schedule horizon or introduce
    an additional validation/model-selection point. Resume uses the same
    resolved configuration, architecture, metadata and train/validation IDs.
    """
    config = resolved_config(config_path)
    if resume is not None and init_weights is not None:
        raise ValueError("Choose resume or weight initialization, not both")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    data = PreparedData(data_dir)
    identity = _identity(data)
    architecture = dict(config["model"])
    for key, expected in [("input_dim", data.x.shape[1]), ("num_outputs", data.num_outputs)]:
        if key in architecture and architecture[key] != expected:
            raise ValueError(f"Model {key} differs from the prepared data")
        architecture[key] = expected
    model_config = ModelConfig(**architecture)
    model_identity = json.loads(json.dumps(model_config.to_dict()))
    output = Path(output_dir)
    if resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new output directory or explicitly resume its checkpoint")
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume is not None else None
    if saved is not None:
        if (saved.get("format") != "gpm-more-checkpoint-v1" or saved["config"] != config
                or saved["model_config"] != model_identity or saved["data_identity"] != identity):
            raise ValueError("Resume configuration, architecture, or dataset identity differs")
        if saved["device_type"] != device.type:
            raise ValueError("Exact resume requires the same device type")
        if (output / "latest.pt").exists():
            current = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
            if current["update"] != saved["update"] or current["config"] != saved["config"]:
                raise ValueError("Output directory contains a different or newer checkpoint")
        for name, expected in [("config.json", config), ("dataset.json", identity)]:
            if (output / name).exists() and json.loads((output / name).read_text(encoding="utf8")) != expected:
                raise ValueError(f"Existing {name} differs from the resumed experiment")
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "config.json", config)
    _save_json(output / "dataset.json", identity)
    _save_json(output / "model_config.json", model_identity)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(config["seed"])
    model = RelationalPatternModel(model_config).to(device)
    optimizer, scheduler = _optimizer(model, config)
    policy = FeedbackSchedule(**config["feedback"])
    x = torch.as_tensor(data.x, dtype=torch.float32, device=device)
    primary = {"regression": "R2", "binary_classification": "AP",
               "multiclass_classification": "accuracy"}[data.task]
    update, history, local_steps = 0, [], []
    train_seconds, validation_seconds = 0., 0.
    best_state, best_metrics, best_update = None, None, 0
    initialization = {"mode": "fresh"}
    if saved is not None:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        for slots in optimizer.state.values():
            for key, value in slots.items():
                if torch.is_tensor(value) and key != "step":
                    slots[key] = value.to(device)
        scheduler.load_state_dict(saved["scheduler"])
        update, history, local_steps = saved["update"], saved["history"], saved["local_steps"]
        best_state, best_metrics, best_update = saved["best_state"], saved["best_metrics"], saved["best_update"]
        train_seconds, validation_seconds = saved["train_seconds"], saved["validation_seconds"]
        initialization = saved["initialization"]
        _rng_restore(saved["rng"], rng, device)
    else:
        if init_weights is not None:
            weights = torch.load(init_weights, map_location="cpu", weights_only=True)
            state = weights.get("state_dict", weights.get("model_state", weights))
            ignored = model.load_historical_state(state)
            initialization = {"mode": "weights_only", "source": str(Path(init_weights).resolve()),
                              "ignored_inactive_keys": ignored, "optimizer_history_imported": False}
        started = time.perf_counter()
        previous_rng = _rng_capture(rng, device)
        metrics, prediction = evaluate(model, data, x, config)
        _rng_restore(previous_rng, rng, device)
        _sync(device)
        validation_seconds += time.perf_counter() - started
        history.append({"update": 0, "validation": metrics})
        best_state, best_metrics = _cpu_tree(model.state_dict()), metrics
        np.savez_compressed(output / "validation_000000.npz", node_ids=data.val_ids,
                            prediction=prediction, target=data.targets(data.val_ids))
    start_update = update
    end = config["steps"] if max_updates is None else min(int(max_updates), config["steps"])
    if end < update:
        raise ValueError("Requested stopping boundary precedes the resumed update")
    wall_start = time.perf_counter()

    def payload():
        return {"format": "gpm-more-checkpoint-v1", "update": update, "config": config,
                "model_config": model_identity, "data_identity": identity, "device_type": device.type,
                "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "rng": _rng_capture(rng, device),
                "history": history, "local_steps": local_steps, "best_state": best_state,
                "best_metrics": best_metrics, "best_update": best_update,
                "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                "initialization": initialization}

    for update in range(update + 1, end + 1):
        started = time.perf_counter()
        nodes = rng.choice(data.train_ids, config["batch_size"],
                           replace=len(data.train_ids) < config["batch_size"])
        budget = int(rng.choice(config["train_budgets"]))
        actions = rng.integers(0, 2, len(nodes)) if budget == 16 else None
        alpha = policy(update)
        before = _rng_capture(rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _backward(model, data, x, nodes, budget, actions, 1., config)
        norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
        coefficient = (config["grad_clip"] / (norm + 1e-6)).clamp(max=1.)
        after_full = _rng_capture(rng, device)
        if alpha != 1.:
            optimizer.zero_grad(set_to_none=True)
            _rng_restore(before, rng, device)
            loss = _backward(model, data, x, nodes, budget, actions, alpha, config)
            if not _rng_equal(after_full, _rng_capture(rng, device)):
                raise AssertionError("Paired backward consumed a different one-pass RNG stream")
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(coefficient.to(parameter.grad))
        factor = config["late_lr"]["factor"] if update > config["late_lr"]["after"] else 1.
        native_lrs = [group["lr"] for group in optimizer.param_groups]
        try:
            for group, lr in zip(optimizer.param_groups, native_lrs):
                group["lr"] = lr * factor
            optimizer.step()
        finally:
            for group, lr in zip(optimizer.param_groups, native_lrs):
                group["lr"] = lr
        scheduler.step()
        _sync(device)
        train_seconds += time.perf_counter() - started
        local_steps.append({"update": update, "budget": budget, "feedback_alpha": alpha,
                            "loss": loss, "full_gradient_norm": float(norm),
                            "common_clip_coefficient": float(coefficient),
                            "effective_lr": native_lrs[0] * factor,
                            "logical_backward_passes": 1 if alpha == 1. else 2})
        if update % config["eval_every"] == 0 or update == config["steps"]:
            started = time.perf_counter()
            previous_rng = _rng_capture(rng, device)
            metrics, prediction = evaluate(model, data, x, config)
            _rng_restore(previous_rng, rng, device)
            _sync(device)
            validation_seconds += time.perf_counter() - started
            history.append({"update": update, "validation": metrics})
            if metrics[primary] > best_metrics[primary]:
                best_state, best_metrics, best_update = _cpu_tree(model.state_dict()), metrics, update
                _save_checkpoint(output / "best.pt", {"model_state": best_state,
                    "model_config": model_identity, "selected_update": best_update,
                    "validation": best_metrics, "config": config})
            np.savez_compressed(output / f"validation_{update:06d}.npz", node_ids=data.val_ids,
                                prediction=prediction, target=data.targets(data.val_ids))
            _save_checkpoint(output / "latest.pt", payload())
            _write_curves(output / "curves.csv", history)
            print(json.dumps({"update": update, "validation": metrics,
                              "best_update": best_update}, allow_nan=False), flush=True)
    # An intentional intermediate interruption saves complete state but does
    # not create an extra validation point or change best-checkpoint selection.
    _save_checkpoint(output / "latest.pt", payload())
    if not (output / "best.pt").exists():
        _save_checkpoint(output / "best.pt", {"model_state": best_state,
            "model_config": model_identity, "selected_update": best_update,
            "validation": best_metrics, "config": config})
    _write_curves(output / "curves.csv", history)
    _save_json(output / "steps.json", local_steps)
    result = {"complete": update == config["steps"], "completed_update": update,
              "new_updates": update - start_update, "schedule_horizon": config["steps"],
              "primary_metric": primary, "best_update": best_update, "best_validation": best_metrics,
              "last_validation_update": history[-1]["update"], "last_validation": history[-1]["validation"],
              "train_seconds": train_seconds, "validation_seconds": validation_seconds,
              "this_call_wall_seconds": time.perf_counter() - wall_start,
              "logical_backward_passes_total": sum(row["logical_backward_passes"] for row in local_steps),
              "test_labels_read": False, "initialization": initialization}
    _save_json(output / "metrics.json", result)
    return result
