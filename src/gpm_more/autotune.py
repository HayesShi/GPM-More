"""Finite, resumable validation search with one training subprocess at a time."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .train import _replace_file, initialize_weights, resolved_config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")
    _replace_file(temporary, path)


def _process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied is conservatively alive.
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class LockBusyError(RuntimeError):
    """A cooperating queue owns the requested OS lock."""


class ActiveChildError(RuntimeError):
    """A previous worker's training child has not exited."""


class QueueLock:
    """OS advisory lock; stale files never prevent a restart after a crash.

    Windows byte-zero locking also cooperates with filelock.WindowsFileLock.
    The locked file stays in place, so another process cannot lock a new inode.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
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
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise LockBusyError(f"Another queue or trainer holds {self.path}") from error
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def load_plan(path):
    path = Path(path).resolve()
    plan = _read(path)
    required = {"schema", "datasets", "base_config", "learning_rates", "late_lr_factors",
                "stop_after", "seeds", "wall_hours", "cpu_threads", "device"}
    optional = {"gpu_lock", "min_free_gb"}
    if set(plan) - required - optional or required - set(plan):
        raise ValueError("Autotune plan has missing or unknown fields")
    if plan["schema"] != "gpm-more-autotune-v1" or plan["seeds"] != [0, 1, 2]:
        raise ValueError("Expected autotune-v1 with tuning seed0 and confirmation seeds1,2")
    if not 1 <= len(plan["datasets"]) <= 3 or not 0 < plan["wall_hours"] <= 12:
        raise ValueError("Use one to three datasets and a positive wall budget no greater than 12 hours")
    if not 1 <= plan["cpu_threads"] <= 16 or plan["stop_after"] < 0:
        raise ValueError("Invalid CPU thread or feedback boundary setting")
    for key in ("learning_rates", "late_lr_factors"):
        values = plan[key]
        if not values or len(set(values)) != len(values) or any(value <= 0 for value in values):
            raise ValueError(f"Expected unique positive {key}")
    if len(plan["learning_rates"]) * len(plan["late_lr_factors"]) * 2 > 12:
        raise ValueError("At most twelve search candidates per dataset")
    plan["base_config"] = resolved_config(plan["base_config"])
    if not plan["base_config"]["deterministic"]:
        raise ValueError("Autotune comparisons require deterministic execution")
    if plan["stop_after"] >= plan["base_config"]["steps"]:
        raise ValueError("Feedback stop must precede the training horizon")
    plan.setdefault("min_free_gb", 2.)
    if plan["min_free_gb"] < 0:
        raise ValueError("min_free_gb must be nonnegative")
    names = []
    for dataset in plan["datasets"]:
        if set(dataset) != {"name", "data", "init_weights"}:
            raise ValueError("Each dataset needs name, data, and seed-indexed init_weights")
        name = dataset["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("Dataset names use lowercase letters, numbers, hyphens and underscores")
        names.append(name)
        if set(dataset["init_weights"]) != {"0", "1", "2"}:
            raise ValueError("Provide genuine structural-pretraining weights for each of seeds0,1,2")
        for key in ("data",):
            dataset[key] = str((path.parent / dataset[key]).resolve())
        dataset["init_weights"] = {seed: str((path.parent / value).resolve())
                                   for seed, value in dataset["init_weights"].items()}
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dataset name")
    if "gpu_lock" in plan:
        plan["gpu_lock"] = str((path.parent / plan["gpu_lock"]).resolve())
    return plan


def validate_pretraining_source(checkpoint, completion, path, dataset, seed, nodes):
    """Accept only two explicit historical structural-pretraining protocols."""
    if checkpoint.get("dataset") != dataset or checkpoint.get("seed") != seed:
        raise ValueError(f"Pretraining dataset or seed differs: {path}")
    if completion.get("complete") is not True or completion.get("dataset") != dataset or completion.get("seed") != seed:
        raise ValueError(f"Pretraining completion record differs: {path.parent}")
    schema = checkpoint.get("schema")
    if schema == "natural-pattern-checkpoint-v1":
        if (checkpoint.get("phase") != "pretrain" or not 1 <= checkpoint.get("step", 0) <= 1000
                or checkpoint.get("test_labels_read") is not False
                or completion.get("phase") != "pretrain" or completion.get("steps") != 1000
                or completion.get("best_step") != checkpoint["step"]
                or completion.get("family") != checkpoint.get("family")
                or completion.get("test_labels_read") is not False
                or str(completion.get("checkpoint", "")).replace("\\", "/").rsplit("/", 1)[-1] != path.name):
            raise ValueError(f"Structural-pretraining completion record differs: {path}")
        return {"phase": "pretrain", "selected_step": checkpoint["step"],
                "completed_pretrain_steps": completion["steps"], "family": checkpoint.get("family"),
                "source_selection": "structural_validation_risk", "test_used_for_source_selection": False,
                "test_labels_read": False}
    if schema == "paper-aligned-natural-gpm-v1":
        execution = checkpoint.get("protocol", {}).get("execution", {})
        expected = {"method": "incidence_rotation", "hidden_dim": 256, "pretrain_epochs": 100,
                    "pretrain_lr": .0003, "optimizer": "AdamW", "schedule": "constant",
                    "logical_batch_size": 256, "early_stopping": False,
                    "pretrain_roots": "all_nodes_unlabeled_full_graph_each_epoch_shuffled_without_replacement"}
        expected_updates = ((nodes + 255) // 256) * 100
        if (path.name != "latest.pt" or checkpoint.get("method") != "incidence_rotation"
                or checkpoint.get("epoch") != 100 or checkpoint.get("updates") != expected_updates
                or checkpoint.get("test_used_for_selection") is not False
                or checkpoint.get("initialization") != "independent_from_scratch_no_task_checkpoint_warmstart"
                or any(execution.get(key) != value for key, value in expected.items())
                or completion.get("method") != "incidence_rotation"
                or completion.get("pretrain_epochs_completed") != 100
                or completion.get("pretrain_updates") != expected_updates
                or completion.get("test_used_for_selection") is not False
                or not {"optimizer", "numpy_rng_state", "torch_rng_state", "cuda_rng_states", "python_rng_state"}.issubset(checkpoint)):
            raise ValueError(f"Expected the fixed final-epoch100 structural-pretraining resume state: {path}")
        return {"phase": "structural_pretraining", "completed_pretrain_epochs": 100,
                "completed_pretrain_steps": expected_updates, "selected_epoch": 100,
                "family": "incidence_rotation", "source_selection": "fixed_final_epoch100",
                "completion_recorded_checkpoint_role": "historical_validation_selected_file; actual_transfer_uses_final_resume_path",
                "test_used_for_source_selection": False,
                "pretraining_probe_rng_interaction": execution.get("probe_rng_interaction"),
                "historical_test_evaluation_after_encoder_selection": True}
    raise ValueError(f"Unsupported structural-pretraining checkpoint schema: {schema}")


def inspect_inputs(plan):
    """Validate real prepared data and all active pretrained tensor shapes on CPU."""
    import torch
    from .data import PreparedData
    from .model import ModelConfig, RelationalPatternModel
    inventory = []
    for dataset in plan["datasets"]:
        data = PreparedData(dataset["data"])
        architecture = {**plan["base_config"]["model"], "input_dim": data.x.shape[1],
                        "num_outputs": data.num_outputs}
        sources = []
        for seed in plan["seeds"]:
            path = Path(dataset["init_weights"][str(seed)])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            completion_path = path.parent / "complete.json"
            completion = _read(completion_path)
            source_metadata = validate_pretraining_source(checkpoint, completion, path,
                                                         dataset["name"], seed, len(data.x))
            model = RelationalPatternModel(ModelConfig(**architecture))
            initialization = initialize_weights(model, path, reset_head=True, seed=seed)
            ignored = initialization["ignored_inactive_keys"]
            stat = path.stat()
            sources.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "schema": checkpoint["schema"], **source_metadata,
                            "completion_path": str(completion_path),
                            "completion_recorded_checkpoint": completion["checkpoint"],
                            "pretraining_seed": seed, "fine_tuning_seed": seed,
                            "task_head_seed": 70000000 + seed, "active_tensors": len(model.state_dict()),
                            "ignored_inactive_keys": ignored, "current_training_test_labels_read": False})
            del checkpoint, model
        inventory.append({"dataset": dataset["name"], "data": dataset["data"],
                          "metadata": data.metadata, "feature_shape": list(data.x.shape),
                          "train_roots": len(data.train_ids), "val_roots": len(data.val_ids),
                          "task": data.task, "sources": sources})
    return inventory


def _number(value):
    return format(value, ".8g").replace(".", "p").replace("-", "m")


def search_jobs(plan):
    jobs = []
    # Interleave datasets and matched feedback settings for early coverage.
    for lr in plan["learning_rates"]:
        for factor in plan["late_lr_factors"]:
            for family in ("full", "late_stop"):
                for dataset in plan["datasets"]:
                    config = copy.deepcopy(plan["base_config"])
                    config.update(seed=0, lr=lr,
                        feedback={"mode": family, "alpha": 1., "stop_after": plan["stop_after"]},
                        late_lr={"after": plan["stop_after"], "factor": factor})
                    key = f"{dataset['name']}__{family}__lr{_number(lr)}__late{_number(factor)}__s0"
                    jobs.append({"id": key, "dataset": dataset["name"], "family": family,
                                 "phase": "search", "seed": 0, "config": resolved_config(config)})
    return jobs


def _metrics(output, job):
    folder = output / "runs" / job["id"]
    path = folder / "metrics.json"
    if path.exists():
        result = _read(path)
        if (folder / "config.json").exists() and _read(folder / "config.json") != job["config"]:
            raise ValueError(f"Configuration differs for existing job {job['id']}")
        return result
    return None


def freeze_selection(plan, output, jobs, state):
    """Freeze both family winners only after the complete dataset grid succeeds."""
    additions = []
    for dataset in plan["datasets"]:
        name = dataset["name"]
        searched = [job for job in jobs if job["dataset"] == name and job["phase"] == "search"]
        if not all(state["jobs"].get(job["id"], {}).get("status") == "complete" for job in searched):
            continue
        path = output / "selections" / f"{name}.json"
        if path.exists():
            selection = _read(path)
        else:
            winners = {}
            for family in ("full", "late_stop"):
                candidates = [job for job in searched if job["family"] == family]
                # Stable grid order breaks equal scores; checkpoint ties use trainer's earlier update.
                winner = max(candidates, key=lambda job:
                             _metrics(output, job)["best_validation"][_metrics(output, job)["primary_metric"]])
                winners[family] = {"job_id": winner["id"], "config": winner["config"],
                                   "metrics": _metrics(output, winner)}
            controlled = winners["late_stop"]["config"]
            counterpart = next(job for job in searched if job["family"] == "full"
                and job["config"]["lr"] == controlled["lr"]
                and job["config"]["late_lr"] == controlled["late_lr"])
            selection = {"dataset": name, "frozen_utc": _now(), "selection_seed": 0,
                         "selection_split": "validation", "candidates_per_family": len(searched) // 2,
                         "winners": winners, "late_stop_matched_full_job": counterpart["id"],
                         "confirmation_seeds": [1, 2], "test_labels_read": False}
            _write(path, selection)
        for seed in (1, 2):
            for family in ("full", "late_stop"):
                winner = selection["winners"][family]
                config = {**copy.deepcopy(winner["config"]), "seed": seed}
                additions.append({"id": winner["job_id"].rsplit("__s", 1)[0] + f"__s{seed}",
                                  "dataset": name, "family": family, "phase": "confirmation",
                                  "seed": seed, "config": config})
    existing = {job["id"] for job in jobs}
    jobs.extend(job for job in additions if job["id"] not in existing)


def _event(output, event, **values):
    row = {"utc": _now(), "event": event, **values}
    with (output / "events.jsonl").open("a", encoding="utf8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def _summary(output, plan, jobs, state):
    rows = []
    tasks = {entry["dataset"]: entry["task"] for entry in _read(output / "inventory.json")}
    primaries = {"regression": "R2", "binary_classification": "AP", "multiclass_classification": "accuracy"}
    for job in jobs:
        status = state["jobs"].get(job["id"], {})
        folder = output / "runs" / job["id"]
        metrics = _metrics(output, job) or {}
        progress = _read(folder / "progress.json") if (folder / "progress.json").exists() else {}
        if progress.get("update", 0) > metrics.get("completed_update", 0):
            metrics = {**metrics, "completed_update": progress["update"],
                       "best_update": progress["best_update"], "best_validation": progress["best_validation"],
                       "last_validation": progress["validation"], "train_seconds": progress["train_seconds"],
                       "validation_seconds": progress["validation_seconds"]}
        row = {"job_id": job["id"], "dataset": job["dataset"], "phase": job["phase"],
               "fine_tuning_seed": job["seed"], "pretraining_seed": job["seed"],
               "family": job["family"], "lr": job["config"]["lr"],
               "late_lr_factor": job["config"]["late_lr"]["factor"],
               "status": status.get("status", "pending"),
               "update": progress.get("update", metrics.get("completed_update", 0)),
               "best_update": metrics.get("best_update", progress.get("best_update")),
               "primary_metric": metrics.get("primary_metric", primaries[tasks[job["dataset"]]]),
               "train_seconds": metrics.get("train_seconds", progress.get("train_seconds")),
               "validation_seconds": metrics.get("validation_seconds", progress.get("validation_seconds")),
               "peak_gpu_allocated_mb": progress.get("peak_gpu_allocated_mb"),
               "error": status.get("error")}
        primary = row["primary_metric"]
        row["best_validation"] = metrics.get("best_validation", progress.get("best_validation", {})).get(primary)
        row["last_validation"] = metrics.get("last_validation", progress.get("validation", {})).get(primary)
        rows.append(row)
    selections = {path.stem: _read(path) for path in (output / "selections").glob("*.json")}
    result = {"utc": _now(), "status": state["status"], "pid": os.getpid(),
              "wall_seconds": state["wall_seconds"], "wall_budget_seconds": plan["wall_hours"] * 3600,
              "complete_jobs": sum(row["status"] == "complete" for row in rows),
              "failed_jobs": sum(row["status"] == "failed" for row in rows),
              "materialized_jobs": len(rows), "maximum_jobs": len(search_jobs(plan)) + 4 * len(plan["datasets"]),
              "selection_seed": 0, "confirmation_seeds": [1, 2], "test_labels_read": False,
              "selections": selections, "jobs": rows}
    _write(output / "summary.json", result)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = output / "summary.csv.tmp"
    temporary.write_text(buffer.getvalue(), encoding="utf8")
    _replace_file(temporary, output / "summary.csv")
    lines = ["# Per-dataset feedback tuning", "", f"Status: {state['status']}; "
             f"completed {result['complete_jobs']}/{result['maximum_jobs']}; "
             f"failed {result['failed_jobs']}; used {state['wall_seconds']/3600:.2f} hours.", "",
             "Search uses seed 0 validation only. Confirmation uses frozen configurations and seeds 1 and 2.",
             "Pretraining and fine-tuning use the corresponding independently pretrained seed. No test labels are read.",
             "The full-feedback control is the matched raw_sum model.", "",
             "| Dataset | Phase | Seed | Feedback | LR | Late LR factor | Status | Update | Best validation | Last validation |",
             "|---|---|---:|---|---:|---:|---|---:|---:|---:|"]
    for row in rows:
        def number(key):
            return "—" if row[key] is None else f"{row[key]:.6f}"
        lines.append(f"| {row['dataset']} | {row['phase']} | {row['fine_tuning_seed']} | {row['family']} "
                     f"| {row['lr']:g} | {row['late_lr_factor']:g} | {row['status']} | {row['update']} "
                     f"| {number('best_validation')} | {number('last_validation')} |")
    for name, selection in selections.items():
        lines.extend(["", f"Frozen {name}: full winner `{selection['winners']['full']['job_id']}`; "
                      f"late-stop winner `{selection['winners']['late_stop']['job_id']}`.",
                      f"Same-LR/same-late-LR seed-0 full counterpart: `{selection['late_stop_matched_full_job']}`."])
    (output / "SUMMARY.md.tmp").write_text("\n".join(lines) + "\n", encoding="utf8")
    _replace_file(output / "SUMMARY.md.tmp", output / "SUMMARY.md")
    return result


def run_queue(plan_path, output_dir, *, _executor=None, _inspect=inspect_inputs, _package_source=None):
    """Run or resume an immutable finite plan; completed jobs are never relaunched."""
    plan, output = load_plan(plan_path), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("runs", "configs", "logs", "selections", "failures"):
        (output / name).mkdir(exist_ok=True)
    gpu_lock = plan.get("gpu_lock", str(output / ".gpu.lock"))
    with QueueLock(output / ".queue.lock"), QueueLock(gpu_lock):
        lease_path = Path(str(gpu_lock) + ".lease.json")
        if lease_path.exists():
            lease = _read(lease_path)
            if lease.get("child_pid") and _process_alive(lease["child_pid"]):
                raise ActiveChildError(f"Previous GPU training child {lease['child_pid']} is still running")
        if (output / "plan.json").exists():
            if _read(output / "plan.json") != plan:
                raise ValueError("Plan differs from the frozen queue plan; use a new output directory")
            inventory = _read(output / "inventory.json")
        else:
            inventory = _inspect(plan)
            _write(output / "inventory.json", inventory)
            _write(output / "plan.json", plan)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            shutil.copytree(_package_source or Path(__file__).resolve().parent, snapshot / "gpm_more",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            _write(output / "code_snapshot_manifest.json", {
                "created_utc": _now(), "package": "gpm_more", "files": [
                    {"path": str(path.relative_to(snapshot)), "bytes": path.stat().st_size}
                    for path in sorted(snapshot.rglob("*.py"))]})
        state = _read(output / "state.json") if (output / "state.json").exists() else {
            "schema": "gpm-more-queue-state-v1", "wall_seconds": 0., "jobs": {}, "status": "ready"}
        for key, record in state["jobs"].items():
            if record.get("status") == "running" and record.get("child_pid") and _process_alive(record["child_pid"]):
                raise ActiveChildError(f"Previous training child {record['child_pid']} is still running for {key}")
        # Conservatively account for any unobserved interval after a worker crash.
        if state["status"] == "running" and "updated_utc" in state:
            state["wall_seconds"] += max(0., (datetime.now(timezone.utc) -
                datetime.fromisoformat(state["updated_utc"])).total_seconds())
        base_wall, invocation_start = state["wall_seconds"], time.monotonic()
        jobs = search_jobs(plan)
        for job in jobs:
            metrics = _metrics(output, job)
            if metrics and metrics["complete"]:
                state["jobs"].setdefault(job["id"], {})["status"] = "complete"
        freeze_selection(plan, output, jobs, state)
        for job in jobs:
            metrics = _metrics(output, job)
            if metrics and metrics["complete"]:
                state["jobs"].setdefault(job["id"], {})["status"] = "complete"
        datasets = {dataset["name"]: dataset for dataset in plan["datasets"]}
        stop_file = output / "STOP_REQUESTED"
        def persist():
            state.update(wall_seconds=base_wall + time.monotonic() - invocation_start,
                         updated_utc=_now(), pid=os.getpid())
            _write(output / "state.json", state)
            return _summary(output, plan, jobs, state)
        state["status"] = "running"
        _event(output, "queue_start", pid=os.getpid(), max_jobs=len(search_jobs(plan)) + 4 * len(datasets))
        persist()
        while True:
            remaining = plan["wall_hours"] * 3600 - (base_wall + time.monotonic() - invocation_start)
            if stop_file.exists() or remaining <= 0:
                state["status"] = "stopped" if stop_file.exists() else "wall_budget"
                break
            if shutil.disk_usage(output).free < plan["min_free_gb"] * 2**30:
                state["status"] = "low_disk"
                break
            pending = [job for job in jobs if state["jobs"].get(job["id"], {}).get("status")
                       not in {"complete", "failed"}]
            if not pending:
                state["status"] = "complete" if len(jobs) == len(search_jobs(plan)) + 4 * len(datasets) \
                    and all(record["status"] == "complete" for record in state["jobs"].values()) else "failed"
                break
            job = pending[0]
            dataset = datasets[job["dataset"]]
            folder = output / "runs" / job["id"]
            config_path = output / "configs" / (job["id"] + ".json")
            _write(config_path, job["config"])
            record = state["jobs"].setdefault(job["id"], {})
            record.update(status="running", started_utc=_now())
            state["active_job"] = job["id"]
            checkpoint = folder / "latest.pt"
            if folder.exists() and any(folder.iterdir()) and not checkpoint.exists():
                # Preserve an initialization interrupted before its first checkpoint.
                archive = output / "failures" / f"{job['id']}__interrupted_{time.time_ns()}"
                folder.rename(archive)
                _event(output, "initialization_archive", job=job["id"], path=str(archive))
            command = [sys.executable, "-m", "gpm_more.cli", "train", "--data", dataset["data"],
                       "--output", str(folder), "--config", str(config_path), "--device", plan["device"],
                       "--stop-file", str(stop_file), "--max-wall-seconds", str(max(.001, remaining))]
            if checkpoint.exists():
                command.extend(["--resume", str(checkpoint)])
            else:
                command.extend(["--init-weights", dataset["init_weights"][str(job["seed"])] , "--reset-head"])
            log_path = output / "logs" / (job["id"] + ".log")
            _event(output, "job_start", job=job["id"], resume=checkpoint.exists())
            persist()
            if _executor is not None:
                returncode = _executor(job, folder, config_path, command)
            else:
                environment = {**os.environ, "PYTHONUNBUFFERED": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "OMP_NUM_THREADS": str(plan["cpu_threads"]), "MKL_NUM_THREADS": str(plan["cpu_threads"]),
                    "OPENBLAS_NUM_THREADS": str(plan["cpu_threads"])}
                source = str(snapshot)
                environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
                with log_path.open("a", encoding="utf8") as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                        env=environment, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    record["child_pid"] = process.pid
                    _write(lease_path, {"queue": str(output), "worker_pid": os.getpid(),
                                        "child_pid": process.pid, "job": job["id"], "utc": _now()})
                    persist()
                    while process.poll() is None:
                        time.sleep(2.)
                        persist()
                    returncode = process.returncode
                    _write(lease_path, {"queue": str(output), "child_pid": None, "utc": _now()})
            metrics = _metrics(output, job)
            record["ended_utc"] = _now()
            record["returncode"] = returncode
            if returncode == 0 and metrics and metrics["complete"]:
                record["status"] = "complete"
                _event(output, "job_complete", job=job["id"], best_validation=metrics["best_validation"])
                freeze_selection(plan, output, jobs, state)
            elif returncode == 0 and metrics and not metrics["complete"]:
                record["status"] = "paused"
                state["status"] = metrics.get("stop_reason") or "interrupted"
                _event(output, "job_paused", job=job["id"], update=metrics["completed_update"])
                break
            else:
                record.update(status="failed", error=f"Child exit {returncode}; see {log_path.name}")
                _write(output / "failures" / (job["id"] + ".json"), {**record, "job": job})
                _event(output, "job_failed", job=job["id"], returncode=returncode)
            persist()
        state.pop("active_job", None)
        result = persist()
        _event(output, "queue_exit", status=state["status"], completed=result["complete_jobs"])
        return result
