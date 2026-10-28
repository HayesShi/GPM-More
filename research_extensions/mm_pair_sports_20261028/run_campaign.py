"""Finite seed-0 Sports validation campaign; no test performance is read.

The official bidirected context is retained. The known reverse-edge exposure
is reported separately using validation-only diagnostic masks.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn.functional as F

from pair_model import PairSportsModel
from walks import CSRGraph, sample_walks
from gpm_more.autotune import QueueLock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mm_pair_sports_20261028"
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"
LOCK = Path(os.environ.get("GPM_GPU_LOCK", str(ROOT / "runs/.gpu0.lock")))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for retry in range(6):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(0.1 * (retry + 1))


def event(kind, **kw):
    row = {"time_unix": time.time(), "event": kind, **kw}
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


class BudgetStop(Exception):
    pass


def guard(deadline=None):
    if (OUT / "STOP_REQUESTED").exists():
        raise BudgetStop("STOP_REQUESTED")
    if deadline is not None and time.time() >= deadline:
        raise BudgetStop("absolute_campaign_deadline")
    if shutil.disk_usage(OUT).free < 2 * 1024 ** 3:
        raise BudgetStop("less_than_2GiB_free_disk")


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_data():
    arrays = {k: np.load(DATA / f"{k}.npy", mmap_mode="r") for k in
              ("x", "train_csr_indptr", "train_csr_indices", "valid_source",
               "valid_target", "valid_neg", "valid_reverse_in_train", "valid_self_loop")}
    arrays["graph"] = CSRGraph(arrays["train_csr_indptr"], arrays["train_csr_indices"])
    arrays["edge_source"] = np.repeat(np.arange(len(arrays["x"])), np.diff(arrays["train_csr_indptr"]))
    arrays["edge_target"] = np.asarray(arrays["train_csr_indices"])
    return arrays


def rank_metrics(scores, data):
    positive = scores[:, :1]
    rank = 1 + 0.5 * ((scores[:, 1:] >= positive).sum(1) + (scores[:, 1:] > positive).sum(1))
    return metrics_from_rank(rank, data), rank


def metrics_from_rank(rank, data):
    masks = {"full": np.ones(len(rank), dtype=bool),
             "clean": ~(data["valid_reverse_in_train"] | data["valid_self_loop"]),
             "reverse_exposed": np.asarray(data["valid_reverse_in_train"])}
    result = {}
    for name, mask in masks.items():
        r = rank[mask]
        result[name] = {"n": len(r), "mrr": float((1 / r).mean()),
                        **{f"hits@{k}": float((r <= k).mean()) for k in (1, 3, 10)}}
    return result


def ensure_bank(data, walks=16):
    bank = OUT / "eval_bank"
    spec = bank / "complete.json"
    if spec.exists():
        return bank
    bank.mkdir(exist_ok=True)
    n = len(data["x"])
    paths = np.lib.format.open_memmap(bank / "paths.npy", mode="w+", dtype="int32", shape=(n, walks, 9))
    rel = np.lib.format.open_memmap(bank / "relations.npy", mode="w+", dtype="float16", shape=(n, walks, 93))
    valid = np.lib.format.open_memmap(bank / "valid.npy", mode="w+", dtype="bool", shape=(n, walks, 9))
    t0 = time.time()
    for begin in range(0, n, 256):
        guard()
        end = min(n, begin + 256)
        obs = sample_walks(data["graph"], np.arange(begin, end), walks, 881000 + begin)
        paths[begin:end], rel[begin:end], valid[begin:end] = obs.paths, obs.relations, obs.occurrence_valid
    paths.flush(); rel.flush(); valid.flush()
    write_json(spec, {"nodes": n, "walks": walks, "length": 8, "seed_rule": "881000+chunk_start",
                      "chunk_size": 256, "wall_seconds": time.time() - t0,
                      "context": "official bidirected train graph; reverse validation exposure retained and reported",
                      "queries": n * walks * 44})
    return bank


def load_bank(bank):
    return {k: np.load(bank / f"{k}.npy", mmap_mode="r") for k in ("paths", "relations", "valid")}


def checkpoint(path, model, optimizer, step, best, seconds):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
             "best": best, "train_seconds": seconds, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


@torch.no_grad()
def evaluate_pair(model, data, bank, x, use_joint, out, step, deadline, final=False):
    model.eval()
    t0 = time.time()
    cache = None
    n = len(x)
    for begin in range(0, n, 128):
        guard(deadline)
        end = min(n, begin + 128)
        encoded = model.encode(x,
            torch.as_tensor(np.array(bank["paths"][begin:end]), device=x.device, dtype=torch.long),
            torch.as_tensor(np.array(bank["relations"][begin:end]), device=x.device, dtype=torch.float32),
            torch.as_tensor(np.array(bank["valid"][begin:end]), device=x.device))
        keys = tuple(encoded) if use_joint else ("z",)
        if cache is None:
            cache = {k: torch.empty((n, *encoded[k].shape[1:]), device=x.device, dtype=encoded[k].dtype) for k in keys}
        for k in keys:
            cache[k][begin:end] = encoded[k]
        if begin % 2048 == 0:
            write_json(out / "progress.json", {"phase": "encode_validation", "step": step, "nodes": end, "total_nodes": n})
    scores = np.empty((len(data["valid_source"]), 151), dtype=np.float32)
    base_scores = np.empty_like(scores) if use_joint and final else None
    for begin in range(0, len(scores), 32):
        guard(deadline)
        end = min(len(scores), begin + 32)
        src = torch.as_tensor(np.repeat(data["valid_source"][begin:end], 151), device=x.device)
        dst = torch.as_tensor(np.concatenate((data["valid_target"][begin:end, None], data["valid_neg"][begin:end]), axis=1).reshape(-1), device=x.device)
        scores[begin:end] = model.score(cache, src, dst, use_joint).reshape(end - begin, 151).cpu().numpy()
        if base_scores is not None:
            base_scores[begin:end] = model.score(cache, src, dst, False).reshape(end - begin, 151).cpu().numpy()
        if begin % 2048 == 0:
            write_json(out / "progress.json", {"phase": "rank_validation", "step": step, "queries": end, "total_queries": len(scores)})
    metrics, rank = rank_metrics(scores, data)
    metrics["eval_seconds"] = time.time() - t0
    if base_scores is not None:
        base_metrics, base_rank = rank_metrics(base_scores, data)
        write_json(out / "fixed_weight_no_joint.json", {"kind": "final checkpoint fixed-weight forward intervention",
                                                        "validation": base_metrics, "test_evaluated": False})
        np.save(out / "no_joint_final_rr.npy", 1 / base_rank)
    del cache
    torch.cuda.empty_cache()
    return metrics, rank, scores


def run_pair(method, data, bank, config, deadline):
    out = OUT / "runs" / method
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and json.loads(metrics_path.read_text())["complete"]:
        event("skip_completed", method=method)
        return json.loads(metrics_path.read_text())
    guard(deadline)
    set_seed(config["seed"])
    x = torch.tensor(np.array(data["x"]), device="cuda", dtype=torch.float32)
    model = PairSportsModel().cuda()
    init_path = OUT / "pair_initial.pt"
    if not init_path.exists():
        torch.save(model.state_dict(), init_path)
    model.load_state_dict(torch.load(init_path, map_location="cuda", weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best = {"full_mrr": -1.0, "step": None}
    start_step, train_seconds = 0, 0.0
    if (out / "latest.pt").exists():
        previous = torch.load(out / "latest.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        start_step, best = previous["step"], previous["best"]
        train_seconds = previous["train_seconds"]
        torch.set_rng_state(previous["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in previous["cuda_rng"]])
    write_json(out / "config.json", {**config, "method": method, "use_joint": method == "pair_joint",
                                    "parameters": sum(p.numel() for p in model.parameters())})
    order = np.random.RandomState(44001).permutation(len(data["edge_source"]))
    use_joint = method == "pair_joint"
    rows, final_metrics, step = [], None, start_step
    completed_step = start_step
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    try:
        prior_rows = []
        if (out / "history.jsonl").exists():
            prior_rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines() if line]
        # A checkpoint is saved before a full validation. Finish a pending
        # evaluation on resume instead of treating its optimizer step as proof
        # that validation (especially the final one) completed.
        if start_step > 0 and (start_step % config["eval_every_steps"] == 0 or start_step == config["max_steps"]):
            previous_eval = next((r for r in prior_rows if r["step"] == start_step), None)
            if previous_eval is None:
                vm, rank, scores = evaluate_pair(model, data, bank, x, use_joint, out, start_step, deadline,
                                                final=start_step == config["max_steps"])
                row = {"step": start_step, "positive_examples": start_step*config["batch_size"],
                       "validation": vm, "train_seconds": train_seconds, "resumed_pending_evaluation": True}
                with (out / "history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if vm["full"]["mrr"] > best["full_mrr"]:
                    best = {"full_mrr": vm["full"]["mrr"], "step": start_step, "validation": vm}
                    torch.save({"model": model.state_dict(), "step": start_step}, out / "best.pt")
                    np.save(out / "valid_rr.npy", 1 / rank)
                    np.save(out / "valid_scores.npy", scores)
                if start_step == config["max_steps"]:
                    final_metrics = vm
                    np.save(out / "valid_rr_final.npy", 1 / rank)
                    np.save(out / "valid_scores_final.npy", scores)
                checkpoint(out / "latest.pt", model, optimizer, start_step, best, train_seconds)
            elif start_step == config["max_steps"]:
                final_metrics = previous_eval["validation"]
        for step in range(start_step + 1, config["max_steps"] + 1):
            guard(deadline)
            t0 = time.time()
            model.train()
            offset = (step - 1) * config["batch_size"]
            chosen = order[np.arange(offset, offset + config["batch_size"]) % len(order)]
            src, dst = data["edge_source"][chosen], data["edge_target"][chosen]
            neg = np.random.RandomState(550000 + step).randint(len(x), size=len(src))
            roots, inverse = np.unique(np.concatenate((src, dst, neg)), return_inverse=True)
            b = len(src)
            obs = sample_walks(data["graph"], roots, config["walks"], 660000 + step,
                               excluded_edges=np.stack((src, dst)))
            encoded = model.encode(x,
                torch.as_tensor(obs.paths, device="cuda", dtype=torch.long),
                torch.as_tensor(obs.relations, device="cuda", dtype=torch.float32),
                torch.as_tensor(obs.occurrence_valid, device="cuda"))
            inv = torch.as_tensor(inverse, device="cuda", dtype=torch.long)
            pos_score = model.score(encoded, inv[:b], inv[b:2*b], use_joint)
            neg_score = model.score(encoded, inv[:b], inv[2*b:], use_joint)
            score = torch.cat((pos_score, neg_score))
            target = torch.cat((torch.ones_like(pos_score), torch.zeros_like(neg_score)))
            loss = F.binary_cross_entropy_with_logits(score, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step}")
            warmup = min(1.0, step / config["warmup_steps"])
            for group in optimizer.param_groups:
                group["lr"] = config["lr"] * warmup
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            if not torch.isfinite(grad):
                raise RuntimeError(f"Nonfinite gradient at step {step}")
            optimizer.step()
            completed_step = step
            train_seconds += time.time() - t0
            if step % 50 == 0 or step == 1:
                write_json(out / "progress.json", {"phase": "training", "method": method, "step": step,
                    "max_steps": config["max_steps"], "positive_examples": step*b,
                    "loss": float(loss), "grad_norm": float(grad), "joint_scale": float(model.joint_scale),
                    "train_seconds": train_seconds, "updated_unix": time.time()})
            del encoded, score, pos_score, neg_score, loss
            if step % 250 == 0:
                checkpoint(out / "latest.pt", model, optimizer, step, best, train_seconds)
            if step % config["eval_every_steps"] == 0 or step == config["max_steps"]:
                checkpoint(out / "latest.pt", model, optimizer, step, best, train_seconds)
                vm, rank, scores = evaluate_pair(model, data, bank, x, use_joint, out, step, deadline,
                                                  final=step == config["max_steps"])
                row = {"step": step, "positive_examples": step*b, "validation": vm,
                       "train_seconds": train_seconds, "joint_scale": float(model.joint_scale)}
                with (out / "history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                rows.append(row)
                event("validation", method=method, **row)
                if vm["full"]["mrr"] > best["full_mrr"]:
                    best = {"full_mrr": vm["full"]["mrr"], "step": step, "validation": vm}
                    torch.save({"model": model.state_dict(), "step": step}, out / "best.pt")
                    np.save(out / "valid_rr.npy", 1 / rank)
                    np.save(out / "valid_scores.npy", scores)
                if step == config["max_steps"]:
                    final_metrics = vm
                    np.save(out / "valid_rr_final.npy", 1 / rank)
                    np.save(out / "valid_scores_final.npy", scores)
                checkpoint(out / "latest.pt", model, optimizer, step, best, train_seconds)
        if final_metrics is None:
            raise RuntimeError("Training cannot complete without full final validation")
        result = {"method": method, "complete": True, "step": completed_step, "best": best,
                  "final_validation": final_metrics, "train_seconds": train_seconds,
                  "wall_seconds_this_execution": time.time() - start,
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                  "test_evaluated": False, "seed": 0}
    except BudgetStop as e:
        checkpoint(out / "latest.pt", model, optimizer, completed_step, best, train_seconds)
        result = {"method": method, "complete": False, "reason": str(e), "step": completed_step,
                  "best": best, "train_seconds": train_seconds, "test_evaluated": False}
    write_json(metrics_path, result)
    del model, optimizer, x
    gc.collect(); torch.cuda.empty_cache()
    return result


def preflight(data):
    set_seed(0)
    out = OUT / "preflight.json"
    x = torch.tensor(np.array(data["x"]), device="cuda", dtype=torch.float32)
    model = PairSportsModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    src, dst = data["edge_source"][:128], data["edge_target"][:128]
    neg = np.random.RandomState(91).randint(len(x), size=len(src))
    roots, inv = np.unique(np.concatenate((src, dst, neg)), return_inverse=True)
    t0 = time.time()
    obs = sample_walks(data["graph"], roots, 16, 17, excluded_edges=np.stack((src, dst)))
    sampling_seconds = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        encoded = model.encode(x, torch.tensor(obs.paths, device="cuda", dtype=torch.long),
                               torch.tensor(obs.relations, device="cuda", dtype=torch.float32))
        inverse = torch.tensor(inv, device="cuda", dtype=torch.long)
        pos = model.score(encoded, inverse[:128], inverse[128:256], True)
        negscore = model.score(encoded, inverse[:128], inverse[256:], True)
        loss = F.softplus(-pos).mean() / 2 + F.softplus(negscore).mean() / 2
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        ss = torch.arange(128, device="cuda").repeat(151)
        dd = (ss * 17 + 3) % len(roots)
        pv = model.score(encoded, ss, dd, True)
        torch.cuda.synchronize()
        score_seconds = time.time() - t0
    result = {"kind": "engineering_only_no_validation_or_test", "positive_batch": 128,
              "actual_unique_roots": len(roots), "sampling_seconds_with_JIT": sampling_seconds,
              "update_seconds": times, "score_pairs": len(ss), "score_seconds": score_seconds,
              "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
              "finite_loss": bool(torch.isfinite(loss)), "finite_scores": bool(torch.isfinite(pv).all())}
    write_json(out, result); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "run"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    if args.action == "prepare":
        print(ensure_bank(data)); return
    with QueueLock(OUT / ".campaign.lock"), QueueLock(LOCK):
        if args.action == "preflight":
            preflight(data); return
        plan = json.loads((OUT / "plan.json").read_text(encoding="utf-8"))
        set_seed(0)
        budget_path = OUT / "budget.json"
        if not budget_path.exists():
            now = time.time()
            write_json(budget_path, {"started_unix": now, "deadline_unix": now + plan["gpu_budget_seconds"],
                                     "seconds": plan["gpu_budget_seconds"]})
        budget = json.loads(budget_path.read_text())
        deadline = budget["deadline_unix"]
        guard(deadline)
        bank = load_bank(ensure_bank(data))
        results = {}
        try:
            for method in plan["order"]:
                guard(deadline)
                existing = OUT / "runs" / method / "metrics.json"
                if existing.exists() and json.loads(existing.read_text()).get("complete"):
                    results[method] = json.loads(existing.read_text())
                    event("skip_completed", method=method)
                    continue
                write_json(OUT / "status.json", {"phase": "running", "method": method, "pid": os.getpid(),
                                                "deadline_unix": deadline, "updated_unix": time.time()})
                event("method_start", method=method)
                if method in ("mlp", "sage"):
                    from baselines import run_baseline
                    last_checkpoint = OUT / "runs" / method / "last.pt"
                    if last_checkpoint.exists():
                        raise RuntimeError(f"{method} has an incomplete run; do not silently restart its training")
                    cfg = {**plan["baselines"], "method": method.upper(), "stop_file": str(OUT / "STOP_REQUESTED")}
                    result = run_baseline(DATA, OUT / "runs" / method, cfg, deadline)
                else:
                    result = run_pair(method, data, bank, plan["pair"], deadline)
                results[method] = result
                write_json(OUT / "summary.json", results)
                event("method_end", method=method, complete=result.get("complete"))
                if not result.get("complete"):
                    break
                gc.collect(); torch.cuda.empty_cache()
            finished = len(results) == len(plan["order"]) and all(r.get("complete") for r in results.values())
            write_json(OUT / "status.json", {"phase": "completed" if finished else "stopped", "pid": os.getpid(),
                                            "updated_unix": time.time(), "complete": finished})
        except BudgetStop as e:
            write_json(OUT / "status.json", {"phase": "stopped", "reason": str(e), "pid": os.getpid()})
        except Exception:
            write_json(OUT / "status.json", {"phase": "failed", "pid": os.getpid(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
