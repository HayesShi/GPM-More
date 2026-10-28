"""Fixed-bank, CPU-only sampling mechanism audit for official Sports validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import time

import numba
from numba import njit, prange
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "pyproject.toml").is_file())
DATA = ROOT / "data/mm_graph/sports-copurchase/processed"
BANK = ROOT / "outputs/mm_pair_sports_20261028/eval_bank"
OUT = ROOT / "outputs/mm_pair_sports_20261028/sampling_mechanism"


@njit(cache=True, parallel=True)
def exact_positive_cn(source, target, indptr, indices):
    result = np.zeros(len(source), dtype=np.int32)
    for i in prange(len(source)):
        u, v = source[i], target[i]
        left, right, end_l, end_r = indptr[u], indptr[v], indptr[u + 1], indptr[v + 1]
        while left < end_l and right < end_r:
            a, b = indices[left], indices[right]
            if a == b:
                result[i] += 1
                left += 1
                right += 1
            elif a < b:
                left += 1
            else:
                right += 1
    return result


@njit(cache=True, parallel=True)
def collision_block(source, candidates, sorted_visits, degree, normalizer, first_step):
    rows, columns = candidates.shape
    raw = np.zeros((rows, columns), dtype=np.float64)
    corrected = np.zeros((rows, columns), dtype=np.float64)
    width = sorted_visits.shape[1]
    for i in prange(rows):
        u = source[i]
        for j in range(columns):
            v = candidates[i, j]
            left, right, dot, weighted = 0, 0, 0, 0.0
            while left < width and sorted_visits[u, left] < 0:
                left += 1
            while right < width and sorted_visits[v, right] < 0:
                right += 1
            while left < width and right < width:
                a, b = sorted_visits[u, left], sorted_visits[v, right]
                if a == b:
                    left_end, right_end = left + 1, right + 1
                    while left_end < width and sorted_visits[u, left_end] == a:
                        left_end += 1
                    while right_end < width and sorted_visits[v, right_end] == b:
                        right_end += 1
                    product = (left_end - left) * (right_end - right)
                    dot += product
                    weighted += product / degree[a]
                    left, right = left_end, right_end
                elif a < b:
                    left += 1
                else:
                    right += 1
            denominator = normalizer[u] * normalizer[v]
            if denominator > 0:
                raw[i, j] = dot / denominator
                if first_step:
                    corrected[i, j] = degree[u] * degree[v] * weighted / denominator
    return raw, corrected


def ranks(values):
    positive = values[:, :1]
    negative = values[:, 1:]
    return 1.0 + 0.5 * ((negative >= positive).sum(axis=1) + (negative > positive).sum(axis=1))


def metrics(rank, mask):
    value = rank[mask]
    return {"rows": int(value.size), "mrr": float((1.0 / value).mean()),
            **{f"hits@{k}": float((value <= k).mean()) for k in (1, 3, 10)}}


def failure_summary(missed, true_cn, mask):
    observed = missed[mask]
    available = (true_cn > 0) & mask
    denominator = int(available.sum())
    false_misses = int((missed & available).sum())
    return {"rows": int(mask.sum()), "positive_shared_witness_zero": int(observed.sum()),
            "positive_shared_witness_zero_rate": float(observed.mean()) if observed.size else None,
            "positive_exact_cn_nonzero": denominator,
            "missed_despite_exact_cn_nonzero": false_misses,
            "miss_rate_given_exact_cn_nonzero": false_misses / denominator if denominator else None}


def binned_failures(values, bounds, missed, true_cn, mask):
    result = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        selected = mask & (values >= low) & (values < high)
        record = failure_summary(missed, true_cn, selected)
        record.update({"lower_inclusive": low, "upper_exclusive": high if np.isfinite(high) else None})
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--block-rows", type=int, default=1024)
    parser.add_argument("--budget-seconds", type=float, default=300)
    args = parser.parse_args()
    numba.set_num_threads(args.threads)
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(DATA / "valid_source.npy", mmap_mode="r")
    target = np.load(DATA / "valid_target.npy", mmap_mode="r")
    negative = np.load(DATA / "valid_neg.npy", mmap_mode="r")
    indptr = np.load(DATA / "train_csr_indptr.npy", mmap_mode="r")
    indices = np.load(DATA / "train_csr_indices.npy", mmap_mode="r")
    reverse = np.load(DATA / "valid_reverse_in_train.npy")
    self_loop = np.load(DATA / "valid_self_loop.npy")
    paths = np.load(BANK / "paths.npy", mmap_mode="r")
    valid = np.load(BANK / "valid.npy", mmap_mode="r")
    if paths.shape != (50250, 16, 9) or valid.shape != paths.shape:
        raise ValueError("Unexpected fixed evaluation bank dimensions")
    n = len(source)
    masks = {"full_valid": np.ones(n, dtype=bool), "clean_valid": ~reverse & ~self_loop,
             "exposed_reverse_train": reverse}
    degree = np.diff(indptr).astype(np.int64)
    degree_product = degree[source] * degree[target]
    true_cn = exact_positive_cn(source, target, indptr, indices)
    all_metrics, all_failures, all_zero_stats = {}, {}, {}
    runtimes = {}
    verification = []
    for k in (2, 4, 8, 16):
        first = np.array(paths[:, :k, 1], dtype=np.int32)
        first[~valid[:, :k, 1]] = -1
        first.sort(axis=1)
        normalize = np.full(len(degree), k, dtype=np.int64)
        method_ranks = {name: np.empty(n, dtype=np.float64) for name in (f"raw_k{k}", f"ra_est_k{k}")}
        missed = np.empty(n, dtype=bool)
        all_zero = np.empty(n, dtype=bool)
        candidate_zero = np.empty(n, dtype=np.int32)
        score_start = time.perf_counter()
        for left in range(0, n, args.block_rows):
            if time.perf_counter() - start > args.budget_seconds:
                raise TimeoutError("Five-minute mechanism audit budget exhausted during required first-step scores")
            right = min(left + args.block_rows, n)
            candidates = np.column_stack([target[left:right], negative[left:right]])
            raw, corrected = collision_block(source[left:right], candidates, first, degree, normalize, True)
            method_ranks[f"raw_k{k}"][left:right] = ranks(raw)
            method_ranks[f"ra_est_k{k}"][left:right] = ranks(corrected)
            missed[left:right] = raw[:, 0] == 0
            all_zero[left:right] = (raw == 0).all(axis=1)
            candidate_zero[left:right] = (raw == 0).sum(axis=1)
        runtimes[f"first_step_k{k}_seconds"] = time.perf_counter() - score_start
        for name, rank in method_ranks.items():
            np.save(OUT / f"{name}_rr.npy", 1.0 / rank, allow_pickle=False)
            all_metrics[name] = {group: metrics(rank, mask) for group, mask in masks.items()}
        expected = np.divide(k * k * true_cn.astype(np.float64), degree_product,
                             out=np.zeros(n, dtype=np.float64), where=degree_product > 0)
        all_failures[f"k{k}"] = {}
        all_zero_stats[f"k{k}"] = {}
        for group, mask in masks.items():
            all_failures[f"k{k}"][group] = {
                "overall": failure_summary(missed, true_cn, mask),
                "by_degree_product": binned_failures(degree_product, [0, 64, 256, 1024, 4096, 16384, 65536, np.inf], missed, true_cn, mask),
                "by_expected_pairwise_witness_count": binned_failures(expected, [0, .1, .5, 1, 2, 4, 8, 16, np.inf], missed, true_cn, mask),
            }
            all_zero_stats[f"k{k}"][group] = {
                "rows": int(mask.sum()), "all_151_scores_zero_rows": int(all_zero[mask].sum()),
                "all_151_scores_zero_rate": float(all_zero[mask].mean()),
                "zero_candidate_pairs": int(candidate_zero[mask].sum()),
                "zero_candidate_pair_rate": float(candidate_zero[mask].sum() / (151 * mask.sum())),
            }
        # Independent count-vector dot product check, with duplicates retained.
        for i in range(8):
            u, v = int(source[i]), int(target[i])
            raw, corrected = collision_block(np.array([u]), np.array([[v]]), first, degree, normalize, True)
            a, ac = np.unique(first[u][first[u] >= 0], return_counts=True)
            b, bc = np.unique(first[v][first[v] >= 0], return_counts=True)
            left_count, right_count = dict(zip(a.tolist(), ac.tolist())), dict(zip(b.tolist(), bc.tolist()))
            shared = set(left_count) & set(right_count)
            dot = sum(left_count[w] * right_count[w] for w in shared)
            weighted = sum(left_count[w] * right_count[w] / degree[w] for w in sorted(shared))
            assert np.isclose(raw[0, 0], dot / (k * k), atol=1e-15)
            assert np.isclose(corrected[0, 0], degree[u] * degree[v] * weighted / (k * k), rtol=1e-13)
        verification.append(f"K={k}: eight validation positive pairs matched independent count-vector calculations")
        print(json.dumps({"K": k, "elapsed_seconds": time.perf_counter() - start,
                          "clean_raw": all_metrics[f"raw_k{k}"]["clean_valid"],
                          "clean_ra_est": all_metrics[f"ra_est_k{k}"]["clean_valid"],
                          "clean_failure": all_failures[f"k{k}"]["clean_valid"]["overall"]}), flush=True)
    # Optional phase-free collision score over all eight non-root walk positions.
    all_steps_status = "skipped: insufficient remaining five-minute budget"
    if time.perf_counter() - start < args.budget_seconds - 30:
        visits = np.array(paths[:, :, 1:], dtype=np.int32).reshape(len(degree), -1)
        visits[~np.array(valid[:, :, 1:]).reshape(len(degree), -1)] = -1
        normalize = (visits >= 0).sum(axis=1).astype(np.int64)
        visits.sort(axis=1)
        rank = np.empty(n, dtype=np.float64)
        missed = np.empty(n, dtype=bool)
        all_zero = np.empty(n, dtype=bool)
        score_start = time.perf_counter()
        finished = True
        for left in range(0, n, args.block_rows):
            if time.perf_counter() - start > args.budget_seconds:
                finished = False
                break
            right = min(left + args.block_rows, n)
            candidates = np.column_stack([target[left:right], negative[left:right]])
            raw, _ = collision_block(source[left:right], candidates, visits, degree, normalize, False)
            rank[left:right] = ranks(raw)
            missed[left:right] = raw[:, 0] == 0
            all_zero[left:right] = (raw == 0).all(axis=1)
        runtimes["visit8_k16_seconds"] = time.perf_counter() - score_start
        if finished:
            for i in range(8):
                u, v = int(source[i]), int(target[i])
                actual, _ = collision_block(np.array([u]), np.array([[v]]), visits, degree, normalize, False)
                a, ac = np.unique(visits[u][visits[u] >= 0], return_counts=True)
                b, bc = np.unique(visits[v][visits[v] >= 0], return_counts=True)
                left_count, right_count = dict(zip(a.tolist(), ac.tolist())), dict(zip(b.tolist(), bc.tolist()))
                dot = sum(left_count[w] * right_count[w] for w in set(left_count) & set(right_count))
                denominator = normalize[u] * normalize[v]
                expected = dot / denominator if denominator else 0.0
                assert np.isclose(actual[0, 0], expected, atol=1e-15)
            verification.append("K16 all-eight-step visits: eight validation positive pairs matched independent count-vector calculations")
            np.save(OUT / "visit8_k16_rr.npy", 1.0 / rank, allow_pickle=False)
            all_metrics["visit8_k16"] = {group: metrics(rank, mask) for group, mask in masks.items()}
            all_failures["visit8_k16"] = {group: failure_summary(missed, true_cn, mask) for group, mask in masks.items()}
            all_zero_stats["visit8_k16"] = {group: {"rows": int(mask.sum()),
                "all_151_scores_zero_rows": int(all_zero[mask].sum()),
                "all_151_scores_zero_rate": float(all_zero[mask].mean())} for group, mask in masks.items()}
            all_steps_status = "completed over the same full official validation candidates"
        else:
            all_steps_status = "skipped after partial work: five-minute budget reached; no partial RR exported"
    exact = {}
    for name in ("cn", "ra"):
        rr = np.load(OUT.parent / f"structural_baselines/{name}_rr.npy")
        exact[name] = {group: {"rows": int(mask.sum()), "mrr": float(rr[mask].mean())} for group, mask in masks.items()}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation_rows": n,
        "candidate_pairs_per_method": n * 151, "same_validation_candidates_for_all_k": True,
        "fixed_bank": json.loads((BANK / "complete.json").read_text()),
        "definitions": {
            "first_step": "bank paths[:,:,1], with bank validity applied; roots at position 0 excluded",
            "raw": "sum_w c_u(w)c_v(w)/K^2",
            "ra_est": "degree(u)*degree(v)/K^2 * sum_w c_u(w)c_v(w)/degree(w)",
            "ra_est_scope": "sampling diagnostic for first-step neighbor encounters, not a proposed new method; no K selected",
            "expected_pairwise_witness_count": "K^2 * exact_CN(u,v)/(degree(u)*degree(v)); interprets independent uniform first-step draws",
            "miss_rate_given_exact_cn_nonzero": "fraction of positive validation pairs with no sampled shared witness, among those with exact graph CN>0",
            "visit8_k16": "sum_w visit_u(w)visit_v(w)/(valid_nonroot_visits_u*valid_nonroot_visits_v), over all 16 walks and positions 1..8; includes revisits, no phase or content",
            "clean_valid": "diagnostic slice excluding original-train reverse exposure and self-loops; not a new official split",
            "tie_rank": "1 + 0.5*(count(negative>=positive)+count(negative>positive))",
        },
        "metrics": all_metrics, "exact_structural_baseline": exact,
        "positive_witness_failures": all_failures, "zero_sampled_scores": all_zero_stats,
        "all_eight_steps_status": all_steps_status,
        "runtime": {**runtimes, "total_seconds": time.perf_counter() - start,
                    "cpu_threads": args.threads, "gpu_used": False, "budget_seconds": args.budget_seconds,
                    "python": platform.python_version(), "numba": numba.__version__, "numpy": np.__version__},
        "verification": verification, "test_read_or_scored": False,
        "reproduce_command": 'python research_extensions/mm_pair_sports_20261028/sampling_mechanism.py --threads 2 --budget-seconds 300',
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(__file__, OUT / "source.py")
    rows = ["# Fixed-bank walk sampling mechanism check", "",
            "All K values use the same complete official validation rows and 150 fixed negatives. No training, test scoring, model/plan edits or GPU work is involved.", "",
            "| Score | K | Full MRR | Clean MRR | Positive witness-zero (clean) | Miss rate given exact CN>0 (clean) |",
            "|---|---:|---:|---:|---:|---:|"]
    for k in (2, 4, 8, 16):
        failure = all_failures[f"k{k}"]["clean_valid"]["overall"]
        for name in (f"raw_k{k}", f"ra_est_k{k}"):
            rows.append(f"| {name} | {k} | {all_metrics[name]['full_valid']['mrr']:.6f} | {all_metrics[name]['clean_valid']['mrr']:.6f} | {failure['positive_shared_witness_zero_rate']:.3%} | {failure['miss_rate_given_exact_cn_nonzero']:.3%} |")
    if "visit8_k16" in all_metrics:
        f = all_failures["visit8_k16"]["clean_valid"]
        rows.append(f"| all-eight-step visit collision | 16 | {all_metrics['visit8_k16']['full_valid']['mrr']:.6f} | {all_metrics['visit8_k16']['clean_valid']['mrr']:.6f} | {f['positive_shared_witness_zero_rate']:.3%} | {f['miss_rate_given_exact_cn_nonzero']:.3%} |")
    k16_failure = all_failures["k16"]["clean_valid"]["overall"]
    rows += ["", f"At K16, first-step sampling misses every shared witness for {k16_failure['missed_despite_exact_cn_nonzero']:,}/{k16_failure['positive_exact_cn_nonzero']:,} clean positive pairs that have an exact common neighbor. This directly demonstrates missing one-hop structural evidence in this fixed bank."]
    if "visit8_k16" in all_metrics:
        f = all_failures["visit8_k16"]["clean_valid"]
        rows += ["", f"After pooling all eight steps, only {f['missed_despite_exact_cn_nonzero']:,}/{f['positive_exact_cn_nonzero']:,} of those pairs have no shared visit, yet the phase-free collision MRR remains {all_metrics['visit8_k16']['clean_valid']['mrr']:.6f}. Thus complete absence of cross-endpoint encounters cannot explain the full 16-by-8 bank's ranking gap: witness coverage alone does not make this pooled collision score discriminative."]
    rows += ["", f"Exact graph CN clean MRR: {exact['cn']['clean_valid']['mrr']:.6f}; RA: {exact['ra']['clean_valid']['mrr']:.6f}.", "",
             "The first-step estimators preserve repeat counts and use K^2 normalization. Degree correction can rescale observed witnesses but cannot recover a witness absent from the fixed bank. The expected-witness groups and degree-product groups in metrics.json report this failure directly. All K use nested prefixes of one already-generated fixed bank; no K is selected and sampling uncertainty across independent banks is not measured. These are mechanism diagnostics, not proposed estimators or a K-tuning experiment.", "",
             "The all-eight-step score excludes initial roots, pools all valid later visits without step/phase identity, includes repeated visits, and has no learned H/content features. It can use longer-range encounters, so it is not an RA estimator.", "",
             "Clean is the previously defined diagnostic validation slice (37,290 rows), with reverse-exposed rows and self-loops excluded. Published full validation still contains 45,557 rows. No new official split is claimed.", "",
             f"CPU threads: {args.threads}; wall time: {report['runtime']['total_seconds']:.2f} seconds; all-eight-step status: {all_steps_status}.", "",
             "Run from the workspace root:", "", "```powershell", "& 'python' research_extensions/mm_pair_sports_20261028/sampling_mechanism.py --threads 2 --budget-seconds 300", "```", ""]
    (OUT / "README.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps({"metrics": all_metrics, "exact": exact, "runtime": report["runtime"],
                      "all_eight_steps_status": all_steps_status}, indent=2), flush=True)


if __name__ == "__main__":
    main()
