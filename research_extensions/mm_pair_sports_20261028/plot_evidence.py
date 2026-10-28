"""Evidence figure from actual single-seed validation and fixed-bank probes."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mm_pair_sports_20261028"


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published', action='store_true', help='Render only the committed aggregate results; no data or checkpoints needed')
    if parser.parse_args().published:
        OUT = ROOT / 'results/20261028/sports_pilot'
    histories = OUT / ("histories" if "results" in OUT.parts else "runs")
    probe = json.loads((OUT / "sampling_mechanism/metrics.json").read_text())
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.1, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "legend.frameon": False})
    fig, axs = plt.subplots(2, 2, figsize=(11.4, 8.2))
    axs = axs.ravel()
    blue, red, teal, gray = "#0F4D92", "#B64342", "#268779", "#767676"
    k = np.array([2, 4, 8, 16])
    exact = 100 * probe["exact_structural_baseline"]["ra"]["clean_valid"]["mrr"]
    raw = [100 * probe["metrics"][f"raw_k{x}"]["clean_valid"]["mrr"] for x in k]
    corrected = [100 * probe["metrics"][f"ra_est_k{x}"]["clean_valid"]["mrr"] for x in k]
    a = axs[0]
    a.plot(k, raw, "o-", color=gray, lw=2, label="First-step collision")
    a.plot(k, corrected, "s-", color=teal, lw=2, label="Degree-corrected estimate")
    a.axhline(exact, color="black", ls="--", lw=1.4, label="Exact RA")
    a.set(xscale="log", xlabel="Walks per endpoint (K)", ylabel="Validation MRR (%)", ylim=(0, 47))
    a.set_xticks(k, labels=k)
    a.xaxis.set_minor_locator(NullLocator())
    a.set_title("A  Sampling changes the statistic", loc="left", fontweight="bold", fontsize=12)
    a.legend(loc="lower right", fontsize=8.5)
    a.text(16, exact + 1, f"{exact:.2f}", ha="right", fontsize=10)

    a = axs[1]
    miss = [100 * probe["positive_witness_failures"][f"k{x}"]["clean_valid"]["overall"]["miss_rate_given_exact_cn_nonzero"] for x in k]
    a.plot(k, miss, "o-", color=red, lw=2)
    a.set(xscale="log", xlabel="Walks per endpoint (K)", ylabel="Missed positive pairs (%)", ylim=(-2, 100))
    a.set_xticks(k, labels=k)
    a.xaxis.set_minor_locator(NullLocator())
    a.set_title("B  First-step evidence can be absent", loc="left", fontweight="bold", fontsize=12)
    a.text(.03, .04, "Among positives with a real\ncommon neighbor; first step only", transform=a.transAxes,
           fontsize=9, va="bottom", color="#444444")
    a.annotate(f"{miss[-1]:.2f}%", (16, miss[-1]), xytext=(-10, -20), textcoords="offset points", ha="right", color=red)

    a = axs[2]
    arm_defs = [("mlp", "MLP", gray), ("sage", "SAGE", "#B78A31"),
                ("pair_independent", "raw_sum: independent ends", blue),
                ("pair_joint", "raw_sum: joint witnesses", teal)]
    for method, label, color in arm_defs:
        f = histories / method / "history.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not rows:
            continue
        xx = [r["positive_targets_seen"] / 1000 if method in ("mlp", "sage") else r["positive_examples"] / 1000 for r in rows]
        yy = [100*r["valid_clean"]["mrr"] if method in ("mlp", "sage") else 100*r["validation"]["clean"]["mrr"] for r in rows]
        a.plot(xx, yy, "o-", color=color, lw=2, label=label)
    a.axhline(exact, color="black", ls="--", lw=1.4, label="Exact RA (no training)")
    a.set(xlabel="Training positives processed (thousands)", ylabel="Validation MRR (%)", xlim=(100, 540))
    a.set_title("C  Same positive count; distinct models", loc="left", fontweight="bold", fontsize=12)
    a.set_ylim(12, 54)
    a.legend(loc="upper left", fontsize=8, ncol=2, columnspacing=1.2)
    a = axs[3]
    final = histories / "pair_joint/fixed_weight_no_joint.json"
    if final.exists():
        on_values = json.loads((histories / 'pair_joint/metrics.json').read_text())['final_validation']
        off_values = json.loads(final.read_text())['validation']
        independent_values = json.loads((histories / 'pair_independent/metrics.json').read_text())['final_validation']
        on = 100 * on_values['clean']['mrr']
        off = 100 * off_values['clean']['mrr']
        independent = 100 * independent_values['clean']['mrr']
        a.plot([0, 1], [on, off], "o-", color=teal, lw=2.4, markersize=8,
               label="No direct reverse exposure (n=37,290)")
        exposed_on, exposed_off = 100 * on_values['reverse_exposed']['mrr'], 100 * off_values['reverse_exposed']['mrr']
        a.plot([0, 1], [exposed_on, exposed_off], "s-", color=red, lw=2.4, markersize=7,
               label="Reverse edge in train (n=8,264)")
        a.axhline(independent, color=blue, ls="--", lw=1.5, label="Independent training: unexposed slice")
        for x, y in [(0, on), (1, off)]:
            a.annotate(f"{y:.2f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", color=teal)
        for x, y in [(0, exposed_on), (1, exposed_off)]:
            a.annotate(f"{y:.2f}", (x, y), xytext=(0, -17), textcoords="offset points", ha="center", color=red)
        a.set(xticks=[0, 1], xticklabels=["Joint score on", "Joint score off"],
              ylabel="Validation MRR (%)", xlim=(-.25, 1.25), ylim=(16, max(independent, on, off) + 7))
        a.legend(loc="lower left", fontsize=8.2)
        a.text(.03, .95, f"Same final weights. All queries: {100*on_values['full']['mrr']:.2f} to {100*off_values['full']['mrr']:.2f}",
               transform=a.transAxes, va="top", fontsize=9, color="#444444")
    a.set_title("D  One branch, opposite subgroup effects", loc="left", fontweight="bold", fontsize=12)
    for a in axs:
        a.grid(axis="y", color="#e7e7e7", lw=.6)
        a.set_axisbelow(True)
    fig.text(.5, .018, "A-C: 37,290 Sports validation queries without direct reverse-train exposure or self-loops; D also shows exposed queries.\n"
             "Released negative candidates retained, including known-positive overlaps. One seed / one bank; no test results.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .065, 1, 1), w_pad=2, h_pad=2.8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"core_evidence.{ext}", dpi=250, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    caption = ("A: Exact RA and finite first-step encounter scores are evaluated on identical official candidates; "
               "degree correction changes weights without obtaining new observations. "
               "B: Positive pairs with at least one exact common neighbor can still have no sampled first-step witness. "
               "This does not imply that the full eight-step bank has no collision: its corresponding miss rate is about 0.109%. "
               "C: Recorded checkpoint trajectories at the same processed-positive budgets; model compute and parameter counts differ. "
               "D: Only the added cross-endpoint score is removed from the final jointly trained checkpoint; the ordinary root relation token and all weights are retained. "
               "The independent reference was trained separately from the matched initial state. These comparisons do not isolate relation gradients. "
               "The clean slice is diagnostic, while checkpoints are selected by full official validation MRR. "
               "Released negative candidates overlap original directed training positives at 185,714 positions; this candidate-label issue is retained, not fixed by the slice. "
               "All numbers are real single-seed or fixed-bank validation measurements, not test performance or multi-seed uncertainty.\n")
    (OUT / "core_evidence_caption.txt").write_text(caption, encoding="utf-8")
    print(OUT / "core_evidence.png")


if __name__ == "__main__":
    main()
