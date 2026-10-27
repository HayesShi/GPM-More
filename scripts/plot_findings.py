"""Regenerate the archived-result figures from the repository's aggregate CSVs.

This script performs no model training and needs only NumPy and Matplotlib.
All lines connect measured values; no result is interpolated or synthesized.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#777777"


def read_csv(folder: Path, name: str) -> list[dict[str, str]]:
    with (folder / name).open(encoding="utf8", newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def save(fig, folder, name):
    for extension in ("png", "svg", "pdf"):
        path = folder / f"{name}.{extension}"
        metadata = ({"Date": None} if extension == "svg" else
                    {"CreationDate": None, "ModDate": None} if extension == "pdf" else None)
        fig.savefig(path, dpi=180, metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                            path.read_text(encoding="utf8").splitlines()) + "\n", encoding="utf8")
    plt.close(fig)


def plot_findings(results: Path, figures: Path):
    content = read_csv(results, "forward_content_interventions.csv")
    interventions = read_csv(results, "training_interventions.csv")
    steps = read_csv(results, "conditional_probe_steps.csv")
    curves = read_csv(results, "conditional_validation_curves.csv")
    summary = json.loads((results / "conditional_probe_summary.json").read_text(encoding="utf8"))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.7), layout="constrained")

    ax = axes[0, 0]
    settings = [("prefix", "Shared prefix (400)", GRAY, "o"),
                ("native", "Full-feedback endpoint", BLUE, "s"),
                ("detach_global_stepmatched", "Stop + step-match endpoint", ORANGE, "^")]
    for state, label, color, marker in settings:
        selected = {row["condition"]: float(row["R2"]) for row in content if row["state"] == state}
        pair = [selected["original"], selected["within_collision_shuffle_direction_keep_norm"]]
        ax.plot([0, 1], pair, marker=marker, color=color, label=label, lw=1.6, ms=5)
    ax.set_xticks([0, 1], ["Original relation content", "Direction permuted within\ncollision strata; norm kept"])
    ax.set(xlim=(-.15, 1.15), ylim=(.31, .52), ylabel="Validation R²",
           title="A  Relation content affects prediction")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.text(.98, .97, "Fixed weights and ordinary tokens; K = 12", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5)

    ax = axes[0, 1]
    for factor, color, label, style in [(1., BLUE, "Original LR schedule", "-"),
                                       (.1, ORANGE, "Additional all-parameter LR × 0.1", "--")]:
        selected = {row["training_intervention"]: float(row["best_validation_R2"])
                    for row in interventions if float(row["extra_all_lr_factor"]) == factor}
        pair = [selected["native"], selected["detach_global"]]
        ax.plot([0, 1], pair, "o", linestyle=style, color=color, lw=1.8, label=label)
        ax.annotate(f"Δ = {pair[1]-pair[0]:+.5f}", (.5, sum(pair)/2),
                    xytext=(0, 10 if factor == .1 else -20), textcoords="offset points",
                    color=color, ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["Full relation feedback", "Stop relation feedback"])
    ax.set(xlim=(-.15, 1.15), ylim=(.481, .514), ylabel="Best validation R²",
           title="B  Feedback and update scale interact")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.text(.02, .97, "Matched seed 0, prefix 400 → 1000", transform=ax.transAxes, va="top", fontsize=8.5)

    ax = axes[1, 0]
    q = values(steps, "Q_full_minus_stop_MSE") * 1e5
    audit = values(steps, "audit_full_minus_stop_MSE") * 1e5
    full = values(steps, "chosen_alpha").astype(bool)
    lim = max(np.max(np.abs(q)), np.max(np.abs(audit))) * 1.08
    ax.axhline(0, color=GRAY, lw=.8)
    ax.axvline(0, color=GRAY, lw=.8)
    ax.scatter(q[full], audit[full], c=BLUE, marker="o", s=18, alpha=.65, label="Full chosen")
    ax.scatter(q[~full], audit[~full], c=ORANGE, marker="x", s=21, alpha=.7, label="Stop chosen")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim),
           xlabel="Selection probe Q: full − stop MSE (×10⁻⁵)",
           ylabel="Non-selection Qaudit: same contrast (×10⁻⁵)",
           title="C  Immediate preferences do not transfer reliably")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    agreement = summary["audit"]["fraction_selection_agrees_with_lower_risk"]
    ax.text(.02, .04, f"Preference agreement: {agreement:.1%}\nBoth probes use training nodes",
            transform=ax.transAxes, va="bottom", fontsize=8.5)

    ax = axes[1, 1]
    settings = [("conditional", "Conditional feedback (current)", BLUE, "-", "o"),
                ("stop", "Fixed stop (matched replay)", ORANGE, "-", "s"),
                ("full_historical", "Fixed full (historical reference)", GRAY, ":", "^")]
    for arm, label, color, style, marker in settings:
        rows = [row for row in curves if row["arm"] == arm]
        ax.plot(values(rows, "update"), values(rows, "R2"), color=color, linestyle=style,
                marker=marker, label=label, lw=1.7, ms=4)
    gain = summary["conditional_minus_current_stop_final_R2"]
    ax.set(xlabel="Training update", ylabel="Validation R²",
           title=f"D  Tested controller: endpoint gain {gain:+.6f}")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, figures, "findings")


def plot_local_paths(results: Path, figures: Path):
    rows = read_csv(results, "local_path_exchange.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), layout="constrained")
    for ax, direction, label in zip(axes, ["relation_frozen_metric", "adam_relation_increment"],
                                     ["Pure relation direction, normalized", "Natural Adam relation increment"]):
        selected = [r for r in rows if r["direction"] == direction
                    and r["split"] == "validation" and float(r["scale"]) == 1.]
        selected.sort(key=lambda r: int(r["state"]))
        ax.axhline(0, color=GRAY, lw=.8)
        for field, name, color, marker in [
            ("delta_MSE_P1R0", "Ordinary path only changed", ORANGE, "s"),
            ("delta_MSE_P0R1", "Relation path only changed", BLUE, "^"),
            ("delta_MSE_P1R1", "Both paths changed", GREEN, "o"),
        ]:
            ax.plot(values(selected, "state"), values(selected, field), color=color,
                    marker=marker, label=name, lw=1.5)
        ax.set(xlabel="Frozen training checkpoint", ylabel="Validation MSE change",
               title=label)
        ax.set_xticks([400, 500, 600, 800])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Local spillover can coexist with a beneficial joint update", fontsize=13)
    save(fig, figures, "local_paths")


def plot_probe_transfer(results: Path, figures: Path):
    rows = read_csv(results, "conditional_probe_steps.csv")
    fig, ax = plt.subplots(figsize=(7.3, 4.2), layout="constrained")
    for field, label, color, style in [
        ("Q_selected_minus_rejected_MSE", "Selection Q (guaranteed by the rule)", BLUE, "-"),
        ("audit_selected_minus_rejected_MSE", "Non-selection Qaudit", GREEN, "--"),
    ]:
        raw = values(rows, field)
        means = np.cumsum(raw) / np.arange(1, len(raw)+1)
        ax.plot(values(rows, "update"), means * 1e5, label=label, color=color, linestyle=style, lw=1.7)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.set(xlabel="Training update", ylabel="Running mean selected − rejected MSE (×10⁻⁵)",
           title="Selection benefit on one training probe does not persist on the other")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, figures, "probe_transfer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "semibold", "svg.fonttype": "none",
                         "svg.hashsalt": "gpm-more", "pdf.fonttype": 42})
    plot_findings(args.results_dir, args.output_dir)
    plot_local_paths(args.results_dir, args.output_dir)
    plot_probe_transfer(args.results_dir, args.output_dir)
    print(f"Wrote findings, local_paths and probe_transfer in PNG/SVG/PDF to {args.output_dir}")


if __name__ == "__main__":
    main()
