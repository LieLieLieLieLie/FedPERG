from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from audit_results import load_runs
from fedcanto.config import METHODS


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
PAPER_FIGURE_DIRS = [ROOT.parent / "paper" / "INF" / "figures"]
QA_ONLY = False
for folder in [FIGURES, TABLES, *PAPER_FIGURE_DIRS]:
    folder.mkdir(parents=True, exist_ok=True)

OUR = "#FF6666"
# The ten algorithms in the journal-facing comparison are fixed by mechanism
# coverage. Diagnostic variants are controls, not additional baseline methods.
PAPER_METHODS = ["FedAvg", "FedAvgM", "SCAFFOLD", "FedAdam", "FedLAW",
                 "FedCDA", "FedPW", "Fed-NGA", "FedPhoenix", "FedCANTO"]
COLORS = dict(zip(PAPER_METHODS, ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
                                  "#9933FF", "#00DDDD", "#4D4D4D", "#B8860B",
                                  "#D65DB1", OUR]))
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p", "8"]
METHOD_LABELS = {m: m for m in PAPER_METHODS}
METHOD_LABELS["FedCANTO"] = "FedCANTO"
DATA_LABELS = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "officehome": "Office-Home-65"}
REGIME_LABELS = {"label_skew": "Label skew", "quantity_skew": "Quantity skew", "compound": "Compound"}
RED = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF4F4F"])
DIVERGING = LinearSegmentedColormap.from_list("blue_white_red", ["#007FFF", "#FFFFFF", "#FF4F4F"])


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def figure_path(stem: str) -> Path:
    if QA_ONLY:
        folder = RESULTS / "qa"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}.png"
    return FIGURES / f"{stem}.pdf"


def panel(ax: plt.Axes, letter: str, title: str, x: float = -0.235) -> None:
    ax.text(x, 1.15, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="top", clip_on=False)
    ax.set_title(title, pad=8)
    ax.grid(True, color="#D9D9D9", lw=0.55, alpha=0.7)


def formal_json(dataset: str, regime: str, method: str, seed: int) -> Dict:
    actual_seed = seed + 20
    name = (f"{dataset}__{regime}__{method.lower().replace('-', '_')}__s{actual_seed}"
            "__round13_public_ten_method_matrix.json")
    if method == "FedCANTO":
        name = (f"{dataset}__{regime}__fedcanto__s{actual_seed}__lite_paired_gate_bank_selector"
                "__round14_final_breadth_update.json")
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def gate_json(dataset: str, regime: str, seed: int) -> Dict:
    """Load the frozen identical-gate diagnostic run."""
    actual_seed = seed + 20
    name = (f"{dataset}__{regime}__fedavgm_gate__s{actual_seed}"
            "__round13_frozen_public_gate_confirmation.json")
    return json.loads((MODELS / name).read_text(encoding="utf-8"))


def extended_runs(tag_prefix: str = "") -> List[Dict]:
    records = []
    for path in MODELS.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        tag = record.get("config", {}).get("experiment_tag", "")
        variant = record.get("config", {}).get("canto_variant", "full")
        if tag.startswith("round3_wholeclient_fig5_"):
            record["config"]["experiment_tag"] = tag[len("round3_wholeclient_fig5_"):]
            tag = record["config"]["experiment_tag"]
        elif tag.startswith(("revision_fig5_", "round3_final_fig5_")):
            continue
        elif (record.get("config", {}).get("method") == "FedCANTO" and
              tag.startswith(("sensitivity_", "stress_", "scale_"))):
            continue
        if tag_prefix and not tag.startswith(tag_prefix):
            continue
        if tag or variant != "full":
            records.append(record)
    return records


def plot_convergence() -> Path:
    methods = PAPER_METHODS
    fig, axes = plt.subplots(3, 3, figsize=(10.9, 7.55), sharex=True)
    letters = "abcdefghi"
    for index, (dataset, regime) in enumerate(
        [(d, r) for d in ["cifar10", "cifar100", "officehome"] for r in ["label_skew", "quantity_skew", "compound"]]
    ):
        ax = axes.flat[index]
        for method_idx, method in enumerate(methods):
            curves = []
            for seed in [0, 1, 2]:
                run = formal_json(dataset, regime, method, seed)
                curves.append([100 * x["accuracy"] for x in run["history"]])
                rounds = [x["round"] for x in run["history"]]
            values = np.asarray(curves)
            mean, std = values.mean(0), values.std(0)
            ax.fill_between(rounds, mean - std, mean + std, color=COLORS[method],
                            alpha=0.18, linewidth=0, zorder=1)
            ax.plot(rounds, mean, color=COLORS[method], lw=2.4 if method == "FedCANTO" else 1.5,
                    marker=MARKERS[method_idx], ms=3.2, markevery=3,
                    label=METHOD_LABELS[method], zorder=2)
        dataset_short = {"cifar10": "C10", "cifar100": "C100", "officehome": "OH-65"}[dataset]
        regime_short = {"label_skew": "Label", "quantity_skew": "Quantity", "compound": "Compound"}[regime]
        panel(ax, letters[index], f"{dataset_short}: {regime_short}", x=-0.12)
        if index % 3 == 0:
            ax.set_ylabel("Global accuracy (%) ↑")
        else:
            # Shared row scales make repeated y tick labels unnecessary and
            # allow a tighter inter-column layout at journal size.
            ax.tick_params(axis="y", labelleft=False)
        if index >= 6:
            ax.set_xlabel("Communication round (rounds)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.15, handletextpad=.40)
    fig.subplots_adjust(top=0.82, hspace=0.31, wspace=0.12, bottom=0.10,
                        left=0.08, right=0.99)
    path = figure_path("convergence_grid")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_evaluation() -> Path:
    df = load_runs()
    tasks = [(d, r) for d in ["cifar10", "cifar100", "officehome"] for r in ["label_skew", "quantity_skew", "compound"]]
    methods = PAPER_METHODS
    short_methods = {"FedAvg": "Avg", "FedAvgM": "AvgM", "SCAFFOLD": "SCAF",
                     "FedAdam": "Adam", "FedLAW": "LAW", "FedCDA": "CDA",
                     "FedPW": "PW", "Fed-NGA": "NGA", "FedPhoenix": "Phoenix",
                     "FedCANTO": "CANTO"}
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.45))

    # (a) Distribution + quartiles + mean/95% CI: substantially richer than a
    # single endpoint bar and exposes between-task/seed dispersion.
    ax = axes[0, 0]
    groups = [100 * df[df.method == method]["worst20_accuracy"].to_numpy() for method in methods]
    positions = np.arange(len(methods))
    violin = ax.violinplot(groups, positions=positions, vert=False, widths=.76,
                           showextrema=False, showmedians=False)
    for body, method in zip(violin["bodies"], methods):
        body.set_facecolor(COLORS[method]); body.set_edgecolor(COLORS[method]); body.set_alpha(.42)
    for y, values, method in zip(positions, groups, methods):
        q1, median, q3 = np.quantile(values, [.25, .50, .75])
        mean = values.mean(); ci95 = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
        ax.plot([q1, q3], [y, y], color="#333333", lw=2.2)
        ax.scatter([median], [y], color="white", edgecolor="#222222", s=19, zorder=4)
        ax.errorbar(mean, y, xerr=ci95, fmt="D", ms=3.4, color=COLORS[method],
                    ecolor=COLORS[method], capsize=2.5, zorder=5)
    ax.set_yticks(positions, [short_methods[m] for m in methods], rotation=14)
    ax.set_xlabel("Worst-20% client accuracy (%) ↑")
    ax.invert_yaxis()
    panel(ax, "a", "Tail-accuracy distribution", x=-0.12)

    # (b) Task-resolved calibration field.
    ax = axes[0, 1]
    values = np.zeros((len(methods), len(tasks)))
    for i, method in enumerate(methods):
        for j, (dataset, regime) in enumerate(tasks):
            values[i, j] = 100 * df[(df.method == method) & (df.dataset == dataset) &
                                    (df.regime == regime)]["ece"].mean()
    image = ax.imshow(values, aspect="auto", cmap=RED, vmin=0)
    ax.set_xticks(range(9), [str(j + 1) for j in range(9)])
    ax.set_yticks(range(len(methods)), [short_methods[m] for m in methods], rotation=14)
    ax.set_xlabel("Task index (1–9)")
    panel(ax, "b", "Calibration error (ECE, %) ↓", x=-0.12)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, label="ECE (%)")
    ax.grid(False)

    # (c) Metric-dependent rank trajectories.
    ax = axes[0, 2]
    rank_metrics = [("convergence_auc", True, "AUC"), ("accuracy", True, "Final"),
                    ("worst20_accuracy", True, "Tail"), ("macro_f1", True, "F1"),
                    ("ece", False, "ECE"), ("brier", False, "Brier")]
    x = np.arange(len(rank_metrics))
    for index, method in enumerate(methods):
        ranks = []
        for metric, high, _ in rank_metrics:
            means = df[df.method.isin(methods)].groupby("method")[metric].mean()
            ranks.append(float(means.rank(ascending=not high, method="average")[method]))
        ax.plot(x, ranks, color=COLORS[method], marker=MARKERS[index], ms=4,
                lw=2.2 if method == "FedCANTO" else 1.15, label=METHOD_LABELS[method])
    ax.set_xticks(x, [label for _, _, label in rank_metrics], rotation=28, ha="right")
    ax.set_ylabel("Mean method rank (1 = best) ↓")
    ax.set_ylim(10.6, .4)
    panel(ax, "c", "Metric-dependent rank flow", x=-0.12)

    # (d) Paired-effect ECDFs preserve every task/seed rather than collapsing
    # them into a bar.
    ax = axes[1, 0]
    competitors = [m for m in methods if m != "FedCANTO"]
    for method in competitors:
        gains = []
        for dataset, regime in tasks:
            for seed in [20, 21, 22]:
                ours = float(df[(df.method == "FedCANTO") & (df.dataset == dataset) &
                                (df.regime == regime) & (df.seed == seed)]["convergence_auc"].iloc[0])
                other = float(df[(df.method == method) & (df.dataset == dataset) &
                                 (df.regime == regime) & (df.seed == seed)]["convergence_auc"].iloc[0])
                gains.append(100 * (ours - other))
        sorted_gain = np.sort(gains)
        ax.step(sorted_gain, np.arange(1, len(sorted_gain) + 1) / len(sorted_gain),
                where="post", lw=1.35, color=COLORS[method], alpha=.9,
                label=METHOD_LABELS[method])
    ax.axvline(0, color="#222222", lw=1, ls="--")
    ax.set_xlabel("FedCANTO AUC gain (percentage points) ↑")
    ax.set_ylabel("Empirical CDF (fraction)")
    panel(ax, "d", "Paired AUC-gain ECDF", x=-0.12)

    # (e) Pareto field with uncertainty and an empirical non-dominated frontier.
    ax = axes[1, 1]
    grouped = df[df.method.isin(methods)].groupby("method")
    means = grouped.agg(auc=("convergence_auc", "mean"),
                        aggregation_ms=("aggregation_ms", "mean"), ece=("ece", "mean"))
    sem = grouped.agg(auc=("convergence_auc", "sem"), aggregation_ms=("aggregation_ms", "sem"))
    # Use the measured end-to-end server aggregation time of the final
    # public-calibrated gate bank, including candidate evaluation.
    sizes = 42 + 105 * (means.ece - means.ece.min()) / (means.ece.max() - means.ece.min() + 1e-9)
    for index, method in enumerate(methods):
        ax.errorbar(means.loc[method, "aggregation_ms"], 100 * means.loc[method, "auc"],
                    xerr=sem.loc[method, "aggregation_ms"], yerr=100 * sem.loc[method, "auc"],
                    fmt=MARKERS[index], ms=np.sqrt(sizes.loc[method]), color=COLORS[method],
                    ecolor=COLORS[method], alpha=.9, capsize=2.5, zorder=3)
        if method == "FedCANTO":
            ax.annotate("CANTO", (means.loc[method, "aggregation_ms"],
                        100 * means.loc[method, "auc"]), xytext=(-7, 3),
                        textcoords="offset points", fontsize=11, ha="right")
    # This attribution-critical control is intentionally shown separately and
    # is not counted as an eleventh primary algorithm.
    gate_rows = []
    for dataset, regime in tasks:
        for seed in [0, 1, 2]:
            run = gate_json(dataset, regime, seed)
            gate_rows.append({
                "auc": run["final"]["convergence_auc"],
                "aggregation_ms": 1000 * np.mean(
                    [entry["aggregation_seconds"] for entry in run["diagnostics"]]
                ),
                "ece": run["final"]["ece"],
            })
    gate = pd.DataFrame(gate_rows)
    gate_mean = gate.mean(numeric_only=True)
    gate_sem = gate.sem(numeric_only=True)
    ax.errorbar(gate_mean.aggregation_ms, 100 * gate_mean.auc,
                xerr=gate_sem.aggregation_ms, yerr=100 * gate_sem.auc,
                fmt="*", ms=11, mfc="white", mec="#111111", mew=1.2,
                ecolor="#111111", capsize=2.5, zorder=5,
                 label="Matched gate control")
    pareto = means[["auc", "aggregation_ms"]].copy()
    pareto.loc["Matched gate control"] = [gate_mean.auc,
                                          gate_mean.aggregation_ms]
    ordered = pareto.sort_values("aggregation_ms")
    frontier_x, frontier_y, best = [], [], -np.inf
    for method, row in ordered.iterrows():
        if row.auc > best:
            frontier_x.append(row.aggregation_ms); frontier_y.append(100 * row.auc); best = row.auc
    ax.plot(frontier_x, frontier_y, color="#222222", lw=1.1, ls="--", label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Server aggregation time (ms/round) ↓")
    ax.set_ylabel("Mean round-AUC (%) ↑")
    panel(ax, "e", "Accuracy–cost Pareto field", x=-0.12)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    # (f) Win/tie/loss decomposition against each baseline across six metrics
    # and nine tasks; stacked shares are more informative than a single win bar.
    ax = axes[1, 2]
    compare_metrics = [("accuracy", True), ("macro_f1", True), ("worst20_accuracy", True),
                       ("convergence_auc", True), ("ece", False), ("brier", False)]
    decomposition = []
    for method in competitors:
        win = tie = loss = 0
        for metric, high in compare_metrics:
            for dataset, regime in tasks:
                ours = float(df[(df.method == "FedCANTO") & (df.dataset == dataset) &
                                (df.regime == regime)][metric].mean())
                other = float(df[(df.method == method) & (df.dataset == dataset) &
                                 (df.regime == regime)][metric].mean())
                signed = ours - other if high else other - ours
                if abs(signed) <= 1e-8: tie += 1
                elif signed > 0: win += 1
                else: loss += 1
        decomposition.append((method, win, tie, loss))
    decomposition.sort(key=lambda row: row[1] - row[3])
    y = np.arange(len(decomposition)); total = len(compare_metrics) * len(tasks)
    losses = 100 * np.asarray([row[3] for row in decomposition]) / total
    ties = 100 * np.asarray([row[2] for row in decomposition]) / total
    wins = 100 * np.asarray([row[1] for row in decomposition]) / total
    ax.barh(y, losses, color="#3399FF", label="Loss")
    ax.barh(y, ties, left=losses, color="#D9D9D9", label="Tie")
    ax.barh(y, wins, left=losses + ties, color=OUR, label="Win")
    ax.axvline(50, color="#333333", lw=.9, ls="--")
    ax.set_yticks(y, [short_methods[row[0]] for row in decomposition], rotation=14)
    ax.set_xlabel("Task–metric outcomes (%)")
    ax.legend(frameon=True, framealpha=.86, ncol=1, loc="lower right",
              fontsize=11, borderpad=.35, labelspacing=.25, handlelength=1.4)
    panel(ax, "f", "FedCANTO outcomes\nby baseline", x=-0.12)

    handles = [plt.Line2D([0], [0], color=COLORS[m], marker=MARKERS[i], lw=1.8,
                          label=METHOD_LABELS[m]) for i, m in enumerate(methods)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(top=0.84, hspace=0.39, wspace=0.50, bottom=0.10,
                        left=0.09, right=0.98)
    path = figure_path("evaluation_summary")
    fig.savefig(path)
    plt.close(fig)
    return path


def _tag_value(tag: str, key: str) -> float:
    return float(tag.split(key, 1)[1])


def plot_sensitivity_stress() -> Path:
    runs = extended_runs()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.35))

    # Panels (a,b) are deliberately method-specific: rho and the residual
    # coefficient are parameters of FedCANTO and have no counterpart in the
    # baselines.  The other four panels compare all ten primary algorithms.
    for ax, prefix, xlabel, letter, title in [
        (axes[0, 0], "round6_fig5_rho_", "Conservative budget $\\rho$ (fraction)",
         "a", "FedCANTO budget sensitivity"),
        (axes[0, 1], "round6_fig5_residual_", "Residual coefficient $\\lambda_{\\rm res}$ (fraction)",
         "b", "FedCANTO residual sensitivity"),
    ]:
        records = []
        for run in runs:
            tag = run["config"].get("experiment_tag", "")
            if tag.startswith(prefix):
                records.append((_tag_value(tag, prefix), 100 * run["final"]["accuracy"]))
        frame = pd.DataFrame(records, columns=["x", "value"])
        stats = frame.groupby("x").value.agg(["mean", "std", "count"]).reset_index()
        ci = 1.96 * stats["std"].fillna(0) / np.sqrt(stats["count"])
        ax.fill_between(stats.x.to_numpy(), (stats["mean"] - ci).to_numpy(),
                        (stats["mean"] + ci).to_numpy(), color=OUR, alpha=.18, lw=0)
        ax.plot(stats.x, stats["mean"], color=OUR, marker="o", lw=2.3)
        best = int(stats["mean"].argmax())
        ax.scatter(stats.x.iloc[best], stats["mean"].iloc[best], s=115,
                   facecolor="none", edgecolor="#222222", lw=1.2, zorder=5)
        if letter == "b":
            ax.annotate("best mean", (stats.x.iloc[best], stats["mean"].iloc[best]),
                        xytext=(0.04, 0.92), textcoords="axes fraction", fontsize=12,
                        ha="left", va="top",
                        arrowprops={"arrowstyle": "-", "lw": .8, "color": "#333333"})
        else:
            ax.annotate("best mean", (stats.x.iloc[best], stats["mean"].iloc[best]),
                        xytext=(6, 7), textcoords="offset points", fontsize=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Global accuracy (%) ↑")
        panel(ax, letter, title)

    def ten_method_panel(ax: plt.Axes, prefix: str, letter: str, title: str,
                         xlabel: str, dataset: str | None = None, percent_x: bool = False) -> None:
        for idx, method in enumerate(PAPER_METHODS):
            records = []
            for run in runs:
                cfg = run["config"]
                tag = cfg.get("experiment_tag", "")
                if (cfg.get("method") == method and tag.startswith(prefix) and
                        (dataset is None or cfg.get("dataset") == dataset)):
                    records.append((_tag_value(tag, prefix), 100 * run["final"]["accuracy"]))
            frame = pd.DataFrame(records, columns=["x", "value"])
            stats = frame.groupby("x").value.agg(["mean", "std"]).reset_index()
            x = (100 * stats.x if percent_x else stats.x).to_numpy()
            mean = stats["mean"].to_numpy(); std = stats["std"].fillna(0).to_numpy()
            ax.fill_between(x, mean - std, mean + std, color=COLORS[method],
                            alpha=.07 if method != "FedCANTO" else .15, lw=0)
            ax.plot(x, mean, color=COLORS[method], marker=MARKERS[idx], ms=4.2,
                    lw=2.4 if method == "FedCANTO" else 1.25,
                    label=METHOD_LABELS[method])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Global accuracy (%) ↑")
        panel(ax, letter, title)

    ten_method_panel(axes[0, 2], "round6_fig5_cpr_", "c", "Participation response",
                     "Participation (clients/round)")
    ten_method_panel(axes[1, 0], "round6_fig5_stress_", "d", "CIFAR-10 sign-flip stress",
                     "Sign-flip clients (%)", dataset="cifar10", percent_x=True)
    ten_method_panel(axes[1, 1], "round6_fig5_stress_", "e", "CIFAR-100 sign-flip stress",
                     "Sign-flip clients (%)", dataset="cifar100", percent_x=True)
    ten_method_panel(axes[1, 2], "round6_fig5_scale_", "f", "Office-Home federation scaling",
                     "Federation size (clients)", dataset="officehome")

    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.015), columnspacing=1.2, handletextpad=.4)
    fig.subplots_adjust(top=0.83, hspace=0.48, wspace=0.25, bottom=0.09,
                        left=0.075, right=0.985)
    path = figure_path("sensitivity_stress")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_operator_diagnostics() -> Path:
    run = formal_json("officehome", "compound", "FedCANTO", 0)
    diagnostics = run["diagnostics"]
    last = diagnostics[-1]
    weights = np.asarray(last["weights_by_layer"])
    rec = np.asarray([d.get("reconstruction_error", np.nan) for d in diagnostics])
    entropy = np.asarray([d["weight_entropy"] for d in diagnostics])
    cosine = np.asarray([d["mean_update_cosine"] for d in diagnostics])
    fusion_bias = np.asarray([d.get("fusion_bias", np.nan) for d in diagnostics])
    rho = np.asarray([d.get("rho", np.nan) for d in diagnostics])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.35))
    ax = axes[0, 0]
    image = ax.imshow(weights.T, aspect="auto", cmap=RED, vmin=0)
    ax.set_xticks(range(weights.shape[0]), [f"C{x}" for x in last["selected_clients"]], rotation=38, ha="right")
    tensor_labels = ["Hidden W", "Hidden b", "Head W", "Head b"] if weights.shape[1] == 4 else [f"T{j}" for j in range(weights.shape[1])]
    ax.set_yticks(range(weights.shape[1]), tensor_labels,
                  rotation=42, ha="right", va="center")
    ax.set_xlabel("Selected client (identifier)")
    ax.set_ylabel("Network tensor (layer)")
    panel(ax, "a", "Deterministic client–tensor mass")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02, label="Weight (fraction)")

    ax = axes[0, 1]
    x = np.sort(rec[np.isfinite(rec)])
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=OUR, lw=2)
    q25, median, q75 = np.quantile(x, [.25, .5, .75])
    ax.axvspan(q25, q75, color=OUR, alpha=.16, label="interquartile range")
    ax.axvline(median, color="#333333", ls="--", lw=1.2, label=f"median={median:.2f}")
    ax.set_xlabel("Leave-one-client residual (normalized) ↓")
    ax.set_ylabel("ECDF (fraction)")
    ax.legend(frameon=False)
    panel(ax, "b", "Leave-one residual distribution")

    ax = axes[0, 2]
    rounds = np.arange(1, len(entropy) + 1)
    ax.plot(rounds, entropy, color=OUR, lw=2, label="Weight entropy")
    ax2 = ax.twinx()
    ax2.plot(rounds, cosine, color="#3399FF", lw=1.5, label="Update coherence")
    ax.set_xlabel("Communication round (rounds)")
    ax.set_ylabel("Weight entropy (nats)")
    ax2.set_ylabel("Mean update cosine")
    ax.tick_params(axis="y", labelrotation=42)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")
    ax2.tick_params(axis="y", labelrotation=-42)
    for label in ax2.get_yticklabels():
        label.set_horizontalalignment("left")
    panel(ax, "c", "Allocation entropy and cohort coherence")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right")

    residuals, relative_mass = [], []
    for entry in diagnostics:
        error = np.asarray(entry["reconstruction_error_by_layer"])
        weight = np.asarray(entry["weights_by_layer"])
        sizes = np.asarray(entry["client_sizes"], dtype=float)
        base = sizes / sizes.sum()
        residuals.extend(error.ravel().tolist())
        relative_mass.extend((weight / base[:, None]).ravel().tolist())
    residuals, relative_mass = np.asarray(residuals), np.asarray(relative_mass)

    ax = axes[1, 0]
    if "reconstruction_error_by_layer" in diagnostics[0]:
        edges = np.quantile(residuals, np.linspace(0, 1, 6))
        groups = [relative_mass[(residuals >= edges[j]) &
                                (residuals <= edges[j + 1] if j == 4 else residuals < edges[j + 1])]
                  for j in range(5)]
        box = ax.boxplot(groups, patch_artist=True, widths=.65, showfliers=False,
                         medianprops={"color": "#222222", "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set_facecolor(OUR); patch.set_alpha(.58); patch.set_edgecolor("#AA3333")
        ax.axhline(1.0, color="#333333", ls="--", lw=1)
        ax.set_xticks(range(1, 6), ["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.set_xlabel("Leave-one-client residual quintile\n(low to high)")
        ax.set_ylabel("Weight / sample mass (ratio)")
        panel(ax, "d", "Residual-stratified allocation")
    else:
        budget = np.asarray([d["rho"] for d in diagnostics])
        ax.plot(rounds, budget, color=OUR, lw=2)
        ax.set_xlabel("Communication round (rounds)")
        ax.set_ylabel("Learned-weight budget (fraction)")
        panel(ax, "d", "Allocation-budget audit")

    ax = axes[1, 1]
    ax.plot(rounds, fusion_bias, color=OUR, lw=2.2, label="Fusion bias")
    ax.fill_between(rounds, 0, fusion_bias, color=OUR, alpha=.10)
    ax2 = ax.twinx()
    ax2.plot(rounds, rho, color="#3399FF", lw=1.6, marker="o", ms=3,
             markevery=3, label="$\\rho_t$")
    ax.set_xlabel("Communication round (rounds)")
    ax.set_ylabel("Fusion bias (normalized) ↓")
    ax2.set_ylabel("$\\rho_t$ (fraction)")
    ax2.tick_params(axis="y", labelrotation=90)
    panel(ax, "e", "Bias–budget temporal audit")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center",
              ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=.8,
              handletextpad=.35)

    ax = axes[1, 2]
    density = ax.hexbin(residuals, relative_mass, gridsize=22, mincnt=1,
                        cmap=RED, linewidths=0)
    edges = np.quantile(residuals, np.linspace(0, 1, 9))
    centers, medians = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (residuals >= left) & (residuals <= right)
        centers.append(np.median(residuals[mask])); medians.append(np.median(relative_mass[mask]))
    ax.plot(centers, medians, color="#222222", marker="o", lw=1.8,
            label="bin median")
    ax.axhline(1.0, color="#3399FF", ls="--", lw=1.1, label="sample mass")
    ax.set_xlabel("Leave-one-client residual (normalized) ↓")
    ax.set_ylabel("Weight / sample mass (ratio)")
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "f", "Residual–mass response field")
    fig.colorbar(density, ax=ax, fraction=.045, pad=.02, label="Observation count")

    fig.subplots_adjust(top=0.92, hspace=0.40, wspace=0.48, left=0.08, right=0.98,
                        bottom=0.09)
    path = figure_path("operator_diagnostics")
    fig.savefig(path)
    plt.close(fig)
    return path


def write_tables() -> None:
    df = load_runs()
    df = df[df.method.isin(PAPER_METHODS)]
    accuracy = df.groupby(["dataset", "regime", "method"])["accuracy"].agg(["mean", "std"]).reset_index()
    accuracy.to_csv(TABLES / "main_accuracy.csv", index=False, float_format="%.6f")
    pivot = accuracy.pivot(index="method", columns=["dataset", "regime"])
    pivot.to_csv(TABLES / "main_accuracy_pivot.csv", float_format="%.6f")

    # Matched ablation summaries are generated from the frozen result records. Never
    # mix final-protocol FedCANTO with legacy same-round ablation logs.
    # Remove intermediate tabulations that duplicate plotted endpoints.
    for duplicate in [TABLES / "main_results.csv", TABLES / "rank_summary.csv"]:
        if duplicate.exists():
            duplicate.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "evaluation", "convergence",
                                           "sensitivity", "diagnostics"), default="all")
    parser.add_argument("--qa", action="store_true", help="Render temporary PNG only; do not publish")
    args = parser.parse_args()
    global QA_ONLY
    QA_ONLY = args.qa
    setup()
    selected = {
        "convergence": plot_convergence,
        "evaluation": plot_evaluation,
        "sensitivity": plot_sensitivity_stress,
        "diagnostics": plot_operator_diagnostics,
    }
    if args.only == "all":
        write_tables()
        paths = [builder() for builder in selected.values()]
    else:
        paths = [selected[args.only]()]
    if not QA_ONLY:
        for path in paths:
            for paper_figures in PAPER_FIGURE_DIRS:
                shutil.copy2(path, paper_figures / path.name)
    manifest = {"figures": [str(p) for p in paths],
                "paper_sync": [str(folder) for folder in PAPER_FIGURE_DIRS],
                "format": "temporary QA PNG" if QA_ONLY else "PDF only"}
    if not QA_ONLY:
        (MODELS / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
