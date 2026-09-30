from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
OUT = ROOT / "results" / "figures" / "calibration_robustness.pdf"
PAPER = ROOT.parent / "paper" / "INF" / "figures" / OUT.name
RED = "#FF6666"
ORANGE = "#FFAA53"
GREEN = "#50CC55"
BLUE = "#3399FF"
PURPLE = "#6666FF"


def panel(ax, letter: str, title: str) -> None:
    ax.text(-0.19, 1.14, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="top", clip_on=False)
    ax.set_title(title, pad=7)
    ax.grid(True, color="#D9D9D9", lw=.55, alpha=.7)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 13,
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "axes.linewidth": 1,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .04,
    })
    robust = json.loads((MODELS / "round14_calibration_robustness_statistics.json").read_text())["summary"]
    final = json.loads((MODELS / "round14_final_paired_statistics.json").read_text())
    raw = json.loads((MODELS / "round14_raw_final_paired_statistics.json").read_text())
    labels = ["1%", "2%", "4%", "8%", "4%\n+ label shift"]
    x = np.arange(5)
    effect = np.array([r["canto_minus_router_pp"] for r in robust])
    sd = np.array([r["effect_sd_pp"] for r in robust])
    select = 100 * np.array([r["evidence_selection_rate"] for r in robust])
    agree = 100 * np.array([r["decision_agreement"] for r in robust])
    cal = 1000 * np.array([r["calibration_loss_gain"] for r in robust])
    test = 1000 * np.array([r["test_loss_gain"] for r in robust])

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.25))
    ax = axes[0, 0]
    ax.errorbar(x, effect, yerr=sd, color=RED, marker="o", lw=2, capsize=3)
    ax.axhline(0, color="#4D4D4D", ls="--", lw=1)
    ax.set_xticks(x, labels); ax.set_ylabel("AUC gain over Router-only (pp) ↑")
    panel(ax, "a", "Calibration size and shift")

    ax = axes[0, 1]
    ax.plot(x, select, color=RED, marker="o", lw=2, label="Evidence selected")
    ax.plot(x, agree, color=BLUE, marker="s", lw=2, label="Cal./test agree")
    ax.set_xticks(x, labels); ax.set_ylabel("Rate (%) ↑"); ax.set_ylim(20, 65)
    panel(ax, "b", "Routing diagnostics")

    ax = axes[0, 2]
    ax.scatter(cal, test, s=85, c=[ORANGE, GREEN, RED, PURPLE, BLUE], edgecolor="#333333")
    for i, lab in enumerate(labels):
        text = lab.replace("\n", " ")
        if i == len(labels) - 1:
            # The shifted 4% pool is genuinely in the lower-right quadrant:
            # calibration loss improves while test loss worsens.  Place its
            # label inside the axes so the diagnostic reads as deliberate
            # evidence of calibration/test misalignment rather than overflow.
            ax.annotate(text, (cal[i], test[i]), xytext=(-8, 7),
                        textcoords="offset points", ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="-", color="#4D4D4D", lw=.7))
        else:
            ax.annotate(text, (cal[i], test[i]), xytext=(4, 4),
                        textcoords="offset points")
    ax.axhline(0, color="#4D4D4D", ls="--", lw=1); ax.axvline(0, color="#4D4D4D", ls="--", lw=1)
    ax.set_xlabel("Calibration-loss gain (×10⁻³) ↑"); ax.set_ylabel("Test-loss gain (×10⁻³) ↑")
    panel(ax, "c", "Calibration–test alignment")

    ax = axes[1, 0]
    tasks = final["task_summary"]
    vals = np.array([r["delta_pp"] for r in tasks])
    order = np.argsort(vals)
    names = [f"{r['dataset'].replace('officehome','OH').replace('mnist_mobilenet','MNIST')}–{r['regime'][0].upper()}" for r in tasks]
    colors = [RED if vals[i] >= 0 else BLUE for i in order]
    ax.barh(np.arange(len(vals)), vals[order], color=colors, alpha=.9)
    ax.axvline(0, color="#333333", lw=1); ax.set_yticks(np.arange(len(vals)), np.array(names)[order], rotation=28)
    ax.set_xlabel("FedCANTO − Router-only AUC (pp) ↑")
    panel(ax, "d", "Task-resolved evidence effect")

    ax = axes[1, 1]
    methods = ["AvgM+Gate", "Router-only", "FedCANTO"]
    fixed = [final["mean_server_ms_round"][m] for m in methods]
    raw_ms = [raw["mean_server_ms_round"][m] for m in methods]
    xx = np.arange(3); width = .36
    ax.bar(xx-width/2, fixed, width, color=[ORANGE, BLUE, RED], alpha=.75, label="Fixed features")
    ax.bar(xx+width/2, raw_ms, width, color=[ORANGE, BLUE, RED], hatch="//", alpha=.45, label="Raw pixels")
    ax.set_yscale("log"); ax.set_xticks(xx, methods, rotation=28, ha="right")
    ax.set_ylabel("Server latency (ms/round) ↓"); ax.legend(frameon=False, loc="upper left")
    panel(ax, "e", "Matched-resource cost")

    ax = axes[1, 2]
    values = [final["router_minus_gate_source_balanced_pp"], final["fedcanto_minus_gate_source_balanced_pp"], raw["canto_minus_router_pp"]]
    lo = [final["router_minus_gate_source_balanced_ci95_pp"][0], final["fedcanto_minus_gate_source_balanced_ci95_pp"][0], raw["canto_minus_router_ci95_pp"][0]]
    hi = [final["router_minus_gate_source_balanced_ci95_pp"][1], final["fedcanto_minus_gate_source_balanced_ci95_pp"][1], raw["canto_minus_router_ci95_pp"][1]]
    names2 = ["Router − Gate", "CANTO − Gate", "Raw CANTO − Router"]
    err = np.vstack([np.array(values)-np.array(lo), np.array(hi)-np.array(values)])
    ax.errorbar(np.arange(3), values, yerr=err, fmt="o", color=RED, capsize=4, lw=2)
    ax.axhline(0, color="#4D4D4D", ls="--", lw=1); ax.set_xticks(np.arange(3), names2, rotation=28, ha="right")
    ax.set_ylabel("AUC effect (pp) ↑")
    panel(ax, "f", "Confirmatory effects (95% CI)")

    handles = [mpl.lines.Line2D([], [], color=RED, marker="o", label="FedCANTO evidence path"),
               mpl.lines.Line2D([], [], color=BLUE, marker="s", label="Matched diagnostic")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.015))
    fig.subplots_adjust(left=.08, right=.99, top=.93, bottom=.18, wspace=.48, hspace=.48)
    OUT.parent.mkdir(parents=True, exist_ok=True); PAPER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT); plt.close(fig); shutil.copy2(OUT, PAPER)
    print(OUT)


if __name__ == "__main__":
    main()
