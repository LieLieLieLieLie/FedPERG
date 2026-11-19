"""Summarize the frozen public-calibrated gate confirmation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TAG = "round13_frozen_public_gate_confirmation"
TASKS = [(d, r) for d in ("cifar10", "cifar100", "officehome")
         for r in ("label_skew", "quantity_skew", "compound")]
TASKS.append(("mnist_mobilenet", "compound"))


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], q: float) -> float:
    values = sorted(values); pos = q * (len(values) - 1)
    low = int(pos); high = min(low + 1, len(values) - 1); w = pos - low
    return values[low] * (1 - w) + values[high] * w


def ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [mean(rng.choices(values, k=len(values))) for _ in range(20000)]
    return quantile(draws, 0.025), quantile(draws, 0.975)


def clustered_ci(rows: list[dict], source_balanced: bool, seed: int) -> tuple[float, float]:
    """Paired seed-cluster bootstrap; all task cells for a seed move together."""
    seed_ids = sorted({int(x["seed"]) for x in rows})
    by_seed = {}
    for seed_id in seed_ids:
        current = [x for x in rows if int(x["seed"]) == seed_id]
        if source_balanced:
            source_means = []
            for source in sorted({x["dataset"] for x in current}):
                source_means.append(mean(x["paired_auc_delta_pp"] for x in current
                                         if x["dataset"] == source))
            by_seed[seed_id] = mean(source_means)
        else:
            by_seed[seed_id] = mean(x["paired_auc_delta_pp"] for x in current)
    rng = random.Random(seed)
    draws = [mean(by_seed[s] for s in rng.choices(seed_ids, k=len(seed_ids)))
             for _ in range(20000)]
    return quantile(draws, .025), quantile(draws, .975)


def main() -> None:
    rows = []
    for dataset, regime in TASKS:
        for seed in range(20, 30):
            prefix = f"{dataset}__{regime}"
            gate = read(MODELS / f"{prefix}__fedavgm_gate__s{seed}__{TAG}.json")
            perg = read(MODELS / (f"{prefix}__fedperg__s{seed}__lite_gate_bank_selector__"
                                   f"{TAG}.json"))
            effect = 100 * (perg["final"]["convergence_auc"] - gate["final"]["convergence_auc"])
            selected = sum(d.get("selector_used_evidence", False) for d in perg["diagnostics"])
            rows.append({
                "dataset": dataset, "regime": regime, "seed": seed,
                "gate_auc_pct": 100 * gate["final"]["convergence_auc"],
                "fedperg_auc_pct": 100 * perg["final"]["convergence_auc"],
                "paired_auc_delta_pp": effect,
                "gate_final_accuracy_pct": 100 * gate["final"]["accuracy"],
                "fedperg_final_accuracy_pct": 100 * perg["final"]["accuracy"],
                "evidence_rounds": selected,
                "gate_runtime_s": gate["runtime_seconds"],
                "fedperg_runtime_s": perg["runtime_seconds"],
            })
    run_path = TABLES / "round13_frozen_public_gate_confirmation_runs.csv"
    with run_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for index, (dataset, regime) in enumerate(TASKS):
        task = [x for x in rows if x["dataset"] == dataset and x["regime"] == regime]
        effects = [x["paired_auc_delta_pp"] for x in task]; low, high = ci(effects, 9028 + index)
        summaries.append({
            "dataset": dataset, "regime": regime,
            "gate_auc_mean_pct": mean(x["gate_auc_pct"] for x in task),
            "fedperg_auc_mean_pct": mean(x["fedperg_auc_pct"] for x in task),
            "paired_delta_mean_pp": mean(effects), "paired_delta_sd_pp": stdev(effects),
            "paired_delta_ci_low_pp": low, "paired_delta_ci_high_pp": high,
            "positive_seeds": sum(x > 0 for x in effects),
            "evidence_selection_rate": mean(x["evidence_rounds"] / 30 for x in task),
        })
    summary_path = TABLES / "round13_frozen_public_gate_confirmation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    fixed = [x for x in rows if x["dataset"] != "mnist_mobilenet"]
    fixed_effects = [x["paired_auc_delta_pp"] for x in fixed]
    all_effects = [x["paired_auc_delta_pp"] for x in rows]
    source_effects = []
    for dataset in sorted({x["dataset"] for x in rows}):
        source_effects.append(mean(x["paired_auc_delta_pp"] for x in rows
                                   if x["dataset"] == dataset))
    report = {
        "status": "frozen_confirmation_complete",
        "runs_per_method": 100,
        "fixed_feature_mean_delta_pp": mean(fixed_effects),
        "fixed_feature_seed_cluster_ci95_pp": clustered_ci(fixed, False, 211),
        "fixed_feature_positive_runs": sum(x > 0 for x in fixed_effects),
        "fixed_feature_positive_task_means": sum(x["paired_delta_mean_pp"] > 0 for x in summaries[:-1]),
        "mnist_mean_delta_pp": summaries[-1]["paired_delta_mean_pp"],
        "mnist_ci95_pp": [summaries[-1]["paired_delta_ci_low_pp"], summaries[-1]["paired_delta_ci_high_pp"]],
        "all_task_weighted_mean_delta_pp": mean(all_effects),
        "all_task_weighted_seed_cluster_ci95_pp": clustered_ci(rows, False, 223),
        "source_balanced_mean_delta_pp": mean(source_effects),
        "source_balanced_seed_cluster_ci95_pp": clustered_ci(rows, True, 227),
        "bootstrap_unit": "paired seed cluster; all source/regime cells move together",
        "all_positive_runs": sum(x > 0 for x in all_effects),
        "task_summary": summaries,
    }
    (MODELS / "round13_frozen_public_gate_confirmation_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
