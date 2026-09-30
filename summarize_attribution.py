"""Statistics for the frozen final H=7 paired-bank confirmation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TASKS = [(d, r) for d in ("cifar10", "cifar100", "officehome")
         for r in ("label_skew", "quantity_skew", "compound")]
TASKS.append(("mnist_mobilenet", "compound"))
TAG = "round14_final_paired_confirmation"


def load(dataset: str, regime: str, variant: str, seed: int) -> dict:
    path = MODELS / (f"{dataset}__{regime}__fedcanto__s{seed}__{variant}__{TAG}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_gate(dataset: str, regime: str, seed: int) -> dict:
    path = MODELS / (f"{dataset}__{regime}__fedavgm_gate__s{seed}__"
                     "round14_gate_reference.json")
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], p: float) -> float:
    values = sorted(values); pos = p * (len(values) - 1)
    lo = int(pos); hi = min(lo + 1, len(values) - 1); w = pos - lo
    return values[lo] * (1 - w) + values[hi] * w


def cluster_effect(rows: list[dict], source_balanced: bool, seed: int,
                   field: str = "delta_pp") -> tuple[float, list[float]]:
    seeds = sorted({x["seed"] for x in rows})
    estimates = {}
    for value in seeds:
        cells = [x for x in rows if x["seed"] == value]
        if source_balanced:
            estimates[value] = mean(mean(x[field] for x in cells if x["dataset"] == source)
                                    for source in sorted({x["dataset"] for x in cells}))
        else:
            estimates[value] = mean(x[field] for x in cells)
    rng = random.Random(seed)
    boot = [mean(estimates[s] for s in rng.choices(seeds, k=len(seeds)))
            for _ in range(20000)]
    return mean(estimates.values()), [quantile(boot, .025), quantile(boot, .975)]


def main() -> None:
    rows = []
    for dataset, regime in TASKS:
        for seed in range(50, 60):
            router = load(dataset, regime, "lite_paired_gate_bank_router_control", seed)
            canto = load(dataset, regime, "lite_paired_gate_bank_selector", seed)
            gate = load_gate(dataset, regime, seed)
            router_auc = 100 * router["final"]["convergence_auc"]
            canto_auc = 100 * canto["final"]["convergence_auc"]
            gate_auc = 100 * gate["final"]["convergence_auc"]
            rows.append({
                "dataset": dataset, "regime": regime, "seed": seed,
                "gate_auc_pct": gate_auc,
                "router_auc_pct": router_auc, "fedcanto_auc_pct": canto_auc,
                "delta_pp": canto_auc - router_auc,
                "router_minus_gate_pp": router_auc - gate_auc,
                "canto_minus_gate_pp": canto_auc - gate_auc,
                "gate_server_ms_round": 1000 * mean(d["aggregation_seconds"]
                                                      for d in gate["diagnostics"]),
                "router_server_ms_round": 1000 * mean(d["aggregation_seconds"]
                                                        for d in router["diagnostics"]),
                "fedcanto_server_ms_round": 1000 * mean(d["aggregation_seconds"]
                                                         for d in canto["diagnostics"]),
                "router_wall_s": router["runtime_seconds"],
                "fedcanto_wall_s": canto["runtime_seconds"],
                "gate_wall_s": gate["runtime_seconds"],
                "evidence_selection_rate": mean(d.get("selector_used_target_evidence", False)
                                                for d in canto["diagnostics"]),
            })
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "round14_final_paired_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    task_summary = []
    for dataset, regime in TASKS:
        cells = [x for x in rows if x["dataset"] == dataset and x["regime"] == regime]
        effects = [x["delta_pp"] for x in cells]
        task_summary.append({
            "dataset": dataset, "regime": regime,
            "router_auc_pct": mean(x["router_auc_pct"] for x in cells),
            "fedcanto_auc_pct": mean(x["fedcanto_auc_pct"] for x in cells),
            "delta_pp": mean(effects), "paired_sd_pp": stdev(effects),
            "positive_seeds": sum(x > 0 for x in effects),
            "evidence_selection_rate": mean(x["evidence_selection_rate"] for x in cells),
        })
    with (TABLES / "round14_final_paired_task_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(task_summary[0])); writer.writeheader(); writer.writerows(task_summary)
    source_mean, source_ci = cluster_effect(rows, True, 1450)
    task_mean, task_ci = cluster_effect(rows, False, 1451)
    router_gate_mean, router_gate_ci = cluster_effect(rows, True, 1452, "router_minus_gate_pp")
    canto_gate_mean, canto_gate_ci = cluster_effect(rows, True, 1453, "canto_minus_gate_pp")
    report = {
        "status": "frozen_final_paired_confirmation_complete",
        "paired_runs": len(rows),
        "primary_estimand": "source-balanced FedCANTO minus Router-only AUC",
        "bootstrap_unit": "paired seed cluster; all source/regime cells move together",
        "source_balanced_delta_pp": source_mean,
        "source_balanced_ci95_pp": source_ci,
        "task_weighted_delta_pp": task_mean,
        "task_weighted_ci95_pp": task_ci,
        "router_minus_gate_source_balanced_pp": router_gate_mean,
        "router_minus_gate_source_balanced_ci95_pp": router_gate_ci,
        "fedcanto_minus_gate_source_balanced_pp": canto_gate_mean,
        "fedcanto_minus_gate_source_balanced_ci95_pp": canto_gate_ci,
        "positive_pairs": sum(x["delta_pp"] > 0 for x in rows),
        "mean_evidence_selection_rate": mean(x["evidence_selection_rate"] for x in rows),
        "mean_server_ms_round": {
            "AvgM+Gate": mean(x["gate_server_ms_round"] for x in rows),
            "Router-only": mean(x["router_server_ms_round"] for x in rows),
            "FedCANTO": mean(x["fedcanto_server_ms_round"] for x in rows)},
        "mean_wall_s": {"AvgM+Gate": mean(x["gate_wall_s"] for x in rows),
                        "Router-only": mean(x["router_wall_s"] for x in rows),
                        "FedCANTO": mean(x["fedcanto_wall_s"] for x in rows)},
        "task_summary": task_summary,
    }
    (MODELS / "round14_final_paired_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
