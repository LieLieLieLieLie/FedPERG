"""Confirmatory residual attribution, cluster sensitivity, and time-domain audit."""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TASKS = [(d, r) for d in ("cifar10", "cifar100", "officehome")
         for r in ("label_skew", "quantity_skew", "compound")]
TASKS.append(("mnist_mobilenet", "compound"))
SOURCES = ["cifar10", "cifar100", "officehome", "mnist_mobilenet"]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_path(dataset: str, regime: str, seed: int, variant: str, tag: str) -> Path:
    return MODELS / f"{dataset}__{regime}__fedperg__s{seed}__{variant}__{tag}.json"


def q(values: list[float], p: float) -> float:
    values = sorted(values); x = p * (len(values) - 1)
    lo = int(x); hi = min(lo + 1, len(values) - 1); a = x - lo
    return values[lo] * (1 - a) + values[hi] * a


def boot_ci(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = [mean(rng.choices(values, k=len(values))) for _ in range(20000)]
    return [q(draws, .025), q(draws, .975)]


def source_balanced_by_seed(rows: list[dict], field: str) -> dict[int, float]:
    out = {}
    for seed in sorted({x["seed"] for x in rows}):
        cells = [x for x in rows if x["seed"] == seed]
        out[seed] = mean(mean(x[field] for x in cells if x["dataset"] == source)
                         for source in SOURCES)
    return out


def exact_sign_flip_p(values: list[float]) -> float:
    observed = abs(mean(values))
    null = [abs(mean(sign * value for sign, value in zip(signs, values)))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(x >= observed - 1e-15 for x in null) / len(null)


def time_auc(run: dict, horizon: float) -> float:
    history = sorted(run["history"], key=lambda x: x["elapsed_seconds"])
    xs = np.array([float(x["elapsed_seconds"]) for x in history])
    ys = 100 * np.array([float(x["accuracy"]) for x in history])
    if horizon < xs[-1]:
        y_h = np.interp(horizon, xs, ys)
        keep = xs < horizon
        xs = np.r_[xs[keep], horizon]; ys = np.r_[ys[keep], y_h]
    return float(np.trapz(ys, xs) / max(horizon, 1e-12))


def main() -> None:
    rows = []
    evidence_rows = []
    for dataset, regime in TASKS:
        for seed in range(50, 60):
            full = read(run_path(dataset, regime, seed,
                                 "lite_paired_gate_bank_selector",
                                 "round14_final_paired_confirmation"))
            nores = read(run_path(dataset, regime, seed,
                                  "lite_paired_gate_bank_no_residual",
                                  "round15_frozen_residual_ablation"))
            router = read(run_path(dataset, regime, seed,
                                   "lite_paired_gate_bank_router_control",
                                   "round14_final_paired_confirmation"))
            rows.append({
                "dataset": dataset, "regime": regime, "seed": seed,
                "full_auc_pp": 100 * full["final"]["convergence_auc"],
                "nores_auc_pp": 100 * nores["final"]["convergence_auc"],
                "delta_pp": 100 * (full["final"]["convergence_auc"] -
                                    nores["final"]["convergence_auc"]),
                "full_wall_s": full["runtime_seconds"],
                "nores_wall_s": nores["runtime_seconds"],
            })
            evidence_rows.append({"dataset": dataset, "regime": regime, "seed": seed,
                                  "delta_pp": 100 * (full["final"]["convergence_auc"] -
                                                     router["final"]["convergence_auc"])})
    seed_effects = source_balanced_by_seed(rows, "delta_pp")
    seed_values = list(seed_effects.values())
    estimate = mean(seed_values)
    se = stdev(seed_values) / math.sqrt(len(seed_values))
    tcrit = t.ppf(.975, len(seed_values) - 1)
    source_summary = []
    for source in SOURCES:
        vals = []
        for seed in range(50, 60):
            vals.append(mean(x["delta_pp"] for x in rows
                             if x["dataset"] == source and x["seed"] == seed))
        source_summary.append({"source": source, "delta_pp": mean(vals),
                               "ci95_pp": boot_ci(vals, 1510 + len(source_summary)),
                               "positive_seeds": sum(x > 0 for x in vals)})
    source_means = {x["source"]: x["delta_pp"] for x in source_summary}
    leave_source = {source: mean(v for s, v in source_means.items() if s != source)
                    for source in SOURCES}
    leave_seed = {seed: mean(v for s, v in seed_effects.items() if s != seed)
                  for seed in seed_effects}

    evidence_seed = source_balanced_by_seed(evidence_rows, "delta_pp")
    evidence_sources = []
    for source in SOURCES:
        vals = [mean(x["delta_pp"] for x in evidence_rows
                     if x["dataset"] == source and x["seed"] == seed)
                for seed in range(50, 60)]
        evidence_sources.append({"source": source, "delta_pp": mean(vals),
                                 "ci95_pp": boot_ci(vals, 1530 + len(evidence_sources)),
                                 "positive_seeds": sum(x > 0 for x in vals)})
    evidence_source_means = {x["source"]: x["delta_pp"] for x in evidence_sources}
    evidence_leave_source = {
        source: mean(v for s, v in evidence_source_means.items() if s != source)
        for source in SOURCES}

    # Common-horizon wall-clock AUC for the primary fixed-feature/frozen-encoder runs.
    time_rows = []
    for dataset, regime in TASKS:
        for seed in range(50, 60):
            gate = read(MODELS / f"{dataset}__{regime}__fedavgm_gate__s{seed}__round14_gate_reference.json")
            router = read(run_path(dataset, regime, seed,
                                   "lite_paired_gate_bank_router_control",
                                   "round14_final_paired_confirmation"))
            full = read(run_path(dataset, regime, seed,
                                 "lite_paired_gate_bank_selector",
                                 "round14_final_paired_confirmation"))
            horizon = min(gate["history"][-1]["elapsed_seconds"],
                          router["history"][-1]["elapsed_seconds"],
                          full["history"][-1]["elapsed_seconds"])
            values = {"AvgM+Gate": time_auc(gate, horizon),
                      "Router-only": time_auc(router, horizon),
                      "FedPERG": time_auc(full, horizon)}
            time_rows.append({"dataset": dataset, "regime": regime, "seed": seed,
                              "horizon_s": horizon, **values,
                              "perg_minus_gate": values["FedPERG"] - values["AvgM+Gate"],
                              "perg_minus_router": values["FedPERG"] - values["Router-only"]})
    time_gate = source_balanced_by_seed(time_rows, "perg_minus_gate")
    time_router = source_balanced_by_seed(time_rows, "perg_minus_router")

    # Raw-MNIST common-horizon time AUC.
    raw_rows = []
    for seed in range(60, 65):
        gate = read(MODELS / f"mnist__label_skew__fedavgm_gate__s{seed}__round14_raw_final_paired.json")
        router = read(MODELS / f"mnist__label_skew__fedperg__s{seed}__lite_paired_gate_bank_router_control__round14_raw_final_paired.json")
        full = read(MODELS / f"mnist__label_skew__fedperg__s{seed}__lite_paired_gate_bank_selector__round14_raw_final_paired.json")
        horizon = min(x["history"][-1]["elapsed_seconds"] for x in (gate, router, full))
        g, r, f = (time_auc(x, horizon) for x in (gate, router, full))
        raw_rows.append({"seed": seed, "horizon_s": horizon, "AvgM+Gate": g,
                         "Router-only": r, "FedPERG": f,
                         "perg_minus_gate": f-g, "perg_minus_router": f-r})

    TABLES.mkdir(parents=True, exist_ok=True)
    import csv
    for name, data in (("round15_residual_ablation_runs.csv", rows),
                       ("round15_time_auc_runs.csv", time_rows),
                       ("round15_raw_time_auc_runs.csv", raw_rows)):
        with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)

    report = {
        "status": "complete",
        "evidence_attribution_sensitivity": {
            "source_balanced_delta_pp": mean(evidence_seed.values()),
            "cluster_bootstrap_ci95_pp": boot_ci(list(evidence_seed.values()), 1520),
            "t_cluster_ci95_pp": [
                mean(evidence_seed.values()) - t.ppf(.975, 9) * stdev(evidence_seed.values()) / math.sqrt(10),
                mean(evidence_seed.values()) + t.ppf(.975, 9) * stdev(evidence_seed.values()) / math.sqrt(10),
            ],
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(list(evidence_seed.values())),
            "seed_effects_pp": evidence_seed,
            "source_summary": evidence_sources,
            "leave_one_source_out_mean_pp": evidence_leave_source,
            "non_mnist_source_balanced_pp": evidence_leave_source["mnist_mobilenet"],
        },
        "residual_ablation": {
            "source_balanced_delta_pp": estimate,
            "cluster_bootstrap_ci95_pp": boot_ci(seed_values, 1501),
            "t_cluster_ci95_pp": [estimate - tcrit * se, estimate + tcrit * se],
            "exact_two_sided_sign_flip_p": exact_sign_flip_p(seed_values),
            "positive_seed_clusters": sum(x > 0 for x in seed_values),
            "positive_task_seed_cells": sum(x["delta_pp"] > 0 for x in rows),
            "seed_effects_pp": seed_effects,
            "leave_one_seed_out_mean_pp": leave_seed,
            "source_summary": source_summary,
            "leave_one_source_out_mean_pp": leave_source,
            "non_mnist_source_balanced_pp": leave_source["mnist_mobilenet"],
        },
        "time_auc": {
            "definition": "accuracy-time area divided by the fastest method's paired common horizon",
            "fixed_source_balanced_perg_minus_gate_pp": mean(time_gate.values()),
            "fixed_source_balanced_perg_minus_gate_ci95_pp": boot_ci(list(time_gate.values()), 1502),
            "fixed_source_balanced_perg_minus_router_pp": mean(time_router.values()),
            "fixed_source_balanced_perg_minus_router_ci95_pp": boot_ci(list(time_router.values()), 1503),
            "raw_perg_minus_gate_pp": mean(x["perg_minus_gate"] for x in raw_rows),
            "raw_perg_minus_router_pp": mean(x["perg_minus_router"] for x in raw_rows),
        },
    }
    (MODELS / "round15_rereview_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
