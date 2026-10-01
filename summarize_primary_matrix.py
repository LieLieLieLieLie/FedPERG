"""Summarize the matched public-split ten-method matrix and emit LaTeX rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
DATASETS = ("cifar10", "cifar100", "officehome")
REGIMES = ("label_skew", "quantity_skew", "compound")
METHODS = ("FedAvg", "FedAvgM", "SCAFFOLD", "FedAdam", "FedLAW",
           "FedCDA", "FedPW", "Fed-NGA", "FedPhoenix", "FedPERG")


def load(dataset: str, regime: str, method: str, seed: int) -> dict:
    tag = ("round13_frozen_public_gate_confirmation" if method == "FedPERG"
           else "round13_public_ten_method_matrix")
    stem = f"{dataset}__{regime}__{method.lower().replace('-', '_')}__s{seed}"
    if method == "FedPERG": stem += "__lite_gate_bank_selector"
    return json.loads((MODELS / f"{stem}__{tag}.json").read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            for regime in REGIMES:
                values = [100 * load(dataset, regime, method, seed)["final"]["convergence_auc"]
                          for seed in range(20, 23)]
                rows.append({"method": method, "dataset": dataset, "regime": regime,
                             "auc_mean_pct": mean(values)})
    with (TABLES / "round13_public_ten_method_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matrix = {}
    for method in METHODS:
        dataset_means = []
        for dataset in DATASETS:
            dataset_means.append(mean(x["auc_mean_pct"] for x in rows
                                      if x["method"] == method and x["dataset"] == dataset))
        matrix[method] = [*dataset_means, mean(dataset_means)]
    for method, values in matrix.items():
        print(method + " &" + "&".join(f"{x:.2f}" for x in values) + r"\\")
    (MODELS / "round13_public_ten_method_summary.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
