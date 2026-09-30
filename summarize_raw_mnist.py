"""Summarize final H=7 paired-bank raw-MNIST confirmation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TAG = "round14_raw_final_paired"


def read(method: str, variant: str, seed: int) -> dict:
    stem = f"mnist__label_skew__{method}__s{seed}"
    if variant != "full":
        stem += f"__{variant}"
    return json.loads((MODELS / f"{stem}__{TAG}.json").read_text(encoding="utf-8"))


def interval(values: list[float], seed: int) -> list[float]:
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(20000))
    return [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]]


def main() -> None:
    rows = []
    for seed in range(60, 65):
        gate = read("fedavgm_gate", "full", seed)
        router = read("fedcanto", "lite_paired_gate_bank_router_control", seed)
        canto = read("fedcanto", "lite_paired_gate_bank_selector", seed)
        auc = lambda r: 100 * r["final"]["convergence_auc"]
        server = lambda r: 1000 * mean(d["aggregation_seconds"] for d in r["diagnostics"])
        rows.append({"seed": seed, "gate_auc_pct": auc(gate),
                     "router_auc_pct": auc(router), "fedcanto_auc_pct": auc(canto),
                     "canto_minus_router_pp": auc(canto) - auc(router),
                     "canto_minus_gate_pp": auc(canto) - auc(gate),
                     "gate_server_ms_round": server(gate),
                     "router_server_ms_round": server(router),
                     "canto_server_ms_round": server(canto),
                     "gate_wall_s": gate["runtime_seconds"],
                     "router_wall_s": router["runtime_seconds"],
                     "canto_wall_s": canto["runtime_seconds"]})
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "round14_raw_final_paired.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cr = [x["canto_minus_router_pp"] for x in rows]
    cg = [x["canto_minus_gate_pp"] for x in rows]
    report = {"status": "complete", "paired_seeds": 5,
              "canto_minus_router_pp": mean(cr), "canto_minus_router_sd_pp": stdev(cr),
              "canto_minus_router_ci95_pp": interval(cr, 1460),
              "canto_minus_router_positive": sum(x > 0 for x in cr),
              "canto_minus_gate_pp": mean(cg), "canto_minus_gate_ci95_pp": interval(cg, 1461),
              "mean_server_ms_round": {
                  "AvgM+Gate": mean(x["gate_server_ms_round"] for x in rows),
                  "Router-only": mean(x["router_server_ms_round"] for x in rows),
                  "FedCANTO": mean(x["canto_server_ms_round"] for x in rows)},
              "mean_wall_s": {"AvgM+Gate": mean(x["gate_wall_s"] for x in rows),
                              "Router-only": mean(x["router_wall_s"] for x in rows),
                              "FedCANTO": mean(x["canto_wall_s"] for x in rows)}}
    (MODELS / "round14_raw_final_paired_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
