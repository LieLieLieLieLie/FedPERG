from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fedperg.config import DATASETS, METHODS, REGIMES


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
FORMAL_DATASETS = ["cifar10", "cifar100", "officehome"]
PAPER_METHODS = ["FedAvg", "FedAvgM", "SCAFFOLD", "FedAdam", "FedLAW",
                 "FedCDA", "FedPW", "Fed-NGA", "FedPhoenix", "FedPERG"]
TABLES.mkdir(parents=True, exist_ok=True)


def load_runs() -> pd.DataFrame:
    rows = []
    for path in MODELS.glob("*.json"):
        if path.name in {"run_summary.json", "aggregate_summary.json"}:
            continue
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(run, dict):
            continue
        cfg = run.get("config", {})
        if cfg.get("rounds") != 30 or cfg.get("clients") != 20 or cfg.get("clients_per_round") != 8:
            continue
        if cfg.get("method") == "FedPERG":
            if (cfg.get("perg_variant") != "lite_paired_gate_bank_selector" or
                    cfg.get("experiment_tag") != "round14_final_breadth_update" or
                    cfg.get("seed") not in range(20, 23)):
                continue
        else:
            if (cfg.get("method") not in PAPER_METHODS or
                    cfg.get("perg_variant", "full") != "full" or
                    cfg.get("experiment_tag") != "round13_public_ten_method_matrix" or
                    cfg.get("seed") not in range(20, 23)):
                continue
        if float(cfg.get("attack_fraction", 0.0)) != 0.0:
            continue
        if cfg.get("dataset") not in FORMAL_DATASETS or cfg.get("regime") not in REGIMES:
            continue
        final = run["final"]
        diag = run["diagnostics"]
        rows.append(
            {
                "dataset": cfg["dataset"],
                "regime": cfg["regime"],
                "method": cfg["method"],
                "seed": cfg["seed"],
                "accuracy": final["accuracy"],
                "macro_f1": final["macro_f1"],
                "worst20_accuracy": final["worst20_accuracy"],
                "client_std": final["client_std"],
                "ece": final["ece"],
                "brier": final["brier"],
                "nll": final["nll"],
                "convergence_auc": final["convergence_auc"],
                "runtime_seconds": run["runtime_seconds"],
                "aggregation_ms": 1000 * np.mean([x["aggregation_seconds"] for x in diag]),
                "communication_mb": run["communication_mb"],
                "update_cosine": np.mean([x["mean_update_cosine"] for x in diag]),
                "weight_entropy": np.mean([x["weight_entropy"] for x in diag]),
                "path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    expected = len(FORMAL_DATASETS) * len(REGIMES) * len(PAPER_METHODS) * 3
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} formal runs, found {len(frame)}")
    return frame


def main() -> None:
    df = load_runs()
    metrics = [
        "accuracy",
        "macro_f1",
        "worst20_accuracy",
        "client_std",
        "ece",
        "brier",
        "convergence_auc",
        "runtime_seconds",
        "aggregation_ms",
        "communication_mb",
    ]
    grouped = df.groupby(["dataset", "regime", "method"])[metrics].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    grouped.reset_index().to_csv(TABLES / "main_results.csv", index=False, float_format="%.6f")

    means = df.groupby(["dataset", "regime", "method"])["accuracy"].mean().reset_index()
    means["rank"] = means.groupby(["dataset", "regime"])["accuracy"].rank(ascending=False, method="average")
    mean_ranks = means.groupby("method")["rank"].mean().sort_values().reset_index(name="mean_rank")
    wins = (
        means.assign(best=means.groupby(["dataset", "regime"])["accuracy"].transform("max"))
        .assign(win=lambda x: np.isclose(x["accuracy"], x["best"]))
        .groupby("method")["win"]
        .sum()
        .reset_index(name="task_wins")
    )
    rank_summary = mean_ranks.merge(wins, on="method")
    rank_summary.to_csv(TABLES / "rank_summary.csv", index=False, float_format="%.6f")

    # The nine settings share only three underlying datasets, and the three
    # seeds within each setting are paired replicates.  A 27-row Wilcoxon test
    # or flat bootstrap would pseudoreplicate the experimental units, so the
    # final audit deliberately emits no such inferential p-value table.

    summary = {
        "formal_runs": len(df),
        "mean_accuracy": df.groupby("method")["accuracy"].mean().sort_values(ascending=False).to_dict(),
        "mean_worst20": df.groupby("method")["worst20_accuracy"].mean().sort_values(ascending=False).to_dict(),
        "mean_ece": df.groupby("method")["ece"].mean().sort_values().to_dict(),
        "mean_auc": df.groupby("method")["convergence_auc"].mean().sort_values(ascending=False).to_dict(),
        "rank_summary": rank_summary.to_dict(orient="records"),
        "fedperg_task_accuracy": means[means.method == "FedPERG"].to_dict(orient="records"),
        "gate": {
            "accuracy_rank": int(df.groupby("method")["accuracy"].mean().rank(ascending=False)["FedPERG"]),
            "worst20_rank": int(df.groupby("method")["worst20_accuracy"].mean().rank(ascending=False)["FedPERG"]),
            "auc_rank": int(df.groupby("method")["convergence_auc"].mean().rank(ascending=False)["FedPERG"]),
            "ece_rank": int(df.groupby("method")["ece"].mean().rank(ascending=True)["FedPERG"]),
        },
    }
    (MODELS / "aggregate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
