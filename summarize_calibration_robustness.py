"""Summarize server-calibration size and label-shift robustness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "results" / "models"
TABLES = ROOT / "results" / "tables"
TASKS = (("cifar10", "compound"), ("cifar100", "compound"),
         ("officehome", "compound"))
CONDITIONS = ((.01, "none"), (.02, "none"), (.04, "none"), (.08, "none"),
              (.04, "low_label_bias"))
TAG = "round14_calibration_robustness"


def load(dataset: str, regime: str, variant: str, suffix: str, seed: int) -> dict:
    path = MODELS / (f"{dataset}__{regime}__fedperg__s{seed}__{variant}__"
                     f"{TAG}_{suffix}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for dataset, regime in TASKS:
        for fraction, shift in CONDITIONS:
            suffix = f"p{int(100*fraction):02d}_{shift}"
            for seed in range(40, 45):
                router = load(dataset, regime, "lite_paired_gate_bank_router_control", suffix, seed)
                perg = load(dataset, regime, "lite_paired_gate_bank_selector", suffix, seed)
                diagnostics = perg["diagnostics"]
                selected_evidence = []
                calibration_gains = []
                test_gains = []
                agreements = []
                for item in diagnostics:
                    losses = item["selector_expert_losses"]
                    tests = item["selector_expert_test_losses"]
                    chosen = item["selector_expert_index"]
                    router_best = min((0, 1, 3, 5), key=losses.__getitem__)
                    evidence_best = min((2, 4, 6), key=losses.__getitem__)
                    test_router = min((0, 1, 3, 5), key=tests.__getitem__)
                    test_evidence = min((2, 4, 6), key=tests.__getitem__)
                    use_evidence = chosen in (2, 4, 6)
                    test_prefers_evidence = tests[test_evidence] < tests[test_router]
                    selected_evidence.append(use_evidence)
                    calibration_gains.append(losses[router_best] - losses[chosen])
                    test_gains.append(tests[test_router] - tests[chosen])
                    agreements.append(use_evidence == test_prefers_evidence)
                rows.append({
                    "dataset": dataset, "fraction": fraction, "shift": shift, "seed": seed,
                    "router_auc_pct": 100 * router["final"]["convergence_auc"],
                    "fedperg_auc_pct": 100 * perg["final"]["convergence_auc"],
                    "perg_minus_router_pp": 100 * (perg["final"]["convergence_auc"] -
                                                      router["final"]["convergence_auc"]),
                    "evidence_selection_rate": mean(selected_evidence),
                    "mean_calibration_loss_gain": mean(calibration_gains),
                    "mean_test_loss_gain": mean(test_gains),
                    "calibration_test_decision_agreement": mean(agreements),
                })
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "round14_calibration_robustness_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for fraction, shift in CONDITIONS:
        cells = [x for x in rows if x["fraction"] == fraction and x["shift"] == shift]
        effects = [x["perg_minus_router_pp"] for x in cells]
        summary.append({
            "fraction": fraction, "shift": shift,
            "perg_minus_router_pp": mean(effects), "effect_sd_pp": stdev(effects),
            "positive_pairs": sum(x > 0 for x in effects),
            "evidence_selection_rate": mean(x["evidence_selection_rate"] for x in cells),
            "calibration_loss_gain": mean(x["mean_calibration_loss_gain"] for x in cells),
            "test_loss_gain": mean(x["mean_test_loss_gain"] for x in cells),
            "decision_agreement": mean(x["calibration_test_decision_agreement"] for x in cells),
        })
    with (TABLES / "round14_calibration_robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    report = {"status": "complete", "runs": len(rows) * 2,
              "diagnostic_test_labels_used_for_routing": False,
              "summary": summary}
    (MODELS / "round14_calibration_robustness_statistics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
