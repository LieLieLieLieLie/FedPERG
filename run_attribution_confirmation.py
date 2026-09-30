"""Frozen final paired-bank attribution confirmation on untouched seeds."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedcanto.config import ExperimentConfig
from fedcanto.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
TASKS = [(d, r) for d in ("cifar10", "cifar100", "officehome")
         for r in ("label_skew", "quantity_skew", "compound")]
TASKS.append(("mnist_mobilenet", "compound"))
METHODS = (("Router-only", "lite_paired_gate_bank_router_control"),
           ("FedCANTO", "lite_paired_gate_bank_selector"))
TAG = "round14_final_paired_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = [(d, r, name, variant, seed)
            for (d, r), (name, variant), seed in
            itertools.product(TASKS, METHODS, range(50, 60))]
    for index, (dataset, regime, name, variant, seed) in enumerate(jobs, 1):
        cfg = ExperimentConfig(
            dataset=dataset, regime=regime, method="FedCANTO", seed=seed,
            rounds=30, canto_variant=variant, canto_meta_steps=1,
            canto_gate_low=.82, canto_gate_span=.32,
            selector_calibration=True, selector_margin=.001,
            experiment_tag=TAG, device=args.device,
        ).resolved(ROOT)
        path = ROOT / "results" / "models" / (
            f"{dataset}__{regime}__fedcanto__s{seed}__{variant}__{TAG}.json")
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True)
            continue
        result = FederatedExperiment(cfg).run(verbose=False)
        evidence = sum(d.get("selector_used_target_evidence", False)
                       for d in result["diagnostics"])
        print(f"[{index}/{len(jobs)}] {dataset}/{regime}/{name}/s{seed} "
              f"AUC={100 * result['final']['convergence_auc']:.3f} "
              f"evidence={evidence}/30", flush=True)


if __name__ == "__main__":
    main()
