"""Matched three-seed ten-method matrix for the public-calibrated final rule."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
DATASETS = ("cifar10", "cifar100", "officehome")
REGIMES = ("label_skew", "quantity_skew", "compound")
METHODS = ("FedAvg", "FedAvgM", "SCAFFOLD", "FedAdam", "FedLAW",
           "FedCDA", "FedPW", "Fed-NGA", "FedPhoenix")
TAG = "round13_public_ten_method_matrix"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = list(itertools.product(DATASETS, REGIMES, METHODS, range(20, 23)))
    for index, (dataset, regime, method, seed) in enumerate(jobs, 1):
        cfg = ExperimentConfig(
            dataset=dataset, regime=regime, method=method, seed=seed, rounds=30,
            selector_calibration=True, calibration_fraction=0.04,
            experiment_tag=TAG, device=args.device,
        ).resolved(ROOT)
        stem = f"{dataset}__{regime}__{method.lower().replace('-', '_')}__s{seed}__{TAG}.json"
        path = ROOT / "results" / "models" / stem
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True); continue
        result = FederatedExperiment(cfg).run(verbose=False)
        print(f"[{index}/{len(jobs)}] {dataset}/{regime}/{method}/s{seed} "
              f"AUC={100 * result['final']['convergence_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
