"""Update only the FedPERG row/curves in the descriptive ten-method breadth matrix."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
TASKS = [(d, r) for d in ("cifar10", "cifar100", "officehome")
         for r in ("label_skew", "quantity_skew", "compound")]
TAG = "round14_final_breadth_update"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = [(d, r, s) for (d, r), s in itertools.product(TASKS, range(20, 23))]
    for index, (dataset, regime, seed) in enumerate(jobs, 1):
        cfg = ExperimentConfig(
            dataset=dataset, regime=regime, method="FedPERG", seed=seed,
            rounds=30, perg_variant="lite_paired_gate_bank_selector",
            perg_meta_steps=1, perg_gate_low=.82, perg_gate_span=.32,
            selector_calibration=True, selector_margin=.001,
            experiment_tag=TAG, device=args.device,
        ).resolved(ROOT)
        path = ROOT / "results" / "models" / (
            f"{dataset}__{regime}__fedperg__s{seed}__lite_paired_gate_bank_selector__{TAG}.json")
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True)
            continue
        result = FederatedExperiment(cfg).run(verbose=False)
        print(f"[{index}/{len(jobs)}] {dataset}/{regime}/s{seed} "
              f"AUC={100 * result['final']['convergence_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
