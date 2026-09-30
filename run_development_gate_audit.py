"""Run the frozen untouched-seed public-calibrated gate confirmation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedcanto.config import ExperimentConfig
from fedcanto.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
DATASETS = ("cifar10", "cifar100", "officehome")
REGIMES = ("label_skew", "quantity_skew", "compound")
METHODS = (("FedAvgM-Gate", "full"), ("FedCANTO", "lite_gate_bank_selector"))
TAG = "round13_frozen_public_gate_confirmation"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = [(d, r, m, v, s) for d, r, (m, v), s in
            itertools.product(DATASETS, REGIMES, METHODS, range(20, 30))]
    jobs += [("mnist_mobilenet", "compound", m, v, s)
             for (m, v), s in itertools.product(METHODS, range(20, 30))]
    for index, (dataset, regime, method, variant, seed) in enumerate(jobs, 1):
        cfg = ExperimentConfig(
            dataset=dataset, regime=regime, method=method, seed=seed,
            rounds=30, canto_variant=variant, canto_meta_steps=1,
            canto_gate_low=0.82, canto_gate_span=0.32,
            client_validation_fraction=0.0, selector_calibration=True,
            selector_margin=0.001, experiment_tag=TAG, device=args.device,
        ).resolved(ROOT)
        method_tag = method.lower().replace("-", "_")
        stem = f"{dataset}__{regime}__{method_tag}__s{seed}"
        if variant != "full": stem += f"__{variant}"
        stem += f"__{TAG}.json"
        path = ROOT / "results" / "models" / stem
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True); continue
        result = FederatedExperiment(cfg).run(verbose=False)
        selected = sum(d.get("selector_used_evidence", False) for d in result["diagnostics"])
        print(f"[{index}/{len(jobs)}] {dataset}/{regime}/{method}/s{seed} "
              f"AUC={100 * result['final']['convergence_auc']:.3f} "
              f"evidence={selected}/30", flush=True)


if __name__ == "__main__":
    main()
