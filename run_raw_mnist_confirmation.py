"""End-to-end raw-MNIST final paired-bank confirmation."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
METHODS = (("AvgM+Gate", "FedAvgM-Gate", "full"),
           ("Router-only", "FedPERG", "lite_paired_gate_bank_router_control"),
           ("FedPERG", "FedPERG", "lite_paired_gate_bank_selector"))
TAG = "round14_raw_final_paired"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = list(itertools.product(METHODS, range(60, 65)))
    for index, ((name, method, variant), seed) in enumerate(jobs, 1):
        cfg = ExperimentConfig(
            dataset="mnist", regime="label_skew", method=method, seed=seed,
            rounds=100, architecture="raw_cnn", local_lr=.025, batch_size=128,
            perg_variant=variant, perg_meta_steps=1,
            perg_gate_low=.82, perg_gate_span=.32,
            selector_calibration=True, selector_margin=.001,
            experiment_tag=TAG, device=args.device,
        ).resolved(ROOT)
        method_tag = method.lower().replace("-", "_")
        stem = f"mnist__label_skew__{method_tag}__s{seed}"
        if variant != "full": stem += f"__{variant}"
        path = ROOT / "results" / "models" / f"{stem}__{TAG}.json"
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True)
            continue
        result = FederatedExperiment(cfg).run(verbose=False)
        print(f"[{index}/{len(jobs)}] {name}/s{seed} AUC="
              f"{100 * result['final']['convergence_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
