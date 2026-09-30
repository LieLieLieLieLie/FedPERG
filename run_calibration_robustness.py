"""Frozen server-calibration size and label-shift robustness study."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from fedcanto.config import ExperimentConfig
from fedcanto.trainer import FederatedExperiment

ROOT = Path(__file__).resolve().parent
TASKS = (("cifar10", "compound"), ("cifar100", "compound"),
         ("officehome", "compound"))
VARIANTS = (("Router-only", "lite_paired_gate_bank_router_control"),
            ("FedCANTO", "lite_paired_gate_bank_selector"))
CONDITIONS = ((.01, "none"), (.02, "none"), (.04, "none"), (.08, "none"),
              (.04, "low_label_bias"))
TAG = "round14_calibration_robustness"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    jobs = list(itertools.product(TASKS, VARIANTS, CONDITIONS, range(40, 45)))
    for index, ((dataset, regime), (name, variant), (fraction, shift), seed) in enumerate(jobs, 1):
        suffix = f"p{int(100*fraction):02d}_{shift}"
        cfg = ExperimentConfig(
            dataset=dataset, regime=regime, method="FedCANTO", seed=seed,
            rounds=30, canto_variant=variant, canto_meta_steps=1,
            canto_gate_low=.82, canto_gate_span=.32,
            calibration_fraction=fraction, selector_calibration=True,
            selector_calibration_shift=shift, selector_test_diagnostics=True,
            selector_margin=.001, experiment_tag=f"{TAG}_{suffix}",
            device=args.device,
        ).resolved(ROOT)
        stem = (f"{dataset}__{regime}__fedcanto__s{seed}__{variant}__"
                f"{TAG}_{suffix}.json")
        path = ROOT / "results" / "models" / stem
        if path.exists():
            print(f"[{index}/{len(jobs)}] SKIP {path.name}", flush=True)
            continue
        result = FederatedExperiment(cfg).run(verbose=False)
        print(f"[{index}/{len(jobs)}] {dataset}/{name}/{suffix}/s{seed} "
              f"AUC={100 * result['final']['convergence_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
