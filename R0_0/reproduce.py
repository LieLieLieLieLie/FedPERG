"""Run the frozen FedPERG experiment suites and regenerate reported artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = {
    "primary": (
        "run_primary_matrix.py",
        "run_primary_fedperg.py",
    ),
    "attribution": (
        "run_gate_reference.py",
        "run_attribution_confirmation.py",
        "run_residual_ablation.py",
        "summarize_attribution.py",
        "summarize_confirmatory_audit.py",
    ),
    "robustness": (
        "run_calibration_robustness.py",
        "run_raw_mnist_confirmation.py",
        "summarize_calibration_robustness.py",
        "summarize_raw_mnist.py",
    ),
    "development": (
        "run_development_gate_audit.py",
        "summarize_development_gate_audit.py",
    ),
    "report": (
        "audit_results.py",
        "summarize_primary_matrix.py",
        "summarize_primary_breadth.py",
        "plot_calibration_robustness.py",
        "analyze_and_plot.py",
    ),
}


def run_script(script: str, device: str) -> None:
    command = [sys.executable, str(ROOT / script)]
    text = (ROOT / script).read_text(encoding="utf-8")
    if 'add_argument("--device"' in text or "add_argument('--device'" in text:
        command.extend(["--device", device])
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(*STAGES, "all"),
        default="primary",
        help="Frozen experiment group to execute.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    selected = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for stage, scripts in selected.items():
        print(f"\n=== {stage.upper()} ===", flush=True)
        for script in scripts:
            run_script(script, args.device)


if __name__ == "__main__":
    main()
