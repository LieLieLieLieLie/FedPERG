"""Reproduce the GPU-only FedPERG optimization and formal experiments."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    'development': (
        ('experiments.optimization.study', '--study', 'initial'),
        ('experiments.optimization.study', '--study', 'robust'),
        ('experiments.optimization.study', '--study', 'temporal'),
        ('reporting.optimization', '--study', 'combined'),
    ),
    'experiments': (('experiments.optimization.queue', '--group', 'all'),
                    ('experiments.optimization.reviewer_baselines',)),
    'report': (('reporting.optimization','--study','combined'),
               ('reporting.optimization_results',), ('checks.verify_gpu',),
               ('reporting.make_manuscript',)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='report')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selected = list(STAGES) if args.stage == 'all' else [args.stage]
    for stage in selected:
        for module, *options in STAGES[stage]:
            command = [sys.executable, '-u', '-m', module, *options]
            if module == 'experiments.optimization.study':
                command += ['--device', args.device]
            print(subprocess.list2cmdline(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
