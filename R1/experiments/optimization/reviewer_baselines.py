"""Extend the nine-setting GPU matrix with PA3Fed and Fed-LAA."""
import argparse
import json
from dataclasses import replace
from hashlib import sha256

import torch

from experiments.optimization.confirm import FEATURE_CONTEXTS, scenarios
from experiments.optimization.locking import record_lock
from fedperg.config import ExperimentConfig
from fedperg.paths import FORMAL, MODELS, PROTOCOLS, ROOT, prepare_output_directories
from fedperg.reviewer_baselines import ReviewerBaselineExperiment
from fedperg.trainer import _safe_name


METHODS = ('PA3Fed', 'Fed-LAA')
SEEDS = (220, 221, 222)
GROUP = 'reviewer_baselines'


def reference_runs():
    runs = {}
    for context, regime, _, _ in scenarios('matrix', {}, None):
        for seed in SEEDS:
            stem = (f'{context}__{regime}__fedperg__s{seed}__lite_paired_gate_bank_selector'
                    f'__r1_opt_confirm_combined_matrix_{context}_{regime}_fedperg')
            path = FORMAL / 'matrix' / f'{stem}.json'
            runs[context, regime, seed] = path
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--contexts', nargs='+', choices=tuple(FEATURE_CONTEXTS), default=list(FEATURE_CONTEXTS))
    parser.add_argument('--seeds', nargs='+', choices=SEEDS, type=int, default=SEEDS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('The matched reviewer comparison requires CUDA.')
    torch.set_num_threads(2)
    prepare_output_directories()
    originals = reference_runs()
    plan = {
        'methods': list(METHODS), 'seeds': list(SEEDS), 'device': 'cuda',
        'references': {path.name: sha256(path.read_bytes()).hexdigest() for path in originals.values()},
        'source': {name: sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in ['fedperg/reviewer_baselines.py', 'fedperg/trainer.py',
                                'experiments/optimization/reviewer_baselines.py']},
    }
    protocol = PROTOCOLS / 'r1_reviewer_baselines_gpu.json'
    with record_lock('reviewer_baselines_protocol'):
        if protocol.exists():
            if json.loads(protocol.read_text(encoding='utf-8')) != plan:
                raise RuntimeError('Reviewer baseline source or matched protocol changed.')
        else:
            protocol.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for (context, regime, seed), original in originals.items():
        if context not in args.contexts or seed not in args.seeds:
            continue
        old = json.loads(original.read_text(encoding='utf-8'))
        reference = ExperimentConfig(**old['config'])
        for method in args.methods:
            tag = f'r1_reviewer_{context}_{regime}'
            config = replace(reference, method=method, experiment_tag=tag,
                             utility_evidence=False, model_group=GROUP, device='cuda',
                             data_dir=ExperimentConfig().data_dir, results_dir=ExperimentConfig().results_dir)
            config = config.resolved(ROOT)
            stem = (f'{context}__{regime}__{_safe_name(method)}__s{seed}'
                    f'__{config.perg_variant}__{tag}')
            target = MODELS / GROUP / f'{stem}.json'
            with record_lock(stem):
                if target.is_file():
                    continue
                print('START', method, context, regime, seed, flush=True)
                result = ReviewerBaselineExperiment(config).run(False)
                print('COMPLETE', method, context, regime, seed,
                      round(100 * result['final']['convergence_auc'], 3), flush=True)


if __name__ == '__main__':
    main()
