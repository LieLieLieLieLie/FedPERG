"""Run the frozen selected configuration on independent confirmation seeds."""
import argparse
import json
import os
from hashlib import sha256
from itertools import product
from pathlib import Path

import torch

from experiments.optimization.study import CONTEXTS
from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment
from fedperg.paths import FORMAL, PROTOCOLS, prepare_output_directories
from experiments.optimization import METHODS, CONTROLS
from experiments.optimization.locking import record_lock

ROOT = Path(__file__).resolve().parents[2]
VARIANT = 'lite_paired_gate_bank_selector'
FEATURE_CONTEXTS = {
    'cifar10': {'dataset': 'cifar10'}, 'cifar100': {'dataset': 'cifar100'},
    'officehome': {'dataset': 'officehome'},
}


def scenarios(suite, selected, regimes=None):
    if suite == 'matrix':
        for context, regime in product(FEATURE_CONTEXTS, ['label_skew', 'quantity_skew', 'compound']):
            yield context, regime, {**FEATURE_CONTEXTS[context], 'regime': regime}, METHODS
    elif suite == 'natural':
        yield 'natural', 'migration31', CONTEXTS['natural'], list(dict.fromkeys([*METHODS, *CONTROLS]))
    elif suite == 'attribution':
        for context, regime in product(FEATURE_CONTEXTS, regimes or ['compound']):
            yield context, regime, {**FEATURE_CONTEXTS[context], 'regime': regime}, CONTROLS
    elif suite == 'stress':
        for context in ['cifar10', 'officehome']:
            for name, k, dropout, power in [('small', 2, 0., 1.), ('medium', 4, 0., 1.),
                                           ('large', 16, 0., 1.), ('dropout', 8, .5, 1.),
                                           ('imbalance', 8, 0., 3.)]:
                yield context, name, {**FEATURE_CONTEXTS[context], 'clients_per_round': k,
                                      'dropout_rate': dropout, 'client_imbalance_power': power}, ['FedPERG', 'Router']
    elif suite == 'quality':
        for context, rate, reward in product(['cifar10', 'officehome'], [0., .18, .36], ['raw', 'none']):
            yield context, f'fault{round(100*rate):02d}_{reward}', {
                **FEATURE_CONTEXTS[context], 'compound_label_fault_rate': rate,
                'reward_mode': reward, 'unequal_work': True, 'class_difficulty': .5}, (
                    ['FedPERG', 'Router', 'TemporalOnly'] if reward == 'raw' else ['FedPERG'])
    elif suite == 'reward':
        for context in ['cifar10', 'officehome']:
            for work, difficulty in [(False, 0.), (True, 0.), (True, .5)]:
                for mode in ['raw', 'none', 'per_step', 'relative']:
                    yield context, f'work{int(work)}_difficulty{difficulty}_{mode}'.replace('.', 'p'), {
                        **FEATURE_CONTEXTS[context], 'unequal_work': work,
                        'class_difficulty': difficulty, 'reward_mode': mode,
                        'reward_diagnostics': True}, ['FedPERG']
    elif suite in ['calibration', 'shift']:
        modes = ['centralized', 'distributed', 'unlabeled'] if suite == 'calibration' else ['centralized']
        for mode in modes:
            changes = {'dataset': 'officehome', 'calibration_mode': mode}
            if suite == 'shift':
                changes['selector_calibration_shift'] = 'low_label_bias'
            yield 'officehome', mode, changes, ['FedPERG', 'Router', 'Permuted']
    elif suite == 'sensitivity':
        base = {'utility_strength': .1, 'utility_field_radius': .2, 'utility_prior_strength': .25,
                'utility_pool_fraction': .5, 'selector_margin': .001, 'gate_bank_scale': 1.,
                'utility_momentum_step': 0., 'perg_reconstruction_weight': .85,
                'perg_median_weight': 1.2, 'perg_temperature': .8,
                'improvement_weight': .35, 'gate_bank_offset': 0.}
        base.update(selected)
        for context in ['cifar10', 'officehome']:
            for key in ['utility_strength', 'utility_field_radius', 'utility_prior_strength',
                        'utility_pool_fraction', 'selector_margin', 'gate_bank_scale',
                        'utility_momentum_step', 'perg_reconstruction_weight',
                        'perg_median_weight', 'perg_temperature', 'improvement_weight',
                        'gate_bank_offset']:
                for factor in [.5, 1.5]:
                    value = (-.08 if factor == .5 else .08) if key == 'gate_bank_offset' else base[key] * factor
                    yield context, f'{key}_{factor}'.replace('.', 'p'), {
                        **FEATURE_CONTEXTS[context], key: value}, ['FedPERG']
    elif suite == 'deep':
        yield 'officehome_raw', 'resnet18', {
            'dataset': 'officehome_raw', 'rounds': 100, 'architecture': 'raw_rgb_resnet18',
            'batch_size': 64, 'local_lr': .02, 'max_local_batches': 100000, 'eval_every': 10,
        }, ['FedPERG', 'Router', 'RandomInput', 'FedAvgM-Gate']
    elif suite == 'cnn':
        yield 'officehome_raw', 'cnn', {
            'dataset': 'officehome_raw', 'rounds': 100, 'architecture': 'raw_rgb_cnn',
            'batch_size': 32, 'local_lr': .02, 'max_local_batches': 3,
        }, ['FedPERG', 'Router']
    elif suite == 'timing':
        yield 'officehome', 'isolated', {'dataset': 'officehome'}, [
            'FedPERG', 'Router', 'SpatialOnly', 'TemporalOnly', 'FedAvgM-Gate']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=[
        'matrix', 'natural', 'attribution', 'stress', 'quality', 'reward', 'calibration',
        'shift', 'sensitivity', 'deep', 'cnn', 'timing'])
    parser.add_argument('--study', choices=['robust', 'temporal', 'combined'], default='combined')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--contexts', nargs='+')
    parser.add_argument('--regimes', nargs='+', choices=['label_skew', 'quantity_skew', 'compound'])
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('Formal experiments require the configured CUDA GPU.')
    prepare_output_directories()
    seeds = args.seeds or list(range(220, 230 if args.suite in ['natural', 'attribution'] else
                                    223 if args.suite == 'matrix' else 225))
    if any(seed not in range(220, 230) for seed in seeds):
        parser.error('Confirmation uses seeds 220--229.')
    selection_path = PROTOCOLS / f'r1_optimization_{args.study}_selection.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    selected = selection['configuration_overrides']
    plan = {
        'selection_sha256': sha256(selection_path.read_bytes()).hexdigest(),
        'configuration_overrides': selected, 'common_settings': selection['common_settings'],
        'matrix_seeds': [220, 221, 222], 'natural_attribution_seeds': list(range(220, 230)),
        'other_seeds': list(range(220, 225)),
        'reported_metrics': ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece',
                             'nll', 'macro_f1', 'communication_mb', 'runtime_seconds'],
        'source_sha256': {name: sha256((ROOT / 'fedperg' / name).read_bytes()).hexdigest()
                          for name in ['aggregators.py', 'config.py', 'data.py', 'trainer.py',
                                       'evidence_fit.py', 'update_reference.py']},
    }
    plan['device'] = 'cuda'
    plan['model_storage'] = 'results/models/formal/<suite>'
    frozen = PROTOCOLS / f'r1_opt_confirmation_{args.study}_gpu_protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8')) == plan, 'Frozen source or protocol changed'
    else:
        temporary = frozen.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(plan, indent=2), encoding='utf-8')
        temporary.replace(frozen)
    torch.set_num_threads(2)
    for seed in seeds:
        for context, scenario, changes, methods in scenarios(args.suite, selected, args.regimes):
            if args.contexts and context not in args.contexts:
                continue
            for label in methods:
                if args.methods and label not in args.methods:
                    continue
                method = 'FedPERG' if label in CONTROLS else label
                variant = ('lite_paired_gate_bank_router_control' if label == 'Router' else
                           'lite_paired_gate_bank_no_residual' if label == 'NoResidual' else VARIANT)
                tag = f'r1_opt_confirm_{args.study}_{args.suite}_{context}_{scenario}_{label}'.lower().replace('-', '_')
                settings = dict(regime='compound', method=method, seed=seed, rounds=60,
                                device=args.device, perg_variant=variant, selector_calibration=True,
                                selector_margin=.001, utility_evidence=method == 'FedPERG',
                                utility_gate_mode='field', utility_strength=.1, experiment_tag=tag,
                                model_group=f'formal/{args.suite}',
                                evidence_control='random_input' if label == 'RandomInput' else
                                                 'permuted' if label == 'Permuted' else 'informative')
                settings.update(selection['common_settings'])
                settings.update(selected)
                settings.update(changes)
                if label == 'SpatialOnly':
                    settings['utility_momentum_step'] = 0.
                elif label == 'TemporalOnly':
                    settings['utility_strength'] = 0.
                    settings['utility_prior_strength'] = 0.
                config = ExperimentConfig(**settings).resolved(ROOT)
                name = (f'{config.dataset}__{config.regime}__{method.lower().replace("-", "_")}__s{seed}'
                        f'__{variant}__{tag}.json')
                if (FORMAL / args.suite / name).is_file():
                    continue
                with record_lock(name):
                    if (FORMAL / args.suite / name).is_file():continue
                    print('START', args.suite, context, scenario, label, seed, flush=True)
                    result = FederatedExperiment(config).run(args.suite == 'deep')
                    print('COMPLETE', label, seed, {metric: round(100 * result['final'][metric], 4)
                                                  for metric in ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']},
                          flush=True)


if __name__ == '__main__':
    main()
