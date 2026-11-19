"""Develop a common field configuration using an independent validation pool."""
import argparse
import json
from pathlib import Path

import torch

from fedperg.config import ExperimentConfig
from fedperg.trainer import FederatedExperiment
from fedperg.paths import DEVELOPMENT, PROTOCOLS, prepare_output_directories

ROOT = Path(__file__).resolve().parents[2]

PROFILES = {
    'reference': {},
    'larger_field': {'utility_strength': .3, 'utility_field_radius': .5},
    'wider_bank': {'utility_strength': .3, 'utility_field_radius': .5, 'gate_bank_scale': 1.5},
    'normalized_reference': {'update_reference': 'work_normalized'},
    'normalized_field': {'update_reference': 'work_normalized',
                         'utility_strength': .3, 'utility_field_radius': .5},
    'normalized_bank': {'update_reference': 'work_normalized',
                        'utility_strength': .3, 'utility_field_radius': .5, 'gate_bank_scale': 1.5},
    'proper_score': {'utility_strength': .3, 'utility_field_radius': .5,
                     'calibration_objective': 'brier_balance'},
    'normalized_score': {'update_reference': 'work_normalized', 'utility_strength': .3,
                         'utility_field_radius': .5, 'calibration_objective': 'brier_balance'},
}

ROBUST_PROFILES = {
    'reference': {},
    'proper_score': {'utility_strength': .3, 'utility_field_radius': .5,
                     'calibration_objective': 'brier_balance'},
    'brier_field': {'utility_strength': .3, 'utility_field_radius': .5,
                    'calibration_objective': 'brier'},
    'normalized_brier': {'utility_strength': .3, 'utility_field_radius': .5,
                         'calibration_objective': 'brier', 'update_reference': 'work_normalized'},
    'robust_field': {'utility_strength': .3, 'utility_field_radius': .5,
                     'calibration_objective': 'gce'},
    'normalized_robust': {'utility_strength': .3, 'utility_field_radius': .5,
                          'calibration_objective': 'gce', 'update_reference': 'work_normalized'},
    'balanced_field': {'utility_strength': .3, 'utility_field_radius': .5,
                       'calibration_objective': 'balanced_brier'},
    'normalized_balanced': {'utility_strength': .3, 'utility_field_radius': .5,
                            'calibration_objective': 'balanced_brier', 'update_reference': 'work_normalized'},
    'momentum_gate': {'method': 'FedAvgM-Gate'},
}

TEMPORAL_PROFILES = {
    'reference': {},
    'proper_score': {'utility_strength': .3, 'utility_field_radius': .5,
                     'calibration_objective': 'brier_balance'},
    **{f'{objective}_momentum_{round(10*step)}': {
        'utility_strength': .3, 'utility_field_radius': .5,
        'calibration_objective': objective, 'utility_momentum_step': step,
    } for objective in ('brier_balance', 'brier', 'gce') for step in (.3, .6)},
    'momentum_gate': {'method': 'FedAvgM-Gate'},
}

CONTEXTS = {
    'cifar10': {'dataset': 'cifar10'},
    'officehome': {'dataset': 'officehome'},
    'natural': {'dataset': 'officehome', 'clients': 4, 'clients_per_round': 4,
                'natural_domains': True, 'unequal_work': True, 'dropout_rate': .25,
                'domain_migration': True, 'domain_migration_round': 31, 'simulated_network': True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['initial', 'robust', 'temporal'], default='initial')
    parser.add_argument('--contexts', nargs='+', choices=tuple(CONTEXTS), default=list(CONTEXTS))
    parser.add_argument('--profiles', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[200, 201, 202])
    parser.add_argument('--device', default='cuda', choices=['cuda'])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('These experiments require the configured CUDA GPU.')
    prepare_output_directories()
    profiles = {'initial': PROFILES, 'robust': ROBUST_PROFILES, 'temporal': TEMPORAL_PROFILES}[args.study]
    chosen_profiles = args.profiles or list(profiles)
    if set(chosen_profiles) - profiles.keys():
        parser.error('Unknown development profile for this study.')
    if any(seed not in range(200, 210) for seed in args.seeds):
        parser.error('This development study uses seeds 200--209.')
    torch.set_num_threads(2)
    manifest = {
        'selection_data': 'Independent 4% training-index pool excluded from client SGD and both aggregation splits.',
        'profiles': profiles, 'contexts': CONTEXTS, 'development_seeds': [200, 201, 202],
        'selection_metrics': ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece'],
        'selection_rule': 'Mean validation rank across four metrics, contexts and seeds. Lower ECE is better.',
        'confirmation_seeds': list(range(220, 230)),
    }
    study_suffix = '' if args.study == 'initial' else '_'+args.study
    manifest_path = PROTOCOLS / f'r1_optimization{study_suffix}_protocol.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding='utf-8')) == manifest
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for context in args.contexts:
        for seed in args.seeds:
            for profile in chosen_profiles:
                prefix = 'r1_opt'+study_suffix
                tag = f'{prefix}_development_{context}_{profile}'
                settings = dict(
                    regime='compound', method='FedPERG', seed=seed, rounds=60,
                    device=args.device, perg_variant='lite_paired_gate_bank_selector',
                    utility_evidence=True, utility_gate_mode='field', utility_strength=.1,
                    utility_anchor_matches_bank=True, selector_calibration=True,
                    selector_margin=.001, confidence_fraction=.04,
                    record_development_validation=True, experiment_tag=tag,
                    clean_confidence_pool=args.study != 'initial',
                    model_group=f'development/{args.study}',
                )
                settings.update(CONTEXTS[context])
                settings.update(profiles[profile])
                config = ExperimentConfig(**settings).resolved(ROOT)
                method_name = config.method.lower().replace('-', '_')
                name = f'{config.dataset}__compound__{method_name}__s{seed}__{config.perg_variant}__{tag}.json'
                if (DEVELOPMENT / args.study / name).exists():
                    continue
                print('START', context, seed, profile, flush=True)
                result = FederatedExperiment(config).run(False)
                # Model selection consumes only the independent development pool.
                validation = result['development_validation']
                print('VALIDATION', {k: round(100 * validation[k], 4)
                                     for k in manifest['selection_metrics']}, flush=True)


if __name__ == '__main__':
    main()
