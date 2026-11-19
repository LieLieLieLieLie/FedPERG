"""Select a field profile using development-pool metrics and record its provenance."""
import csv
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from experiments.optimization.study import CONTEXTS, PROFILES, ROBUST_PROFILES, TEMPORAL_PROFILES
from fedperg.paths import DEVELOPMENT, PROTOCOLS, prepare_output_directories

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('convergence_auc', 'accuracy', 'worst20_accuracy', 'ece')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', choices=['initial', 'robust', 'temporal', 'combined'], default='initial')
    args = parser.parse_args()
    prepare_output_directories()
    candidates = {'initial': PROFILES, 'robust': ROBUST_PROFILES, 'temporal': TEMPORAL_PROFILES,
                  'combined': {**ROBUST_PROFILES, **TEMPORAL_PROFILES}}[args.study]
    suffix = '' if args.study == 'initial' else '_'+args.study
    study_prefix = 'r1_opt'+suffix
    records = {}
    paths = {}
    # Shared profiles retain their earlier robust-study observation.
    prefixes = ['r1_opt_robust', 'r1_opt_temporal'] if args.study == 'combined' else [study_prefix]
    for study_prefix in prefixes:
        for path in sorted(DEVELOPMENT.rglob(f'*__{study_prefix}_development_*.json')):
            payload = json.loads(path.read_text(encoding='utf-8'))
            config = payload['config']
            for context in CONTEXTS:
                prefix = f'{study_prefix}_development_{context}_'
                if config['experiment_tag'].startswith(prefix):
                    profile = config['experiment_tag'][len(prefix):]
                    key = (context, config['seed'], profile)
                    if key in records:
                        assert args.study == 'combined'
                        continue
                    records[key] = payload['development_validation']
                    paths[key] = path
    expected = {(context, seed, profile) for context in CONTEXTS
                for seed in (200, 201, 202) for profile in candidates}
    missing = expected - set(records)
    if missing:
        raise RuntimeError(f'{len(missing)} development runs remain incomplete')
    profiles = list(candidates)
    ranks = {profile: [] for profile in profiles}
    for context in CONTEXTS:
        for seed in (200, 201, 202):
            for metric in METRICS:
                values = np.array([records[context, seed, profile][metric] for profile in profiles])
                metric_ranks = rankdata(values if metric == 'ece' else -values)
                for profile, rank in zip(profiles, metric_ranks):
                    ranks[profile].append(float(rank))
    ordered = sorted(profiles, key=lambda profile: (np.mean(ranks[profile]), profile))
    chosen = next(profile for profile in ordered if candidates[profile].get('method', 'FedPERG') == 'FedPERG')
    selection = {
        'profile': chosen,
        'configuration_overrides': candidates[chosen],
        'common_settings': {'utility_anchor_matches_bank': True,
                            'confidence_fraction': .04, 'clean_confidence_pool': args.study != 'initial'},
        'selection_metrics': list(METRICS),
        'validation_mean_rank': {profile: float(np.mean(ranks[profile])) for profile in ordered},
        'development_seeds': [200, 201, 202],
        'confirmation_seeds': list(range(220, 230)),
        'data_access': 'Only development_validation fields were read for model selection.',
        'development_records_sha256': {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
                                       for path in paths.values()},
    }
    target = PROTOCOLS / f'r1_optimization{suffix}_selection.json'
    if target.exists():
        prior = json.loads(target.read_text(encoding='utf-8'))
        prior_hashes = prior.pop('development_records_sha256')
        current = dict(selection)
        current_hashes = current.pop('development_records_sha256')
        assert prior == current
        assert sorted(prior_hashes.values()) == sorted(current_hashes.values())
    else:
        target.write_text(json.dumps(selection, indent=2), encoding='utf-8')
    rows = []
    for context in CONTEXTS:
        for profile in ordered:
            values = [records[context, seed, profile] for seed in (200, 201, 202)]
            rows.append([context, profile, selection['validation_mean_rank'][profile],
                         *[100 * np.mean([v[metric] for v in values]) for metric in METRICS]])
    with (PROTOCOLS / f'r1_optimization{suffix}_development.csv').open('w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows([
            ['Context', 'Profile', 'Mean validation rank', 'Validation AUC (%)', 'Validation accuracy (%)',
             'Validation tail (%)', 'Validation ECE (%)'], *rows,
        ])
    print('Selected from independent validation metrics:', chosen)
    print(json.dumps(selection['validation_mean_rank'], indent=2))


if __name__ == '__main__':
    main()
