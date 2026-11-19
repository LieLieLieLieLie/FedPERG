"""Summarize frozen confirmation runs without changing their selected configuration."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import rankdata

from reporting.statistics import interval
from experiments.optimization import METHODS, CONTROLS
from fedperg.paths import FORMAL, PROTOCOLS, prepare_output_directories

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']


def arm(record):
    tag = record['config']['experiment_tag']
    for label in sorted(set(METHODS + CONTROLS), key=len, reverse=True):
        if tag.endswith('_' + label.lower().replace('-', '_')):
            return label
    raise ValueError(tag)


def load(study='combined'):
    records = []
    for path in sorted(FORMAL.rglob(f'*__r1_opt_confirm_{study}_*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') == 'complete':
            body = record['config']['experiment_tag'].split(f'r1_opt_confirm_{study}_', 1)[1]
            record['_suite'] = body.split('_', 1)[0]
            record['_arm'] = arm(record)
            record['_path'] = str(path)
            records.append(record)
    return records


def paired(items, control, metric):
    keys = {}
    for record in items:
        cfg = record['config']
        key = (cfg['dataset'], cfg['regime'], cfg['seed'], record['_arm'])
        assert key not in keys
        keys[key] = 100 * record['final'][metric]
    seeds = sorted({r['config']['seed'] for r in items})
    sources = sorted({(r['config']['dataset'], r['config']['regime']) for r in items})
    effects = []
    for seed in seeds:
        if all((d, g, seed, a) in keys for d, g in sources for a in ['FedPERG', control]):
            effects.append(mean(keys[d, g, seed, 'FedPERG'] - keys[d, g, seed, control]
                                for d, g in sources))
    return interval(effects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='combined')
    args = parser.parse_args()
    prepare_output_directories()
    records = load(args.study)
    report = {'counts': {}, 'effects': {}, 'ranks': {}, 'method_summaries': {}}
    for suite in sorted({r['_suite'] for r in records}):
        items = [r for r in records if r['_suite'] == suite]
        report['counts'][suite] = len(items)
        if suite == 'attribution':
            items = [r for r in items if r['config']['regime'] == 'compound']
        if suite not in ['matrix', 'natural', 'attribution', 'deep']:
            continue
        rows = []
        for label in dict.fromkeys([*METHODS, *CONTROLS]):
            group = [r for r in items if r['_arm'] == label]
            if not group:
                continue
            rows.append([label, len(group), *[100 * mean(r['final'][metric] for r in group) for metric in METRICS],
                         100 * mean(r['confidence_calibration']['temperature_scaled']['ece'] for r in group)])
        report['method_summaries'][suite] = {
            row[0]:dict(zip(['runs','auc','final','tail','raw_ece','scaled_ece'],row[1:])) for row in rows}
        if any(r['_arm'] == 'FedPERG' for r in items):
            report['effects'][suite] = {
                label: {metric: paired(items, label, metric) for metric in METRICS}
                for label in dict.fromkeys([*METHODS, *CONTROLS]) if label != 'FedPERG'
            }
        if suite in ['matrix', 'natural']:
            indexed = {(r['config']['dataset'], r['config']['regime'], r['config']['seed'], r['_arm']): r
                       for r in items}
            settings = sorted({key[:3] for key in indexed})
            complete = [setting for setting in settings if all((*setting, m) in indexed for m in METHODS)]
            ranking = defaultdict(list)
            for setting in complete:
                for metric in METRICS:
                    values = np.array([indexed[(*setting, m)]['final'][metric] for m in METHODS])
                    ranks = rankdata(values if metric == 'ece' else -values)
                    for method, rank in zip(METHODS, ranks):
                        ranking[method].append(float(rank))
            report['ranks'][suite] = {
                'complete_settings': len(complete),
                'mean_metric_rank': {m: mean(ranking[m]) for m in sorted(ranking, key=lambda m: mean(ranking[m]))},
            }
    target = PROTOCOLS / f'r1_optimization_{args.study}_confirmation_summary.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Completed confirmation records:', report['counts'])
    print(json.dumps(report['ranks'], indent=2))


if __name__ == '__main__':
    main()
