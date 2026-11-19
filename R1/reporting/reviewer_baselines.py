"""Validate paired GPU records and summarize the additional recent baselines."""
import csv
import json
from hashlib import sha256
from statistics import mean

from experiments.optimization.reviewer_baselines import GROUP, METHODS, reference_runs
from fedperg.paths import MODELS, PROTOCOLS, RESULTS, ROOT
from fedperg.trainer import _safe_name
from reporting.statistics import interval


def load_extension():
    protocol = json.loads((PROTOCOLS / 'r1_reviewer_baselines_gpu.json').read_text(encoding='utf-8'))
    for name, digest in protocol['source'].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    references = reference_runs()
    expected_files = set()
    records = []
    hashes = {}
    for (context, regime, seed), original in references.items():
        reference_bytes = original.read_bytes()
        assert sha256(reference_bytes).hexdigest() == protocol['references'][original.name]
        reference = json.loads(reference_bytes)
        for method in METHODS:
            variant = reference['config']['perg_variant']
            stem = (f'{context}__{regime}__{_safe_name(method)}__s{seed}__{variant}'
                    f'__r1_reviewer_{context}_{regime}')
            path = MODELS / GROUP / f'{stem}.json'
            checkpoint = path.with_suffix('.pth')
            expected_files.add(path)
            new = json.loads(path.read_text(encoding='utf-8'))
            assert new['status'] == 'complete' and new['device'].startswith('cuda')
            assert new['peak_gpu_memory_mb'] > 0 and checkpoint.stat().st_size > 1000
            for key, value in reference['config'].items():
                if key not in {'method', 'utility_evidence', 'experiment_tag', 'model_group', 'data_dir', 'results_dir'}:
                    assert new['config'][key] == value, (path, key)
            assert new['config']['method'] == method and new['parameter_count'] == reference['parameter_count']
            assert [x['round'] for x in new['history']] == [x['round'] for x in reference['history']]
            assert new['history'][0]['accuracy'] == reference['history'][0]['accuracy']
            assert len(new['diagnostics']) == len(reference['diagnostics']) == 60
            for new_round, old_round in zip(new['diagnostics'], reference['diagnostics']):
                for key in ('selected_clients', 'client_sizes', 'local_steps'):
                    assert new_round[key] == old_round[key], (path, key)
                clients = len(new_round['selected_clients'])
                if method == 'PA3Fed':
                    weights = [new_round['pa3fed_client_weights']]
                    expected_bytes = 8 * clients
                else:
                    weights = new_round['layerwise_client_weights']
                    assert len(weights) == 2, path
                    expected_bytes = 0
                assert all(len(w) == clients and abs(sum(w) - 1) < 1e-5 for w in weights), path
                assert new_round['reviewer_baseline_extra_upload_bytes'] == expected_bytes, path
            records.append({**new, '_suite': GROUP, '_arm': method})
            for artifact in (path, checkpoint):
                hashes[artifact.name] = sha256(artifact.read_bytes()).hexdigest()
    assert expected_files == set((MODELS / GROUP).glob('*.json'))
    assert not list((MODELS / GROUP).glob('*.tmp'))
    (PROTOCOLS / 'reviewer_baselines_manifest.json').write_text(json.dumps({
        'device': 'cuda', 'matched_runs': len(records), 'sha256': hashes,
    }, indent=2), encoding='utf-8')
    return records


def summarize(extension, matrix):
    methods = ('FedPERG', 'PA3Fed', 'Fed-LAA', 'FedLBW')
    metrics = ('convergence_auc', 'accuracy', 'worst20_accuracy', 'ece')
    records = [run for run in [*extension, *matrix] if run['_arm'] in methods]
    results = {(cfg['dataset'], cfg['regime'], cfg['seed'], run['_arm']): run['final']
               for run in records for cfg in [run['config']]}
    design = reference_runs()
    assert len(results) == len(records) == len(design) * len(methods)
    csv_path = RESULTS / 'tables' / 'reviewer_baselines_paired_runs.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('dataset', 'regime', 'seed', 'method', *(m + '_percent' for m in metrics)))
        for context, regime, seed in design:
            for method in methods:
                values = results[context, regime, seed, method]
                writer.writerow((context, regime, seed, method, *(100 * values[m] for m in metrics)))

    settings = tuple(dict.fromkeys((context, regime) for context, regime, _ in design))
    seeds = tuple(dict.fromkeys(seed for _, _, seed in design))
    estimates = {
        method: {setting: mean(100 * results[(*setting, seed, method)]['convergence_auc'] for seed in seeds)
                 for setting in settings} for method in methods
    }
    pooled = {method: mean(estimates[method].values()) for method in methods}
    differences = {}
    for method in methods[1:]:
        differences[method] = {}
        for metric in metrics:
            pairs = {setting: [100 * (results[(*setting, seed, 'FedPERG')][metric]
                                      - results[(*setting, seed, method)][metric]) for seed in seeds]
                     for setting in settings}
            difference = interval([mean(values[i] for values in pairs.values()) for i in range(len(seeds))])
            difference['settings_positive'] = sum(mean(values) > 0 for values in pairs.values())
            differences[method][metric] = difference
    summary = {'methods': list(methods), 'pooled_auc': pooled, 'effects': differences,
               'runs': len(extension), 'settings': len(settings), 'seeds': list(seeds)}
    (PROTOCOLS / 'reviewer_baselines_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return estimates, pooled
