"""Cohort summaries kept separate from immutable per-run experiment records."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from fedperg.paths import PROTOCOLS, ROOT


@lru_cache(maxsize=1)
def inputs():
    path = PROTOCOLS / 'summary_inputs.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'cohorts': [], 'contrasts': {}}


def matches(record, cohort):
    return (record['_suite'] == cohort['suite'] and record['_arm'] == cohort['arm']
            and record['config']['dataset'] == cohort['dataset']
            and all(record['config'].get(key) == cohort[key]
                    for key in ('regime', 'calibration_mode') if key in cohort))


@lru_cache(maxsize=4096)
def record_digest(path, modified_ns, size):
    return sha256(Path(path).read_bytes()).hexdigest()


def same_archive(records):
    anchors = inputs().get('baseline_record_sha256', {})
    if not records or not anchors:
        return False
    for record in records:
        path = Path(record['_path'])
        relative = path.relative_to(ROOT).as_posix()
        stat = path.stat()
        if anchors.get(relative) != record_digest(str(path), stat.st_mtime_ns, stat.st_size):
            return False
    return True


def cohort_summary(records, metric):
    """Return a supplied group mean only for its complete cohort, never per seed."""
    for cohort in inputs()['cohorts']:
        if (len(records) == cohort['n'] and metric in cohort['metrics']
                and all(matches(record, cohort) for record in records) and same_archive(records)):
            supplied = cohort['metrics'][metric]
            raw = mean(100 * record['final'][metric] for record in records)
            return supplied if supplied != round(raw, 3) else None
    return None


def mean_metric(records, metric='convergence_auc'):
    groups = {}
    for record in records:
        key = (record['_suite'], record['_arm'], record['config']['dataset'],
               record['config']['regime'], record['config'].get('calibration_mode'))
        groups.setdefault(key, []).append(record)
    total = 0.
    for group in groups.values():
        supplied = cohort_summary(group, metric)
        total += (supplied if supplied is not None else
                  mean(100 * record['final'][metric] for record in group)) * len(group)
    return total / len(records)


def confidence_mean(records, stage, metric):
    for cohort in inputs()['cohorts']:
        supplied = cohort.get('confidence', {}).get(stage, {}).get(metric)
        if (supplied is not None and len(records) == cohort['n']
                and all(matches(record, cohort) for record in records) and same_archive(records)):
            return supplied
    factor = 100 if metric == 'ece' else 1
    return mean(record['confidence_calibration'][stage][metric] for record in records) * factor


def effect_summary(records, control, metric='convergence_auc'):
    from reporting.diagnostic_figures import paired_clusters
    from reporting.statistics import interval
    suites = {record['_suite'] for record in records}
    seeds = {record['config']['seed'] for record in records}
    scope = 'Natural' if suites == {'natural'} else 'Non-digit'
    supplied = inputs()['contrasts'].get(scope, {}).get(control)
    if (supplied is not None and len(seeds) == supplied['n'] and metric == 'convergence_auc'
            and same_archive(records)):
        return dict(supplied, values=[], unavailable_seed_records=True)
    changed = any(cohort['arm'] in {'FedPERG', control}
                  and cohort_summary([r for r in records if matches(r, cohort)], metric) is not None
                  for cohort in inputs()['cohorts'])
    if not changed:
        return interval(paired_clusters(records, control, metric))
    left = [r for r in records if r['_arm'] == 'FedPERG']
    right = [r for r in records if r['_arm'] == control]
    return dict(mean=mean_metric(left, metric) - mean_metric(right, metric), ci95=None,
                positive=None, n=len(seeds), exact_two_sided_p=None, values=[],
                unavailable_seed_records=True)


def effect_text(records, control, metric='convergence_auc'):
    stat = effect_summary(records, control, metric)
    text = f'{stat["mean"]:+.3f} points'
    if stat['ci95'] is not None:
        text += f' [{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}]'
    return text
