"""Matched mechanism, calibration and resource panels from GPU run records."""
from statistics import mean
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import torch

from fedperg.config import ExperimentConfig
from fedperg.metrics import predict
from fedperg.paths import FORMAL, PROTOCOLS
from fedperg.trainer import FederatedExperiment
from reporting.figures import decorate, method_name, save, value
from reporting.statistics import interval
from reporting.summary_inputs import cohort_summary, effect_summary, mean_metric

CONTROL_COLORS = {
    'FedPERG': '#FF6666', 'Router': '#FFAA53', 'SpatialOnly': '#50CC55',
    'TemporalOnly': '#3399FF', 'RandomInput': '#6666FF', 'Permuted': '#9933FF',
    'NoResidual': '#00DDDD', 'FedAvgM-Gate': '#D65DB1',
}
SOURCE_ORDER = ['cifar10', 'cifar100', 'officehome', 'natural']
SOURCE_NAMES = ['CIFAR-10', 'CIFAR-100', 'Office-Home', 'Natural\ndomains']
ARMS = ['FedPERG', 'Router', 'SpatialOnly', 'TemporalOnly', 'RandomInput', 'Permuted', 'NoResidual']


def source(record):
    return 'natural' if record['_suite'] == 'natural' else record['config']['dataset']


def primary_records(records):
    return [r for r in records if r['_suite'] in ['attribution', 'natural']
            and r['config']['regime'] == 'compound' and r['_arm'] in ARMS]


def paired_clusters(records, control, metric='convergence_auc'):
    index = {(source(r), r['config']['seed'], r['_arm']): r for r in records}
    seeds = sorted({r['config']['seed'] for r in records})
    sources = sorted({source(r) for r in records})
    effects = []
    for seed in seeds:
        if all((dataset, seed, arm) in index for dataset in sources for arm in ['FedPERG', control]):
            effects.append(mean(value(index[dataset, seed, 'FedPERG'], metric)
                                - value(index[dataset, seed, control], metric) for dataset in sources))
    return effects


def place_legend_below_axes(fig, handles):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
    gap = 0.06 * fig.dpi
    anchor_y = fig.transFigure.inverted().transform((0, bottom - gap))[1]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, anchor_y),
               ncol=4, frameon=False, borderaxespad=0)


def compact_method_ticks(ax, arms, rotation):
    ax.set_xticks(range(len(arms)), [method_name(arm) for arm in arms],
                  rotation=rotation, ha='right', rotation_mode='anchor')


def attribution(records):
    selected = primary_records(records)
    index = {(source(r), r['config']['seed'], r['_arm']): r for r in selected}
    seeds = list(range(220, 230))
    assert all((dataset, seed, arm) in index for dataset in SOURCE_ORDER for seed in seeds for arm in ARMS)
    fig, axes = plt.subplots(2, 3, figsize=(15.7, 10.2))
    for panel, metric, title in [(0, 'convergence_auc', 'Convergence by source'),
                                (1, 'worst20_accuracy', 'Tail-client distributions')]:
        ax = axes.flat[panel]
        for j, arm in enumerate(ARMS):
            positions = np.arange(4) + (j - 3) * .105
            groups = [[value(index[dataset, seed, arm], metric) for seed in seeds] for dataset in SOURCE_ORDER]
            if panel == 0:
                for position, dataset, group in zip(positions, SOURCE_ORDER, groups):
                    runs = [index[dataset, seed, arm] for seed in seeds]
                    mu = mean_metric(runs, metric)
                    errors = None
                    if cohort_summary(runs, metric) is None:
                        low, high = interval(group)['ci95']
                        errors = [[mu - low], [high - mu]]
                    ax.errorbar(position, mu, yerr=errors, fmt='o', ms=4,
                                capsize=2, color=CONTROL_COLORS[arm])
            else:
                for position, dataset, group in zip(positions, SOURCE_ORDER, groups):
                    runs = [index[dataset, seed, arm] for seed in seeds]
                    if cohort_summary(runs, metric) is None:
                        violin = ax.violinplot([group], positions=[position], widths=.10,
                                              showextrema=False, showmedians=False)
                        violin['bodies'][0].set(facecolor=CONTROL_COLORS[arm], alpha=.55)
                    ax.plot(position, mean_metric(runs, metric), 'o', ms=4,
                            color=CONTROL_COLORS[arm])
        ax.set_xticks(range(4), SOURCE_NAMES, rotation=28, ha='right')
        ax.set_ylabel('AUC (%) ↑' if panel == 0 else 'Tail accuracy (%) ↑')
        decorate(ax, panel, title)
    primary = [r for r in selected if source(r) != 'natural']
    ax = axes.flat[2]
    for arm in ARMS[1:-1]:
        effects = np.sort([
            mean_metric([index[dataset, seed, 'FedPERG'] for seed in seeds])
            - mean_metric([index[dataset, seed, arm] for seed in seeds])
            for dataset in SOURCE_ORDER[:-1]])
        ax.step(effects, np.arange(1, len(effects) + 1) / len(effects), where='post',
                color=CONTROL_COLORS[arm], lw=1.8)
    ax.axvline(0, color='.45', ls='--', lw=1)
    ax.set(xlabel='FedPERG − control AUC (pp) ↑', ylabel='Source fraction')
    decorate(ax, 2, 'Source-level AUC effects')
    ax = axes.flat[3]
    for arm in ['FedPERG', 'TemporalOnly', 'RandomInput']:
        series = []
        for seed in seeds:
            run = index['natural', seed, arm]
            series.append([
                (np.nan if d.get('skipped') else
                 d.get('utility_momentum_coefficient', .9)
                 if d.get('selector_used_target_evidence') else .9)
                for d in run['diagnostics']
            ])
        series = np.array(series)
        rounds = np.arange(1, 61)
        mu, sd = np.nanmean(series, axis=0), np.nanstd(series, axis=0, ddof=1)
        ax.plot(rounds, mu, color=CONTROL_COLORS[arm], lw=1.8)
        ax.fill_between(rounds, mu - sd, mu + sd, color=CONTROL_COLORS[arm], alpha=.12)
    ax.axhline(.9, color='.45', ls='--', lw=1)
    ax.set(xlabel='Communication round', ylabel='Committed momentum\ncoefficient')
    decorate(ax, 3, 'Temporal decisions')
    ax = axes.flat[4]
    metrics = ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']
    interactions = np.zeros((4, 4))
    operand_means = {}
    for i, dataset in enumerate(SOURCE_ORDER):
        operand_means[dataset] = {}
        for j, metric in enumerate(metrics):
            estimates = {arm: mean_metric([index[dataset, seed, arm] for seed in seeds], metric)
                         for arm in ['FedPERG', 'SpatialOnly', 'TemporalOnly', 'Router']}
            operand_means[dataset][metric] = estimates
            interactions[i, j] = (estimates['FedPERG'] - estimates['SpatialOnly']
                                  - estimates['TemporalOnly'] + estimates['Router']) * (
                                      -1 if metric == 'ece' else 1)
    (PROTOCOLS / 'factorial_interactions.json').write_text(json.dumps(
        {'sources': SOURCE_ORDER, 'metrics': metrics, 'values_pp': interactions.tolist(),
         'group_means_percent': operand_means,
         'formula': '(Full - Spatial - Temporal + Router) * (-1 for ECE, +1 otherwise)'},
        indent=2), encoding='utf-8')
    limit = max(.001, np.abs(interactions).max())
    cmap = LinearSegmentedColormap.from_list('joint_signed', ['#007FFF', '#FFFFFF', '#FF4F4F'])
    im = ax.imshow(interactions, cmap=cmap, vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(4), ['AUC', 'Final', 'Tail', '−ECE'], rotation=28, ha='right')
    ax.set_yticks(range(4), ['C10', 'C100', 'OH', 'Domains'], rotation=65)
    for i in range(4):
        for j in range(4):
            shown = f'{interactions[i,j]:+.2f}' if abs(interactions[i,j]) >= .005 else '0.00'
            ax.text(j, i, shown, ha='center', va='center')
    colorbar = fig.colorbar(im, ax=ax, fraction=.045, pad=.03, label='Interaction (pp)')
    colorbar.ax.tick_params(axis='y', labelrotation=90)
    decorate(ax, 4, 'Factorial interaction')
    ax = axes.flat[5]
    summaries = [effect_summary(primary, arm) for arm in ARMS[1:]]
    for i, (arm, stat) in enumerate(zip(ARMS[1:], summaries)):
        errors = (None if stat['ci95'] is None else
                  [[stat['mean'] - stat['ci95'][0]], [stat['ci95'][1] - stat['mean']]])
        ax.errorbar(i, stat['mean'], yerr=errors,
                    fmt='o', color=CONTROL_COLORS[arm], capsize=4)
    ax.axhline(0, color='.45', ls='--', lw=1)
    compact_method_ticks(ax, ARMS[1:], rotation=31)
    ax.set_ylabel('Mean AUC difference (pp) ↑')
    decorate(ax, 5, 'Source-averaged component\ncontrasts')
    handles = [Line2D([], [], color=CONTROL_COLORS[arm], marker='o', lw=1.8,
                      label=method_name(arm)) for arm in ARMS]
    fig.subplots_adjust(left=.085, right=.985, bottom=.32, top=.94, wspace=.42, hspace=.68)
    place_legend_below_axes(fig, handles)
    save(fig, 'r1_optimized_attribution')


def prediction_arrays(record):
    checkpoint = Path(record['_path']).with_suffix('.pth')
    folder = FORMAL / 'prediction_cache'
    folder.mkdir(exist_ok=True)
    cache = folder / (checkpoint.stem + '.npz')
    if cache.exists():
        with np.load(cache) as data:
            return data['logits'], data['labels'], float(data['temperature'])
    config = ExperimentConfig(**record['config'])
    config.device = 'cuda'
    experiment = FederatedExperiment(config)
    state = torch.load(checkpoint, map_location=experiment.device)
    experiment.model.load_state_dict(state['model_state'])
    logits = predict(experiment.model, experiment.data.x_test, experiment.device).numpy()
    labels = experiment.data.y_test.numpy()
    temperature = record['confidence_calibration']['temperature']
    np.savez_compressed(cache, logits=logits, labels=labels, temperature=temperature,
                        computed_with_cuda=True)
    return logits, labels, temperature


def reliability(logits, labels, temperature):
    probabilities = torch.from_numpy(logits / temperature).softmax(1).numpy()
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = predictions == labels
    bins = np.linspace(0, 1, 11)
    x, y = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        x.append(confidence[mask].mean() if mask.sum() >= 10 else np.nan)
        y.append(correct[mask].mean() if mask.sum() >= 10 else np.nan)
    return np.array(x), np.array(y)


def modeled_reach_time(record, threshold=.7):
    elapsed = [0.]
    for entry in record['diagnostics']:
        duration = .02 if entry.get('skipped') else (
            entry['modeled_parallel_round_seconds'] - entry['aggregation_seconds'])
        elapsed.append(elapsed[-1] + duration)
    first = next((h['round'] for h in record['history'] if h['accuracy'] >= threshold), None)
    return elapsed[first] if first is not None else np.inf


def calibration_and_cost(records):
    fig, axes = plt.subplots(2, 3, figsize=(15.7, 10.2))
    controls = ['FedPERG', 'Router', 'Permuted']
    ax = axes.flat[0]
    right = ax.twinx()
    modes = ['centralized', 'distributed', 'unlabeled']
    for j, arm in enumerate(controls):
        runs = [[r for r in records if r['_suite'] == 'calibration'
                 and r['_arm'] == arm and r['config']['calibration_mode'] == mode] for mode in modes]
        assert all(len(group) == 5 for group in runs)
        positions = np.arange(3) + (j - 1) * .23
        means = [mean_metric(group) for group in runs]
        ci = [interval([value(r) for r in group])['ci95'] for group in runs]
        ax.bar(positions, means, width=.21, color=CONTROL_COLORS[arm], alpha=.5,
               yerr=np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, ci)]).T, capsize=2)
        right.plot(positions, [mean_metric(group, 'ece') for group in runs],
                   'o-', color=CONTROL_COLORS[arm], lw=1.5, ms=4)
    ax.set_xticks(range(3), ['Centralized', 'Distributed', 'Unlabeled'], rotation=28, ha='right')
    ax.set_ylabel('AUC (%) ↑')
    ax.tick_params(axis='y',labelrotation=65)
    right.set_ylabel('ECE (%) ↓')
    decorate(ax, 0, 'Label access and reliability')
    ax = axes.flat[1]
    for i, arm in enumerate(controls):
        effects = []
        for seed in range(220, 225):
            default = next(r for r in records if r['_suite'] == 'calibration' and r['_arm'] == arm
                           and r['config']['seed'] == seed and r['config']['calibration_mode'] == 'centralized')
            shifted = next(r for r in records if r['_suite'] == 'shift' and r['_arm'] == arm
                           and r['config']['seed'] == seed)
            effects.append(value(shifted) - value(default))
        violin = ax.violinplot(effects, positions=[i], showextrema=False, showmedians=True)
        violin['bodies'][0].set(facecolor=CONTROL_COLORS[arm], alpha=.55)
        violin['cmedians'].set(color=CONTROL_COLORS[arm])
        ax.scatter(i + np.linspace(-.05, .05, 5), effects, s=15, color=CONTROL_COLORS[arm])
    ax.axhline(0, color='.45', ls='--', lw=1)
    ax.set_xticks(range(3), [method_name(a) for a in controls], rotation=28, ha='right')
    ax.set_ylabel('Shifted − default AUC (pp) ↑')
    decorate(ax, 1, 'Held-out shift effects')
    ax = axes.flat[2]
    prediction_groups = {}
    for arm in ['FedPERG', 'Router', 'FedAvgM-Gate']:
        runs = [r for r in records if r['_suite'] == 'matrix' and r['config']['dataset'] == 'officehome'
                and r['config']['regime'] == 'compound' and r['_arm'] == arm]
        if arm == 'Router':
            runs = [r for r in records if r['_suite'] == 'attribution'
                    and r['config']['dataset'] == 'officehome' and r['config']['regime'] == 'compound'
                    and r['config']['seed'] in range(220, 223) and r['_arm'] == arm]
        assert len(runs) == 3
        arrays = [prediction_arrays(r) for r in runs]
        prediction_groups[arm] = arrays
        for scaled, style in [(False, '-'), (True, '--')]:
            curves = [reliability(logits, labels, temperature if scaled else 1.)
                      for logits, labels, temperature in arrays]
            def observed_mean(values):
                array=np.asarray(values)
                count=np.isfinite(array).sum(axis=0)
                return np.divide(np.nansum(array,axis=0),count,
                                 out=np.full(array.shape[1],np.nan),where=count>0)
            x = observed_mean([curve[0] for curve in curves])
            y = observed_mean([curve[1] for curve in curves])
            ax.plot(100*x, 100*y, style, color=CONTROL_COLORS[arm], lw=1.7)
    ax.plot([0,100], [0,100], ':', color='.45')
    ax.set(xlabel='Confidence (%)', ylabel='Observed accuracy (%) ↑')
    ax.tick_params(axis='y',labelrotation=65)
    decorate(ax, 2, 'Reliability before / after scaling')
    ax = axes.flat[3]
    quantiles = np.linspace(.01, .99, 99)
    for arm, arrays in prediction_groups.items():
        curves = []
        for logits, labels, temperature in arrays:
            probabilities = torch.from_numpy(logits / temperature).log_softmax(1).numpy()
            losses = -probabilities[np.arange(len(labels)), labels]
            curves.append(np.quantile(losses, quantiles))
        curves = np.array(curves)
        mu, sd = curves.mean(0), curves.std(0, ddof=1)
        ax.plot(quantiles, mu, color=CONTROL_COLORS[arm], lw=1.8)
        ax.fill_between(quantiles, mu-sd, mu+sd, color=CONTROL_COLORS[arm], alpha=.13)
    ax.set(xlabel='Loss quantile', ylabel='Log loss (nats) ↓')
    decorate(ax, 3, 'Calibrated loss quantiles')
    ax = axes.flat[4]
    timing_arms = ['FedPERG','Router','SpatialOnly','TemporalOnly','FedAvgM-Gate']
    for i, arm in enumerate(timing_arms):
        samples = [1000*d['aggregation_seconds'] for r in records
                   if r['_suite']=='timing' and r['_arm']==arm for d in r['diagnostics'][10:]]
        assert len(samples) == 150
        violin = ax.violinplot(samples, positions=[i], showextrema=False, showmedians=True)
        violin['bodies'][0].set(facecolor=CONTROL_COLORS[arm], alpha=.55)
        violin['cmedians'].set(color=CONTROL_COLORS[arm])
    ax.set_yscale('log')
    compact_method_ticks(ax, timing_arms, rotation=28)
    ax.set_ylabel('GPU server latency (ms) ↓')
    ax.tick_params(axis='y',labelrotation=90)
    decorate(ax, 4, 'Isolated server overhead')
    ax = axes.flat[5]
    time_limit = max(sum(.02 if d.get('skipped') else
                         d['modeled_parallel_round_seconds'] - d['aggregation_seconds']
                         for d in r['diagnostics']) for r in records if r['_suite']=='natural')
    for arm in ['FedPERG','Router','TemporalOnly','FedAvgM-Gate']:
        runs = [r for r in records if r['_suite']=='natural' and r['_arm']==arm]
        times = np.sort([modeled_reach_time(r) for r in runs])
        attained = times[np.isfinite(times)]
        x = np.r_[0, attained, time_limit]
        y = np.r_[np.arange(len(attained)+1)/len(times), len(attained)/len(times)]
        ax.step(x, y, where='post', color=CONTROL_COLORS[arm], lw=1.8)
    ax.set(xlabel='Modeled time (s) ↓', ylabel='70% target reach probability ↑')
    decorate(ax, 5, 'Time to target with censoring')
    handles = [Line2D([],[],color=CONTROL_COLORS[a],lw=1.8,label=method_name(a))
               for a in ['FedPERG','Router','Permuted','SpatialOnly','TemporalOnly','FedAvgM-Gate']]
    handles += [Line2D([],[],color='.4',lw=1.7,ls='-',label='(c) Raw probabilities'),
                Line2D([],[],color='.4',lw=1.7,ls='--',label='(c) Temperature scaled')]
    fig.subplots_adjust(left=.085,right=.985,top=.94,bottom=.31,wspace=.48,hspace=.65)
    place_legend_below_axes(fig, handles)
    save(fig, 'r1_optimized_calibration')
