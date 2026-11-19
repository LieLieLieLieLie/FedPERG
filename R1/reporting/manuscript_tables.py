"""Build native table data and LaTeX from the GPU-only confirmation records."""
import csv
import json
from collections import Counter
from statistics import mean, stdev

import numpy as np

from experiments.optimization import METHODS
from fedperg.config import ExperimentConfig
from fedperg.paths import DEVELOPMENT, PROTOCOLS, RESULTS, ROOT
from reporting.diagnostic_figures import modeled_reach_time, paired_clusters
from reporting.figures import NAMES, method_name, value
from reporting.reviewer_baselines import load_extension, summarize
from reporting.statistics import interval
from reporting.summary_inputs import confidence_mean, effect_summary, mean_metric
from reporting.manuscript_sections import write_section

PAPER = ROOT.parents[1] / 'paper' / 'INS' / 'R1'
EXPECTED = {
    'matrix': 270, 'natural': 160, 'attribution': 300, 'stress': 100,
    'quality': 120, 'reward': 120, 'sensitivity': 240, 'calibration': 45,
    'shift': 15, 'deep': 20, 'cnn': 10, 'timing': 15,
}
TABLES = {}
FULL_WIDTH_TABLES = {2, 23}
COMPACT_TABLES = {5, 11, 15, 19, 20, 22}
CALIBRATION_HEADERS = (
    'Method', 'Accuracy', 'ECE\nraw', 'ECE\nscaled', 'NLL\nraw', 'NLL\nscaled',
)
STRESS_CASES = [
    ('Two invitations', 2, 0., 1.), ('Four invitations', 4, 0., 1.),
    ('Sixteen invitations', 16, 0., 1.), ('50% dropout', 8, .5, 1.),
    ('Severe imbalance', 8, 0., 3.),
]


def select(records, suite, arm=None, dataset=None, regime=None, seeds=None):
    return [
        r for r in records if r['_suite'] == suite
        and (arm is None or r['_arm'] == arm)
        and (dataset is None or r['config']['dataset'] == dataset)
        and (regime is None or r['config']['regime'] == regime)
        and (seeds is None or r['config']['seed'] in seeds)
    ]


def cell(records, metric, sd=False):
    samples = [value(record, metric) for record in records]
    estimate = f'{mean_metric(records, metric):.3f}'
    return estimate + (f' ({stdev(samples):.3f})' if sd else '')


def paired_values(left, right, metric='convergence_auc'):
    left_index = {r['config']['seed']: r for r in left}
    right_index = {r['config']['seed']: r for r in right}
    assert left_index.keys() == right_index.keys()
    return [value(left_index[seed], metric) - value(right_index[seed], metric)
            for seed in sorted(left_index)]


def effect_cell(values):
    stat = interval(values)
    return f'{stat["mean"]:+.3f}', f'[{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}]'


def register(number, label, caption, headers, rows, grouped=False, ranking=None, wide=False, underline=True,row_ranking=None):
    assert label not in TABLES
    assert all(len(row) == len(headers) for row in rows), label
    # Keep default manuscript type size; use the full page when columns require it.
    shown_columns = len(headers) - int(grouped)
    wide = wide or shown_columns >= 5
    entry = dict(number=number, label=label, caption=caption, headers=headers, rows=rows,
                 grouped=grouped, ranking=ranking or {}, wide=wide, underline=underline,row_ranking=row_ranking or [])
    TABLES[label] = entry
    with (RESULTS / 'tables' / f'table{number:02d}_{label.split(":")[1]}.csv').open(
            'w', encoding='utf-8', newline='') as handle:
        csv.writer(handle).writerows([headers, *rows])
    return entry


def latex(entry):
    def escape(text):
        return (str(text).replace('%', r'\%').replace('_', r'\_').replace('±', r' $\pm$ ')
                .replace('×', r'$\times$').replace('−', r'$-$').replace('–', '--').replace('—', '---'))
    offset = int(entry['grouped'])
    headers, rows = entry['headers'], entry['rows']
    number = entry['number']
    is_wide = number in FULL_WIDTH_TABLES
    environment = 'table*' if is_wide else 'table'
    placement = 'pos=!tbp' if is_wide else 'pos=!htbp'
    caption = escape(entry['caption'])
    shown_headers = CALIBRATION_HEADERS if number in {11, 20} else headers[offset:]
    if number == 19:
        shown_headers = ('Perturbation', 'AUC\nchange', '95% CI', 'Accept')
    header_cells = []
    for header in shown_headers:
        if '\n' in header:
            header_cells.append(r'\shortstack[r]{' + r'\\'.join(map(escape, header.splitlines())) + '}')
        else:
            header_cells.append(escape(header))
    padding = '1.5pt' if number in COMPACT_TABLES else '2.5pt'
    lines = [
        rf'\begin{{{environment}}}[{placement}]', r'\centering\normalfont\normalsize',
        rf'\caption{{{caption}}}\label{{{entry["label"]}}}',
        rf'\setlength{{\tabcolsep}}{{{padding}}}',
    ]
    if number == 19:
        lines.append(r'\renewcommand{\arraystretch}{0.94}')
    spec = 'l' + 'r' * (len(headers) - offset - 1)
    def tabular(body, fill_width):
        name = 'tabular*' if fill_width else 'tabular'
        start = (r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' if fill_width
                 else r'\begin{tabular}{@{}')
        return [start + spec + '@{}}', r'\toprule', ' & '.join(header_cells) + r' \\',
                r'\midrule', *body, r'\bottomrule', rf'\end{{{name}}}']

    previous = None
    body = []
    for row_index,row in enumerate(rows):
        if offset and row[0] != previous:
            if previous is not None:
                body.append(r'\midrule')
            body.append(rf'\multicolumn{{{len(headers)-1}}}{{@{{}}l}}{{\textit{{{escape(row[0])}}}}} \\*')
            previous = row[0]
        fields = [escape(v) for v in row[offset:]]
        if number == 19:
            fields[0] = fields[0].replace('coefficient', 'coeff.').replace('temperature', 'temp.')
        if number == 15:
            for column in range(1, 5):
                if ' (' in fields[column]:
                    seconds, count = fields[column].split(' (', 1)
                    fields[column] = rf'\shortstack[r]{{{seconds}\\({count}}}'
        for column, higher in entry['ranking'].items():
            peers = [r for r in rows if not offset or r[0] == row[0]]
            scores = sorted({float(str(r[column]).split()[0]) for r in peers}, reverse=higher)
            score = float(str(row[column]).split()[0])
            shown = column - offset
            if score == scores[0]:
                fields[shown] = r'\textbf{' + fields[shown] + '}'
            elif entry['underline'] and len(scores) > 1 and score == scores[1]:
                fields[shown] = r'\underline{' + fields[shown] + '}'
        if entry['row_ranking']:
            scores=sorted({float(row[column]) for column in entry['row_ranking']},reverse=True)
            for column in entry['row_ranking']:
                if float(row[column])==scores[0]:fields[column-offset]=r'\textbf{'+fields[column-offset]+'}'
                elif len(scores)>1 and float(row[column])==scores[1]:fields[column-offset]=r'\underline{'+fields[column-offset]+'}'
        keep_pair=entry['number'] in [10,11] and row_index+1<len(rows) and rows[row_index+1][0]==row[0]
        body.append(' & '.join(fields) + (r' \\*' if keep_pair else r' \\'))
    lines.extend(tabular(body, fill_width=not is_wide))
    return '\n'.join([*lines, rf'\end{{{environment}}}'])


def performance_tables(records, extension):
    rows = []
    for arm in METHODS:
        values_by_source = []
        for dataset in NAMES:
            runs = select(records, 'matrix', arm, dataset)
            clusters = [mean(value(r) for r in runs if r['config']['seed'] == seed)
                        for seed in range(220, 223)]
            values_by_source.append(clusters)
        grand = [mean(values[seed] for values in values_by_source) for seed in range(3)]
        rows.append([method_name(arm),
                     *[f'{mean(v):.3f} ({stdev(v):.3f})' for v in values_by_source],
                     f'{mean(grand):.3f} ({stdev(grand):.3f})'])
    register(2, 'tab:main',
             'Convergence AUC (%) over 60 rounds. Each source averages three heterogeneity regimes. '
             'Parentheses give SD across the three paired seed averages (220–222).',
             ['Method', 'C10', 'C100', 'OH-65', 'Mean'], rows,
             ranking={1: True, 2: True, 3: True, 4: True})
    rows = []
    for dataset, name in NAMES.items():
        for arm in ['Router', 'SpatialOnly', 'TemporalOnly', 'FedPERG']:
            runs = select(records, 'attribution', arm, dataset, 'compound')
            rows.append([name, method_name(arm), *[cell(runs, m) for m in
                         ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
    register(3, 'tab:gate-confirm',
             'Matched spatial–temporal factorial, ten paired seeds per source. AUC, final accuracy, '
             'tail accuracy and expected calibration error (ECE) are percentages. Every arm uses one gradient and seven candidate evaluations.',
             ['Source', 'Method', 'AUC', 'Final', 'Tail', 'ECE'], rows, grouped=True,
             ranking={2: True, 3: True, 4: True, 5: False})
    rows = []
    for arm in ['FedNova', 'FedLBW', 'FedLAW', 'FedLWS', 'FedAWA', 'FedDS', 'PA3Fed',
                'Fed-LAA', 'FedAvgM-Gate', 'FedPERG']:
        for dataset, name in NAMES.items():
            runs = (select(extension, 'reviewer_baselines', arm, dataset, 'compound')
                    if arm in {'PA3Fed', 'Fed-LAA'} else select(records, 'matrix', arm, dataset, 'compound'))
            rows.append([name, method_name(arm), *[cell(runs, metric, sd=metric == 'convergence_auc')
                         for metric in ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
    rows.sort(key=lambda row: list(NAMES.values()).index(row[0]))
    register(7, 'tab:r1-baselines',
             'Compound-heterogeneity defining-mechanism comparisons, three paired seeds and 60 rounds. '
             'Parentheses give AUC SD across seeds. All reported values are percentages.',
             ['Source', 'Method', 'AUC (SD)', 'Final', 'Tail', 'ECE'], rows, grouped=True,
             ranking={2: True, 3: True, 4: True, 5: False})


def reviewer_baseline_table(records, extension):
    estimates, pooled = summarize(extension, select(records, 'matrix'))
    methods = ('FedLBW', 'PA3Fed', 'Fed-LAA', 'FedPERG')
    regimes = {'label_skew': 'Label skew', 'quantity_skew': 'Quantity skew', 'compound': 'Compound'}
    rows = [[NAMES[source], regimes[regime], *[f'{estimates[method][source, regime]:.3f}' for method in methods]]
            for source, regime in estimates['FedPERG']]
    rows.append(['Pooled', 'Equal weight', *[f'{pooled[method]:.3f}' for method in methods]])
    register(24, 'tab:reviewer-baselines',
             'Convergence AUC (%) for the requested adaptive baselines over 60 rounds and three paired seeds. '
             'Each of the nine settings has equal weight in the pooled row.',
             ['Source', 'Regime', *methods], rows, grouped=True, row_ranking=[2, 3, 4, 5])


def selection_table():
    from experiments.optimization.study import CONTEXTS
    from fedperg.paths import DEVELOPMENT
    names = {
        'brier_balance_momentum_3': 'Joint, CE+Brier',
        'proper_score': 'Spatial, CE+Brier', 'brier_field': 'Spatial, Brier',
        'reference': 'Spatial, CE', 'momentum_gate': 'AvgM+Gate',
    }
    rows = []
    for context in CONTEXTS:
        for profile, title in names.items():
            study = 'temporal' if profile == 'brier_balance_momentum_3' else 'robust'
            paths = list((DEVELOPMENT / study).glob(f'*__r1_opt_{study}_development_{context}_{profile}.json'))
            assert len(paths) == 3
            estimates = [json.loads(p.read_text(encoding='utf-8'))['development_validation'] for p in paths]
            rows.append([{'cifar10': 'CIFAR-10', 'officehome': 'Office-Home', 'natural': 'Natural domains'}[context],
                         title, *[f'{100*mean(v[k] for v in estimates):.3f}' for k in
                                  ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
    register(6, 'tab:r1-sensitivity',
             'Representative configurations from the independent development comparison. '
             'All fifteen configurations were ranked using four validation metrics across three contexts '
             'and seeds 200–202. The complete configuration grid remains in the saved results. Values are percentages.',
             ['Source', 'Configuration', 'AUC', 'Final', 'Tail', 'ECE'], rows, grouped=True,
             ranking={2: True, 3: True, 4: True, 5: False})


def development_grid():
    """Describe the saved selection grid without reranking confirmation results."""
    from experiments.optimization.study import ROBUST_PROFILES, TEMPORAL_PROFILES
    selection = json.loads((PROTOCOLS / 'r1_optimization_combined_selection.json').read_text())
    profiles = {**ROBUST_PROFILES, **TEMPORAL_PROFILES}
    objective = {'original': 'E', 'brier_balance': 'P', 'brier': 'B',
                 'gce': 'G', 'balanced_brier': 'W'}
    names = ['calibration_objective', 'update_reference', 'utility_strength',
             'utility_field_radius', 'utility_momentum_step', 'utility_prior_strength',
             'utility_pool_fraction', 'selector_margin', 'gate_bank_scale', 'gate_bank_offset',
             'perg_temperature', 'perg_reconstruction_weight', 'perg_median_weight',
             'improvement_weight', 'method', 'perg_gate_low', 'perg_gate_span']
    lines = [r'''The final grid uses tuples $(L,U,\alpha,r_g,\alpha_v)$. Here $L$ is E (cross-entropy), P (cross-entropy plus half Brier), B (Brier), G (generalized cross-entropy with exponent .7), or W (class-balanced Brier). The update reference $U$ is M (sample mass) or N (actual-work normalization). The last three entries are the spatial step, trust radius, and temporal step. The complete fifteen configurations and their mean validation ranks are
\begin{enumerate}[label=D\arabic*.,leftmargin=*,nosep]''']
    configurations = []
    for profile in profiles:
        observations = []
        for context in ('cifar10', 'officehome', 'natural'):
            for seed in (200, 201, 202):
                paths = []
                for study in ('robust', 'temporal'):
                    paths = list((DEVELOPMENT / study).glob(
                        f'*__s{seed}__*__r1_opt_{study}_development_{context}_{profile}.json'))
                    if paths:
                        break
                assert len(paths) == 1, (context, profile, seed)
                saved = json.loads(paths[0].read_text(encoding='utf-8'))
                assert saved['device'].startswith('cuda')
                config = ExperimentConfig(**saved['config'])
                observations.append({name: getattr(config, name) for name in names})
        assert all(config == observations[0] for config in observations), profile
        config = observations[0]
        rank = selection['validation_mean_rank'][profile]
        if config['method'] == 'FedAvgM-Gate':
            assert config['perg_gate_low'] == .82 and config['perg_gate_span'] == .32
            description = r'AvgM+Gate control ($g=.82+.32C^p$, momentum .9)'
            displayed = ['AvgM+Gate', '—', 'M', '—', '—', '—']
        else:
            reference = 'M' if config['update_reference'] == 'sample_mass' else 'N'
            steps = [f'{config[name]:g}' for name in
                     ('utility_strength', 'utility_field_radius', 'utility_momentum_step')]
            values = ','.join([objective[config['calibration_objective']], reference,
                               *steps])
            description = '$(' + values + ')$'
            displayed = ['FedPERG', objective[config['calibration_objective']], reference, *steps]
        selected = r' (selected)' if profile == selection['profile'] else ''
        lines.append(rf'\item {description}, rank {rank:.3f}{selected}.')
        configurations.append(dict(profile=profile, parameters=config, validation_mean_rank=rank,
                                   display_fields=[f'D{len(configurations)+1}', *displayed, f'{rank:.3f}']))
    assert len(configurations) == 15
    lines.append(r'''\end{enumerate}
The tested values for the fitted profiles are $\alpha\in\{.1,.3\}$, $r_g\in\{.2,.5\}$ and $\alpha_v\in\{0,.3,.6\}$, together with the five objectives and two references above. The search is limited to the fifteen listed configurations. Gate-span scale 1, intercept 0, prior strength .25, fitting fraction .5, margin .001, prior temperature .8, and residual/alignment/reward coefficients $.85/1.2/.35$ remain fixed for the fitted profiles. D9 uses sample-mass sign agreement in its fixed gate and momentum .9. It performs no evidence fitting or candidate routing, so the fitting objective, spatial step, trust radius and temporal fitting step are inapplicable. D10 has the lowest validation mean rank, 5.931, and defines the unchanged confirmation configuration.

The initial eight profiles used the same spatial-step and radius values, mass or work normalization, cross-entropy or CE plus half Brier, and gate-span scales 1 or 1.5. They produced 72 runs. The two later nine-profile grids produced 81 runs each with clean validation populations. Three profiles coincide between those two grids, yielding fifteen distinct clean-validation configurations and 135 distinct context--seed--configuration observations. Their 27 repeated observations remain in the 234-run development record but enter selection once. Every final-grid configuration has nine observations across three contexts and seeds 200--202. Ranking uses each of four validation metrics equally within every context and seed, and then averages those ranks. Test observations do not enter that ranking.''')
    write_section('development_grid', '\n'.join(lines) + '\n')
    (PROTOCOLS / 'development_grid_details.json').write_text(
        json.dumps(configurations, indent=2), encoding='utf-8')


def attribution_tables(records):
    controls = ['Router', 'SpatialOnly', 'TemporalOnly', 'RandomInput', 'Permuted', 'NoResidual']
    rows = []
    for scope, selected in [
        ('Non-digit', select(records, 'attribution', regime='compound')),
        ('Natural', select(records, 'natural')),
    ]:
        for control in controls:
            stats = effect_summary(selected, control)
            rows.append([scope, method_name(control), f'{stats["mean"]:+.3f}',
                         (f'[{stats["ci95"][0]:+.3f}, {stats["ci95"][1]:+.3f}]'
                          if stats['ci95'] is not None else '--'),
                         (f'{stats["positive"]}/{stats["n"]}' if stats['positive'] is not None else '--'),
                         (f'{stats["exact_two_sided_p"]:.4f}'
                          if stats['exact_two_sided_p'] is not None else '--')])
    register(12, 'tab:field-attribution',
             'FedPERG minus matched controls in AUC percentage points. Non-digit estimates average '
             'CIFAR-10, CIFAR-100 and Office-Home within each of ten seeds. Intervals summarize paired seed clusters. '
             'A dash denotes an unavailable statistic. Exact two-sided sign-flip probabilities require '
             'the corresponding seed-level contrasts.',
             ['Mode', 'Control', 'AUC', '95% CI', 'Wins', 'Exact p'], rows, grouped=True)
    rows = []
    for regime, title in [('label_skew', 'Label skew'), ('quantity_skew', 'Quantity skew'), ('compound', 'Compound')]:
        selected = select(records, 'attribution', regime=regime, seeds=range(220, 223))
        for control in ['Router', 'SpatialOnly', 'TemporalOnly', 'NoResidual']:
            stat = interval(paired_clusters(selected, control))
            rows.append([title, method_name(control), f'{stat["mean"]:+.3f}',
                         f'[{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}]'])
    register(13, 'tab:field-regimes',
             'Regime-specific FedPERG minus control AUC (percentage points). Each row uses the same '
             'three sources and three paired seeds, 220–222. Intervals are descriptive paired-seed bootstrap summaries.',
             ['Mode', 'Control', 'AUC', '95% CI'], rows, grouped=True)
    rows = []
    scopes = [
        ('Three non-digit sources', select(records, 'attribution', regime='compound')),
        ('Natural domains', select(records, 'natural')),
    ]
    for title, selected in scopes:
        values = paired_clusters(selected, 'Router')
        stat = effect_summary(selected, 'Router')
        effect = f'{stat["mean"]:+.3f}'
        uncertainty = (f'[{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}]'
                       if stat['ci95'] is not None else '--')
        rows.append([title, len(values), effect, uncertainty])
    joint = []
    core = scopes[0][1]
    natural = scopes[1][1]
    for a, b in zip(paired_clusters(core, 'Router'), paired_clusters(natural, 'Router')):
        joint.append((3*a+b)/4)
    core_stat, natural_stat = (effect_summary(selected, 'Router') for _, selected in scopes)
    if core_stat['ci95'] is None or natural_stat['ci95'] is None:
        effect = f'{(3 * core_stat["mean"] + natural_stat["mean"]) / 4:+.3f}'
        uncertainty = '--'
    else:
        effect, uncertainty = effect_cell(joint)
    rows.append(['Four contexts equally', len(joint), effect, uncertainty])
    register(5, 'tab:r1-weighting',
             'Sensitivity of the FedPERG minus Router-only AUC estimate to its evaluation population. '
             'Each estimate uses all ten paired seeds. No digit source enters the primary comparison.',
             ['Population', 'Seeds', 'AUC (pp)', '95% CI'], rows, wide=True)


def stress_rows(records, arm, dataset, case):
    _, k, dropout, power = case
    return [r for r in select(records, 'stress', arm, dataset)
            if (r['config']['clients_per_round'], r['config']['dropout_rate'],
                r['config']['client_imbalance_power']) == (k, dropout, power)]


def stress_tables(records):
    pooled = []
    detailed = []
    for dataset in ['cifar10', 'officehome']:
        for arm in ['FedPERG', 'Router']:
            runs = select(records, 'stress', arm, dataset)
            pooled.append([NAMES[dataset], method_name(arm),
                           *[cell(runs, m) for m in ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
        for case in STRESS_CASES:
            full = stress_rows(records, 'FedPERG', dataset, case)
            router = stress_rows(records, 'Router', dataset, case)
            auc, ci = effect_cell(paired_values(full, router))
            tail = mean(paired_values(full, router, 'worst20_accuracy'))
            skipped = mean(sum(d.get('skipped', False) for d in r['diagnostics']) for r in full)
            detailed.append([NAMES[dataset], case[0], auc, ci, f'{tail:+.3f}', f'{skipped:.1f}'])
    register(9, 'tab:r1-participation',
             'Performance under the five preset participation/dropout/imbalance conditions, equally '
             'weighted within each seed. Each source–method group contains five seeds and 25 runs. Values are percentages.',
             ['Source', 'Method', 'AUC', 'Final', 'Tail', 'ECE'], pooled, grouped=True,
             ranking={2: True, 3: True, 4: True, 5: False})
    register(22, 'tab:field-participation',
             'FedPERG minus Router-only under participation stress. Five paired seeds per condition. '
             'AUC and tail differences are percentage points. Skip counts nominal rounds with fewer than two uploads.',
             ['Source', 'Condition', 'AUC', '95% CI', 'Tail', 'Skip'], detailed, grouped=True)


def representation_tables(records):
    rows = []
    for seed in range(220, 225):
        for arm in ['FedPERG', 'Router']:
            run = select(records, 'cnn', arm, seeds=[seed])
            assert len(run) == 1
            rows.append([f'Seed {seed}', method_name(arm),
                         *[cell(run, k) for k in ['convergence_auc', 'accuracy', 'worst20_accuracy']]])
    register(10, 'tab:r1-endtoend',
             'Fully trainable Office-Home RGB CNN, 100 rounds, five paired seeds and at most three '
             'batches per local epoch. Bold marks each seed-wise maximum.',
             ['Seed', 'Method', 'AUC', 'Final', 'Tail'], rows, grouped=True,
             ranking={2: True, 3: True, 4: True}, underline=False)
    rows = []
    for arm in ['FedPERG', 'Router', 'RandomInput', 'FedAvgM-Gate']:
        runs = select(records, 'deep', arm)
        rows.append([method_name(arm), *[cell(runs, k) for k in
                     ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
    register(16, 'tab:field-deep',
             'Fully trainable GroupNorm ResNet-18 on Office-Home RGB images. All local batches are '
             'processed over 100 rounds, with five paired seeds. Values are percentages.',
             ['Method', 'AUC', 'Final', 'Tail', 'ECE'], rows,
             ranking={1: True, 2: True, 3: True, 4: False})
    rows = []
    for suite, title in [('cnn', 'RGB CNN'), ('deep', 'ResNet-18')]:
        for arm in ['FedPERG', 'Router']:
            runs = select(records, suite, arm)
            raw = confidence_mean(runs, 'raw', 'ece')
            scaled = confidence_mean(runs, 'temperature_scaled', 'ece')
            nll_raw = confidence_mean(runs, 'raw', 'nll')
            nll_scaled = confidence_mean(runs, 'temperature_scaled', 'nll')
            rows.append([title, method_name(arm), cell(runs, 'accuracy'),
                         f'{raw:.3f}', f'{scaled:.3f}', f'{nll_raw:.3f}', f'{nll_scaled:.3f}'])
    register(11, 'tab:r1-confidence',
             'Independent temperature fitting for the trainable representations. Accuracy is exactly '
             'preserved. Raw/scaled ECE is in percent and negative log-likelihood (NLL) is in nats.',
             ['Source', 'Method', 'Accuracy', 'ECE raw', 'ECE scaled', 'NLL raw', 'NLL scaled'], rows, grouped=True,
             ranking={2: True,3: False,4: False,5: False,6: False})


def calibration_tables(records):
    rows = []
    for mode in ['centralized', 'distributed', 'unlabeled']:
        for arm in ['FedPERG', 'Router', 'Permuted']:
            runs = [r for r in select(records, 'calibration', arm) if r['config']['calibration_mode'] == mode]
            acceptance = mean(mean(bool(d.get('selector_used_target_evidence')) for d in r['diagnostics'])
                              for r in runs) * 100
            rows.append([mode.capitalize(), method_name(arm),
                         *[cell(runs, m) for m in ['convergence_auc', 'worst20_accuracy', 'ece']],
                         f'{acceptance:.1f}'])
    register(17, 'tab:field-calibration',
             'Label-access comparison on Office-Home, five paired seeds and 60 rounds. Centralized '
             'and distributed fitting use CE+Brier. Unlabeled fitting and routing use predictive entropy minus '
             'half the marginal entropy. All values are percentages. Accept is a diagnostic selection frequency.',
             ['Mode', 'Method', 'AUC', 'Tail', 'ECE', 'Accept'], rows, grouped=True,
             ranking={2: True, 3: True, 4: False})
    rows = []
    for mode in ['Default', 'Lower-label biased']:
        for arm in ['FedPERG', 'Router', 'Permuted']:
            runs = (select(records, 'shift', arm) if mode != 'Default' else
                    [r for r in select(records, 'calibration', arm)
                     if r['config']['calibration_mode'] == 'centralized'])
            rows.append([mode, method_name(arm), *[cell(runs, k) for k in
                         ['convergence_auc', 'worst20_accuracy', 'ece']]])
    register(18, 'tab:field-shift',
             'Default versus lower-label-biased fitting/routing populations, five paired Office-Home seeds. '
             'The biased pool draws 80% of its examples from the lower half of class labels. Values are percentages.',
             ['Mode', 'Method', 'AUC', 'Tail', 'ECE'], rows, grouped=True,
             ranking={2: True, 3: True, 4: False})
    rows = []
    for dataset, name in NAMES.items():
        for arm in METHODS:
            runs = select(records, 'matrix', arm, dataset, 'compound')
            raw = confidence_mean(runs, 'raw', 'ece')
            scaled = confidence_mean(runs, 'temperature_scaled', 'ece')
            raw_nll = confidence_mean(runs, 'raw', 'nll')
            scaled_nll = confidence_mean(runs, 'temperature_scaled', 'nll')
            rows.append([name, method_name(arm), cell(runs, 'accuracy'), f'{raw:.3f}', f'{scaled:.3f}',
                         f'{raw_nll:.3f}', f'{scaled_nll:.3f}'])
    register(20, 'tab:field-confidence',
             'Confidence calibration for the compound matrix, three paired seeds per source. All methods '
             'fit temperature on the same separate 4% label pool. Accuracy is preserved; ECE is percent and NLL is nats.',
             ['Source', 'Method', 'Accuracy', 'ECE raw', 'ECE scaled', 'NLL raw', 'NLL scaled'], rows, grouped=True,
             ranking={2: True,3: False,4: False,5: False,6: False})


def natural_tables(records):
    rows = []
    for arm in METHODS:
        runs = select(records, 'natural', arm)
        rows.append([method_name(arm), *[cell(runs, m) for m in
                     ['convergence_auc', 'accuracy', 'worst20_accuracy', 'ece']]])
    register(14, 'tab:field-natural',
             'Four natural Office-Home domain silos with unequal local work, dropout, and domain migration '
             'at round 31. Ten paired seeds and 60 rounds. Final is last-iterate accuracy. Values are percentages.',
             ['Method', 'AUC', 'Final', 'Tail', 'ECE'], rows,
             ranking={1: True, 2: True, 3: True, 4: False})
    rows = []
    for arm in ['FedPERG', 'Router', 'TemporalOnly', 'FedAvgM-Gate']:
        runs = select(records, 'natural', arm)
        targets = []
        for threshold in [.5, .6, .65, .7]:
            times = [modeled_reach_time(r, threshold) for r in runs]
            finite = [t for t in times if np.isfinite(t)]
            targets.append(f'{np.median(finite):.1f} ({len(finite)}/10)' if finite else '0/10')
        skipped = mean(sum(d.get('skipped', False) for d in r['diagnostics']) for r in runs)
        rows.append([method_name(arm), *targets, f'{skipped:.1f}'])
    register(15, 'tab:field-resources',
             'Modeled client-compute plus upload seconds to target accuracy (lower is better). Parenthesized '
             'counts are successful runs out of ten (higher is better). Skip counts missed rounds (lower is better). '
             'Medians condition on reaching a target and exclude server evaluation.',
             ['Method', '50% (s)', '60% (s)', '65% (s)', '70% (s)', 'Skip'], rows)


def sensitivity_table(records):
    names = {
        'utility_strength': 'Spatial step', 'utility_field_radius': 'Trust radius',
        'utility_prior_strength': 'Prior strength', 'utility_pool_fraction': 'Fitting fraction',
        'selector_margin': 'Routing margin', 'gate_bank_scale': 'Gate span',
        'utility_momentum_step': 'Temporal step',
        'perg_reconstruction_weight': 'Residual coefficient',
        'perg_median_weight': 'Alignment coefficient', 'perg_temperature': 'Prior temperature',
        'improvement_weight': 'Reward coefficient', 'gate_bank_offset': 'Gate intercept',
    }
    rows = []
    for dataset in ['cifar10', 'officehome']:
        baseline = select(records, 'attribution', 'FedPERG', dataset, 'compound', range(220, 225))
        for parameter, title in names.items():
            for factor in ['0p5', '1p5']:
                suffix = f'_{parameter}_{factor}_fedperg'
                runs = [r for r in select(records, 'sensitivity', 'FedPERG', dataset)
                        if r['config']['experiment_tag'].endswith(suffix)]
                assert len(runs) == 5
                delta, ci = effect_cell(paired_values(runs, baseline))
                acceptance = mean(mean(bool(d.get('selector_used_target_evidence')) for d in r['diagnostics'])
                                  for r in runs) * 100
                suffix_text = (' −0.08' if factor == '0p5' else ' +0.08') if parameter=='gate_bank_offset' else (' ×0.5' if factor == '0p5' else ' ×1.5')
                rows.append([NAMES[dataset], title + suffix_text,
                             delta, ci, f'{acceptance:.1f}'])
    register(19, 'tab:field-sensitivity',
             'One-factor perturbations of the selected configuration, paired seeds 220–224 per row. Gain is '
             'AUC difference in percentage points from the unchanged default. Acceptance is percent. '
             'Gate intercept uses offsets −0.08/+0.08. All perturbed outcomes are retained.',
             ['Source', 'Perturbation', 'AUC change', '95% CI', 'Accept'], rows, grouped=True, wide=True)


def reward_tables(records):
    rows = []
    for dataset in ['cifar10', 'officehome']:
        for work, difficulty in [(False, 0.), (True, 0.), (True, .5)]:
            group = f'{NAMES[dataset]}, ' + ('work 1/2/4/8' if work else 'reference work') + f', difficulty {difficulty:g}'
            for mode in ['raw', 'none', 'per_step', 'relative']:
                runs = [r for r in select(records, 'reward', 'FedPERG', dataset)
                        if r['config']['unequal_work'] == work and r['config']['class_difficulty'] == difficulty
                        and r['config']['reward_mode'] == mode]
                assert len(runs) == 5
                rows.append([group, mode.replace('_', ' '), cell(runs, 'convergence_auc'),
                             cell(runs, 'worst20_accuracy')])
    register(8, 'tab:r1-reward',
             'Reward definitions under independently varied local work and class difficulty. Five paired '
             'seeds per condition; values are percentages. Proxy-reward forwards are diagnostic overhead, '
             'so this experiment is excluded from runtime comparisons.',
             ['Source', 'Reward', 'AUC', 'Tail'], rows, grouped=True,ranking={2: True,3: True})
    rows = []
    for dataset in ['cifar10', 'officehome']:
        for rate in [0., .18, .36]:
            def group(arm, reward='raw'):
                return [r for r in select(records, 'quality', arm, dataset)
                        if r['config']['compound_label_fault_rate'] == rate
                        and r['config']['reward_mode'] == reward]
            full, router, temporal, zero = group('FedPERG'), group('Router'), group('TemporalOnly'), group('FedPERG', 'none')
            effect, ci = effect_cell(paired_values(full, router))
            reward_delta = mean(paired_values(full, zero))
            rows.append([NAMES[dataset], f'{100*rate:.0f}', cell(full, 'convergence_auc'),
                         cell(router, 'convergence_auc'), cell(temporal, 'convergence_auc'),
                         effect, ci, f'{reward_delta:+.3f}'])
    register(23, 'tab:field-quality',
             'Label-quality stress at fixed unequal work and class difficulty, five paired seeds. Fault is '
             'the label-flip percentage among affected clients. AUC is percent; Full−Router and Raw−Zero '
             'reward differences are percentage points.',
             ['Source', 'Fault', 'FedPERG', 'Router', 'Temporal', 'Full−Router', '95% CI', 'Raw−Zero'],
             rows, grouped=True, wide=True,row_ranking=[2,3,4])


def timing_tables(records):
    arms = ['FedPERG', 'Router', 'SpatialOnly', 'TemporalOnly', 'FedAvgM-Gate']
    rows, runtime = [], []
    for arm in arms:
        runs = select(records, 'timing', arm)
        medians = [1000*np.median([d['aggregation_seconds'] for d in r['diagnostics'][10:]]) for r in runs]
        rows.append([method_name(arm), f'{np.median(medians):.2f}',
                     f'[{min(medians):.2f}, {max(medians):.2f}]'])
        horizons = []
        for seed in range(220, 223):
            paired_runs = [r for r in select(records, 'timing', seeds=[seed])]
            horizon = min(r['history'][-1]['elapsed_seconds'] for r in paired_runs)
            run = next(r for r in paired_runs if r['_arm'] == arm)
            times = np.array([h['elapsed_seconds'] for h in run['history']])
            accuracies = np.array([h['accuracy'] for h in run['history']])
            interior = times < horizon
            x = np.r_[times[interior], horizon]
            y = np.r_[accuracies[interior], np.interp(horizon, times, accuracies)]
            horizons.append(100*np.trapz(y, x)/horizon)
        runtime.append([method_name(arm), f'{mean(r["runtime_seconds"] for r in runs):.2f}',
                        f'{mean(horizons):.3f}', f'{mean(r["communication_mb"] for r in runs):.3f}'])
    register(21, 'tab:field-cost',
             'Isolated server-update latency after warmup. Each seed contributes its median over '
             'rounds 11–60. The range covers the three seed medians. The benchmark runs after all other experiment queues finish.',
             ['Method', 'Median (ms)', 'Range (ms)'], rows, ranking={1: False})
    register(4, 'tab:runtime',
             'Isolated common-horizon wall-time comparison. Time AUC integrates accuracy to the fastest '
             'paired run horizon and divides by that horizon. Communication counts transmitted model updates and reward scalars.',
             ['Method', 'Wall (s)', 'Time AUC (%)', 'Upload (MB)'], runtime,
             ranking={1: False, 2: True, 3: False})
    crossings, details = [], []
    index = {(run['_arm'], run['config']['seed']): run for run in select(records, 'timing')}
    for target in (.50, .60, .65, .70):
        for seed in range(220, 223):
            reached = [next((h for h in index[arm, seed]['history'] if h['accuracy'] >= target), None)
                       for arm in ('FedPERG', 'FedAvgM-Gate')]
            rounds = [str(h['round']) if h else '>60' for h in reached]
            times = [1000 * h['elapsed_seconds'] / h['round'] if h else None for h in reached]
            critical = None
            if all(reached):
                full, baseline = reached
                assert baseline['round'] > full['round']
                critical = 1000 * (full['elapsed_seconds'] - baseline['elapsed_seconds']) / (
                    baseline['round'] - full['round'])
                equality = full['elapsed_seconds'] + full['round'] * critical / 1000
                assert abs(equality - baseline['elapsed_seconds'] - baseline['round'] * critical / 1000) < 1e-9
            crossings.append([f'{round(100 * target)}%', str(seed), *rounds,
                              *[f'{time:.2f}' if time is not None else '---' for time in times],
                              f'{critical:.2f}' if critical is not None else '---'])
            details.append(dict(target=target, seed=seed, reached=reached,
                                per_round_ms=times, critical_delay_ms=critical))
    register(25, 'tab:break-even',
             'Added communication-delay thresholds from the isolated Office-Home timing histories. '
             'P denotes FedPERG and B denotes AvgM+Gate. R is the first observed target round. '
             's is elapsed simulation time to that observation divided by R. All s and critical delays are ms. '
             '>60 is right-censored and no threshold is assigned. Evaluation is every two rounds.',
             ['Target', 'Seed', 'R(P)', 'R(B)', 's(P)', 's(B)', 'Critical c'], crossings, grouped=True)
    (PROTOCOLS / 'break_even_details.json').write_text(json.dumps(details, indent=2), encoding='utf-8')


def build(records):
    actual = Counter(r['_suite'] for r in records)
    if dict(actual) != EXPECTED:
        missing = {name: expected-actual[name] for name, expected in EXPECTED.items() if actual[name] != expected}
        raise RuntimeError(f'GPU experiments remain incomplete: {missing}')
    assert all(str(r['device']).startswith('cuda') for r in records)
    TABLES.clear()
    extension = load_extension()
    performance_tables(records, extension)
    reviewer_baseline_table(records, extension)
    selection_table()
    development_grid()
    attribution_tables(records)
    stress_tables(records)
    representation_tables(records)
    calibration_tables(records)
    natural_tables(records)
    sensitivity_table(records)
    reward_tables(records)
    timing_tables(records)
    assert len(TABLES) == 24
    assert sorted(entry['number'] for entry in TABLES.values()) == list(range(2, 26))
    (PROTOCOLS / 'manuscript_tables.json').write_text(json.dumps(TABLES, indent=2), encoding='utf-8')
    return TABLES
