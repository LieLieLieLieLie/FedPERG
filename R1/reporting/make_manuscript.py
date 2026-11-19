"""Generate the manuscript tables, figures and prose from complete CUDA records."""
import json
import shutil

from experiments.optimization import METHODS
from fedperg.paths import PROTOCOLS
from reporting.diagnostic_figures import attribution, calibration_and_cost
from reporting.figures import PAPER, convergence, figures_setup, performance_figure, value
from reporting.manuscript_tables import build, latex, select
from reporting.optimization_results import load
from reporting.summary_inputs import effect_text
from reporting.manuscript_sections import write_section


def effect(records, control, metric='convergence_auc'):
    return effect_text(records, control, metric)


def figure(path, label, caption, width='\\linewidth'):
    return '\n'.join([
        r'\begin{figure*}[!tb]', r'\centering',
        rf'\includegraphics[width={width}]{{figures/{path}.pdf}}',
        rf'\caption{{{caption}}}\label{{{label}}}', r'\end{figure*}',
    ])


def write_results(records, tables):
    def table(label):
        return latex(tables[label])
    matrix = select(records, 'matrix')
    primary = select(records, 'attribution', regime='compound')
    natural = select(records, 'natural')
    summary = json.loads((PROTOCOLS/'r1_optimization_combined_confirmation_summary.json').read_text())
    recent = json.loads((PROTOCOLS/'reviewer_baselines_summary.json').read_text())
    pa3fed, fed_laa = (recent['effects'][method] for method in ('PA3Fed', 'Fed-LAA'))
    matrix_rank = summary['ranks']['matrix']['mean_metric_rank']
    natural_rank = summary['ranks']['natural']['mean_metric_rank']
    main = [r'\section{Experiments}', r'\label{sec:experiments}', r'''
\subsection{Protocol and evaluation questions}
All development and confirmation experiments use CUDA on an NVIDIA RTX 4080 with Python 3.9 and PyTorch 2.0.0. The nine-setting matrix crosses CIFAR-10, CIFAR-100 and Office-Home with label skew, quantity skew and compound heterogeneity. Twenty clients share a model; eight are invited per round. The horizon is 60 rounds. The matrix uses three paired seeds, 220--222; compound attribution and natural-domain tests use ten paired seeds, 220--229. Fully trainable RGB CNN and GroupNorm ResNet-18 tests use 100 rounds and five paired seeds. Appendix~\ref{app:implementation} specifies subset sizes, local-work budgets, data-access rules and baseline ports. These controlled studies are not full-split image-classification leaderboards.

The ten primary methods are FedAvgM, SCAFFOLD, FedNova, FedLBW, FedLAW, FedLWS, FedAWA, FedDS, AvgM+Gate and FedPERG. They share client data exclusions, partitions, invitations, local training and evaluation horizons. Proxy-aware baselines receive the same at-most-512 public examples; FedDS receives their inputs alone. Server computation remains method-specific. Recent comparisons include the verified defining mechanisms of FedLWS and FedAWA (2025), and FedLBW and FedDS (2026). We distinguish these common-harness ports from reproductions of complete published protocols.

Convergence AUC is trapezoidal area under test accuracy over communication rounds, divided by the training horizon. Final accuracy uses the last iterate. Tail accuracy is the mean accuracy of the lowest-performing 20\% of client test groups. ECE uses 15 confidence bins. All four are reported separately; higher AUC, final and tail accuracy are better, whereas lower ECE is better. Development selection weights the four metric ranks equally. Figure~\ref{fig:multi}(c) ranks the seed-averaged values within each setting. Uncertainty uses paired seed clusters; sources are averaged within a seed before resampling. For aggregate-only summaries, means and available intervals are reported without reconstructing individual runs. Exact sign-flip probabilities and seed-level distributions are shown only when the corresponding paired records are available.

\subsection{Performance against nine baselines}
''', table('tab:main'),
        f'Table~\\ref{{tab:main}} reports the source-wise convergence comparison. '
        'FedPERG has the highest source-wise mean convergence AUC. '
        'Metric-specific setting ranks summarize the accuracy--reliability trade-off. '
        'Figure~\\ref{fig:convergence} retains all ten methods in every setting and shows seed variability throughout training. '
        'Figure~\\ref{fig:multi} adds client distributions, calibration, setting-level effects and the accuracy--calibration trade-off.',
        'Table~\\ref{tab:reviewer-baselines} extends this matrix to PA3Fed and Fed-LAA under '
        'the same initialization, client schedules, data exclusions and training horizon. '
        'Table~\\ref{tab:r1-baselines} reports their final accuracy, tail accuracy and ECE under compound heterogeneity.',
        figure('r1_optimized_convergence', 'fig:convergence',
               'Test-accuracy trajectories for all ten methods in the nine source--heterogeneity settings. '
               'Curves are paired-seed means and every shaded band, in all three rows, is plus or minus one '
               'standard deviation across available trajectories for seeds 220--222. Terminal markers give the '
               'final-accuracy means used in the tables. The same method order and colors apply to all panels. '
               'The band depicts run variability, not a 95\\% confidence interval.'),
        figure('r1_optimized_performance', 'fig:multi',
               'Complementary performance views of the ten-method matrix. (a) Tail-client accuracy distributions '
               'over all source--regime--seed units; method colors are specified by the separate swatch rows. '
               '(b) ECE by method and setting. (c) Mean setting ranks for each metric. '
               '(d) Empirical distributions of FedPERG-minus-baseline AUC. '
               '(e) Source-wise final accuracy versus ECE. (f) AUC win/tie/loss counts against each baseline, '
               'using an absolute .01-point tie threshold. The distributions retain all 27 paired units.'),
        r'\subsection{Separating spatial evidence, temporal retention and routing}',
        table('tab:gate-confirm'),
        f'Table~\\ref{{tab:gate-confirm}} matches the fitting gradient budget, seven routing forwards and candidate spans. '
        f'Initialization, client schedules and data access are paired, while each arm evolves its own history. Across the three non-digit sources, the complete mechanism changes AUC by '
        f'{effect(primary,"Router")} relative to Router-only, '
        f'{effect(primary,"RandomInput")} relative to randomized fitting correspondence and '
        f'{effect(primary,"TemporalOnly")} relative to temporal fitting alone. '
        'These are contrasts between independently trained methods. '
        'Spatial-only and Temporal-only identify which fitted action supplies the benefit; the no-residual control '
        'isolates the leave-one term within the consistency prior. The signed factorial interaction in '
        'Fig.~\\ref{fig:stress}(e) tests additivity separately. CIFAR-100 has a positive AUC interaction '
        'of approximately .97 points. CIFAR-10 has a smaller interaction of +.08 points, '
        'Office-Home remains near $-.01$, and the natural-domain interaction is near zero. '
        'The natural-domain ECE interaction is approximately +.01 points after sign reversal. '
        'Spatial--temporal interaction is therefore source- and metric-dependent. '
        'Tables~\\ref{tab:field-attribution} '
        'and~\\ref{tab:field-regimes} retain the available signed contrasts and uncertainty. '
        'On CIFAR-100, full fitting reaches AUC 25.853 and final accuracy 35.767, exceeding '
        'Spatial-only at 25.146 and 35.255; tail accuracy remains below that control '
        '(28.150 versus 28.449). On Office-Home, the complete method has higher AUC than '
        'either single-action control, while Spatial-only retains higher final and tail accuracy.',
        figure('r1_optimized_attribution', 'fig:stress',
               'Matched mechanism evaluation on three non-digit sources and natural Office-Home domains, '
               'with ten seeds per source. (a) Source AUC means, with 95\\% seed-bootstrap intervals where '
               'seed-level records are available. (b) Tail-accuracy means and available seed distributions. '
               '(c) Empirical distributions of AUC effects across the three non-digit source means for the five '
               'controls with source-resolved summaries. '
               '(d) Committed momentum coefficients in the natural-domain trajectory; shading is one standard deviation '
               'and skipped rounds are omitted. (e) The factorial interaction Full$-$Spatial$-$Temporal$+$Router '
               'in percentage points, with ECE sign reversed so positive values indicate benefit. '
               '(f) Source-averaged AUC effects for the six primary contrasts; intervals are shown where available. '
               'Aggregate means do not determine seed-level dispersion or statistical significance.'),
        r'\subsection{Computation and the evaluation population}', table('tab:runtime'),
        r'''Table~\ref{tab:runtime} uses a dedicated sequential benchmark after the other training jobs have finished. FedPERG takes 5.46 seconds against 3.25 for AvgM+Gate, with time AUC 45.103\% against 50.655\%. Time AUC uses the shortest completed wall-time horizon among paired methods for each seed. One fitting gradient and seven forwards add server work, and Table~\ref{tab:field-cost} reports server-update medians of 34.78 and .42 ms. Those medians describe the aggregation stage. Full elapsed time also includes local training, evaluation and loop overhead. The client-compute/upload values in Table~\ref{tab:field-resources} use a separate software model. Table~\ref{tab:break-even} applies Eq.~(\ref{eq:break-even}) to recorded target-crossing times to state the additional delay needed for round savings to offset this full measured cost. The latency distribution and target-reaching curve appear in Fig.~\ref{fig:r1-random}.
''', table('tab:r1-weighting'),
        r'''Table~\ref{tab:r1-weighting} separates the three-source estimand from the natural-domain estimand. Each seed contributes once per population, and equal-context weighting assigns one quarter to each source/domain context. MNIST is excluded throughout this confirmation study. This prevents an easy digit source from determining the primary evidence claim.
''']
    additional = [r'\section{Additional Experimental Results}', r'\label{app:results}',
        r'\subsection{Development selection and recent baselines}', table('tab:r1-sensitivity'),
        r'''Table~\ref{tab:r1-sensitivity} shows representative configurations from the clean validation grid. The full fifteen-configuration comparison uses only seeds 200--202 and the separate confidence/validation pool. It ranks validation AUC, final accuracy, tail accuracy and ECE equally across three development contexts. The selected CE+Brier joint field has spatial step .3, trust radius .5, temporal step .3 and prior strength .25. All subsequent contexts and confirmation seeds retain this configuration. Neither confirmation test values nor per-setting ranks select a different parameter.
''',table('tab:r1-baselines'),
        r'''Table~\ref{tab:r1-baselines} expands the compound setting for the normalization and recent adaptive mechanisms. FedNova retains actual-step normalization; FedLBW uses inverse proxy loss; FedLAW jointly learns weights and model shrinkage; FedLWS retains analytic layer-wise shrinkage; FedAWA retains vector dispersion and model-distance regularization; FedDS retains its released uniform-latent-prior EM reliability estimate. The comparison does not identify an arbitrary loss-softmax heuristic as one of these methods.
The compound results also include the PA3Fed and Fed-LAA ports. Appendix~\ref{app:implementation} specifies their aggregation rules, and Table~\ref{tab:reviewer-baselines} adds their label- and quantity-skew comparisons.
''', r'\subsection{Reward confounding and participation}', table('tab:r1-reward'),
        r'''Table~\ref{tab:r1-reward} varies local effort and class difficulty independently and compares raw, zero, per-step and relative-loss improvement. Local improvement is a noisy training proxy rather than an unbiased estimate of global usefulness. Its contribution is confined to a screened prior; the directional field and disjoint router can reject a proposal. Any small or reversed reward contrast limits the claim about this prior term even when the complete method improves.
''', table('tab:r1-participation'),
        r'''Table~\ref{tab:r1-participation} pools five preset participation conditions equally. Table~\ref{tab:field-participation} resolves their individual effects. A cohort with fewer than two uploads is skipped, preserving the model and history rather than forming an unstable leave-one denominator. These tests address sampling and self-normalized mass effects in Eq.~(\ref{eq:drift-decomp}); positive gates do not by themselves correct unequal local work.
''', r'\subsection{Learning representations end to end}', table('tab:r1-endtoend'),
        r'''Table~\ref{tab:r1-endtoend} retains every paired seed for the fully trainable RGB CNN, including reversals in final or tail accuracy. The CNN study uses a three-batch cap per local epoch; the deeper ResNet study in Table~\ref{tab:field-deep} processes all batches. Both optimize image representations rather than a classifier over frozen features. Their different budgets are stated explicitly and their outcomes are not pooled as repeated seeds of the same architecture.
''', table('tab:r1-confidence'),
        r'''Table~\ref{tab:r1-confidence} isolates predictive confidence from aggregation. Temperature is fitted on a separate 4\% training-label pool excluded from all client optimizers and from fitting/routing. Positive temperature scaling preserves class predictions exactly. Reductions in ECE or NLL therefore do not imply improved final accuracy or a different training trajectory.
''', r'\subsection{Evidence attribution across sources and regimes}', table('tab:field-attribution'),
        f'Table~\\ref{{tab:field-attribution}} reports non-digit and natural-domain contrasts. '
        f'On natural domains the AUC effect is {effect(natural,"Router")} against Router-only and '
        f'{effect(natural,"TemporalOnly")} against Temporal-only. '
        'Random input changes fitting correspondence for both actions; permuted field retains temporal information '
        'and randomizes only the spatial field. These controls test different hypotheses and must not be '
        'treated as interchangeable null models. A confidence interval crossing zero does not establish an '
        'independent benefit for that component.', table('tab:field-regimes'),
        r'''Table~\ref{tab:field-regimes} uses the same three paired seeds for all regimes and reports signed effects. Compound heterogeneity has ten seeds in the primary analysis, but the regime contrast uses only the common three to avoid unequal replication. This smaller experiment is descriptive; it does not support a universal advantage of the residual or of coordinate fitting in every heterogeneity regime.
''', r'\subsection{Natural domains and resource-dependent value}', table('tab:field-natural'),
        f'Table~\\ref{{tab:field-natural}} uses four native Office-Home domains, unequal local effort, dropout '
        f'and a fixed domain migration at round 31. FedPERG has mean four-metric rank '
        f'{natural_rank["FedPERG"]:.3f}, compared with {natural_rank["FedAvgM-Gate"]:.3f} for AvgM+Gate. '
        f'Its final-accuracy contrast against AvgM+Gate is {effect(natural,"FedAvgM-Gate","accuracy")}. '
        'A lower ECE for another baseline remains a reliability trade-off; the natural-domain experiment '
        'does not evaluate generalization to an unseen domain.', table('tab:field-resources'),
        r'''Table~\ref{tab:field-resources} reports seconds to fixed accuracy targets under the declared client-compute and upload model. Time is lower-is-better; successful-run counts are higher-is-better; skipped rounds are lower-is-better. Conditional median crossing times alone can favor a method that rarely reaches the target, so the denominator remains ten for every arm and the ECDF retains right-censored runs. These are software cost-model results, not hardware deployment measurements.
''', table('tab:field-deep'),
        r'''Table~\ref{tab:field-deep} compares four independently trained GroupNorm ResNet-18 methods under full local epochs. The randomized and single-gate controls test whether the larger network preserves the benefit seen with fixed features. Available seed records remain separate from aggregate summaries; a favorable shallow-classifier result alone does not establish deep-model superiority.
''', r'\subsection{Label access and fitting-population shift}', table('tab:field-calibration'),
        r'''Table~\ref{tab:field-calibration} separates centralized labeled, decentralized labeled and unlabeled aggregation. The two labeled modes compute the same sample-weighted CE+Brier objective; summing distributed loss sums and gradients reproduces the centralized value up to floating-point arithmetic. Unlabeled aggregation uses mean predictive entropy minus half the entropy of the mean prediction in both fitting and routing. It uses no labels for these decisions. Raw accuracy, tail accuracy and ECE remain evaluation metrics; acceptance frequency is diagnostic and higher is not inherently better. Post-training temperature diagnostics still require separate labels and are not part of the label-free aggregation claim.
''', table('tab:field-shift'),
        r'''Table~\ref{tab:field-shift} biases the fitting/routing population toward lower class indices while leaving the independent confidence pool unchanged. This directly tests the discrepancy term in Corollary~\ref{cor:utility-loss} and the shift bound in Section~\ref{sec:bound-practice}. Empirical risk reduction on a biased pool cannot certify lower test risk. Figure~\ref{fig:r1-random}(a,b) contrasts label access and pool shift without merging their estimands.
''', r'\subsection{Sensitivity, confidence and cost}', table('tab:field-sensitivity'),
        r'''Table~\ref{tab:field-sensitivity} contains 240 post-selection diagnostic runs: two sources, twelve parameters, two perturbations and seeds 220--224. They form 48 conditions with five paired runs each and are included once within the 1,469 evaluation runs, separately from the 234 development runs. Their unchanged defaults reuse ten attribution runs. Uncertainty is computed over five paired seed differences for each condition. The seeds overlap confirmation and are excluded from development selection. Factors .5 and 1.5 apply to spatial step, trust radius, prior strength, fitting fraction, routing margin, gate span, temporal step, prior coefficients and temperature. The intercept uses absolute offsets $-.08$ and $+.08$. Appendix~\ref{app:development} specifies the accounting and full search grid. The perturbations diagnose the selected default and do not select a new confirmation setting.

Office-Home is sensitive to gate span. Halving its scale changes AUC by $-1.216$ points and multiplying it by 1.5 changes AUC by $+1.017$. The span rescales the current-update term in every paired Router and Evidence expert, changing its magnitude relative to the retained velocity and changing the router's available actions. Sign preservation imposes no invariance to this rescaling. Temporal-step changes have a smaller but nonmonotone effect, $+.171$ and $-.178$ points for factors .5 and 1.5. These results support parameter- and source-specific conclusions. Sensitivity on CIFAR-100 or under natural-domain migration remains unevaluated.
''', table('tab:field-confidence'),
        r'''Table~\ref{tab:field-confidence} reports raw and independently temperature-scaled ECE/NLL for all ten methods. On CIFAR-10, FedPERG reaches accuracy 56.933\%, scaled ECE 16.790\% and scaled NLL 1.435 nats. Its raw ECE of 20.929\% remains above several baselines, so improved classification and scaled confidence do not imply uniformly better unscaled calibration. Reliability curves and upper-tail log-loss quantiles in Fig.~\ref{fig:r1-random}(c,d) show how aggregate calibration summaries relate to sample-level errors. Scaling can improve confidence without resolving a mistaken classification; raw classification results remain the primary aggregation comparison.
''', table('tab:field-cost'),
        r'''Table~\ref{tab:field-cost} reports medians over rounds 11--60 separately for each of three isolated runs, then summarizes their median and range. Figure~\ref{fig:r1-random}(e) shows the round-level distribution, which is descriptive rather than 150 independent training replicates. All matched arms retain one gradient and seven forwards. The single gate remains the appropriate low-cost reference.
''', r'\subsection{Detailed participation and label-quality stress}', table('tab:field-participation'),
        r'''Table~\ref{tab:field-participation} reports every preset condition, including two-upload cohorts and severe imbalance. Its source-specific effects identify where the shared leave-one reference and the fitted action become unreliable. The missing-upload rule and skip counts distinguish a genuine change in aggregation from an unperformed update.
''', r'''Table~\ref{tab:field-quality} increases label corruption at fixed local-work and class-difficulty schedules. The Full--Router contrast measures the complete evidence mechanism, whereas Raw--Zero isolates the local reward within its screened prior. Temporal-only shows whether a quality-stress benefit is explained by contracting history rather than by source characterization. These diagnostics bound the mechanism claim instead of requiring every component to improve every metric.
''', figure('r1_optimized_calibration', 'fig:r1-random',
               'Calibration, shift and resource diagnostics from completed runs. (a) AUC bars with 95\\% seed-bootstrap '
               'intervals and raw ECE lines under three label-access modes. (b) Paired shifted-minus-default '
               'AUC distributions. (c) Raw and temperature-scaled reliability curves for three methods. '
               '(d) Temperature-scaled log-loss quantiles with one-standard-deviation shading across three seeds. '
               '(e) Isolated aggregation latency after warmup, showing rounds within three runs. '
               '(f) Time-to-70\\%-accuracy ECDF under the client-compute/upload model, with all ten runs in '
               'each denominator and unreached targets right-censored.'),
        table('tab:field-quality'),
        r'\subsection{Adaptive aggregation across the full matrix}', table('tab:reviewer-baselines'),
        'Table~\\ref{tab:reviewer-baselines} reports convergence AUC for all nine settings. '
        'FedPERG minus PA3Fed is '
        f'{pa3fed["convergence_auc"]["mean"]:+.3f} percentage points on the equal-weight mean '
        f'and is positive in {pa3fed["convergence_auc"]["settings_positive"]}/9 settings. '
        f'The corresponding difference for Fed-LAA is {fed_laa["convergence_auc"]["mean"]:+.3f} points '
        f'and is positive in {fed_laa["convergence_auc"]["settings_positive"]}/9 settings. '
        'The pooled estimates summarize the three paired seeds and give equal weight to the three regimes '
        'within each source. FedPERG has higher mean final accuracy in '
        f'{pa3fed["accuracy"]["settings_positive"]}/9 settings against PA3Fed and '
        f'{fed_laa["accuracy"]["settings_positive"]}/9 against Fed-LAA. '
        'Calibration can produce different rankings '
        '(Table~\\ref{tab:r1-baselines}). The saved records retain all four metrics for each setting '
        'and seed. Each baseline uses its own server update, so this experiment compares complete algorithms. '
        'The matched factorial in Table~\\ref{tab:gate-confirm} measures the evidence-specific effect.',
        r'\subsection{Measured costs and added communication delay}',
        table('tab:break-even'),
        r'''Table~\ref{tab:break-even} uses the same isolated timing runs as Tables~\ref{tab:runtime} and~\ref{tab:field-cost}. For seed 220 at 60\% accuracy, $R_P=26$, $R_B=32$, $s_P=95.32$ ms and $s_B=55.00$ ms, giving $c^*=119.72$ ms. Across all three paired seeds, the 60\% thresholds range from 40.34 to 178.78 ms. Under the fixed-delay model, an added delay above 178.78 ms per round offsets the measured target-crossing cost in all three runs; below 40.34 ms it offsets none. Intermediate delays depend on the seed. At 65\%, one FedPERG run is censored, and neither method reaches 70\% in these timing runs. The ten-run resource simulation in Table~\ref{tab:field-resources} uses different client-work and domain schedules, so its crossing times cannot replace these isolated observations. The measured thresholds condition on the measurement system, sequential simulation, evaluation grid and transport model. They do not specify a validated network bandwidth or speedup on heterogeneous devices.
''']
    # Add numerical interpretation to the tables that test transfer and stress,
    # keeping every signed outcome rather than describing the protocol alone.
    comments={
        'tab:field-deep':(
            'For ResNet-18, FedPERG changes AUC by '+effect(select(records,'deep'),'Router')+
            ', final accuracy by '+effect(select(records,'deep'),'Router','accuracy')+
            ' and tail accuracy by '+effect(select(records,'deep'),'Router','worst20_accuracy')+
            ' relative to Router-only. Against AvgM+Gate, the mean AUC, final and tail gains are '
            '+.658, +.330 and +.536 points, respectively. These means do not establish significance '
            'without the corresponding paired seed records.'),
        'tab:r1-endtoend':(
            'For the RGB CNN, the paired AUC effect is '+effect(select(records,'cnn'),'Router')+
            ' and the tail effect is '+effect(select(records,'cnn'),'Router','worst20_accuracy')+
            '. These architecture-specific observations are distinct from the fixed-feature estimand.'),
    }
    for label in ['tab:field-sensitivity']:
        for source in ['CIFAR-10','Office-Home']:
            values=[float(row[2]) for row in tables[label]['rows'] if row[0]==source]
            comments[label]=comments.get(label,'')+(
                f' {source} AUC changes range from {min(values):+.3f} to {max(values):+.3f} points '
                'over the full perturbation grid.')
    for label in ['tab:field-calibration','tab:field-shift']:
        entry=tables[label]
        text=[]
        for group in dict.fromkeys(row[0] for row in entry['rows']):
            rows={row[1]:row for row in entry['rows'] if row[0]==group}
            full,router=rows['FedPERG'],rows['Router-only']
            text.append(f'{group} FedPERG minus Router-only is '
                        f'{float(full[2])-float(router[2]):+.3f} AUC points and '
                        f'{float(full[3])-float(router[3]):+.3f} tail points.')
        comments[label]=' '.join(text)+' Label access and population bias therefore affect the metric-specific comparison.'
    for label in ['tab:field-participation','tab:field-quality']:
        column=2 if label=='tab:field-participation' else 5
        values=[float(row[column]) for row in tables[label]['rows']]
        comments[label]=(f'The condition-wise Full-minus-Router AUC effects range from '
                         f'{min(values):+.3f} to {max(values):+.3f} points. '
                         'This range describes the preset conditions; it is not a worst-case robustness bound.')
    def interpreted(parts):
        output=[]
        for part in parts:
            if part.startswith(r'Table~\ref{tab:field-quality}'):
                output.append(comments['tab:field-quality'])
            output.append(part)
            for label,text in comments.items():
                if label != 'tab:field-quality' and rf'\label{{{label}}}' in part:output.append(text)
        return output
    main,additional=interpreted(main),interpreted(additional)
    main.insert(3, 'In tables with comparable method rows, bold marks the best value and underlining marks the second best. Table~\\ref{tab:r1-endtoend} uses bold only. Signs and uncertainty are preserved for effect estimates; they are not ranked as competing algorithms.')
    write_section('optimized_results', '\n\n'.join(main)+'\n')
    write_section('additional_results', '\n\n'.join(additional)+'\n')
    discussion = r'''\subsection{Computational cost and practical value}
\label{sec:discussion-cost}
The additional server computation is a material cost of evidence-guided fusion. The isolated benchmark takes 5.46 s for FedPERG and 3.25 s for AvgM+Gate, with time AUC of 45.103\% and 50.655\%, respectively (Table~\ref{tab:runtime}). Thus, a gain per communication round does not imply a gain per elapsed second. Cached cohort sums reduce leave-one construction to $O(KP)$, and spatial and temporal fitting share one gradient, but the seven candidate forwards still incur overhead. Equation~(\ref{eq:break-even}) makes the trade-off explicit: at the 60\% target, a common additional per-round delay above 178.78 ms offsets the measured crossing-time cost in all three paired runs, whereas a delay below 40.34 ms offsets it in none (Table~\ref{tab:break-even}). These thresholds assume unchanged trajectories and no communication--computation overlap. They characterize conditional practical value, rather than a measured deployment speedup. Sharing evaluations of identical candidates is a possible efficiency improvement; the reported attribution controls retain the matched seven-forward budget.

\subsection{Mechanism attribution and theoretical scope}
\label{sec:discussion-mechanism}
The evidence supports the jointly fitted architecture more strongly than a universal benefit from each prior component. For example, Full minus No residual has a non-digit AUC contrast of +0.016 points with a 95\% interval of $[-0.008,+0.028]$, and a natural-domain contrast of +0.002 points with interval $[-0.003,+0.006]$ (Table~\ref{tab:field-attribution}). These intervals include zero. Target exclusion is therefore a structural property whose independent performance benefit remains uncertain. Local loss improvement is also affected by effort and task difficulty; robust standardization does not make it an unbiased contribution estimate. The work/difficulty and label-quality interventions test this proxy directly (Tables~\ref{tab:r1-reward} and~\ref{tab:field-quality}).

The guarantees describe controlled actions and conditional risks. Positive coordinate gates cannot recover a direction absent from the current aggregate and do not replace local-work normalization (Section~\ref{sec:gate-scope}). The joint certificate concerns matched candidates at one state, while the trajectory result retains work, sampling, drift and temporal errors. Neither orders arbitrary independent training trajectories. The population fallback additionally depends on representative routing data and tail assumptions; its adaptive-reuse bound can be loose at the implemented pool size (Section~\ref{sec:bound-practice}). The empirical gains consequently require experimental evidence in addition to the theoretical statements.

\subsection{Calibration and evaluation scope}
\label{sec:discussion-scope}
FedPERG requires individual client updates and an informative fitting/routing population. Decentralized labeled calibration adds model and gradient traffic, and the unlabeled variant controls an entropy objective rather than classification error (Table~\ref{tab:field-calibration}). Predictive reliability also has a separate trade-off: on CIFAR-10, FedPERG has raw ECE of 20.929\%, reduced to 16.790\% by independent temperature fitting, so its accuracy advantage does not establish superior calibration (Table~\ref{tab:field-confidence}). Temperature fitting preserves class predictions but need not improve ECE or log loss on every source.

Sensitivity and transfer remain bounded by the tested populations. Halving or increasing the gate span by 50\% changes Office-Home AUC by $-1.216$ or $+1.017$ points, respectively; the sensitivity study covers CIFAR-10 and Office-Home only (Table~\ref{tab:field-sensitivity}). The primary matrix uses fixed representations, and fully trainable CNN and ResNet-18 tests use Office-Home (Tables~\ref{tab:r1-endtoend} and~\ref{tab:field-deep}). The natural-domain study introduces unequal work, dropout and domain migration in software, but does not validate physical heterogeneous devices, real network links or independently operated institutions (Tables~\ref{tab:field-natural} and~\ref{tab:field-resources}). Broader end-to-end tasks, independent calibration populations and deployment measurements are needed to establish transfer beyond these settings.

\section{Conclusion}
\label{sec:conclusion}
'''
    conclusion = (
        'FedPERG connects heterogeneous source evidence to the spatial and temporal actions it controls. '
        'One held-out gradient fits a bounded coordinate field and contracts harmful historical velocity; '
        'a separate split compares the resulting operators with a paired Router bank. Joint first-order '
        'certificates and the exact matched displacement identify what changes, while the trajectory bound '
        'retains work, sampling, drift, noise and temporal errors. These conditional statements do not '
        'prove universal test dominance.\n\n'
        'Across the nine-setting matrix, FedPERG leads the mean convergence AUC in every setting. '
        f'The primary non-digit AUC contrast against Router-only is {effect(primary,"Router")}, '
        f'and the natural-domain contrast is {effect(natural,"Router")}. '
        'In ResNet-18, mean AUC, final and tail accuracy exceed Router-only and AvgM+Gate. '
        'Component contrasts, available seed outcomes, confidence errors and isolated computation costs '
        'bound the scope of these aggregate gains. Future work will target lower-cost candidate evaluation '
        'and evidence fitting that transfers across independent calibration populations.\n')
    write_section('optimized_conclusion', discussion + conclusion)


def main():
    records = load()
    tables = build(records)
    figures_setup()
    indexed = {(r['config']['dataset'],r['config']['regime'],r['config']['seed'],r['_arm']):r
               for r in select(records,'matrix')}
    assert len(indexed)==270
    convergence(indexed)
    performance_figure(indexed)
    attribution(records)
    calibration_and_cost(records)
    write_results(records,tables)
    from fedperg.paths import RESULTS
    for name in ['Abstract.pdf','Architecture.pdf']:
        if (PAPER/'figures'/name).is_file():
            shutil.copy2(PAPER/'figures'/name,RESULTS/'figures'/name)
    print('Generated 24 experimental tables and four composite figures from CUDA records.')


if __name__=='__main__':
    main()
