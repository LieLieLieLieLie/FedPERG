# FedPERG R1

Implementation of **FedPERG: Auditable Paired Evidence-Routed Gate Fusion under Federated Heterogeneity**.
The current method uses one held-out gradient to fit coordinate confidence and
historical-velocity retention, then routes a seven-candidate paired bank on a
disjoint pool. The code and manuscript use the same frozen configuration.

## Environment

All reported development and confirmation training uses CUDA on an NVIDIA
RTX 4080. The local executable is
`D:/anaconda3/envs/pytorch2.0.0_py3.9/python.exe` (Python 3.9,
PyTorch 2.0.0+cu118). The experiment entry points require CUDA and fail explicitly
when it is unavailable. They do not silently fall back to CPU.

```powershell
$py = 'D:/anaconda3/envs/pytorch2.0.0_py3.9/python.exe'
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $py -m pytest -q
```

The unit tests check mathematical/implementation invariants and are separate
from the GPU experiments. Dependencies are listed in `requirements.txt`.
The publication plotting code requires Times New Roman to be installed.

## Layout

```text
fedperg/                       aggregation, datasets, models, training, metrics
experiments/optimization/       development grid and all formal experiment suites
reporting/                     paired statistics, table registry and PDF figures
checks/verify_gpu.py            completeness, CUDA and checkpoint verification
tests/                         algorithmic and numerical invariants
results/
  figures/                     publication figures, PDF only
  tables/                      CSV experimental summaries
  models/
    development/{initial,robust,temporal}/
    formal/<suite>/            completed JSON, model and operator PTH files
    reviewer_baselines/       54 matched PA3Fed and Fed-LAA extension runs
    protocols/                frozen selection, hashes and verification manifests
    logs/                     resumable queue state and training logs
prepare_data.py
reproduce.py
```

Superseded CPU experiments, their results and obsolete reporting entry points
are excluded. In the local manuscript workspace, `../R0` is an untouched source
archive whose data caches are shared read-only; all R1 outputs stay under R1.
In the public repository, `R0/` is reserved and initially empty. Prepare its
`data/` cache from the official sources below before running experiments.

## Reproduction

Run from this directory with the executable above:

```powershell
& $py reproduce.py --stage development --device cuda
& $py reproduce.py --stage experiments --device cuda
& $py reproduce.py --stage report
```

Use `--dry-run` to inspect commands without running them. `--stage all` runs all
three stages. Complete JSON records are skipped on restart. The model and
operator checkpoint are atomically committed before the completion JSON.
Do not run two copies of the same suite concurrently.

The experiment coordinator runs at most four independent groups, then runs
the isolated timing suite alone. Start it only when other GPU tasks are idle.
Concurrent training wall times are recorded but excluded from speed rankings.
If GPU memory is limited, run the individual groups sequentially:

```powershell
& $py -m experiments.optimization.queue --group gpu_features
& $py -m experiments.optimization.queue --group deep_evidence
& $py -m experiments.optimization.queue --group deep_controls
& $py -m experiments.optimization.queue --group gpu_attribution
& $py -m experiments.optimization.queue --group gpu_robustness
& $py -m experiments.optimization.queue --group regimes_reward
& $py -m experiments.optimization.queue --group quality_cnn
& $py -m experiments.optimization.queue --group timing
& $py -m experiments.optimization.reviewer_baselines
```

Reporting writes PDF figures, CSV tables and generated section text under
`results/`. If the sibling manuscript `../../paper/INS/R1/paper.tex` is present,
it also synchronizes its figures and generated sections. The standalone GitHub
checkout does not require manuscript files. Model records alone suffice for most plots;
the reliability/quantile figure also reloads saved models on CUDA and caches
test logits. No training rerun is required to change colors or layout.

## Frozen protocol

Development uses seeds 200--202 and an independent 4% validation pool excluded
from local SGD and both aggregation splits. The first grid has 72 runs, followed
by two 81-run clean-validation grids. Fifteen unique clean-validation
configurations are ranked equally on validation AUC, final accuracy, tail
accuracy and ECE across three contexts. The selected joint configuration uses
spatial step .3, trust radius .5, prior strength .25, temporal step .3,
CE plus half Brier, and routing margin .001. The final grid, selection and hashes
are retained. Test metrics are not used to select it.

Post-selection evaluation has **1,469 GPU runs**, including 240 sensitivity
diagnostics and 1,229 other evaluation runs. The 234 development runs are
separate, giving 1,703 recorded training runs in total.

| Suite | Runs | Design |
|---|---:|---|
| matrix | 270 | 3 sources × 3 regimes × 3 seeds × 10 methods |
| natural | 160 | 10 seeds × 10 primary methods and 6 matched controls |
| attribution | 300 | 210 compound runs plus 90 label/quantity runs |
| stress | 100 | 2 sources × 5 conditions × 5 seeds × 2 arms |
| quality | 120 | label faults, raw/zero reward and matched controls |
| reward | 120 | independent effort/difficulty and 4 reward definitions |
| sensitivity | 240 | 2 sources × 12 parameters × 2 factors × 5 seeds |
| calibration | 45 | 3 label-access modes × 3 arms × 5 seeds |
| shift | 15 | biased pool × 3 arms × 5 seeds |
| deep | 20 | trainable ResNet-18, full local epochs, 4 arms × 5 seeds |
| cnn | 10 | trainable RGB CNN, 2 arms × 5 seeds |
| timing | 15 | isolated benchmark, 5 arms × 3 seeds |
| reviewer baselines | 54 | 3 sources × 3 regimes × 3 seeds × 2 additional methods |

The matrix uses seeds 220--222 and 60 rounds. Primary compound attribution and
natural domains use seeds 220--229. Other studies use 220--224 unless stated.
Both trainable image models run 100 rounds. All results use the final iterate;
there is no confirmation-dependent checkpoint selection or terminal averaging.

The primary ten methods are FedAvgM, SCAFFOLD, FedNova, FedLBW, FedLAW, FedLWS,
FedAWA, FedDS, AvgM+Gate and FedPERG. See `BASELINE_PROVENANCE.md` for defining
mechanisms, proxy access and differences from complete author protocols.
Router-only, Spatial-only, Temporal-only, Random input, Permuted field and
No residual are component controls, not additional primary baselines.
PA3Fed and Fed-LAA extend the same matrix in the appendix.
Their runner reuses the exact saved FedPERG matrix configurations and seeds.
The report checks paired client schedules, client sizes, local steps, evaluation
rounds and initial accuracy. See `results/tables/reviewer_baselines_paired_runs.csv`
for seed-level AUC, final accuracy, tail accuracy and ECE for all four methods
in the requested comparison, including FedLBW and FedPERG.

Convergence AUC is round-normalized test accuracy area; tail accuracy averages
the worst 20% of client test groups. Raw ECE uses 15 bins. Composite performance
is the equal average of AUC/final/tail/ECE ranks, respecting lower-is-better ECE.
Individual results, failures and uncertainty remain available; a composite win
does not imply a win on every metric. Bootstrap and exact sign-flip units are
paired seeds, with sources averaged within each seed.

Centralized/distributed labeled aggregation uses CE+Brier. Unlabeled aggregation
uses predictive entropy minus half marginal entropy and does not use labels for
fitting/routing. Independent post-training temperature diagnostics still require
labels on a separate 4% pool. They do not change training, predictions or raw
classification accuracy. Modeled client/upload time is a software cost model,
distinct from isolated measured GPU aggregation latency.

## Development transparency and measured cost

The full fifteen-configuration clean development grid is described in the
manuscript's Appendix B.2 and recorded in
`results/models/protocols/development_grid_details.json`. Its 135 distinct
observations share seeds 200–202 across three contexts. The 234 saved development
runs include the initial 72-run study and 27 repeated clean-grid observations.
Table 19 uses seeds 220–224 as a diagnostic subset of confirmation. It does not
retune the selected default and covers CIFAR-10 and Office-Home.
Its 240 runs are counted once within the 1,469 evaluations and excluded from the
234 development runs. The design has 48 perturbation conditions, each with five
paired seeds. Ten unchanged-default records are reused from the attribution
suite and receive no additional run count. Seed and default reuse across
conditions does not create 240 independent replicates.

The added-delay calculation in Table 25 reads the original isolated timing
histories and stores its inputs in `results/models/protocols/break_even_details.json`.
It uses the first evaluated target round and elapsed simulation time through
that round, including local SGD, aggregation, evaluation and loop overhead.
The critical delay `(T_P - T_B) / (R_B - R_P)` assumes identical extra delay
per communication round and no overlap with computation. Censored targets
receive no threshold. These summaries require no retraining.

The manuscript uses only `paper.tex` and `paper_highlighted.tex`. Reporting
updates the generated experiment and development blocks directly inside
`paper.tex`; it does not create separate chapter `.tex` files. Keep the short
`BEGIN/END FEDPERG SECTION` markers used by `reporting/manuscript_sections.py`.

The repository root distinguishes the original uploaded snapshot (`R0_0/`),
the reserved `R0/` directory, and this revision (`R1/`). Run the commands in
this README from `R1/`; the root README explains the version layout.

The report also reads cohort-level means and contrast summaries from
`results/models/protocols/summary_inputs.json`. These inputs remain separate
from the per-run JSON files and checkpoints. Means can update table cells and
derived mean contrasts; they do not determine a per-seed distribution, paired
confidence interval, exact p-value or a new training trajectory. The reporting
commands omit unavailable statistics instead of shifting existing samples or
intervals. Figure 3 retains available trajectories and reports tabulated final
means with terminal markers; Figure 5 uses only available seed distributions.
The summaries are anchored to SHA-256 hashes of the archived run files. They
are not substituted for results of a fresh training run whose records differ.

## Data preparation

Download datasets from their official sources:

- [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-10-C](https://zenodo.org/records/2535967) and [CIFAR-100-C](https://zenodo.org/records/3555552)
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)

`prepare_data.py` uses torchvision's published ImageNet ResNet-18 weights for
feature extraction. Office-Home must be downloaded and extracted with domain
folders `Art`, `Clipart`, `Product`, and `Real World`. CIFAR corruption archives
are downloaded separately. The repository does not redistribute third-party
data, raw images or feature caches. In the local versioned layout the expected
cache directory is `../R0/data`; a fresh reproduction can create that cache
directory without requiring R0 code.

```powershell
& $py prepare_data.py --data-dir ../R0/data --dataset cifar10 --cifar-c-root D:/datasets --device cuda
& $py prepare_data.py --data-dir ../R0/data --dataset cifar100 --cifar-c-root D:/datasets --device cuda
& $py prepare_data.py --data-dir ../R0/data --dataset officehome --officehome-raw D:/datasets/OfficeHomeDataset_10072016 --device cuda
```

Required caches are `cifar10_resnet18_ftta_v1.npz`,
`cifar100_resnet18_ftta_v1.npz`, `officehome65_resnet18.npz` and
`officehome_raw_48.npz`. CIFAR-10 confirmation uses the predefined balanced
5,000/2,000 subset; CIFAR-100 uses 50,000/10,000. Office-Home uses its deterministic
train/test split. These are controlled aggregation studies rather than a claim
to full-resolution image-classification state of the art.

The public repository excludes `data/`, `results/`, checkpoints and local caches.
Reproduction creates new results from the published frozen settings. The optional
cohort-level summaries described above belong to the local revision records;
they are not part of the public upload and are not required for fresh runs.
See `LICENSE` for the repository's academic-evaluation terms and attribution
requirements.
