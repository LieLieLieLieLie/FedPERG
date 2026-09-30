# FedCANTO

Official research implementation of **FedCANTO: Paired Evidence-Routed Gate
Fusion for Heterogeneous Federated Learning**. FedCANTO constructs a
client--tensor evidence map and uses a fixed-margin router to choose among
span-matched, sign-preserving gated aggregates.

This repository contains the frozen implementation and experiment scripts used
for the submitted manuscript. Third-party datasets, derived feature caches, and
generated results are intentionally not redistributed.

## Environment

The reported experiments were executed with Python 3.9, PyTorch 2.0, and an
NVIDIA CUDA GPU. A clean environment can be prepared as follows:

```bash
conda create -n fedcanto python=3.9 -y
conda activate fedcanto
pip install -r requirements.txt
```

CUDA is the default for experiment scripts. Use `--device cpu` only for a
functional check; the complete matrix is intended for GPU execution. Random
seeds, client counts, participation rates, communication rounds, heterogeneity
settings, and method-specific frozen parameters are defined in the respective
`run_*.py` scripts and protocol files.

## Data download and feature preparation

Obtain the raw datasets only from their official sources:

- [CIFAR-10 and CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-10-C](https://zenodo.org/records/2535967) and
  [CIFAR-100-C](https://zenodo.org/records/3555552)
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)
- [MNIST](https://yann.lecun.com/exdb/mnist/)

`prepare_data.py` downloads the clean CIFAR training sets and MNIST through
torchvision and constructs the frozen ImageNet-backbone feature caches expected
by the experiments. Download and extract CIFAR-10-C and CIFAR-100-C separately;
their multi-gigabyte archives are not fetched implicitly. Office-Home must be
downloaded and extracted so that the given directory contains `Art`, `Clipart`,
`Product`, and `Real World`.

```bash
python prepare_data.py --dataset cifar10 --cifar-c-root "D:/datasets" --device cuda
python prepare_data.py --dataset cifar100 --cifar-c-root "D:/datasets" --device cuda
python prepare_data.py --dataset mnist --device cuda
python prepare_data.py --dataset officehome \
  --officehome-raw "D:/datasets/OfficeHomeDataset_10072016" --device cuda
```

The resulting files are written below `data/`:

```text
data/
├── cifar10_resnet18_ftta_v1.npz
├── cifar100_resnet18_ftta_v1.npz
├── officehome65_resnet18.npz
├── officehome_raw_48.npz
└── mnist_mobilenet_v3_64.npz
```

The extraction uses torchvision's published ImageNet weights and deterministic,
class-balanced indices. The six CIFAR-C target episodes are Gaussian noise,
motion blur, contrast, brightness, pixelation, and JPEG compression at severity
levels 1, 2, 3, 4, 5, and 3, respectively. CIFAR-10 uses 500 training and 200
test examples per class; CIFAR-100 uses its full 500/100 per-class split.
Dataset images and generated `.npz` caches remain excluded by `.gitignore`;
redistribution rights remain with the original dataset owners.

## Reproducing the experiments

Run the invariant tests first:

```bash
pytest -q
```

The frozen experiment groups can then be executed from the repository root:

```bash
# Ten-method, three-dataset, three-regime comparison (seeds 20--22)
python reproduce.py --stage primary --device cuda

# Paired AvgM+Gate comparison, operator attribution, and residual ablation
python reproduce.py --stage attribution --device cuda

# Calibration-shift and end-to-end/raw-input checks
python reproduce.py --stage robustness --device cuda

# Prospectively frozen four-candidate development audit
python reproduce.py --stage development --device cuda

# Statistical summaries, tables, and publication PDF figures
python reproduce.py --stage report --device cuda
```

`python reproduce.py --stage all --device cuda` runs all groups in the order
above. Completed run records are skipped by the individual runners, so an
interrupted study can be resumed. JSON run records are written to
`results/models/`, tabular summaries to `results/tables/`, and figures to
`results/figures/`. These generated artifacts are not committed.

The primary matrix uses 20 clients, 8 participating clients per round, 30
communication rounds, one local epoch, and three paired seeds per condition.
The confirmatory gate/operator attribution uses untouched seeds 50--59. See the
frozen protocol documents for the precise estimands and interpretation limits.

## Repository structure

```text
fedcanto/          core models, aggregation operators, data partitions, trainer
tests/             permutation-equivariance and masking invariants
protocols/         descriptive frozen protocols for the reported experiments
prepare_data.py    official-data download and feature-cache construction
reproduce.py       grouped reproduction entry point
run_*.py           experiment runners
summarize_*.py     statistical summaries and table builders
analyze_and_plot.py publication figure generation
audit_results.py   result-integrity and paired-design audit
```

## License and research integrity

The code is released under the **FedCANTO Academic Evaluation License 1.0** for
non-commercial evaluation and result reproduction during manuscript review. It
is source-available, not OSI-approved open-source software. Reuse requires clear
attribution; redistribution, rebranding, commercial use, or presenting the
code, algorithms, experimental material, or substantial derivatives as one's
own work is prohibited without written permission. See [LICENSE](LICENSE).
