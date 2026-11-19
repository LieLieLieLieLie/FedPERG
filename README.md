# FedPERG

**FedPERG: Auditable Paired Evidence-Routed Gate Fusion under Federated Heterogeneity**

This repository separates the original submitted implementation from the current revision so that reviewers can select the relevant version.

## Versions

- [`R0_0/`](R0_0/): the complete repository snapshot uploaded for the initial submission, preserved without changing its file contents.
- `R0/`: a reserved version directory, currently containing only a placeholder. It is not the revised implementation.
- [`R1/`](R1/): the revised implementation, experiment protocols, tests and reporting code corresponding to the current revision. Start here to reproduce the revised study.

The original snapshot remains available in Git history as well. Its historical README describes that snapshot; use the R1 instructions for the revised study.

## Reproduce R1

```bash
git clone https://github.com/LieLieLieLieLie/FedPERG.git
cd FedPERG/R1
python -m pip install -r requirements.txt
python reproduce.py --stage all --device cuda
```

Prepare the datasets before starting experiments. The detailed [`R1 README`](R1/README.md) gives the CUDA environment, official download links, data-cache preparation, development and confirmation seeds, frozen parameters, baseline provenance and individual experiment commands. In the versioned layout, data caches are prepared under `R0/data/`; no R0 implementation is needed to create these caches.

The revised implementation uses a held-out gradient to jointly fit coordinate confidence and temporal retention, then evaluates a span-matched paired candidate bank on a disjoint routing split. The reporting scripts generate PDF figures and CSV tables in `R1/results/`; a manuscript checkout is optional.

## Data and generated outputs

Third-party datasets, derived data caches, `results/` directories, trained checkpoints and machine-specific caches are not uploaded. Download CIFAR-10/100 and their corruption archives, and Office-Home, from the official sources linked in the R1 README. Reproduction creates fresh experiment records and figures locally. The local revision's optional cohort-level summaries and archived models are not distributed here; fresh runs are evaluated from their own records.

## Use and attribution

See [`LICENSE`](LICENSE) for the academic-evaluation terms and attribution requirements. Do not present the code, method, figures or results as your own work. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
