# Baseline mechanisms in the revision

All methods use the same dataset caches, seed-specific client split, held-out
pool removal, invitations, local SGD, and evaluation horizon within a comparison.
Server-side optimization budgets remain method-specific. These are common-harness
mechanism ports, not reproductions of the authors' complete training protocols.
No old heuristic result is relabelled as a new faithful baseline result.

| Method | Primary source | Mechanism retained and scope |
|---|---|---|
| FedNova | NeurIPS 2020, 10.48550/arXiv.2007.07481 | Actual local-SGD step normalization with no local momentum; weighted effective step count. |
| FedLBW | Expert Systems with Applications 302 (2026) 130487, 10.1016/j.eswa.2025.130487; arXiv:2608.07007 | Inverse held-out loss weights, normalized without multiplying by sample mass. Same at-most-512 public examples, no test labels. |
| FedLAW | ICML 2023, ZexiLee/ICML-2023-FedLAW | Joint simplex weights and global model shrinkage, learned on the held-out pool; 10 Adam steps, learning rate .01. The precursor ProxySoftmax heuristic is not FedLAW. |
| FedLWS | ICLR 2025, Shi et al. | Analytic layer-wise shrinkage with adjacent weight/bias tensors as layers in the feature classifier. This port is not used for the GroupNorm ResNet because tensor pairing would differ. |
| FedAWA | CVPR 2025; https://github.com/ChanglongShi/FedAWA | Client-vector dispersion plus cosine model-distance cost; one Adam step, .001, betas (.5,.999), gamma 1. The released last pre-step probabilities form the aggregate. To support varying selected clients, persistent logits are indexed by client identity; the release's selected-slot state is not copied as an identity mapping. Initial logits are first-observed participating sample masses. |
| FedDS | Information Sciences 745 (2026) 123425; https://github.com/Gia99999/FedDS | Majority-prediction initialization, EM confusion matrices, mean diagonal reliability, and reliability-only model weights. Uses the released standard algorithm's uniform latent prior, 500 maximum iterations and tolerance 1e-5. The paper also describes a learned class-prior variant; this result is explicitly the released uniform-prior variant. Log-space E steps and sparse M steps are algebraically equivalent optimizations, tested against independent literal loops. Public inputs only; public labels are not used. |
| SCAFFOLD | ICML 2020 | Each round's clients receive one shared pre-round server control; server control updates are committed without changing that snapshot. |
| FedAvgM | FedAvg with server momentum | Unnormalized server velocity beta .9, matching FedPERG's reference Router path. Evidence candidates fit retention in [.6,.9]; matched Spatial-only/Temporal-only controls identify that difference. |
| AvgM+Gate | Matched architecture control | Fixed .82+.32 sample-mass sign agreement and the same beta .9 velocity. |

The paired extension uses the same nine settings and seeds 220–222 as the
primary matrix. The two additional methods require no server proxy examples.
All held-out exclusions remain in force.

- **PA3Fed** ([AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33912)).
  Algorithm 1 and Eqs. (4)–(6), (8) and (10) define the relative FGN threshold
  of .01 and exponential score adjustment with β=.3. The port evaluates local
  cross-entropy and mean per-example squared gradient norm at the downloaded
  model. It stores each client's last observed score, uses a unit correction
  for new clients, and normalizes corrected sample masses within each cohort.
  It retains the common horizon, feature model and SGD budget. Two extra
  scalars per participating client are counted in upload volume.
- **Fed-LAA** ([Applied Soft Computing 2026](https://doi.org/10.1016/j.asoc.2026.115676)).
  The [published algorithm](https://ars.els-cdn.com/content/image/1-s2.0-S1568494626011245-fx1_lrg.jpg)
  reconstructs each client's gradients from parameter deltas divided by
  its learning rate and local epochs. The reference is their sample-mass
  average. The layer includes its weight and bias, and each client's angle
  starts at zero, persists when the client is absent, and receives the
  specified new-observation weight `1/t` at global round `t`. The server
  computes the sample-mass-weighted Gompertz softmax independently per layer.
  Its parameter β is an input to the pseudocode. The common-harness port
  fixes β=5 using the setting for the same mapping in
  [FedAdp](https://doi.org/10.1109/TCCN.2021.3084406), with denominator
  regularization at `1e-12`. The server uses model deltas already uploaded
  by clients and the scheduled local SGD configuration.

Timing collected during concurrent training is retained for reproducibility but
must not be interpreted as an isolated speed ranking. Dedicated timing and any
bandwidth/throughput model must be identified separately in the manuscript.
