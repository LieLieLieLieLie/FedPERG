# Frozen Confirmatory Attribution Protocol

This protocol was fixed before seeds 50--59 were executed. The sources are
CIFAR-10, CIFAR-100, and Office-Home under all three heterogeneity regimes plus
frozen MobileNetV3/MNIST under compound heterogeneity. AvgM+Gate, Router-only,
FedCANTO, and the leave-one-residual ablation share data, client sampling,
local optimization, incoming momentum, seven candidate spans, seven calibration
forwards, the 0.001 margin, and the disjoint 4% server calibration pool.

Router-only replaces evidence-weighted agreement with sample-mass agreement in
every paired candidate. The residual ablation changes only the standardized
leave-one-residual coefficient from 0.85 to zero. The primary estimand averages
regimes within each source and then sources equally. All ten paired seed clusters
are retained; uncertainty uses paired seed-cluster bootstrap, t intervals,
exact sign-flip tests, and leave-one-seed/source audits.

