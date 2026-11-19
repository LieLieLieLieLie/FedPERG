# Frozen Primary Matrix Protocol

The primary breadth matrix compares FedPERG with FedAvg, FedAvgM, SCAFFOLD,
FedAdam, FedLAW, FedCDA, FedPW, Fed-NGA, and FedPhoenix. CIFAR-10, CIFAR-100,
and Office-Home are evaluated under label, quantity, and compound heterogeneity.
Every cell uses 20 clients, 8 participants per round, 30 rounds, one local epoch,
and paired seeds 20--22. A seed-determined 4% server calibration split is removed
from every client's local data for all methods. FedPERG uses the frozen
`lite_paired_gate_bank_selector` variant; no result from this matrix changes its
coefficients, candidates, margin, or evaluation rule.

