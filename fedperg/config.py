from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    regime: str = "compound"
    method: str = "FedPERG"
    seed: int = 0
    clients: int = 20
    clients_per_round: int = 8
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 128
    hidden_dim: int = 64
    architecture: str = "shallow"
    local_lr: float = 0.06
    server_lr: float = 1.0
    weight_decay: float = 1e-4
    dirichlet_alpha: float = 0.2
    eval_every: int = 2
    device: str = "cuda"
    perg_hidden: int = 48
    perg_heads: int = 4
    perg_meta_steps: int = 4
    perg_lr: float = 3e-3
    perg_rho: float = 0.30
    perg_temperature: float = 0.8
    perg_reconstruction_weight: float = 0.85
    perg_median_weight: float = 1.20
    perg_memory_weight: float = 0.40
    perg_gate_low: float = 0.82
    perg_gate_span: float = 0.32
    perg_variant: str = "full"
    attack_fraction: float = 0.0
    attack_type: str = "none"
    experiment_tag: str = ""
    calibration_fraction: float = 0.04
    specialization: float = 0.0
    client_validation_fraction: float = 0.0
    selector_margin: float = 0.0
    selector_calibration: bool = False
    selector_calibration_shift: str = "none"
    selector_test_diagnostics: bool = False
    data_dir: str = "data"
    results_dir: str = "results"

    def resolved(self, root: Path) -> "ExperimentConfig":
        clone = ExperimentConfig(**asdict(self))
        clone.data_dir = str((root / self.data_dir).resolve())
        clone.results_dir = str((root / self.results_dir).resolve())
        if clone.dataset == "cifar100":
            clone.local_lr = min(clone.local_lr, 0.045)
            clone.hidden_dim = max(clone.hidden_dim, 96)
        if clone.dataset == "officehome":
            clone.local_lr = min(clone.local_lr, 0.04)
            clone.hidden_dim = max(clone.hidden_dim, 96)
        if clone.dataset == "eurosat_resnet18":
            clone.local_lr = min(clone.local_lr, 0.05)
            clone.hidden_dim = max(clone.hidden_dim, 96)
        return clone

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


METHODS = [
    "FedAvg",
    "FedAvgM",
    "FedProx",
    "SCAFFOLD",
    "FedNova",
    "FedDyn",
    "FedAdam",
    "FedLAW",
    "FedCDA",
    "FedPW",
    "Fed-NGA",
    "FedPhoenix",
    "FedPERG",
]

DATASETS = ["cifar10", "cifar100", "officehome", "eurosat_resnet18",
            "synthetic_complementarity"]
REGIMES = ["label_skew", "quantity_skew", "compound"]
