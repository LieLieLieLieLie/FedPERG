"""FedCANTO research implementation.

The trainer is imported lazily so result-audit and plotting utilities do not
initialise CUDA/PyTorch merely to read configuration constants.
"""

from .config import ExperimentConfig

__all__ = ["ExperimentConfig", "FederatedExperiment"]


def __getattr__(name: str):
    if name == "FederatedExperiment":
        from .trainer import FederatedExperiment
        return FederatedExperiment
    raise AttributeError(name)
