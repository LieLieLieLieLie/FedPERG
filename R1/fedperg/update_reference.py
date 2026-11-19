"""Construct client updates with a common effective local-work scale."""
from typing import List, Sequence

import torch
from torch import Tensor


def normalize_local_work(
    deltas: Sequence[Sequence[Tensor]], masses: Tensor, steps: Tensor
) -> List[List[Tensor]]:
    """Match FedNova's effective-work reference before coordinate gating."""
    if len(deltas) != len(masses) or len(deltas) != len(steps):
        raise ValueError('Updates, sample masses and local step counts must align')
    if torch.any(steps <= 0):
        raise ValueError('Local-work normalization requires positive step counts')
    effective_steps = torch.dot(masses, steps)
    return [
        [tensor * (effective_steps / count) for tensor in delta]
        for delta, count in zip(deltas, steps)
    ]
