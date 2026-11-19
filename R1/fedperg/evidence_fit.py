"""Projected fitting operators for the spatial and temporal evidence actions."""
from typing import Sequence, Tuple

import torch
from torch import Tensor


def fit_momentum(
    gradients: Sequence[Tensor], velocity: Sequence[Tensor], step: float,
    random_input: bool = False, seed: int = 0, round_idx: int = 0,
) -> Tuple[float, float]:
    """Project the momentum coefficient into [max(0, .9-step), .9]."""
    if not 0 <= step <= .9:
        raise ValueError('Momentum fitting step must lie between zero and .9')
    derivative = torch.zeros((), device=gradients[0].device)
    for layer, (gradient, previous) in enumerate(zip(gradients, velocity)):
        if random_input:
            generator = torch.Generator(device=gradient.device).manual_seed(
                7193 * seed + round_idx * 101 + layer)
            order = torch.randperm(gradient.numel(), generator=generator, device=gradient.device)
            gradient = gradient.flatten()[order].reshape_as(gradient)
        derivative = derivative + (gradient * previous).sum()
    value = float(derivative)
    if value <= 0 or step == 0:
        return .9, 0.
    coefficient = max(0., .9 - step * value / max(value, 1e-12))
    return coefficient, value * (coefficient - .9)
