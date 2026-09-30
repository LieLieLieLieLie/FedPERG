from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn


class FeatureMLP(nn.Module):
    """Compact nonlinear classifier used on fixed, public pretrained features."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, classes)

    def forward(self, x: Tensor, return_features: bool = False):
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return (logits, h) if return_features else logits


class ResidualFeatureNet(nn.Module):
    """A 20-tensor trainable residual head for an architecture-scale control.

    This still consumes fixed features; it is not a raw-image CNN experiment.
    """

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({"linear": nn.Linear(hidden_dim, hidden_dim),
                           "norm": nn.LayerNorm(hidden_dim)}) for _ in range(4)
        ])
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, x: Tensor, return_features: bool = False):
        h = torch.nn.functional.gelu(self.stem(x))
        for block in self.blocks:
            h = h + 0.25 * torch.nn.functional.gelu(block["norm"](block["linear"](h)))
        logits = self.head(h)
        return (logits, h) if return_features else logits


class RawDigitCNN(nn.Module):
    """Twenty trainable tensors; all representation layers train on raw pixels."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, 16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 32)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm4 = nn.GroupNorm(8, 32)
        self.fc = nn.Linear(32 * 7 * 7, 64)
        self.head = nn.Linear(64, classes)

    def forward(self, x: Tensor, return_features: bool = False):
        x = torch.nn.functional.gelu(self.norm1(self.conv1(x)))
        x = torch.nn.functional.gelu(self.norm2(self.conv2(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.gelu(self.norm3(self.conv3(x)))
        x = torch.nn.functional.gelu(self.norm4(self.conv4(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        h = torch.nn.functional.gelu(self.fc(x.flatten(1)))
        logits = self.head(h)
        return (logits, h) if return_features else logits


class RawRGBCNN(nn.Module):
    """Twenty trainable tensors for end-to-end 48x48 RGB classification."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm4 = nn.GroupNorm(8, 64)
        self.fc = nn.Linear(64 * 6 * 6, 128)
        self.head = nn.Linear(128, classes)

    def forward(self, x: Tensor, return_features: bool = False):
        x = torch.nn.functional.gelu(self.norm1(self.conv1(x)))
        x = torch.nn.functional.gelu(self.norm2(self.conv2(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.gelu(self.norm3(self.conv3(x)))
        x = torch.nn.functional.gelu(self.norm4(self.conv4(x)))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (6, 6))
        h = torch.nn.functional.gelu(self.fc(x.flatten(1)))
        logits = self.head(h)
        return (logits, h) if return_features else logits


def state_names(model: nn.Module) -> List[str]:
    return [name for name, _ in model.named_parameters()]


def clone_state(model: nn.Module) -> OrderedDict[str, Tensor]:
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())


def state_delta(local: Mapping[str, Tensor], global_state: Mapping[str, Tensor]) -> List[Tensor]:
    return [(local[k] - global_state[k]).detach() for k in global_state]


def add_delta_(model: nn.Module, delta: Sequence[Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for param, step in zip(model.parameters(), delta):
            param.add_(step, alpha=scale)


def flatten_delta(delta: Sequence[Tensor]) -> Tensor:
    return torch.cat([x.reshape(-1) for x in delta])


def unflatten_like(vector: Tensor, reference: Sequence[Tensor]) -> List[Tensor]:
    result: List[Tensor] = []
    offset = 0
    for ref in reference:
        count = ref.numel()
        result.append(vector[offset : offset + count].reshape_as(ref))
        offset += count
    if offset != vector.numel():
        raise ValueError("Vector length does not match reference tensors")
    return result


def zeros_like_delta(model: nn.Module) -> List[Tensor]:
    return [torch.zeros_like(p) for p in model.parameters()]


def weighted_delta(deltas: Sequence[Sequence[Tensor]], weights: Tensor) -> List[Tensor]:
    if len(deltas) == 0:
        raise ValueError("Cannot aggregate an empty delta set")
    output = [torch.zeros_like(x) for x in deltas[0]]
    for i, client_delta in enumerate(deltas):
        for layer, value in enumerate(client_delta):
            output[layer].add_(value, alpha=float(weights[i]))
    return output
