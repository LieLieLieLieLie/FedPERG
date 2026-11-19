"""Numerical-equivalence and synchronized GPU benchmark for cached FedPERG.

The tensor profiles match the fixed-feature classifiers used by the three
primary sources and the 20-tensor raw-MNIST CNN.  Both paths execute the full
FedPERG server aggregation; only the leave-one residual implementation is
swapped.  Results are written under ``results/{tables,models}``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor

import fedperg.aggregators as aggregators
from fedperg.aggregators import PERGAggregator, cached_leave_one_residual


ROOT = Path(__file__).resolve().parent


def direct_leave_one_residual(vectors: Tensor, base_weights: Tensor) -> Tensor:
    """Audited pre-cache implementation retained only for this benchmark."""
    result = torch.zeros(vectors.shape[0], device=vectors.device, dtype=vectors.dtype)
    for client in range(vectors.shape[0]):
        visible = torch.ones(vectors.shape[0], dtype=torch.bool, device=vectors.device)
        visible[client] = False
        visible_weights = base_weights[visible]
        visible_weights = visible_weights / visible_weights.sum()
        reference = (vectors[visible] * visible_weights[:, None]).sum(dim=0)
        denominator = vectors[client].norm() + reference.norm() + 1e-12
        result[client] = ((vectors[client] - reference).norm() / denominator).pow(2)
    return result


def tensor_shapes(profile: str) -> Sequence[Sequence[int]]:
    if profile == "cifar10_fixed":
        return ((64, 512), (64,), (10, 64), (10,))
    if profile == "cifar100_fixed":
        return ((96, 512), (96,), (100, 96), (100,))
    if profile == "officehome_fixed":
        return ((96, 512), (96,), (65, 96), (65,))
    if profile == "mnist_raw":
        return (
            (16, 1, 3, 3), (16,), (16,), (16,),
            (16, 16, 3, 3), (16,), (16,), (16,),
            (32, 16, 3, 3), (32,), (32,), (32,),
            (32, 32, 3, 3), (32,), (32,), (32,),
            (64, 32 * 7 * 7), (64,), (10, 64), (10,),
        )
    raise ValueError(profile)


def make_inputs(profile: str, clients: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(20260927 + clients)
    reference = [torch.zeros(tuple(shape), device=device) for shape in tensor_shapes(profile)]
    deltas = [
        [torch.randn(value.shape, generator=generator, device=device) * 0.02
         for value in reference]
        for _ in range(clients)
    ]
    mass = torch.arange(1, clients + 1, device=device, dtype=torch.float32)
    mass /= mass.sum()
    improvement = torch.linspace(-0.1, 0.1, clients, device=device)
    return reference, deltas, mass, improvement


def aggregate_once(profile: str, clients: int, device: torch.device,
                   residual_fn: Callable[[Tensor, Tensor], Tensor]):
    reference, deltas, mass, improvement = make_inputs(profile, clients, device)
    original = aggregators.cached_leave_one_residual
    aggregators.cached_leave_one_residual = residual_fn
    try:
        torch.manual_seed(20260927)
        server = PERGAggregator(reference, 48, 4, 3e-3, 4, 0.30, device,
                                 variant="lite_simple_residual")
        output = server.aggregate(deltas, mass, improvement, 1)
        server.update_predictor(deltas, mass, improvement, 1)
        return output
    finally:
        aggregators.cached_leave_one_residual = original


def benchmark(profile: str, clients: int, device: torch.device,
              repeats: int = 100, warmup: int = 20) -> Dict[str, object]:
    direct = aggregate_once(profile, clients, device, direct_leave_one_residual)
    cached = aggregate_once(profile, clients, device, cached_leave_one_residual)
    max_residual_error = float((direct.reconstruction_error - cached.reconstruction_error).abs().max())
    max_weight_error = float((direct.weights - cached.weights).abs().max())
    max_delta_error = max(float((a - b).abs().max()) for a, b in zip(direct.delta, cached.delta))

    timings: Dict[str, List[float]] = {"direct": [], "cached": []}
    for label, residual_fn in (("direct", direct_leave_one_residual),
                               ("cached", cached_leave_one_residual)):
        reference, deltas, mass, improvement = make_inputs(profile, clients, device)
        original = aggregators.cached_leave_one_residual
        aggregators.cached_leave_one_residual = residual_fn
        try:
            torch.manual_seed(20260927)
            server = PERGAggregator(reference, 48, 4, 3e-3, 4, 0.30, device,
                                     variant="lite_simple_residual")
            for step in range(warmup + repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                server.aggregate(deltas, mass, improvement, step + 1)
                server.update_predictor(deltas, mass, improvement, step + 1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = 1000.0 * (time.perf_counter() - started)
                if step >= warmup:
                    timings[label].append(elapsed_ms)
        finally:
            aggregators.cached_leave_one_residual = original

    direct_mean = float(np.mean(timings["direct"]))
    cached_mean = float(np.mean(timings["cached"]))
    return {
        "profile": profile,
        "clients": clients,
        "tensors": len(tensor_shapes(profile)),
        "parameters": int(sum(np.prod(shape) for shape in tensor_shapes(profile))),
        "repeats": repeats,
        "device": str(device),
        "direct_ms_mean": direct_mean,
        "direct_ms_sd": float(np.std(timings["direct"], ddof=1)),
        "cached_ms_mean": cached_mean,
        "cached_ms_sd": float(np.std(timings["cached"], ddof=1)),
        "speedup": direct_mean / cached_mean,
        "max_residual_abs_error": max_residual_error,
        "max_weight_abs_error": max_weight_error,
        "max_delta_abs_error": max_delta_error,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [benchmark(profile, 8, device) for profile in
            ("cifar10_fixed", "cifar100_fixed", "officehome_fixed", "mnist_raw")]
    tables = ROOT / "results" / "tables"
    models = ROOT / "results" / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    with (tables / "round7_cached_residual_benchmark.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "purpose": "pre-cache direct versus algebraically cached final FedPERG",
        "synchronization": "torch.cuda.synchronize before and after each timed aggregation",
        "rows": rows,
    }
    (models / "round7_cached_residual_benchmark.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
