from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.no_grad()
def predict(model: nn.Module, x: Tensor, device: torch.device, batch_size: int = 1024) -> Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(x), batch_size):
        outputs.append(model(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def expected_calibration_error(probs: Tensor, labels: Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)
    value = torch.tensor(0.0)
    edges = torch.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (conf > low) & (conf <= high)
        if mask.any():
            value += mask.float().mean() * (conf[mask].mean() - correct[mask].float().mean()).abs()
    return float(value)


def classification_metrics(logits: Tensor, labels: Tensor, classes: int) -> Dict[str, float]:
    probs = logits.softmax(dim=1)
    pred = probs.argmax(dim=1)
    accuracy = float(pred.eq(labels).float().mean())
    confmat = torch.zeros(classes, classes, dtype=torch.float64)
    for truth, guess in zip(labels, pred):
        confmat[int(truth), int(guess)] += 1
    tp = confmat.diag()
    precision = tp / confmat.sum(dim=0).clamp_min(1)
    recall = tp / confmat.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = confmat.sum(dim=1) > 0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    nll = float(F.cross_entropy(logits, labels))
    one_hot = F.one_hot(labels, num_classes=classes).float()
    brier = float((probs - one_hot).pow(2).sum(dim=1).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "nll": nll,
        "ece": expected_calibration_error(probs, labels),
        "brier": brier,
    }


def fairness_metrics(logits: Tensor, labels: Tensor, client_test: Sequence[np.ndarray]) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    scores: List[float] = []
    for idx in client_test:
        ids = torch.as_tensor(idx, dtype=torch.long)
        scores.append(float(pred[ids].eq(labels[ids]).float().mean()))
    values = np.asarray(scores, dtype=float)
    tail = max(1, int(np.ceil(0.2 * len(values))))
    return {
        "client_mean": float(values.mean()),
        "client_std": float(values.std(ddof=0)),
        "worst20_accuracy": float(np.sort(values)[:tail].mean()),
        "client_min": float(values.min()),
        "client_accuracies": scores,
    }
