"""Vectorized confusion counts preserve the previously reported macro F1."""
import torch

from fedperg.metrics import classification_metrics


def test_macro_f1_matches_literal_confusion_matrix_counts():
    generator = torch.Generator().manual_seed(822)
    logits = torch.randn(91, 7, generator=generator)
    labels = torch.randint(0, 5, (91,), generator=generator)
    counts = torch.zeros(7, 7, dtype=torch.float64)
    for truth, guess in zip(labels, logits.argmax(1)):
        counts[int(truth), int(guess)] += 1
    true_positive = counts.diag()
    precision = true_positive / counts.sum(0).clamp_min(1)
    recall = true_positive / counts.sum(1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    expected = float(f1[counts.sum(1) > 0].mean())
    assert classification_metrics(logits, labels, 7)['macro_f1'] == expected
