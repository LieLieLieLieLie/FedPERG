"""Development holdouts and robust risk sums preserve their intended populations."""
from types import SimpleNamespace

import numpy as np
import torch

from fedperg.config import ExperimentConfig
from fedperg.data import independent_confidence_indices, load_federated_data
from fedperg.trainer import FederatedExperiment


def test_clean_validation_restoration_preserves_training_examples():
    settings = dict(data_dir='.', dataset='synthetic_complementarity', regime='compound',
                    clients=3, alpha=.2, seed=200, calibration_fraction=.04,
                    confidence_fraction=.04)
    original = load_federated_data(**settings)
    protected = load_federated_data(**settings, clean_confidence_pool=True)
    heldout = independent_confidence_indices(len(original.y_train), original.calibration_indices, .04, 200)
    training = np.setdiff1d(np.arange(len(original.y_train)), heldout)
    np.testing.assert_array_equal(original.calibration_indices, protected.calibration_indices)
    torch.testing.assert_close(original.x_train[training], protected.x_train[training], rtol=0, atol=0)
    torch.testing.assert_close(original.y_train[training], protected.y_train[training], rtol=0, atol=0)
    assert not np.intersect1d(heldout, protected.calibration_indices).size


def test_balanced_brier_shards_match_complete_pool_gradient():
    experiment = FederatedExperiment.__new__(FederatedExperiment)
    experiment.cfg = ExperimentConfig(calibration_objective='balanced_brier')
    experiment.device = torch.device('cpu')
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    experiment.data = SimpleNamespace(y_train=labels)
    logits = torch.randn(6, 3, generator=torch.Generator().manual_seed(51), requires_grad=True)
    indices = np.arange(6)
    full = experiment._labeled_calibration_loss(logits, labels, reference_indices=indices)
    split = sum(experiment._labeled_calibration_loss(logits[part], labels[part],
                                                    reference_indices=indices)
                for part in (slice(0, 2), slice(2, 6)))
    torch.testing.assert_close(full, split)
    a, = torch.autograd.grad(full, logits, retain_graph=True)
    b, = torch.autograd.grad(split, logits)
    torch.testing.assert_close(a, b)
