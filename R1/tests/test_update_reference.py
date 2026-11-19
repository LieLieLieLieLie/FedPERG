"""Local-work reference equations agree with effective-step normalization."""
import pytest
import torch

from fedperg.update_reference import normalize_local_work
from fedperg.evidence_fit import fit_momentum


def test_reference_matches_effective_step_formula():
    updates = [[torch.tensor([2., -4.])], [torch.tensor([-3., 6.])]]
    masses = torch.tensor([.25, .75])
    steps = torch.tensor([2., 6.])
    transformed = normalize_local_work(updates, masses, steps)
    actual = sum(p * delta[0] for p, delta in zip(masses, transformed))
    expected = (masses * steps).sum() * sum(p * delta[0] / count
                                           for p, delta, count in zip(masses, updates, steps))
    torch.testing.assert_close(actual, expected)
    equal = normalize_local_work(updates, masses, torch.tensor([3., 3.]))
    for a, b in zip(updates, equal):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    with pytest.raises(ValueError):
        normalize_local_work(updates, masses, torch.tensor([0., 3.]))


def test_projected_momentum_has_a_nonpositive_directional_effect():
    coefficient, effect = fit_momentum([torch.ones(4)], [torch.ones(4)], .3)
    assert coefficient == pytest.approx(.6)
    assert effect == pytest.approx(-1.2)
    coefficient, effect = fit_momentum([-torch.ones(4)], [torch.ones(4)], .6)
    assert coefficient == .9 and effect == 0
    coefficient, effect = fit_momentum([torch.zeros(4)], [torch.ones(4)], .6)
    assert coefficient == .9 and effect == 0
