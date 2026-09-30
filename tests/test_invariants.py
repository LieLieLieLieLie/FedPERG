from __future__ import annotations

import torch

from fedcanto.aggregators import ClientLayerSetOperator


def test_client_permutation_equivariance() -> None:
    torch.manual_seed(3)
    operator = ClientLayerSetOperator(feature_dim=7, layers=4, hidden=24, heads=4).eval()
    x = torch.randn(6, 4, 7)
    perm = torch.tensor([4, 1, 5, 0, 3, 2])
    with torch.no_grad():
        score, reconstruction = operator(x)
        score_p, reconstruction_p = operator(x[perm])
    assert torch.allclose(score_p, score[perm], atol=1e-5)
    assert torch.allclose(reconstruction_p, reconstruction[perm], atol=1e-5)


def test_mask_shapes() -> None:
    operator = ClientLayerSetOperator(feature_dim=7, layers=4, hidden=24, heads=4)
    x = torch.randn(5, 4, 7)
    mask = torch.zeros(5, 4, dtype=torch.bool)
    mask[0] = True
    score, reconstruction = operator(x, mask)
    assert score.shape == (5, 4)
    # The current operator reconstructs the 16 update-specific target
    # statistics, not the full input token (which also has metadata).
    assert reconstruction.shape == (5, 4, 16)
