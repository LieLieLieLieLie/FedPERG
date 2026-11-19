import torch
from torch.nn import functional as F

from fedperg.model import FeatureMLP
from fedperg.reviewer_baselines import LayerwiseAlignment, PeriodAwareAggregation, mlp_sample_gradient_energy


def test_per_sample_gradient_energy_matches_parameter_gradients():
    torch.manual_seed(13)
    model = FeatureMLP(5, 3, 4).double()
    x = torch.randn(7, 5, dtype=torch.float64)
    y = torch.arange(len(x)) % 4
    losses, energy = mlp_sample_gradient_energy(model, x, y)
    expected = []
    for features, label in zip(x, y):
        loss = F.cross_entropy(model(features[None]), label[None])
        grads = torch.autograd.grad(loss, model.parameters())
        expected.append(sum(v.square().sum() for v in grads))
    torch.testing.assert_close(losses, F.cross_entropy(model(x), y, reduction='none'))
    torch.testing.assert_close(energy, torch.stack(expected))


def test_period_check_uses_last_participation_and_disables_corrections_outside_clp():
    state = PeriodAwareAggregation()
    mass = torch.tensor([.6, .4])
    norm = torch.tensor([1., 1.])
    first, _ = state.weights([1, 2], torch.tensor([.8, 1.]), norm, mass, .06)
    torch.testing.assert_close(first, mass)
    second, record = state.weights([1, 3], torch.tensor([1.8, .3]), 2 * norm, mass, .06)
    assert record['pa3fed_period_active']
    torch.testing.assert_close(second, (mass.log() + torch.tensor([.3, 0.])).softmax(0))
    third, record = state.weights([1, 2], torch.tensor([.8, .1]), norm, mass, .06)
    assert not record['pa3fed_period_active']
    torch.testing.assert_close(third, mass)


def test_layerwise_state_shares_layer_weights_but_can_reverse_client_order():
    alignment = LayerwiseAlignment(('a.weight', 'a.bias', 'b.weight', 'b.bias'))
    gradients = [[torch.tensor([v]) for v in values] for values in ((3., 1., -1., -.5), (-1., -.5, 3., 1.))]
    deltas = [[-g for g in client] for client in gradients]
    weights, _ = alignment.weights([5, 1], deltas, torch.tensor([.5, .5]), torch.ones(2), 1)
    torch.testing.assert_close(weights[0], weights[1])
    torch.testing.assert_close(weights[2], weights[3])
    assert weights[0][0] > weights[0][1] and weights[2][0] < weights[2][1]
    assert all(torch.isclose(w.sum(), torch.ones(())) for w in weights)
    mass = torch.tensor([.5, .5])
    rates = torch.tensor([2., 1.])
    alignment = LayerwiseAlignment(('a.weight', 'a.bias', 'b.weight', 'b.bias'))
    alignment.weights([5, 1], deltas, mass, rates, 4)
    first_layer = torch.stack([torch.cat(d[:2]) for d in deltas]).neg() / rates[:, None]
    reference = first_layer.mean(0)
    angle = (first_layer[0] @ reference / (first_layer[0].norm() * reference.norm())).clamp(-1, 1).acos()
    torch.testing.assert_close(alignment.angles[5][0], angle / 4)
