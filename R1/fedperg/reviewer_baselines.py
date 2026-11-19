"""Stateful implementations for the reviewer-requested aggregation mechanisms."""
import math
from collections import defaultdict

import torch
from torch.nn import functional as F

from .model import FeatureMLP
from .trainer import FederatedExperiment


def mlp_sample_gradient_energy(model, x, y):
    """Exact squared parameter-gradient norm for each example in a FeatureMLP."""
    with torch.enable_grad():
        preactivation = model.fc1(x).detach().requires_grad_()
        hidden = model.act(preactivation)
        logits = model.fc2(hidden)
        losses = F.cross_entropy(logits, y, reduction='none')
        hidden_gradient, output_gradient = torch.autograd.grad(losses.sum(), (preactivation, logits))
    # A linear layer's per-example weight gradient is the outer product of
    # its input and its preactivation gradient. The extra 1 accounts for bias.
    first = (x.square().sum(1) + 1) * hidden_gradient.square().sum(1)
    second = (hidden.detach().square().sum(1) + 1) * output_gradient.square().sum(1)
    return losses.detach(), first + second


def weighted_layer_updates(deltas, weights):
    return [torch.einsum('i,i...->...', weight, torch.stack(tensors))
            for tensors, weight in zip(zip(*deltas), weights)]


class PeriodAwareAggregation:
    """PA3Fed Algorithm 1, including per-client history under partial participation."""
    def __init__(self, beta=.3, threshold=.01):
        self.beta = beta
        self.threshold = threshold
        self.previous_scores = {}
        self.previous_fgn = None

    def weights(self, ids, losses, gradient_energy, sample_mass, learning_rate):
        # The magnitude has the same relative change as the signed FGN in Eq. 5.
        fgn = learning_rate * torch.dot(sample_mass, gradient_energy).item()
        change = None if not self.previous_fgn else (fgn - self.previous_fgn) / self.previous_fgn
        active = change is not None and change > self.threshold
        past_losses = losses.new_tensor([self.previous_scores.get(cid, loss)
                                         for cid, loss in zip(ids, losses.tolist())])
        log_correction = self.beta * (losses - past_losses)
        weights = (sample_mass.log() + log_correction).softmax(0) if active else sample_mass
        self.previous_scores.update(zip(ids, losses.tolist()))
        self.previous_fgn = fgn
        return weights, {'pa3fed_fgn_magnitude': fgn, 'pa3fed_fgn_relative_change': change,
                         'pa3fed_period_active': active, 'pa3fed_client_weights': weights.tolist()}


class LayerwiseAlignment:
    """Fed-LAA aggregation as specified in the publisher's algorithm figure."""
    def __init__(self, parameter_names, scale=5.):
        self.scale = scale
        groups = defaultdict(list)
        for index, name in enumerate(parameter_names):
            groups[name.rpartition('.')[0]].append(index)
        self.layers = tuple(groups.values())
        self.angles = {}
    def weights(self, ids, deltas, sample_mass, rate_epochs, round_idx):
        angles = []
        for layer in self.layers:
            vectors = -torch.stack([torch.cat([delta[i].flatten() for i in layer])
                                    for delta in deltas]) / rate_epochs[:, None]
            reference = torch.einsum('i,ij->j', sample_mass, vectors)
            cosine = (vectors @ reference) / (vectors.norm(dim=1) * reference.norm() + 1.e-12)
            angles.append(cosine.clamp(-1., 1.).acos())
        observed = torch.stack(angles, dim=1)
        smoothed = []
        for row, cid in zip(observed, ids):
            previous = self.angles.get(cid, torch.zeros_like(row))
            angle = previous.lerp(row, 1 / round_idx)
            self.angles[cid] = angle
            smoothed.append(angle)
        angle = torch.stack(smoothed)
        # Evaluate the decreasing Gompertz transform in the log weights.
        score = -self.scale * torch.expm1(-torch.exp(-self.scale * (angle - 1)))
        layer_weights = (sample_mass.log()[:, None] + score).softmax(0).T
        tensor_weights = [None] * sum(map(len, self.layers))
        for layer, weights in zip(self.layers, layer_weights):
            for index in layer:
                tensor_weights[index] = weights
        return tensor_weights, {'layerwise_client_weights': layer_weights.tolist()}


class ReviewerBaselineExperiment(FederatedExperiment):
    """Use the frozen harness and add only the diagnostic needed by each port."""
    def __init__(self, config):
        super().__init__(config)
        if not isinstance(self.model, FeatureMLP):
            raise ValueError('Reviewer baseline evaluation uses the shared FeatureMLP protocol')
        self.baseline = self
        self.period_aware = PeriodAwareAggregation() if config.method == 'PA3Fed' else None
        self.layerwise = LayerwiseAlignment([name for name, _ in self.model.named_parameters()])
        self.client_statistics = []
        self.baseline_diagnostics = {}

    def _train_client(self, cid, round_idx):
        if self.period_aware is None:
            epochs = self.cfg.local_epochs + int(self.cfg.regime != 'label_skew' and cid % 4 == 0)
            if self.cfg.unequal_work:
                epochs = (1, 2, 4, 8)[cid % 4]
            self.client_statistics.append((cid, self.cfg.local_lr * epochs))
            return super()._train_client(cid, round_idx)
        indices, _ = self._client_split(cid)
        total_loss = torch.zeros((), device=self.device)
        total_energy = torch.zeros_like(total_loss)
        self.model.eval()
        for batch in range(0, len(indices), self.cfg.batch_size):
            ids = indices[batch:batch + self.cfg.batch_size]
            x = self.data.x_train[ids].to(self.device)
            y = self.data.y_train[ids].to(self.device)
            losses, energies = mlp_sample_gradient_energy(self.model, x, y)
            total_loss += losses.sum()
            total_energy += energies.sum()
        statistics = torch.stack((total_loss, total_energy)) / len(indices)
        self.client_statistics.append((cid, statistics))
        return super()._train_client(cid, round_idx)

    def aggregate(self, deltas, sample_mass, improvement, steps, round_idx, global_flat, public_losses):
        ids, values = zip(*self.client_statistics)
        self.client_statistics.clear()
        if self.period_aware is not None:
            losses, energies = torch.stack(values).T
            weights, diagnostics = self.period_aware.weights(ids, losses, energies, sample_mass, self.cfg.local_lr)
            tensor_weights = [weights] * len(deltas[0])
        else:
            rate_epochs = sample_mass.new_tensor(values)
            tensor_weights, diagnostics = self.layerwise.weights(ids, deltas, sample_mass, rate_epochs, round_idx)
            weights = torch.stack(tensor_weights).mean(0)
        self.baseline_diagnostics[round_idx] = diagnostics
        return weighted_layer_updates(deltas, tensor_weights), weights

    def _save(self, result, averaged_state=None):
        for entry in result['diagnostics']:
            if entry.get('skipped'):
                continue
            entry.update(self.baseline_diagnostics[entry['round']])
            clients = len(entry['selected_clients'])
            # PA3Fed uploads two scalars. Fed-LAA reconstructs gradients on
            # the server from deltas and the known local SGD configuration.
            payload = 8 * clients if self.period_aware is not None else 0
            entry['reviewer_baseline_extra_upload_bytes'] = payload
            result['communication_mb'] += payload / 1e6
        super()._save(result, averaged_state)
