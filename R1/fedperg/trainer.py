from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .aggregators import BaselineAggregator, PERGAggregator
from .config import ExperimentConfig
from .data import FederatedData, independent_confidence_indices, load_federated_data
from .metrics import classification_metrics, fairness_metrics, predict
from .recent_baselines import FedAWAState, dawid_skene_weights
from .update_reference import normalize_local_work
from .model import FeatureMLP, RawDigitCNN, RawRGBCNN, RawRGBResNet18, ResidualFeatureNet, add_delta_, clone_state, flatten_delta, state_delta


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_name(text: str) -> str:
    return text.lower().replace("-", "_").replace(" ", "_")


class FederatedExperiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        _seed_everything(config.seed)
        if config.device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cpu")
        self.data: FederatedData = load_federated_data(
            config.data_dir,
            config.dataset,
            config.regime,
            config.clients,
            config.dirichlet_alpha,
            config.seed,
            config.calibration_fraction,
            config.specialization,
            config.selector_calibration_shift,
            config.natural_domains,
            config.compound_label_fault_rate,
            config.confidence_fraction, config.clean_confidence_pool,
        )
        self.utility_indices=np.empty(0,dtype=np.int64)
        self.routing_indices=self.data.calibration_indices[:512].copy()
        if config.utility_evidence:
            count=max(16,int(len(self.routing_indices)*config.utility_pool_fraction))
            self.utility_indices=self.routing_indices[:count].copy()
            self.routing_indices=self.routing_indices[count:].copy()
            if len(self.routing_indices)<16:
                raise ValueError('Independent evidence and routing pools must each contain >=16 examples')
        self.confidence_indices = independent_confidence_indices(
            len(self.data.y_train), self.data.calibration_indices, config.confidence_fraction, config.seed)
        if config.class_difficulty > 0:
            if self.data.x_train.ndim != 2:
                raise ValueError('Class-difficulty intervention is defined for fixed features only')
            # Class IDs, not client identities or corruption labels, define the
            # harder classes. Use an independent RNG and disclose train-only shift.
            mask = self.data.y_train < self.data.classes // 2
            if config.clean_confidence_pool:
                mask[self.confidence_indices] = False
            generator = torch.Generator().manual_seed(config.seed + 94021)
            x = self.data.x_train.clone()
            noise = torch.randn(x[mask].shape, generator=generator)
            x[mask] = (1-config.class_difficulty)*x[mask] + config.class_difficulty*noise
            self.data.x_train = x
        self.model_class = {"shallow": FeatureMLP, "residual_feature": ResidualFeatureNet,
                            "raw_cnn": RawDigitCNN, "raw_rgb_cnn": RawRGBCNN,
                            "raw_rgb_resnet18": RawRGBResNet18}[config.architecture]
        self.model = self.model_class(self.data.input_dim, config.hidden_dim, self.data.classes).to(self.device)
        self.initial_state = clone_state(self.model)
        self.scaffold_server = [torch.zeros_like(p) for p in self.model.parameters()]
        self.scaffold_clients = [
            [torch.zeros_like(p) for p in self.model.parameters()] for _ in range(config.clients)
        ]
        self.feddyn_clients = [
            [torch.zeros_like(p) for p in self.model.parameters()] for _ in range(config.clients)
        ]
        reference = [torch.zeros_like(p) for p in self.model.parameters()]
        self.perg: Optional[PERGAggregator] = None
        self.baseline: Optional[BaselineAggregator] = None
        if config.method == "FedPERG":
            self.perg = PERGAggregator(
                reference,
                config.perg_hidden,
                config.perg_heads,
                config.perg_lr,
                config.perg_meta_steps,
                config.perg_rho,
                self.device,
                config.perg_variant,
                temperature=config.perg_temperature,
                reconstruction_weight=config.perg_reconstruction_weight,
                median_weight=config.perg_median_weight,
                memory_weight=config.perg_memory_weight,
                gate_low=config.perg_gate_low,
                gate_span=config.perg_gate_span,
            )
        else:
            self.baseline = BaselineAggregator(
                config.method, reference, config.seed,
                gate_low=config.perg_gate_low,
                gate_span=config.perg_gate_span,
            )
            self.baseline.state.initial = torch.cat([v.reshape(-1) for v in self.initial_state.values()]).detach()
        self.rng = np.random.default_rng(config.seed + 1181)
        self.fedawa = FedAWAState(config.clients) if config.method == 'FedAWA' else None
        if self.perg is not None:
            self.perg.evidence_control = config.evidence_control
            self.perg.improvement_weight = config.improvement_weight
            self.perg.gate_bank_scale = config.gate_bank_scale
            self.perg.gate_bank_offset = config.gate_bank_offset
            self.perg.control_seed = config.seed
            self.perg.utility_strength=config.utility_strength
            self.perg.utility_prior_strength=config.utility_prior_strength
            self.perg.utility_gate_mode=config.utility_gate_mode
            self.perg.utility_field_radius=config.utility_field_radius
            self.perg.utility_momentum_step=config.utility_momentum_step
        if config.client_imbalance_power != 1.0:
            for cid, idx in enumerate(self.data.client_train):
                fraction = ((cid + 1) / config.clients) ** (config.client_imbalance_power - 1)
                self.data.client_train[cid] = idx[:max(16, int(len(idx) * fraction))]

    def _sample_loss(self, model: nn.Module, idx: np.ndarray, limit: int = 256) -> float:
        if len(idx) > limit:
            idx = idx[:limit]
        with torch.no_grad():
            logits = model(self.data.x_train[idx].to(self.device))
            return float(F.cross_entropy(logits, self.data.y_train[idx].to(self.device)))

    def _client_split(self, cid: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.data.client_train[cid]
        if self.cfg.selector_calibration:
            # The server calibration pool is disjoint from every client's
            # local optimization data for both FedPERG and matched controls.
            idx = idx[~np.isin(idx, self.data.calibration_indices)]
        if len(self.confidence_indices):
            idx = idx[~np.isin(idx, self.confidence_indices)]
        fraction = float(self.cfg.client_validation_fraction)
        if fraction <= 0.0 or len(idx) < 20:
            return idx.copy(), np.empty(0, dtype=np.int64)
        count = max(8, int(round(len(idx) * fraction)))
        count = min(count, max(1, len(idx) // 3))
        return idx[:-count].copy(), idx[-count:].copy()

    def _selected_validation_loss(self, selected: np.ndarray) -> float:
        weighted, mass = 0.0, 0
        was_training = self.model.training
        self.model.eval()
        for cid in selected:
            _, val_idx = self._client_split(int(cid))
            if len(val_idx) == 0:
                continue
            weighted += len(val_idx) * self._sample_loss(self.model, val_idx, limit=256)
            mass += len(val_idx)
        self.model.train(was_training)
        return weighted / max(mass, 1)

    def _selector_validation_loss(self, selected: np.ndarray) -> float:
        if not self.cfg.selector_calibration:
            return self._selected_validation_loss(selected)
        was_training = self.model.training
        self.model.eval()
        if self.cfg.calibration_mode=='centralized':
            with torch.no_grad():
                ids = self.routing_indices
                value = float(self._labeled_calibration_loss(
                    self.model(self.data.x_train[ids].to(self.device)),
                    self.data.y_train[ids].to(self.device), reduction='mean',
                    reference_indices=self.routing_indices))
        elif self.cfg.calibration_mode=='distributed':
            total=0.;count=0
            with torch.no_grad():
                for ids in np.array_split(self.routing_indices,self.cfg.calibration_sites):
                    if len(ids):
                        loss=self._labeled_calibration_loss(self.model(self.data.x_train[ids].to(self.device)),
                            self.data.y_train[ids].to(self.device), reference_indices=self.routing_indices)
                        total+=float(loss);count+=len(ids)
            value=total/max(count,1)
        elif self.cfg.calibration_mode=='unlabeled':
            with torch.no_grad():
                logits = self.model(self.data.x_train[self.routing_indices].to(self.device))
                value = float(self._unlabeled_calibration_loss(logits, reduction='mean'))
        else:raise ValueError('Unknown calibration mode')
        self.model.train(was_training)
        return value

    def _labeled_calibration_loss(self, logits: Tensor, labels: Tensor, reduction: str = 'sum',
                                 reference_indices=None) -> Tensor:
        objective = self.cfg.calibration_objective
        if objective in {'brier', 'balanced_brier', 'gce'}:
            probabilities = logits.softmax(-1)
            if objective == 'gce':
                correct = probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
                losses = (1 - correct.pow(.7)) / .7
            else:
                truth = F.one_hot(labels, num_classes=logits.shape[-1])
                losses = (probabilities - truth).square().sum(-1)
                if objective == 'balanced_brier':
                    if reference_indices is None:
                        raise ValueError('Balanced risk needs the complete fitting or routing label counts')
                    counts = torch.bincount(self.data.y_train[reference_indices],
                                            minlength=logits.shape[-1]).to(self.device)
                    weights = len(reference_indices) / ((counts > 0).sum() * counts.clamp_min(1))
                    losses = losses * weights[labels]
            return losses.mean() if reduction == 'mean' else losses.sum()
        loss = F.cross_entropy(logits, labels, reduction=reduction)
        if objective == 'brier_balance':
            probabilities = logits.softmax(-1)
            truth = F.one_hot(labels, num_classes=logits.shape[-1])
            brier = (probabilities - truth).square().sum(-1)
            loss = loss + .5 * (brier.mean() if reduction == 'mean' else brier.sum())
        return loss

    def _unlabeled_calibration_loss(self, logits: Tensor, reduction: str = 'sum') -> Tensor:
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        loss = entropy.mean() if reduction == 'mean' else entropy.sum()
        if self.cfg.calibration_objective in {'brier_balance', 'brier', 'balanced_brier', 'gce'}:
            marginal = probabilities.mean(0)
            # Discourage confident collapse to one class without using labels.
            batch_weight = 1 if reduction == 'mean' else len(logits)
            loss = loss + .5 * batch_weight * (marginal * marginal.clamp_min(1e-12).log()).sum()
        return loss

    def _selector_test_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        count = min(512, len(self.data.y_test))
        with torch.no_grad():
            logits = self.model(self.data.x_test[:count].to(self.device))
            value = float(F.cross_entropy(logits, self.data.y_test[:count].to(self.device)))
        self.model.train(was_training)
        return value

    def _fedlaw_update(self, deltas, base_weights):
        """Learn simplex client weights AND global model shrinkage on the proxy.

        Implements the FedLAW aggregation parameterization (Li et al., ICML
        2023), not the R0 heuristic softmax of individual client proxy losses.
        Shared data/local-training harness is retained, not the official full
        training protocol. Only proxy labels enter the server optimization.
        """
        from torch.func import functional_call
        names = [name for name, _ in self.model.named_parameters()]
        global_params = [p.detach().clone() for p in self.model.parameters()]
        local_params = [torch.stack([g + d[j] for d in deltas])
                        for j, g in enumerate(global_params)]
        alpha = base_weights.clamp_min(1e-12).log().detach().requires_grad_(True)
        log_gamma = torch.zeros((), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha, log_gamma], lr=self.cfg.fedlaw_server_lr)
        ids = self.data.calibration_indices[:512]
        x, y = self.data.x_train[ids].to(self.device), self.data.y_train[ids].to(self.device)
        was_training = self.model.training
        self.model.eval()
        for _ in range(self.cfg.fedlaw_server_steps):
            weights = alpha.softmax(0)
            gamma = log_gamma.exp()
            params = {name: gamma * torch.einsum('i,i...->...', weights, local)
                      for name, local in zip(names, local_params)}
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(functional_call(self.model, params, (x,)), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            weights, gamma = alpha.softmax(0), log_gamma.exp()
            update = [gamma * torch.einsum('i,i...->...', weights, local) - g
                      for local, g in zip(local_params, global_params)]
        self.model.train(was_training)
        return update, weights.detach()

    def _utility_gradient(self,deltas,base_weights):
        """Evidence fitting uses A; the outer router sees only disjoint pool B."""
        saved=clone_state(self.model)
        reference=[]
        with torch.no_grad():
            for layer in range(len(deltas[0])):
                stacked=torch.stack([d[layer] for d in deltas])
                view=(len(deltas),)+(1,)*(stacked.ndim-1)
                base=(stacked*base_weights.reshape(view)).sum(0)
                agreement=((stacked.sign()==base.sign()[None]).float()
                           *base_weights.reshape(view)).sum(0)
                low, span = .98, .48
                if self.cfg.utility_anchor_matches_bank:
                    low += self.cfg.gate_bank_offset
                    span *= self.cfg.gate_bank_scale
                reference.append(.9*self.perg.momentum[layer]+base*(low+span*agreement))
        add_delta_(self.model,reference,self.cfg.server_lr)
        self.model.eval();self.model.zero_grad(set_to_none=True)
        chunks=(np.array_split(self.utility_indices,self.cfg.calibration_sites)
                if self.cfg.calibration_mode=='distributed' else [self.utility_indices])
        total_loss=0.
        for ids in chunks:
            if not len(ids):continue
            logits=self.model(self.data.x_train[ids].to(self.device))
            if self.cfg.calibration_mode=='unlabeled':
                term=self._unlabeled_calibration_loss(logits)
            else:
                term=self._labeled_calibration_loss(logits,self.data.y_train[ids].to(self.device),
                                                     reference_indices=self.utility_indices)
            loss=term/len(self.utility_indices)
            loss.backward();total_loss+=float(loss.detach())
        gradients=[p.grad.detach().clone() for p in self.model.parameters()]
        self.model.load_state_dict(saved);self.model.zero_grad(set_to_none=True)
        return gradients,{'utility_fit_n':len(self.utility_indices),
            'outer_routing_n':len(self.routing_indices),'utility_reference_loss':total_loss}

    def _fedds_update(self,deltas):
        saved=clone_state(self.model);self.model.eval();predictions=[]
        ids=self.data.calibration_indices[:512]
        with torch.no_grad():
            for delta in deltas:
                self.model.load_state_dict(saved);add_delta_(self.model,delta,1.)
                predictions.append(self.model(self.data.x_train[ids].to(self.device)).argmax(-1).cpu().numpy())
        self.model.load_state_dict(saved)
        weights,diag=dawid_skene_weights(np.stack(predictions,axis=1),self.data.classes)
        weights=torch.as_tensor(weights,device=self.device,dtype=torch.float32)
        update=[torch.einsum('i,i...->...',weights,torch.stack([d[j] for d in deltas]))
                for j in range(len(deltas[0]))]
        return update,weights,diag

    def _train_client(self, cid: int, round_idx: int) -> Tuple[List[Tensor], float, int, float]:
        cfg, method = self.cfg, self.cfg.method
        local = self.model_class(self.data.input_dim, cfg.hidden_dim, self.data.classes).to(self.device)
        local.load_state_dict(self.model.state_dict())
        global_params = [p.detach().clone() for p in self.model.parameters()]
        idx, _ = self._client_split(cid)
        generator = np.random.default_rng(cfg.seed * 100003 + round_idx * 101 + cid)
        generator.shuffle(idx)
        before = self._sample_loss(local, idx)
        proxy_before = (self._sample_loss(local, self.data.calibration_indices, limit=512)
                        if cfg.reward_diagnostics else None)
        optimizer = torch.optim.SGD(local.parameters(), lr=cfg.local_lr, momentum=0.0, weight_decay=cfg.weight_decay)
        epochs = cfg.local_epochs + int(cfg.regime != "label_skew" and cid % 4 == 0)
        if cfg.unequal_work:
            epochs = (1, 2, 4, 8)[cid % 4]
        steps = 0
        local.train()
        max_batches = cfg.max_local_batches or (3 if cfg.dataset != "cifar100" else 2)
        for epoch in range(epochs):
            generator.shuffle(idx)
            for batch_no, start in enumerate(range(0, len(idx), cfg.batch_size)):
                if batch_no >= max_batches:
                    break
                ids = idx[start : start + cfg.batch_size]
                x = self.data.x_train[ids].to(self.device)
                y = self.data.y_train[ids].to(self.device)
                loss = F.cross_entropy(local(x), y)
                if method == "FedProx":
                    prox = sum((p - g).pow(2).sum() for p, g in zip(local.parameters(), global_params))
                    loss = loss + 0.012 * prox
                elif method == "FedDyn":
                    prox = sum((p - g).pow(2).sum() for p, g in zip(local.parameters(), global_params))
                    linear = sum((h * p).sum() for h, p in zip(self.feddyn_clients[cid], local.parameters()))
                    loss = loss + 0.018 * prox - linear
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if method == "SCAFFOLD":
                    with torch.no_grad():
                        for p, c_global, c_local in zip(
                            local.parameters(), self.scaffold_round_reference, self.scaffold_clients[cid]
                        ):
                            p.grad.add_(c_global - c_local)
                torch.nn.utils.clip_grad_norm_(local.parameters(), 8.0)
                optimizer.step()
                steps += 1
        after = self._sample_loss(local, idx)
        improvement = before - after
        raw_improvement = improvement
        if cfg.reward_mode == "per_step":
            improvement /= max(steps, 1)
        elif cfg.reward_mode == "relative":
            improvement /= max(before, 1e-8)
        elif cfg.reward_mode == "none":
            improvement = 0.0
        if cfg.reward_diagnostics:
            proxy_after = self._sample_loss(local, self.data.calibration_indices, limit=512)
            self.round_reward_diagnostics.append({
                'client': cid, 'steps': steps, 'initial_local_loss': before,
                'raw_reward': raw_improvement, 'used_reward': improvement,
                'proxy_gain': proxy_before-proxy_after,
                'hard_class_fraction': float((self.data.y_train[idx] < self.data.classes//2).float().mean()),
                'compound_fault_client': cid % 7 == 0})
        delta = state_delta(local.state_dict(), self.model.state_dict())

        if method == "SCAFFOLD" and steps > 0:
            old = [x.clone() for x in self.scaffold_clients[cid]]
            with torch.no_grad():
                for layer in range(len(delta)):
                    self.scaffold_clients[cid][layer] = (
                        self.scaffold_clients[cid][layer]
                        - self.scaffold_round_reference[layer]
                        - delta[layer] / (steps * cfg.local_lr)
                    )
                for layer in range(len(delta)):
                    self.scaffold_server[layer].add_(
                        self.scaffold_clients[cid][layer] - old[layer], alpha=1.0 / cfg.clients
                    )
        elif method == "FedDyn":
            with torch.no_grad():
                for layer in range(len(delta)):
                    self.feddyn_clients[cid][layer].sub_(delta[layer], alpha=0.018)

        calibration_loss = math.nan
        if method == "FedLBW":
            calibration_idx = self.data.calibration_indices
            calibration_loss = self._sample_loss(local, calibration_idx, limit=512)
        del local
        return delta, improvement, steps, calibration_loss

    @torch.no_grad()
    def _evaluate(self, round_idx: int) -> Dict[str, object]:
        logits = predict(self.model, self.data.x_test, self.device)
        metrics: Dict[str, object] = classification_metrics(logits, self.data.y_test, self.data.classes)
        metrics.update(fairness_metrics(logits, self.data.y_test, self.data.client_test))
        metrics["round"] = round_idx
        if self.cfg.record_development_validation:
            if not len(self.confidence_indices):
                raise ValueError('Development validation requires an independent confidence pool')
            ids = self.confidence_indices
            validation_logits = predict(self.model, self.data.x_train[ids], self.device)
            validation_labels = self.data.y_train[ids]
            groups = [np.flatnonzero(np.isin(ids, client_ids)) for client_ids in self.data.client_train]
            groups = [group for group in groups if len(group)]
            metrics['development'] = {
                **classification_metrics(validation_logits, validation_labels, self.data.classes),
                **fairness_metrics(validation_logits, validation_labels, groups),
            }
        return metrics

    def run(self, verbose: bool = True) -> Dict[str, object]:
        cfg = self.cfg
        if not 0 <= cfg.terminal_average_rounds <= cfg.rounds:
            raise ValueError('Terminal averaging window must fit within the training horizon')
        averaged_state = None
        averaged_rounds = 0

        @torch.no_grad()
        def average_snapshot(round_idx):
            nonlocal averaged_state, averaged_rounds
            if round_idx <= cfg.rounds - cfg.terminal_average_rounds:
                return
            averaged_rounds += 1
            if averaged_state is None:
                averaged_state = clone_state(self.model)
                return
            for name, tensor in self.model.state_dict().items():
                if tensor.is_floating_point():
                    averaged_state[name].lerp_(tensor, 1 / averaged_rounds)
                else:
                    averaged_state[name].copy_(tensor)

        history: List[Dict[str, object]] = [self._evaluate(0)]
        history[0]["elapsed_seconds"] = 0.0
        diagnostics: List[Dict[str, object]] = []
        total_communication = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        all_clients = np.arange(cfg.clients)
        for round_idx in range(1, cfg.rounds + 1):
            self.round_reward_diagnostics = []
            # All selected clients must observe the same pre-round control.
            self.scaffold_round_reference = [c.clone() for c in self.scaffold_server]
            migration_round = cfg.domain_migration_round or cfg.rounds // 2 + 1
            if cfg.domain_migration and round_idx == migration_round:
                # Abrupt local-domain migration uses real images from another
                # domain. Global examples/labels remain unchanged.
                self.data.client_train=self.data.client_train[2:]+self.data.client_train[:2]
                self.data.client_test=self.data.client_test[2:]+self.data.client_test[:2]
            selected = self.rng.choice(all_clients, size=cfg.clients_per_round, replace=False)
            if cfg.dropout_rate > 0:
                keep = self.rng.random(len(selected)) >= cfg.dropout_rate
                selected = selected[keep]
                if len(selected) < 2:
                    # Skip this round rather than inventing successful clients.
                    diagnostics.append({"round": round_idx, "skipped": True,
                                        "reason": "fewer_than_two_returned_updates"})
                    average_snapshot(round_idx)
                    continue
            deltas: List[List[Tensor]] = []
            improvements: List[float] = []
            steps: List[int] = []
            calibration_losses: List[float] = []
            sizes: List[int] = []
            local_started = time.perf_counter()
            for cid in selected:
                delta, improvement, count, calibration_loss = self._train_client(int(cid), round_idx)
                if cfg.attack_fraction > 0 and int(cid) < int(math.ceil(cfg.clients * cfg.attack_fraction)):
                    if cfg.attack_type == "signflip":
                        delta = [x.mul(-3.0) for x in delta]
                    elif cfg.attack_type == "gaussian":
                        attack_gen = torch.Generator(device=self.device).manual_seed(
                            cfg.seed * 811 + round_idx * 43 + int(cid)
                        )
                        delta = [
                            x
                            + torch.randn(x.shape, generator=attack_gen, device=x.device)
                            * x.std().clamp_min(1e-4)
                            * 4.0
                            for x in delta
                        ]
                deltas.append(delta)
                improvements.append(improvement)
                steps.append(count)
                calibration_losses.append(calibration_loss)
                train_idx, _ = self._client_split(int(cid))
                sizes.append(len(train_idx))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            local_seconds = time.perf_counter() - local_started
            base_weights = torch.tensor(sizes, dtype=torch.float32, device=self.device)
            base_weights /= base_weights.sum()
            imp_tensor = torch.tensor(improvements, dtype=torch.float32, device=self.device)
            step_tensor = torch.tensor(steps, dtype=torch.float32, device=self.device)
            agg_started = time.perf_counter()
            perg_diag: Dict[str, object] = {}
            if self.perg is not None:
                utility_diag={}
                reference_deltas = deltas
                if cfg.update_reference == 'work_normalized':
                    reference_deltas = normalize_local_work(deltas, base_weights, step_tensor)
                elif cfg.update_reference != 'sample_mass':
                    raise ValueError('Unknown update reference')
                if cfg.utility_evidence:
                    self.perg.utility_gradient,utility_diag=self._utility_gradient(reference_deltas,base_weights)
                out = self.perg.aggregate(reference_deltas, base_weights, imp_tensor, round_idx)
                aggregated = out.delta
                client_weights = out.weights.mean(dim=1)
                perg_diag = {
                    **utility_diag,
                    'utility_fitted_linear_effects':getattr(self.perg,'utility_linear_effects',None),
                    'utility_momentum_coefficient':self.perg.utility_momentum_coefficient,
                    'utility_momentum_linear_effect':self.perg.utility_momentum_linear_effect,
                    "rho": out.rho,
                    "meta_loss": out.meta_loss,
                    "reconstruction_error": float(out.reconstruction_error.mean()),
                    "reconstruction_error_by_layer": out.reconstruction_error.cpu().tolist(),
                    "layer_weight_std": float(out.weights.std()),
                    "weights_by_layer": out.weights.cpu().tolist(),
                    "prediction_seconds": out.prediction_seconds,
                    "fusion_bias": out.fusion_bias,
                    "attentive_bias": out.attentive_bias,
                }
                if (cfg.perg_variant in {"lite_full_gate_selector",
                                          "lite_full_direction_selector"} and
                        out.control_delta is not None):
                    saved = {name: tensor.detach().clone()
                             for name, tensor in self.model.state_dict().items()}
                    add_delta_(self.model, out.delta, cfg.server_lr)
                    evidence_loss = self._selector_validation_loss(selected)
                    self.model.load_state_dict(saved)
                    add_delta_(self.model, out.control_delta, cfg.server_lr)
                    control_loss = self._selector_validation_loss(selected)
                    self.model.load_state_dict(saved)
                    # The evidence expert must clear a predeclared absolute
                    # validation-loss margin; otherwise the controller falls
                    # back exactly to the matched AvgM+Gate branch.
                    use_evidence = evidence_loss + cfg.selector_margin < control_loss
                    aggregated = out.delta if use_evidence else out.control_delta
                    for memory, chosen in zip(self.perg.momentum, aggregated):
                        memory.copy_(chosen)
                    perg_diag.update({
                        "selector_evidence_loss": evidence_loss,
                        "selector_control_loss": control_loss,
                        "selector_used_evidence": bool(use_evidence),
                        "selector_margin": float(cfg.selector_margin),
                    })
                elif (cfg.perg_variant in {"lite_gate_bank_selector",
                                            "lite_gate_bank_router_control",
                                            "lite_paired_gate_bank_selector",
                                            "lite_paired_gate_bank_no_residual",
                                            "lite_paired_gate_bank_router_control"} and
                      out.expert_deltas is not None and out.expert_names is not None):
                    saved = {name: tensor.detach().clone()
                             for name, tensor in self.model.state_dict().items()}
                    expert_losses: List[float] = []
                    expert_test_losses: List[float] = []
                    for candidate_index, candidate in enumerate(out.expert_deltas):
                        duplicate = None
                        if cfg.deduplicate_candidates:
                            duplicate = next((j for j in range(candidate_index)
                                if all(torch.equal(a, b) for a, b in
                                       zip(candidate, out.expert_deltas[j]))), None)
                        if duplicate is not None:
                            expert_losses.append(expert_losses[duplicate])
                            if cfg.selector_test_diagnostics:
                                expert_test_losses.append(expert_test_losses[duplicate])
                            continue
                        add_delta_(self.model, candidate, cfg.server_lr)
                        expert_losses.append(self._selector_validation_loss(selected))
                        if cfg.selector_test_diagnostics:
                            expert_test_losses.append(self._selector_test_loss())
                        self.model.load_state_dict(saved)
                    if cfg.perg_variant in {"lite_paired_gate_bank_selector",
                                             "lite_paired_gate_bank_no_residual",
                                             "lite_paired_gate_bank_router_control"}:
                        router_indices = [0, 1, 3, 5]
                        evidence_indices = [2, 4, 6]
                        best_router = min(router_indices, key=expert_losses.__getitem__)
                        best_evidence = min(evidence_indices, key=expert_losses.__getitem__)
                        best = (best_evidence
                                if expert_losses[best_evidence] + cfg.selector_margin
                                < expert_losses[best_router] else best_router)
                    else:
                        best = min(range(len(expert_losses)), key=expert_losses.__getitem__)
                        if expert_losses[best] + cfg.selector_margin >= expert_losses[0]:
                            best = 0
                    aggregated = out.expert_deltas[best]
                    for memory, chosen in zip(self.perg.momentum, aggregated):
                        memory.copy_(chosen)
                    perg_diag.update({
                        "selector_expert_losses": expert_losses,
                        "selector_expert_names": out.expert_names,
                        "selector_expert_index": int(best),
                        "selector_used_evidence": bool(best != 0),
                        "selector_margin": float(cfg.selector_margin),
                        "selector_used_target_evidence": bool(
                            cfg.perg_variant in {"lite_paired_gate_bank_selector",
                                                  "lite_paired_gate_bank_no_residual",
                                                  "lite_paired_gate_bank_router_control"}
                            and best in {2, 4, 6}
                        ),
                    })
                    if expert_test_losses:
                        test_best = min(range(len(expert_test_losses)),
                                        key=expert_test_losses.__getitem__)
                        perg_diag.update({
                            "selector_expert_test_losses": expert_test_losses,
                            "selector_test_best_index": int(test_best),
                            "selector_decision_agreement": bool(best == test_best),
                            "selector_test_gain_vs_control": float(
                                expert_test_losses[0] - expert_test_losses[best]
                            ),
                        })
            else:
                global_flat = torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])
                losses = torch.tensor(calibration_losses, dtype=torch.float32, device=self.device)
                if cfg.method == "FedLAW":
                    aggregated, client_weights = self._fedlaw_update(deltas, base_weights)
                elif cfg.method == 'FedAWA':
                    aggregated,client_weights=self.fedawa.aggregate(deltas,base_weights,global_flat,selected)
                elif cfg.method == 'FedDS':
                    aggregated,client_weights,ds_diag=self._fedds_update(deltas)
                    perg_diag.update(ds_diag)
                else:
                    aggregated, client_weights = self.baseline.aggregate(
                        deltas, base_weights, imp_tensor, step_tensor, round_idx,
                        global_flat, losses if cfg.method == "FedLBW" else None)
            add_delta_(self.model, aggregated, cfg.server_lr)
            average_snapshot(round_idx)
            if self.perg is not None:
                # Strict temporal holdout: all current-round scoring and the
                # global update precede training on current masked targets.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                predictor_train_started = time.perf_counter()
                perg_diag["meta_loss"] = self.perg.update_predictor(
                    reference_deltas, base_weights, imp_tensor, round_idx
                )
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                perg_diag["predictor_train_seconds"] = (
                    time.perf_counter() - predictor_train_started
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            agg_seconds = time.perf_counter() - agg_started
            flat_updates = torch.stack([flatten_delta(d) for d in deltas])
            mean_update = (flat_updates * base_weights[:, None]).sum(dim=0)
            cosines = F.cosine_similarity(flat_updates, mean_update[None], dim=1)
            entropy = float(-(client_weights.clamp_min(1e-9) * client_weights.clamp_min(1e-9).log()).sum())
            num_parameters = sum(p.numel() for p in self.model.parameters())
            round_bytes = len(selected) * num_parameters * 4
            extra_scalar_bytes = (4 * len(selected)
                                  if cfg.method in {"FedPERG", "LossWeightedM"} else 0)
            total_communication += round_bytes + extra_scalar_bytes
            diagnostics.append(
                {
                    "round": round_idx,
                    "selected_clients": selected.tolist(),
                    "client_sizes": sizes,
                    "loss_improvements": improvements,
                    "local_steps": steps,
                    "mean_update_cosine": float(cosines.mean()),
                    "min_update_cosine": float(cosines.min()),
                    "weight_entropy": entropy,
                    "aggregate_norm": float(flatten_delta(aggregated).norm()),
                    "local_seconds": local_seconds,
                    "aggregation_seconds": agg_seconds,
                    "communication_mb": round_bytes / 1e6,
                    "extra_scalar_bytes": extra_scalar_bytes,
                    "reward_diagnostics": self.round_reward_diagnostics,
                    **perg_diag,
                }
            )
            if cfg.simulated_network:
                parallel_cost=max(float(count)/[10.,20.,50.,100.][int(cid)%4]
                    + num_parameters*4*8/(1e6*[5.,20.,50.,100.][int(cid)%4])+.02
                    for cid,count in zip(selected,steps))
                diagnostics[-1]['modeled_parallel_round_seconds']=parallel_cost+agg_seconds
                diagnostics[-1]['distributed_validation_bytes']=(
                    cfg.calibration_sites*7*(num_parameters*4+12)
                    if cfg.calibration_mode=='distributed' else 0)
            if round_idx % cfg.eval_every == 0 or round_idx == cfg.rounds:
                current = self._evaluate(round_idx)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                current["elapsed_seconds"] = time.perf_counter() - started
                history.append(current)
                if verbose:
                    print(
                        f"[{cfg.dataset}/{cfg.regime}/{cfg.method}/s{cfg.seed}] "
                        f"round={round_idx:02d} acc={current['accuracy']:.4f} "
                        f"worst20={current['worst20_accuracy']:.4f} ece={current['ece']:.4f}",
                        flush=True,
                    )
        elapsed = time.perf_counter() - started
        if int(history[-1]['round']) != cfg.rounds:
            final_eval = self._evaluate(cfg.rounds)
            final_eval['elapsed_seconds'] = time.perf_counter() - started
            history.append(final_eval)
        peak_memory_mb = (torch.cuda.max_memory_allocated(self.device) / 1e6
                          if self.device.type == "cuda" else None)
        accuracies = np.asarray([float(v["accuracy"]) for v in history])
        rounds = np.asarray([int(v["round"]) for v in history])
        auc = float(np.trapz(accuracies, rounds) / max(1, cfg.rounds))
        result: Dict[str, object] = {
            "config": cfg.to_dict(),
            "device": str(self.device),
            "history": history,
            "diagnostics": diagnostics,
            "final": {**history[-1], "convergence_auc": auc},
            "runtime_seconds": elapsed,
            "peak_gpu_memory_mb": peak_memory_mb,
            "communication_mb": total_communication / 1e6,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "status": "complete",
        }
        if cfg.record_development_validation:
            validation_accuracy = np.asarray([h['development']['accuracy'] for h in history])
            result['development_validation'] = {
                **history[-1]['development'],
                'convergence_auc': float(np.trapz(validation_accuracy, rounds) / max(1, cfg.rounds)),
                'examples': len(self.confidence_indices),
            }
        if len(self.confidence_indices):
            # Separate labels, removed from BOTH client SGD and expert routing.
            # The scalar fit never uses test labels or candidate selection loss.
            from scipy.optimize import minimize_scalar
            cal_logits = predict(self.model, self.data.x_train[self.confidence_indices], self.device)
            cal_labels = self.data.y_train[self.confidence_indices]
            fitted = minimize_scalar(lambda log_t: float(F.cross_entropy(
                cal_logits / math.exp(log_t), cal_labels)), bounds=(-3.,3.), method='bounded')
            temperature = math.exp(float(fitted.x))
            test_logits = predict(self.model, self.data.x_test, self.device)
            raw_metrics = classification_metrics(test_logits, self.data.y_test, self.data.classes)
            scaled_metrics = classification_metrics(test_logits/temperature, self.data.y_test, self.data.classes)
            assert raw_metrics['accuracy'] == scaled_metrics['accuracy']
            result['confidence_calibration'] = {'temperature': temperature,
                'independent_fit_examples':len(self.confidence_indices),
                'raw':raw_metrics,'temperature_scaled':scaled_metrics}
        if averaged_state is not None:
            terminal_state = clone_state(self.model)
            self.model.load_state_dict(averaged_state)
            result['terminal_average'] = {'rounds': averaged_rounds, **self._evaluate(cfg.rounds)}
            self.model.load_state_dict(terminal_state)
        self._save(result, averaged_state)
        return result

    def _save(self, result: Dict[str, object], averaged_state=None) -> None:
        root = Path(self.cfg.results_dir)
        models = root / "models" / self.cfg.model_group
        models.mkdir(parents=True, exist_ok=True)
        stem = f"{self.cfg.dataset}__{self.cfg.regime}__{_safe_name(self.cfg.method)}__s{self.cfg.seed}"
        tags = []
        if self.cfg.perg_variant != "full":
            tags.append(_safe_name(self.cfg.perg_variant))
        if self.cfg.experiment_tag:
            tags.append(_safe_name(self.cfg.experiment_tag))
        if tags:
            stem += "__" + "__".join(tags)
        json_path = models / f"{stem}.json"
        snapshot = {"model_state": self.model.state_dict(), "config": self.cfg.to_dict()}
        if averaged_state is not None:
            snapshot['terminal_average_state'] = averaged_state
        model_path = models / f"{stem}.pth"
        model_temporary = model_path.with_suffix('.pth.tmp')
        torch.save(snapshot, model_temporary)
        model_temporary.replace(model_path)
        if self.perg is not None:
            operator_path = models / f"{stem}__operator.pth"
            operator_temporary = operator_path.with_suffix('.pth.tmp')
            torch.save(
                {"operator_state": self.perg.operator.state_dict(), "config": self.cfg.to_dict()},
                operator_temporary,
            )
            operator_temporary.replace(operator_path)
        json_temporary = json_path.with_suffix('.json.tmp')
        json_temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        json_temporary.replace(json_path)
