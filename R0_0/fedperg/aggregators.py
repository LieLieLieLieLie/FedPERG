from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import flatten_delta, unflatten_like, weighted_delta


def _cos(a: Tensor, b: Tensor) -> Tensor:
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=1).squeeze(0)


def _robust_z(x: Tensor) -> Tensor:
    med = x.median()
    mad = (x - med).abs().median().clamp_min(1e-4)
    return (x - med) / (1.4826 * mad)


def _pool_sketch(vector: Tensor, size: int) -> Tensor:
    if vector.numel() < size:
        return F.pad(vector, (0, size - vector.numel()))
    return F.adaptive_avg_pool1d(vector.reshape(1, 1, -1), size).reshape(-1)


def cached_leave_one_residual(vectors: Tensor, base_weights: Tensor) -> Tensor:
    """Mass-weighted leave-one residuals in O(KP) arithmetic.

    ``vectors`` has shape ``[clients, parameters]``. The cached cohort sum is
    algebraically identical to renormalizing the visible-client weights for
    every held-out client, but avoids constructing K masks and recomputing K
    weighted sums.
    """
    if vectors.ndim != 2 or base_weights.ndim != 1:
        raise ValueError("Expected vectors [K,P] and base_weights [K]")
    if vectors.shape[0] != base_weights.numel() or vectors.shape[0] < 2:
        raise ValueError("Leave-one residual requires matching K >= 2")
    total_mass = base_weights.sum()
    remaining_mass = (total_mass - base_weights).clamp_min(1e-12)
    cohort_sum = (vectors * base_weights[:, None]).sum(dim=0)
    references = (
        cohort_sum[None, :] - base_weights[:, None] * vectors
    ) / remaining_mass[:, None]
    denominator = vectors.norm(dim=1) + references.norm(dim=1) + 1e-12
    return ((vectors - references).norm(dim=1) / denominator).pow(2)


class AxialSetBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mode: str = "full") -> None:
        super().__init__()
        self.mode = mode
        self.client_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.layer_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm_c = nn.LayerNorm(hidden)
        self.norm_l = nn.LayerNorm(hidden)
        self.norm_f = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, x: Tensor) -> Tensor:
        # x: clients x layers x hidden.  No client positional embedding is used.
        if self.mode in {"full", "client_only"}:
            across_clients = x.transpose(0, 1)
            update, _ = self.client_attn(across_clients, across_clients, across_clients, need_weights=False)
            x = self.norm_c(x + update.transpose(0, 1))
        if self.mode in {"full", "tensor_only"}:
            update, _ = self.layer_attn(x, x, x, need_weights=False)
            x = self.norm_l(x + update)
        return self.norm_f(x + self.ff(x))


class ClientLayerSetOperator(nn.Module):
    """Permutation-equivariant client/layer operator used by FedPERG."""

    def __init__(self, feature_dim: int, layers: int, hidden: int = 48, heads: int = 4,
                 mode: str = "full", reconstruction_dim: int = 16,
                 use_score: bool = True) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim + 1, hidden)
        self.layer_embedding = nn.Parameter(torch.randn(layers, hidden) * 0.02)
        self.blocks = nn.ModuleList([AxialSetBlock(hidden, heads, mode), AxialSetBlock(hidden, heads, mode)])
        self.score = (nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
                      if use_score else None)
        self.reconstruct = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, reconstruction_dim))

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if mask is None:
            mask = torch.zeros(features.shape[:2], device=features.device, dtype=torch.bool)
        visible = features.masked_fill(mask.unsqueeze(-1), 0.0)
        x = torch.cat([visible, mask.float().unsqueeze(-1)], dim=-1)
        x = self.input(x) + self.layer_embedding.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        score = (self.score(x).squeeze(-1) if self.score is not None
                 else torch.zeros(features.shape[:2], device=features.device,
                                  dtype=features.dtype))
        return score, self.reconstruct(x)


@dataclass
class PERGOutput:
    delta: List[Tensor]
    weights: Tensor
    reconstruction_error: Tensor
    rho: float
    meta_loss: float
    prediction_seconds: float
    fusion_bias: float
    attentive_bias: float
    control_delta: Optional[List[Tensor]] = None
    expert_deltas: Optional[List[List[Tensor]]] = None
    expert_names: Optional[List[str]] = None


class PERGAggregator:
    def __init__(
        self,
        reference: Sequence[Tensor],
        hidden: int,
        heads: int,
        lr: float,
        meta_steps: int,
        rho_max: float,
        device: torch.device,
        variant: str = "full",
        sketch_dim: int = 12,
        temperature: float = 0.8,
        reconstruction_weight: float = 0.85,
        median_weight: float = 1.20,
        memory_weight: float = 0.40,
        gate_low: float = 0.82,
        gate_span: float = 0.32,
    ) -> None:
        self.reference = [x.detach().clone() for x in reference]
        self.sketch_dim = sketch_dim
        self.feature_dim = sketch_dim + 9
        self.target_dim = sketch_dim + 4
        architecture = variant if variant in {"client_only", "tensor_only", "mlp"} else "full"
        lite_variants = {"pruned_lite", "pruned_lite_noscore", "lite_frozen",
                         "lite_no_msr", "lite_simple_residual",
                         "lite_residual_only",
                         "lite_residual_logit_off_dynamic",
                         "lite_residual_logit_on_fixed",
                         "lite_residual_gate_fixed",
                         "lite_residual_gate_dynamic",
                         "lite_full_gate_fixed",
                         "lite_residual_gate_unmixed",
                         "lite_full_gate_unmixed",
                         "lite_full_gate_selector",
                         "lite_full_direction_selector",
                         "lite_gate_bank_selector",
                         "lite_gate_bank_router_control",
                         "lite_paired_gate_bank_selector",
                         "lite_paired_gate_bank_no_residual",
                         "lite_paired_gate_bank_router_control"}
        use_score = variant not in lite_variants
        self.operator = ClientLayerSetOperator(
            self.feature_dim, len(reference), hidden, heads, architecture,
            self.target_dim, use_score=use_score
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.operator.parameters(), lr=lr, weight_decay=1e-4)
        self.meta_steps = meta_steps
        self.rho_max = rho_max
        self.device = device
        self.variant = variant
        self.temperature = temperature
        self.reconstruction_weight = reconstruction_weight
        self.median_weight = median_weight
        self.memory_weight = memory_weight
        self.gate_low = gate_low
        self.gate_span = gate_span
        self.momentum = [torch.zeros_like(x) for x in reference]
        self._pending_round: Optional[int] = None
        self._pending_memory: Optional[List[Tensor]] = None
        self.replay: List[Tuple[List[List[Tensor]], Tensor, Tensor, List[Tensor]]] = []

    def _features(
        self,
        deltas: Sequence[Sequence[Tensor]],
        base_weights: Tensor,
        improvements: Tensor,
        mask: Optional[Tensor] = None,
        momentum_snapshot: Optional[Sequence[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        n, layers = len(deltas), len(deltas[0])
        if mask is None:
            mask = torch.zeros(n, layers, device=self.device, dtype=torch.bool)
        features = torch.zeros(n, layers, self.feature_dim, device=self.device)
        targets = torch.zeros(n, layers, self.target_dim, device=self.device)
        cos_med = torch.zeros(n, layers, device=self.device)
        cos_mom = torch.zeros(n, layers, device=self.device)
        norms = torch.zeros(n, layers, device=self.device)
        for layer in range(layers):
            vectors = torch.stack([d[layer].reshape(-1) for d in deltas])
            visible = ~mask[:, layer]
            if not visible.any():
                raise ValueError("At least one client per tensor must remain visible")
            median = vectors[visible].median(dim=0).values
            observed_weights = base_weights[visible] / base_weights[visible].sum()
            visible_mass = torch.zeros_like(base_weights)
            visible_mass[visible] = observed_weights
            mean = (vectors[visible] * observed_weights[:, None]).sum(dim=0)
            memory = self.momentum if momentum_snapshot is None else momentum_snapshot
            mom = memory[layer].reshape(-1)
            for client in range(n):
                v = vectors[client]
                sketch = _pool_sketch(v, self.sketch_dim)
                norm = v.norm().clamp_min(1e-12)
                norms[client, layer] = norm
                cos_med[client, layer] = _cos(v, median)
                cos_mom[client, layer] = _cos(v, mom) if mom.norm() > 0 else _cos(v, mean)
                own = torch.stack([norm.log1p(), v.mean(), v.std(unbiased=False), v.sign().mean()])
                targets[client, layer] = torch.cat([sketch, own])
                tail = torch.stack(
                    [
                        *own,
                        _cos(v, mean),
                        cos_med[client, layer],
                        cos_mom[client, layer],
                        visible_mass[client],
                        improvements[client],
                    ]
                )
                features[client, layer] = torch.cat([sketch, tail])
        # All cohort statistics, including the normalization, use only visible
        # updates. A hidden update is present solely in its loss target.
        center = torch.zeros(1, layers, self.feature_dim, device=self.device)
        scale = torch.zeros_like(center)
        for layer in range(layers):
            observed = features[~mask[:, layer], layer]
            center[0, layer] = observed.mean(dim=0)
            scale[0, layer] = observed.std(dim=0, unbiased=False).clamp_min(1e-4)
        standardized = ((features - center) / scale).clamp(-8.0, 8.0)
        target_standardized = ((targets - center[:, :, :self.target_dim]) /
                               scale[:, :, :self.target_dim]).clamp(-8.0, 8.0)
        return standardized, target_standardized, {"cos_med": cos_med, "cos_mom": cos_mom, "norms": norms}

    def _visible_subset_consistency(
        self, masked_features: Tensor, mask: Tensor, generator: torch.Generator
    ) -> Tensor:
        """Both views use only clients visible under this training mask."""
        visible_clients = (~mask).all(dim=1)
        visible_features = masked_features[visible_clients]
        if len(visible_features) < 3:
            return masked_features.new_zeros(())
        keep = torch.rand(len(visible_features), generator=generator,
                          device=self.device) > 0.2
        if keep.sum() < 2:
            keep[:2] = True
        subset_score, _ = self.operator(visible_features[keep])
        full_score, _ = self.operator(visible_features)
        return (subset_score.mean(dim=0) - full_score[keep].mean(dim=0)).pow(2).mean()

    def _training_mask(self, n: int, layers: int, generator: torch.Generator,
                       round_idx: int, step_index: int) -> Tensor:
        """Draw a nonempty client mask; only the legacy cell ablation masks cells."""
        if self.variant == "cell_mask":
            mask = torch.rand((n, layers), generator=generator,
                              device=self.device) < 0.28
        else:
            clients = torch.rand(n, generator=generator,
                                 device=self.device) < 0.28
            mask = clients[:, None].expand(n, layers).clone()
        if not mask.any():
            client = (round_idx + step_index) % n
            if self.variant == "cell_mask":
                mask[client, (round_idx + step_index) % layers] = True
            else:
                mask[client, :] = True
        if self.variant != "cell_mask":
            visible_count = int((~mask[:, 0]).sum())
            for client in range(n):
                if visible_count >= 2:
                    break
                if mask[client, 0]:
                    mask[client, :] = False
                    visible_count += 1
            assert torch.equal(mask, mask[:, :1].expand_as(mask))
            if not mask.any():
                raise ValueError("Whole-client masking requires at least three clients")
        elif (~mask).sum(dim=0).min() < 2:
            mask[0] = False
            mask[1] = False
        return mask

    def update_predictor(
        self,
        deltas: Sequence[Sequence[Tensor]],
        base_weights: Tensor,
        improvements: Tensor,
        round_idx: int,
    ) -> float:
        """Train on round t only after its global update is committed.

        The pre-round velocity is used for masked training features: the
        just-computed velocity already contains the round's hidden uploads.
        """
        if self._pending_round != round_idx or self._pending_memory is None:
            raise RuntimeError("Call aggregate, then commit the global step, before update_predictor")
        observed_improvements = torch.zeros_like(improvements) if self.variant == "no_r" else improvements
        n, layers = len(deltas), len(deltas[0])
        generator = torch.Generator(device=self.device).manual_seed(7717 + round_idx)
        no_training = {"frozen", "global_weights_frozen", "residual_frozen",
                       "replay_residual_frozen", "handcrafted", "gate_only",
                       "memory_feature_only", "lite_frozen", "lite_no_msr",
                       "lite_simple_residual", "lite_residual_only",
                       "lite_residual_logit_off_dynamic",
                       "lite_residual_logit_on_fixed",
                       "lite_residual_gate_fixed",
                       "lite_residual_gate_dynamic",
                       "lite_full_gate_fixed",
                       "lite_residual_gate_unmixed",
                       "lite_full_gate_unmixed",
                       "lite_full_gate_selector",
                       "lite_full_direction_selector",
                       "lite_gate_bank_selector",
                       "lite_gate_bank_router_control",
                       "lite_paired_gate_bank_selector",
                       "lite_paired_gate_bank_no_residual",
                       "lite_paired_gate_bank_router_control"}
        no_reconstruction = {"no_msr", "global_weights_no_msr", "replay_no_msr",
                             "lite_no_msr"}
        train_steps = 0 if self.variant in no_training else self.meta_steps
        losses: List[float] = []
        self.operator.train()
        current_record = (
            [[tensor.detach().cpu().clone() for tensor in client] for client in deltas],
            base_weights.detach().cpu().clone(), observed_improvements.detach().cpu().clone(),
            [tensor.detach().cpu().clone() for tensor in self._pending_memory],
        )
        pool = [*self.replay, current_record] if self.variant.startswith("replay") else [current_record]
        for step_index in range(train_steps):
            record = pool[-1 - (step_index % len(pool))]
            step_deltas = [[tensor.to(self.device) for tensor in client] for client in record[0]]
            step_weights = record[1].to(self.device)
            step_improvements = record[2].to(self.device)
            step_memory = [tensor.to(self.device) for tensor in record[3]]
            step_n, step_layers = len(step_deltas), len(step_deltas[0])
            mask = self._training_mask(step_n, step_layers, generator,
                                       round_idx, step_index)
            masked_features, targets, _ = self._features(
                step_deltas, step_weights, step_improvements, mask,
                momentum_snapshot=step_memory
            )
            score, reconstruction = self.operator(masked_features, mask)
            residual = (reconstruction - targets).pow(2).mean(dim=-1)
            loss_cons = self._visible_subset_consistency(masked_features, mask, generator)
            if self.variant in no_reconstruction:
                loss = 0.05 * loss_cons
            else:
                loss_recon = residual[mask].mean()
                if self.variant in {"pruned_lite", "pruned_lite_noscore"}:
                    loss = loss_recon + 0.05 * loss_cons
                else:
                    loss_score = F.smooth_l1_loss(score[mask], -residual.detach()[mask])
                    loss = loss_recon + 0.25 * loss_score + 0.05 * loss_cons
            if not loss.requires_grad:
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.operator.parameters(), 2.0)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        if self.variant.startswith("replay"):
            self.replay.append(current_record)
            self.replay = self.replay[-4:]
        self._pending_round = None
        self._pending_memory = None
        return sum(losses) / len(losses) if losses else 0.0

    def aggregate(
        self,
        deltas: Sequence[Sequence[Tensor]],
        base_weights: Tensor,
        improvements: Tensor,
        round_idx: int,
    ) -> PERGOutput:
        if self._pending_round is not None:
            raise RuntimeError("Previous round predictor update has not completed")
        self._pending_round = round_idx
        self._pending_memory = [m.detach().clone() for m in self.momentum]
        observed_improvements = torch.zeros_like(improvements) if self.variant == "no_r" else improvements
        features, _, raw = self._features(deltas, base_weights, observed_improvements)
        n, layers, _ = features.shape
        self.operator.eval()
        reconstruction_error = torch.zeros(n, layers, device=self.device)
        learned_score = torch.zeros(n, layers, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prediction_started = time.perf_counter()
        if self.variant in {"lite_simple_residual", "lite_residual_only",
                            "lite_residual_logit_off_dynamic",
                            "lite_residual_logit_on_fixed",
                            "lite_residual_gate_fixed",
                            "lite_residual_gate_dynamic",
                            "lite_full_gate_fixed",
                            "lite_residual_gate_unmixed",
                            "lite_full_gate_unmixed",
                            "lite_full_gate_selector",
                            "lite_full_direction_selector",
                            "lite_gate_bank_selector",
                            "lite_gate_bank_router_control",
                            "lite_paired_gate_bank_selector",
                            "lite_paired_gate_bank_no_residual",
                            "lite_paired_gate_bank_router_control"}:
            # Deterministic leave-one-client-out residual: no learned parameters,
            # no current-client leakage, and the same uploaded tensors as Lite.
            # A cached mass-weighted cohort sum makes this O(KP), rather than
            # recomputing K visible-client sums for O(K^2 P) arithmetic.
            with torch.no_grad():
                for layer in range(layers):
                    vectors = torch.stack([client[layer].reshape(-1) for client in deltas])
                    reconstruction_error[:, layer] = cached_leave_one_residual(
                        vectors, base_weights
                    )
        elif self.variant != "lite_no_msr":
            with torch.no_grad():
                for client in range(n):
                    mask = torch.zeros((n, layers), dtype=torch.bool, device=self.device)
                    mask[client, :] = True
                    masked_features, targets, _ = self._features(
                        deltas, base_weights, observed_improvements, mask
                    )
                    score, reconstruction = self.operator(masked_features, mask)
                    reconstruction_error[client] = (
                        reconstruction[client] - targets[client]
                    ).pow(2).mean(dim=-1)
                    learned_score[client] = score[client]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        prediction_seconds = time.perf_counter() - prediction_started

        norm_penalty = _robust_z(raw["norms"].log1p()).abs().clamp_max(5.0)
        imp = _robust_z(observed_improvements).clamp(-4.0, 4.0).unsqueeze(1)
        rec_z = torch.stack([_robust_z(reconstruction_error[:, j]) for j in range(layers)], dim=1)
        score_coefficient = (0.0 if self.variant in {"handcrafted", "gate_only", "memory_feature_only",
                                                     "residual_only", "residual_frozen",
                                                     "replay_residual", "replay_residual_frozen",
                                                     "drop_score_logit", "pruned_lite",
                                                     "pruned_lite_noscore", "lite_frozen",
                                                      "lite_no_msr", "lite_simple_residual",
                                                      "lite_residual_only",
                                                      "lite_residual_logit_off_dynamic",
                                                      "lite_residual_logit_on_fixed",
                                                      "lite_residual_gate_fixed",
                                                      "lite_residual_gate_dynamic",
                                                      "lite_full_gate_fixed",
                                                      "lite_residual_gate_unmixed",
                                                      "lite_full_gate_unmixed",
                                                      "lite_full_gate_selector",
                                                      "lite_full_direction_selector",
                                                      "lite_gate_bank_selector",
                                                       "lite_gate_bank_router_control",
                                                       "lite_paired_gate_bank_selector",
                                                       "lite_paired_gate_bank_no_residual",
                                                       "lite_paired_gate_bank_router_control"}
                             else 1.0)
        residual_coefficient = (0.0 if self.variant in {"handcrafted", "gate_only", "memory_feature_only",
                                                       "drop_residual_logit", "lite_no_msr",
                                                       "lite_residual_logit_off_dynamic",
                                                       "lite_paired_gate_bank_no_residual"}
                                else 1.0)
        handcrafted_coefficient = (0.0 if self.variant in {"learned_only", "gate_only",
                                                            "memory_feature_only",
                                                            "lite_residual_only",
                                                            "lite_residual_gate_fixed",
                                                            "lite_residual_gate_dynamic",
                                                            "lite_residual_gate_unmixed"}
                                   else 1.0)
        improvement_coefficient = 0.0 if self.variant in {"no_r", "drop_improvement_logit"} else 1.0
        median_coefficient = 0.0 if self.variant == "drop_median_logit" else 1.0
        memory_coefficient = 0.0 if self.variant in {"drop_memory_logit", "pruned_lite",
                                                      "pruned_lite_noscore", "lite_frozen",
                                                      "lite_no_msr", "lite_simple_residual",
                                                      "lite_residual_only",
                                                      "lite_residual_logit_off_dynamic",
                                                      "lite_residual_logit_on_fixed",
                                                      "lite_residual_gate_fixed",
                                                      "lite_residual_gate_dynamic",
                                                      "lite_full_gate_fixed",
                                                      "lite_residual_gate_unmixed",
                                                      "lite_full_gate_unmixed",
                                                      "lite_full_gate_selector",
                                                      "lite_full_direction_selector",
                                                      "lite_gate_bank_selector",
                                                       "lite_gate_bank_router_control",
                                                       "lite_paired_gate_bank_selector",
                                                       "lite_paired_gate_bank_no_residual",
                                                       "lite_paired_gate_bank_router_control"} else 1.0
        norm_coefficient = 0.0 if self.variant in {"drop_norm_logit", "pruned_lite",
                                                    "pruned_lite_noscore", "lite_frozen",
                                                    "lite_no_msr", "lite_simple_residual",
                                                    "lite_residual_only",
                                                    "lite_residual_logit_off_dynamic",
                                                    "lite_residual_logit_on_fixed",
                                                    "lite_residual_gate_fixed",
                                                    "lite_residual_gate_dynamic",
                                                    "lite_full_gate_fixed",
                                                    "lite_residual_gate_unmixed",
                                                    "lite_full_gate_unmixed",
                                                    "lite_full_gate_selector",
                                                    "lite_full_direction_selector",
                                                    "lite_gate_bank_selector",
                                                     "lite_gate_bank_router_control",
                                                     "lite_paired_gate_bank_selector",
                                                     "lite_paired_gate_bank_no_residual",
                                                     "lite_paired_gate_bank_router_control"} else 1.0
        logits = (
            score_coefficient * 0.35 * learned_score
            - residual_coefficient * self.reconstruction_weight * rec_z
            + handcrafted_coefficient * (median_coefficient * self.median_weight * raw["cos_med"] +
                                         memory_coefficient * self.memory_weight * raw["cos_mom"]
                                         - norm_coefficient * 0.20 * norm_penalty)
            + improvement_coefficient * handcrafted_coefficient * 0.35 * imp
            + base_weights.clamp_min(1e-8).log().unsqueeze(1)
        )
        if self.variant == "memory_feature_only":
            logits = self.memory_weight * raw["cos_mom"] + base_weights.clamp_min(1e-8).log().unsqueeze(1)
        attentive = torch.softmax(logits / self.temperature, dim=0)
        uncertainty = (0.5 if self.variant in {"handcrafted", "gate_only", "memory_feature_only",
                                               "lite_no_msr", "lite_residual_logit_on_fixed",
                                               "lite_residual_gate_fixed", "lite_full_gate_fixed"}
                       else float(torch.sigmoid(reconstruction_error.median() - 1.0)))
        rho = float(self.rho_max * (1.0 - 0.45 * uncertainty))
        if self.variant == "gate_only":
            rho = 0.0
        if self.variant == "no_conservative":
            rho = 1.0
        if self.variant in {"lite_residual_gate_unmixed", "lite_full_gate_unmixed",
                            "lite_full_gate_selector",
                            "lite_full_direction_selector",
                            "lite_gate_bank_selector",
                            "lite_gate_bank_router_control",
                            "lite_paired_gate_bank_selector",
                            "lite_paired_gate_bank_no_residual",
                            "lite_paired_gate_bank_router_control"}:
            rho = 1.0
        weights = (1.0 - rho) * base_weights[:, None] + rho * attentive
        weights = weights / weights.sum(dim=0, keepdim=True)
        if self.variant in {"lite_gate_bank_router_control",
                            "lite_paired_gate_bank_router_control"}:
            # Equal-resource attribution control: compute the same evidence
            # pipeline but prevent it from affecting any candidate operator.
            # Every gate uses sample-mass sign agreement C^p.
            weights = base_weights[:, None].expand_as(weights)
        if self.variant in {"global_weights", "global_weights_no_msr", "global_weights_frozen"}:
            weights = weights.mean(dim=1, keepdim=True).expand_as(weights)
            weights = weights / weights.sum(dim=0, keepdim=True)

        output: List[Tensor] = []
        control_output: List[Tensor] = []
        bank_variants = {"lite_gate_bank_selector", "lite_gate_bank_router_control",
                         "lite_paired_gate_bank_selector",
                         "lite_paired_gate_bank_no_residual",
                         "lite_paired_gate_bank_router_control"}
        paired_bank = self.variant in {"lite_paired_gate_bank_selector",
                                       "lite_paired_gate_bank_no_residual",
                                       "lite_paired_gate_bank_router_control"}
        expert_outputs: Optional[List[List[Tensor]]] = (
            [[] for _ in range(7 if paired_bank else 4)]
            if self.variant in bank_variants else None
        )
        base_aggregates: List[Tensor] = []
        fused_aggregates: List[Tensor] = []
        attentive_aggregates: List[Tensor] = []
        for layer in range(layers):
            stacked = torch.stack([deltas[client][layer] for client in range(n)])
            view_shape = (n,) + (1,) * (stacked.ndim - 1)
            layer_weights = weights[:, layer].reshape(view_shape)
            agg = (stacked * layer_weights).sum(dim=0)
            base_agg = (stacked * base_weights.reshape(view_shape)).sum(dim=0)
            attentive_agg = (stacked * attentive[:, layer].reshape(view_shape)).sum(dim=0)
            base_aggregates.append(base_agg)
            fused_aggregates.append(agg)
            attentive_aggregates.append(attentive_agg)

            # The conservative coordinate gate cannot reverse the weighted
            # direction: it only contracts conflicted coordinates and mildly
            # amplifies coordinates supported by most of the cohort.
            base_view = base_weights.reshape(view_shape)
            consensus_sign = torch.sign((stacked * base_view).sum(dim=0))
            agreement_mass = (
                (torch.sign(stacked) == consensus_sign.unsqueeze(0)).float() * layer_weights
            ).sum(dim=0)
            coordinate_gate = self.gate_low + self.gate_span * agreement_mass
            base_agreement_mass = (
                (torch.sign(stacked) == consensus_sign.unsqueeze(0)).float() * base_view
            ).sum(dim=0)
            base_coordinate_gate = self.gate_low + self.gate_span * base_agreement_mass
            if self.variant == "no_coordinate_gate":
                coordinate_gate = torch.ones_like(coordinate_gate)
            if self.variant in {"lite_residual_gate_fixed",
                                "lite_residual_gate_dynamic",
                                "lite_full_gate_fixed",
                                "lite_residual_gate_unmixed",
                                "lite_full_gate_unmixed",
                                "lite_full_gate_selector",
                                "lite_gate_bank_selector",
                                "lite_gate_bank_router_control",
                                "lite_paired_gate_bank_selector",
                                "lite_paired_gate_bank_no_residual",
                                "lite_paired_gate_bank_router_control"}:
                # Preserve the sample-mass direction and use target-excluded
                # evidence only to calibrate the bounded coordinate gate.
                agg = base_agg
            agg = agg * coordinate_gate

            # A causal velocity token uses the same 0.9 momentum budget as
            # FedAvgM; this makes gains attributable to the client-layer
            # operator rather than to a weaker or stronger server optimizer.
            previous = self.momentum[layer]
            accelerated = 0.9 * previous + agg
            control_output.append(0.9 * previous + base_agg * base_coordinate_gate)
            if expert_outputs is not None:
                # Expert 0 is the exact matched AvgM+Gate control. The final
                # paired bank then exposes sample-mass/evidence expert pairs at
                # each span, enabling an exact equal-resource no-evidence test.
                expert_outputs[0].append(control_output[-1])
                if paired_bank:
                    for level, (low, span) in enumerate(
                        ((0.90, 0.40), (0.98, 0.48), (1.06, 0.56))
                    ):
                        router_gate = low + span * base_agreement_mass
                        evidence_gate = low + span * agreement_mass
                        expert_outputs[1 + 2 * level].append(
                            0.9 * previous + base_agg * router_gate)
                        expert_outputs[2 + 2 * level].append(
                            0.9 * previous + base_agg * evidence_gate)
                else:
                    for expert_index, (low, span) in enumerate(
                        ((0.90, 0.40), (0.98, 0.48), (1.06, 0.56)), start=1
                    ):
                        expert_gate = low + span * agreement_mass
                        expert_outputs[expert_index].append(
                            0.9 * previous + base_agg * expert_gate)
            if self.variant == "no_memory":
                accelerated = agg
            output.append(accelerated)
            self.momentum[layer].copy_(accelerated)
        base_norm = torch.sqrt(sum(x.pow(2).sum() for x in base_aggregates)).clamp_min(1e-12)
        fusion_bias = torch.sqrt(sum((x - b).pow(2).sum()
                                     for x, b in zip(fused_aggregates, base_aggregates))) / base_norm
        attentive_bias = torch.sqrt(sum((x - b).pow(2).sum()
                                        for x, b in zip(attentive_aggregates, base_aggregates))) / base_norm
        return PERGOutput(output, weights.detach(), reconstruction_error.detach(), rho, 0.0,
                           prediction_seconds, float(fusion_bias), float(attentive_bias),
                           control_delta=control_output,
                           expert_deltas=expert_outputs,
                           expert_names=(
                               ["AvgM+Gate", "Router-C", "PERG-C", "Router-B",
                                "PERG-B", "Router-A", "PERG-A"]
                               if paired_bank else
                               ["AvgM+Gate", "PERG-C", "PERG-B", "PERG-A"]
                               if self.variant == "lite_gate_bank_selector" else
                               ["AvgM+Gate", "Router-C", "Router-B", "Router-A"]
                               if expert_outputs is not None else None))


@dataclass
class ServerState:
    momentum: Optional[Tensor] = None
    second: Optional[Tensor] = None
    memory: List[Tensor] = field(default_factory=list)
    initial: Optional[Tensor] = None


class BaselineAggregator:
    def __init__(self, method: str, reference: Sequence[Tensor], seed: int = 0,
                 gate_low: float = 0.82, gate_span: float = 0.32) -> None:
        self.method = method
        self.reference = [x.detach().clone() for x in reference]
        flat = flatten_delta(reference)
        self.state = ServerState(initial=flat.clone())
        self.seed = seed
        self.gate_low = gate_low
        self.gate_span = gate_span

    def aggregate(
        self,
        deltas: Sequence[Sequence[Tensor]],
        base_weights: Tensor,
        improvements: Tensor,
        local_steps: Tensor,
        round_idx: int,
        global_flat: Tensor,
        public_losses: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], Tensor]:
        flat = torch.stack([flatten_delta(d) for d in deltas])
        weights = base_weights.clone()
        method = self.method
        if method == "FedLAW" and public_losses is not None:
            quality = torch.softmax(-public_losses / 0.18, dim=0)
            weights = 0.15 * base_weights + 0.85 * quality
            weights /= weights.sum()
        elif method == "FedPW":
            mean = (flat * base_weights[:, None]).sum(dim=0)
            consensus = F.cosine_similarity(flat, mean[None], dim=1)
            phase = min(1.0, round_idx / 20.0)
            score = (1.1 - 0.5 * phase) * consensus + (0.35 + 0.4 * phase) * _robust_z(improvements)
            weights = torch.softmax(score / 0.75 + base_weights.log(), dim=0)
        elif method == "LossWeightedM":
            weights = torch.softmax(base_weights.clamp_min(1e-8).log() +
                                    0.35 * _robust_z(improvements).clamp(-4, 4), dim=0)

        if method == "FedNova":
            normalized = flat / local_steps[:, None].clamp_min(1.0)
            step = (normalized * weights[:, None]).sum(dim=0) * (local_steps * weights).sum()
        elif method == "Fed-NGA":
            norms = flat.norm(dim=1).clamp_min(1e-10)
            scale = norms.median()
            step = ((flat / norms[:, None]) * weights[:, None]).sum(dim=0) * scale
        elif method == "FedPW":
            masked = flat.clone()
            threshold = torch.quantile(masked.abs(), 0.18, dim=1, keepdim=True)
            masked = torch.where(masked.abs() >= threshold, masked, torch.zeros_like(masked))
            mean_sign = torch.sign((masked * weights[:, None]).sum(dim=0))
            agree = (torch.sign(masked) == mean_sign).float()
            masked = masked * (1.0 + 0.18 * agree)
            step = (masked * weights[:, None]).sum(dim=0)
        elif method == "FedAvgM-Gate":
            # Exact sample-weighted coordinate gate used by FedPERG with
            # q_i=p_i and the same 0.9 unnormalized server momentum.
            base_step = (flat * base_weights[:, None]).sum(dim=0)
            agreement = (torch.sign(flat) == torch.sign(base_step)[None]).float()
            agreement_mass = (agreement * base_weights[:, None]).sum(dim=0)
            step = base_step * (self.gate_low + self.gate_span * agreement_mass)
        else:
            step = (flat * weights[:, None]).sum(dim=0)

        if method == "FedAdam":
            beta1, beta2 = 0.9, 0.99
            if self.state.momentum is None:
                self.state.momentum = torch.zeros_like(step)
                self.state.second = torch.zeros_like(step)
            self.state.momentum.mul_(beta1).add_(step, alpha=1 - beta1)
            self.state.second.mul_(beta2).addcmul_(step, step, value=1 - beta2)
            adaptive = 0.08 * self.state.momentum / (self.state.second.sqrt() + 0.02)
            cap = 1.5 * step.norm().clamp_min(1e-8)
            step = adaptive * min(1.0, float(cap / adaptive.norm().clamp_min(1e-8)))
        elif method in {"FedAvgM", "LossWeightedM", "FedAvgM-Gate", "FedLWSM"}:
            if self.state.momentum is None:
                self.state.momentum = torch.zeros_like(step)
            self.state.momentum.mul_(0.9).add_(step)
            step = self.state.momentum.clone()
        elif method == "FedCDA":
            if self.state.memory:
                sims = torch.stack([_cos(step, past) for past in self.state.memory])
                historical = self.state.memory[int(sims.argmax())]
                blend = 0.22 * max(0.0, float(sims.max()))
                step = (1.0 - blend) * step + blend * historical
            self.state.memory.append(step.detach().clone())
            self.state.memory = self.state.memory[-6:]
        elif method == "FedPhoenix":
            generator = torch.Generator(device=step.device).manual_seed(self.seed * 1009 + round_idx)
            reset = torch.rand(step.shape, generator=generator, device=step.device) < 0.035
            target = self.state.initial.to(step.device) - global_flat
            step = torch.where(reset, target, step)

        if method in {"FedLWS", "FedLWSM"}:
            # Port of the author-released FedLWS layer-shrinkage equation to
            # this common harness. Adjacent weight/bias tensors form a layer.
            # This is not an official-code reproduction of their full protocol.
            if len(self.reference) % 2:
                raise ValueError("FedLWS port expects weight/bias tensor pairs")
            previous = unflatten_like(global_flat, self.reference)
            proposed = unflatten_like(step, self.reference)
            adjusted: List[Tensor] = []
            for start in range(0, len(previous), 2):
                group_previous = torch.cat([x.reshape(-1) for x in previous[start:start + 2]])
                group_delta = torch.cat([x.reshape(-1) for x in proposed[start:start + 2]])
                client_groups = torch.stack([
                    torch.cat([deltas[i][j].reshape(-1) for j in (start, start + 1)])
                    for i in range(len(deltas))
                ])
                client_mean = client_groups.mean(dim=0)
                variation = (client_groups - client_mean).norm(dim=1).mean()
                tau = (0.03 * variation).clamp(min=0.01, max=0.2)
                gamma = group_previous.norm() / (
                    group_previous.norm() + tau * group_delta.norm() + 1e-12
                )
                for j in (start, start + 1):
                    adjusted.append(gamma * (previous[j] + proposed[j]) - previous[j])
            step = flatten_delta(adjusted)

        return unflatten_like(step, self.reference), weights.detach()
