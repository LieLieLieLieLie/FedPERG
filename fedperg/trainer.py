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
from .data import FederatedData, load_federated_data
from .metrics import classification_metrics, fairness_metrics, predict
from .model import FeatureMLP, RawDigitCNN, RawRGBCNN, ResidualFeatureNet, add_delta_, clone_state, flatten_delta, state_delta


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
        )
        self.model_class = {"shallow": FeatureMLP, "residual_feature": ResidualFeatureNet,
                            "raw_cnn": RawDigitCNN, "raw_rgb_cnn": RawRGBCNN}[config.architecture]
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
        value = self._sample_loss(self.model, self.data.calibration_indices, limit=512)
        self.model.train(was_training)
        return value

    def _selector_test_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        count = min(512, len(self.data.y_test))
        with torch.no_grad():
            logits = self.model(self.data.x_test[:count].to(self.device))
            value = float(F.cross_entropy(logits, self.data.y_test[:count].to(self.device)))
        self.model.train(was_training)
        return value

    def _train_client(self, cid: int, round_idx: int) -> Tuple[List[Tensor], float, int, float]:
        cfg, method = self.cfg, self.cfg.method
        local = self.model_class(self.data.input_dim, cfg.hidden_dim, self.data.classes).to(self.device)
        local.load_state_dict(self.model.state_dict())
        global_params = [p.detach().clone() for p in self.model.parameters()]
        idx, _ = self._client_split(cid)
        generator = np.random.default_rng(cfg.seed * 100003 + round_idx * 101 + cid)
        generator.shuffle(idx)
        before = self._sample_loss(local, idx)
        optimizer = torch.optim.SGD(local.parameters(), lr=cfg.local_lr, momentum=0.0, weight_decay=cfg.weight_decay)
        epochs = cfg.local_epochs + int(cfg.regime != "label_skew" and cid % 4 == 0)
        steps = 0
        local.train()
        max_batches = 3 if cfg.dataset != "cifar100" else 2
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
                            local.parameters(), self.scaffold_server, self.scaffold_clients[cid]
                        ):
                            p.grad.add_(c_global - c_local)
                torch.nn.utils.clip_grad_norm_(local.parameters(), 8.0)
                optimizer.step()
                steps += 1
        after = self._sample_loss(local, idx)
        improvement = before - after
        delta = state_delta(local.state_dict(), self.model.state_dict())

        if method == "SCAFFOLD" and steps > 0:
            old = [x.clone() for x in self.scaffold_clients[cid]]
            with torch.no_grad():
                for layer in range(len(delta)):
                    self.scaffold_clients[cid][layer] = (
                        self.scaffold_clients[cid][layer]
                        - self.scaffold_server[layer]
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
        if method == "FedLAW":
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
        return metrics

    def run(self, verbose: bool = True) -> Dict[str, object]:
        cfg = self.cfg
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
            selected = self.rng.choice(all_clients, size=cfg.clients_per_round, replace=False)
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
                out = self.perg.aggregate(deltas, base_weights, imp_tensor, round_idx)
                aggregated = out.delta
                client_weights = out.weights.mean(dim=1)
                perg_diag = {
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
                    for candidate in out.expert_deltas:
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
                aggregated, client_weights = self.baseline.aggregate(
                    deltas,
                    base_weights,
                    imp_tensor,
                    step_tensor,
                    round_idx,
                    global_flat,
                    losses if cfg.method == "FedLAW" else None,
                )
            add_delta_(self.model, aggregated, cfg.server_lr)
            if self.perg is not None:
                # Strict temporal holdout: all current-round scoring and the
                # global update precede training on current masked targets.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                predictor_train_started = time.perf_counter()
                perg_diag["meta_loss"] = self.perg.update_predictor(
                    deltas, base_weights, imp_tensor, round_idx
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
            round_bytes = cfg.clients_per_round * num_parameters * 4
            extra_scalar_bytes = (4 * cfg.clients_per_round
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
                    **perg_diag,
                }
            )
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
        self._save(result)
        return result

    def _save(self, result: Dict[str, object]) -> None:
        root = Path(self.cfg.results_dir)
        models = root / "models"
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
        json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        torch.save(
            {"model_state": self.model.state_dict(), "config": self.cfg.to_dict()},
            models / f"{stem}.pth",
        )
        if self.perg is not None:
            torch.save(
                {"operator_state": self.perg.operator.state_dict(), "config": self.cfg.to_dict()},
                models / f"{stem}__operator.pth",
            )
