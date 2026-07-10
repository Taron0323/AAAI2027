"""Small numpy trainer for smoke-validating LAP/PCR plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from foreact.training.losses import ce_soft, forward_kl_stopgrad, softmax
from foreact.types import AlignedExample


@dataclass
class ToyForeActConfig:
    hidden_dim: int
    num_sketches: int
    horizon: int
    lambda_future: float = 0.3
    mu_consistency: float = 0.1
    eta_success: float = 0.05
    learning_rate: float = 0.05
    steps: int = 8
    seed: int = 0


class ToyForeActModel:
    """Forecast and success heads over deterministic smoke features."""

    def __init__(self, config: ToyForeActConfig) -> None:
        rng = np.random.default_rng(config.seed)
        self.config = config
        self.forecast = rng.normal(0, 0.05, size=(config.horizon, config.hidden_dim, config.num_sketches))
        self.success = rng.normal(0, 0.05, size=(config.hidden_dim,))

    def forecast_probs(self, features: Sequence[float]) -> np.ndarray:
        hidden = np.asarray(features, dtype=float)
        logits = np.einsum("d,hds->hs", hidden, self.forecast)
        return softmax(logits)

    def success_prob(self, features: Sequence[float]) -> float:
        hidden = np.asarray(features, dtype=float)
        return float(1.0 / (1.0 + np.exp(-np.dot(hidden, self.success))))


def target_index_distribution(distribution: Dict[str, float], sketch_to_idx: Dict[str, int]) -> Dict[int, float]:
    return {sketch_to_idx[sketch]: prob for sketch, prob in distribution.items() if sketch in sketch_to_idx}


def train_toy_foreact(
    examples: Sequence[AlignedExample],
    sketches: Sequence[str],
    config: ToyForeActConfig,
) -> tuple[ToyForeActModel, List[dict]]:
    model = ToyForeActModel(config)
    sketch_to_idx = {sketch: idx for idx, sketch in enumerate(sketches)}
    metrics: List[dict] = []

    for step in range(config.steps):
        total_future = 0.0
        total_cons = 0.0
        total_success = 0.0
        active_horizon = _active_horizon(step, config.steps, config.horizon)
        for ex in examples:
            probs = model.forecast_probs(ex.hidden_features)
            hidden = np.asarray(ex.hidden_features, dtype=float)
            for h_idx, target in enumerate(ex.future_targets[:active_horizon]):
                target_dist = target_index_distribution(dict(target.distribution), sketch_to_idx)
                if not target_dist:
                    continue
                total_future += target.branch_weight * ce_soft(probs[h_idx], target_dist)
                grad_logits = probs[h_idx].copy()
                for idx, weight in target_dist.items():
                    grad_logits[idx] -= weight
                grad_logits *= target.branch_weight * config.lambda_future
                model.forecast[h_idx] -= config.learning_rate * np.outer(hidden, grad_logits)

            if ex.next_hidden_features is not None and active_horizon >= 2:
                next_probs = model.forecast_probs(ex.next_hidden_features)
                for h_idx in range(1, active_horizon):
                    weight = ex.future_targets[h_idx].branch_weight if h_idx < len(ex.future_targets) else 1.0
                    total_cons += weight * forward_kl_stopgrad(next_probs[h_idx - 1], probs[h_idx])
                    grad_logits = (probs[h_idx] - next_probs[h_idx - 1]) * weight * config.mu_consistency
                    model.forecast[h_idx] -= config.learning_rate * np.outer(hidden, grad_logits)

            succ = model.success_prob(ex.hidden_features)
            label = 1.0 if ex.success else 0.0
            total_success += -(label * np.log(succ + 1e-9) + (1 - label) * np.log(1 - succ + 1e-9))
            model.success -= config.learning_rate * config.eta_success * (succ - label) * hidden

        denom = max(1, len(examples))
        metrics.append(
            {
                "step": step,
                "active_horizon": active_horizon,
                "future_loss": total_future / denom,
                "consistency_loss": total_cons / denom,
                "success_loss": total_success / denom,
                "mean_entropy": float(_mean_entropy(model, examples)),
            }
        )
    return model, metrics


def _active_horizon(step: int, total_steps: int, horizon: int) -> int:
    if step < total_steps / 3:
        return min(2, horizon)
    if step < 2 * total_steps / 3:
        return min(4, horizon)
    return horizon


def _mean_entropy(model: ToyForeActModel, examples: Sequence[AlignedExample]) -> float:
    if not examples:
        return 0.0
    entropies = []
    for ex in examples:
        probs = model.forecast_probs(ex.hidden_features)
        entropies.append(float(-np.sum(probs * np.log(probs + 1e-9)) / probs.shape[0]))
    return float(np.mean(entropies))

