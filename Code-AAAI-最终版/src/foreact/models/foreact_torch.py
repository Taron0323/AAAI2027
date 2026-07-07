"""Optional PyTorch modules for full LAP/PCR training."""

from __future__ import annotations

from dataclasses import dataclass


class TorchUnavailable(RuntimeError):
    pass


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise TorchUnavailable("Install Code-AAAI with the [torch] extra to use full training modules.") from exc
    return torch, nn


def build_forecast_heads(hidden_dim: int, sketch_size: int, horizon: int, width: int = 1024):
    """Build H independent two-layer MLP forecast heads."""

    _torch, nn = require_torch()
    return nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, width), nn.GELU(), nn.Linear(width, sketch_size)) for _ in range(horizon)]
    )


@dataclass(frozen=True)
class ForeActTorchConfig:
    hidden_dim: int
    sketch_size: int
    horizon: int = 8
    width: int = 1024
    lambda_future: float = 0.3
    mu_consistency: float = 0.1
    eta_success: float = 0.05


def build_foreact_heads(config: ForeActTorchConfig):
    """Build forecast and success heads for a backbone hidden state."""

    _torch, nn = require_torch()
    return nn.ModuleDict(
        {
            "forecast": build_forecast_heads(
                config.hidden_dim,
                config.sketch_size,
                config.horizon,
                width=config.width,
            ),
            "success": nn.Sequential(
                nn.Linear(config.hidden_dim, config.width),
                nn.GELU(),
                nn.Linear(config.width, 1),
            ),
        }
    )


def active_horizon(step: int, total_steps: int, horizon: int) -> int:
    if step < total_steps / 3:
        return min(2, horizon)
    if step < 2 * total_steps / 3:
        return min(4, horizon)
    return horizon


def forecast_logits(heads, hidden_states):
    """Return tensor shaped [batch, horizon, sketch_size]."""

    torch, _nn = require_torch()
    logits = [head(hidden_states) for head in heads["forecast"]]
    return torch.stack(logits, dim=1)


def soft_target_ce(logits, targets, weights=None):
    """Cross entropy against soft sketch targets.

    Args:
        logits: [batch, active_horizon, sketch_size]
        targets: same shape, each row a probability distribution.
        weights: optional [batch, active_horizon] branch-aware weights.
    """

    torch, _nn = require_torch()
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1)
    if weights is not None:
        loss = loss * weights
        return loss.sum() / weights.sum().clamp_min(1.0)
    return loss.mean()


def pcr_forward_kl(current_logits, next_logits, branch_weights=None):
    """PCR loss KL(sg[q_{t+1}^{h-1}] || q_t^h).

    The direction is intentionally fixed to the Proposition 1 direction in the
    paper plan. `next_logits` is detached before forming the target.
    """

    torch, _nn = require_torch()
    if current_logits.shape[1] < 2:
        return current_logits.sum() * 0.0
    pred_log_probs = torch.log_softmax(current_logits[:, 1:, :], dim=-1)
    target_probs = torch.softmax(next_logits[:, :-1, :].detach(), dim=-1)
    target_log_probs = torch.log(target_probs.clamp_min(1e-9))
    kl = (target_probs * (target_log_probs - pred_log_probs)).sum(dim=-1)
    if branch_weights is not None:
        weights = branch_weights[:, 1:]
        kl = kl * weights
        return kl.sum() / weights.sum().clamp_min(1.0)
    return kl.mean()


def success_bce(heads, hidden_states, labels):
    torch, _nn = require_torch()
    logits = heads["success"](hidden_states).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())


def foreact_auxiliary_loss(
    heads,
    hidden_states,
    next_hidden_states,
    soft_targets,
    branch_weights,
    success_labels,
    config: ForeActTorchConfig,
    step: int,
    total_steps: int,
):
    """Compute LAP + PCR + success-head loss for full-scale trainers.

    The trainer should add this auxiliary value to the backbone NTP/SFT loss.
    """

    ah = active_horizon(step, total_steps, config.horizon)
    logits = forecast_logits(heads, hidden_states)[:, :ah, :]
    next_logits = forecast_logits(heads, next_hidden_states)[:, :ah, :]
    targets = soft_targets[:, :ah, :]
    weights = branch_weights[:, :ah] if branch_weights is not None else None
    future = soft_target_ce(logits, targets, weights)
    consistency = pcr_forward_kl(logits, next_logits, weights)
    succ = success_bce(heads, hidden_states, success_labels)
    total = config.lambda_future * future + config.mu_consistency * consistency + config.eta_success * succ
    return {
        "loss": total,
        "future_loss": future.detach(),
        "consistency_loss": consistency.detach(),
        "success_loss": succ.detach(),
        "active_horizon": ah,
    }
