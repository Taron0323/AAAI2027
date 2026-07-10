"""Mechanism diagnostics for ForeAct smoke and full experiments.

The full paper's Effective Lookahead Depth (ELD) is an intervention probe over
backbone representations. The functions here implement the same artifact
contract for smoke runs: a depth-indexed future-information curve, a machine
computed ELD crossing, and PlanDepth boundary diagnostics. Smoke curves are
proxies and must not be reported as real paper results.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np

from foreact.training.toy_trainer import ToyForeActModel
from foreact.types import AlignedExample


def smoke_eld_curve(model: ToyForeActModel, examples: Sequence[AlignedExample]) -> Dict[int, float]:
    """Approximate ELD artifact shape from forecast confidence.

    This is not a causal intervention probe. It validates the artifact contract
    that the real representation intervention analysis will fill.
    """

    if not examples:
        return {}
    curve: Dict[int, float] = {}
    for depth in range(1, model.config.horizon + 1):
        confidence = []
        for ex in examples:
            probs = model.forecast_probs(ex.hidden_features)
            confidence.append(float(np.max(probs[depth - 1])))
        curve[depth] = max(0.0, float(np.mean(confidence)) - (1.0 / model.config.num_sketches))
    return curve


def forecast_entropy_by_depth(model: ToyForeActModel, examples: Sequence[AlignedExample]) -> list[float]:
    if not examples:
        return [0.0 for _ in range(model.config.horizon)]
    values = []
    for depth in range(model.config.horizon):
        entropies = []
        for ex in examples:
            probs = model.forecast_probs(ex.hidden_features)[depth]
            entropies.append(float(-np.sum(probs * np.log(probs + 1e-9))))
        values.append(float(np.mean(entropies)))
    return values


def effective_lookahead_depth(
    signal_by_depth: Mapping[int | str, float],
    *,
    absolute_threshold: float = 0.05,
    relative_threshold: float = 0.25,
) -> int:
    """Return the deepest depth with recoverable future signal.

    In full experiments `signal_by_depth` should be the quality drop caused by
    removing a future-action probe subspace. In smoke tests it is a confidence
    proxy with the same shape. A depth counts as "covered" when its signal is at
    least the larger of an absolute floor and a fraction of the maximum signal.
    """

    normalized = {int(depth): float(value) for depth, value in signal_by_depth.items()}
    if not normalized:
        return 0
    max_signal = max(normalized.values())
    if max_signal <= 0:
        return 0
    threshold = max(absolute_threshold, relative_threshold * max_signal)
    covered = [depth for depth, value in normalized.items() if value >= threshold]
    return max(covered) if covered else 0


def eld_curve_rows(
    method: str,
    signal_by_depth: Mapping[int | str, float],
    *,
    figure_role: str,
    absolute_threshold: float = 0.05,
    relative_threshold: float = 0.25,
    note: str = "smoke proxy; replace with intervention-probe measurements for paper figures",
) -> list[dict]:
    """Rows for Fig.2/Fig.5 ELD artifacts with shared axis metadata."""

    normalized = {int(depth): float(value) for depth, value in signal_by_depth.items()}
    if not normalized:
        return []
    depths = sorted(normalized)
    max_signal = max(normalized.values())
    threshold = max(absolute_threshold, relative_threshold * max_signal)
    eld = effective_lookahead_depth(
        normalized,
        absolute_threshold=absolute_threshold,
        relative_threshold=relative_threshold,
    )
    return [
        {
            "figure_role": figure_role,
            "method": method,
            "depth": depth,
            "future_signal": normalized[depth],
            "threshold": threshold,
            "above_threshold": normalized[depth] >= threshold,
            "effective_lookahead_depth": eld,
            "axis_depth_min": min(depths),
            "axis_depth_max": max(depths),
            "axis_signal_min": 0.0,
            "axis_signal_max": max_signal,
            "note": note,
        }
        for depth in depths
    ]


def success_collapse_depth(sr_by_depth: Mapping[int | str, float], *, threshold: float = 0.5) -> int:
    """Smallest depth where SR(d) drops below threshold, or the deepest depth."""

    normalized = {int(depth): float(value) for depth, value in sr_by_depth.items()}
    if not normalized:
        return 0
    for depth, value in sorted(normalized.items()):
        if value < threshold:
            return depth
    return max(normalized)


def plandepth_boundary_rows(
    metrics_by_method: Mapping[str, Mapping[str, object]],
    eld_by_method: Mapping[str, int],
    *,
    trained_horizon: int,
) -> list[dict]:
    """Build H5 boundary rows: gain vs depth and ELD crossing."""

    rows: list[dict] = []
    for method, metrics in metrics_by_method.items():
        sr_by_depth = metrics.get("sr_by_depth", {})
        if not isinstance(sr_by_depth, Mapping):
            continue
        eld = int(eld_by_method.get(method, 0))
        collapse = success_collapse_depth(sr_by_depth)
        for depth_raw, success_rate in sorted(sr_by_depth.items(), key=lambda item: int(item[0])):
            depth = int(depth_raw)
            rows.append(
                {
                    "method": method,
                    "depth": depth,
                    "success_rate": float(success_rate),
                    "effective_lookahead_depth": eld,
                    "success_collapse_depth": collapse,
                    "within_eld": depth <= eld,
                    "beyond_trained_horizon": depth > trained_horizon,
                    "note": "PlanDepth smoke boundary row; not a paper result",
                }
            )
    return rows


def granularity_diagnostic_rows() -> list[dict]:
    """Expected A-prime granularity spectrum rows for job/reporting contracts."""

    return [
        {
            "ablation": "A_prime",
            "zeta_mode": "token",
            "expected_shape": "low",
            "reason": "surface token prediction is not aligned with decision feasibility",
        },
        {
            "ablation": "A_prime",
            "zeta_mode": "fsp_summary",
            "expected_shape": "mid",
            "reason": "future summary loses per-depth temporal structure",
        },
        {
            "ablation": "A_prime",
            "zeta_mode": "type",
            "expected_shape": "mid",
            "reason": "operation type is often too coarse",
        },
        {
            "ablation": "A_prime",
            "zeta_mode": "type_arg",
            "expected_shape": "high",
            "reason": "main semantic sketch: operation type x primary argument slot",
        },
        {
            "ablation": "A_prime",
            "zeta_mode": "vq_mock",
            "expected_shape": "high_or_mid",
            "reason": "manual-schema-free VQ-style smoke interface",
        },
    ]
