"""Mechanism diagnostics for ForeAct smoke and full experiments."""

from __future__ import annotations

from typing import Dict, Sequence

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

