"""LAP and PCR loss functions in numpy."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

EPS = 1e-9


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def ce_soft(probs: np.ndarray, target: Mapping[int, float]) -> float:
    return -sum(weight * math.log(float(probs[index]) + EPS) for index, weight in target.items())


def forward_kl_stopgrad(target_probs: np.ndarray, pred_probs: np.ndarray) -> float:
    """KL(sg[target] || pred), the direction required by ForeAct PCR."""

    return float(np.sum(target_probs * (np.log(target_probs + EPS) - np.log(pred_probs + EPS))))


def branch_weight_from_entropy(entropy: float) -> float:
    return max(0.0, min(1.0, 1.0 - entropy))
