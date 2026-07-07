"""Lightweight learned VQ-style action sketch codebooks."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

import numpy as np

from foreact.types import Action


def build_vq_schema(actions: Iterable[Action], codebook_size: int = 512, dim: int = 64, iterations: int = 20) -> Dict[str, object]:
    action_list = list(actions)
    if not action_list:
        return {"mode": "vq", "size": 0, "sketches": [], "counts": {}, "centroids": []}
    vectors = np.stack([_text_vector(action.text, dim) for action in action_list])
    k = min(codebook_size, len(action_list))
    centroids = vectors[np.linspace(0, len(vectors) - 1, k, dtype=int)].copy()
    for _ in range(iterations):
        distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        labels = distances.argmin(axis=1)
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centroids[idx] = vectors[mask].mean(axis=0)
    sketches = [f"vq::{idx}" for idx in range(k)]
    counts = {sketch: 0 for sketch in sketches}
    distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
    labels = distances.argmin(axis=1)
    for label in labels:
        counts[f"vq::{int(label)}"] += 1
    return {
        "mode": "vq",
        "size": len(sketches),
        "sketches": sketches,
        "counts": counts,
        "centroids": centroids.round(6).tolist(),
        "embedding_dim": dim,
    }


def encode_vq_action(action: Action, schema: Dict[str, object]) -> str:
    centroids = np.array(schema.get("centroids", []), dtype=np.float32)
    if centroids.size == 0:
        return "<unk>"
    vector = _text_vector(action.text, int(schema.get("embedding_dim", centroids.shape[-1])))
    distances = ((centroids - vector[None, :]) ** 2).sum(axis=-1)
    return f"vq::{int(distances.argmin())}"


def _text_vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: List[float] = []
    counter = 0
    while len(values) < dim:
        block = hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
        values.extend((byte / 127.5) - 1.0 for byte in block)
        counter += 1
    vec = np.array(values[:dim], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
