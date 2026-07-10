"""K-rollout future-sketch soft target construction."""

from __future__ import annotations

import math
import random
import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from foreact.types import Action, AlignedExample, Observation, SoftTarget, Trajectory, action_history_text
from foreact.zeta import SketchMapper


def normalized_entropy(distribution: Dict[str, float]) -> float:
    if not distribution:
        return 0.0
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in distribution.values())
    if len(distribution) <= 1:
        return 0.0
    return entropy / math.log(len(distribution))


def aggregate_future_targets(
    continuations: Sequence[Sequence[Action]],
    mapper: SketchMapper,
    horizon: int,
) -> List[SoftTarget]:
    targets: List[SoftTarget] = []
    for depth in range(1, horizon + 1):
        sketches: List[str] = []
        for continuation in continuations:
            if len(continuation) >= depth:
                sketches.append(mapper.encode(continuation[depth - 1]))
        counter = Counter(sketches)
        total = sum(counter.values())
        distribution = {sketch: count / total for sketch, count in sorted(counter.items())} if total else {}
        entropy = normalized_entropy(distribution)
        targets.append(
            SoftTarget(
                depth=depth,
                distribution=distribution,
                entropy=entropy,
                branch_weight=(1.0 - entropy) if distribution else 0.0,
            )
        )
    return targets


def deterministic_features(text: str, dim: int) -> List[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def build_aligned_examples(
    trajectories: Iterable[Trajectory],
    mapper: SketchMapper,
    continuation_lookup: Dict[tuple, Sequence[Sequence[Action]]],
    horizon: int,
    hidden_dim: int,
) -> List[AlignedExample]:
    examples: List[AlignedExample] = []
    for trajectory in trajectories:
        for idx, action in enumerate(trajectory.actions):
            continuations = continuation_lookup.get((trajectory.task_id, idx), [])
            if not continuations:
                continue
            history_text = trajectory.goal + "\n" + "\n".join(a.text for a in trajectory.actions[:idx])
            next_history = history_text + "\n" + action.text
            prefix_text = _decision_text(
                trajectory.goal,
                trajectory.actions[:idx],
                trajectory.observations[:idx],
                idx + 1,
                action.text,
            )
            next_prefix_text = _decision_text(
                trajectory.goal,
                trajectory.actions[: idx + 1],
                trajectory.observations[: idx + 1],
                idx + 2,
            )
            examples.append(
                AlignedExample(
                    task_id=trajectory.task_id,
                    prefix_index=idx,
                    goal=trajectory.goal,
                    current_action=action,
                    current_sketch=mapper.encode(action),
                    future_targets=aggregate_future_targets(continuations, mapper, horizon),
                    success=trajectory.success,
                    hidden_features=deterministic_features(history_text, hidden_dim),
                    next_hidden_features=deterministic_features(next_history, hidden_dim),
                    prefix_text=prefix_text,
                    next_prefix_text=next_prefix_text,
                    metadata=dict(trajectory.metadata),
                )
            )
    return examples


def examples_to_rows(examples: Iterable[AlignedExample]) -> List[dict]:
    rows: List[dict] = []
    for ex in examples:
        rows.append(
            {
                "task_id": ex.task_id,
                "prefix_index": ex.prefix_index,
                "goal": ex.goal,
                "current_action": {
                    "text": ex.current_action.text,
                    "tool": ex.current_action.tool,
                    "args": dict(ex.current_action.args),
                },
                "current_sketch": ex.current_sketch,
                "future_targets": [
                    {
                        "depth": target.depth,
                        "distribution": dict(target.distribution),
                        "entropy": target.entropy,
                        "branch_weight": target.branch_weight,
                    }
                    for target in ex.future_targets
                ],
                "success": ex.success,
                "hidden_features": ex.hidden_features,
                "next_hidden_features": ex.next_hidden_features,
                "prefix_text": ex.prefix_text,
                "next_prefix_text": ex.next_prefix_text,
                "metadata": ex.metadata,
            }
        )
    return rows


def _decision_text(
    goal: str,
    actions: Sequence[Action],
    observations: Sequence[Observation],
    action_number: int,
    action_text: str = "",
) -> str:
    prefix = action_history_text(goal, actions, observations)
    suffix = f"Action {action_number}: {action_text}".rstrip()
    return f"{prefix}\n{suffix}" if prefix else suffix
