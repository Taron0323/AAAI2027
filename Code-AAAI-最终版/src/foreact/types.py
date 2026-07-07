"""Shared dataclasses for ForeAct phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Action:
    """A complete environment-changing action."""

    text: str
    tool: str
    args: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Environment feedback after an action."""

    text: str
    state: Mapping[str, str] = field(default_factory=dict)
    branch_id: Optional[str] = None


@dataclass
class Trajectory:
    """One ReAct-style trajectory with outcome labels."""

    task_id: str
    goal: str
    actions: List[Action]
    observations: List[Observation]
    success: bool
    dead_end: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def prefix(self, action_index: int) -> "TrajectoryPrefix":
        if action_index < 0 or action_index > len(self.actions):
            raise ValueError(f"invalid prefix action_index={action_index}")
        return TrajectoryPrefix(
            task_id=self.task_id,
            goal=self.goal,
            actions=self.actions[:action_index],
            observations=self.observations[:action_index],
            next_action_index=action_index,
            success=self.success,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class TrajectoryPrefix:
    """A decision prefix h_t before selecting the next action."""

    task_id: str
    goal: str
    actions: Sequence[Action]
    observations: Sequence[Observation]
    next_action_index: int
    success: bool
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftTarget:
    """Empirical future-sketch distribution for one prefix and depth."""

    depth: int
    distribution: Mapping[str, float]
    entropy: float
    branch_weight: float


@dataclass
class AlignedExample:
    """Training row consumed by LAP/PCR."""

    task_id: str
    prefix_index: int
    goal: str
    current_action: Action
    current_sketch: str
    future_targets: List[SoftTarget]
    success: bool
    hidden_features: List[float]
    next_hidden_features: Optional[List[float]] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def action_history_text(goal: str, actions: Sequence[Action], observations: Sequence[Observation]) -> str:
    """Render a ReAct-style history without adding special model tokens."""

    lines = [f"Goal: {goal}"]
    for idx, action in enumerate(actions):
        lines.append(f"Action {idx + 1}: {action.text}")
        if idx < len(observations):
            lines.append(f"Observation {idx + 1}: {observations[idx].text}")
    return "\n".join(lines)
