"""ReAct trajectory parsing and action-boundary location."""

from __future__ import annotations

import re
from typing import List, Tuple

from foreact.types import Action, Observation, Trajectory

ACTION_RE = re.compile(r"^\s*(?:Action|Tool)\s*\d*\s*:\s*(?P<text>.+)$", re.IGNORECASE)
ACTION_START_RE = re.compile(r"^\s*(?:Action|Tool)\s*\d*\s*:", re.IGNORECASE)
OBS_RE = re.compile(r"^\s*Observation\s*\d*\s*:\s*(?P<text>.+)$", re.IGNORECASE)


def infer_tool(action_text: str) -> Tuple[str, dict]:
    parts = action_text.strip().split()
    tool = parts[0] if parts else "unknown"
    args = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            args[key] = value.strip(",")
    return tool, args


def parse_react_text(task_id: str, goal: str, text: str, success: bool = False) -> Trajectory:
    actions: List[Action] = []
    observations: List[Observation] = []
    for line in text.splitlines():
        action_match = ACTION_RE.match(line)
        if action_match:
            action_text = action_match.group("text").strip()
            tool, args = infer_tool(action_text)
            actions.append(Action(text=action_text, tool=tool, args=args))
            continue
        obs_match = OBS_RE.match(line)
        if obs_match:
            observations.append(Observation(text=obs_match.group("text").strip()))
    return Trajectory(
        task_id=task_id,
        goal=goal,
        actions=actions,
        observations=observations,
        success=success,
        dead_end=any("dead_end" in obs.text for obs in observations),
    )


def locate_action_start_offsets(text: str) -> List[int]:
    """Return character offsets for ReAct action starts.

    Token-level offset conversion is tokenizer-specific; this stable character
    list is the adapter boundary used by HF tokenizers in full training.
    """

    offsets: List[int] = []
    running = 0
    for line in text.splitlines(keepends=True):
        if ACTION_START_RE.match(line):
            offsets.append(running + line.lower().find(":") + 1)
        running += len(line)
    return offsets
