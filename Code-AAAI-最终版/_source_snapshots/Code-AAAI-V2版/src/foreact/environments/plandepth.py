"""PlanDepth: controlled long-horizon tasks for ForeAct smoke and analysis."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from foreact.types import Action, Observation, Trajectory


@dataclass(frozen=True)
class PlanDepthTask:
    task_id: str
    depth: int
    stochastic: bool
    delayed_deadend: bool
    goal: str
    path: Sequence[str]
    trap_step: int


class PlanDepthEnv:
    """A tiny graph-navigation and resource-scheduling environment.

    Each task has a required chain of semantic actions. A premature commit action
    creates a delayed dead end, mirroring irreversible tool calls in the paper.
    """

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)

    def make_tasks(
        self,
        num_tasks: int,
        depths: Sequence[int],
        stochastic: bool = True,
        delayed_deadend: bool = True,
    ) -> List[PlanDepthTask]:
        tasks: List[PlanDepthTask] = []
        for index in range(num_tasks):
            depth = depths[index % len(depths)]
            path = [f"node_{index}_{step}" for step in range(depth)]
            trap_step = max(1, depth // 2)
            tasks.append(
                PlanDepthTask(
                    task_id=f"pd-{index:04d}",
                    depth=depth,
                    stochastic=stochastic,
                    delayed_deadend=delayed_deadend,
                    goal=f"Reach node_{index}_{depth - 1} without triggering the delayed trap.",
                    path=path,
                    trap_step=trap_step,
                )
            )
        return tasks

    def expert_actions(self, task: PlanDepthTask) -> List[Action]:
        actions: List[Action] = []
        for step, node in enumerate(task.path):
            tool = "inspect" if step == 0 else "move"
            if step == task.trap_step:
                tool = "verify"
            if step == len(task.path) - 1:
                tool = "commit"
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node}))
        return actions

    def trap_action(self, task: PlanDepthTask) -> Action:
        target = task.path[min(task.trap_step, len(task.path) - 1)]
        return Action(text=f"commit target={target}", tool="commit", args={"target": target, "trap": "true"})

    def rollout_expert(self, task: PlanDepthTask, max_steps: int | None = None) -> Trajectory:
        actions = self.expert_actions(task)
        if max_steps is not None:
            actions = actions[:max_steps]
        observations: List[Observation] = []
        dead_end = False
        for step, action in enumerate(actions):
            branch = "nominal"
            if task.stochastic and self.random.random() < 0.2 and action.tool != "commit":
                branch = "noisy_hint"
            if action.args.get("trap") == "true":
                dead_end = True
            observations.append(
                Observation(
                    text=f"{branch}: completed {action.tool} at step {step + 1}",
                    state={"step": str(step + 1), "depth": str(task.depth)},
                    branch_id=branch,
                )
            )
        success = (len(actions) >= task.depth) and not dead_end
        return Trajectory(
            task_id=task.task_id,
            goal=task.goal,
            actions=actions,
            observations=observations,
            success=success,
            dead_end=dead_end,
            metadata={"depth": str(task.depth), "benchmark": "plandepth"},
        )

    def rollout_myopic(self, task: PlanDepthTask, max_steps: int | None = None) -> Trajectory:
        expert = self.expert_actions(task)
        actions: List[Action] = []
        for step, action in enumerate(expert):
            if task.delayed_deadend and step == task.trap_step:
                actions.append(self.trap_action(task))
                break
            actions.append(action)
        if max_steps is not None:
            actions = actions[:max_steps]
        observations = [
            Observation(
                text=f"completed {action.tool}; {'dead_end' if action.args.get('trap') else 'ok'}",
                state={"step": str(i + 1), "depth": str(task.depth)},
                branch_id="trap" if action.args.get("trap") else "nominal",
            )
            for i, action in enumerate(actions)
        ]
        dead_end = any(action.args.get("trap") == "true" for action in actions)
        return Trajectory(
            task_id=task.task_id,
            goal=task.goal,
            actions=actions,
            observations=observations,
            success=False,
            dead_end=dead_end,
            metadata={"depth": str(task.depth), "benchmark": "plandepth", "policy": "myopic"},
        )

    def teacher_continuations(
        self,
        task: PlanDepthTask,
        prefix_len: int,
        k: int,
        horizon: int,
    ) -> List[List[Action]]:
        expert = self.expert_actions(task)
        continuations: List[List[Action]] = []
        for sample_id in range(k):
            future = list(expert[prefix_len : prefix_len + horizon])
            if task.stochastic and future and self.random.random() < 0.25:
                replace_index = min(len(future) - 1, self.random.randrange(len(future)))
                target = future[replace_index].args.get("target", "unknown")
                future[replace_index] = Action(
                    text=f"query_constraint target={target}",
                    tool="query_constraint",
                    args={"target": target, "sample": str(sample_id)},
                )
            continuations.append(future)
        return continuations


def trajectories_to_rows(trajectories: Iterable[Trajectory]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for trajectory in trajectories:
        rows.append(
            {
                "task_id": trajectory.task_id,
                "goal": trajectory.goal,
                "success": trajectory.success,
                "dead_end": trajectory.dead_end,
                "metadata": trajectory.metadata,
                "actions": [
                    {"text": action.text, "tool": action.tool, "args": dict(action.args)}
                    for action in trajectory.actions
                ],
                "observations": [
                    {"text": obs.text, "state": dict(obs.state), "branch_id": obs.branch_id}
                    for obs in trajectory.observations
                ],
            }
        )
    return rows

