"""Task, method, and system metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence

from foreact.environments.plandepth import PlanDepthTask
from foreact.inference.policies import PolicyTrace
from foreact.types import Action


def evaluate_plandepth(tasks: Sequence[PlanDepthTask], traces: Sequence[PolicyTrace]) -> Dict[str, object]:
    task_by_id = {task.task_id: task for task in tasks}
    successes: List[bool] = []
    dead_ends: List[bool] = []
    churn_rates: List[float] = []
    by_depth = defaultdict(lambda: {"success": 0, "total": 0})
    token_count = 0
    extra_forward = 0
    for trace in traces:
        task = task_by_id[trace.task_id]
        success = _success(task, trace.actions)
        dead_end = _dead_end(task, trace.actions)
        successes.append(success)
        dead_ends.append(dead_end)
        churn_rates.append(behavioral_churn_rate(trace.actions))
        by_depth[task.depth]["success"] += int(success)
        by_depth[task.depth]["total"] += 1
        token_count += trace.token_count
        extra_forward += trace.extra_forward_count
    return {
        "success_rate": _mean(successes),
        "dead_end_rate": _mean(dead_ends),
        "behavioral_churn_rate": _mean(churn_rates),
        "tokens_per_task": token_count / max(1, len(traces)),
        "extra_forward_per_task": extra_forward / max(1, len(traces)),
        "sr_by_depth": {
            str(depth): values["success"] / max(1, values["total"]) for depth, values in sorted(by_depth.items())
        },
    }


def behavioral_churn_rate(actions: Sequence[Action]) -> float:
    if not actions:
        return 0.0
    repeats = 0
    seen = set()
    reversals = 0
    for action in actions:
        sketch = (action.tool, action.args.get("target", ""))
        if sketch in seen:
            repeats += 1
        if action.tool in {"undo", "restore", "rollback"}:
            reversals += 1
        seen.add(sketch)
    return (repeats + reversals) / len(actions)


def _success(task: PlanDepthTask, actions: Sequence[Action]) -> bool:
    if _dead_end(task, actions):
        return False
    if len(actions) < task.depth:
        return False
    return actions[-1].tool == "commit" and actions[-1].args.get("target") == task.path[-1]


def _dead_end(task: PlanDepthTask, actions: Sequence[Action]) -> bool:
    for idx, action in enumerate(actions):
        if action.args.get("trap") == "true":
            return True
        if action.tool == "commit" and idx < task.depth - 1:
            return True
    return False


def _mean(values: Sequence[float | bool]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))

