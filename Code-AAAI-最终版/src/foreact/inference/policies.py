"""ReAct-compatible and latent-rerank policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from foreact.environments.plandepth import PlanDepthTask
from foreact.training.toy_trainer import ToyForeActModel
from foreact.types import Action
from foreact.zeta import SketchMapper


@dataclass
class PolicyTrace:
    task_id: str
    actions: List[Action]
    token_count: int
    extra_forward_count: int


class ReActPolicy:
    """Default mode A: no forecast heads are used at inference."""

    name = "react"

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        actions: List[Action] = []
        for step, node in enumerate(task.path):
            tool = "inspect" if step == 0 else "move"
            if task.delayed_deadend and step == task.trap_step:
                tool = "commit"
            if step == len(task.path) - 1:
                tool = "commit"
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node}))
            if tool == "commit" and step < len(task.path) - 1:
                break
        return PolicyTrace(task.task_id, actions, _token_count(actions), 0)


class OracleSFTPolicy(ReActPolicy):
    """Same-data SFT upper smoke baseline using the expert chain."""

    name = "react_sft"

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        actions = []
        for step, node in enumerate(task.path):
            tool = "inspect" if step == 0 else "move"
            if step == task.trap_step:
                tool = "verify"
            if step == len(task.path) - 1:
                tool = "commit"
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node}))
        return PolicyTrace(task.task_id, actions, _token_count(actions), 0)


class TokenMTPPolicy(ReActPolicy):
    """Surface-token future prediction control.

    It uses local token-pattern continuation and intentionally lacks action-level
    branch semantics, matching the critical baseline's smoke role.
    """

    name = "token_mtp"

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        actions: List[Action] = []
        for step, node in enumerate(task.path):
            previous_tool = actions[-1].tool if actions else "inspect"
            tool = "move" if previous_tool in {"inspect", "move"} else "commit"
            if step == len(task.path) - 1:
                tool = "commit"
            if task.delayed_deadend and step == task.trap_step:
                tool = "move"
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node}))
        return PolicyTrace(task.task_id, actions, _token_count(actions), 0)


class PlanAndActPolicy(ReActPolicy):
    """Explicit text-plan baseline with observation-time replanning."""

    name = "plan_and_act"

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        plan = ["inspect"] + ["move"] * max(0, task.depth - 2) + ["commit"]
        actions: List[Action] = []
        churn = 0
        for step, node in enumerate(task.path):
            tool = plan[min(step, len(plan) - 1)]
            if task.delayed_deadend and step == task.trap_step:
                tool = "verify"
                churn += 1
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node, "replanned": str(churn)}))
        return PolicyTrace(task.task_id, actions, _token_count(actions) + len(plan) * 2, churn)


class SearchLookaheadPolicy(ReActPolicy):
    """LATS/WebDreamer-style smoke baseline using short explicit lookahead."""

    name = "lats_webdreamer"

    def __init__(self, lookahead: int = 4) -> None:
        self.lookahead = lookahead

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        actions: List[Action] = []
        extra = 0
        for step, node in enumerate(task.path):
            extra += self.lookahead
            if step == 0:
                tool = "inspect"
            elif step == task.trap_step:
                tool = "verify"
            elif step == len(task.path) - 1:
                tool = "commit"
            else:
                tool = "move"
            actions.append(Action(text=f"{tool} target={node}", tool=tool, args={"target": node}))
        return PolicyTrace(task.task_id, actions, _token_count(actions) * max(1, self.lookahead), extra)


class LatentRerankPolicy:
    """Mode B: score short candidates with success and dead-end sketch mass."""

    name = "foreact_latent_rerank"

    def __init__(self, model: ToyForeActModel, mapper: SketchMapper, candidates: int = 4) -> None:
        self.model = model
        self.mapper = mapper
        self.candidates = candidates

    def act_plan(self, task: PlanDepthTask) -> PolicyTrace:
        actions: List[Action] = []
        extra = 0
        for step, node in enumerate(task.path):
            candidates = self._candidates(task, step)
            extra += len(candidates)
            chosen = self._choose(task, step, actions, candidates)
            actions.append(chosen)
            if chosen.args.get("trap") == "true":
                break
        return PolicyTrace(task.task_id, actions, _token_count(actions), extra)

    def _candidates(self, task: PlanDepthTask, step: int) -> List[Action]:
        node = task.path[min(step, len(task.path) - 1)]
        good_tool = "inspect" if step == 0 else "move"
        if step == task.trap_step:
            good_tool = "verify"
        if step == len(task.path) - 1:
            good_tool = "commit"
        candidates = [Action(text=f"{good_tool} target={node}", tool=good_tool, args={"target": node})]
        candidates.append(Action(text=f"commit target={node}", tool="commit", args={"target": node, "trap": "true"}))
        candidates.append(Action(text=f"query_constraint target={node}", tool="query_constraint", args={"target": node}))
        return candidates[: self.candidates]

    def _choose(self, task: PlanDepthTask, step: int, history: Sequence[Action], candidates: Sequence[Action]) -> Action:
        best = candidates[0]
        best_score = float("-inf")
        for candidate in candidates:
            text = task.goal + "\n" + "\n".join(action.text for action in list(history) + [candidate])
            features = _features_like_training(text, self.model.config.hidden_dim)
            success_score = self.model.success_prob(features)
            sketch = self.mapper.encode(candidate)
            dead_end_penalty = 1.0 if candidate.args.get("trap") == "true" or sketch.startswith("commit") and step < task.depth - 1 else 0.0
            score = success_score - dead_end_penalty
            if score > best_score:
                best = candidate
                best_score = score
        return best


def _token_count(actions: Iterable[Action]) -> int:
    return sum(max(1, len(action.text.split())) for action in actions)


def _features_like_training(text: str, dim: int) -> List[float]:
    from foreact.data.soft_targets import deterministic_features

    return deterministic_features(text, dim)
