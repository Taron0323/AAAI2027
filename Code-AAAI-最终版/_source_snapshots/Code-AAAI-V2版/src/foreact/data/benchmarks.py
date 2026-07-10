"""Dataset loaders that convert official assets into ForeAct trajectories.

These loaders intentionally avoid running benchmark evaluation. Their job is to
make the pre-experiment data layer real: official task files become a shared
trajectory/evaluation-row format that training, teacher rollouts, and harness
wrappers can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from foreact.io import read_jsonl
from foreact.types import Action, Observation, Trajectory


@dataclass(frozen=True)
class BenchmarkLoadResult:
    benchmark: str
    trajectories: List[Trajectory]
    eval_rows: List[Dict[str, object]]
    tool_schema: Dict[str, object]
    source_paths: List[str]
    warnings: List[str]


def load_benchmark_trajectories(
    benchmark: str,
    root: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> BenchmarkLoadResult:
    key = benchmark.lower().replace("-", "_")
    if key in {"tau2", "tau2_bench", "tau"}:
        return load_tau2(root, split=split, limit=limit)
    if key == "appworld":
        return load_appworld(root, split=split, limit=limit)
    if key in {"swe_gym", "swegym"}:
        return load_swe_gym(root, split=split, limit=limit)
    if key in {"swe_bench_verified", "swebench_verified", "swe_bench"}:
        return load_swe_bench_verified(root, limit=limit)
    raise ValueError(f"unknown benchmark loader: {benchmark}")


def load_tau2(root: str | Path, split: str = "train", limit: int | None = None) -> BenchmarkLoadResult:
    root_path = Path(root)
    domain_roots = []
    if (root_path / "tasks.json").exists():
        domain_roots = [root_path]
    else:
        domain_roots = sorted(path for path in root_path.iterdir() if (path / "tasks.json").exists())
    trajectories: List[Trajectory] = []
    eval_rows: List[Dict[str, object]] = []
    source_paths: List[str] = []
    warnings: List[str] = []
    for domain_root in domain_roots:
        tasks_path = domain_root / "tasks.json"
        source_paths.append(str(tasks_path))
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        split_ids = _load_tau2_split_ids(domain_root, split)
        for task in tasks:
            task_id = str(task.get("id", len(trajectories)))
            if split_ids is not None and task_id not in split_ids:
                continue
            goal = _tau2_goal(task, domain_root.name)
            actions = _tau2_expected_actions(task)
            trajectories.append(
                Trajectory(
                    task_id=f"tau2-{domain_root.name}-{task_id}",
                    goal=goal,
                    actions=actions,
                    observations=[
                        Observation(
                            text="official tau2 task specification; execute through tau2 harness for real observations",
                            state={"domain": domain_root.name, "split": split},
                        )
                    ]
                    if actions
                    else [],
                    success=bool(actions),
                    metadata={
                        "benchmark": "tau2_bench",
                        "domain": domain_root.name,
                        "split": split,
                        "source_task_id": task_id,
                    },
                )
            )
            eval_rows.append(
                {
                    "benchmark": "tau2_bench",
                    "domain": domain_root.name,
                    "task_id": task_id,
                    "goal": goal,
                    "raw": task,
                }
            )
            if limit is not None and len(trajectories) >= limit:
                return _result("tau2_bench", trajectories, eval_rows, _tool_schema_from_actions(trajectories), source_paths, warnings)
    if not trajectories:
        warnings.append("No tau2 trajectories with action criteria were found; teacher generation must create training actions.")
    return _result("tau2_bench", trajectories, eval_rows, _tool_schema_from_actions(trajectories), source_paths, warnings)


def load_appworld(root: str | Path, split: str = "train", limit: int | None = None) -> BenchmarkLoadResult:
    root_path = Path(root)
    data_root = root_path / "data" if (root_path / "data").exists() else root_path
    split_path = data_root / "datasets" / f"{split}.txt"
    if not split_path.exists() and split == "test":
        split_path = data_root / "datasets" / "test_normal.txt"
    task_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        task_ids = task_ids[:limit]
    trajectories: List[Trajectory] = []
    eval_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for task_id in task_ids:
        spec_path = data_root / "tasks" / task_id / "specs.json"
        if not spec_path.exists():
            warnings.append(f"missing AppWorld specs.json for {task_id}")
            continue
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        goal = str(spec.get("instruction", ""))
        required_apps = _read_optional_json(data_root / "tasks" / task_id / "ground_truth" / "required_apps.json")
        required_apis = _read_optional_json(data_root / "tasks" / task_id / "ground_truth" / "required_apis.json")
        actions = _appworld_actions(required_apps, required_apis)
        trajectories.append(
            Trajectory(
                task_id=f"appworld-{task_id}",
                goal=goal,
                actions=actions,
                observations=[
                    Observation(
                        text="official AppWorld task specification; execute through AppWorld for real observations",
                        state={"split": split},
                    )
                ]
                if actions
                else [],
                success=bool(actions),
                metadata={"benchmark": "appworld", "split": split, "source_task_id": task_id},
            )
        )
        eval_rows.append({"benchmark": "appworld", "task_id": task_id, "goal": goal, "raw": spec})
    if not trajectories:
        warnings.append("No AppWorld trajectory actions were derived; use teacher generation before SFT.")
    return _result("appworld", trajectories, eval_rows, _tool_schema_from_actions(trajectories), [str(split_path)], warnings)


def load_swe_gym(root: str | Path, split: str = "train", limit: int | None = None) -> BenchmarkLoadResult:
    root_path = Path(root)
    rows_path = root_path / "openhands_sft_trajectories" / "train.success.oss.sample.jsonl"
    source_path = rows_path
    rows = _load_hf_rows(root_path / "openhands_sft_trajectories" / "hf_dataset", "train.success.oss", limit=limit)
    if not rows:
        rows = _load_hf_rows(root_path / "openhands_verifier_trajectories" / "hf_dataset", "train.mixture", limit=limit)
        source_path = root_path / "openhands_verifier_trajectories" / "hf_dataset"
    if not rows:
        if not rows_path.exists():
            rows_path = root_path / "openhands_verifier_trajectories" / "train.mixture.sample.jsonl"
        rows = list(read_jsonl(rows_path))
        source_path = rows_path
    if limit is not None:
        rows = rows[:limit]
    trajectories = []
    eval_rows = []
    for idx, row in enumerate(rows):
        messages = row.get("messages", [])
        goal = _messages_goal(messages, fallback=f"SWE-Gym trajectory {idx}")
        actions = _messages_to_actions(messages)
        trajectories.append(
            Trajectory(
                task_id=f"swe-gym-{idx:05d}",
                goal=goal,
                actions=actions,
                observations=_messages_to_observations(messages),
                success=bool(row.get("resolved", True)),
                metadata={"benchmark": "swe_gym", "split": split, "source_index": str(idx)},
            )
        )
        eval_rows.append({"benchmark": "swe_gym", "task_id": f"swe-gym-{idx:05d}", "goal": goal})
    return _result("swe_gym", trajectories, eval_rows, _tool_schema_from_actions(trajectories), [str(source_path)], [])


def load_swe_bench_verified(root: str | Path, limit: int | None = None) -> BenchmarkLoadResult:
    root_path = Path(root)
    rows_path = root_path / "verified" / "test.jsonl"
    if not rows_path.exists():
        rows_path = root_path / "test.jsonl"
    rows = list(read_jsonl(rows_path))
    if limit is not None:
        rows = rows[:limit]
    trajectories = []
    eval_rows = []
    for row in rows:
        instance_id = str(row["instance_id"])
        goal = str(row.get("problem_statement", ""))
        patch = str(row.get("patch", ""))
        actions = [
            Action(text=f"edit file=patch instance={instance_id}", tool="edit", args={"file": "patch", "instance": instance_id}),
            Action(text=f"test target={instance_id}", tool="test", args={"target": instance_id}),
            Action(text=f"submit target={instance_id}", tool="submit", args={"target": instance_id}),
        ]
        trajectories.append(
            Trajectory(
                task_id=f"swe-bench-verified-{instance_id}",
                goal=goal,
                actions=actions,
                observations=[Observation(text="gold patch available for teacher/SFT construction")],
                success=bool(patch),
                metadata={
                    "benchmark": "swe_bench_verified",
                    "instance_id": instance_id,
                    "repo": str(row.get("repo", "")),
                    "base_commit": str(row.get("base_commit", "")),
                },
            )
        )
        eval_rows.append(
            {
                "benchmark": "swe_bench_verified",
                "instance_id": instance_id,
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "problem_statement": goal,
                "gold_patch": patch,
            }
        )
    return _result("swe_bench_verified", trajectories, eval_rows, _tool_schema_from_actions(trajectories), [str(rows_path)], [])


def trajectories_to_rows(trajectories: Iterable[Trajectory]) -> List[Dict[str, object]]:
    rows = []
    for tr in trajectories:
        rows.append(
            {
                "task_id": tr.task_id,
                "goal": tr.goal,
                "success": tr.success,
                "dead_end": tr.dead_end,
                "metadata": dict(tr.metadata),
                "actions": [{"text": a.text, "tool": a.tool, "args": dict(a.args)} for a in tr.actions],
                "observations": [{"text": o.text, "state": dict(o.state), "branch_id": o.branch_id} for o in tr.observations],
            }
        )
    return rows


def rows_to_trajectories(rows: Iterable[Mapping[str, object]]) -> List[Trajectory]:
    trajectories = []
    for row in rows:
        actions = [
            Action(text=str(item.get("text", "")), tool=str(item.get("tool", "unknown")), args=dict(item.get("args", {})))
            for item in row.get("actions", [])  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        ]
        observations = [
            Observation(
                text=str(item.get("text", "")),
                state=dict(item.get("state", {})),
                branch_id=item.get("branch_id"),  # type: ignore[arg-type]
            )
            for item in row.get("observations", [])  # type: ignore[union-attr]
            if isinstance(item, Mapping)
        ]
        trajectories.append(
            Trajectory(
                task_id=str(row.get("task_id", "")),
                goal=str(row.get("goal", "")),
                actions=actions,
                observations=observations,
                success=bool(row.get("success", False)),
                dead_end=bool(row.get("dead_end", False)),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return trajectories


def _tau2_goal(task: Mapping[str, object], domain: str) -> str:
    scenario = task.get("user_scenario", {})
    instructions = scenario.get("instructions", {}) if isinstance(scenario, Mapping) else {}
    parts = [f"tau2 {domain} task {task.get('id')}"]
    for key in ("reason_for_call", "task_instructions", "known_info"):
        value = instructions.get(key) if isinstance(instructions, Mapping) else None
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def _tau2_expected_actions(task: Mapping[str, object]) -> List[Action]:
    criteria = task.get("evaluation_criteria", {})
    raw_actions = criteria.get("actions", []) if isinstance(criteria, Mapping) else []
    actions = []
    for idx, item in enumerate(raw_actions):
        if isinstance(item, Mapping):
            name = str(item.get("action_name") or item.get("name") or item.get("tool") or f"expected_action_{idx}")
            args = item.get("arguments") or item.get("args") or {}
            if not isinstance(args, Mapping):
                args = {"value": str(args)}
            text = " ".join([name] + [f"{key}={value}" for key, value in sorted(args.items())])
            actions.append(Action(text=text, tool=name, args={str(k): str(v) for k, v in args.items()}))
        else:
            text = str(item)
            actions.append(Action(text=text, tool=text.split()[0] if text.split() else "expected_action", args={}))
    return actions


def _load_tau2_split_ids(domain_root: Path, split: str) -> set[str] | None:
    split_path = domain_root / "split_tasks.json"
    if not split_path.exists():
        return None
    data = json.loads(split_path.read_text(encoding="utf-8"))
    ids = data.get(split)
    if not isinstance(ids, list):
        return None
    return {str(item) for item in ids}


def _read_optional_json(path: Path) -> object:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _appworld_actions(required_apps: object, required_apis: object) -> List[Action]:
    actions = []
    apps = required_apps if isinstance(required_apps, list) else []
    apis = required_apis if isinstance(required_apis, list) else []
    for idx, api in enumerate(apis):
        app = str(apps[idx]) if idx < len(apps) else "appworld"
        api_name = str(api)
        actions.append(Action(text=f"{app}.{api_name}", tool=f"{app}.{api_name}", args={"app": app, "api": api_name}))
    return actions


def _messages_goal(messages: object, fallback: str) -> str:
    if not isinstance(messages, list):
        return fallback
    for msg in messages:
        if isinstance(msg, Mapping) and msg.get("role") in {"user", "human"} and msg.get("content"):
            return str(msg["content"])[:4000]
    return fallback


def _messages_to_actions(messages: object) -> List[Action]:
    if not isinstance(messages, list):
        return []
    actions = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        if role in {"assistant", "tool"} and content:
            tool = "execute_bash" if "execute_bash" in content or "bash" in content.lower() else "respond"
            actions.append(Action(text=content[:2000], tool=tool, args={}))
    return actions


def _messages_to_observations(messages: object) -> List[Observation]:
    if not isinstance(messages, list):
        return []
    observations = []
    for msg in messages:
        if isinstance(msg, Mapping) and str(msg.get("role", "")) in {"tool", "observation", "user"}:
            observations.append(Observation(text=str(msg.get("content", ""))[:2000]))
    return observations


def _load_hf_rows(path: Path, split: str, limit: int | None = None) -> List[Mapping[str, object]]:
    if not path.exists():
        return []
    try:
        from datasets import load_from_disk
    except Exception:
        return []
    dataset = load_from_disk(str(path))
    if hasattr(dataset, "keys"):
        if split in dataset:
            dataset = dataset[split]
        else:
            first_key = next(iter(dataset.keys()))
            dataset = dataset[first_key]
    count = len(dataset) if limit is None else min(limit, len(dataset))
    return [dict(dataset[idx]) for idx in range(count)]


def _tool_schema_from_actions(trajectories: Sequence[Trajectory]) -> Dict[str, object]:
    tools: Dict[str, Dict[str, str]] = {}
    for tr in trajectories:
        for action in tr.actions:
            params = tools.setdefault(action.tool, {})
            for key in action.args:
                params[str(key)] = "string"
    return {"tools": [{"name": name, "parameters": params} for name, params in sorted(tools.items())]}


def _result(
    benchmark: str,
    trajectories: List[Trajectory],
    eval_rows: List[Dict[str, object]],
    tool_schema: Dict[str, object],
    source_paths: List[str],
    warnings: List[str],
) -> BenchmarkLoadResult:
    return BenchmarkLoadResult(
        benchmark=benchmark,
        trajectories=trajectories,
        eval_rows=eval_rows,
        tool_schema=tool_schema,
        source_paths=source_paths,
        warnings=warnings,
    )
