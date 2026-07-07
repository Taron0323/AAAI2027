"""Generate teacher main trajectories before soft-target construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

from foreact.data.benchmarks import rows_to_trajectories, trajectories_to_rows
from foreact.io import read_jsonl, write_json, write_jsonl
from foreact.teachers.providers import CachedAPIProvider, MockTeacherProvider, TeacherProvider
from foreact.types import Action, Observation, Trajectory, TrajectoryPrefix


def generate_teacher_trajectories(config: Mapping[str, object]) -> Dict[str, object]:
    input_path = Path(str(config.get("eval_rows", config.get("input", "outputs/foreact_4b_milestone/eval_rows.jsonl"))))
    output_path = Path(str(config.get("out", "outputs/teacher/trajectories.jsonl")))
    horizon = int(config.get("horizon", 12))
    limit = config.get("limit")
    provider = _provider(config)
    rows = list(read_jsonl(input_path))
    if limit is not None:
        rows = rows[: int(limit)]
    trajectories = []
    for row in rows:
        prefix = TrajectoryPrefix(
            task_id=str(row.get("task_id", row.get("instance_id", len(trajectories)))),
            goal=str(row.get("goal", row.get("problem_statement", ""))),
            actions=[],
            observations=[],
            next_action_index=0,
            success=False,
            metadata={"benchmark": str(row.get("benchmark", ""))},
        )
        actions = provider.continue_actions(prefix, horizon=horizon, k=1)[0]
        success, dead_end = _validate_trajectory(row, actions)
        trajectories.append(
            Trajectory(
                task_id=prefix.task_id,
                goal=prefix.goal,
                actions=actions,
                observations=[Observation(text="teacher-generated trajectory; validate with official harness for final labels")],
                success=success,
                dead_end=dead_end,
                metadata={"benchmark": str(row.get("benchmark", "")), "teacher_generated": "true"},
            )
        )
    write_jsonl(output_path, trajectories_to_rows(trajectories))
    manifest = {
        "input": str(input_path),
        "out": str(output_path),
        "num_input_rows": len(rows),
        "num_trajectories": len(trajectories),
        "provider": str(config.get("provider", "mock")),
        "horizon": horizon,
        "note": "Success labels are lightweight validators; official benchmark harnesses remain authoritative.",
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def _provider(config: Mapping[str, object]) -> TeacherProvider:
    provider = str(config.get("provider", "mock")).lower()
    if provider == "mock":
        return MockTeacherProvider()
    model_id = str(config.get("model_id", config.get("main_trajectory_model", "deepseek-v4-pro")))
    if model_id.startswith("deepseek"):
        key_env = str(config.get("api_key_env", "DEEPSEEK_API_KEY"))
    else:
        key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    return CachedAPIProvider(
        model_id=model_id,
        api_key_env=key_env,
        cache_dir=str(config.get("cache_dir", ".cache/teacher_main_trajectories")),
        base_url=config.get("base_url"),  # type: ignore[arg-type]
    )


def _validate_trajectory(row: Mapping[str, object], actions: list[Action]) -> tuple[bool, bool]:
    if not actions:
        return False, True
    text = "\n".join(action.text.lower() for action in actions)
    dead_end = any(marker in text for marker in ("error", "dead_end", "rollback failed", "unauthorized"))
    benchmark = str(row.get("benchmark", ""))
    if benchmark == "swe_bench_verified":
        success = any(action.tool in {"submit", "final", "patch"} or "diff --git" in action.text for action in actions)
    elif benchmark == "appworld":
        success = not dead_end and any("." in action.tool or action.tool.startswith("api") for action in actions)
    elif benchmark == "tau2_bench":
        success = not dead_end and bool(actions)
    else:
        success = not dead_end
    return success, dead_end
