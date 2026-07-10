"""Command wrappers for official/public baseline implementations."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from foreact.io import write_json


@dataclass(frozen=True)
class BaselineCommand:
    key: str
    command: List[str]
    cwd: str
    source_paths: List[str]
    notes: str


def build_baseline_command(config: Mapping[str, object]) -> BaselineCommand:
    key = str(config.get("baseline", config.get("key", ""))).lower()
    python = str(config.get("python", "python3"))
    if key in {"plan_and_act", "preact"}:
        script = "run_plan_and_act_with_replanning.py" if key == "plan_and_act" else "run_plan_and_act.py"
        return BaselineCommand(
            key=key,
            command=[python, script, "--help"],
            cwd="third_party/baselines/plan-and-act",
            source_paths=["third_party/baselines/plan-and-act"],
            notes="Official Plan-and-Act/PreAct checkout. Replace --help with benchmark-specific args after selecting task adapter.",
        )
    if key in {"lats", "lats_webdreamer"}:
        return BaselineCommand(
            key=key,
            command=[python, "programming/main.py", "--help"],
            cwd="third_party/baselines/LanguageAgentTreeSearch",
            source_paths=["third_party/baselines/LanguageAgentTreeSearch"],
            notes="Official LATS checkout. Use programming/main.py for SWE-like tasks or domain-specific run.py scripts.",
        )
    if key == "webdreamer":
        return BaselineCommand(
            key=key,
            command=[python, "-c", "import controller, world_model, simulation_scoring; print('webdreamer modules importable')"],
            cwd="third_party/baselines/WebDreamer",
            source_paths=["third_party/baselines/WebDreamer"],
            notes="Official WebDreamer modules. Full runs require a world-model/provider configuration.",
        )
    if key == "react":
        return BaselineCommand(
            key=key,
            command=[python, "-c", "import wikienv, wrappers; print('react reference importable')"],
            cwd="third_party/baselines/ReAct",
            source_paths=["third_party/baselines/ReAct"],
            notes="Reference ReAct checkout; ForeAct also has built-in ReAct-compatible policy/training variants.",
        )
    raise ValueError(f"unknown baseline command: {key}")


def run_baseline_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    out = Path(str(config.get("out", "outputs/baselines/run.json")))
    execute = bool(config.get("execute", False))
    spec = build_baseline_command(config)
    missing_paths = [path for path in spec.source_paths if not Path(path).exists()]
    result: Dict[str, object] = {
        "baseline": spec.key,
        "command": spec.command,
        "cwd": spec.cwd,
        "source_paths": spec.source_paths,
        "missing_paths": missing_paths,
        "notes": spec.notes,
        "executed": False,
    }
    if missing_paths:
        result["status"] = "not_ready"
    elif execute:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(spec.source_paths + [env.get("PYTHONPATH", "")])
        completed = subprocess.run(spec.command, cwd=spec.cwd, text=True, capture_output=True, check=False, env=env)
        result.update(
            {
                "executed": True,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "status": "ok" if completed.returncode == 0 else "failed",
            }
        )
    else:
        result["status"] = "ready_source"
    write_json(out, result)
    return result
