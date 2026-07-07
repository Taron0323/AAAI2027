"""Official benchmark runner command construction and execution."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from foreact.io import write_json


@dataclass(frozen=True)
class BenchmarkRunSpec:
    benchmark: str
    command: List[str]
    cwd: str
    required_packages: List[str]
    source_paths: List[str]
    notes: str


def build_benchmark_run_spec(config: Mapping[str, object]) -> BenchmarkRunSpec:
    benchmark = str(config.get("benchmark", config.get("name", ""))).lower().replace("-", "_")
    if benchmark in {"tau2", "tau2_bench", "tau"}:
        domain = str(config.get("domain", "airline"))
        agent_llm = str(config.get("agent_llm", config.get("model", "gpt-4.1")))
        user_llm = str(config.get("user_llm", agent_llm))
        num_tasks = str(config.get("num_tasks", 5))
        num_trials = str(config.get("num_trials", 1))
        return BenchmarkRunSpec(
            benchmark="tau2_bench",
            command=[
                "python3",
                "-m",
                "tau2.cli",
                "run",
                "--domain",
                domain,
                "--agent-llm",
                agent_llm,
                "--user-llm",
                user_llm,
                "--num-trials",
                num_trials,
                "--num-tasks",
                num_tasks,
            ],
            cwd=str(config.get("cwd", ".")),
            required_packages=["tau2"],
            source_paths=["third_party/benchmarks/tau2-bench/src"],
            notes="Runs the official tau2 CLI. Use a LiteLLM model id/API keys accepted by tau2.",
        )
    if benchmark == "appworld":
        experiment_name = str(config.get("experiment_name", "foreact"))
        split = str(config.get("split", "test_normal"))
        python = str(config.get("python", "python3"))
        root = str(config.get("root", "datasets/appworld"))
        return BenchmarkRunSpec(
            benchmark="appworld",
            command=[
                python,
                "-m",
                "appworld.cli",
                "evaluate",
                experiment_name,
                split,
                "--root",
                root,
            ],
            cwd=str(config.get("cwd", ".")),
            required_packages=["appworld"],
            source_paths=["third_party/benchmarks/appworld/src"],
            notes="Evaluates an AppWorld experiment directory produced by a runnable agent.",
        )
    if benchmark in {"swe_bench_verified", "swebench_verified", "swe_bench"}:
        predictions = str(config.get("predictions_path", "gold"))
        run_id = str(config.get("run_id", "foreact-verified"))
        max_workers = str(config.get("max_workers", 1))
        dataset_name = str(config.get("dataset_name", "SWE-bench/SWE-bench_Verified"))
        command = [
            "python3",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--predictions_path",
            predictions,
            "--max_workers",
            max_workers,
            "--run_id",
            run_id,
        ]
        if config.get("modal"):
            command.extend(["--modal", "true"])
        return BenchmarkRunSpec(
            benchmark="swe_bench_verified",
            command=command,
            cwd=str(config.get("cwd", ".")),
            required_packages=["swebench"],
            source_paths=["third_party/benchmarks/SWE-bench"],
            notes="Runs official SWE-bench harness; requires Docker or Modal for real scoring.",
        )
    if benchmark in {"mini_swe_agent", "mini"}:
        return BenchmarkRunSpec(
            benchmark="mini_swe_agent",
            command=["python3", "-m", "minisweagent.run.mini", "--help"],
            cwd=str(config.get("cwd", ".")),
            required_packages=["minisweagent"],
            source_paths=["third_party/benchmarks/mini-swe-agent/src"],
            notes="Smoke-checks mini-SWE-agent CLI availability for SWE-bench execution.",
        )
    raise ValueError(f"unknown benchmark run spec: {benchmark}")


def run_benchmark_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    output_path = Path(str(config.get("out", "outputs/benchmark_run.json")))
    execute = bool(config.get("execute", False))
    spec = build_benchmark_run_spec(config)
    source_ready = [path for path in spec.source_paths if Path(path).exists()]
    missing = [pkg for pkg in spec.required_packages if importlib.util.find_spec(pkg) is None]
    if source_ready:
        missing = []
    result: Dict[str, object] = {
        "benchmark": spec.benchmark,
        "command": spec.command,
        "cwd": spec.cwd,
        "required_packages": spec.required_packages,
        "source_paths": spec.source_paths,
        "missing_packages": missing,
        "notes": spec.notes,
        "executed": False,
    }
    if missing:
        result["status"] = "not_ready"
        result["message"] = f"Install missing packages first: {', '.join(missing)}"
    elif execute:
        env = os.environ.copy()
        if source_ready:
            env["PYTHONPATH"] = os.pathsep.join(source_ready + [env.get("PYTHONPATH", "")])
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
        result["status"] = "ready_source" if source_ready else "ready"
        result["message"] = "Command is ready; rerun with execute=true to launch official harness."
    write_json(output_path, result)
    return result


def write_swe_predictions(path: str | Path, predictions: Mapping[str, str]) -> Dict[str, object]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for instance_id, patch in predictions.items():
            handle.write(json.dumps({"instance_id": instance_id, "model_patch": patch}, sort_keys=True) + "\n")
    return {"path": str(target), "num_predictions": len(predictions)}
