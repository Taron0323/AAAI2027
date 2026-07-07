"""Deployment and efficiency measurement helpers."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping
from urllib.request import Request, urlopen

from foreact.io import read_jsonl, write_json


def build_vllm_command(config: Mapping[str, object]) -> Dict[str, object]:
    model = str(config.get("model", config.get("model_path", "outputs/foreact_4b_milestone/mode_a")))
    port = str(config.get("port", 8000))
    tensor_parallel = str(config.get("tensor_parallel_size", 1))
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        port,
        "--tensor-parallel-size",
        tensor_parallel,
    ]
    return {"command": command, "base_url": f"http://127.0.0.1:{port}/v1/chat/completions"}


def measure_openai_compatible_latency(config: Mapping[str, object]) -> Dict[str, object]:
    prompts_path = Path(str(config.get("prompts", "outputs/foreact_4b_milestone/eval_rows.jsonl")))
    out = Path(str(config.get("out", "outputs/efficiency/latency.json")))
    base_url = str(config.get("base_url", "http://127.0.0.1:8000/v1/chat/completions"))
    model = str(config.get("model", "foreact-mode-a"))
    limit = int(config.get("limit", 16))
    rows = list(read_jsonl(prompts_path))[:limit]
    measurements = []
    for row in rows:
        prompt = str(row.get("goal", row.get("problem_statement", "")))
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(config.get("max_tokens", 128)),
            "temperature": float(config.get("temperature", 0.0)),
        }
        started = time.perf_counter()
        response = _post_json(base_url, payload)
        elapsed = time.perf_counter() - started
        usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
        measurements.append(
            {
                "task_id": row.get("task_id", row.get("instance_id")),
                "latency_s": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    summary = _summary(measurements)
    result = {"base_url": base_url, "model": model, "num_requests": len(measurements), "summary": summary, "measurements": measurements}
    write_json(out, result)
    return result


def run_efficiency_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    mode = str(config.get("mode", "plan")).lower()
    out = Path(str(config.get("out", "outputs/efficiency/plan.json")))
    if mode == "plan":
        result = build_vllm_command(config)
        write_json(out, result)
        return result
    if mode == "measure":
        return measure_openai_compatible_latency(config)
    raise ValueError(f"unknown efficiency mode: {mode}")


def _post_json(url: str, payload: Mapping[str, object]) -> Mapping[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _summary(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    values = list(rows)
    latencies = [float(row["latency_s"]) for row in values]
    tokens = [float(row["total_tokens"]) for row in values if row.get("total_tokens") is not None]
    return {
        "latency_mean_s": sum(latencies) / max(1, len(latencies)),
        "latency_max_s": max(latencies) if latencies else 0.0,
        "tokens_mean": sum(tokens) / max(1, len(tokens)) if tokens else 0.0,
    }
