"""Pre-experiment data preparation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from foreact.data.benchmarks import (
    BenchmarkLoadResult,
    load_benchmark_trajectories,
    trajectories_to_rows,
)
from foreact.data.soft_targets import build_aligned_examples, examples_to_rows
from foreact.io import write_json, write_jsonl
from foreact.teachers.providers import CachedAPIProvider, MockTeacherProvider, TeacherProvider
from foreact.types import Action, Trajectory
from foreact.zeta import SketchMapper, build_schema_from_actions
from foreact.zeta.schema import build_schema_from_tool_schema
from foreact.zeta.vq import build_vq_schema


def prepare_real_data(config: Mapping[str, object]) -> Dict[str, object]:
    output_dir = Path(str(config.get("output_dir", "outputs/real_data")))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = _mapping(config.get("data"))
    zeta_cfg = _mapping(config.get("zeta"))
    soft_cfg = _mapping(config.get("soft_targets"))
    teacher_cfg = _mapping(config.get("teacher"))

    limit = _optional_int(data_cfg.get("limit") or data_cfg.get("target_trajectories"))
    split = str(data_cfg.get("split", "train"))
    benchmark_specs = _benchmark_specs(data_cfg)
    loaded: List[BenchmarkLoadResult] = []
    trajectories: List[Trajectory] = []
    eval_rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for spec in benchmark_specs:
        result = load_benchmark_trajectories(
            str(spec["name"]),
            spec["root"],
            split=str(spec.get("split", split)),
            limit=_optional_int(spec.get("limit")) or limit,
        )
        loaded.append(result)
        trajectories.extend(result.trajectories)
        eval_rows.extend(result.eval_rows)
        warnings.extend(result.warnings)

    if not trajectories:
        raise RuntimeError("No trajectories were loaded; check data.benchmarks roots/splits.")

    write_jsonl(output_dir / "trajectories.jsonl", trajectories_to_rows(trajectories))
    write_jsonl(output_dir / "eval_rows.jsonl", eval_rows)

    all_actions = [action for tr in trajectories for action in tr.actions]
    if zeta_cfg.get("tool_schema"):
        import json

        schema = build_schema_from_tool_schema(json.loads(Path(str(zeta_cfg["tool_schema"])).read_text(encoding="utf-8")))
    elif str(zeta_cfg.get("mode", "type_arg")) == "vq":
        schema = build_vq_schema(
            all_actions,
            codebook_size=int(zeta_cfg.get("vq_codebook_size", 512)),
            dim=int(zeta_cfg.get("vq_dim", 64)),
            iterations=int(zeta_cfg.get("vq_iterations", 20)),
        )
    else:
        schema = build_schema_from_actions(all_actions, mode=str(zeta_cfg.get("mode", "type_arg")))
    if "<unk>" not in schema["sketches"]:
        schema["sketches"].append("<unk>")
        schema["size"] = len(schema["sketches"])
    write_json(output_dir / "schema.json", schema)
    mapper = SketchMapper(schema)

    provider = _build_provider(teacher_cfg)
    horizon = int(soft_cfg.get("horizon", 8))
    rollouts = int(soft_cfg.get("rollouts", 8))
    prefix_fraction = float(soft_cfg.get("prefix_fraction", data_cfg.get("prefix_sample_fraction", 1.0)))
    continuation_lookup: Dict[tuple, Sequence[Sequence[Action]]] = {}
    prefix_count = 0
    for tr in trajectories:
        for idx, _action in enumerate(tr.actions):
            if prefix_fraction < 1.0 and (idx + 1) / max(1, len(tr.actions)) > prefix_fraction:
                continue
            prefix = tr.prefix(idx)
            continuation_lookup[(tr.task_id, idx)] = provider.continue_actions(prefix, horizon=horizon, k=rollouts)
            prefix_count += 1

    hidden_dim = int(_mapping(config.get("training")).get("hidden_dim", 64))
    examples = build_aligned_examples(
        trajectories,
        mapper,
        continuation_lookup,
        horizon=horizon,
        hidden_dim=hidden_dim,
    )
    write_jsonl(output_dir / "aligned_examples.jsonl", examples_to_rows(examples))

    manifest = {
        "output_dir": str(output_dir),
        "benchmarks": [
            {
                "benchmark": item.benchmark,
                "trajectories": len(item.trajectories),
                "eval_rows": len(item.eval_rows),
                "source_paths": item.source_paths,
                "warnings": item.warnings,
            }
            for item in loaded
        ],
        "num_trajectories": len(trajectories),
        "num_eval_rows": len(eval_rows),
        "num_prefixes": prefix_count,
        "num_aligned_examples": len(examples),
        "schema_size": len(schema["sketches"]),
        "horizon": horizon,
        "rollouts": rollouts,
        "teacher": {
            "provider": str(teacher_cfg.get("provider", "mock")),
            "model_id": str(teacher_cfg.get("continuation_model", teacher_cfg.get("model_id", ""))),
            "cache_dir": str(teacher_cfg.get("cache_dir", "")),
        },
        "warnings": warnings,
        "files": {
            "trajectories": str(output_dir / "trajectories.jsonl"),
            "eval_rows": str(output_dir / "eval_rows.jsonl"),
            "schema": str(output_dir / "schema.json"),
            "aligned_examples": str(output_dir / "aligned_examples.jsonl"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _benchmark_specs(data_cfg: Mapping[str, object]) -> List[Mapping[str, object]]:
    raw = data_cfg.get("benchmarks")
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    roots = _mapping(data_cfg.get("roots"))
    if roots:
        return [{"name": name, "root": root} for name, root in roots.items()]
    domains = data_cfg.get("train_domains", [])
    specs = []
    if isinstance(domains, list):
        for name in domains:
            key = str(name)
            if key.startswith("tau2"):
                specs.append({"name": "tau2_bench", "root": "third_party/benchmarks/tau2-bench/data/tau2/domains"})
            elif key == "appworld":
                specs.append({"name": "appworld", "root": "datasets/appworld"})
            elif key == "swe_gym":
                specs.append({"name": "swe_gym", "root": "datasets/swe-gym"})
            elif key == "swe_bench_verified":
                specs.append({"name": "swe_bench_verified", "root": "datasets/swe-bench"})
    return specs


def _build_provider(teacher_cfg: Mapping[str, object]) -> TeacherProvider:
    provider = str(teacher_cfg.get("provider", "mock")).lower()
    if provider == "mock":
        return MockTeacherProvider()
    model_id = str(teacher_cfg.get("continuation_model", teacher_cfg.get("model_id", "deepseek-v4-flash")))
    env_cfg = _mapping(teacher_cfg.get("api_key_env"))
    if provider == "deepseek" or model_id.startswith("deepseek"):
        key_env = str(env_cfg.get("deepseek", teacher_cfg.get("api_key_env", "DEEPSEEK_API_KEY")))
    else:
        key_env = str(env_cfg.get("openai", teacher_cfg.get("api_key_env", "OPENAI_API_KEY")))
    return CachedAPIProvider(
        model_id=model_id,
        api_key_env=key_env,
        cache_dir=str(teacher_cfg.get("cache_dir", ".cache/teacher_rollouts")),
        base_url=teacher_cfg.get("base_url"),  # type: ignore[arg-type]
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
