"""Pre-experiment readiness report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from foreact.evaluation.adapters import external_harness_status
from foreact.evaluation.assets import asset_status, asset_summary
from foreact.evaluation.datasets import dataset_status
from foreact.io import load_yaml, read_json, write_json


def readiness_report(config: Mapping[str, object], out: str | Path = "outputs/readiness.json") -> Dict[str, object]:
    asset_manifest = str(config.get("asset_manifest", "third_party/assets.yaml"))
    dataset_config = str(config.get("dataset_config", "configs/datasets.yaml"))
    experiment_config = str(config.get("experiment_config", "configs/foreact_4b_milestone.yaml"))
    assets = asset_status(load_yaml(asset_manifest), project_root=Path.cwd())
    datasets = dataset_status(load_yaml(dataset_config))
    harness = external_harness_status()
    exp = load_yaml(experiment_config)
    paths = exp.get("paths", {}) if isinstance(exp.get("paths"), Mapping) else {}
    prepared = {name: {"path": str(path), "exists": Path(str(path)).exists()} for name, path in paths.items()}
    ablation_manifest_path = Path(str(config.get("ablation_manifest", "outputs/ablation_jobs/ablation_manifest.json")))
    ablation = read_json(ablation_manifest_path) if ablation_manifest_path.exists() else None
    extra_files = {
        "teacher_trajectories": Path(str(config.get("teacher_trajectories", "outputs/teacher/trajectories.jsonl"))),
        "control_task": Path(str(config.get("control_task", "outputs/control_tasks/random_labels.jsonl"))),
        "efficiency_plan": Path(str(config.get("efficiency_plan", "outputs/efficiency/vllm_plan.json"))),
        "lats_baseline_plan": Path(str(config.get("lats_baseline_plan", "outputs/baselines/lats_run_plan.json"))),
        "plan_and_act_baseline_plan": Path(str(config.get("plan_and_act_baseline_plan", "outputs/baselines/plan_and_act_run_plan.json"))),
    }
    extra_status = {name: {"path": str(path), "exists": path.exists()} for name, path in extra_files.items()}
    blocking = []
    for name, status in harness.items():
        if not status.get("available"):
            blocking.append(f"harness:{name}:{status.get('message')}")
    for name, item in prepared.items():
        if not item["exists"]:
            blocking.append(f"prepared_file:{name}:{item['path']}")
    for name, item in extra_status.items():
        if not item["exists"]:
            blocking.append(f"prepared_file:{name}:{item['path']}")
    report = {
        "asset_summary": asset_summary(assets),
        "datasets": datasets,
        "harness": harness,
        "prepared_files": prepared,
        "extra_prepared_files": extra_status,
        "ablation_manifest": str(ablation_manifest_path),
        "ablation_jobs": ablation.get("num_jobs") if isinstance(ablation, Mapping) else None,
        "blocking_items": blocking,
        "ready_except_external_runtime": not any(item.startswith("prepared_file:") for item in blocking),
    }
    write_json(out, report)
    return report
