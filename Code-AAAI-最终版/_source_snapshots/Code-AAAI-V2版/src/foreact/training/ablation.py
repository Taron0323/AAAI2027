"""Ablation matrix helpers."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, List, Mapping

from foreact.io import write_json


def expand_ablation_jobs(matrix: Mapping[str, object]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for item in matrix.get("main_controls", []):
        if isinstance(item, dict):
            jobs.append({"group": "main_controls", **item})
    for mode in matrix.get("sketch_granularity", []):
        jobs.append({"group": "sketch_granularity", "key": f"zeta_{mode}", "zeta_mode": mode})
    for horizon in matrix.get("horizon_sweep", []):
        jobs.append({"group": "horizon_sweep", "key": f"H{horizon}", "horizon": horizon})
    for rollouts in matrix.get("rollout_sweep", []):
        jobs.append({"group": "rollout_sweep", "key": f"K{rollouts}", "rollouts": rollouts})
    for lam, mu in product(matrix.get("lambda_sweep", []), matrix.get("mu_sweep", [])):
        jobs.append(
            {
                "group": "loss_sweep",
                "key": f"lambda{lam}_mu{mu}",
                "lambda_future": lam,
                "mu_consistency": mu,
            }
        )
    return jobs


def materialize_ablation_jobs(
    matrix: Mapping[str, object],
    base_config: Mapping[str, object],
    output_dir: str | Path,
    execute: bool = False,
) -> Dict[str, object]:
    """Write per-ablation configs and train/probe command plans.

    The command plans are the real experiment entrypoints. They deliberately do
    not call smoke code; execution can be enabled by the caller once external
    GPU/model resources are configured.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job in expand_ablation_jobs(matrix):
        cfg = _deepcopy_mapping(base_config)
        _apply_job(cfg, job)
        job_dir = out / str(job["key"])
        job_dir.mkdir(parents=True, exist_ok=True)
        cfg["output_dir"] = str(job_dir)
        cfg_path = job_dir / "config.yaml"
        _write_yaml(cfg_path, cfg)
        command = ["python3", "-m", "foreact.cli", "train-foreact", "--config", str(cfg_path)]
        jobs.append({"job": job, "config": str(cfg_path), "command": command, "execute": execute})
    manifest = {"output_dir": str(out), "num_jobs": len(jobs), "jobs": jobs}
    write_json(out / "ablation_manifest.json", manifest)
    return manifest


def _apply_job(cfg: Dict[str, object], job: Mapping[str, object]) -> None:
    training = cfg.setdefault("training", {})
    zeta = cfg.setdefault("zeta", {})
    soft = cfg.setdefault("soft_targets", {})
    if not isinstance(training, dict) or not isinstance(zeta, dict) or not isinstance(soft, dict):
        raise ValueError("base config training/zeta/soft_targets must be mappings")
    key = str(job.get("key", ""))
    if key == "react_sft" or key == "A":
        training["variant"] = "react_sft"
        training["lambda_future"] = 0.0
        training["mu_consistency"] = 0.0
    elif key == "token_mtp":
        training["variant"] = "token_mtp"
    elif key == "no_pcr":
        training["variant"] = "no_pcr"
        training["mu_consistency"] = 0.0
    elif key == "predict_past" or key == "G_predict_past":
        training["variant"] = "predict_past"
    if "zeta_mode" in job:
        zeta["mode"] = job["zeta_mode"]
    if "horizon" in job:
        training["horizon"] = job["horizon"]
        soft["horizon"] = job["horizon"]
    if "rollouts" in job:
        soft["rollouts"] = job["rollouts"]
    if "lambda_future" in job:
        training["lambda_future"] = job["lambda_future"]
    if "mu_consistency" in job:
        training["mu_consistency"] = job["mu_consistency"]


def _deepcopy_mapping(value: Mapping[str, object]) -> Dict[str, object]:
    import copy

    return copy.deepcopy(dict(value))


def _write_yaml(path: Path, data: Mapping[str, object]) -> None:
    import yaml

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)
