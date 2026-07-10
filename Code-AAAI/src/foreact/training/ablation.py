"""Ablation matrix helpers."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Mapping


def expand_ablation_jobs(matrix: Mapping[str, object]) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for item in matrix.get("main_controls", []):
        if isinstance(item, dict):
            jobs.append({"group": "main_controls", **item})
    for mode in matrix.get("sketch_granularity", []):
        jobs.append({"group": "A_prime_sketch_granularity", "key": f"A_prime_zeta_{mode}", "zeta_mode": mode})
    for horizon in matrix.get("horizon_sweep", []):
        jobs.append({"group": "D_horizon_sweep", "key": f"D_H{horizon}", "horizon": horizon})
    for rollouts in matrix.get("rollout_sweep", []):
        jobs.append({"group": "E_rollout_sweep", "key": f"E_K{rollouts}", "rollouts": rollouts})
    for lam, mu in product(matrix.get("lambda_sweep", []), matrix.get("mu_sweep", [])):
        jobs.append(
            {
                "group": "F_loss_sweep",
                "key": f"F_lambda{lam}_mu{mu}",
                "lambda_future": lam,
                "mu_consistency": mu,
            }
        )
    for item in matrix.get("rerank_controls", []):
        if isinstance(item, dict):
            jobs.append({"group": "C_rerank_controls", **item})
    for item in matrix.get("teacher_controls", []):
        if isinstance(item, dict):
            jobs.append({"group": "E_prime_teacher_controls", **item})
    for item in matrix.get("anti_confounds", []):
        if isinstance(item, dict):
            jobs.append({"group": "G_anti_confounds", **item})
    return jobs
