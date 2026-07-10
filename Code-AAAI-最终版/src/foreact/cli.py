"""ForeAct command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from foreact.analysis.exports import (
    write_depth_curve,
    write_efficiency,
    write_forecast_entropy,
    write_mechanism_probe,
    write_rows,
)
from foreact.analysis.mechanism import (
    effective_lookahead_depth,
    eld_curve_rows,
    forecast_entropy_by_depth,
    granularity_diagnostic_rows,
    plandepth_boundary_rows,
    smoke_eld_curve,
)
from foreact.analysis.case_study import plandepth_case_rows
from foreact.analysis.control_tasks import build_control_task_from_config
from foreact.analysis.eld_probe import run_eld_probe_from_config
from foreact.analysis.visualize import (
    write_case_heatmap_svg,
    write_depth_svg,
    write_efficiency_svg,
    write_eld_svg,
    write_pareto_svg,
)
from foreact.baselines.registry import registry
from foreact.baselines.runners import run_baseline_from_config
from foreact.data.prepare import prepare_real_data
from foreact.data.soft_targets import build_aligned_examples, examples_to_rows
from foreact.environments.plandepth import PlanDepthEnv, trajectories_to_rows
from foreact.evaluation.assets import asset_audit, asset_status, asset_summary
from foreact.evaluation.adapters import external_harness_status
from foreact.evaluation.datasets import dataset_status
from foreact.evaluation.efficiency import run_efficiency_from_config
from foreact.evaluation.metrics import evaluate_plandepth
from foreact.evaluation.readiness import readiness_report
from foreact.evaluation.runners import run_benchmark_from_config
from foreact.inference.policies import LatentRerankPolicy, OracleSFTPolicy, ReActPolicy
from foreact.inference.policies import PlanAndActPolicy, SearchLookaheadPolicy, TokenMTPPolicy
from foreact.io import load_yaml, write_json, write_jsonl
from foreact.teachers.trajectory_generation import generate_teacher_trajectories
from foreact.training.ablation import expand_ablation_jobs, materialize_ablation_jobs
from foreact.training.full_trainer import export_mode_a_checkpoint, train_foreact_from_config
from foreact.training.toy_trainer import ToyForeActConfig, train_toy_foreact
from foreact.zeta import SketchMapper, build_schema_from_actions
from foreact.zeta.schema import build_schema_from_tool_schema


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foreact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run the tiny end-to-end PlanDepth pipeline.")
    smoke_parser.add_argument("--config", required=True, help="YAML config path.")

    registry_parser = subparsers.add_parser("registry", help="Print required baselines/ablations.")
    registry_parser.add_argument("--out", help="Optional JSON output path.")

    harness_parser = subparsers.add_parser("harness-status", help="Check external benchmark harness availability.")
    harness_parser.add_argument("--out", help="Optional JSON output path.")

    dataset_parser = subparsers.add_parser("dataset-status", help="Check configured external dataset paths.")
    dataset_parser.add_argument("--config", default="configs/datasets.yaml")
    dataset_parser.add_argument("--out", help="Optional JSON output path.")

    asset_parser = subparsers.add_parser("asset-status", help="Check third-party benchmark/baseline/data assets.")
    asset_parser.add_argument("--manifest", default="third_party/assets.yaml")
    asset_parser.add_argument("--out", help="Optional JSON output path.")

    asset_audit_parser = subparsers.add_parser(
        "asset-audit",
        help="Audit that external assets exclude model weights and use allowed groups.",
    )
    asset_audit_parser.add_argument("--manifest", default="third_party/assets.yaml")
    asset_audit_parser.add_argument("--out", help="Optional JSON output path.")

    ablation_parser = subparsers.add_parser("ablation-jobs", help="Expand the configured ablation matrix.")
    ablation_parser.add_argument("--config", required=True, help="Ablation matrix YAML path.")
    ablation_parser.add_argument("--out", required=True, help="JSON output path.")

    materialize_parser = subparsers.add_parser("materialize-ablation", help="Write train/probe job configs for an ablation matrix.")
    materialize_parser.add_argument("--matrix", required=True, help="Ablation matrix YAML path.")
    materialize_parser.add_argument("--base-config", required=True, help="Base experiment YAML path.")
    materialize_parser.add_argument("--out-dir", required=True, help="Output directory for job configs.")

    schema_parser = subparsers.add_parser("schema-from-tools", help="Build a zeta schema from a tool schema JSON file.")
    schema_parser.add_argument("--tool-schema", required=True)
    schema_parser.add_argument("--mode", default="type_arg")
    schema_parser.add_argument("--out", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare official benchmark trajectories and ForeAct soft targets.")
    prepare_parser.add_argument("--config", required=True, help="Experiment YAML path.")

    train_parser = subparsers.add_parser("train-foreact", help="Run the reference HF/Qwen ForeAct trainer.")
    train_parser.add_argument("--config", required=True, help="Experiment YAML path.")

    probe_parser = subparsers.add_parser("probe-eld", help="Run hidden-state ELD probes.")
    probe_parser.add_argument("--config", required=True, help="Experiment YAML path.")

    benchmark_parser = subparsers.add_parser("run-benchmark", help="Build or execute official benchmark harness commands.")
    benchmark_parser.add_argument("--config", required=True, help="Benchmark run YAML path.")
    benchmark_parser.add_argument("--execute", action="store_true", help="Actually run the official harness command.")

    export_parser = subparsers.add_parser("export-mode-a", help="Export a zero-overhead mode-A backbone checkpoint.")
    export_parser.add_argument("--checkpoint", required=True)
    export_parser.add_argument("--out-dir", required=True)

    readiness_parser = subparsers.add_parser("readiness", help="Write a pre-experiment readiness report.")
    readiness_parser.add_argument("--config", default="configs/readiness.yaml")
    readiness_parser.add_argument("--out", default="outputs/readiness.json")

    gen_parser = subparsers.add_parser("generate-trajectories", help="Generate teacher main trajectories from eval rows.")
    gen_parser.add_argument("--config", required=True)

    baseline_parser = subparsers.add_parser("run-baseline", help="Build or execute official baseline commands.")
    baseline_parser.add_argument("--config", required=True)
    baseline_parser.add_argument("--execute", action="store_true")

    efficiency_parser = subparsers.add_parser("run-efficiency", help="Plan vLLM serving or measure OpenAI-compatible latency.")
    efficiency_parser.add_argument("--config", required=True)

    control_parser = subparsers.add_parser("build-control-task", help="Build Hewitt-Liang random-label control targets.")
    control_parser.add_argument("--config", required=True)

    args = parser.parse_args(argv)
    if args.command == "smoke":
        run_smoke(args.config)
        return 0
    if args.command == "registry":
        data = registry()
        if args.out:
            write_json(args.out, data)
        else:
            print(data)
        return 0
    if args.command == "harness-status":
        data = external_harness_status()
        if args.out:
            write_json(args.out, data)
        else:
            print(data)
        return 0
    if args.command == "dataset-status":
        data = dataset_status(load_yaml(args.config))
        if args.out:
            write_json(args.out, data)
        else:
            print(data)
        return 0
    if args.command == "asset-status":
        status = asset_status(load_yaml(args.manifest), project_root=Path(args.manifest).resolve().parent.parent)
        data = {"summary": asset_summary(status), "assets": status}
        if args.out:
            write_json(args.out, data)
        else:
            print(data)
        return 0
    if args.command == "asset-audit":
        data = asset_audit(load_yaml(args.manifest), project_root=Path(args.manifest).resolve().parent.parent)
        if args.out:
            write_json(args.out, data)
        else:
            print(data)
        return 0
    if args.command == "ablation-jobs":
        write_json(args.out, expand_ablation_jobs(load_yaml(args.config)))
        return 0
    if args.command == "materialize-ablation":
        materialize_ablation_jobs(load_yaml(args.matrix), load_yaml(args.base_config), args.out_dir)
        return 0
    if args.command == "schema-from-tools":
        import json

        with Path(args.tool_schema).open("r", encoding="utf-8") as handle:
            tool_schema = json.load(handle)
        write_json(args.out, build_schema_from_tool_schema(tool_schema, mode=args.mode))
        return 0
    if args.command == "prepare-data":
        data = prepare_real_data(load_yaml(args.config))
        print(data)
        return 0
    if args.command == "train-foreact":
        data = train_foreact_from_config(load_yaml(args.config))
        print(data)
        return 0
    if args.command == "probe-eld":
        data = run_eld_probe_from_config(load_yaml(args.config))
        print(data)
        return 0
    if args.command == "run-benchmark":
        cfg = load_yaml(args.config)
        if args.execute:
            cfg["execute"] = True
        data = run_benchmark_from_config(cfg)
        print(data)
        return 0
    if args.command == "export-mode-a":
        data = export_mode_a_checkpoint(args.checkpoint, args.out_dir)
        print(data)
        return 0
    if args.command == "readiness":
        cfg = load_yaml(args.config) if Path(args.config).exists() else {}
        data = readiness_report(cfg, out=args.out)
        print(data)
        return 0
    if args.command == "generate-trajectories":
        data = generate_teacher_trajectories(load_yaml(args.config))
        print(data)
        return 0
    if args.command == "run-baseline":
        cfg = load_yaml(args.config)
        if args.execute:
            cfg["execute"] = True
        data = run_baseline_from_config(cfg)
        print(data)
        return 0
    if args.command == "run-efficiency":
        data = run_efficiency_from_config(load_yaml(args.config))
        print(data)
        return 0
    if args.command == "build-control-task":
        data = build_control_task_from_config(load_yaml(args.config))
        print(data)
        return 0
    raise AssertionError(args.command)


def run_smoke(config_path: str) -> Dict[str, object]:
    cfg = load_yaml(config_path)
    output_dir = Path(cfg.get("output_dir", "outputs/smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)

    env = PlanDepthEnv(seed=int(cfg.get("seed", 0)))
    pd_cfg = cfg["plandepth"]
    tasks = env.make_tasks(
        num_tasks=int(pd_cfg["num_tasks"]),
        depths=list(pd_cfg["depths"]),
        stochastic=bool(pd_cfg.get("stochastic", True)),
        delayed_deadend=bool(pd_cfg.get("delayed_deadend", True)),
    )
    trajectories = [env.rollout_expert(task, max_steps=pd_cfg.get("max_steps")) for task in tasks]
    write_jsonl(output_dir / "plandepth_trajectories.jsonl", trajectories_to_rows(trajectories))

    all_actions = [action for trajectory in trajectories for action in trajectory.actions]
    schema = build_schema_from_actions(all_actions, mode=cfg.get("zeta", {}).get("mode", "type_arg"))
    if "<unk>" not in schema["sketches"]:
        schema["sketches"].append("<unk>")
        schema["size"] = len(schema["sketches"])
    schema_path = Path(cfg.get("zeta", {}).get("schema_out", output_dir / "plandepth_schema.json"))
    write_json(schema_path, schema)
    mapper = SketchMapper(schema)

    st_cfg = cfg["soft_targets"]
    horizon = int(st_cfg["horizon"])
    rollouts = int(st_cfg["rollouts"])
    continuation_lookup = {}
    for task in tasks:
        for prefix_idx in range(min(task.depth, int(pd_cfg.get("max_steps", task.depth)))):
            continuation_lookup[(task.task_id, prefix_idx)] = env.teacher_continuations(task, prefix_idx, rollouts, horizon)

    tr_cfg = cfg["training"]
    examples = build_aligned_examples(
        trajectories,
        mapper,
        continuation_lookup,
        horizon=horizon,
        hidden_dim=int(tr_cfg["hidden_dim"]),
    )
    write_jsonl(output_dir / "aligned_examples.jsonl", examples_to_rows(examples))

    train_cfg = ToyForeActConfig(
        hidden_dim=int(tr_cfg["hidden_dim"]),
        num_sketches=len(schema["sketches"]),
        horizon=int(tr_cfg["horizon"]),
        lambda_future=float(tr_cfg["lambda_future"]),
        mu_consistency=float(tr_cfg["mu_consistency"]),
        eta_success=float(tr_cfg["eta_success"]),
        learning_rate=float(tr_cfg["learning_rate"]),
        steps=int(tr_cfg["steps"]),
        seed=int(cfg.get("seed", 0)),
    )
    model, train_metrics = train_toy_foreact(examples, list(schema["sketches"]), train_cfg)
    write_json(output_dir / "toy_training_metrics.json", train_metrics)
    foreact_eld_curve = smoke_eld_curve(model, examples)
    react_pilot_curve = {depth: max(0.0, value * 0.45) for depth, value in foreact_eld_curve.items()}
    write_mechanism_probe(output_dir / "smoke_eld_curve.csv", "foreact_smoke", foreact_eld_curve)
    fig2_rows = eld_curve_rows("react_sft_smoke", react_pilot_curve, figure_role="Fig2_pilot")
    fig5_rows = eld_curve_rows("foreact_smoke", foreact_eld_curve, figure_role="Fig5_recovery")
    eld_fieldnames = [
        "figure_role",
        "method",
        "depth",
        "future_signal",
        "threshold",
        "above_threshold",
        "effective_lookahead_depth",
        "axis_depth_min",
        "axis_depth_max",
        "axis_signal_min",
        "axis_signal_max",
        "note",
    ]
    write_rows(output_dir / "fig2_pilot_eld.csv", fig2_rows, eld_fieldnames)
    write_rows(output_dir / "fig5_recovery_eld.csv", fig5_rows, eld_fieldnames)
    write_eld_svg(output_dir / "fig2_pilot_eld.svg", fig2_rows, title="Fig.2 Pilot ELD Smoke Proxy")
    write_eld_svg(output_dir / "fig5_recovery_eld.svg", fig5_rows, title="Fig.5 ELD Recovery Smoke Proxy")
    write_forecast_entropy(output_dir / "forecast_entropy.csv", forecast_entropy_by_depth(model, examples))

    policies = {
        "react": ReActPolicy(),
        "react_sft": OracleSFTPolicy(),
        "token_mtp": TokenMTPPolicy(),
        "plan_and_act": PlanAndActPolicy(),
        "lats_webdreamer": SearchLookaheadPolicy(),
        "foreact_latent_rerank": LatentRerankPolicy(
            model,
            mapper,
            candidates=int(cfg.get("inference", {}).get("candidates", 4)),
        ),
    }
    metrics_by_method = {}
    traces_by_method = {}
    for method, policy in policies.items():
        traces = [policy.act_plan(task) for task in tasks]
        traces_by_method[method] = traces
        metrics = evaluate_plandepth(tasks, traces)
        metrics_by_method[method] = metrics
        write_json(output_dir / f"{method}_metrics.json", metrics)
        write_depth_curve(output_dir / f"{method}_depth_curve.csv", method, metrics["sr_by_depth"])
    eld_by_method = {
        "react": effective_lookahead_depth(react_pilot_curve),
        "react_sft": effective_lookahead_depth(react_pilot_curve),
        "token_mtp": max(1, effective_lookahead_depth(react_pilot_curve)),
        "plan_and_act": max(1, effective_lookahead_depth(react_pilot_curve)),
        "lats_webdreamer": horizon,
        "foreact_latent_rerank": effective_lookahead_depth(foreact_eld_curve),
    }
    boundary_rows = plandepth_boundary_rows(metrics_by_method, eld_by_method, trained_horizon=horizon)
    write_rows(
        output_dir / "plandepth_boundary_h5.csv",
        boundary_rows,
        [
            "method",
            "depth",
            "success_rate",
            "effective_lookahead_depth",
            "success_collapse_depth",
            "within_eld",
            "beyond_trained_horizon",
            "note",
        ],
    )
    write_rows(
        output_dir / "granularity_a_prime.csv",
        granularity_diagnostic_rows(),
        ["ablation", "zeta_mode", "expected_shape", "reason"],
    )
    write_efficiency(output_dir / "efficiency.csv", metrics_by_method)
    write_depth_svg(
        output_dir / "depth_curves.svg",
        {method: metrics["sr_by_depth"] for method, metrics in metrics_by_method.items()},
    )
    write_efficiency_svg(output_dir / "efficiency.svg", metrics_by_method)
    write_pareto_svg(output_dir / "success_overhead_pareto.svg", metrics_by_method)

    case_task = max(tasks, key=lambda task: task.depth)
    case_trace = next(trace for trace in traces_by_method["foreact_latent_rerank"] if trace.task_id == case_task.task_id)
    case_rows = plandepth_case_rows(case_task, case_trace.actions, model, mapper)
    write_json(
        output_dir / "plandepth_case_study.json",
        {
            "task_id": case_task.task_id,
            "depth": case_task.depth,
            "goal": case_task.goal,
            "rows": case_rows,
            "note": "Smoke artifact only; replace probabilities with real forecast heads for paper figures.",
        },
    )
    write_case_heatmap_svg(output_dir / "plandepth_case_study.svg", case_rows)

    manifest = {
        "config": config_path,
        "num_tasks": len(tasks),
        "num_examples": len(examples),
        "schema_size": len(schema["sketches"]),
        "effective_lookahead_depth": {
            "react_sft_smoke": effective_lookahead_depth(react_pilot_curve),
            "foreact_smoke": effective_lookahead_depth(foreact_eld_curve),
        },
        "v3_contract": {
            "north_star": "Plan like diffusion, act like autoregressive.",
            "mode_A_zero_overhead": True,
            "pcr_kl_direction": "KL(sg[q_{t+1}^{h-1}] || q_t^h)",
            "no_paper_numbers": "smoke artifacts only; not real AAAI results",
        },
        "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.iterdir()),
        "external_harness_status": external_harness_status(),
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
