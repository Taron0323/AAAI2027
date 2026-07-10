import json
from pathlib import Path

from foreact.data.benchmarks import load_appworld, load_plandepth, load_swe_bench_verified, load_tau2
from foreact.data.prepare import _benchmark_specs, _build_provider, prepare_real_data
from foreact.data.react_parser import locate_action_start_offsets
from foreact.evaluation.runners import build_benchmark_run_spec, run_benchmark_from_config
from foreact.baselines.runners import run_baseline_from_config
from foreact.evaluation.efficiency import run_efficiency_from_config
from foreact.analysis.control_tasks import build_control_task_from_config
from foreact.teachers.trajectory_generation import generate_teacher_trajectories
from foreact.teachers.providers import CachedAPIProvider
from foreact.training.ablation import materialize_ablation_jobs
from foreact.types import Action
from foreact.zeta import SketchMapper
from foreact.zeta.vq import build_vq_schema


def test_official_dataset_loaders_minimal(tmp_path: Path):
    plandepth = load_plandepth(limit=3)
    assert plandepth.benchmark == "plandepth"
    assert len(plandepth.trajectories) == 3
    assert plandepth.trajectories[0].metadata["benchmark"] == "plandepth"

    tau = tmp_path / "tau" / "airline"
    tau.mkdir(parents=True)
    (tau / "split_tasks.json").write_text(json.dumps({"train": ["1"]}), encoding="utf-8")
    (tau / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "user_scenario": {"instructions": {"reason_for_call": "cancel flight"}},
                    "evaluation_criteria": {"actions": [{"name": "refund", "args": {"target": "ticket"}}]},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_tau2(tau.parent, split="train").trajectories[0].actions[0].tool == "refund"

    app = tmp_path / "appworld" / "data"
    (app / "datasets").mkdir(parents=True)
    (app / "tasks" / "task1" / "ground_truth").mkdir(parents=True)
    (app / "datasets" / "train.txt").write_text("task1\n", encoding="utf-8")
    (app / "tasks" / "task1" / "specs.json").write_text(json.dumps({"instruction": "buy item"}), encoding="utf-8")
    (app / "tasks" / "task1" / "ground_truth" / "required_apps.json").write_text(json.dumps(["amazon"]), encoding="utf-8")
    (app / "tasks" / "task1" / "ground_truth" / "required_apis.json").write_text(json.dumps(["checkout"]), encoding="utf-8")
    assert load_appworld(app.parent).trajectories[0].actions[0].tool == "amazon.checkout"

    swe = tmp_path / "swe-bench" / "verified"
    swe.mkdir(parents=True)
    (swe / "test.jsonl").write_text(
        json.dumps({"instance_id": "x", "repo": "r", "base_commit": "b", "problem_statement": "fix", "patch": "diff"}) + "\n",
        encoding="utf-8",
    )
    assert load_swe_bench_verified(swe.parent).eval_rows[0]["instance_id"] == "x"


def test_train_domains_map_to_distinct_benchmark_specs():
    specs = _benchmark_specs({"train_domains": ["plandepth", "tau2_airline", "tau2_retail", "appworld", "tau2_airline"]})
    assert specs == [
        {"name": "plandepth", "root": ""},
        {"name": "tau2_bench", "root": "third_party/benchmarks/tau2-bench/data/tau2/domains/airline"},
        {"name": "tau2_bench", "root": "third_party/benchmarks/tau2-bench/data/tau2/domains/retail"},
        {"name": "appworld", "root": "datasets/appworld"},
    ]


def test_teacher_provider_infers_api_from_model_id():
    provider = _build_provider({"continuation_model": "deepseek-v4-flash", "api_key_env": {"deepseek": "DEEPSEEK_TEST_KEY"}})
    assert isinstance(provider, CachedAPIProvider)
    assert provider.api_key_env == "DEEPSEEK_TEST_KEY"


def test_prepare_real_data_with_mock_teacher(tmp_path: Path):
    tau = tmp_path / "tau" / "airline"
    tau.mkdir(parents=True)
    (tau / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "user_scenario": {"instructions": {"reason_for_call": "cancel"}},
                    "evaluation_criteria": {"actions": [{"name": "refund", "args": {"target": "ticket"}}]},
                }
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    manifest = prepare_real_data(
        {
            "output_dir": str(out),
            "data": {"benchmarks": [{"name": "tau2_bench", "root": str(tau.parent)}]},
            "teacher": {"provider": "mock"},
            "soft_targets": {"horizon": 2, "rollouts": 2},
            "training": {"hidden_dim": 4},
        }
    )
    assert manifest["num_aligned_examples"] == 1
    assert (out / "aligned_examples.jsonl").exists()
    assert (out / "schema.json").exists()
    row = json.loads((out / "aligned_examples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["prefix_text"].startswith("Goal:")
    assert "Action 1:" in row["prefix_text"]
    assert "Action 2:" in row["next_prefix_text"]


def test_action_start_offsets_include_empty_next_decision():
    assert locate_action_start_offsets("Goal: x\nAction 1:") == [17]


def test_vq_schema_encodes_actions():
    actions = [Action(text=f"tool target={idx}", tool="tool", args={"target": str(idx)}) for idx in range(4)]
    schema = build_vq_schema(actions, codebook_size=2, iterations=2)
    mapper = SketchMapper(schema)
    assert mapper.encode(actions[0]).startswith("vq::")


def test_benchmark_runner_dry_run(tmp_path: Path):
    spec = build_benchmark_run_spec({"benchmark": "swe_bench_verified", "predictions_path": "gold"})
    assert "swebench.harness.run_evaluation" in spec.command
    result = run_benchmark_from_config({"benchmark": "swe_bench_verified", "out": str(tmp_path / "run.json")})
    assert result["status"] in {"ready", "ready_source", "not_ready"}
    assert (tmp_path / "run.json").exists()


def test_materialize_ablation_jobs(tmp_path: Path):
    manifest = materialize_ablation_jobs(
        {"main_controls": [{"key": "react_sft"}], "sketch_granularity": ["type_arg"]},
        {"training": {"horizon": 2}, "soft_targets": {"horizon": 2}, "zeta": {"mode": "type_arg"}},
        tmp_path,
    )
    assert manifest["num_jobs"] == 2
    assert (tmp_path / "react_sft" / "config.yaml").exists()


def test_teacher_baseline_efficiency_and_control_entrypoints(tmp_path: Path):
    eval_rows = tmp_path / "eval_rows.jsonl"
    eval_rows.write_text(json.dumps({"benchmark": "tau2_bench", "task_id": "t1", "goal": "do task"}) + "\n", encoding="utf-8")
    teacher_manifest = generate_teacher_trajectories(
        {"input": str(eval_rows), "out": str(tmp_path / "teacher.jsonl"), "provider": "mock", "horizon": 2}
    )
    assert teacher_manifest["num_trajectories"] == 1

    baseline = run_baseline_from_config({"baseline": "react", "out": str(tmp_path / "baseline.json")})
    assert baseline["status"] in {"ready_source", "not_ready"}

    efficiency = run_efficiency_from_config({"mode": "plan", "out": str(tmp_path / "eff.json"), "model": "demo-model"})
    assert "vllm.entrypoints.openai.api_server" in efficiency["command"]

    aligned = tmp_path / "aligned.jsonl"
    aligned.write_text(
        json.dumps({"task_id": "t1", "prefix_index": 0, "current_sketch": "move::target"}) + "\n",
        encoding="utf-8",
    )
    control = build_control_task_from_config({"aligned_examples": str(aligned), "out": str(tmp_path / "control.jsonl")})
    assert control["num_rows"] == 1
