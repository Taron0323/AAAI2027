#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

python3 - <<'PY'
from tests.test_zeta import test_schema_mapper_type_arg
from tests.test_soft_targets import test_aggregate_future_targets_soft_distribution
from tests.test_losses import test_forward_kl_direction_penalizes_wrong_prediction
from tests.test_plandepth import test_plandepth_distinguishes_myopic_and_expert
from tests.test_ablation import test_expand_ablation_jobs
from tests.test_torch_optional import test_torch_optional_import_contract
from tests.test_visualize import test_visualization_exports
from tests.test_schema_tools import test_schema_from_tool_schema
from tests.test_provider_cache import test_cached_provider_roundtrip
from tests.test_policies import test_smoke_baseline_policies_run
from tests.test_hf_optional import test_hf_optional_import_contract
from tests.test_dataset_status import (
    test_dataset_status_marks_missing_paths,
    test_dataset_status_resolves_relative_paths,
)
from tests.test_asset_status import test_asset_status_distinguishes_missing_code_and_dataset
from tests.test_v3_contract import (
    test_asset_audit_rejects_model_weights,
    test_effective_lookahead_depth_contract,
    test_plandepth_boundary_and_granularity_rows,
    test_registry_contains_v3_hypotheses_and_guardrails,
)
from tests.test_real_readiness import (
    test_benchmark_runner_dry_run,
    test_materialize_ablation_jobs,
    test_official_dataset_loaders_minimal,
    test_prepare_real_data_with_mock_teacher,
    test_teacher_baseline_efficiency_and_control_entrypoints,
    test_vq_schema_encodes_actions,
)
import tempfile
from pathlib import Path
import os
from tests.test_cli_smoke import test_cli_smoke_runs

test_schema_mapper_type_arg()
test_aggregate_future_targets_soft_distribution()
test_forward_kl_direction_penalizes_wrong_prediction()
with tempfile.TemporaryDirectory() as d:
    test_official_dataset_loaders_minimal(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_prepare_real_data_with_mock_teacher(Path(d))
test_vq_schema_encodes_actions()
with tempfile.TemporaryDirectory() as d:
    test_benchmark_runner_dry_run(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_materialize_ablation_jobs(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_teacher_baseline_efficiency_and_control_entrypoints(Path(d))
test_effective_lookahead_depth_contract()
test_plandepth_boundary_and_granularity_rows()
test_registry_contains_v3_hypotheses_and_guardrails()
with tempfile.TemporaryDirectory() as d:
    test_asset_audit_rejects_model_weights(Path(d))
test_plandepth_distinguishes_myopic_and_expert()
test_expand_ablation_jobs()
test_torch_optional_import_contract()
test_schema_from_tool_schema()
test_smoke_baseline_policies_run()
test_hf_optional_import_contract()
test_dataset_status_marks_missing_paths()
with tempfile.TemporaryDirectory() as d:
    from types import SimpleNamespace

    old_cwd = os.getcwd()
    try:
        test_dataset_status_resolves_relative_paths(Path(d), SimpleNamespace(chdir=os.chdir))
    finally:
        os.chdir(old_cwd)
with tempfile.TemporaryDirectory() as d:
    test_asset_status_distinguishes_missing_code_and_dataset(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_visualization_exports(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_cached_provider_roundtrip(Path(d))
with tempfile.TemporaryDirectory() as d:
    test_cli_smoke_runs(Path(d))
print("standard-library test runner: 25 tests passed")
PY

python3 -m foreact.cli smoke --config configs/smoke.yaml
python3 -m foreact.cli registry --out outputs/smoke/registry.json
python3 -m foreact.cli harness-status --out outputs/smoke/harness_status.json
python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
python3 -m foreact.cli asset-audit --out outputs/smoke/asset_audit.json
python3 -m foreact.cli ablation-jobs --config configs/ablation_matrix.yaml --out outputs/smoke/ablation_jobs.json
python3 -m foreact.cli materialize-ablation \
  --matrix configs/ablation_matrix.yaml \
  --base-config configs/foreact_4b_milestone.yaml \
  --out-dir outputs/smoke/ablation_real_jobs
python3 -m foreact.cli schema-from-tools --tool-schema examples/tool_schemas/plandepth_tools.json --out outputs/smoke/schema_from_tools.json
python3 -m foreact.cli run-baseline --config configs/baseline_lats.yaml
python3 -m foreact.cli run-baseline --config configs/baseline_plan_and_act.yaml
python3 -m foreact.cli run-efficiency --config configs/efficiency_vllm.yaml
python3 -m foreact.cli build-control-task --config configs/control_task.yaml
python3 -m foreact.cli readiness --out outputs/readiness.json
