#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

python3 -m pytest -q
python3 -m foreact.cli smoke --config configs/smoke.yaml
python3 -m foreact.cli registry --out outputs/smoke/registry.json
python3 -m foreact.cli harness-status --out outputs/smoke/harness_status.json
python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
python3 -m foreact.cli ablation-jobs --config configs/ablation_matrix.yaml --out outputs/smoke/ablation_jobs.json
python3 -m foreact.cli materialize-ablation \
  --matrix configs/ablation_matrix.yaml \
  --base-config configs/foreact_4b_milestone.yaml \
  --out-dir outputs/smoke/ablation_real_jobs
python3 -m foreact.cli schema-from-tools \
  --tool-schema examples/tool_schemas/plandepth_tools.json \
  --out outputs/smoke/schema_from_tools.json
python3 -m foreact.cli run-baseline --config configs/baseline_lats.yaml
python3 -m foreact.cli run-baseline --config configs/baseline_plan_and_act.yaml
python3 -m foreact.cli run-efficiency --config configs/efficiency_vllm.yaml
python3 -m foreact.cli build-control-task --config configs/control_task.yaml
python3 -m foreact.cli readiness --out outputs/readiness.json
