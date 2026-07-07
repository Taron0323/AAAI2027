#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

out_dir="${1:-outputs/ablation_jobs}"
base_config="${2:-configs/foreact_4b_milestone.yaml}"
matrix="${3:-configs/ablation_matrix.yaml}"

python3 -m foreact.cli materialize-ablation \
  --matrix "$matrix" \
  --base-config "$base_config" \
  --out-dir "$out_dir"

echo "wrote $out_dir/ablation_manifest.json"
echo "Launch individual jobs with the command arrays in the manifest after model/API/GPU resources are configured."
