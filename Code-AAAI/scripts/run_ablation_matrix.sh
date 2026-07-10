#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "This script records the intended ablation entrypoint."
echo "Full runs require Qwen checkpoints, benchmark data, and GPU training."
python3 -m foreact.cli registry --out outputs/smoke/registry.json
python3 -m foreact.cli smoke --config configs/smoke.yaml

