#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

config="${1:-configs/foreact_4b_milestone.yaml}"

python3 -m foreact.cli prepare-data --config "$config"
python3 -m foreact.cli train-foreact --config "$config"
python3 -m foreact.cli probe-eld --config "$config"

echo "Full pre-experiment pipeline finished for $config"
