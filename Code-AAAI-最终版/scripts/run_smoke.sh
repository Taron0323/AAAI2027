#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m foreact.cli smoke --config configs/smoke.yaml
