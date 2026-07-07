#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

venv="${1:-.venv-appworld}"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install "pydantic>=2.12,<3" "SQLAlchemy>=2.0" "sqlmodel>=0.0.19"
"$venv/bin/python" -m pip install -e third_party/benchmarks/appworld

APPWORLD_ROOT="$(pwd)/datasets/appworld" "$venv/bin/python" - <<'PY'
import appworld
print("appworld import ok", getattr(appworld, "__version__", "unknown"))
PY

echo "Use this interpreter for AppWorld official runs: $venv/bin/python"
