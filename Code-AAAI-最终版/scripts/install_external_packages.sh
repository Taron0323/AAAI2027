#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pip_install() {
  python3 -m pip install ${PIP_EXTRA_ARGS:-} "$@"
}

pip_install -e .
pip_install -e third_party/benchmarks/tau2-bench
pip_install -e third_party/benchmarks/SWE-bench
pip_install -e third_party/benchmarks/mini-swe-agent

echo "Skipping AppWorld in the main environment; use scripts/create_appworld_env.sh to avoid SQLAlchemy conflicts."

if [[ -f third_party/auxiliary/OpenRLHF/pyproject.toml || -f third_party/auxiliary/OpenRLHF/setup.py ]]; then
  pip_install -e third_party/auxiliary/OpenRLHF || true
fi

PYTHONPATH=src python3 -m foreact.cli harness-status
