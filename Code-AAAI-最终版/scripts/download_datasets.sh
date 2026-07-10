#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"

PY312="${FOREACT_ASSET_PYTHON:-/Users/futaoran/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
if [[ ! -x "$PY312" ]]; then
  PY312="python3"
fi

if [[ ! -x ".venv-assets/bin/python" ]]; then
  "$PY312" -m venv .venv-assets
fi

pip_install() {
  .venv-assets/bin/python -m pip install ${PIP_EXTRA_ARGS:-} "$@"
}

pip_install --upgrade pip setuptools wheel
pip_install datasets huggingface_hub

mkdir -p datasets/appworld datasets/swe-gym datasets/swe-bench

echo "tau2-bench: official domain data is included in third_party/benchmarks/tau2-bench/data/tau2/domains"

echo "Downloading SWE-bench Verified and SWE-Gym datasets from Hugging Face..."
.venv-assets/bin/python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json


def save_dataset(repo: str, root: Path, marker_name: str = "DATASET_READY.json") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    dsd = load_dataset(repo)
    dsd.save_to_disk(str(root / "hf_dataset"))
    manifest = {"repo": repo, "splits": {}}
    for split, ds in dsd.items():
        manifest["splits"][split] = {"rows": len(ds), "columns": ds.column_names}
        sample_path = root / f"{split}.sample.jsonl"
        with sample_path.open("w", encoding="utf-8") as handle:
            for row in ds.select(range(min(20, len(ds)))):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (root / marker_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


swe_verified = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
swe_root = Path("datasets/swe-bench/verified")
swe_root.mkdir(parents=True, exist_ok=True)
swe_verified.save_to_disk(str(swe_root / "hf_dataset"))
with (swe_root / "test.jsonl").open("w", encoding="utf-8") as handle:
    for row in swe_verified:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
(swe_root / "DATASET_READY.json").write_text(
    json.dumps(
        {
            "name": "SWE-bench/SWE-bench_Verified",
            "split": "test",
            "rows": len(swe_verified),
            "columns": swe_verified.column_names,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

repos = {
    "swe_gym": "SWE-Gym/SWE-Gym",
    "swe_gym_lite": "SWE-Gym/SWE-Gym-Lite",
    "openhands_sft_trajectories": "SWE-Gym/OpenHands-SFT-Trajectories",
    "openhands_verifier_trajectories": "SWE-Gym/OpenHands-Verifier-Trajectories",
}
root = Path("datasets/swe-gym")
manifest = {}
for key, repo in repos.items():
    manifest[key] = save_dataset(repo, root / key)
(root / "DATASET_READY.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "Preparing AppWorld official data..."
pip_install -e third_party/benchmarks/appworld

.venv-assets/bin/python - <<'PY'
import json
import pathlib
import urllib.request

repo = "StonyBrookNLP/appworld"
root = pathlib.Path("third_party/benchmarks/appworld")
pointers = []
for path in root.rglob("*.bundle"):
    text = path.read_text("utf-8", errors="ignore")
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        continue
    oid = None
    size = None
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1]
        if line.startswith("size "):
            size = int(line.split()[1])
    if oid and size:
        pointers.append((path, oid, size))

if pointers:
    body = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size} for _, oid, size in pointers],
        }
    ).encode()
    req = urllib.request.Request(
        f"https://github.com/{repo}.git/info/lfs/objects/batch",
        data=body,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "ForeAct-AAAI2027-local-repro/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        objects = {obj["oid"]: obj for obj in json.load(response).get("objects", [])}
    for path, oid, size in pointers:
        action = objects[oid]["actions"]["download"]
        headers = action.get("header", {})
        headers.setdefault("User-Agent", "ForeAct-AAAI2027-local-repro/1.0")
        tmp = path.with_suffix(path.suffix + ".tmp")
        with urllib.request.urlopen(
            urllib.request.Request(action["href"], headers=headers),
            timeout=180,
        ) as src:
            data = src.read()
        if len(data) != size:
            raise RuntimeError(f"Git LFS size mismatch for {path}: {len(data)} != {size}")
        tmp.write_bytes(data)
        tmp.replace(path)
PY

(
  cd third_party/benchmarks/appworld
  APPWORLD_ROOT="$(pwd)/../../../datasets/appworld" ../../../.venv-assets/bin/appworld install --repo
  APPWORLD_ROOT="$(pwd)/../../../datasets/appworld" ../../../.venv-assets/bin/appworld download data --root "$(pwd)/../../../datasets/appworld"
)

python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
echo "wrote outputs/smoke/dataset_status.json and outputs/smoke/asset_status.json"
