"""External dataset path checks with explicit non-vendoring semantics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping


def expand_env_path(value: str) -> str:
    return os.path.expandvars(value)


def dataset_status(config: Mapping[str, object]) -> Dict[str, dict]:
    project_root = Path.cwd()
    statuses: Dict[str, dict] = {}
    for name, raw_spec in config.items():
        if not isinstance(raw_spec, Mapping):
            continue
        required = raw_spec.get("required_paths", {})
        paths = {}
        missing = []
        if isinstance(required, Mapping):
            for key, value in required.items():
                expanded = expand_env_path(str(value))
                path = Path(expanded)
                if not path.is_absolute():
                    path = project_root / path
                exists = "$" not in expanded and path.exists()
                paths[str(key)] = {"path": str(path), "exists": exists}
                if not exists:
                    missing.append(str(key))
        statuses[name] = {
            "available": not missing,
            "paths": paths,
            "missing": missing,
            "setup": str(raw_spec.get("setup", "Configure official dataset path.")),
            "note": "Official datasets/baselines are not vendored or faked by this repository.",
        }
    return statuses
