"""Status checks for third-party benchmark, baseline, and dataset assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping


ASSET_GROUPS = (
    "benchmark_code",
    "baseline_code",
    "auxiliary_code",
    "nested_archive_code",
    "datasets",
)


def _project_root(start: str | Path | None = None) -> Path:
    if start is not None:
        return Path(start).resolve()
    return Path(__file__).resolve().parents[3]


def _first_existing_path(root: Path, spec: Mapping[str, object]) -> Path:
    env_var = spec.get("env_var")
    if env_var:
        value = os.environ.get(str(env_var), "")
        if value:
            return Path(value).expanduser()
    return root / str(spec.get("local_path", ""))


def _marker_status(base: Path, markers: Iterable[object]) -> Dict[str, bool]:
    return {str(marker): (base / str(marker)).exists() for marker in markers}


def asset_status(manifest: Mapping[str, object], project_root: str | Path | None = None) -> Dict[str, dict]:
    """Return machine-readable local availability for every configured asset."""

    root = _project_root(project_root)
    result: Dict[str, dict] = {}
    for group in ASSET_GROUPS:
        group_spec = manifest.get(group, {})
        if not isinstance(group_spec, Mapping):
            continue
        result[group] = {}
        for name, raw_spec in group_spec.items():
            if not isinstance(raw_spec, Mapping):
                continue
            local = _first_existing_path(root, raw_spec)
            markers = _marker_status(local, raw_spec.get("required_markers", []) or [])
            exists = local.exists()
            markers_ok = all(markers.values()) if markers else exists
            entry: MutableMapping[str, object] = {
                "title": str(raw_spec.get("title", name)),
                "available": bool(exists and markers_ok),
                "path": str(local),
                "exists": exists,
                "markers": markers,
                "url": str(raw_spec.get("url", "")),
                "archive_url": str(raw_spec.get("archive_url", "")),
                "install_command": str(raw_spec.get("install_command", "")),
                "download_command": str(raw_spec.get("download_command", "")),
                "notes": str(raw_spec.get("notes", "")),
            }
            if raw_spec.get("env_var"):
                env_name = str(raw_spec["env_var"])
                entry["env_var"] = env_name
                entry["env_value"] = os.environ.get(env_name, "")
            result[group][str(name)] = dict(entry)
    return result


def asset_summary(status: Mapping[str, Mapping[str, Mapping[str, object]]]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for group, assets in status.items():
        total = len(assets)
        available = sum(1 for item in assets.values() if item.get("available"))
        summary[group] = {
            "available": available,
            "missing": total - available,
            "total": total,
        }
    return summary
