"""CSV exports for paper plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_depth_curve(path: str | Path, method: str, sr_by_depth: Mapping[str, float]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "depth", "success_rate"])
        writer.writeheader()
        for depth, success_rate in sorted(sr_by_depth.items(), key=lambda item: int(item[0])):
            writer.writerow({"method": method, "depth": depth, "success_rate": success_rate})


def write_efficiency(path: str | Path, metrics_by_method: Mapping[str, Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "success_rate", "tokens_per_task", "extra_forward_per_task"],
        )
        writer.writeheader()
        for method, metrics in metrics_by_method.items():
            writer.writerow(
                {
                    "method": method,
                    "success_rate": metrics.get("success_rate", 0.0),
                    "tokens_per_task": metrics.get("tokens_per_task", 0.0),
                    "extra_forward_per_task": metrics.get("extra_forward_per_task", 0.0),
                }
            )


def write_mechanism_probe(path: str | Path, method: str, quality_drop_by_depth: Mapping[int, float]) -> None:
    """Export an ELD-style intervention curve table shape."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "depth", "quality_drop"])
        writer.writeheader()
        for depth, quality_drop in sorted(quality_drop_by_depth.items()):
            writer.writerow({"method": method, "depth": depth, "quality_drop": quality_drop})


def write_forecast_entropy(path: str | Path, entropy_by_depth: Sequence[float]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["depth", "entropy"])
        writer.writeheader()
        for idx, entropy in enumerate(entropy_by_depth, start=1):
            writer.writerow({"depth": idx, "entropy": entropy})


def write_rows(path: str | Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
