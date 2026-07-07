"""Hewitt-Liang-style control-task readiness and label generation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping

from foreact.io import read_jsonl, write_json, write_jsonl


def build_control_task_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    aligned = Path(str(config.get("aligned_examples", "outputs/foreact_4b_milestone/aligned_examples.jsonl")))
    out = Path(str(config.get("out", "outputs/control_tasks/random_labels.jsonl")))
    seed = int(config.get("seed", 13))
    rows = list(read_jsonl(aligned))
    rng = random.Random(seed)
    sketches = sorted({str(row.get("current_sketch", "<unk>")) for row in rows}) or ["<unk>"]
    control_rows = []
    for row in rows:
        control_rows.append(
            {
                "task_id": row.get("task_id"),
                "prefix_index": row.get("prefix_index"),
                "true_sketch": row.get("current_sketch"),
                "control_sketch": rng.choice(sketches),
            }
        )
    write_jsonl(out, control_rows)
    source = Path(str(config.get("control_tasks_source", "third_party/auxiliary/control-tasks")))
    manifest = {
        "aligned_examples": str(aligned),
        "out": str(out),
        "num_rows": len(control_rows),
        "num_labels": len(sketches),
        "control_tasks_source": str(source),
        "source_available": source.exists(),
        "note": "Use this random-label file as the Hewitt-Liang control target for ELD probes.",
    }
    write_json(out.with_suffix(".manifest.json"), manifest)
    return manifest
