"""Effective Lookahead Depth probe over real backbone hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from foreact.io import read_json, read_jsonl, write_json
from foreact.models.foreact_torch import require_torch


@dataclass(frozen=True)
class ELDProbeConfig:
    model_name_or_path: str
    aligned_examples: Path
    schema_path: Path
    output_dir: Path
    horizon: int = 8
    max_examples: int = 512
    train_steps: int = 200
    learning_rate: float = 1e-2
    seed: int = 13
    max_length: int = 2048


def run_eld_probe_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    probe_cfg = _parse_config(config)
    return run_eld_probe(probe_cfg)


def run_eld_probe(cfg: ELDProbeConfig) -> Dict[str, object]:
    torch, _nn = require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install transformers to run ELD probes.") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model.eval()
    schema = read_json(cfg.schema_path)
    sketch_to_id = {sketch: idx for idx, sketch in enumerate(schema.get("sketches", []))}
    rows = list(read_jsonl(cfg.aligned_examples))[: cfg.max_examples]
    if not rows:
        raise RuntimeError(f"No aligned examples found: {cfg.aligned_examples}")

    hidden_states = []
    labels_by_depth: List[List[int]] = [[] for _ in range(cfg.horizon)]
    with torch.no_grad():
        for row in rows:
            text = _example_text(row)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg.max_length)
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden_states.append(outputs.hidden_states[-1][0, -1, :].detach().float().cpu())
            targets = _target_ids(row, sketch_to_id, cfg.horizon)
            for depth, label in enumerate(targets):
                labels_by_depth[depth].append(label)
    states = torch.stack(hidden_states)
    split = max(1, int(0.8 * len(states)))
    results = []
    for depth in range(cfg.horizon):
        labels = torch.tensor(labels_by_depth[depth], dtype=torch.long)
        result = _fit_probe(states, labels, len(sketch_to_id), split, cfg, random_labels=False)
        control = _fit_probe(states, labels, len(sketch_to_id), split, cfg, random_labels=True)
        result["control_accuracy"] = control["accuracy"]
        result["depth"] = depth + 1
        results.append(result)
    eld = _estimate_eld(results)
    manifest = {
        "model_name_or_path": cfg.model_name_or_path,
        "aligned_examples": str(cfg.aligned_examples),
        "schema_path": str(cfg.schema_path),
        "num_examples": len(rows),
        "horizon": cfg.horizon,
        "eld": eld,
        "depth_results": results,
        "notes": "accuracy is held-out linear-probe accuracy; intervention_delta is accuracy drop after projecting out the learned probe direction.",
    }
    write_json(cfg.output_dir / "eld_probe.json", manifest)
    _write_curve_csv(cfg.output_dir / "eld_curve.csv", results)
    return manifest


def _fit_probe(states, labels, num_classes: int, split: int, cfg: ELDProbeConfig, random_labels: bool) -> Dict[str, float]:
    torch, nn = require_torch()
    y = labels.clone()
    if random_labels:
        y = y[torch.randperm(len(y))]
    probe = nn.Linear(states.shape[-1], num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=cfg.learning_rate)
    train_x = states[:split]
    train_y = y[:split]
    test_x = states[split:] if split < len(states) else states[:split]
    test_y = y[split:] if split < len(states) else y[:split]
    for _step in range(cfg.train_steps):
        logits = probe(train_x)
        loss = torch.nn.functional.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = probe(test_x)
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == test_y).float().mean())
        ablated = _project_out(test_x, probe.weight.detach())
        ablated_preds = probe(ablated).argmax(dim=-1)
        ablated_accuracy = float((ablated_preds == test_y).float().mean())
    return {
        "accuracy": accuracy,
        "ablated_accuracy": ablated_accuracy,
        "intervention_delta": max(0.0, accuracy - ablated_accuracy),
        "train_size": float(len(train_x)),
        "test_size": float(len(test_x)),
    }


def _project_out(states, weight):
    basis = weight - weight.mean(dim=0, keepdim=True)
    q, _r = np.linalg.qr(basis.cpu().numpy().T)
    import torch

    q_tensor = torch.tensor(q, dtype=states.dtype, device=states.device)
    return states - (states @ q_tensor) @ q_tensor.T


def _target_ids(row: Mapping[str, object], sketch_to_id: Mapping[str, int], horizon: int) -> List[int]:
    unk = sketch_to_id.get("<unk>", 0)
    labels = [unk for _ in range(horizon)]
    targets = row.get("future_targets", [])
    if isinstance(targets, list):
        for item in targets[:horizon]:
            if not isinstance(item, Mapping):
                continue
            depth = int(item.get("depth", 1)) - 1
            dist = item.get("distribution", {})
            if 0 <= depth < horizon and isinstance(dist, Mapping) and dist:
                sketch = max(dist.items(), key=lambda kv: float(kv[1]))[0]
                labels[depth] = sketch_to_id.get(str(sketch), unk)
    return labels


def _example_text(row: Mapping[str, object]) -> str:
    current = row.get("current_action", {})
    action_text = current.get("text", "") if isinstance(current, Mapping) else ""
    return f"Goal: {row.get('goal', '')}\nAction: {action_text}"


def _estimate_eld(results: Sequence[Mapping[str, float]]) -> int:
    threshold = 0.05
    eld = 0
    for item in results:
        if float(item.get("intervention_delta", 0.0)) >= threshold and float(item.get("accuracy", 0.0)) > float(item.get("control_accuracy", 0.0)) + threshold:
            eld = int(item["depth"])
    return eld


def _write_curve_csv(path: Path, results: Sequence[Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("depth,accuracy,control_accuracy,ablated_accuracy,intervention_delta\n")
        for item in results:
            handle.write(
                f"{item['depth']},{item['accuracy']},{item['control_accuracy']},{item['ablated_accuracy']},{item['intervention_delta']}\n"
            )


def _parse_config(config: Mapping[str, object]) -> ELDProbeConfig:
    model_cfg = _mapping(config.get("model"))
    paths = _mapping(config.get("paths"))
    probe_cfg = _mapping(config.get("eld_probe"))
    output_dir = Path(str(config.get("output_dir", probe_cfg.get("output_dir", "outputs/eld_probe"))))
    return ELDProbeConfig(
        model_name_or_path=str(model_cfg.get("backbone", config.get("model_name_or_path", "Qwen/Qwen3-4B"))),
        aligned_examples=Path(str(paths.get("aligned_examples", config.get("aligned_examples", output_dir / "aligned_examples.jsonl")))),
        schema_path=Path(str(paths.get("schema", config.get("schema", output_dir / "schema.json")))),
        output_dir=output_dir,
        horizon=int(probe_cfg.get("horizon", _mapping(config.get("training")).get("horizon", 8))),
        max_examples=int(probe_cfg.get("max_examples", 512)),
        train_steps=int(probe_cfg.get("train_steps", 200)),
        learning_rate=float(probe_cfg.get("learning_rate", 1e-2)),
        seed=int(config.get("seed", probe_cfg.get("seed", 13))),
        max_length=int(model_cfg.get("context_length", probe_cfg.get("max_length", 2048))),
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
