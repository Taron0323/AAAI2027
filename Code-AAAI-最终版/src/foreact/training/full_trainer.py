"""Single-node HF/Qwen training entrypoint for pre-experiment readiness.

This is deliberately simple: it is the reference implementation that validates
the ForeAct objectives on real hidden states before scaling the same data
contract to verl/OpenRLHF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from foreact.data.benchmarks import rows_to_trajectories
from foreact.data.react_parser import locate_action_start_offsets
from foreact.io import read_json, read_jsonl, write_json
from foreact.models.foreact_torch import (
    ForeActTorchConfig,
    TorchUnavailable,
    build_foreact_heads,
    foreact_auxiliary_loss,
    require_torch,
)


@dataclass(frozen=True)
class TrainForeActConfig:
    model_name_or_path: str
    train_data: Path
    schema_path: Path
    output_dir: Path
    variant: str = "foreact"
    max_steps: int = 100
    batch_size: int = 1
    learning_rate: float = 2e-5
    max_length: int = 2048
    seed: int = 13
    save_every: int = 0
    horizon: int = 8
    lambda_future: float = 0.3
    mu_consistency: float = 0.1
    eta_success: float = 0.05
    train_backbone: bool = True


def train_foreact_from_config(config: Mapping[str, object]) -> Dict[str, object]:
    cfg = _parse_config(config)
    return train_foreact(cfg)


def train_foreact(cfg: TrainForeActConfig) -> Dict[str, object]:
    torch, _nn = require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise TorchUnavailable("Install transformers to use train-foreact.") from exc

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    model.train()
    schema = read_json(cfg.schema_path)
    sketch_to_id = {sketch: idx for idx, sketch in enumerate(schema.get("sketches", []))}
    examples = list(read_jsonl(cfg.train_data))
    if not examples:
        raise RuntimeError(f"No training rows found in {cfg.train_data}")

    hidden_dim = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    if hidden_dim <= 0:
        raise RuntimeError("Could not infer backbone hidden size from model config.")
    heads = build_foreact_heads(
        ForeActTorchConfig(
            hidden_dim=hidden_dim,
            sketch_size=len(sketch_to_id),
            horizon=cfg.horizon,
            lambda_future=cfg.lambda_future,
            mu_consistency=cfg.mu_consistency,
            eta_success=cfg.eta_success,
        )
    )
    token_mtp_heads = None
    if cfg.variant == "token_mtp":
        token_mtp_heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, int(getattr(model.config, "vocab_size"))) for _ in range(cfg.horizon)]
        )
    params = list(heads.parameters()) + (list(token_mtp_heads.parameters()) if token_mtp_heads is not None else []) + (list(model.parameters()) if cfg.train_backbone else [])
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    metrics: List[Dict[str, float]] = []

    for step in range(cfg.max_steps):
        batch_rows = [examples[(step * cfg.batch_size + offset) % len(examples)] for offset in range(cfg.batch_size)]
        batch = _collate_batch(batch_rows, tokenizer, sketch_to_id, cfg)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
            use_cache=False,
        )
        ntp_loss = outputs.loss
        hidden = _select_indices(outputs.hidden_states[-1], batch["current_indices"])
        with torch.no_grad():
            next_outputs = model(
                input_ids=batch["next_input_ids"],
                attention_mask=batch["next_attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        next_hidden = _select_indices(next_outputs.hidden_states[-1], batch["next_indices"]).detach()
        aux_cfg = ForeActTorchConfig(
            hidden_dim=hidden_dim,
            sketch_size=len(sketch_to_id),
            horizon=cfg.horizon,
            lambda_future=0.0 if cfg.variant in {"react_sft", "token_mtp"} else cfg.lambda_future,
            mu_consistency=0.0 if cfg.variant in {"react_sft", "no_pcr", "token_mtp", "predict_past"} else cfg.mu_consistency,
            eta_success=cfg.eta_success,
        )
        soft_targets = batch["soft_targets"]
        if cfg.variant == "predict_past":
            soft_targets = batch["past_targets"]
        aux = foreact_auxiliary_loss(
            heads,
            hidden,
            next_hidden,
            soft_targets,
            batch["branch_weights"],
            batch["success_labels"],
            aux_cfg,
            step=step,
            total_steps=max(1, cfg.max_steps),
        )
        token_mtp = (
            _token_mtp_loss(token_mtp_heads, outputs.hidden_states[-1], batch["input_ids"], batch["attention_mask"])
            if token_mtp_heads is not None
            else ntp_loss * 0.0
        )
        loss = ntp_loss + aux["loss"] + (cfg.lambda_future * token_mtp if token_mtp_heads is not None else 0.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.append(
            {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "ntp_loss": float(ntp_loss.detach().cpu()),
                "future_loss": float(aux["future_loss"].cpu()),
                "consistency_loss": float(aux["consistency_loss"].cpu()),
                "success_loss": float(aux["success_loss"].cpu()),
                "token_mtp_loss": float(token_mtp.detach().cpu()),
                "active_horizon": float(aux["active_horizon"]),
            }
        )
        if cfg.save_every and (step + 1) % cfg.save_every == 0:
            _save_checkpoint(cfg.output_dir / f"step-{step + 1}", model, tokenizer, heads, cfg, metrics)

    _save_checkpoint(cfg.output_dir / "final", model, tokenizer, heads, cfg, metrics)
    write_json(cfg.output_dir / "training_metrics.json", metrics)
    manifest = {
        "variant": cfg.variant,
        "model_name_or_path": cfg.model_name_or_path,
        "train_data": str(cfg.train_data),
        "schema_path": str(cfg.schema_path),
        "output_dir": str(cfg.output_dir),
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "horizon": cfg.horizon,
        "train_backbone": cfg.train_backbone,
        "final_loss": metrics[-1]["loss"] if metrics else None,
    }
    write_json(cfg.output_dir / "train_manifest.json", manifest)
    return manifest


def export_mode_a_checkpoint(checkpoint_dir: str | Path, output_dir: str | Path) -> Dict[str, object]:
    """Copy only the backbone/tokenizer files for zero-overhead mode A."""

    source = Path(checkpoint_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source.iterdir():
        if path.name in {"foreact_heads.pt", "train_config.json", "training_metrics.json"}:
            continue
        if path.is_file():
            data = path.read_bytes()
            (target / path.name).write_bytes(data)
            copied.append(path.name)
    manifest = {"source": str(source), "output_dir": str(target), "copied_files": copied, "mode": "A_zero_overhead"}
    write_json(target / "mode_a_manifest.json", manifest)
    return manifest


def _parse_config(config: Mapping[str, object]) -> TrainForeActConfig:
    model_cfg = _mapping(config.get("model"))
    train_cfg = _mapping(config.get("training"))
    paths = _mapping(config.get("paths"))
    trainer = str(train_cfg.get("trainer", "hf_reference"))
    if trainer not in {"hf_reference", "single_node_hf"}:
        raise RuntimeError(
            f"train-foreact supports trainer=hf_reference only; got {trainer!r}. "
            "Materialize scale-out jobs separately for verl/OpenRLHF."
        )
    output_dir = Path(str(config.get("output_dir", train_cfg.get("output_dir", "outputs/train_foreact"))))
    train_data = Path(str(paths.get("aligned_examples", config.get("train_data", output_dir / "aligned_examples.jsonl"))))
    schema_path = Path(str(paths.get("schema", config.get("schema", output_dir / "schema.json"))))
    return TrainForeActConfig(
        model_name_or_path=str(model_cfg.get("backbone", config.get("model_name_or_path", "Qwen/Qwen3-4B"))),
        train_data=train_data,
        schema_path=schema_path,
        output_dir=output_dir,
        variant=str(train_cfg.get("variant", config.get("variant", "foreact"))),
        max_steps=int(train_cfg.get("steps", config.get("max_steps", 100))),
        batch_size=int(train_cfg.get("batch_size", config.get("batch_size", 1))),
        learning_rate=float(train_cfg.get("learning_rate", config.get("learning_rate", 2e-5))),
        max_length=int(model_cfg.get("context_length", config.get("max_length", 2048))),
        seed=int(config.get("seed", train_cfg.get("seed", 13))),
        save_every=int(train_cfg.get("save_every", 0)),
        horizon=int(train_cfg.get("horizon", 8)),
        lambda_future=float(train_cfg.get("lambda_future", 0.3)),
        mu_consistency=float(train_cfg.get("mu_consistency", 0.1)),
        eta_success=float(train_cfg.get("eta_success", 0.05)),
        train_backbone=bool(train_cfg.get("train_backbone", True)),
    )


def _collate_batch(rows: Sequence[Mapping[str, object]], tokenizer, sketch_to_id: Mapping[str, int], cfg: TrainForeActConfig):
    torch, _nn = require_torch()
    texts = [_example_to_react_text(row) for row in rows]
    next_texts = [_next_example_to_react_text(row) for row in rows]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    next_encoded = tokenizer(
        next_texts,
        padding=True,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    current_indices = []
    next_indices = []
    for text in texts:
        offsets = locate_action_start_offsets(text)
        token_indices = _char_offsets_to_token_indices(tokenizer, text, offsets)
        current_indices.append(token_indices[-1] if token_indices else int(encoded["attention_mask"][len(current_indices)].sum() - 1))
    for text in next_texts:
        offsets = locate_action_start_offsets(text)
        token_indices = _char_offsets_to_token_indices(tokenizer, text, offsets)
        next_indices.append(token_indices[-1] if token_indices else int(next_encoded["attention_mask"][len(next_indices)].sum() - 1))
    soft_targets = []
    branch_weights = []
    past_targets = []
    for row in rows:
        probs, weights = _targets_to_tensor(row.get("future_targets", []), sketch_to_id, cfg.horizon)
        soft_targets.append(probs)
        branch_weights.append(weights)
        past_targets.append(_past_target(row, sketch_to_id, cfg.horizon))
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "next_input_ids": next_encoded["input_ids"],
        "next_attention_mask": next_encoded["attention_mask"],
        "labels": labels,
        "current_indices": torch.tensor(current_indices, dtype=torch.long),
        "next_indices": torch.tensor(next_indices, dtype=torch.long),
        "soft_targets": torch.tensor(np.stack(soft_targets), dtype=torch.float32),
        "past_targets": torch.tensor(np.stack(past_targets), dtype=torch.float32),
        "branch_weights": torch.tensor(np.stack(branch_weights), dtype=torch.float32),
        "success_labels": torch.tensor([1.0 if row.get("success") else 0.0 for row in rows], dtype=torch.float32),
    }


def _example_to_react_text(row: Mapping[str, object]) -> str:
    if row.get("prefix_text"):
        return str(row["prefix_text"])
    lines = [f"Goal: {row.get('goal', '')}"]
    current = row.get("current_action", {})
    if isinstance(current, Mapping):
        lines.append(f"Action: {current.get('text', '')}")
    return "\n".join(lines)


def _next_example_to_react_text(row: Mapping[str, object]) -> str:
    if row.get("next_prefix_text"):
        return str(row["next_prefix_text"])
    return _example_to_react_text(row)


def _char_offsets_to_token_indices(tokenizer, text: str, char_offsets: Sequence[int]) -> List[int]:
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    indices = []
    for char_offset in char_offsets:
        chosen = 0
        for idx, (start, end) in enumerate(offsets):
            if start <= char_offset < max(end, start + 1):
                chosen = idx
                break
            if start <= char_offset:
                chosen = idx
        indices.append(chosen)
    return indices


def _targets_to_tensor(raw_targets: object, sketch_to_id: Mapping[str, int], horizon: int):
    probs = np.zeros((horizon, len(sketch_to_id)), dtype=np.float32)
    weights = np.zeros((horizon,), dtype=np.float32)
    unk = sketch_to_id.get("<unk>", 0)
    for depth in range(horizon):
        probs[depth, unk] = 1.0
    if isinstance(raw_targets, list):
        for item in raw_targets[:horizon]:
            if not isinstance(item, Mapping):
                continue
            depth = int(item.get("depth", 1)) - 1
            if depth < 0 or depth >= horizon:
                continue
            probs[depth, :] = 0.0
            dist = item.get("distribution", {})
            has_distribution = False
            if isinstance(dist, Mapping):
                for sketch, value in dist.items():
                    numeric = float(value)
                    if numeric > 0:
                        has_distribution = True
                    probs[depth, sketch_to_id.get(str(sketch), unk)] += numeric
            total = probs[depth].sum()
            if total > 0 and has_distribution:
                probs[depth] = probs[depth] / total
                weights[depth] = float(item.get("branch_weight", 1.0))
            else:
                probs[depth, :] = 0.0
                probs[depth, unk] = 1.0
                weights[depth] = 0.0
    return probs, weights


def _past_target(row: Mapping[str, object], sketch_to_id: Mapping[str, int], horizon: int):
    probs = np.zeros((horizon, len(sketch_to_id)), dtype=np.float32)
    sketch = str(row.get("current_sketch", "<unk>"))
    idx = sketch_to_id.get(sketch, sketch_to_id.get("<unk>", 0))
    probs[:, idx] = 1.0
    return probs


def _select_indices(hidden_states, indices):
    torch, _nn = require_torch()
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, indices.to(hidden_states.device), :]


def _token_mtp_loss(token_heads, hidden_states, input_ids, attention_mask):
    torch, _nn = require_torch()
    if token_heads is None:
        return hidden_states.sum() * 0.0
    losses = []
    for depth, head in enumerate(token_heads, start=1):
        if input_ids.shape[1] <= depth:
            continue
        logits = head(hidden_states[:, :-depth, :])
        labels = input_ids[:, depth:]
        mask = attention_mask[:, depth:].bool()
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_mask = mask.reshape(-1)
        if flat_mask.any():
            losses.append(torch.nn.functional.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask]))
    if not losses:
        return hidden_states.sum() * 0.0
    return torch.stack(losses).mean()


def _save_checkpoint(path: Path, model, tokenizer, heads, cfg: TrainForeActConfig, metrics: Sequence[Mapping[str, float]]) -> None:
    torch, _nn = require_torch()
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(heads.state_dict(), path / "foreact_heads.pt")
    write_json(path / "train_config.json", _cfg_to_json(cfg))
    write_json(path / "training_metrics.json", list(metrics))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _cfg_to_json(cfg: TrainForeActConfig) -> Dict[str, object]:
    data = dict(cfg.__dict__)
    for key in ("train_data", "schema_path", "output_dir"):
        data[key] = str(data[key])
    return data
