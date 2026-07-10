"""HF-backed latent reranking for ForeAct mode B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from foreact.io import read_json
from foreact.models.foreact_torch import ForeActTorchConfig, build_foreact_heads, forecast_logits, require_torch
from foreact.types import Action
from foreact.zeta import SketchMapper


@dataclass(frozen=True)
class RerankResult:
    selected: Action
    scores: List[Dict[str, object]]
    extra_forward_count: int


class HFLatentReranker:
    def __init__(self, checkpoint_dir: str | Path, schema_path: str | Path, head_path: str | Path | None = None) -> None:
        torch, _nn = require_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        self.model.eval()
        self.schema = read_json(schema_path)
        self.mapper = SketchMapper(self.schema)
        hidden_dim = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", 0)))
        self.heads = build_foreact_heads(
            ForeActTorchConfig(hidden_dim=hidden_dim, sketch_size=len(self.schema.get("sketches", [])), horizon=8)
        )
        state_path = Path(head_path) if head_path else self.checkpoint_dir / "foreact_heads.pt"
        if state_path.exists():
            self.heads.load_state_dict(torch.load(state_path, map_location="cpu"))
        else:
            raise FileNotFoundError(f"missing ForeAct head checkpoint: {state_path}")
        self.heads.eval()

    def rerank(self, goal: str, history: Sequence[Action], candidates: Sequence[Action]) -> RerankResult:
        torch, _nn = require_torch()
        sketch_names = list(self.schema.get("sketches", []))
        scores = []
        with torch.no_grad():
            for candidate in candidates:
                text = _render(goal, list(history) + [candidate])
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
                outputs = self.model(**encoded, output_hidden_states=True, use_cache=True)
                hidden = outputs.hidden_states[-1][:, -1, :]
                logits = forecast_logits(self.heads, hidden)
                probs = torch.softmax(logits[0], dim=-1)
                success = torch.sigmoid(self.heads["success"](hidden)).item()
                dead_mass = _dead_end_mass(probs, sketch_names)
                score = float(success - dead_mass)
                scores.append(
                    {
                        "action": {"text": candidate.text, "tool": candidate.tool, "args": dict(candidate.args)},
                        "success_score": float(success),
                        "dead_end_mass": float(dead_mass),
                        "score": score,
                    }
                )
        best_idx = max(range(len(scores)), key=lambda idx: float(scores[idx]["score"]))
        return RerankResult(selected=candidates[best_idx], scores=scores, extra_forward_count=len(candidates))


def _dead_end_mass(probs, sketch_names: Sequence[str]) -> float:
    mass = 0.0
    for idx, sketch in enumerate(sketch_names):
        if sketch.startswith("commit") or "delete" in sketch or "cancel" in sketch:
            mass += float(probs[:, idx].mean())
    return mass


def _render(goal: str, actions: Sequence[Action]) -> str:
    lines = [f"Goal: {goal}"]
    for idx, action in enumerate(actions):
        lines.append(f"Action {idx + 1}: {action.text}")
    return "\n".join(lines)
