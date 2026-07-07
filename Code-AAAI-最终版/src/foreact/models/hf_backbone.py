"""Optional Hugging Face backbone helpers for Qwen-style ForeAct training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from foreact.data.react_parser import locate_action_start_offsets


class HFUnavailable(RuntimeError):
    pass


def require_transformers():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise HFUnavailable("Install Code-AAAI with the [torch] extra and transformers to use HF backbones.") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ActionStateBatch:
    input_ids: object
    attention_mask: object
    action_token_indices: List[List[int]]


def char_offsets_to_token_indices(tokenizer, text: str, char_offsets: Sequence[int]) -> List[int]:
    """Map ReAct action-start character offsets to token indices."""

    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    token_indices: List[int] = []
    for char_offset in char_offsets:
        best_idx = 0
        for idx, (start, end) in enumerate(offsets):
            if start <= char_offset < max(end, start + 1):
                best_idx = idx
                break
            if start <= char_offset:
                best_idx = idx
        token_indices.append(best_idx)
    return token_indices


def build_action_state_batch(tokenizer, texts: Sequence[str]) -> ActionStateBatch:
    torch, _model_cls, _tok_cls = require_transformers()
    encoded = tokenizer(list(texts), padding=True, return_tensors="pt", add_special_tokens=False)
    action_token_indices = [
        char_offsets_to_token_indices(tokenizer, text, locate_action_start_offsets(text))
        for text in texts
    ]
    return ActionStateBatch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])),
        action_token_indices=action_token_indices,
    )


def extract_action_start_states(model, batch: ActionStateBatch):
    """Return final-layer hidden states at action-start token positions.

    The output is a list of tensors, one tensor per sequence, because each
    trajectory can contain a different number of decisions.
    """

    torch, _model_cls, _tok_cls = require_transformers()
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    final_hidden = outputs.hidden_states[-1]
    states = []
    for row, indices in enumerate(batch.action_token_indices):
        if not indices:
            states.append(final_hidden[row, :0, :])
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=final_hidden.device)
        states.append(final_hidden[row].index_select(0, index_tensor))
    return states


def load_causal_lm_and_tokenizer(model_name_or_path: str, **kwargs):
    """Load an HF causal LM and tokenizer without hard-coding Qwen paths."""

    _torch, model_cls, tokenizer_cls = require_transformers()
    tokenizer = tokenizer_cls.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
    model = model_cls.from_pretrained(model_name_or_path, trust_remote_code=True, **kwargs)
    return model, tokenizer

