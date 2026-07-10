# Full Qwen/HF Training Path

The smoke path uses deterministic feature vectors so it can run on a laptop. Full experiments replace those features with action-start hidden states from a Qwen3 backbone.

## Trainer Contract

For each batch, the full trainer should provide:

- `hidden_states`: `[batch, hidden_dim]`, final-layer state at the current action start.
- `next_hidden_states`: `[batch, hidden_dim]`, state at the next decision point for PCR.
- `soft_targets`: `[batch, H, |Z|]`, K-rollout empirical sketch distributions.
- `branch_weights`: `[batch, H]`, usually `1 - normalized_entropy(soft_target)`.
- `success_labels`: `[batch]`, trajectory-level success/failure labels.
- `ntp_loss`: standard next-token/action SFT loss from the backbone.

Then add:

```python
from foreact.models.foreact_torch import (
    ForeActTorchConfig,
    build_foreact_heads,
    foreact_auxiliary_loss,
)

config = ForeActTorchConfig(hidden_dim=hidden_dim, sketch_size=sketch_size, horizon=8)
heads = build_foreact_heads(config)
aux = foreact_auxiliary_loss(
    heads,
    hidden_states,
    next_hidden_states,
    soft_targets,
    branch_weights,
    success_labels,
    config,
    step=global_step,
    total_steps=total_steps,
)
loss = ntp_loss + aux["loss"]
```

## HF/Qwen Action-State Extraction

`foreact.models.hf_backbone` provides the optional Hugging Face adapter layer:

- `load_causal_lm_and_tokenizer(model_name_or_path)`
- `build_action_state_batch(tokenizer, react_texts)`
- `extract_action_start_states(model, batch)`

These helpers map ReAct action-start character offsets to tokenizer positions and return final-layer hidden states for LAP/PCR heads. They require `torch` and `transformers`; smoke tests verify that missing dependencies fail with setup guidance rather than cryptic import errors.

## Required Invariants

- Do not insert ForeAct special tokens into prompts.
- Locate action starts through the tokenizer-aligned version of `foreact.data.react_parser.locate_action_start_offsets`.
- Keep forecast heads training-only for mode A.
- Use `foreact.models.foreact_torch.pcr_forward_kl`, which implements `KL(sg[q_{t+1}] || q_t)`.
- Use the token-level MTP baseline and A' sketch spectrum with the same data and FLOPs before claiming action-granularity gains.
