# Full Qwen/HF Training Path

The smoke path uses deterministic feature vectors so it can run on a laptop. The real reference path uses `foreact.training.full_trainer` to extract action-start hidden states from a Hugging Face/Qwen backbone and train the backbone plus ForeAct heads.

## Reference Commands

```bash
PYTHONPATH=src python3 -m foreact.cli prepare-data --config configs/foreact_4b_milestone.yaml
PYTHONPATH=src python3 -m foreact.cli train-foreact --config configs/foreact_4b_milestone.yaml
PYTHONPATH=src python3 -m foreact.cli export-mode-a \
  --checkpoint outputs/foreact_4b_milestone/final \
  --out-dir outputs/foreact_4b_milestone/mode_a
```

The reference trainer supports `training.variant` values:

- `foreact`: NTP + LAP + PCR + success head.
- `react_sft`: NTP/SFT only, FLOPs-controlled by the same loop.
- `no_pcr`: NTP + LAP without PCR.
- `token_mtp`: NTP + token-level MTP heads for the critical action-vs-token baseline.
- `predict_past`: same head shape as ForeAct, but targets past/current sketches as an anti-confounding control.

## Trainer Contract

For verl/OpenRLHF scale-up, each batch should provide:

- `hidden_states`: `[batch, hidden_dim]`, final-layer state at the current action start.
- `next_hidden_states`: `[batch, hidden_dim]`, state at the next decision point for PCR.
- `soft_targets`: `[batch, H, |Z|]`, K-rollout empirical sketch distributions.
- `branch_weights`: `[batch, H]`, usually `1 - normalized_entropy(soft_target)`.
- `success_labels`: `[batch]`, trajectory-level success/failure labels.
- `ntp_loss`: standard next-token/action SFT loss from the backbone.

The reference implementation already adds this value in `foreact.training.full_trainer`; scale-up trainers should preserve the same calculation:

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
