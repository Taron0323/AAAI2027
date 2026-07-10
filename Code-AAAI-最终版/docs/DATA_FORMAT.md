# Data Format

ForeAct uses a small set of stable records so generated data can be shared across training backends.

## Trajectory JSONL

Written by the smoke pipeline as `plandepth_trajectories.jsonl`.

```json
{
  "task_id": "pd-0000",
  "goal": "Reach ...",
  "success": true,
  "dead_end": false,
  "metadata": {"depth": "4", "benchmark": "plandepth"},
  "actions": [{"text": "inspect target=node", "tool": "inspect", "args": {"target": "node"}}],
  "observations": [{"text": "nominal: completed inspect", "state": {"step": "1"}, "branch_id": "nominal"}]
}
```

## Aligned Example JSONL

Written as `aligned_examples.jsonl`.

Each row corresponds to one decision point and contains:

- `current_action`: the demonstrated action at that decision.
- `current_sketch`: `zeta(current_action)`.
- `future_targets`: one soft distribution for each depth `h`.
- `hidden_features`: smoke-time deterministic features; full training replaces this with Qwen action-start hidden states.
- `next_hidden_features`: smoke-time stand-in for the adjacent decision state used by PCR.
- `success`: trajectory-level outcome for the success head.

## Sketch Schema

The main setting is `type_arg`, equivalent to operation type times primary argument slot. The required ablation spectrum is:

- `token`
- `fsp_summary`
- `type`
- `type_arg`
- `vq_mock` as a smoke stand-in for VQ-512

Production VQ-512 should train a codebook over action strings and write the same schema contract.

## Tool Schema Input

`foreact schema-from-tools` accepts a small JSON schema:

```json
{
  "tools": [
    {"name": "inspect", "parameters": {"target": "string"}}
  ]
}
```

Example:

```bash
PYTHONPATH=src python3 -m foreact.cli schema-from-tools \
  --tool-schema examples/tool_schemas/plandepth_tools.json \
  --out outputs/smoke/schema_from_tools.json
```
