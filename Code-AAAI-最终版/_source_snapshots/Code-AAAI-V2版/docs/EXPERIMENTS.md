# ForeAct Experiment Map

This document maps the paper requirements to executable repository modules.

## Required Main Path

1. **PlanDepth mock data:** `foreact.environments.plandepth`
2. **Action sketches:** `foreact.zeta.schema`
3. **K-rollout soft targets:** `foreact.data.soft_targets`
4. **LAP/PCR smoke training:** `foreact.training.toy_trainer`
5. **Official data preparation:** `foreact.data.benchmarks`, `foreact.data.prepare`
6. **Reference HF/Qwen training:** `foreact.training.full_trainer`
7. **Real ELD probe:** `foreact.analysis.eld_probe`
8. **Official harness command runner:** `foreact.evaluation.runners`
5. **Mode A/B inference:** `foreact.inference.policies`
6. **Task/system metrics:** `foreact.evaluation.metrics`
7. **Plot exports:** `foreact.analysis.exports`
8. **SVG visualizations and case studies:** `foreact.analysis.visualize`, `foreact.analysis.case_study`

Run:

```bash
python3 -m foreact.cli smoke --config configs/smoke.yaml
```

Real pre-experiment path:

```bash
python3 -m foreact.cli prepare-data --config configs/foreact_4b_milestone.yaml
python3 -m foreact.cli train-foreact --config configs/foreact_4b_milestone.yaml
python3 -m foreact.cli probe-eld --config configs/foreact_4b_milestone.yaml
python3 -m foreact.cli materialize-ablation \
  --matrix configs/ablation_matrix.yaml \
  --base-config configs/foreact_4b_milestone.yaml \
  --out-dir outputs/ablation_jobs
```

## Full-Scale Training

The smoke trainer is not a replacement for Qwen3 finetuning. It verifies the loss plumbing without requiring GPUs. `foreact.training.full_trainer` is the single-node reference finetuner; use it to validate data, targets, and checkpoint artifacts before scaling to verl/OpenRLHF.

For the full paper run, `prepare-data` writes aligned examples and `train-foreact` replaces deterministic smoke features with action-start hidden states from the Qwen3 backbone. The same artifacts can be consumed by distributed trainers:

- forecast heads from `foreact.models.foreact_torch`
- soft targets from cached DeepSeek-V4-Flash continuations
- successful and failed trajectories from DeepSeek-V4-Pro plus environment validators
- external harness adapters from `foreact.evaluation.adapters`

The production trainer must preserve these invariants:

- no special prompt tokens for ForeAct forecasts
- mode A discards forecast and success heads
- PCR uses `KL(sg[q_{t+1}^{h-1}] || q_t^h)`, not the reverse direction
- branch-aware weights downweight high-entropy K-rollout targets
- headline numbers must be written only after real measured runs

## Baselines and Ablations

The required registry is available with:

```bash
python3 -m foreact.cli registry --out outputs/smoke/registry.json
```

Critical main-paper controls:

- ReAct-SFT
- AR + token-level MTP
- A/A' sketch granularity spectrum
- B no-PCR
- G predict-past and FLOPs-matched SFT controls
