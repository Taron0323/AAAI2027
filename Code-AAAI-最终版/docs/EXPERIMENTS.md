# ForeAct Experiment Map

This document maps the paper requirements to executable repository modules.

## Required Main Path

1. **PlanDepth mock data:** `foreact.environments.plandepth`
2. **Action sketches:** `foreact.zeta.schema`
3. **K-rollout soft targets:** `foreact.data.soft_targets`
4. **LAP/PCR smoke training:** `foreact.training.toy_trainer`
5. **Mode A/B inference:** `foreact.inference.policies`
6. **Task/system metrics:** `foreact.evaluation.metrics`
7. **Plot exports:** `foreact.analysis.exports`
8. **SVG visualizations and case studies:** `foreact.analysis.visualize`, `foreact.analysis.case_study`

Run:

```bash
python3 -m foreact.cli smoke --config configs/smoke.yaml
```

## Full-Scale Training

The smoke trainer is not a replacement for Qwen3 finetuning. It verifies the loss plumbing without requiring GPUs.

For the full paper run, replace deterministic smoke features with action-start hidden states from the Qwen3 backbone and use:

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

## V3 Hypotheses And Figure Contracts

The registry command exports the v3 paper map:

- `H1 effectiveness`: ForeAct should beat same-data same-FLOPs ReAct-SFT and token-MTP on long-horizon tasks.
- `H2 mechanism_eld`: Fig.2 and Fig.5 use the same Effective Lookahead Depth contract.
- `H3 robustness`: PCR should reduce churn/dead-end behavior under stochastic branching.
- `H4 efficiency`: Mode A is ReAct-identical; Mode B reports bounded rerank overhead.
- `H5 boundaries`: gains should vanish or shrink for shallow tasks, deterministic settings, and bad `zeta` granularity.

Smoke outputs corresponding to those contracts:

- `fig2_pilot_eld.csv` / `fig2_pilot_eld.svg`
- `fig5_recovery_eld.csv` / `fig5_recovery_eld.svg`
- `plandepth_boundary_h5.csv`
- `granularity_a_prime.csv`
- `success_overhead_pareto.svg`

These files are contract checks only. Full Fig.2/Fig.5 values require real intervention probes over Qwen hidden states.

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
- B' no branch-aware weighting
- G predict-past and FLOPs-matched SFT controls

Appendix/job-expansion controls include C/C' rerank variants, D horizon sweep, E rollout sweep, E' second-teacher subset, and F curriculum/lambda/mu sweeps.
