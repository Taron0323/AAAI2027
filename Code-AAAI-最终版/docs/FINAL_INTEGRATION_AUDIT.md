# Final Integration Audit

`Code-AAAI-最终版` merges the tested v3 smoke repository (`Code-AAAI`) with the real pre-experiment entrypoints from `Code-AAAI-V2版`.

## Source Of Truth

- Highest-priority product document: `ForeAct_论文整体框架_v3模板填充版.md`
- Draft consistency reference: `AuthorKit27/ForeAct_AAAI2027.pdf`
- Non-negotiable constraints preserved:
  - action-level vs token-level comparison
  - token-MTP baseline
  - A' sketch granularity spectrum
  - B/B' PCR and branch-aware weighting
  - G predict-past and FLOPs-matched controls
  - Mode A zero-overhead ReAct-shaped inference
  - PCR direction `KL(sg[q_{t+1}^{h-1}] || q_t^h)`
  - no fabricated paper results

## Merged From `Code-AAAI`

- v3 smoke pipeline with Fig.2/Fig.5-compatible ELD artifacts
- H1-H5 registry, C1-C3 alignment, design choices, Remark-1 degenerate corners
- asset-status and asset-audit with model-weight exclusion
- PlanDepth smoke benchmark and built-in baselines
- docs and tests for v3 contract compliance

## Merged From `Code-AAAI-V2版`

- official benchmark dataset loaders and `prepare-data`
- reference HF/Qwen `train-foreact`
- real hidden-state `probe-eld`
- baseline/harness command planners
- vLLM efficiency planning
- teacher trajectory generation
- control-task generation
- readiness reporting
- VQ-style zeta schema
- materialized ablation job configs

## External Asset Strategy

Official code and datasets are not duplicated. This final directory uses symlinks:

- `third_party/benchmarks`
- `third_party/baselines`
- `third_party/auxiliary`
- `datasets`

The target assets are the already-downloaded local assets under `Code-AAAI/`. Run:

```bash
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
PYTHONPATH=src python3 -m foreact.cli asset-audit --out outputs/smoke/asset_audit.json
PYTHONPATH=src python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
```

Expected local state: benchmark code 5/5, baseline code 6/6, auxiliary code 3/3, nested archive code 2/2, datasets 4/4, and no model weights under code assets.

## Verification

```bash
bash scripts/check_repo.sh
PYTHONPATH=src /Users/futaoran/Desktop/AAAI2027/Code-AAAI/.venv-assets/bin/python -m pytest -q
```

The check script writes smoke artifacts and readiness files under `outputs/`. These outputs are ignored and are not paper results.
