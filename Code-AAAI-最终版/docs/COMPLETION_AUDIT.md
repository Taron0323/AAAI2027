# Completion Audit

This document maps the requested ForeAct repository requirements to current evidence.

## Git Management

- Local repository root: `/Users/futaoran/Desktop/AAAI2027`
- Project code path: `Code-AAAI-最终版/`
- Ignore policy: `.gitignore` excludes secrets, local envs, model weights, datasets, large outputs, logs, scratch, and LaTeX build products.
- Generated smoke artifacts are under `Code-AAAI-最终版/outputs/` and intentionally ignored.
- Official datasets are local-only under `Code-AAAI-最终版/datasets/`.
- Third-party benchmark, baseline, and auxiliary code assets are local-only under `Code-AAAI-最终版/third_party/`.

## Requirement Evidence

1. **Repository structure**
   - Evidence: `src/foreact/{data,zeta,environments,training,models,inference,evaluation,baselines,analysis,teachers}`, `configs/`, `scripts/`, `docs/`, `tests/`.

2. **Phase 1 data construction**
   - Trajectory records: `foreact.types`, `foreact.environments.plandepth`
   - ReAct parser and action offsets: `foreact.data.react_parser`
   - Zeta schema: `foreact.zeta.schema`
   - Tool schema example: `examples/tool_schemas/plandepth_tools.json`
   - K-rollout soft targets: `foreact.data.soft_targets`
   - Teacher abstraction/cache shell: `foreact.teachers.providers`
   - Official benchmark data conversion: `foreact.data.benchmarks`, `foreact.data.prepare`

3. **Phase 2 LAP**
   - Smoke trainer: `foreact.training.toy_trainer`
   - PyTorch head/loss contract: `foreact.models.foreact_torch`
   - Optional HF/Qwen action-state extraction: `foreact.models.hf_backbone`
   - Reference HF/Qwen trainer: `foreact.training.full_trainer`
   - Full training notes: `docs/FULL_TRAINING.md`

4. **Phase 3 PCR**
   - Numpy forward KL: `foreact.training.losses.forward_kl_stopgrad`
   - PyTorch forward KL: `foreact.models.foreact_torch.pcr_forward_kl`
   - Entropy/branch weights: `foreact.data.soft_targets`, `foreact.analysis.mechanism`

5. **Phase 4 inference**
   - Mode A/ReAct: `foreact.inference.policies.ReActPolicy`
   - Mode B latent rerank: `foreact.inference.policies.LatentRerankPolicy`
   - HF-backed rerank adapter: `foreact.inference.hf_rerank`
   - Efficiency metrics: `foreact.evaluation.metrics`, `foreact.analysis.exports`

6. **PlanDepth**
   - Environment and teacher continuations: `foreact.environments.plandepth`
   - SR(d): `foreact.evaluation.metrics.evaluate_plandepth`
   - Case study: `foreact.analysis.case_study`, `foreact.analysis.visualize`
   - ELD probe/control-task entrypoints: `foreact.analysis.eld_probe`, `foreact.analysis.control_tasks`

7. **Baselines and ablations**
   - Registry: `foreact.baselines.registry`
   - Official baseline command planners: `foreact.baselines.runners`
   - Runnable PlanDepth smoke policies: `foreact.inference.policies`
   - Ablation matrix: `configs/ablation_matrix.yaml`
   - Job expansion/materialization: `foreact.training.ablation`

8. **External harness adapters**
   - tau2/AppWorld/SWE-Gym/SWE-bench Verified graceful skips: `foreact.evaluation.adapters`
   - External dataset path checks: `foreact.evaluation.datasets`, `configs/datasets.yaml`
   - External benchmark/baseline code manifest: `third_party/assets.yaml`
   - Official harness command planners: `foreact.evaluation.runners`
   - Readiness report: `foreact.evaluation.readiness`
   - Asset availability command: `foreact.evaluation.assets`, CLI `asset-status`
   - Asset weight/group audit command: CLI `asset-audit`
   - Fetch scripts: `scripts/bootstrap_external_assets.sh`, `scripts/download_datasets.sh`
   - Fetch attempt evidence: `docs/EXTERNAL_ASSET_FETCH_LOG.md`
   - Current local status: benchmark code 5/5, baseline code 6/6, auxiliary code 3/3, nested archive code 2/2, datasets 4/4 available.
   - Current asset audit: no model-weight-like files under code/third-party assets.
   - Setup guidance: `docs/BENCHMARK_SETUP.md`
   - Three-layer implementation distinction: `docs/DATASETS_AND_BASELINES.md`

9. **Config, CLI, docs, tests**
   - CLI: `foreact.cli`
   - Smoke config: `configs/smoke.yaml`
   - Milestone config: `configs/foreact_4b_milestone.yaml`
   - Tests: `tests/`
   - Check script: `scripts/check_repo.sh`

10. **No fabricated results**
   - Smoke outputs are labeled as smoke artifacts.
   - Full benchmark and large-scale training commands are documented as requiring real data/API/GPU resources.

## V3 Template Alignment

- North-star contract and Mode-A zero-overhead invariant: `foreact.cli.run_smoke` manifest, `README.md`, `docs/FULL_TRAINING.md`.
- ELD operator and Fig.2/Fig.5 smoke artifact contract: `foreact.analysis.mechanism`, `foreact.analysis.visualize`, `outputs/smoke/fig2_pilot_eld.*`, `outputs/smoke/fig5_recovery_eld.*`.
- H1-H5, C1-C3, design choices, and Remark-1 degenerate corners: `foreact.baselines.registry`, exported by `python3 -m foreact.cli registry`.
- A'/B'/C'/D/E/E'/F/G job expansion: `configs/ablation_matrix.yaml`, `foreact.training.ablation`.
- H5 PlanDepth boundary and A' granularity diagnostics: `outputs/smoke/plandepth_boundary_h5.csv`, `outputs/smoke/granularity_a_prime.csv`.
- External asset restriction to benchmark/harness code, dataset path checks, baseline/auxiliary code, and no model weights: `third_party/assets.yaml`, `foreact.evaluation.assets.asset_audit`.

## Verification Commands

```bash
cd /Users/futaoran/Desktop/AAAI2027/Code-AAAI-最终版
bash scripts/check_repo.sh
PYTHONPATH=src python3 -m foreact.cli smoke --config configs/smoke.yaml
PYTHONPATH=src python3 -m foreact.cli harness-status --out outputs/smoke/harness_status.json
PYTHONPATH=src python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
PYTHONPATH=src python3 -m foreact.cli asset-audit --out outputs/smoke/asset_audit.json
```

## Known External Dependencies

The repository is complete as an executable scaffold and smoke harness. Real paper numbers still require:

- Qwen3 checkpoints and GPU/cluster training.
- DeepSeek/OpenAI API keys or self-hosted teacher models.
- Docker/OpenHands/Modal or equivalent execution infrastructure for real SWE-Gym/SWE-bench evaluation runs.
- Real ELD intervention probes over backbone hidden states.

These are intentionally not mocked as real results.
