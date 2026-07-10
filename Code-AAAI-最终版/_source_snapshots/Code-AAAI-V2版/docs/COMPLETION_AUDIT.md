# Completion Audit

This document maps the requested ForeAct repository requirements to current evidence.

## Git Management

- Local repository root: `/Users/futaoran/Desktop/AAAI2027`
- Project code path: `Code-AAAI/`
- Ignore policy: `.gitignore` excludes secrets, local envs, model weights, datasets, large outputs, logs, scratch, and LaTeX build products.
- Generated smoke artifacts are under `Code-AAAI/outputs/` and intentionally ignored.
- Official datasets under `Code-AAAI/datasets/` are intentionally ignored.
- Third-party code is tracked through git submodules under `Code-AAAI/third_party/`.

## Requirement Evidence

1. **Repository structure**
   - Evidence: `src/foreact/{data,zeta,environments,training,models,inference,evaluation,baselines,analysis,teachers}`, `configs/`, `scripts/`, `docs/`, `tests/`.

2. **Phase 1 data construction**
   - Trajectory records: `foreact.types`, `foreact.environments.plandepth`
   - Official benchmark loaders: `foreact.data.benchmarks`
   - Real data preparation pipeline: `foreact.data.prepare`, CLI `prepare-data`
   - ReAct parser and action offsets: `foreact.data.react_parser`
   - Zeta schema: `foreact.zeta.schema`
   - Tool schema example: `examples/tool_schemas/plandepth_tools.json`
   - K-rollout soft targets: `foreact.data.soft_targets`
   - Teacher abstraction/cache shell: `foreact.teachers.providers`

3. **Phase 2 LAP**
   - Smoke trainer: `foreact.training.toy_trainer`
   - PyTorch head/loss contract: `foreact.models.foreact_torch`
   - Optional HF/Qwen action-state extraction: `foreact.models.hf_backbone`
   - Reference HF/Qwen trainer: `foreact.training.full_trainer`, CLI `train-foreact`
   - Full training notes: `docs/FULL_TRAINING.md`

4. **Phase 3 PCR**
   - Numpy forward KL: `foreact.training.losses.forward_kl_stopgrad`
   - PyTorch forward KL: `foreact.models.foreact_torch.pcr_forward_kl`
   - Entropy/branch weights: `foreact.data.soft_targets`, `foreact.analysis.mechanism`

5. **Phase 4 inference**
   - Mode A/ReAct: `foreact.inference.policies.ReActPolicy`
   - Mode B latent rerank: `foreact.inference.policies.LatentRerankPolicy`
   - HF latent reranker: `foreact.inference.hf_rerank`
   - Mode-A head-discard export: CLI `export-mode-a`
   - Efficiency metrics: `foreact.evaluation.metrics`, `foreact.analysis.exports`

6. **PlanDepth**
   - Environment and teacher continuations: `foreact.environments.plandepth`
   - SR(d): `foreact.evaluation.metrics.evaluate_plandepth`
   - Case study: `foreact.analysis.case_study`, `foreact.analysis.visualize`

7. **Baselines and ablations**
   - Registry: `foreact.baselines.registry`
   - Runnable PlanDepth smoke policies: `foreact.inference.policies`
   - Ablation matrix: `configs/ablation_matrix.yaml`
   - Job expansion/materialization: `foreact.training.ablation`, CLI `materialize-ablation`
   - Real training variants: `foreact`, `react_sft`, `no_pcr`, `token_mtp`, `predict_past`

8. **External harness adapters**
   - tau2/AppWorld/SWE-Gym/SWE-bench Verified graceful skips: `foreact.evaluation.adapters`
   - Official harness command runner: `foreact.evaluation.runners`, CLI `run-benchmark`
   - External dataset path checks: `foreact.evaluation.datasets`, `configs/datasets.yaml`
   - External benchmark/baseline code manifest: `third_party/assets.yaml`
   - Asset availability command: `foreact.evaluation.assets`, CLI `asset-status`
   - Fetch scripts: `scripts/bootstrap_external_assets.sh`, `scripts/download_datasets.sh`
   - Fetch attempt evidence: `docs/EXTERNAL_ASSET_FETCH_LOG.md`
   - Current local status: benchmark code 5/5, baseline code 6/6, auxiliary code 3/3, nested archive code 2/2, datasets 4/4 available.
   - Setup guidance: `docs/BENCHMARK_SETUP.md`
   - Three-layer implementation distinction: `docs/DATASETS_AND_BASELINES.md`

9. **Config, CLI, docs, tests**
   - CLI: `foreact.cli`
   - Smoke config: `configs/smoke.yaml`
   - Milestone config: `configs/foreact_4b_milestone.yaml`
   - Tests: `tests/`
   - Check script: `scripts/check_repo.sh`
   - Full pipeline script: `scripts/run_full_pipeline.sh`
   - External package install script: `scripts/install_external_packages.sh`

10. **No fabricated results**
   - Smoke outputs are labeled as smoke artifacts.
   - Full benchmark and large-scale training commands are documented as requiring real data/API/GPU resources.

## Verification Commands

```bash
cd /Users/futaoran/Desktop/AAAI2027/Code-AAAI
bash scripts/check_repo.sh
PYTHONPATH=src python3 -m foreact.cli smoke --config configs/smoke.yaml
PYTHONPATH=src python3 -m foreact.cli harness-status --out outputs/smoke/harness_status.json
PYTHONPATH=src python3 -m foreact.cli dataset-status --config configs/datasets.yaml --out outputs/smoke/dataset_status.json
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
```

## Known External Dependencies

The repository now has pre-experiment entrypoints for official data preparation, reference HF/Qwen training, ablation job materialization, real ELD probing, and official harness command execution. Real paper numbers still require:

- Qwen3 checkpoints and GPU/cluster training.
- DeepSeek/OpenAI API keys or self-hosted teacher models when `teacher.provider` is not `mock`.
- Installing official harness packages into the active environment (`scripts/install_external_packages.sh`).
- Docker/OpenHands/Modal or equivalent execution infrastructure for real SWE-Gym/SWE-bench evaluation runs.

These are intentionally not mocked as real results.
