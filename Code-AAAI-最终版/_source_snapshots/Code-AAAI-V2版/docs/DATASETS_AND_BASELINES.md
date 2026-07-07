# Datasets, Benchmarks, and Baseline Implementation Levels

This repository deliberately separates three implementation levels.

## 1. Built-In Runnable Smoke Benchmark/Baselines

These run without external datasets, GPUs, API keys, `torch`, or `transformers`:

- `PlanDepth`
- `ReAct`
- `ReAct-SFT`
- `token-MTP`
- `Plan-and-Act / Pre-Act` smoke baseline
- `LATS / WebDreamer` smoke baseline
- `ForeAct latent rerank`

Run them with:

```bash
PYTHONPATH=src python3 -m foreact.cli smoke --config configs/smoke.yaml
```

Outputs are written under `outputs/smoke/` and are smoke artifacts, not paper results.

## 2. External Benchmark And Baseline Code Assets

Official/public code assets are tracked in `third_party/assets.yaml` and should be fetched into `third_party/`:

- tau2-bench
- AppWorld
- SWE-Gym
- SWE-bench Verified
- mini-SWE-agent
- ReAct reference
- LATS / LanguageAgentTreeSearch
- WebDreamer
- Plan-and-Act
- PreAct
- Reflexion
- verl-recipe nested code used by verl
- leetcode-hard-gym nested code used by Reflexion

Fetch/check them with:

```bash
bash scripts/bootstrap_external_assets.sh
bash scripts/bootstrap_external_assets.sh nested
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
```

If git submodule cloning is blocked by the local network, use the archive fallback:

```bash
bash scripts/bootstrap_external_assets.sh archive
```

The built-in smoke baselines remain runnable without those external repos, but exact official-baseline reproduction requires the upstream code to be present locally.

Current local status:

- benchmark code: downloaded as git submodules, 5/5 available
- baseline code: downloaded as git submodules, 6/6 available
- auxiliary code: `control-tasks`, `verl`, and `OpenRLHF` available locally
- nested archive code: `verl-recipe` and Reflexion `leetcode-hard-gym` available locally

## 3. External Benchmark Adapters

Check Python harness availability:

```bash
PYTHONPATH=src python3 -m foreact.cli harness-status --out outputs/smoke/harness_status.json
```

Check configured dataset paths:

```bash
PYTHONPATH=src python3 -m foreact.cli dataset-status \
  --config configs/datasets.yaml \
  --out outputs/smoke/dataset_status.json
```

If a package or path is missing, the command reports a skipped/missing status with setup guidance.

## 4. Real Dataset / Official Baseline Layer

The repository does not commit official benchmark datasets, private API outputs, or model weights. Official datasets are downloaded into git-ignored local paths.

Current local official dataset status:

- tau2-bench: official domain data present inside the tau2-bench submodule
- AppWorld: official minimal data present under `datasets/appworld/data`
- SWE-Gym: official Hugging Face datasets and OpenHands trajectories present under `datasets/swe-gym`
- SWE-bench Verified: official 500-instance test split present under `datasets/swe-bench/verified`

Before real paper runs, provide:

- Qwen3 checkpoints
- DeepSeek/OpenAI API keys or self-hosted teacher model endpoints
- Docker/OpenHands/Modal or equivalent execution infrastructure for SWE-style evaluation runs

This separation prevents smoke tests from being mistaken for real benchmark results.
