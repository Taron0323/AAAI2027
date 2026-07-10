# External Benchmark Setup

The smoke path does not require external benchmarks. Full paper experiments need the official harnesses below.

## Fetch External Code

The official/public benchmark and baseline repositories are listed in `third_party/assets.yaml`.

```bash
cd /Users/futaoran/Desktop/AAAI2027/Code-AAAI
bash scripts/bootstrap_external_assets.sh
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
```

If `git submodule add` cannot reach GitHub from this machine, retry archive mode:

```bash
bash scripts/bootstrap_external_assets.sh archive
```

Some official repositories carry their own nested submodules. If those nested clones are blocked by GitHub transport issues, fetch the commit-pinned archive-managed copies:

```bash
bash scripts/bootstrap_external_assets.sh nested
```

This step downloads official/public code only.

The current local checkout also includes:

- `mini-swe-agent` for lightweight SWE-bench execution
- `Reflexion` as an official baseline implementation
- `control-tasks` for probing/control-task analyses
- `verl` and `OpenRLHF` as optional training-framework code
- `verl-recipe` and Reflexion `leetcode-hard-gym` nested code as archive-managed checkouts

## tau2-bench

Role: stochastic user/environment branch benchmark for H2.

Expected domains:

- airline
- retail
- telecom in appendix/reference reporting

Adapter: `foreact.evaluation.adapters.Tau2BenchAdapter`

If missing, `python3 -m foreact.cli harness-status` reports a skipped status with setup guidance.
Dataset paths are checked separately with `python3 -m foreact.cli dataset-status --config configs/datasets.yaml`.

## AppWorld

Role: long-horizon API composition benchmark for H1.

Adapter: `foreact.evaluation.adapters.AppWorldAdapter`

Use official splits. Do not mix test tasks into synthetic training generation.
Set `APPWORLD_ROOT` before real runs.

## SWE-Gym and SWE-bench Verified

Role: train on non-overlapping SWE-Gym trajectories, evaluate on SWE-bench Verified.

Adapters:

- `SweGymAdapter`
- `SweBenchVerifiedAdapter`

This is the longest-tail domain in the project plan. If it cannot finish before submission, report it as an appendix/camera-ready completion item rather than fabricating numbers.
Set `SWE_GYM_ROOT` and `SWE_BENCH_ROOT` before real runs.

## Dataset Status

Download local ignored datasets and emit a status report:

```bash
bash scripts/download_datasets.sh
```

The script downloads:

- AppWorld official minimal data via `appworld download data`
- SWE-Gym official Hugging Face datasets and OpenHands trajectories
- SWE-bench Verified official Hugging Face test split

tau2-bench official domain data is included in the tau2-bench submodule.

If you keep datasets elsewhere, set:

```bash
export TAU2_BENCH_ROOT=/path/to/tau2-bench-data
export APPWORLD_ROOT=/path/to/appworld
export SWE_GYM_ROOT=/path/to/swe-gym
export SWE_BENCH_ROOT=/path/to/swe-bench
```

Re-check:

```bash
PYTHONPATH=src python3 -m foreact.cli dataset-status \
  --config configs/datasets.yaml \
  --out outputs/smoke/dataset_status.json
```

Large datasets, generated trajectories, model weights, and benchmark outputs are intentionally git-ignored.

## Provider Keys

Teacher APIs must read keys from environment variables:

- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`

The code must use `deepseek-v4-pro` and `deepseek-v4-flash` model IDs for the planned pipeline, not retired aliases.
