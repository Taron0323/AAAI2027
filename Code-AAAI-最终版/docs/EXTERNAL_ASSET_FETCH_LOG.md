# External Asset Fetch Log

This log records attempts to fetch official/public benchmark and baseline code into `third_party/`.

## 2026-07-07

Commands attempted from `/Users/futaoran/Desktop/AAAI2027/Code-AAAI-最终版`:

```bash
git submodule add https://github.com/sierra-research/tau2-bench.git third_party/benchmarks/tau2-bench
GIT_HTTP_VERSION=HTTP/1.1 git -c http.version=HTTP/1.1 ls-remote https://github.com/sierra-research/tau2-bench.git HEAD
bash scripts/bootstrap_external_assets.sh archive
```

Observed local network results:

- `git submodule add` failed before adding any submodule: `Error in the HTTP2 framing layer`.
- `git ls-remote` attempts later failed or stalled with inability to connect to `github.com:443`.
- Archive downloads reached GitHub/codeload but were rejected with HTTP 429 for all configured code assets.

Current state after the attempt:

- `third_party/assets.yaml` contains the official/public code asset targets.
- `third_party/benchmarks/` and `third_party/baselines/` exist as empty target directories.
- `outputs/smoke/asset_status.json` reports `benchmark_code: 0/4`, `baseline_code: 0/5`, `datasets: 0/4`.
- No third-party code, official dataset, model weight, or fake benchmark result was committed.

Retry when GitHub access is healthy:

```bash
cd /Users/futaoran/Desktop/AAAI2027/Code-AAAI-最终版
bash scripts/bootstrap_external_assets.sh
# or
bash scripts/bootstrap_external_assets.sh archive
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
```

## 2026-07-07 Follow-Up

The external fetch was retried successfully.

Official/public code now exists as git submodules:

- tau2-bench: `1901a301961cbbe3fd11f3e84a2a376530c759e3`
- AppWorld: `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`
- SWE-Gym: `b681068ca20628c6987b7416cc4cf03f06b77ba5`
- SWE-bench: `f7bbbb2ccdf479001d6467c9e34af59e44a840f9`
- ReAct: `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`
- LanguageAgentTreeSearch: `853d81614607dd27433faf17c7b0a7d660f95d22`
- WebDreamer: `e58941109170ca9f12658116d8e65bc70c1c57f3`
- plan-and-act: `534ed56f0d75e3059e54907859b89c901094f293`
- PreAct: `89cad5a79b9d3f023177e2186ef1a371f53229a5`
- mini-SWE-agent: `e187bcb2ff5825d85761a6f9c1f98c9fa6cfbc79`
- Reflexion: `218cf0ef1df84b05ce379dd4a8e47f17766733a0`
- verl: `e52747a403f55044578d9435069825f949b549bf`
- OpenRLHF: `3f8ae08c99db23a3532abc3159144f6a0821a6d0`
- control-tasks: archive-managed checkout from `john-hewitt/control-tasks` at tarball prefix `john-hewitt-control-tasks-be70d7f`

Official datasets now exist locally but are git-ignored:

- tau2-bench official domain data: `third_party/benchmarks/tau2-bench/data/tau2/domains`
- AppWorld official minimal data: `datasets/appworld/data`
- SWE-Gym official Hugging Face datasets and OpenHands trajectories: `datasets/swe-gym`
- SWE-bench Verified official 500-instance test split: `datasets/swe-bench/verified`

Current machine-readable status:

```text
benchmark_code: 4/4
baseline_code: 5/5
datasets: 4/4
```

Additional follow-up code requested by the user is also present:

```text
benchmark_code: 5/5 including mini-SWE-agent
baseline_code: 6/6 including Reflexion
auxiliary_code: 3/3 for control-tasks, verl, OpenRLHF
nested_archive_code: 2/2 for verl-recipe and Reflexion leetcode-hard-gym
```

Two nested upstream submodules were repeatedly blocked by GitHub clone transport errors, so they were fetched as commit-pinned codeload tarballs into the exact nested paths expected by their parent repositories:

- verl recipe: `third_party/auxiliary/verl/recipe` from `verl-project/verl-recipe@e7f889574b8301cc0f0fc1d57c6d67f31ffeb689`
- Reflexion leetcode env: `third_party/baselines/reflexion/programming_runs/executors/leetcode_env` from `GammaTauAI/leetcode-hard-gym@228163abdc983712bebfd8e26f7e7d360830e648`

Each archive-managed nested checkout includes a `.foreact_archive_source` file. Rebuild them with:

```bash
bash scripts/bootstrap_external_assets.sh nested
```
