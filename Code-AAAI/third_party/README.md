# Third-Party Assets

This directory is for official benchmark and baseline code that is too large or externally maintained.

Use:

```bash
bash scripts/bootstrap_external_assets.sh
PYTHONPATH=src python3 -m foreact.cli asset-status --out outputs/smoke/asset_status.json
```

The manifest is `third_party/assets.yaml`.

Datasets, model checkpoints, API caches, benchmark outputs, and environment images are not committed here. They belong under ignored local paths such as `datasets/`, `data/`, `runs/`, or paths configured through environment variables.
