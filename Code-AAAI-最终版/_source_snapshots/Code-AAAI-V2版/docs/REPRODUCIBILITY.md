# Reproducibility Checklist Notes

This repository is designed to support the AAAI reproducibility checklist without claiming unmeasured results.

Report for each real run:

- git commit hash
- config YAML
- backbone checkpoint and context length
- prompt/tool-call template
- teacher model IDs and cache policy
- number of trajectories, prefixes, rollouts `K`, and horizon `H`
- sketch mode and schema file
- optimizer, batch size, learning rate schedule, update tokens, seeds
- benchmark harness version and split
- bootstrap confidence intervals and paired tests
- serving stack and hardware for mode A/B efficiency

Never promote smoke metrics to paper results. Smoke metrics only prove that code paths execute.

