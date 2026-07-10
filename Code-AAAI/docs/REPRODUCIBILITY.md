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
- asset audit JSON proving external code assets contain no model weights

For v3 paper alignment, also report:

- H1-H5 mapping from `outputs/smoke/registry.json`
- Fig.2/Fig.5 ELD probe definition and shared axis metadata
- A'/B'/G anti-confound job keys used in the final tables
- whether Mode A used byte-for-byte ReAct prompts and no forecast decoding

Never promote smoke metrics to paper results. Smoke metrics only prove that code paths execute.
