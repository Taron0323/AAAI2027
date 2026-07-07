# Local Git Workflow

The project root `/Users/futaoran/Desktop/AAAI2027` is the local git repository.

Recommended commit sequence:

1. `scaffold foreact repository`
2. `add plandepth data pipeline`
3. `add lap pcr training smoke path`
4. `add inference evaluation adapters`
5. `add tests and reproducibility docs`

Rules:

- Run `git status --short --branch` before and after each phase.
- Do not commit API keys, model weights, benchmark datasets, or generated large outputs.
- Do not use destructive commands such as `git reset --hard`.
- Keep paper placeholders as placeholders until real experiments produce measured numbers.
