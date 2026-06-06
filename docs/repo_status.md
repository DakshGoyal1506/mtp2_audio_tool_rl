# Repository Status

This clean repository currently contains the first safe extraction pass from a much larger MTP research workspace.

## Current status

- `141` files were copied into the clean repo during the safe extraction pass.
- The two copied cleanup issues were resolved inside the clean repo.
- Six patch directories were created under `patches/`.
- No files above `10 MB` are currently present.
- Git has not been initialized in this repository yet.

## Extracted areas

The current extraction includes:

- GRPO
- GRPO single phase
- GRPO LLM decoupled
- Audio-Maestro GRPO
- DeSTA vLLM
- Dataset construction scripts
- Synthetic dataset scripts
- Evaluation scripts
- Tests
- Prompts and configs

## Excluded areas

The current clean repo excludes:

- Datasets
- Raw audio
- Copied audio
- Checkpoints
- Logs
- `wandb/`
- Generated outputs
- Container images

## Notes

This is a cleaned research code repository, not a finalized release. Patch preservation is in place, but documentation, setup curation, import cleanup, and reproducible command validation are still in progress.
