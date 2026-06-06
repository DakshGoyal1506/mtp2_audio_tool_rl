# Reproducibility

This repository currently supports a limited, practical reproducibility surface rather than a full end-to-end training reproduction.

## Reproducible Now

- repository safety checks
- lightweight package import checks
- manifest validation with the included fake example
- patch dry-run checks against pinned submodules
- stable code, config, and path layout for the cleaned repo

## Not Reproducible Directly From This Repo

- full training runs
- model checkpoints
- raw datasets or raw audio
- generated tool outputs
- heavyweight evaluation or inference runs that depend on external assets

## External Requirements

- datasets must live outside Git
- submodules must be initialized
- model caches, checkpoints, and heavyweight artifacts must remain external

## Pointers

- Dataset reconstruction notes: [docs/dataset_reconstruction.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/dataset_reconstruction.md)
- Patch preview workflow: [docs/patch_application_plan.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/patch_application_plan.md)
