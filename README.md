# MTP2 Audio Tool RL

This repository contains extracted research code for tool-augmented audio-language model experiments, GRPO-style training, DeSTA/Audio-Maestro/ToolRL integrations, dataset construction, inference, evaluation, and patch preservation.

It is a cleaned code repository extracted from a much larger research workspace. The goal of this repository is to preserve project-owned code and patch metadata while excluding bulky datasets, raw audio, checkpoints, logs, caches, and other generated artifacts.

## Repository status

This repository is still being prepared for its first Git commit. Extraction, cleanup, and patch preservation have been completed for the initial safe code pass, but full documentation, import cleanup, environment curation, and reproducible training commands are still in progress.

This codebase is research-oriented and should not be treated as production-ready.

## What is included

- Extracted Python packages under `src/mtp2_audio_tool_rl/`
- Training and run scripts under `scripts/`
- Selected configs under `configs/`
- Extracted tests under `tests/`
- Audit, migration, and patch-planning docs under `docs/`
- Patch preservation records under `patches/`

Included code areas currently cover:

- Canonical GRPO training code
- GRPO single-phase variants
- GRPO LLM-decoupled variants
- Audio-Maestro GRPO variants
- DeSTA vLLM inference and rollout helpers
- Dataset construction and synthetic data scripts
- Evaluation, diagnostics, prompts, and config files

## What is excluded

This repository intentionally excludes:

- Raw audio and copied audio subsets
- Full datasets and large manifest dumps
- Checkpoints, model weights, and embedding caches
- Logs, Slurm outputs, `wandb/`, and generated results
- Full third-party cloned repositories in the first commit
- Local machine files, virtual environments, and editor state

## Repository layout

```text
mtp2-audio-tool-rl/
├── README.md
├── .gitignore
├── .gitattributes
├── configs/
├── docs/
├── manifests/
├── patches/
├── scripts/
├── src/
├── tests/
└── third_party/
```

## External repositories and patches

This repository does not vendor full upstream third-party repositories in the first commit. Instead, local tracked modifications are preserved under `patches/`, and future submodules or forks can be added after the first clean commit.

Current preserved patch areas include:

- `patches/audio-maestro-main/`
- `patches/audio-maestro-bak/`
- `patches/desta25-audio/`
- `patches/desta-grpo/`
- `patches/desta-grpo-toolrl/`
- `patches/grpo-dataset-cache-audio-maestro/`

See [third_party/README.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/third_party/README.md) and [docs/submodule_plan.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/submodule_plan.md) for the planned upstream relationships.

## Dataset policy

Datasets are intentionally kept outside Git. This repository should only commit tiny examples, schemas, and instructions that help reconstruct or reference external data.

Do not commit raw audio, copied audio folders, full JSON or JSONL datasets, large parquet exports, checkpoints, result dumps, or caches.

## Setup placeholder

Environment cleanup is still in progress. A curated `environment/` setup has not been finalized yet, and several extracted scripts still need dependency review and path cleanup before they can be treated as reproducible entry points.

For now, treat the repository as an extracted research snapshot with code and patch preservation first, setup standardization second.

## Reproducibility status

Reproducibility is not complete yet. We have preserved code, configs, tests, and patch metadata, but the following are still in progress:

- Import cleanup and package-level verification
- Canonical environment selection
- Reproducible training and inference commands
- Dataset manifest curation
- Third-party submodule or fork decisions

## GitHub target

- Owner: `DakshGoyal1506`
- Repo: `mtp2_audio_tool_rl`
- Preferred remote: `git@github.com:DakshGoyal1506/mtp2_audio_tool_rl.git`
- HTTPS fallback: `https://github.com/DakshGoyal1506/mtp2_audio_tool_rl.git`

Git initialization, the first commit, and any push should happen only after the remaining metadata, license review, and safety checks are complete.
