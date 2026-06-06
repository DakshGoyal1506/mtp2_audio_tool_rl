# MTP2 Audio Tool RL

MTP2 Audio Tool RL is a cleaned research-code repository for tool-augmented audio-language model experiments. It collects extracted project-owned code for GRPO-style training, DeSTA/Audio-Maestro/ToolRL integrations, dataset construction helpers, inference utilities, evaluation scripts, and patch-preservation metadata from a larger research workspace.

## Current Repository Status

This is a public research repository, not a polished reproduction package. Lightweight setup checks, manifest validation, CI, submodule pinning, and patch dry-runs are in place. Full end-to-end training reproduction, dataset reconstruction, and heavyweight test coverage are still incomplete.

## What Is Included

- Python package code under `src/mtp2_audio_tool_rl/`
- Run scripts and setup helpers under `scripts/`
- Selected configs under `configs/`
- Tests and diagnostics that were safe to extract
- Dataset skeleton files under `manifests/`
- Preserved patch metadata under `patches/`
- Pinned third-party submodules under `third_party/`

## What Is Not Included

- Raw datasets or copied audio
- Model checkpoints, weights, embeddings, or caches
- Generated tool outputs, logs, or results
- Applied third-party patch changes inside submodules
- A complete paper-style reproducibility package

## Repository Layout

```text
mtp2-audio-tool-rl/
├── configs/
├── docs/
├── manifests/
├── patches/
├── scripts/
├── src/
├── tests/
└── third_party/
```

## Quickstart

Clone with submodules and install the lightweight base environment:

```bash
git clone --recurse-submodules https://github.com/DakshGoyal1506/mtp2_audio_tool_rl.git
cd mtp2_audio_tool_rl
python3 -m pip install -e .
python3 -m pip install -r requirements/base.txt
```

Run the lightweight checks:

```bash
python3 scripts/setup/check_repo_setup.py
python3 scripts/setup/check_imports.py
python3 scripts/data/validate_manifest.py manifests/example_manifest.jsonl
bash scripts/setup/apply_patches_preview.sh
```

More detail: [docs/quickstart.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/quickstart.md)

## Submodules

This repository uses pinned submodules for upstream dependencies such as `Audio-Maestro`, `DeSTA2.5-Audio`, and `ToolRL`. They are checked out for reference and patch dry-run validation only. The repository does not claim that those third-party components are relicensed under the top-level MIT license.

## Setup Checks

The current lightweight validation surface includes:

- repository safety and artifact checks
- package import checks without heavy ML/audio dependencies
- example manifest validation
- patch dry-runs against pinned submodules

These same checks are also used in CI. Full `pytest` is intentionally not part of the default path yet.

## Dataset Policy

Datasets stay outside Git. This repository includes only lightweight dataset metadata such as schemas, tiny fake/example manifests, and reconstruction notes. See [manifests/README.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/manifests/README.md) and [docs/dataset_reconstruction.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/dataset_reconstruction.md).

## Patch Preview Workflow

Patch files under `patches/` preserve local tracked modifications from earlier working copies. The repository provides a dry-run-only preview script:

```bash
bash scripts/setup/apply_patches_preview.sh
```

This checks whether selected preserved patches apply cleanly to the pinned submodules, but it does not apply them.

## License And Third-Party Notice

The top-level [LICENSE](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/LICENSE) is MIT for original repository-authored code, scripts, configuration, documentation, and scaffolding. Third-party submodules and patch-derived content retain their own terms or unresolved status. See [THIRD_PARTY_NOTICES.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/THIRD_PARTY_NOTICES.md).

## Current Limitations

- Full training reproduction is not turnkey yet.
- Datasets, raw audio, checkpoints, and generated outputs are not included.
- Heavy tests and model-dependent scripts require external assets and optional dependencies.
- Patch overlays are preserved and validated, but not applied automatically.

Reproducibility notes: [docs/reproducibility.md](/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/reproducibility.md)
