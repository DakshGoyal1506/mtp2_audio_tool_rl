# Remaining Work

## Current State

The repository is in a solid private-working state: setup is complete, lightweight CI is working, submodules are pinned, patch dry-runs pass, and the dataset skeleton is in place. The current public-readiness verdict is still `PRIVATE OK` but `NOT READY FOR PUBLIC`, mainly because licensing, patch-publication review, and public-facing documentation are not finished.

## Blocking Before Public Release

- License review:
  Confirm upstream licenses for Audio-Maestro and DeSTA2.5-Audio, then decide whether a top-level project license is compatible with extracted code, submodules, and patch overlays.
- Patch secret-risk review:
  Manually review patch files that mention Gemini/API-key variables and rerun a dedicated secret scan before any visibility change.
- Third-party patch strategy:
  Decide whether preserved third-party changes should stay as patch overlays, move to forks, or remain private-only preservation metadata.
- Public docs cleanup:
  Separate internal audit/migration notes from public-facing setup, reproducibility, and data reconstruction documentation.

## Recommended GitHub Issues

### Title
Review upstream licenses and decide top-level license

- Priority: `blocking`
- Scope: audit upstream license terms for Audio-Maestro, DeSTA2.5-Audio, ToolRL, and the interaction between extracted code and patch files.
- Acceptance criteria: upstream license status is documented; redistribution constraints are understood; decision recorded on whether to add a top-level `LICENSE`.
- Files likely involved: `pyproject.toml`, `docs/license_notes.md`, `docs/license_audit.md`, `third_party/README.md`, `.gitmodules`

### Title
Review patch files for secret-risk before public release

- Priority: `blocking`
- Scope: manually inspect patch files that reference Gemini/API-key variables and confirm they contain only environment-variable references or placeholders.
- Acceptance criteria: secret review is documented; any risky patch content is removed, redacted, or kept private; a follow-up secret scan passes.
- Files likely involved: `docs/secret_risk_audit.md`, `patches/audio-maestro-main/`, `patches/audio-maestro-bak/`, `patches/desta-grpo/`

### Title
Decide patch strategy: overlays vs forks vs private-only

- Priority: `blocking`
- Scope: decide how third-party modifications should be preserved for a future public repo.
- Acceptance criteria: each patch directory has a chosen strategy; duplicate patches are rationalized; public/private treatment is documented.
- Files likely involved: `docs/patch_application_plan.md`, `docs/license_audit.md`, `patches/`, `third_party/README.md`

### Title
Create public-facing README and reproducibility guide

- Priority: `high`
- Scope: write public-safe setup, environment, submodule, manifest, and reconstruction guidance without exposing internal migration detail.
- Acceptance criteria: README and reproducibility docs explain what the repo is, how to set it up, what is intentionally excluded, and what users can realistically run.
- Files likely involved: `README.md`, `docs/dataset_reconstruction.md`, `docs/ci_status.md`, `manifests/README.md`

### Title
Add lightweight unit tests for non-GPU utilities

- Priority: `medium`
- Scope: add tests for manifest validation, path helpers, and other small non-heavy utilities that can run in CI without models or datasets.
- Acceptance criteria: at least a small fast test set runs in CI; tests do not require GPUs, checkpoints, or external datasets.
- Files likely involved: `tests/`, `.github/workflows/repo-check.yml`, `scripts/data/validate_manifest.py`, `scripts/setup/check_repo_setup.py`

### Title
Decide whether audit/migration docs should remain in public repo

- Priority: `medium`
- Scope: review internal audit, migration, and extraction documents for whether they belong in a public release.
- Acceptance criteria: each major audit/migration doc is either retained, summarized, moved private, or replaced with a public-safe version.
- Files likely involved: `docs/public_readiness_audit.md`, `docs/public_release_blockers.md`, `docs/extraction_report.md`, `docs/extraction_inventory.md`, `docs/migration_plan.md`

### Title
Create external artifact and dataset reconstruction checklist

- Priority: `low`
- Scope: turn the current dataset skeleton into a concise operator checklist for reconstructing manifests, external audio exports, and tool outputs outside Git.
- Acceptance criteria: a short checklist exists for external data prep, manifest validation, artifact placement, and what must never be committed.
- Files likely involved: `docs/dataset_reconstruction.md`, `manifests/dataset_sources.md`, `manifests/audio_manifest.schema.json`, `scripts/data/validate_manifest.py`

## Not Needed Now

- No datasets, raw audio, checkpoints, or generated results should be added to Git.
- No patch application inside submodules is needed; dry-run preview is enough for now.
- No public release should happen yet.
- No heavy CI job is needed yet; lightweight checks are the right default until non-GPU tests are added.
