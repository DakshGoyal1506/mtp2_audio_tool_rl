# Remaining Work

## Current State

The repository is in a solid working state: setup is complete, lightweight CI is working, lightweight validation tests have been added, submodules are pinned, patch dry-runs pass, the dataset skeleton is in place, and a scoped MIT license now covers original repository-authored code and documentation. A minimal GRPO/tool-use experiment layer now exists with a canonical tool-call schema, deterministic reward breakdown, smoke config, dry-run runner, and baseline-vs-GRPO comparison script. Public visibility is possible with third-party caveats documented, but there is still worthwhile follow-up work.

## Blocking Before Public Release

- License review:
  Keep issue `#1` open to confirm the unresolved Audio-Maestro license and clarify the ambiguous DeSTA2.5-Audio license signals.
- Patch secret-risk review:
  Manually review patch files that mention Gemini/API-key variables and rerun a dedicated secret scan as follow-up.
- Third-party patch strategy:
  Decide whether preserved third-party changes should stay as patch overlays, move to forks, or remain private-only preservation metadata.
- Public docs cleanup:
  A public documentation map and internal-audit index now exist. The remaining decision is whether to move internal audit/migration notes into `docs/internal/` later.

## Recommended GitHub Issues

### Title
Review upstream licenses and decide top-level license

- Priority: `high`
- Scope: audit upstream license terms for Audio-Maestro, DeSTA2.5-Audio, ToolRL, and the interaction between extracted code, top-level MIT licensing, and patch files.
- Acceptance criteria: upstream license status is documented; Audio-Maestro remains resolved or explicitly unresolved; DeSTA2.5-Audio ambiguity is resolved; repository notices remain accurate.
- Files likely involved: `LICENSE`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, `docs/license_notes.md`, `docs/license_audit.md`, `docs/public_release_blockers.md`, `third_party/README.md`, `.gitmodules`

### Title
Review patch files for secret-risk before public release

- Priority: `high`
- Scope: manually inspect patch files that reference Gemini/API-key variables and confirm they contain only environment-variable references or placeholders.
- Acceptance criteria: secret review is documented; any risky patch content is removed, redacted, or kept private; a follow-up secret scan passes.
- Files likely involved: `docs/secret_risk_audit.md`, `patches/audio-maestro-main/`, `patches/audio-maestro-bak/`, `patches/desta-grpo/`

### Title
Decide patch strategy: overlays vs forks vs private-only

- Priority: `high`
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

- Priority: `low`
- Scope: extend the new lightweight test surface with more non-heavy utilities as they stabilize. Current coverage includes manifest validation, repository setup helpers, tool-call schema validation, tool-use reward breakdowns, and baseline-vs-GRPO comparison.
- Acceptance criteria: additional fast tests run in CI; tests do not require GPUs, checkpoints, or external datasets.
- Files likely involved: `tests/`, `.github/workflows/repo-check.yml`, `scripts/data/validate_manifest.py`, `scripts/setup/check_repo_setup.py`

### Title
Connect GRPO/tool-use debug runner to a real training entrypoint

- Priority: `high`
- Scope: choose the canonical GRPO training entrypoint and wire `scripts/train/grpo/run_grpo_debug.sh` behind `RUN_ACTUAL_GRPO=1`.
- Acceptance criteria: dry-run mode remains safe; actual mode runs only with external manifests/output roots; no outputs are written into Git.
- Files likely involved: `scripts/train/grpo/run_grpo_debug.sh`, `configs/experiments/grpo_tool_use_smoke.yaml`, `src/mtp2_audio_tool_rl/grpo/`, `docs/grpo_tool_use_experiment.md`

### Title
Generate external baseline and GRPO prediction files

- Priority: `high`
- Scope: run baseline and GRPO inference externally using the smoke manifest format, then compare outputs with the new evaluation script.
- Acceptance criteria: prediction JSONL files stay outside Git; comparison metrics are generated under an external output directory; result summaries use `docs/results_table_template.md`.
- Files likely involved: `scripts/eval/compare_baseline_grpo.py`, `docs/results_table_template.md`, external manifests and output directories

### Title
Decide whether audit/migration docs should remain in public repo

- Priority: `medium`
- Scope: review internal audit, migration, and extraction documents for whether they should stay where they are or move under `docs/internal/`.
- Acceptance criteria: each major audit/migration doc is either retained, summarized, moved to `docs/internal/`, moved private, or replaced with a public-safe version.
- Files likely involved: `docs/PUBLIC_DOCS.md`, `docs/INTERNAL_AUDITS.md`, `docs/public_readiness_audit.md`, `docs/public_release_blockers.md`, `docs/extraction_report.md`, `docs/extraction_inventory.md`, `docs/migration_plan.md`

### Title
Create external artifact and dataset reconstruction checklist

- Priority: `low`
- Scope: turn the current dataset skeleton into a concise operator checklist for reconstructing manifests, external audio exports, and tool outputs outside Git.
- Acceptance criteria: a short checklist exists for external data prep, manifest validation, artifact placement, and what must never be committed.
- Files likely involved: `docs/dataset_reconstruction.md`, `manifests/dataset_sources.md`, `manifests/audio_manifest.schema.json`, `scripts/data/validate_manifest.py`

## Not Needed Now

- No datasets, raw audio, checkpoints, or generated results should be added to Git.
- No patch application inside submodules is needed; dry-run preview is enough for now.
- No heavy CI job is needed yet; lightweight checks are the right default until non-GPU tests are added.
- No immediate doc moves are needed; the public/internal indexes are enough for now.
- No claim should be made that third-party submodules or patch-derived code are MIT-licensed by the top-level `LICENSE`.
