# Submodule Plan

Submodules are intentionally deferred until after the first clean commit.

That ordering keeps the first Git history focused on extracted project-owned code, metadata, and patch preservation, without mixing in submodule state.

## Recommended later submodules

- `third_party/Audio-Maestro` -> `https://github.com/gary920209/Audio-Maestro.git`
- `third_party/DeSTA2.5-Audio` -> `https://github.com/kehanlu/DeSTA2.5-Audio.git`
- `third_party/ToolRL` -> `https://github.com/qiancheng0/ToolRL.git`
- Optional `third_party/AHa-Bench` -> `https://github.com/AHa-Bench/AHa-Bench.git`

## Patch overlays to preserve

If submodules are added later, the following patch overlays should remain part of the repository history:

- `patches/audio-maestro-main`
- `patches/desta25-audio`
- `patches/desta-grpo-toolrl`

## Do not submodule as-is

The following workspace areas should not be turned into submodules in their current extracted or mixed state:

- `Desta_grpo`
- `grpo_dataset`
- `rlTool`
- `toolRL/AF3`

`Desta_grpo` contains project-owned extracted research code plus preserved patch context. `grpo_dataset` and `rlTool` are treated as project workspaces rather than clean upstream dependencies. `toolRL/AF3` remains mixed with outputs and needs later human review.
