# Third-Party Repositories

Full upstream third-party repositories are not vendored in the first commit of this clean repository.

Instead, local tracked modifications from the larger research workspace are preserved under `patches/`, and submodules or forks may be added later after the first clean commit.

## Planned upstream references

- `Audio-Maestro`
  - Upstream: `https://github.com/gary920209/Audio-Maestro.git`
  - Local modifications preserved in: `patches/audio-maestro-main/` and `patches/audio-maestro-bak/`
- `DeSTA2.5-Audio`
  - Upstream: `https://github.com/kehanlu/DeSTA2.5-Audio.git`
  - Local modifications preserved in: `patches/desta25-audio/`
- `ToolRL`
  - Upstream: `https://github.com/qiancheng0/ToolRL.git`
  - Local modifications preserved in: `patches/desta-grpo-toolrl/`
  - Clean clone candidate noted in audit: external `toolRL/ToolRL`
- `AHa-Bench`
  - Upstream: `https://github.com/AHa-Bench/AHa-Bench.git`
  - No patch directory was created in this pass because the audited nested repo was clean

## Patch policy

Patch directories preserve tracked changes, changed-file summaries, repo metadata, and untracked file listings without copying bulky artifacts into this clean repo.

Submodules should be added only after the first clean commit, not before. That keeps the initial Git history focused on extracted project-owned code and patch preservation metadata.
