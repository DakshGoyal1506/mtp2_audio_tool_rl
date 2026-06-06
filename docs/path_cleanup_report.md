# Path Cleanup Report

Files changed:

- `scripts/diagnostics/diag_embed_gpu.py`
- `scripts/slurm/run_filter_job.sh`
- `scripts/slurm/run_wavecaps_job.sh`
- `configs/grpo/optimized.yaml`
- `configs/grpo/optimized_v2.yaml`
- `configs/grpo/optimized_v3.yaml`
- `configs/grpo/single_phase/optimized.yaml`
- `configs/grpo/single_phase/optimized_v2.yaml`
- `configs/grpo/single_phase/optimized_v3.yaml`
- `configs/grpo/audio_maestro/optimized.yaml`
- `scripts/inference/desta_vllm/run_vllm_grpo.sh`
- `scripts/inference/desta_vllm/run_benchmark.sh`
- `src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py`

## What changed

|file|old path type|new environment variable|behavior preserved|
|---|---|---|---|
|`scripts/diagnostics/diag_embed_gpu.py`|Hard-coded old workspace imports for `Desta_grpo` and `DeSTA2.5-Audio`.|`MTP2_WORKSPACE_ROOT`, `MTP2_DESTA_REPO`|Yes. The script now prefers repo-local `src/mtp2_audio_tool_rl` and uses env vars for external DeSTA code when needed.|
|`scripts/slurm/run_filter_job.sh`|Hard-coded old `grpo_dataset` workspace path.|`MTP2_WORKSPACE_ROOT`, `MTP2_DATA_ROOT`, `MTP2_FILTER_SUBDIR`|Yes. It still enters a filtering workspace, but the location is configurable.|
|`scripts/slurm/run_wavecaps_job.sh`|Hard-coded old `grpo_dataset/cache_tools` path.|`MTP2_WORKSPACE_ROOT`, `MTP2_DATA_ROOT`, `MTP2_CACHE_TOOLS_ROOT`|Yes. It still runs the same cache-tools pipeline, but from a configurable root.|
|GRPO optimized configs|Hard-coded old `Audio-Maestro/precomputed_embeds` path.|`MTP2_EMBED_ROOT`|Yes. OmegaConf env interpolation now resolves the embed root at runtime, defaulting to `precomputed_embeds`.|
|`scripts/inference/desta_vllm/run_vllm_grpo.sh`|Hard-coded `/scratch/.cache/vllm` cache path.|`VLLM_CACHE_ROOT`|Yes. The cache root remains configurable and defaults to a repo-local `.cache/vllm` path.|
|`scripts/inference/desta_vllm/run_benchmark.sh`|Hard-coded `/scratch/.cache/vllm` cache path.|`VLLM_CACHE_ROOT`|Yes. Same benchmark behavior, configurable cache root.|
|`src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py`|Hard-coded `/scratch/.cache/vllm` cache path in generated job scripts.|`VLLM_CACHE_ROOT`|Yes. Generated SLURM scripts now inherit a configurable cache root.|

## Remaining path risks

- Several scripts still contain environment-specific assumptions unrelated to the flagged path set, such as local Conda installation paths and container file names.
- `scripts/slurm/run_filter_job.sh` now has a repo-local fallback, but the clean repo currently does not include `scripts/data/filtering/filter_dataset.py`, so a real run may still require `MTP2_DATA_ROOT` to point at an external dataset workspace.
- `scripts/diagnostics/diag_embed_gpu.py` still depends on an external DeSTA checkout unless `third_party/DeSTA2.5-Audio` exists later.
- Relative defaults like `precomputed_embeds` preserve behavior for users who arrange local data the same way, but they still require external data placement outside Git.
