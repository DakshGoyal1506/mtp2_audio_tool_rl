# Local Path Hardening Report

## Files Scanned

Scanned only:

- `src/`
- `scripts/`
- `configs/`
- `tests/`

Excluded from this pass:

- `patches/`
- audit, extraction, and migration documentation
- `third_party/` submodules

## Files Changed

- `src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py`
- `scripts/diagnostics/run_diag.sh`
- `scripts/inference/desta_vllm/run_benchmark.sh`
- `scripts/inference/desta_vllm/run_vllm_grpo.sh`
- `scripts/slurm/run_filter_job.sh`
- `scripts/slurm/run_wavecaps_job.sh`
- `scripts/train/grpo/run_debug.sh`
- `scripts/train/grpo/run_trl.sh`
- `scripts/train/grpo/run_trl_v2.sh`
- `scripts/train/grpo/run_trl_v3.sh`
- `scripts/train/grpo_llm_decoupled/bench_judge.sh`
- `scripts/train/grpo_llm_decoupled/judge.sh`
- `scripts/train/grpo_llm_decoupled/run_trl_v4.sh`
- `scripts/train/grpo_llm_decoupled/test_vllm_small.sh`
- `scripts/train/grpo_single_phase/run_trl_v3.sh`

## Old Path Classes Removed

- Removed hard-coded `/home/speech-nlp-cse/24m0756/anaconda3/...` conda bootstrap paths.
- Removed hard-coded `/home/speech-nlp-cse/24m0756/.cache/tiktoken-rs-cache`.
- Confirmed no `/scratch/`, `/mnt/`, or absolute `/data/` paths remain in the scanned areas.

## New Environment Variables Used

- `HOME`: used as the fallback location for `$HOME/anaconda3` and `$HOME/.cache/tiktoken-rs-cache`.
- `TIKTOKEN_RS_CACHE_DIR`: optional override for the tiktoken-rs cache path.

No new `.env.example` entries were required for this pass. Existing `MTP2_*`, `HF_HOME`, `TRANSFORMERS_CACHE`, and `VLLM_CACHE_ROOT` variables remain available for data/model/cache configuration.

## Remaining Path References

The following `/tmp` references remain intentionally:

- `scripts/diagnostics/podman_gpu_test_job.sh`: temporary Podman overlay/runroot paths under `/tmp/$USER-*`.
- `tests/audio_maestro_test_embed_pipeline.py`: example optional output path `/tmp/test_embed.pt`.
- `tests/desta_test_embed_pipeline.py`: example optional output path `/tmp/test_embed.pt`.

These are not persistent dataset, cache, checkpoint, or model-output defaults. They are temporary runtime/test paths and are acceptable for this cleanup pass.

Docs and patch files may still contain old absolute paths as preservation metadata. They were intentionally excluded.

## Validation Commands Run

- `grep -R "/home/speech-nlp-cse/24m0756" -n src scripts configs tests || true`: no matches.
- `grep -R "/scratch/" -n src scripts configs tests || true`: no matches.
- `grep -R "/mnt/" -n src scripts configs tests || true`: no matches.
- `rg -n "(^|[^A-Za-z0-9_])/(home/speech-nlp-cse/24m0756|scratch|mnt|data)/" src scripts configs tests || true`: no matches.
- `grep -R "/tmp/" -n src scripts configs tests || true`: only the temporary/test references listed above.
- `bash -n` on edited shell scripts: PASS.
- `python3 -m py_compile src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py`: PASS; generated `__pycache__` was removed afterward.
- `python3 scripts/setup/check_repo_setup.py`: PASS.
- `python3 scripts/setup/check_imports.py`: PASS.
- `bash scripts/setup/apply_patches_preview.sh`: PASS for Audio-Maestro, DeSTA2.5-Audio, and ToolRL dry-runs.
- `find . -type f -size +10M -not -path "./third_party/*" -print`: no parent-repo files reported.
- `git submodule foreach --quiet 'git status --short'`: no submodule dirty output.

## Result

PASS. No hard-coded private IITB account paths remain in `src/`, `scripts/`, `configs/`, or `tests`. Setup checks and patch dry-runs still pass, and no submodules were modified.
