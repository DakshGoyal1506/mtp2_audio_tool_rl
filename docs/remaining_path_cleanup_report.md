# Remaining Path Cleanup Report

## Files Changed

- `.env.example`
- `src/mtp2_audio_tool_rl/datasets/extract_audio.py`
- `scripts/data/run_gen_job.sh`
- `scripts/data/synthetic/run_embed.sh`
- `scripts/data/synthetic/run_extract.sh`
- `scripts/data/synthetic/run_inf.sh`

## Old Path Categories Removed

- Removed old `/home/speech-nlp-cse/24m0756/abhishek/...` workspace paths from the targeted dataset and synthetic-data scripts.
- Removed hard-coded conda paths under `/home/speech-nlp-cse/24m0756/anaconda3/...` from the targeted shell scripts.
- Removed hard-coded dataset, audio, embedding, and output paths from the targeted scripts.

## New Environment Variables Used

- `MTP2_REPO_ROOT`: optional repo root override for shell scripts.
- `MTP2_WORKSPACE_ROOT`: required in `run_inf.sh` for the external workspace containing `inference/tool_execute.py`.
- `MTP2_DATA_ROOT`: required by synthetic-data scripts for external dataset files.
- `MTP2_OUTPUT_ROOT`: required for generated audio/inference output locations.
- `MTP2_EMBED_ROOT`: optional embedding output/input root, defaulting under `MTP2_DATA_ROOT` where reasonable.
- `MTP2_AUDIO_ROOT`: required by `extract_audio.py`; optional in synthetic scripts, defaulting under `MTP2_DATA_ROOT`.
- `MTP2_MANIFEST_ROOT`: optional manifest root, defaulting to repo `manifests/` or `MTP2_DATA_ROOT` depending on script context.

`.env.example` now includes placeholder-only entries for these variables. No real paths or secrets were added.

## Commands Checked

- `rg` over the targeted files for old workspace, account-root, and vLLM cache paths: no matches.
- `bash -n` for the four edited shell scripts: PASS.
- `python3 -m py_compile src/mtp2_audio_tool_rl/datasets/extract_audio.py`: PASS; generated `__pycache__` was removed afterward.
- `grep -R "/home/speech-nlp-cse/24m0756/abhishek" -n src scripts configs || true`: no matches.
- `grep -R "/scratch/.cache/vllm" -n src scripts configs || true`: no matches.
- `python3 scripts/setup/check_repo_setup.py`: PASS.
- `python3 scripts/setup/check_imports.py`: PASS.
- `bash scripts/setup/apply_patches_preview.sh`: PASS for Audio-Maestro, DeSTA2.5-Audio, and ToolRL dry-runs.
- `find . -type f -size +10M -not -path "./third_party/*" -print`: no parent-repo files reported.

## Remaining Known Path Risks

The broader scan still finds `/home/speech-nlp-cse/24m0756` references in other source/scripts/config paths outside this focused cleanup pass. They are mostly local conda bootstrap or local cache paths in training, inference, Slurm, and diagnostic scripts:

- `src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py`
- `scripts/train/grpo_llm_decoupled/test_vllm_small.sh`
- `scripts/train/grpo_llm_decoupled/judge.sh`
- `scripts/train/grpo_llm_decoupled/bench_judge.sh`
- `scripts/train/grpo_llm_decoupled/run_trl_v4.sh`
- `scripts/train/grpo_single_phase/run_trl_v3.sh`
- `scripts/train/grpo/run_debug.sh`
- `scripts/train/grpo/run_trl_v2.sh`
- `scripts/train/grpo/run_trl.sh`
- `scripts/train/grpo/run_trl_v3.sh`
- `scripts/inference/desta_vllm/run_benchmark.sh`
- `scripts/inference/desta_vllm/run_vllm_grpo.sh`
- `scripts/slurm/run_wavecaps_job.sh`
- `scripts/slurm/run_filter_job.sh`
- `scripts/diagnostics/run_diag.sh`

These should be cleaned in a later pass using the same pattern: `conda` discovery, `$HOME` fallbacks, and `MTP2_*` roots. No old `/home/speech-nlp-cse/24m0756/abhishek` workspace paths remain in `src`, `scripts`, or `configs` after this pass.

## Preservation Metadata Note

Docs and patch files may still contain old absolute paths because they preserve audit history, patch provenance, and original workspace metadata. Those paths should remain unless the preservation policy changes.
