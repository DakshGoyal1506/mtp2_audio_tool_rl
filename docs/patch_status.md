# Patch Status
## Cleanup Completed
- `src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge copy.py` was not a byte-identical duplicate of `judge.py`; it changes the judge prompt, scoring ranges, model defaults, and distributed timing behavior. It was renamed to `src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge_variant.py` so it remains reviewable as an intentional variant.
- `src/mtp2_audio_tool_rl/desta_vllm/__init____Desta_grpo__desta_vllm____init___py.py` contained real package exports. Those exports were merged into `src/mtp2_audio_tool_rl/desta_vllm/__init__.py` with package-relative imports, then the generated collision file was removed from the clean repo.

## Patches Created
| Patch directory | Original path | Remote | Commit | Modified | Tracked files | Untracked listed | Notes |
|---|---|---|---|---:|---:|---:|---|
|`patches/audio-maestro-main/`|`/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro`|`https://github.com/gary920209/Audio-Maestro.git`|`ba443a215573ff2d5b85e8efbec268343b40baa2`|yes|5|2263|duplicate Audio-Maestro tracked patch|
|`patches/audio-maestro-bak/`|`/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro-bak`|`https://github.com/gary920209/Audio-Maestro.git`|`ba443a215573ff2d5b85e8efbec268343b40baa2`|yes|5|191|duplicate Audio-Maestro tracked patch|
|`patches/desta25-audio/`|`/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio`|`https://github.com/kehanlu/DeSTA2.5-Audio.git`|`e9b28ffd97c559eb178096b853321a1bbe5d97cf`|yes|1|1|Submodule candidate later, with this patch applied on top.|
|`patches/desta-grpo/`|`/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo`|`https://github.com/gary920209/Audio-Maestro.git`|`ba443a215573ff2d5b85e8efbec268343b40baa2`|yes|6|113623|Do not submodule; project-owned code was extracted, tracked third-party-derived changes are preserved here.|
|`patches/desta-grpo-toolrl/`|`/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL`|`https://github.com/qiancheng0/ToolRL.git`|`8cee13ec0ca72f0461da372a93a6fd8140dbb840`|yes|3|18|Submodule or fork candidate later, with this patch applied on top.|
|`patches/grpo-dataset-cache-audio-maestro/`|`/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro`|`https://github.com/gary920209/Audio-Maestro.git`|`ba443a215573ff2d5b85e8efbec268343b40baa2`|yes|1|2|Patch only; likely dependency snapshot used by cache tooling.|

## Modified Repos
- `audio-maestro-main`
- `audio-maestro-bak`
- `desta25-audio`
- `desta-grpo`
- `desta-grpo-toolrl`
- `grpo-dataset-cache-audio-maestro`

## Clean Repos
- None of the six patch-preservation targets were clean; each had tracked changes and/or untracked files listed.
- Audit note: `/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL` was previously identified as a clean upstream clone and remains a good submodule candidate; no patch directory was requested for it in this pass.
- Audit note: `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/hallucination/AHa-Bench` was previously identified as clean and optional as a later submodule if needed.

## Submodule Candidates
- `Audio-Maestro`: submodule candidate, with patch overlay from `patches/audio-maestro-main/`.
- `DeSTA2.5-Audio`: submodule candidate, with patch overlay from `patches/desta25-audio/`.
- `ToolRL` clean clone: submodule candidate from `/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL`.
- `AHa-Bench`: optional submodule later if needed.

## Fork Or Patch Decisions
- `Desta_grpo/ToolRL`: patch is captured, but this may become a fork if the `audio_grpo/`, `dataset/audio_grpo/`, and core `verl/` changes need ongoing development.
- `Desta_grpo`: do not submodule; own GRPO code has been extracted, and Audio-Maestro-derived tracked modifications are preserved as a patch.
- `Audio-Maestro-bak`: do not submodule unless human review finds meaningful differences from `Audio-Maestro`; tracked patches are currently byte-identical.
- `grpo_dataset`: do not submodule; dataset-construction code was extracted, while data/artifacts remain outside the clean repo.
- `rlTool`: do not submodule; copied scripts are project-owned, artifacts remain ignored.
- `toolRL/AF3`: review later; do not submodule as-is because it is mixed with outputs/checkpoints/sandbox material.

## Important Untracked Folders Needing Human Review
- `audio-maestro-main` has 2263 untracked entries listed. Examples needing review:
  - `checkpoints/grpo-desta-run2/best/README.md`
  - `checkpoints/grpo-desta-run2/best/adapter_config.json`
  - `checkpoints/grpo-desta-run2/best/adapter_model.safetensors`
  - `checkpoints/grpo-desta-run2/best/training_meta.json`
  - `checkpoints/grpo-desta-run2/epoch-1/README.md`
  - `checkpoints/grpo-desta-run2/epoch-1/adapter_config.json`
  - `checkpoints/grpo-desta-run2/epoch-1/adapter_model.safetensors`
  - `checkpoints/grpo-desta-run2/epoch-1/training_meta.json`
  - `checkpoints/grpo-desta-run2/epoch-2/README.md`
  - `checkpoints/grpo-desta-run2/epoch-2/adapter_config.json`
  - `checkpoints/grpo-desta-run2/epoch-2/adapter_model.safetensors`
  - `checkpoints/grpo-desta-run2/epoch-2/training_meta.json`
- `audio-maestro-bak` has 191 untracked entries listed. Examples needing review:
  - `grpo/__init__.py`
  - `grpo/__pycache__/__init__.cpython-310.pyc`
  - `grpo/__pycache__/dataset.cpython-310.pyc`
  - `grpo/__pycache__/dataset.cpython-313.pyc`
  - `grpo/__pycache__/diversity_sampler.cpython-310.pyc`
  - `grpo/__pycache__/model_wrapper.cpython-310.pyc`
  - `grpo/__pycache__/model_wrapper.cpython-313.pyc`
  - `grpo/__pycache__/modeling_grpo.cpython-310.pyc`
  - `grpo/__pycache__/prompts.cpython-310.pyc`
  - `grpo/__pycache__/prompts.cpython-313.pyc`
  - `grpo/__pycache__/rewards.cpython-310.pyc`
  - `grpo/__pycache__/rewards.cpython-313.pyc`
- `desta25-audio` has 1 untracked entries listed. Examples needing review:
  - `test.py`
- `desta-grpo` has 113623 untracked entries listed. Examples needing review:
  - `ToolRL/`
  - `biased_verified_creation/create_desta_biased.py`
  - `biased_verified_creation/critic_eval_results.json`
  - `biased_verified_creation/critic_react_eval_results.json`
  - `biased_verified_creation/eval_af3.json`
  - `biased_verified_creation/eval_af3_easy.json`
  - `biased_verified_creation/eval_af3_hard.json`
  - `biased_verified_creation/eval_af3_tool_helps.json`
  - `biased_verified_creation/eval_caption_judge.py`
  - `biased_verified_creation/eval_caption_judge_results.json`
  - `biased_verified_creation/eval_critic.py`
  - `biased_verified_creation/eval_difficulty.py`
- `desta-grpo-toolrl` has 18 untracked entries listed. Examples needing review:
  - `audio_grpo/__init__.py`
  - `audio_grpo/audio_rollout.py`
  - `audio_grpo/configs/audio_grpo.yaml`
  - `audio_grpo/dataset.py`
  - `audio_grpo/judge.py`
  - `audio_grpo/main_train.py`
  - `audio_grpo/prepare_data.sh`
  - `audio_grpo/prompts.py`
  - `audio_grpo/rewards.py`
  - `audio_grpo/run_train.sh`
  - `audio_grpo/submit_e2e_test.sh`
  - `audio_grpo/submit_train.sh`
- `grpo-dataset-cache-audio-maestro` has 2 untracked entries listed. Examples needing review:
  - `audio_maestro/__pycache__/__init__.cpython-310.pyc`
  - `audio_maestro/__pycache__/audio_copilot.cpython-310.pyc`

No untracked datasets, checkpoints, logs, caches, model weights, or audio files were copied into the clean repository by the patch step; they are listed only in `untracked_files.txt`.
