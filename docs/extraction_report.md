# Extraction Report

Safe copy pass from `/home/speech-nlp-cse/24m0756/abhishek` into `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl`. Only inventory entries marked `copy` were considered, and safety rules were re-applied. No git init/stage/commit/push was run.

## Summary

- Copied file count: `141`
- Skipped file count during copy pass: `25`
- Inventory `review` entries not copied: `111`
- Secret-risk files not copied: `5`
- Files above 10 MB not copied: `4`
- Copy failures: `0`
- Policy issues found during copy pass: `2`
- Policy issues resolved during cleanup pass: `2`

## Policy Issue Cleanup

Cleanup was performed only inside the clean repo during the patch-preservation pass. The source workspace was not modified.

|issue|source/destination|cleanup action|
|---|---|---|
|A review-marked duplicate file was copied through folder expansion.|source `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge copy.py`; original destination `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge copy.py`|Compared against `judge.py`; it contains meaningful judge-prompt/scoring/default differences, so it was kept as `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge_variant.py`.|
|A source `__init__.py` collided with the empty package initializer created for `desta_vllm`, so the copy pass preserved it under a generated collision name.|original destination `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/__init____Desta_grpo__desta_vllm____init___py.py`|Merged the real package exports into `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/__init__.py` and removed the generated collision file from the clean repo.|

## Source To Destination Mapping

|source|destination|
|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/audio_maestro/audio_copilot.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/tool_calling/audio_maestro_audio_copilot.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/__init__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/train_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/train_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/diversity_sampler.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/diversity_sampler.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/model_wrapper_qwen.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/model_wrapper_qwen.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/trainer.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/trainer.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/dataset.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/dataset.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/system_monitor.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/system_monitor.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/modeling_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/modeling_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/model_wrapper.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/model_wrapper.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/modeling_qwen_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/modeling_qwen_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/configs/optimized.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/audio_maestro/optimized.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/configs/default.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/audio_maestro/default.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/configs/qwen_omni.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/audio_maestro/qwen_omni.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/extract_embeddings.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/audio_maestro_extract_embeddings.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/prompts/audio_maestro_prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/test_embed_pipeline.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/audio_maestro_test_embed_pipeline.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/analyze_tool_comparison.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/analyze_tool_comparison.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/audio_maestro/audio_copilot.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/tool_calling/desta_audio_copilot.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/demo_tool_mask.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/demo_tool_mask.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_benchmark.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/inference/desta_vllm/run_benchmark.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/__init____Desta_grpo__desta_vllm____init___py.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/embed_utils.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/embed_utils.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/tool_execute_vllm.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/tool_execute_vllm.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/grpo_rollout.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/grpo_rollout.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_all_checkpoints.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/run_all_checkpoints.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_benchmark.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/run_benchmark.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/engine.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/engine.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/__main__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/__main__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/trainer.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/trainer.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/__init__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/rewards_llm.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/rewards_llm.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/train_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/train_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/run_vllm_grpo.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/inference/desta_vllm/run_vllm_grpo.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/rewards_rule.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/rewards_rule.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training/configs/vllm_grpo_v1.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/training/configs/vllm_grpo_v1.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/diag_embed.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/diag_embed.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/evaluation.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/evaluation.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo/run_trl.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/__init__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/train_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/train_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/trainer.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/trainer.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/dataset.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/dataset.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_debug.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo/run_debug.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl_v3.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo/run_trl_v3.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/modeling_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/modeling_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/train_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/train_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/REWARD_DESIGN.md|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/grpo/REWARD_DESIGN.md|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl_v2.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo/run_trl_v2.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/trl_grpo_trainer_orig.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/trl_grpo_trainer_orig.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/PIPELINE_DOCS.md|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/grpo/PIPELINE_DOCS.md|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/model_wrapper.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/model_wrapper.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/rewards_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/rewards_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs/optimized.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/optimized.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs/optimized_v3.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/optimized_v3.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs/default.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/default.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs/qwen_omni.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/qwen_omni.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs/optimized_v2.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/optimized_v2.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/__init__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/dataset.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/dataset.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/run_trl_v3.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo_single_phase/run_trl_v3.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/modeling_grpo.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/modeling_grpo.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/train_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/train_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/REWARD_DESIGN.md|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/grpo_single_phase/REWARD_DESIGN.md|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/trl_grpo_trainer_orig.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/trl_grpo_trainer_orig.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/model_wrapper.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/model_wrapper.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/rewards_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/rewards_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs/optimized.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/single_phase/optimized.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs/optimized_v3.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/single_phase/optimized_v3.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs/default.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/single_phase/default.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs/qwen_omni.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/single_phase/qwen_omni.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs/optimized_v2.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/single_phase/optimized_v2.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/rewards_rule.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/rewards_rule.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/__init__.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge copy.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge copy.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/judge.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/train_trl.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/train_trl.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/run_trl_v4.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo_llm_decoupled/run_trl_v4.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/rewards_llm.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/rewards_llm.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/configs/optimized_v4.yaml|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/configs/grpo/llm_decoupled/optimized_v4.yaml|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/test_vllm_small.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo_llm_decoupled/test_vllm_small.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/bench_grpo_judge.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/benchmark/bench_grpo_judge.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/bench_judge.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/benchmark/bench_judge.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/judge.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo_llm_decoupled/judge.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/bench_judge.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/train/grpo_llm_decoupled/bench_judge.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/bench_throughput.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/benchmark/bench_throughput.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark/bench_gpu_staggered.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/benchmark/bench_gpu_staggered.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_tool_empirical.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/judge_tool_empirical.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_tool_helpfulness.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/judge_tool_helpfulness.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/plot_comparison_table.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/plot_comparison_table.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/plot_grpo_metrics.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/plot_grpo_metrics.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/prompts/desta_prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/diag_embed_gpu.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/diag_embed_gpu.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/extract_embeddings.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/extract_embeddings.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/podman_gpu_test_job.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/podman_gpu_test_job.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/run_diag.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/run_diag.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/runtime_matrix_gpu_test.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/diagnostics/runtime_matrix_gpu_test.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/snapshot_expt.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/slurm/snapshot_expt.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/test_embed_pipeline.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/desta_test_embed_pipeline.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/check_audio.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/check_audio.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/convert_to_train_format.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/convert_to_train_format.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/extract_embeddings.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/extract_embeddings.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/merge_tool_outputs.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/merge_tool_outputs.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_audio.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/process_audio.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_chord.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/process_chord.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_stress.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/process_stress.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_embed.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/run_embed.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_extract.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/run_extract.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_inf.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/synthetic/run_inf.sh|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_bpe_boundary.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_bpe_boundary.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_chat_template_mask.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_chat_template_mask.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_gpt_oss.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_gpt_oss.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_qwen.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen2.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_qwen2.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen3.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_qwen3.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen_simple.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_qwen_simple.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen_simple2.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_qwen_simple2.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_tools.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/tests/test_tools.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/verify_top3_tool_capablity.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/evaluation/verify_top3_tool_capablity.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/download_models.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/download_models.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_audio.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_audio.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_audio_wavecaps.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_audio_wavecaps.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_chord.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_chord.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_chord_wavecaps.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_chord_wavecaps.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_stress.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_stress.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_stress_wavecaps.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/cache_tools/process_stress_wavecaps.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/run_wavecaps_job.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/slurm/run_wavecaps_job.sh|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/curate_samples.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/curate_samples.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/extract_audio.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/extract_audio.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/run_filter_job.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/slurm/run_filter_job.sh|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/generate.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/grpo_dataset_prompts.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen_job.sh|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/data/run_gen_job.sh|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/schema.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/schema.py|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/stream_and_download_audioset.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/datasets/stream_and_download_audioset.py|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/audioflammingo_generation_study.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/inference/rltool_audioflammingo_generation_study.py|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau_structured_infer.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/scripts/inference/rltool_mmau_structured_infer.py|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/prompts.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/prompts/rltool_prompts.py|

## Needs Human Review

|source|purpose|reason|
|---|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro|modified third-party Audio-Maestro clone|do not copy whole clone blindly; preserve tracked prompt/tool edits|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/README.md|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/environment.yml|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/evaluation.py|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/extract_embeds.sh|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/extract_embeds_offline.sh|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/gpu_me.sh|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/configs|Audio-Maestro GRPO configs|needs human review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/rewards.py|Audio-Maestro GRPO training/model/reward file (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/run_grpo.sh|Audio-Maestro GRPO training/model/reward file (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/splits|Audio-Maestro GRPO split metadata|review generated/full dataset status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/image/Results.png|deleted tracked image/documentation asset|tracked deletion; preserve patch context, do not recreate blindly|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/image/framework.png|deleted tracked image/documentation asset|tracked deletion; preserve patch context, do not recreate blindly|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/internet.sh|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/requirements-clean.txt|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/requirments.txt|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/run.sh|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/tool_execute.py|Audio-Maestro helper/tool/embedding script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/tool_execute_gemini.py|modified Audio-Maestro tool-calling/tool-execution code (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test_error_cases.py|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test_tools.py|Audio-Maestro eval/test/run/env/doc candidate|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio|third-party DeSTA repo with local model edit|do not copy whole clone blindly|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/README.md|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/models|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/models/modeling_desta25.py|modified DeSTA model code (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/trainer|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/utils|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/dataset.md|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/evaluation_tips.md|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/train.md|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples/evaluation|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples/train|DeSTA third-party source/examples/docs|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/index.html|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/setup.py|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/test.py|DeSTA doc/setup/test/reference file|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/build_flash_attn.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_forced_multi_vs_single.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training|vLLM training helpers|review contents before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dummy_data.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/environment.yml|environment/config file|review against root env.yaml/gemma.yml before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/extract_embeds.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/extract_embeds_offline.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fix_flash_attn.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gemma.yml|environment/config file|review against root env.yaml/gemma.yml before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/getip.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gpu_me.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/rewards.py|GRPO training/reward/prompt/model/run file (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_grpo2.sh|GRPO training/reward/prompt/model/run file (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits|small split/manifest folder if tiny|review whether split files are small metadata or generated dataset content|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits_v2|small split/manifest folder if tiny|review whether split files are small metadata or generated dataset content|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/splits_v2|single-phase split metadata|review size/content before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark|benchmark/evaluation folder|review for generated outputs before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge copy.py|duplicate judge script|copy-named duplicate; compare against judge.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/install_flash_attn.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/install_vllm.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/interactive.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/internet.sh|run/setup/diagnostic script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/load_test.py|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini-cached-top3.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini-cached.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau_af3_biased_eval_results.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/my_jobs.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/run.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/models|local script model templates/configs|review for generated weights before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/select_top_tools.py|tool selection/ranking helper (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/tool_execute.py|tool execution code (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset|synthetic dataset construction scripts mixed with generated outputs|copy only scripts, not JSON outputs/logs/embeds|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/launch_jupyter_gpu.sh|synthetic dataset construction/conversion script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_a40.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_error_cases.py|test/probe script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/tool_empirical_results.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/tool_helpfulness_results.json|small manifest/result JSON candidate|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/vllm_install.sh|run/setup/diagnostic script|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_cached.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_chord.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_extended.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_stress.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/desta_dataset.txt|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/desta_dataset_stats.tx|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtered_curated_candidates.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/curated_tool_candidates.jsonl|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filter_dataset.py|dataset filtering script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/mmau-test-mini-cached.json|cached mini manifest|review as tiny example only|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/google_client.py|LLM API client helper; secret-risk (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/mmau-test-mini-cached.json|dataset manifest/stat file|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/openai_client.py|LLM API client helper; secret-risk (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/wavecaps/wavecaps_curate_and_download.py|WaveCaps curation/download script (secret-risk)|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/biased_tool_verified.json|MMaU cached/result JSON|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-structured-results-100.json|MMaU cached/result JSON|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-structured-results.json|MMaU cached/result JSON|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-test-mini-cached-100.json|MMaU cached/result JSON|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-test-mini-cached.json|MMaU cached/result JSON|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/test|top-level test folder|folder appeared empty in max-depth scan; keep as review note|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3|AF3 project/source copy mixed with outputs/datasets/checkpoints|review selected source only; skip generated dirs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/README.md|AF3 evaluation/test/doc file|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/evaluation.py|AF3 evaluation/test/doc file|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/rewards|AF3 scripts/reward folder|review contents; skip datasets/results/checkpoints|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/scripts|AF3 scripts/reward folder|review contents; skip datasets/results/checkpoints|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/test_parquet_chat.py|AF3 evaluation/test/doc file|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL|third-party ToolRL clone|do not copy whole clone blindly|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/README.md|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/audio_grpo|ToolRL audio GRPO integration folder if present|review because toolRL/ToolRL had no tracked diff but may be dependency code|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/pyproject.toml|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/requirements.txt|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/setup.py|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/train_grpo.sh|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/train_ppo.sh|ToolRL setup/run/doc file|third-party repo content; preserve only if needed|

## Secret-risk Files Not Copied

|source|risk|
|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/rewards.py|secret-risk detected by filename or small text scan|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/run_grpo.sh|secret-risk detected by filename or small text scan|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/rewards.py|secret-risk detected by filename or small text scan|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_grpo2.sh|secret-risk detected by filename or small text scan|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filter_dataset.py|secret-risk detected by filename or small text scan|

## Files Above 10 MB Not Copied

|source|size|reason|
|---|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/curated_tool_candidates.jsonl|10.1 MB|above 10 MB|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gpt_oss/filtering_llm_trace.log|71.6 MB|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gemma/filtering_llm_trace.log|24.9 MB|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering/filtering_llm_trace.log|30.0 MB|blocked extension|

## Duplicate-name Handling Decisions

|source|destination|decision|
|---|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_audio_maestro/__init__.py|identical duplicate already copied|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/desta_vllm/__init____Desta_grpo__desta_vllm____init___py.py|renamed to avoid destination collision|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo/__init__.py|identical duplicate already copied|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_single_phase/__init__.py|identical duplicate already copied|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/__init__.py|/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/src/mtp2_audio_tool_rl/grpo_llm_decoupled/__init__.py|identical duplicate already copied|

## Folders Where Only Selected Files Were Copied

|source folder|candidate files seen|
|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo|19|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm|16|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo|29|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs|5|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase|18|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs|5|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled|15|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/configs|1|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering|15|

## Skipped During Copy Pass

|source|reason|
|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/splits/train.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/splits/test.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/splits/eval.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits/train.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits/test.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits/eval.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits_v2/train.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits_v2/test.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits_v2/eval.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/splits_v2/train.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/splits_v2/test.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/splits_v2/eval.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/curated_tool_candidates.jsonl|above 10 MB|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/mmau-test-mini-cached.json|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gpt_oss/filtering_llm_trace.log|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gpt_oss/filtered_curated_candidates.jsonl|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gpt_oss/filtered_curated_candidates copy.jsonl|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gemma/job_100248.out|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gemma/job_100248.err|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gemma/filtering_llm_trace.log|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering_gemma/filtered_curated_candidates.jsonl|json/jsonl not clearly config/schema/example manifest|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering/job_100266.err|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering/job_100266.out|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering/filtering_llm_trace.log|blocked extension|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filtering/filtered_curated_candidates.jsonl|json/jsonl not clearly config/schema/example manifest|

## Inventory Skip Entries Not Copied

|source|reason|
|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/audio_maestro/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/checkpoints|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/old_logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/precomputed_embeds|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/results|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test-mini-audios|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/wandb|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/.git|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/assets/audios|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta.egg-info|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/checkpoints|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/cuda12.sif|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dataset.tar|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dataset/audio|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/demo_output.txt|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fake_group|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fake_passwd|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/forced_dir|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gemma4.sif|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge-867777.tar|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/logs2|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/old_grpo_logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/precomputed_embeds|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/results|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/selectio_tools.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/precomputed_embeds|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/tool_findings_results|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test-mini-audios|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_a40.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/training_logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/wandb|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/audioset|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/audioset_zips|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/curated_stats.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/downloading.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/final_audio|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate_job_94004.err|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate_job_94004.out|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/lid.176.bin|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output00|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output01|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output02|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output03|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen_2.log|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/wavecaps/wavecaps_freesound|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/__pycache__|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/audioflamingo3_logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/desta_vllm_model|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/test-mini-audios|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3 copy|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/checkpoints|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/dataset|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/flash_attn_pkg|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/logs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/outputs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/results|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/sandbox|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/wandb|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/.git|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/checkpoints|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/dataset|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/outputs|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/toolrl_paper.pdf|skip unless separately cited in docs|

## Copy Failures

None.
