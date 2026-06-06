# Extraction Inventory

Inventory-only extraction plan for selected folders in `/home/speech-nlp-cse/24m0756/abhishek`. This report was written to `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl/docs/extraction_inventory.md`. No source files were copied, moved, staged, committed, pushed, deleted, or rewritten.

## Scope

- Old workspace: `/home/speech-nlp-cse/24m0756/abhishek`
- Clean repo: `/home/speech-nlp-cse/24m0756/mtp2-audio-tool-rl`
- Inspected folders:
  - `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo`
  - `/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset`
  - `/home/speech-nlp-cse/24m0756/abhishek/rlTool`
  - `/home/speech-nlp-cse/24m0756/abhishek/toolRL`
  - `/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro`
  - `/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio`
  - `/home/speech-nlp-cse/24m0756/abhishek/test`
- Candidate decisions: copy `121`, review `111`, skip `74`.
- `copy` still means copy later after final human review; this task did not copy anything.

## Category Coverage

- GRPO training code: `56` candidate(s)
- reward code: `40` candidate(s)
- tool-calling/tool execution: `22` candidate(s)
- dataset construction/mapping: `42` candidate(s)
- prompts: `31` candidate(s)
- configs/environment: `7` candidate(s)
- model templates/wrappers: `34` candidate(s)
- inference/rollout: `14` candidate(s)
- evaluation/analysis: `21` candidate(s)
- run/Slurm scripts: `57` candidate(s)
- tests: `34` candidate(s)
- docs: `48` candidate(s)

## Candidate Files And Folders

|source path|size|file/folder type|inferred purpose|proposed destination|decision|reason|
|---|---|---|---|---|---|---|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/audio_maestro/audio_copilot.py|32.9 KB|.py|modified Audio-Maestro tool-calling/tool-execution code|src/mtp2_audio_tool_rl/tools/|copy|tracked nested-repo modification; preserve as source or patch|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo|5.52 MB in 19 files|folder|Audio-Maestro GRPO package|src/mtp2_audio_tool_rl/grpo_audio_maestro/|copy|project-specific GRPO variant; compare with Desta_grpo/grpo|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/__init__.py|0 B|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/dataset.py|7.35 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/diversity_sampler.py|12.9 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/model_wrapper.py|17.4 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/model_wrapper_qwen.py|15.0 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/modeling_grpo.py|26.7 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/modeling_qwen_grpo.py|4.58 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/prompts.py|10.3 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/system_monitor.py|4.59 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/train_grpo.py|12.3 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/trainer.py|42.9 KB|.py|Audio-Maestro GRPO training/model/reward file|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/extract_embeddings.py|8.52 KB|.py|Audio-Maestro helper/tool/embedding script|src/mtp2_audio_tool_rl/tools/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/prompts.py|10.3 KB|.py|modified Audio-Maestro tool-calling/tool-execution code|src/mtp2_audio_tool_rl/tools/|copy|tracked nested-repo modification; preserve as source or patch|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/test_embed_pipeline.py|11.3 KB|.py|Audio-Maestro helper/tool/embedding script|src/mtp2_audio_tool_rl/tools/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/analyze_tool_comparison.py|13.6 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/audio_maestro/audio_copilot.py|32.9 KB|.py|Audio-Maestro tool-calling code|src/mtp2_audio_tool_rl/tools/audio_copilot.py|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/demo_tool_mask.py|17.4 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm|173 KB in 16 files|folder|DeSTA/vLLM rollout and tool-execution package|src/mtp2_audio_tool_rl/desta_vllm/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/__init__.py|1.38 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/__main__.py|135 B|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/embed_utils.py|12.8 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/engine.py|24.7 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/grpo_rollout.py|6.68 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_all_checkpoints.py|12.6 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_benchmark.py|13.1 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/run_benchmark.sh|7.39 KB|.sh|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/tool_execute_vllm.py|24.2 KB|.py|vLLM inference/rollout/tool execution file|src/mtp2_audio_tool_rl/desta_vllm/ or scripts/inference/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/diag_embed.py|3.16 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/evaluation.py|4.11 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo|11.1 MB in 29 files|folder|canonical GRPO training package|src/mtp2_audio_tool_rl/grpo/|copy|central GRPO package; extract selectively and normalize imports|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/PIPELINE_DOCS.md|27.8 KB|.md|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/REWARD_DESIGN.md|2.74 KB|.md|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/__init__.py|0 B|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/configs|12.7 KB in 5 files|folder|GRPO config files|configs/training/grpo/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/dataset.py|7.01 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/model_wrapper.py|16.6 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/modeling_grpo.py|28.3 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/prompts.py|11.5 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/rewards_trl.py|11.5 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_debug.sh|988 B|.sh|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl.sh|3.02 KB|.sh|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl_v2.sh|2.89 KB|.sh|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_trl_v3.sh|2.53 KB|.sh|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/train_grpo.py|13.7 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/train_trl.py|38.0 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/trainer.py|43.0 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/trl_grpo_trainer_orig.py|125 KB|.py|GRPO training/reward/prompt/model/run file|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase|5.59 MB in 18 files|folder|single-phase GRPO package|src/mtp2_audio_tool_rl/grpo_single_phase/|copy|compare against canonical grpo/ before copying duplicates|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/REWARD_DESIGN.md|2.74 KB|.md|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/__init__.py|0 B|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/configs|12.7 KB in 5 files|folder|single-phase GRPO configs|configs/training/grpo_single_phase/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/dataset.py|7.87 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/model_wrapper.py|9.40 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/modeling_grpo.py|29.1 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/prompts.py|6.50 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/rewards_trl.py|11.2 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/run_trl_v3.sh|2.56 KB|.sh|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/train_trl.py|41.3 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/trl_grpo_trainer_orig.py|125 KB|.py|single-phase GRPO source/script/doc|src/mtp2_audio_tool_rl/grpo_single_phase/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled|174 KB in 15 files|folder|LLM-decoupled single-phase GRPO reward/training package|src/mtp2_audio_tool_rl/grpo_llm_decoupled/|copy|important decoupled reward/judge variant|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/__init__.py|0 B|.py|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/configs|2.38 KB in 1 files|folder|LLM-decoupled training configs|configs/training/grpo_llm_decoupled/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge.py|25.1 KB|.py|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/rewards_llm.py|9.99 KB|.py|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/rewards_rule.py|25.7 KB|.py|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/run_trl_v4.sh|3.52 KB|.sh|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/train_trl.py|8.46 KB|.py|LLM-decoupled GRPO reward/judge/training file|src/mtp2_audio_tool_rl/grpo_llm_decoupled/ or scripts/training/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_tool_empirical.py|14.8 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_tool_helpfulness.py|9.83 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/plot_comparison_table.py|5.16 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/plot_grpo_metrics.py|8.45 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/prompts.py|5.59 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/diag_embed_gpu.py|9.33 KB|.py|embedding diagnostic script|scripts/diagnostics/diag_embed_gpu.py|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/extract_embeddings.py|8.52 KB|.py|embedding extraction script|scripts/dataset/extract_embeddings.py|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/podman_gpu_test_job.sh|1.82 KB|.sh|run/diagnostic shell script|scripts/slurm/ or scripts/diagnostics/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/run_diag.sh|614 B|.sh|run/diagnostic shell script|scripts/slurm/ or scripts/diagnostics/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/runtime_matrix_gpu_test.sh|1.42 KB|.sh|run/diagnostic shell script|scripts/slurm/ or scripts/diagnostics/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/snapshot_expt.sh|4.25 KB|.sh|run/diagnostic shell script|scripts/slurm/ or scripts/diagnostics/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/test_embed_pipeline.py|11.3 KB|.py|embedding pipeline test|tests/test_embed_pipeline.py|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/check_audio.py|7.06 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/convert_to_train_format.py|2.53 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/extract_embeddings.py|9.54 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/merge_tool_outputs.py|2.15 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_audio.py|3.99 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_chord.py|3.54 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/process_stress.py|3.79 KB|.py|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_embed.sh|1023 B|.sh|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_extract.sh|2.33 KB|.sh|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/run_inf.sh|1.82 KB|.sh|synthetic dataset construction/conversion script|scripts/dataset/synthetic/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_bpe_boundary.py|7.97 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_chat_template_mask.py|15.4 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_gpt_oss.py|695 B|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen.py|3.14 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen2.py|3.11 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen3.py|2.99 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen_simple.py|670 B|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_qwen_simple2.py|456 B|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_tools.py|4.27 KB|.py|test/probe script|tests/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/verify_top3_tool_capablity.py|21.5 KB|.py|prompt/evaluation/analysis/diagnostic script|src/mtp2_audio_tool_rl/ or scripts/evaluation/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/download_models.py|1.32 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_audio.py|4.56 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_audio_wavecaps.py|4.33 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_chord.py|4.29 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_chord_wavecaps.py|3.78 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_stress.py|4.87 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/process_stress_wavecaps.py|3.93 KB|.py|audio/tool feature caching or preprocessing script|scripts/dataset/cache_tools/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/run_wavecaps_job.sh|1.51 KB|.sh|WaveCaps caching run script|scripts/slurm/run_wavecaps_job.sh|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/curate_samples.py|14.3 KB|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/extract_audio.py|1.46 KB|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering|144 MB in 15 files|folder|dataset filtering package/scripts|scripts/dataset/filtering/ or src/mtp2_audio_tool_rl/dataset/filtering/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/run_filter_job.sh|505 B|.sh|filtering Slurm/run script|scripts/slurm/run_filter_job.sh|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate.py|9.09 KB|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/prompts.py|4.17 KB|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen_job.sh|479 B|.sh|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/schema.py|885 B|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/stream_and_download_audioset.py|1.41 KB|.py|dataset construction/mapping/generation script|src/mtp2_audio_tool_rl/dataset/ or scripts/dataset/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/audioflammingo_generation_study.py|13.7 KB|.py|AudioFlamingo/MMaU prompt/inference script|scripts/inference/ or src/mtp2_audio_tool_rl/eval/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau_structured_infer.py|9.15 KB|.py|AudioFlamingo/MMaU prompt/inference script|scripts/inference/ or src/mtp2_audio_tool_rl/eval/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/prompts.py|6.50 KB|.py|AudioFlamingo/MMaU prompt/inference script|scripts/inference/ or src/mtp2_audio_tool_rl/eval/|copy|project-specific candidate|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro|28.4 MB in 166 files|folder|modified third-party Audio-Maestro clone|patches/Audio-Maestro/ and selected src/tools/|review|do not copy whole clone blindly; preserve tracked prompt/tool edits|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/README.md|3.30 KB|.md|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|scripts/ environment/ docs/ or tests/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/environment.yml|708 B|.yml|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/evaluation.py|4.11 KB|.py|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/extract_embeds.sh|731 B|.sh|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/extract_embeds_offline.sh|786 B|.sh|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/gpu_me.sh|6.63 KB|.sh|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/configs|7.20 KB in 3 files|folder|Audio-Maestro GRPO configs|configs/training/audio_maestro/|review|needs human review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/rewards.py|10.8 KB|.py|Audio-Maestro GRPO training/model/reward file (secret-risk)|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/run_grpo.sh|9.34 KB|.sh|Audio-Maestro GRPO training/model/reward file (secret-risk)|src/mtp2_audio_tool_rl/grpo_audio_maestro/ or scripts/training/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/grpo/splits|5.35 MB in 3 files|folder|Audio-Maestro GRPO split metadata|manifests/examples/audio_maestro/|review|review generated/full dataset status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/image/Results.png|missing/deleted|missing/deleted|deleted tracked image/documentation asset|patches/Audio-Maestro/ or docs/vendor/|review|tracked deletion; preserve patch context, do not recreate blindly|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/image/framework.png|missing/deleted|missing/deleted|deleted tracked image/documentation asset|patches/Audio-Maestro/ or docs/vendor/|review|tracked deletion; preserve patch context, do not recreate blindly|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/internet.sh|693 B|.sh|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|scripts/ environment/ docs/ or tests/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/requirements-clean.txt|3.40 KB|.txt|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/requirments.txt|8.26 KB|.txt|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/run.sh|3.55 KB|.sh|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/tool_execute.py|36.9 KB|.py|Audio-Maestro helper/tool/embedding script (secret-risk)|src/mtp2_audio_tool_rl/tools/ or scripts/dataset/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/tool_execute_gemini.py|11.9 KB|.py|modified Audio-Maestro tool-calling/tool-execution code (secret-risk)|src/mtp2_audio_tool_rl/tools/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test_error_cases.py|20.1 KB|.py|Audio-Maestro eval/test/run/env/doc candidate (secret-risk)|scripts/ environment/ docs/ or tests/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test_tools.py|4.27 KB|.py|Audio-Maestro eval/test/run/env/doc candidate|scripts/ environment/ docs/ or tests/|review|review machine-specific paths and third-party status|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio|164 KB in 31 files|folder|third-party DeSTA repo with local model edit|third_party/DeSTA2.5-Audio as fork/submodule or patches/|review|do not copy whole clone blindly|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/README.md|5.89 KB|.md|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta|78.4 KB in 8 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/models|28.5 KB in 1 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/models/modeling_desta25.py|28.5 KB|.py|modified DeSTA model code (secret-risk)|patches/DeSTA2.5-Audio/ or src/vendor/desta/models/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/trainer|24.7 KB in 3 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/utils|25.2 KB in 3 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs|16.2 KB in 3 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/dataset.md|11.0 KB|.md|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/evaluation_tips.md|2.78 KB|.md|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/docs/train.md|2.45 KB|.md|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples|27.0 KB in 10 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples/evaluation|16.3 KB in 4 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/examples/train|10.7 KB in 6 files|folder|DeSTA third-party source/examples/docs|third_party submodule/fork or docs/vendor/|review|review; prefer dependency/submodule unless project-specific|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/index.html|22.3 KB|.html|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/setup.py|585 B|.py|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/test.py|8.30 KB|.py|DeSTA doc/setup/test/reference file|docs/vendor/ or patches/DeSTA2.5-Audio/|review|third-party reference; copy only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/build_flash_attn.sh|2.38 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_forced_multi_vs_single.json|324 KB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/desta_vllm/training|69.9 KB in 7 files|folder|vLLM training helpers|src/mtp2_audio_tool_rl/desta_vllm/training/|review|review contents before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dummy_data.json|129 B|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/environment.yml|708 B|.yml|environment/config file|environment/ or configs/|review|review against root env.yaml/gemma.yml before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/extract_embeds.sh|749 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/extract_embeds_offline.sh|804 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fix_flash_attn.sh|1.81 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gemma.yml|10.6 KB|.yml|environment/config file|environment/ or configs/|review|review against root env.yaml/gemma.yml before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/getip.sh|1.62 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gpu_me.sh|6.63 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/rewards.py|10.8 KB|.py|GRPO training/reward/prompt/model/run file (secret-risk)|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/run_grpo2.sh|9.41 KB|.sh|GRPO training/reward/prompt/model/run file (secret-risk)|src/mtp2_audio_tool_rl/grpo/ or scripts/training/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits|5.35 MB in 3 files|folder|small split/manifest folder if tiny|manifests/examples/grpo/|review|review whether split files are small metadata or generated dataset content|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo/splits_v2|5.35 MB in 3 files|folder|small split/manifest folder if tiny|manifests/examples/grpo/|review|review whether split files are small metadata or generated dataset content|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase/splits_v2|5.35 MB in 3 files|folder|single-phase split metadata|manifests/examples/grpo_single_phase/|review|review size/content before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/benchmark|78.3 KB in 7 files|folder|benchmark/evaluation folder|scripts/evaluation/ or tests/benchmarks/|review|review for generated outputs before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/grpo_single_phase_llm_decoupled/judge copy.py|20.8 KB|.py|duplicate judge script|src/mtp2_audio_tool_rl/grpo_llm_decoupled/|review|copy-named duplicate; compare against judge.py|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/install_flash_attn.sh|1019 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/install_vllm.sh|940 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/interactive.sh|92 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/internet.sh|693 B|.sh|run/setup/diagnostic script (secret-risk)|scripts/slurm/ or scripts/setup/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/load_test.py|2.08 KB|.py|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini-cached-top3.json|5.45 MB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini-cached.json|5.29 MB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau-test-mini.json|577 KB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/mmau_af3_biased_eval_results.json|737 KB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/my_jobs.sh|3.01 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/run.sh|3.55 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/models|35.1 KB in 7 files|folder|local script model templates/configs|configs/models/ or src/mtp2_audio_tool_rl/models/|review|review for generated weights before copying|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/select_top_tools.py|7.40 KB|.py|tool selection/ranking helper (secret-risk)|scripts/tools/select_top_tools.py|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/scripts/tool_execute.py|37.4 KB|.py|tool execution code (secret-risk)|src/mtp2_audio_tool_rl/tools/tool_execute.py|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset|77.7 MB in 77 files|folder|synthetic dataset construction scripts mixed with generated outputs|scripts/dataset/synthetic/|review|copy only scripts, not JSON outputs/logs/embeds|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/launch_jupyter_gpu.sh|1.18 KB|.sh|synthetic dataset construction/conversion script (secret-risk)|scripts/dataset/synthetic/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_a40.sh|493 B|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_error_cases.py|20.1 KB|.py|test/probe script (secret-risk)|tests/|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/tool_empirical_results.json|3.88 MB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/tool_helpfulness_results.json|1.14 MB|.json|small manifest/result JSON candidate|manifests/examples/ only if tiny and necessary|review|likely generated/cached result; do not copy full datasets|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/vllm_install.sh|1.55 KB|.sh|run/setup/diagnostic script|scripts/slurm/ or scripts/setup/|review|review for machine-specific paths and temporary usage|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_cached.jsonl|724 KB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_chord.jsonl|227 KB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_extended.jsonl|603 KB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/wavecaps_stress.jsonl|687 KB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/desta_dataset.txt|70.3 KB|.txt|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/desta_dataset_stats.tx|1.29 KB|.tx|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtered_curated_candidates.jsonl|583 KB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/curated_tool_candidates.jsonl|10.1 MB|.jsonl|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/filter_dataset.py|15.2 KB|.py|dataset filtering script (secret-risk)|scripts/dataset/filtering/filter_dataset.py|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/mmau-test-mini-cached.json|5.29 MB|.json|cached mini manifest|manifests/examples/|review|review as tiny example only|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/google_client.py|2.05 KB|.py|LLM API client helper; secret-risk (secret-risk)|src/mtp2_audio_tool_rl/dataset/clients/ or recreate from template|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/mmau-test-mini-cached.json|5.29 MB|.json|dataset manifest/stat file|manifests/examples/ or docs/dataset/|review|review size and generated/full-dataset status before copying|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/openai_client.py|2.10 KB|.py|LLM API client helper; secret-risk (secret-risk)|src/mtp2_audio_tool_rl/dataset/clients/ or recreate from template|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/wavecaps/wavecaps_curate_and_download.py|27.3 KB|.py|WaveCaps curation/download script (secret-risk)|scripts/dataset/wavecaps_curate_and_download.py|review|secret-risk: filename or small text scan suggests credentials/API/token/password; do not print values; sanitize or recreate from template|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/biased_tool_verified.json|1.90 MB|.json|MMaU cached/result JSON|manifests/examples/ only if tiny|review|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-structured-results-100.json|542 KB|.json|MMaU cached/result JSON|manifests/examples/ only if tiny|review|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-structured-results.json|5.99 MB|.json|MMaU cached/result JSON|manifests/examples/ only if tiny|review|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-test-mini-cached-100.json|331 KB|.json|MMaU cached/result JSON|manifests/examples/ only if tiny|review|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/mmau-test-mini-cached.json|5.29 MB|.json|MMaU cached/result JSON|manifests/examples/ only if tiny|review|likely dataset/result artifact; review before copying|
|/home/speech-nlp-cse/24m0756/abhishek/test|0 B in 0 files|folder|top-level test folder|tests/|review|folder appeared empty in max-depth scan; keep as review note|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3|70.9 MB in 3000+ files|folder|AF3 project/source copy mixed with outputs/datasets/checkpoints|src/mtp2_audio_tool_rl/af3_toolrl/ selectively|review|review selected source only; skip generated dirs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/README.md|5.72 KB|.md|AF3 evaluation/test/doc file|scripts/evaluation/ or tests/|review|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/evaluation.py|12.7 KB|.py|AF3 evaluation/test/doc file|scripts/evaluation/ or tests/|review|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/rewards|8.49 KB in 2 files|folder|AF3 scripts/reward folder|scripts/ or src/mtp2_audio_tool_rl/rewards/|review|review contents; skip datasets/results/checkpoints|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/scripts|41.1 KB in 8 files|folder|AF3 scripts/reward folder|scripts/ or src/mtp2_audio_tool_rl/rewards/|review|review contents; skip datasets/results/checkpoints|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/test_parquet_chat.py|3.50 KB|.py|AF3 evaluation/test/doc file|scripts/evaluation/ or tests/|review|review duplicate with AF3 copy and generated outputs|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL|28.5 MB in 205 files|folder|third-party ToolRL clone|third_party/ToolRL as submodule/fork or patches/|review|do not copy whole clone blindly|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/README.md|3.04 KB|.md|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/audio_grpo|missing/deleted|missing/deleted|ToolRL audio GRPO integration folder if present|src/mtp2_audio_tool_rl/toolrl_audio_grpo/ or patches/ToolRL/|review|review because toolRL/ToolRL had no tracked diff but may be dependency code|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/pyproject.toml|2.17 KB|.toml|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/requirements.txt|135 B|.txt|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/setup.py|1.87 KB|.py|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/train_grpo.sh|561 B|.sh|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/train_ppo.sh|557 B|.sh|ToolRL setup/run/doc file|third_party patches or docs/vendor/|review|third-party repo content; preserve only if needed|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/audio_maestro/__pycache__|20.5 KB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/checkpoints|204 MB in 714 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/logs|59.8 MB in 126 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/old_logs|1.23 MB in 54 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/precomputed_embeds|1002 MB in 1000 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/results|130 MB in 99 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/scripts/__pycache__|84.1 KB in 4 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/test-mini-audios|1.70 GB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro/wandb|0 B in 0 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/.git|3.79 MB in 36 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/assets/audios|3.21 MB in 9 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta.egg-info|876 B in 5 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio/desta/__pycache__|776 B in 3 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/checkpoints|19.2 GB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/cuda12.sif|85.3 MB|.sif|artifact/data/cache/log family|do not copy|skip|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dataset.tar|186 MB|.tar|artifact/data/cache/log family|do not copy|skip|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/dataset/audio|187 MB in 1941 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/demo_output.txt|24.6 KB|.txt|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fake_group|0 B|file|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/fake_passwd|0 B|file|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/forced_dir|438 KB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/gemma4.sif|8.25 GB|.sif|artifact/data/cache/log family|do not copy|skip|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge-867777.tar|316 MB|.tar|artifact/data/cache/log family|do not copy|skip|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/judge_logs|136 MB in 22 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/logs|249 MB in 580 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/logs2|45.5 MB in 66 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/old_grpo_logs|49.0 MB in 53 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/precomputed_embeds|1002 MB in 1000 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/results|1.76 GB in 1287 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/selectio_tools.log|6.76 MB|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/logs|3.40 MB in 39 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/precomputed_embeds|490 MB in 489 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/synthetic_dataset/tool_findings_results|38.2 MB in 24 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test-mini-audios|1.70 GB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/test_a40.log|488 B|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/training_logs|5.63 MB in 13 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/wandb|3.33 MB in 3 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/__pycache__|71.1 KB in 16 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/audioset|974 MB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/audioset_zips|67.3 GB in 23 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro|1.19 MB in 8 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/curated_stats.log|2.03 KB|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/downloading.log|732 KB|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/filtering/__pycache__|9.37 KB in 1 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/final_audio|180 MB in 599 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate_job_94004.err|380 KB|.err|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/generate_job_94004.out|41 B|.out|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/lid.176.bin|125 MB|.bin|artifact/data/cache/log family|do not copy|skip|blocked extension/raw audio/model/cache/archive policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output|2.05 MB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output00|16.2 MB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output01|7.72 MB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output02|14.5 MB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output03|1.86 MB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen.log|482 KB|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/run_gen_2.log|190 KB|.log|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/wavecaps/wavecaps_freesound|162 MB in 6 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/__pycache__|13.7 KB in 2 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/audioflamingo3_logs|424 KB in 5 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/desta_vllm_model|0 B in 0 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/rlTool/test-mini-audios|1.69 GB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3 copy|3.80 GB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/checkpoints|147 GB in 68 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/dataset|6.58 MB in 6 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/flash_attn_pkg|8.06 MB in 1 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/logs|3.32 MB in 48 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/outputs|813 KB in 100 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/results|6.06 MB in 1 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/sandbox|56.2 MB in 3000+ files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/AF3/wandb|821 KB in 22 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/.git|9.88 MB in 26 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/checkpoints|missing/deleted|missing/deleted|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/dataset|23.4 MB in 5 files|folder|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL/outputs|missing/deleted|missing/deleted|artifact/data/cache/log family|do not copy|skip|always skip by policy|
|/home/speech-nlp-cse/24m0756/abhishek/toolRL/toolrl_paper.pdf|2.36 MB|.pdf|large/reference paper PDF|do not copy|skip|skip unless separately cited in docs|

## Recommended Extraction Order

1. Start with the canonical GRPO package: `Desta_grpo/grpo`. Then compare `grpo_single_phase` and `grpo_single_phase_llm_decoupled` and copy only non-duplicative variants.
2. Extract shared reward, prompt, dataset-loader, model-wrapper, and trainer code into `src/mtp2_audio_tool_rl/`; put launch scripts in `scripts/training/` and configs in `configs/training/`.
3. Extract tool-calling and tool-execution code next: `Desta_grpo/desta_vllm`, `Desta_grpo/audio_maestro`, and the modified Audio-Maestro prompt/tool files.
4. Extract dataset construction scripts from `grpo_dataset`, `filtering`, `wavecaps`, and selected `cache_tools` scripts. Do not copy full generated JSONL outputs, raw/copied audio, language-ID binaries, logs, or cache folders.
5. Preserve third-party modifications as patches or forks before vendoring anything: `DeSTA2.5-Audio/desta/models/modeling_desta25.py`, Audio-Maestro tracked edits, and `Desta_grpo/ToolRL` tokenizer/FSDP edits.
6. Bring over evaluation/inference scripts and tests after the source layout is stable, then update imports and paths.
7. Review every `review` and `secret-risk` entry manually. Recreate credential-bearing configs from templates rather than copying live values.
