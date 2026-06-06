# Modified Nested Repositories Audit

This report locates nested `.git` repositories and summarizes local modifications without printing full diffs. Untracked files are intentionally collapsed by directory and sampled because several repos contain huge generated/data trees.

## `Audio-Maestro`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro`
- Remote origin URL: `https://github.com/gary920209/Audio-Maestro.git`
- All remotes:
```text
origin	https://github.com/gary920209/Audio-Maestro.git (fetch)
origin	https://github.com/gary920209/Audio-Maestro.git (push)
```
- Current branch: `main`
- Current commit hash: `ba443a215573ff2d5b85e8efbec268343b40baa2`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `3`
- Deleted tracked files: `2`
- Untracked files/folders visible after `--directory` collapse: `64`
- Ignored-looking large/generated untracked entries in sample: `40`

`git status --short --untracked-files=no`:
```text
M audio_maestro/audio_copilot.py
 D image/Results.png
 D image/framework.png
 M scripts/prompts.py
 M scripts/tool_execute_gemini.py
```

`git diff --stat`:
```text
audio_maestro/audio_copilot.py | 434 +++++++++++++++++++++++------------------
 image/Results.png              | Bin 974324 -> 0 bytes
 image/framework.png            | Bin 224071 -> 0 bytes
 scripts/prompts.py             | 278 ++++++++++++++++++++++----
 scripts/tool_execute_gemini.py |  93 +++++----
 5 files changed, 526 insertions(+), 279 deletions(-)
```

`git diff --name-status`:
```text
M	audio_maestro/audio_copilot.py
D	image/Results.png
D	image/framework.png
M	scripts/prompts.py
M	scripts/tool_execute_gemini.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|audio_maestro/audio_copilot.py|modified|my new code|yes, likely important to preserve|patch file|
|image/Results.png|deleted|result|no for Git; preserve outside Git/manifest if useful|ignored/manifest only|
|image/framework.png|deleted|unknown|yes, likely important to preserve|patch file|
|scripts/prompts.py|modified|my new code|yes, likely important to preserve|patch file|
|scripts/tool_execute_gemini.py|modified|my new code|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|.gitignore|16 B|file|unknown|human review|
|audio_maestro/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|checkpoints/|directory; size not expanded here|dir|checkpoint|ignore|
|evaluation.py|4.11 KB|.py|evaluation script|move to main repo or keep as patch/addition|
|extract_embeds.sh|731 B|.sh|my new code|move to main repo or keep as patch/addition|
|extract_embeds_73163.err|721 B|.err|log|ignore|
|extract_embeds_73163.log|419 B|.log|log|ignore|
|extract_embeds_73167.err|41.0 KB|.err|log|ignore|
|extract_embeds_73167.log|3.57 MB|.log|log|ignore|
|extract_embeds_73173.err|72.6 KB|.err|log|ignore|
|extract_embeds_73173.log|89.3 KB|.log|log|ignore|
|extract_embeds_offline.sh|786 B|.sh|my new code|move to main repo or keep as patch/addition|
|gpu_me.sh|6.63 KB|.sh|my new code|move to main repo or keep as patch/addition|
|grpo/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo2/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_slurm_73089.log|26.1 KB|.log|log|ignore|
|grpo_slurm_73109.log|315 KB|.log|log|ignore|
|grpo_slurm_73138.log|22.1 KB|.log|log|ignore|
|grpo_slurm_73178.log|15.6 KB|.log|log|ignore|
|grpo_slurm_73221.log|13.6 KB|.log|log|ignore|
|grpo_slurm_73231.log|82 B|.log|log|ignore|
|grpo_slurm_73233.log|26.6 KB|.log|log|ignore|
|grpo_slurm_73287.log|8.80 KB|.log|log|ignore|
|grpo_slurm_73291.log|17.1 KB|.log|log|ignore|
|grpo_slurm_73309.log|17.3 KB|.log|log|ignore|
|grpo_slurm_73459.log|17.6 KB|.log|log|ignore|
|grpo_slurm_73480.log|17.6 KB|.log|log|ignore|
|grpo_slurm_73595.log|24.6 KB|.log|log|ignore|
|grpo_slurm_73761.log|25.4 KB|.log|log|ignore|
|grpo_slurm_73988.log|519 KB|.log|log|ignore|
|grpo_slurm_74204.log|10.1 KB|.log|log|ignore|
|grpo_slurm_74211.log|13.6 KB|.log|log|ignore|
|grpo_slurm_74215.log|17.7 KB|.log|log|ignore|
|grpo_slurm_74218.log|550 KB|.log|log|ignore|
|inference/|directory; size not expanded here|dir|inference script|move to main repo or keep as patch/addition|
|inference_parallel_slurm_74854.log|633 KB|.log|log|ignore|
|inference_parallel_slurm_74864.log|633 KB|.log|log|ignore|
|inference_parallel_slurm_74867.log|790 KB|.log|log|ignore|
|inference_parallel_slurm_74976.log|385 KB|.log|log|ignore|
|inference_parallel_slurm_74991.log|480 KB|.log|log|ignore|
|inference_run_slurm.log|82 B|.log|log|ignore|
|inference_run_slurm_74656.log|35.8 KB|.log|log|ignore|
|inference_run_slurm_74664.log|554 KB|.log|log|ignore|
|inference_run_slurm_74809.log|390 KB|.log|log|ignore|
|inference_run_slurm_74829.log|774 KB|.log|log|ignore|
|internet.sh|693 B|.sh|my new code|move to main repo or keep as patch/addition|
|logs/|directory; size not expanded here|dir|unknown|human review|
|mmau-test-mini-cached.json|5.29 MB|.json|config change|move to main repo or keep as patch/addition|
|mmau-test-mini.json|577 KB|.json|config change|move to main repo or keep as patch/addition|
|old_logs/|directory; size not expanded here|dir|unknown|human review|
|precomputed_embeds/|directory; size not expanded here|dir|unknown|human review|
|requirements-clean.txt|3.40 KB|.txt|environment change|move to main repo or keep as patch/addition|
|requirments.txt|8.26 KB|.txt|unknown|human review|
|results/|directory; size not expanded here|dir|result|ignore|
|run.sh|3.55 KB|.sh|my new code|move to main repo or keep as patch/addition|
|scripts/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|scripts/agentic/|directory; size not expanded here|dir|unknown|human review|
|scripts/extract_embeddings.py|8.52 KB|.py|my new code|move to main repo or keep as patch/addition|
|scripts/models/|directory; size not expanded here|dir|unknown|human review|
|scripts/test_embed_pipeline.py|11.3 KB|.py|temporary/debug|human review or ignore|
|scripts/tool_execute.py|36.9 KB|.py|my new code|move to main repo or keep as patch/addition|
|test_error_cases.py|20.1 KB|.py|temporary/debug|human review or ignore|
|test_tools.py|4.27 KB|.py|temporary/debug|human review or ignore|
|wandb/|directory; size not expanded here|dir|result|ignore|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/Audio-Maestro
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro diff > patches/Audio-Maestro/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro diff --name-status > patches/Audio-Maestro/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro ls-files --others --exclude-standard > patches/Audio-Maestro/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `Audio-Maestro-bak`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro-bak`
- Remote origin URL: `https://github.com/gary920209/Audio-Maestro.git`
- All remotes:
```text
origin	https://github.com/gary920209/Audio-Maestro.git (fetch)
origin	https://github.com/gary920209/Audio-Maestro.git (push)
```
- Current branch: `main`
- Current commit hash: `ba443a215573ff2d5b85e8efbec268343b40baa2`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `3`
- Deleted tracked files: `2`
- Untracked files/folders visible after `--directory` collapse: `34`
- Ignored-looking large/generated untracked entries in sample: `18`

`git status --short --untracked-files=no`:
```text
M audio_maestro/audio_copilot.py
 D image/Results.png
 D image/framework.png
 M scripts/prompts.py
 M scripts/tool_execute_gemini.py
```

`git diff --stat`:
```text
audio_maestro/audio_copilot.py | 434 +++++++++++++++++++++++------------------
 image/Results.png              | Bin 974324 -> 0 bytes
 image/framework.png            | Bin 224071 -> 0 bytes
 scripts/prompts.py             | 278 ++++++++++++++++++++++----
 scripts/tool_execute_gemini.py |  93 +++++----
 5 files changed, 526 insertions(+), 279 deletions(-)
```

`git diff --name-status`:
```text
M	audio_maestro/audio_copilot.py
D	image/Results.png
D	image/framework.png
M	scripts/prompts.py
M	scripts/tool_execute_gemini.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|audio_maestro/audio_copilot.py|modified|my new code|yes, likely important to preserve|patch file|
|image/Results.png|deleted|result|no for Git; preserve outside Git/manifest if useful|ignored/manifest only|
|image/framework.png|deleted|unknown|yes, likely important to preserve|patch file|
|scripts/prompts.py|modified|my new code|yes, likely important to preserve|patch file|
|scripts/tool_execute_gemini.py|modified|my new code|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|.gitignore|16 B|file|unknown|human review|
|audio_maestro/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|evaluation.py|4.11 KB|.py|evaluation script|move to main repo or keep as patch/addition|
|gpu_me.sh|6.63 KB|.sh|my new code|move to main repo or keep as patch/addition|
|gpu_stat.log|3.40 KB|.log|log|ignore|
|grpo/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_slurm_70408.log|13.9 KB|.log|log|ignore|
|grpo_slurm_70433.log|15.1 KB|.log|log|ignore|
|grpo_slurm_70499.log|13.8 KB|.log|log|ignore|
|grpo_slurm_70736.log|10.9 KB|.log|log|ignore|
|grpo_slurm_70779.log|14.0 KB|.log|log|ignore|
|grpo_slurm_70786.log|12.8 KB|.log|log|ignore|
|grpo_slurm_70879.log|12.4 KB|.log|log|ignore|
|grpo_slurm_70903.log|12.4 KB|.log|log|ignore|
|grpo_slurm_70917.log|39.4 KB|.log|log|ignore|
|grpo_slurm_70921.log|11.4 KB|.log|log|ignore|
|grpo_slurm_70926.log|47.0 KB|.log|log|ignore|
|grpo_slurm_72370.log|12.1 KB|.log|log|ignore|
|grpo_slurm_72396.log|14.9 KB|.log|log|ignore|
|internet.sh|693 B|.sh|my new code|move to main repo or keep as patch/addition|
|logs/|directory; size not expanded here|dir|unknown|human review|
|mmau-test-mini-cached.json|5.29 MB|.json|config change|move to main repo or keep as patch/addition|
|mmau-test-mini.json|577 KB|.json|config change|move to main repo or keep as patch/addition|
|requirements-clean.txt|3.40 KB|.txt|environment change|move to main repo or keep as patch/addition|
|requirments.txt|8.26 KB|.txt|unknown|human review|
|results/|directory; size not expanded here|dir|result|ignore|
|run.sh|3.55 KB|.sh|my new code|move to main repo or keep as patch/addition|
|scripts/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|scripts/agentic/|directory; size not expanded here|dir|unknown|human review|
|scripts/models/|directory; size not expanded here|dir|unknown|human review|
|scripts/tool_execute.py|36.4 KB|.py|my new code|move to main repo or keep as patch/addition|
|test_error_cases.py|20.1 KB|.py|temporary/debug|human review or ignore|
|test_tools.py|4.27 KB|.py|temporary/debug|human review or ignore|
|wandb/|directory; size not expanded here|dir|result|ignore|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/Audio-Maestro-bak
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro-bak diff > patches/Audio-Maestro-bak/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro-bak diff --name-status > patches/Audio-Maestro-bak/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro-bak ls-files --others --exclude-standard > patches/Audio-Maestro-bak/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `DeSTA2.5-Audio`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio`
- Remote origin URL: `https://github.com/kehanlu/DeSTA2.5-Audio.git`
- All remotes:
```text
origin	https://github.com/kehanlu/DeSTA2.5-Audio.git (fetch)
origin	https://github.com/kehanlu/DeSTA2.5-Audio.git (push)
```
- Current branch: `main`
- Current commit hash: `e9b28ffd97c559eb178096b853321a1bbe5d97cf`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `1`
- Deleted tracked files: `0`
- Untracked files/folders visible after `--directory` collapse: `1`
- Ignored-looking large/generated untracked entries in sample: `0`

`git status --short --untracked-files=no`:
```text
M desta/models/modeling_desta25.py
```

`git diff --stat`:
```text
desta/models/modeling_desta25.py | 104 ++++++++++++++++++++-------------------
 1 file changed, 54 insertions(+), 50 deletions(-)
```

`git diff --name-status`:
```text
M	desta/models/modeling_desta25.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|desta/models/modeling_desta25.py|modified|core third-party code edit|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|test.py|8.30 KB|.py|temporary/debug|human review or ignore|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/DeSTA2.5-Audio
git -C /home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio diff > patches/DeSTA2.5-Audio/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio diff --name-status > patches/DeSTA2.5-Audio/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio ls-files --others --exclude-standard > patches/DeSTA2.5-Audio/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `Desta_grpo`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo`
- Remote origin URL: `https://github.com/gary920209/Audio-Maestro.git`
- All remotes:
```text
origin	https://github.com/gary920209/Audio-Maestro.git (fetch)
origin	https://github.com/gary920209/Audio-Maestro.git (push)
```
- Current branch: `main`
- Current commit hash: `ba443a215573ff2d5b85e8efbec268343b40baa2`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `3`
- Deleted tracked files: `3`
- Untracked files/folders visible after `--directory` collapse: `107`
- Ignored-looking large/generated untracked entries in sample: `17`

`git status --short --untracked-files=no`:
```text
D README.md
 M audio_maestro/audio_copilot.py
 D image/Results.png
 D image/framework.png
 M scripts/prompts.py
 M scripts/tool_execute_gemini.py
```

`git diff --stat`:
```text
README.md                      |  87 ---------
 audio_maestro/audio_copilot.py | 434 +++++++++++++++++++++++------------------
 image/Results.png              | Bin 974324 -> 0 bytes
 image/framework.png            | Bin 224071 -> 0 bytes
 scripts/prompts.py             | 278 ++++++++++++++++++++++----
 scripts/tool_execute_gemini.py |  93 +++++----
 6 files changed, 526 insertions(+), 366 deletions(-)
```

`git diff --name-status`:
```text
D	README.md
M	audio_maestro/audio_copilot.py
D	image/Results.png
D	image/framework.png
M	scripts/prompts.py
M	scripts/tool_execute_gemini.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|README.md|deleted|documentation|yes, likely important to preserve|patch file|
|audio_maestro/audio_copilot.py|modified|my new code|yes, likely important to preserve|patch file|
|image/Results.png|deleted|result|no for Git; preserve outside Git/manifest if useful|ignored/manifest only|
|image/framework.png|deleted|unknown|yes, likely important to preserve|patch file|
|scripts/prompts.py|modified|my new code|yes, likely important to preserve|patch file|
|scripts/tool_execute_gemini.py|modified|my new code|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|.gitignore|16 B|file|unknown|human review|
|.vscode/|directory; size not expanded here|dir|unknown|human review|
|ToolRL/|directory; size not expanded here|dir|unknown|human review|
|__pycache__/|directory; size not expanded here|dir|cache|ignore|
|analyze_tool_comparison.py|13.6 KB|.py|my new code|move to main repo or keep as patch/addition|
|audio_maestro/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|biased_verified_creation/|directory; size not expanded here|dir|unknown|human review|
|build_flash_attn.sh|2.38 KB|.sh|my new code|move to main repo or keep as patch/addition|
|checkpoints/|directory; size not expanded here|dir|checkpoint|ignore|
|container_job_97883.err|65.9 KB|.err|log|ignore|
|container_job_97883.log|1.84 KB|.log|log|ignore|
|cuda12.sif|85.3 MB|.sif|unknown|human review|
|dataset.tar|186 MB|.tar|dataset script|move to main repo or keep as patch/addition|
|dataset/|directory; size not expanded here|dir|dataset script|move to main repo or keep as patch/addition|
|demo_output.txt|24.6 KB|.txt|result|ignore|
|demo_tool_mask.py|17.4 KB|.py|my new code|move to main repo or keep as patch/addition|
|desta_forced_multi_vs_single.json|324 KB|.json|config change|move to main repo or keep as patch/addition|
|desta_vllm/|directory; size not expanded here|dir|inference script|move to main repo or keep as patch/addition|
|diag_embed.py|3.16 KB|.py|my new code|move to main repo or keep as patch/addition|
|dummy_data.json|129 B|.json|dataset script|move to main repo or keep as patch/addition|
|evaluation.py|4.11 KB|.py|evaluation script|move to main repo or keep as patch/addition|
|expts/|directory; size not expanded here|dir|unknown|human review|
|extract_embeds.sh|749 B|.sh|my new code|move to main repo or keep as patch/addition|
|extract_embeds_offline.sh|804 B|.sh|my new code|move to main repo or keep as patch/addition|
|fake_group|0 B|file|unknown|human review|
|fake_passwd|0 B|file|unknown|human review|
|fix_flash_attn.sh|1.81 KB|.sh|my new code|move to main repo or keep as patch/addition|
|forced_dir/|directory; size not expanded here|dir|unknown|human review|
|gemma.yml|10.6 KB|.yml|config change|move to main repo or keep as patch/addition|
|gemma4.sif|8.25 GB|.sif|unknown|human review|
|getip.sh|1.62 KB|.sh|my new code|move to main repo or keep as patch/addition|
|gpu_me.sh|6.63 KB|.sh|my new code|move to main repo or keep as patch/addition|
|grpo/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_llm_agentic/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase copy/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase_llm copy 2/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase_llm copy/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase_llm/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_single_phase_llm_decoupled/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|grpo_trainer.py|125 KB|.py|training script|move to main repo or keep as patch/addition|
|hallucination/|directory; size not expanded here|dir|unknown|human review|
|image(6).png|748 KB|.png|unknown|human review|
|infer_101490_direct_base_slurm_104414.log|570 KB|.log|log|ignore|
|inference/|directory; size not expanded here|dir|inference script|move to main repo or keep as patch/addition|
|install_flash_attn.sh|1019 B|.sh|my new code|move to main repo or keep as patch/addition|
|install_vllm.sh|940 B|.sh|inference script|move to main repo or keep as patch/addition|
|interactive.sh|92 B|.sh|my new code|move to main repo or keep as patch/addition|
|internet.sh|693 B|.sh|my new code|move to main repo or keep as patch/addition|
|judge-867777.tar|316 MB|.tar|unknown|human review|
|judge_logs/|directory; size not expanded here|dir|unknown|human review|
|judge_tool_empirical.py|14.8 KB|.py|my new code|move to main repo or keep as patch/addition|
|judge_tool_helpfulness.py|9.83 KB|.py|my new code|move to main repo or keep as patch/addition|
|load_test.py|2.08 KB|.py|temporary/debug|human review or ignore|
|logs/|directory; size not expanded here|dir|unknown|human review|
|logs2/|directory; size not expanded here|dir|unknown|human review|
|metric_93279_cosine_decay.log|1.18 KB|.log|log|ignore|
|mmau-test-mini-cached-top3.json|5.45 MB|.json|config change|move to main repo or keep as patch/addition|
|mmau-test-mini-cached.json|5.29 MB|.json|config change|move to main repo or keep as patch/addition|
|mmau-test-mini.json|577 KB|.json|config change|move to main repo or keep as patch/addition|
|mmau_af3_biased.json|4.16 MB|.json|config change|move to main repo or keep as patch/addition|
|mmau_af3_biased_eval_results.json|737 KB|.json|result|ignore|
|mmau_af3_biased_forced_single_turn.json|140 KB|.json|config change|move to main repo or keep as patch/addition|
|mmau_desta_biased_single_pass.json|4.73 MB|.json|config change|move to main repo or keep as patch/addition|
|my_jobs.sh|3.01 KB|.sh|my new code|move to main repo or keep as patch/addition|
|old_grpo_logs/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|plot_comparison_table.py|5.16 KB|.py|my new code|move to main repo or keep as patch/addition|
|plot_grpo_metrics.py|8.45 KB|.py|training script|move to main repo or keep as patch/addition|
|precomputed_embeds/|directory; size not expanded here|dir|unknown|human review|
|probe_97874.out|2.05 KB|.out|log|ignore|
|prompts.py|5.59 KB|.py|my new code|move to main repo or keep as patch/addition|
|results/|directory; size not expanded here|dir|result|ignore|
|run.sh|3.55 KB|.sh|my new code|move to main repo or keep as patch/addition|
|scripts/__init__.py|0 B|.py|my new code|move to main repo or keep as patch/addition|
|scripts/__pycache__/|directory; size not expanded here|dir|cache|ignore|
|scripts/diag_embed_gpu.py|9.33 KB|.py|my new code|move to main repo or keep as patch/addition|
|scripts/extract_embeddings.py|8.52 KB|.py|my new code|move to main repo or keep as patch/addition|
|scripts/models/|directory; size not expanded here|dir|unknown|human review|
|scripts/podman_gpu_test_job.sh|1.82 KB|.sh|temporary/debug|human review or ignore|
|scripts/run_diag.sh|614 B|.sh|my new code|move to main repo or keep as patch/addition|
|scripts/runtime_matrix_gpu_test.sh|1.42 KB|.sh|temporary/debug|human review or ignore|
|scripts/select_top_tools.py|7.40 KB|.py|my new code|move to main repo or keep as patch/addition|
|scripts/snapshot_expt.sh|4.25 KB|.sh|my new code|move to main repo or keep as patch/addition|
|scripts/test_embed_pipeline.py|11.3 KB|.py|temporary/debug|human review or ignore|
|scripts/tool_execute.py|37.4 KB|.py|my new code|move to main repo or keep as patch/addition|
|selectio_tools.log|6.76 MB|.log|log|ignore|
|synthetic_dataset/|directory; size not expanded here|dir|dataset script|move to main repo or keep as patch/addition|
|test_a40.log|488 B|.log|log|ignore|
|test_a40.sh|493 B|.sh|temporary/debug|human review or ignore|
|test_bpe_boundary.py|7.97 KB|.py|temporary/debug|human review or ignore|
|test_chat_template_mask.py|15.4 KB|.py|temporary/debug|human review or ignore|
|test_error_cases.py|20.1 KB|.py|temporary/debug|human review or ignore|
|test_gpt_oss.py|695 B|.py|temporary/debug|human review or ignore|
|test_qwen.py|3.14 KB|.py|temporary/debug|human review or ignore|
|test_qwen2.py|3.11 KB|.py|temporary/debug|human review or ignore|
|test_qwen3.py|2.99 KB|.py|temporary/debug|human review or ignore|
|test_qwen_simple.py|670 B|.py|temporary/debug|human review or ignore|
|test_qwen_simple2.py|456 B|.py|temporary/debug|human review or ignore|
|test_tools.py|4.27 KB|.py|temporary/debug|human review or ignore|
|tool_empirical_results.json|3.88 MB|.json|result|ignore|
|tool_helpfulness_results.json|1.14 MB|.json|result|ignore|
|training_logs/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|utils/|directory; size not expanded here|dir|unknown|human review|
|verify_top3_tool_capablity.py|21.5 KB|.py|my new code|move to main repo or keep as patch/addition|
|vllm_install.sh|1.55 KB|.sh|inference script|move to main repo or keep as patch/addition|
|wandb/|directory; size not expanded here|dir|result|ignore|
|wavecaps/|directory; size not expanded here|dir|unknown|human review|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/Desta_grpo
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo diff > patches/Desta_grpo/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo diff --name-status > patches/Desta_grpo/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo ls-files --others --exclude-standard > patches/Desta_grpo/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `Desta_grpo/hallucination/AHa-Bench`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/hallucination/AHa-Bench`
- Remote origin URL: `https://github.com/AHa-Bench/AHa-Bench.git`
- All remotes:
```text
origin	https://github.com/AHa-Bench/AHa-Bench.git (fetch)
origin	https://github.com/AHa-Bench/AHa-Bench.git (push)
```
- Current branch: `main`
- Current commit hash: `58657da77f91755ccb5ef275c9232300796ff50d`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `clean`
- Modified tracked files: `0`
- Deleted tracked files: `0`
- Untracked files/folders visible after `--directory` collapse: `0`
- Ignored-looking large/generated untracked entries in sample: `0`

`git status --short --untracked-files=no`:
```text
(no tracked status output)
```

`git diff --stat`:
```text
(no tracked diff stat)
```

`git diff --name-status`:
```text
(no tracked modifications)
```

### 3. Tracked modifications

No tracked modified/deleted/renamed files found.

### 4. Untracked files inside cloned repo

No untracked files/folders found by collapsed listing.

### 5. Patch/fork decision

Primary strategy: **ignore clean repo**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/AHa-Bench
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/hallucination/AHa-Bench diff > patches/AHa-Bench/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/hallucination/AHa-Bench diff --name-status > patches/AHa-Bench/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/hallucination/AHa-Bench ls-files --others --exclude-standard > patches/AHa-Bench/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `Desta_grpo/ToolRL`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL`
- Remote origin URL: `https://github.com/qiancheng0/ToolRL.git`
- All remotes:
```text
origin	https://github.com/qiancheng0/ToolRL.git (fetch)
origin	https://github.com/qiancheng0/ToolRL.git (push)
```
- Current branch: `main`
- Current commit hash: `8cee13ec0ca72f0461da372a93a6fd8140dbb840`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `3`
- Deleted tracked files: `0`
- Untracked files/folders visible after `--directory` collapse: `2`
- Ignored-looking large/generated untracked entries in sample: `0`

`git status --short --untracked-files=no`:
```text
M requirements.txt
 M verl/utils/tokenizer.py
 M verl/workers/fsdp_workers.py
```

`git diff --stat`:
```text
requirements.txt             |  6 +++---
 verl/utils/tokenizer.py      | 26 ++++++++++++++++++++++++++
 verl/workers/fsdp_workers.py | 35 +++++++++++++++++++++++++++++++++++
 3 files changed, 64 insertions(+), 3 deletions(-)
```

`git diff --name-status`:
```text
M	requirements.txt
M	verl/utils/tokenizer.py
M	verl/workers/fsdp_workers.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|requirements.txt|modified|environment change|yes, likely important to preserve|patch file|
|verl/utils/tokenizer.py|modified|core third-party code edit|yes, likely important to preserve|patch file|
|verl/workers/fsdp_workers.py|modified|core third-party code edit|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|audio_grpo/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|
|dataset/audio_grpo/|directory; size not expanded here|dir|training script|move to main repo or keep as patch/addition|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/ToolRL
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL diff > patches/ToolRL/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL diff --name-status > patches/ToolRL/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL ls-files --others --exclude-standard > patches/ToolRL/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `grpo_dataset/cache_tools/Audio-Maestro`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro`
- Remote origin URL: `https://github.com/gary920209/Audio-Maestro.git`
- All remotes:
```text
origin	https://github.com/gary920209/Audio-Maestro.git (fetch)
origin	https://github.com/gary920209/Audio-Maestro.git (push)
```
- Current branch: `main`
- Current commit hash: `ba443a215573ff2d5b85e8efbec268343b40baa2`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `modified/untracked`
- Modified tracked files: `1`
- Deleted tracked files: `0`
- Untracked files/folders visible after `--directory` collapse: `1`
- Ignored-looking large/generated untracked entries in sample: `1`

`git status --short --untracked-files=no`:
```text
M audio_maestro/audio_copilot.py
```

`git diff --stat`:
```text
audio_maestro/audio_copilot.py | 343 ++++++++++++++++++++++-------------------
 1 file changed, 181 insertions(+), 162 deletions(-)
```

`git diff --name-status`:
```text
M	audio_maestro/audio_copilot.py
```

### 3. Tracked modifications

|path|status|rough category|likely important to preserve|should become|
|---|---|---|---|---|
|audio_maestro/audio_copilot.py|modified|my new code|yes, likely important to preserve|patch file|

### 4. Untracked files inside cloned repo

|path|size|extension/type|category|recommended action|
|---|---|---|---|---|
|audio_maestro/__pycache__/|directory; size not expanded here|dir|cache|ignore|

### 5. Patch/fork decision

Primary strategy: **store patch files in patches/<repo>/**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/Audio-Maestro
git -C /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro diff > patches/Audio-Maestro/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro diff --name-status > patches/Audio-Maestro/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/cache_tools/Audio-Maestro ls-files --others --exclude-standard > patches/Audio-Maestro/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

## `toolRL/ToolRL`

### 1. Repository identity

- Local path: `/home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL`
- Remote origin URL: `https://github.com/qiancheng0/ToolRL.git`
- All remotes:
```text
origin	https://github.com/qiancheng0/ToolRL.git (fetch)
origin	https://github.com/qiancheng0/ToolRL.git (push)
```
- Current branch: `main`
- Current commit hash: `8cee13ec0ca72f0461da372a93a6fd8140dbb840`
- Latest tag: `(none)`
- HEAD detached: `no`

### 2. Working tree state

- State: `clean`
- Modified tracked files: `0`
- Deleted tracked files: `0`
- Untracked files/folders visible after `--directory` collapse: `0`
- Ignored-looking large/generated untracked entries in sample: `0`

`git status --short --untracked-files=no`:
```text
(no tracked status output)
```

`git diff --stat`:
```text
(no tracked diff stat)
```

`git diff --name-status`:
```text
(no tracked modifications)
```

### 3. Tracked modifications

No tracked modified/deleted/renamed files found.

### 4. Untracked files inside cloned repo

No untracked files/folders found by collapsed listing.

### 5. Patch/fork decision

Primary strategy: **ignore clean repo**

Rationale: clean repos can be ignored/submoduled later; small tracked source/config changes fit patch preservation; broad internal edits fit forks; standalone untracked scripts/configs should be extracted; generated data/results/checkpoints should be ignored or represented by manifests only.

### 6. Suggested preservation commands (do not run until migration time)

```bash
mkdir -p patches/ToolRL
git -C /home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL diff > patches/ToolRL/tracked_changes.patch
git -C /home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL diff --name-status > patches/ToolRL/tracked_changed_files.txt
git -C /home/speech-nlp-cse/24m0756/abhishek/toolRL/ToolRL ls-files --others --exclude-standard > patches/ToolRL/untracked_files.txt
# Review untracked_files.txt, then copy selected scripts/configs into scripts/, configs/, src/, or docs/.
```

