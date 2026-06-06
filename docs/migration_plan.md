# Migration Plan

Practical, preservation-focused plan for turning this working directory into a clean GitHub repository. Do not execute the commands until you have reviewed the two audit reports.

## 1. Safety steps

- Make a full backup of the current folder before touching Git history or moving files.
- Do not run `git clean`, `git reset --hard`, or deletion commands in the working directory or nested repos.
- Do not delete cloned repos; they may contain local edits and new scripts/configs.
- Do not commit raw datasets, copied audio folders, checkpoints, caches, logs, or full result directories.
- Preserve code changes first as patches/fork branches or extracted scripts before applying `.gitignore` broadly.

## 2. Proposed clean repo layout

```text
mtp2-audio-tool-rl/
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── environment/
├── src/
├── scripts/
├── configs/
├── tests/
├── docs/
├── manifests/
├── patches/
└── third_party/
```

## 3. File movement recommendations

|current path|suggested destination|recommendation|reason/purpose|
|---|---|---|---|
|ALLOWED_PARTITIONS.md|docs/|commit if wanted|small documentation|
|dummy_inference.py|scripts/|review and place if still used|project script; review temporary status and dependencies|
|env.yaml|configs/ or environment/|review and place if still used|commit only one curated environment/config file after review|
|env.yaml.bak|remain external or human review|review and place if still used|commit only one curated environment/config file after review|
|gemma.yml|configs/ or environment/|review and place if still used|commit only one curated environment/config file after review|
|generate_job_94000.err|scripts/|do not commit|generated job/log output|
|generate_job_94000.out|scripts/|do not commit|generated job/log output|
|gpu_stat.sh|scripts/|review and place if still used|project script; review temporary status and dependencies|
|hello.sh|scripts/|review and place if still used|project script; review temporary status and dependencies|
|interactive.sh|scripts/|review and place if still used|project script; review temporary status and dependencies|
|job_diagnose.sh|scripts/|review and place if still used|project script; review temporary status and dependencies|
|nohup.out|remain external or human review|do not commit|generated job/log output|
|run_dummy.sh|scripts/|review and place if still used|project script; review temporary status and dependencies|
|Training.md|docs/|commit if wanted|small documentation|

## 4. Third-party strategy

|repo path|origin|state|strategy|
|---|---|---|---|
|Audio-Maestro|https://github.com/gary920209/Audio-Maestro.git|modified/untracked|patch|
|Audio-Maestro-bak|https://github.com/gary920209/Audio-Maestro.git|modified/untracked|patch|
|DeSTA2.5-Audio|https://github.com/kehanlu/DeSTA2.5-Audio.git|modified/untracked|patch|
|Desta_grpo|https://github.com/gary920209/Audio-Maestro.git|modified/untracked|patch|
|Desta_grpo/hallucination/AHa-Bench|https://github.com/AHa-Bench/AHa-Bench.git|clean|ignore now; submodule later if needed|
|Desta_grpo/ToolRL|https://github.com/qiancheng0/ToolRL.git|modified/untracked|patch|
|grpo_dataset/cache_tools/Audio-Maestro|https://github.com/gary920209/Audio-Maestro.git|modified/untracked|patch|
|toolRL/ToolRL|https://github.com/qiancheng0/ToolRL.git|clean|ignore now; submodule later if needed|

## 5. Dataset policy

Commit these: small JSON schemas, tiny example manifests, download/preparation scripts, mapping scripts, README instructions, and minimal synthetic/test fixtures.

Do not commit these: raw audio, full datasets, copied audio folders, large parquet dumps, checkpoints, model weights, array caches, tool-result caches, full result folders, Slurm logs, `wandb/`, or virtual environments.

## 6. Initial commit plan (commands to run later, not during audit)

```bash
# From a new clean repository directory, after backing up the current workspace:
mkdir mtp2-audio-tool-rl
cd mtp2-audio-tool-rl
git init -b main
mkdir -p environment src scripts configs tests docs manifests patches third_party

# Create .gitignore from docs/repo_audit.md recommended draft, then inspect before adding files.
$EDITOR .gitignore
$EDITOR .gitattributes

# Copy only reviewed safe files from the old workspace into this clean layout.
# Examples, adjust after human review:
cp ../abhishek/Training.md docs/
cp ../abhishek/ALLOWED_PARTITIONS.md docs/
cp ../abhishek/env.yaml environment/
cp ../abhishek/*.sh scripts/  # only after reviewing temporary/job-specific scripts
cp ../abhishek/dummy_inference.py scripts/  # only if it is useful, otherwise skip

# Preserve nested repo changes before ignoring third-party directories.
# Example pattern for each modified repo:
mkdir -p patches/Audio-Maestro
git -C ../abhishek/Audio-Maestro diff > patches/Audio-Maestro/tracked_changes.patch
git -C ../abhishek/Audio-Maestro diff --name-status > patches/Audio-Maestro/tracked_changed_files.txt
git -C ../abhishek/Audio-Maestro ls-files --others --exclude-standard > patches/Audio-Maestro/untracked_files.txt

# Review what Git sees before committing.
git status --short
git add README.md LICENSE .gitignore .gitattributes environment/ src/ scripts/ configs/ tests/ docs/ manifests/ patches/
git status --short
git diff --cached --stat
git diff --cached --name-only

# If staged contents are safe:
git commit -m "Initial clean MTP audio tool RL repository"
git remote add origin git@github.com:<USER_OR_ORG>/mtp2_audio_tool_rl.git
git push -u origin main
```

## 7. Human decisions needed

- Decide which nested repos are true research dependencies versus temporary clones.
- For every modified nested repo, choose patch versus fork after reviewing `docs/modified_repos_audit.md`.
- Decide whether `env.yaml`, `gemma.yml`, or a new minimal `environment/conda.yml` should be the canonical environment.
- Decide whether any generated outputs should be summarized in `manifests/` or omitted entirely.
- Decide whether any checkpoint/model files are essential enough for Git LFS; default recommendation is no.
- Review secret-risk paths and rotate/remove any real credentials before publishing.
- Write a top-level README explaining project purpose, setup, datasets expected outside Git, and reproduction steps.
