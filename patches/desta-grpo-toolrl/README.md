# Desta_grpo/ToolRL Patch Preservation

## Source

- Original local path: `/home/speech-nlp-cse/24m0756/abhishek/Desta_grpo/ToolRL`
- Upstream remote URL: `https://github.com/qiancheng0/ToolRL.git`
- Branch: `main`
- Commit hash: `8cee13ec0ca72f0461da372a93a6fd8140dbb840`
- Detached HEAD: `no`
- Latest reachable tag: `none`
- Working tree modified: `yes`
- Primary strategy: Submodule or fork candidate later, with this patch applied on top.

## Modified Tracked Files

- `M	requirements.txt`
- `M	verl/utils/tokenizer.py`
- `M	verl/workers/fsdp_workers.py`

## Files In This Directory

- `repo_info.txt`: remotes, branch, commit, detached state, latest tag, and `git status --short` output.
- `tracked_changes.patch`: tracked working-tree changes only.
- `tracked_changed_files.txt`: tracked changed/deleted/renamed file list.
- `untracked_files.txt`: untracked file names only; no untracked artifacts are copied here.

## Apply Later

```bash
git clone https://github.com/qiancheng0/ToolRL.git <repo>
cd <repo>
git checkout 8cee13ec0ca72f0461da372a93a6fd8140dbb840
git apply /path/to/tracked_changes.patch
```

Large/generated/untracked artifacts are not copied into this clean repository. Untracked files are listed only in `untracked_files.txt` for human review.
