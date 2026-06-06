# Audio-Maestro main workspace Patch Preservation

## Source

- Original local path: `/home/speech-nlp-cse/24m0756/abhishek/Audio-Maestro`
- Upstream remote URL: `https://github.com/gary920209/Audio-Maestro.git`
- Branch: `main`
- Commit hash: `ba443a215573ff2d5b85e8efbec268343b40baa2`
- Detached HEAD: `no`
- Latest reachable tag: `none`
- Working tree modified: `yes`
- Primary strategy: Submodule candidate later, with this patch applied on top.

## Duplicate Patch Note

The tracked patch is byte-identical to the other Audio-Maestro patch captured in this clean repo. Do not delete either until the backup workspace has been reviewed.

## Modified Tracked Files

- `M	audio_maestro/audio_copilot.py`
- `D	image/Results.png`
- `D	image/framework.png`
- `M	scripts/prompts.py`
- `M	scripts/tool_execute_gemini.py`

## Files In This Directory

- `repo_info.txt`: remotes, branch, commit, detached state, latest tag, and `git status --short` output.
- `tracked_changes.patch`: tracked working-tree changes only.
- `tracked_changed_files.txt`: tracked changed/deleted/renamed file list.
- `untracked_files.txt`: untracked file names only; no untracked artifacts are copied here.

## Apply Later

```bash
git clone https://github.com/gary920209/Audio-Maestro.git <repo>
cd <repo>
git checkout ba443a215573ff2d5b85e8efbec268343b40baa2
git apply /path/to/tracked_changes.patch
```

Large/generated/untracked artifacts are not copied into this clean repository. Untracked files are listed only in `untracked_files.txt` for human review.
