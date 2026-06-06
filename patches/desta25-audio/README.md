# DeSTA2.5-Audio Patch Preservation

## Source

- Original local path: `/home/speech-nlp-cse/24m0756/abhishek/DeSTA2.5-Audio`
- Upstream remote URL: `https://github.com/kehanlu/DeSTA2.5-Audio.git`
- Branch: `main`
- Commit hash: `e9b28ffd97c559eb178096b853321a1bbe5d97cf`
- Detached HEAD: `no`
- Latest reachable tag: `none`
- Working tree modified: `yes`
- Primary strategy: Submodule candidate later, with this patch applied on top.

## Modified Tracked Files

- `M	desta/models/modeling_desta25.py`

## Files In This Directory

- `repo_info.txt`: remotes, branch, commit, detached state, latest tag, and `git status --short` output.
- `tracked_changes.patch`: tracked working-tree changes only.
- `tracked_changed_files.txt`: tracked changed/deleted/renamed file list.
- `untracked_files.txt`: untracked file names only; no untracked artifacts are copied here.

## Apply Later

```bash
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git <repo>
cd <repo>
git checkout e9b28ffd97c559eb178096b853321a1bbe5d97cf
git apply /path/to/tracked_changes.patch
```

Large/generated/untracked artifacts are not copied into this clean repository. Untracked files are listed only in `untracked_files.txt` for human review.
