# Patch Application Plan

## Purpose

This plan is for patch dry-runs only. The helper script uses `git apply --check` to preview whether preserved tracked changes can apply to pinned submodules. It does not apply patches, commit, push, or modify submodule working trees.

## Patch-To-Submodule Mapping

| Patch | Target submodule | Dry-run result | Notes |
| --- | --- | --- | --- |
| `patches/audio-maestro-main/tracked_source_only.patch` | `third_party/Audio-Maestro` | PASS | Active preview patch; excludes binary image deletions from the preserved full tracked diff. |
| `patches/desta25-audio/tracked_changes.patch` | `third_party/DeSTA2.5-Audio` | PASS | Clean `git apply --check` against pinned submodule commit. |
| `patches/desta-grpo-toolrl/tracked_changes.patch` | `third_party/ToolRL` | PASS | Clean `git apply --check` against pinned submodule commit. |

## Patches Not Mapped Yet

| Patch directory | Reason |
| --- | --- |
| `patches/audio-maestro-bak/` | Duplicate of `audio-maestro-main` unless the backup workspace is later proven useful. |
| `patches/desta-grpo/` | Came from a mixed research workspace; keep as preservation metadata until manually reviewed. |
| `patches/grpo-dataset-cache-audio-maestro/` | Cache-tool-specific Audio-Maestro variant; keep as metadata unless manually reviewed. |

## Preview Script

Run:

```bash
bash scripts/setup/apply_patches_preview.sh
```

Expected behavior:

- Prints current submodule status.
- Verifies required submodule directories exist.
- Runs only `git apply --check` against mapped patch files.
- Reports each mapped patch as PASS or FAIL.
- Exits nonzero if any mapped patch fails.
- Does not apply patches.

## Current Preview Result

The latest preview run reported:

- `Audio-Maestro`: PASS using `tracked_source_only.patch`.
- `DeSTA2.5-Audio`: PASS.
- `ToolRL`: PASS.

No patches were applied.

The original `patches/audio-maestro-main/tracked_changes.patch` is still preserved for audit/history. It includes binary image deletion hunks for `image/Results.png` and `image/framework.png`, which are intentionally excluded from the active preview patch. Those deleted binary images are not needed for the current code setup.

## If A Patch Applies Cleanly

Do not apply it directly inside a pinned submodule on the main branch. Prefer one of these strategies:

- Create a fork branch for the upstream repository and apply the patch there.
- Keep the patch as an overlay and document the exact upstream commit it applies to.
- If the change is small and temporary, keep it as a patch until the dependency strategy is finalized.

## If A Patch Fails

Manually review any failing patch before an application attempt. If the original full Audio-Maestro patch is revisited, likely next steps are:

- Split source-code changes from binary asset deletions.
- Regenerate a source-only patch for `audio_maestro/audio_copilot.py`, `scripts/prompts.py`, and `scripts/tool_execute_gemini.py`.
- Handle deleted binary image files separately, or regenerate a binary-capable patch only if those deletions are still desired.

## Warning

Do not run `git apply` directly inside submodules unless intentionally creating a fork branch or a clearly documented local overlay workflow. Direct patch application inside submodules can leave the parent repo pointing at a clean gitlink while the submodule working tree contains uncommitted changes.

## Future Strategy

- Clean patches: convert to fork branches or keep as documented overlays.
- Failing patches: manually review and regenerate narrower patches.
- Duplicate patches: keep one canonical patch later after confirming which workspace copy is authoritative.
