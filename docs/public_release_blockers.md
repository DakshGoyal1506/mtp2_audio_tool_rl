# Public Release Blockers

| Blocker | Severity | Affected files | Recommended fix | Required before public release |
| --- | --- | --- | --- | --- |
| Audio-Maestro license unresolved locally | warning | `third_party/Audio-Maestro`, Audio-Maestro patch dirs | Confirm upstream license from authoritative source and document redistribution terms in notices/docs. | Recommended |
| DeSTA2.5-Audio license ambiguous locally | warning | `third_party/DeSTA2.5-Audio`, `patches/desta25-audio/` | Resolve the mismatch between the MIT classifier, Apache-2.0 file header evidence, and the lack of a top-level license file. | Recommended |
| Patch secret-risk review | warning | `patches/audio-maestro-main/`, `patches/audio-maestro-bak/`, `patches/desta-grpo/` | Manually review API-key variable references and rerun secret scanning as a follow-up. | Recommended |
| Third-party patch strategy unresolved | warning | `patches/`, `third_party/` | Decide between forks, patch overlays, or private-only preservation. | Recommended |
| Patch overlays not applied/forked | warning | `scripts/setup/apply_patches_preview.sh`, `patches/` | Convert clean patches into fork branches or document overlay workflow. | Recommended |
| README/repro docs incomplete | warning | `README.md`, `docs/` | Add public-facing setup, environment, data reconstruction, and reproducibility notes. | Recommended |
| Heavy tests not runnable in CI | warning | `tests/`, `.github/workflows/repo-check.yml` | Mark heavy tests and add lightweight unit tests for public CI. | Recommended |
| No full dataset included by design | warning | `manifests/`, `docs/dataset_reconstruction.md` | Keep this policy, but add clear public dataset reconstruction instructions. | No |
| Audit docs contain preservation metadata | warning | `docs/*audit*.md`, extraction/migration reports | Decide whether to publish detailed audit reports or keep only summarized public docs. | Recommended |
| Generated/model artifacts excluded | later | `.gitignore`, CI, setup checks | Keep exclusion policy; provide external artifact instructions if needed. | No |

## Current Recommendation

The repository can be made public with the scoped MIT license and third-party caveats documented. Issue `#1` remains open as follow-up work rather than a hard blocker for initial public visibility.
