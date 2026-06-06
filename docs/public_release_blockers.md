# Public Release Blockers

| Blocker | Severity | Affected files | Recommended fix | Required before public release |
| --- | --- | --- | --- | --- |
| License pending | blocking | `pyproject.toml`, `docs/license_notes.md`, submodules, patches | Complete upstream license review and choose a compatible top-level license or keep no public license. | Yes |
| Audio-Maestro license not discoverable locally | blocking | `third_party/Audio-Maestro`, Audio-Maestro patch dirs | Confirm upstream license from authoritative source and document redistribution terms. | Yes |
| DeSTA2.5-Audio license not discoverable locally | blocking | `third_party/DeSTA2.5-Audio`, `patches/desta25-audio/` | Confirm upstream license from authoritative source and document redistribution terms. | Yes |
| Patch secret-risk review | blocking | `patches/audio-maestro-main/`, `patches/audio-maestro-bak/`, `patches/desta-grpo/` | Manually review API-key variable references and rerun secret scanning before public release. | Yes |
| Third-party patch strategy unresolved | blocking | `patches/`, `third_party/` | Decide between forks, patch overlays, or private-only preservation. | Yes |
| Patch overlays not applied/forked | warning | `scripts/setup/apply_patches_preview.sh`, `patches/` | Convert clean patches into fork branches or document overlay workflow. | Recommended |
| README/repro docs incomplete | warning | `README.md`, `docs/` | Add public-facing setup, environment, data reconstruction, and reproducibility notes. | Recommended |
| Heavy tests not runnable in CI | warning | `tests/`, `.github/workflows/repo-check.yml` | Mark heavy tests and add lightweight unit tests for public CI. | Recommended |
| No full dataset included by design | warning | `manifests/`, `docs/dataset_reconstruction.md` | Keep this policy, but add clear public dataset reconstruction instructions. | No |
| Audit docs contain preservation metadata | warning | `docs/*audit*.md`, extraction/migration reports | Decide whether to publish detailed audit reports or keep only summarized public docs. | Recommended |
| Generated/model artifacts excluded | later | `.gitignore`, CI, setup checks | Keep exclusion policy; provide external artifact instructions if needed. | No |

## Current Recommendation

Do not make the repository public yet. The main blockers are licensing and patch-publication review.
