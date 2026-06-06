# License Audit

## Current Repository License Status

- Top-level project license: pending.
- `pyproject.toml` currently declares: `License pending upstream review`.
- `docs/license_notes.md` correctly says not to create a top-level `LICENSE` yet.
- Recommendation: keep the repository private until upstream license review and patch strategy decisions are complete.
- Required conclusion: do not add a top-level `LICENSE` yet, and keep issue `#1` open.

## Submodule Licenses Discoverable Locally

| Submodule | Remote | Local commit | License found locally | Notes |
| --- | --- | --- | --- | --- |
| `third_party/Audio-Maestro` | `https://github.com/gary920209/Audio-Maestro.git` | `ba443a215573ff2d5b85e8efbec268343b40baa2` | Unresolved locally | `find third_party -maxdepth 3` found no local `LICENSE`, `COPYING`, or `NOTICE` file for Audio-Maestro. |
| `third_party/DeSTA2.5-Audio` | `https://github.com/kehanlu/DeSTA2.5-Audio.git` | `e9b28ffd97c559eb178096b853321a1bbe5d97cf` | Ambiguous locally | No top-level `LICENSE`, `COPYING`, or `NOTICE` file was found. `setup.py` includes the classifier `License :: OSI Approved :: MIT License`, `desta/utils/audio.py` contains Apache-2.0 header text, and `docs/dataset.md` says original audio files cannot be redistributed. |
| `third_party/ToolRL` | `https://github.com/qiancheng0/ToolRL.git` | `8cee13ec0ca72f0461da372a93a6fd8140dbb840` | Apache License 2.0 | Confirmed locally via `third_party/ToolRL/LICENSE`; file headers are also consistent with Apache-2.0. |

## Patch Files And Third-Party Code

Patch files do preserve modifications to third-party code:

| Patch directory | Upstream project | Modified tracked files |
| --- | --- | --- |
| `patches/audio-maestro-main/` | Audio-Maestro | `audio_maestro/audio_copilot.py`, `scripts/prompts.py`, `scripts/tool_execute_gemini.py`; original full patch also records deleted image files. |
| `patches/audio-maestro-bak/` | Audio-Maestro | Same tracked file set as the main Audio-Maestro patch. |
| `patches/desta-grpo/` | Mixed Audio-Maestro-derived workspace | `README.md`, `audio_maestro/audio_copilot.py`, `scripts/prompts.py`, `scripts/tool_execute_gemini.py`, deleted image files. |
| `patches/desta25-audio/` | DeSTA2.5-Audio | `desta/models/modeling_desta25.py`. |
| `patches/desta-grpo-toolrl/` | ToolRL | `requirements.txt`, `verl/utils/tokenizer.py`, `verl/workers/fsdp_workers.py`. |
| `patches/grpo-dataset-cache-audio-maestro/` | Audio-Maestro snapshot | `audio_maestro/audio_copilot.py`. |

## Top-Level License Recommendation

Do not add a top-level `LICENSE` before public release. The clean repo contains original/extracted research code, submodules, and patch overlays that interact with upstream projects whose local license status is incomplete or ambiguous.

## Private/Public Recommendation

- Private repository: acceptable for current research and preservation use.
- Public repository: not ready until Audio-Maestro license status is resolved, DeSTA2.5-Audio licensing is clarified, and patch overlay distribution is reviewed.

## Unresolved License Decisions

- Confirm upstream license terms for Audio-Maestro.
- Clarify DeSTA2.5-Audio licensing because local evidence is mixed: MIT classifier in `setup.py`, Apache-2.0 header text in at least one file, and no top-level license file.
- Confirm whether DeSTA2.5-Audio redistribution constraints for code and non-redistributable audio are documented consistently.
- Decide whether patch files can be publicly distributed as-is.
- Decide whether modified third-party code should live as forks, patch overlays, or private-only preservation metadata.
- Decide the top-level project license only after the above review.
