# License Notes

A top-level `LICENSE` file is now present and uses MIT for original repository-authored code, scripts, configuration, documentation, and scaffolding only.

That MIT license does not relicense third-party submodules, third-party-derived patch files, or external artifacts.

Before choosing a license, review the upstream licenses and reuse constraints for:

- `Audio-Maestro`
- `DeSTA2.5-Audio`
- `ToolRL`
- `AHa-Bench`

Current local status:

- `ToolRL` is confirmed Apache-2.0 locally.
- `Audio-Maestro` remains unresolved locally because no top-level `LICENSE`, `COPYING`, or `NOTICE` file was found.
- `DeSTA2.5-Audio` remains ambiguous locally: `setup.py` advertises an MIT classifier, some files contain Apache-2.0 header text, and no top-level license file was found.

Recommended next step: keep issue `#1` open and complete upstream license follow-up for Audio-Maestro, DeSTA2.5-Audio, and patch redistribution strategy.

The repository can be public with these caveats documented, but third-party licensing and patch-policy follow-up still matters.
