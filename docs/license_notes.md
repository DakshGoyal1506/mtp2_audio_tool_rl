# License Notes

Do not create a top-level `LICENSE` file yet.

License choice is still pending because this repository contains extracted project-owned code alongside patch metadata that interacts with several third-party repositories.

Before choosing a license, review the upstream licenses and reuse constraints for:

- `Audio-Maestro`
- `DeSTA2.5-Audio`
- `ToolRL`
- `AHa-Bench`

Current local status:

- `ToolRL` is confirmed Apache-2.0 locally.
- `Audio-Maestro` remains unresolved locally because no top-level `LICENSE`, `COPYING`, or `NOTICE` file was found.
- `DeSTA2.5-Audio` remains ambiguous locally: `setup.py` advertises an MIT classifier, some files contain Apache-2.0 header text, and no top-level license file was found.

Recommended next step: keep issue `#1` open and complete an upstream license review before deciding between MIT, Apache-2.0, a custom research-use license, or leaving the repository without a public code license.

Until that review is finished, a private repository is the safest default.
