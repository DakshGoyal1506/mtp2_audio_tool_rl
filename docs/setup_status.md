# Setup Status

## Setup Files Created

- `pyproject.toml` for a setuptools `src/` layout package.
- `requirements.txt` plus `requirements/base.txt`, `requirements/audio.txt`, and `requirements/rl.txt`.
- `.env.example` with placeholder-only local path and API key variables.
- `pytest.ini` with `tests` as the test root and `src` on `pythonpath`.
- `Makefile` with setup, check, import, test, and status targets.
- `scripts/setup/check_repo_setup.py` for lightweight repository safety checks.
- `scripts/setup/check_imports.py` for base package import checks without loading heavy model/GPU modules.
- `scripts/setup/check_repo_setup.py` now reads `.gitmodules`, verifies registered submodule paths exist, and skips submodule working trees during parent-repo artifact scans.

## Checks Run

- `python scripts/setup/check_repo_setup.py`: not runnable on this machine because `python` is not on `PATH`.
- `python scripts/setup/check_imports.py`: not runnable on this machine because `python` is not on `PATH`.
- `python3 scripts/setup/check_repo_setup.py`: PASS.
- `python3 scripts/setup/check_imports.py`: PASS.
- `find . -type f -size +10M -print`: no files reported.
- After submodules were added, `python3 scripts/setup/check_repo_setup.py`: PASS, with registered submodule working trees skipped.
- `find . -type f -size +10M -not -path './third_party/*' -print`: no parent-repo files reported.

## Status

PASS for basic setup scaffolding and lightweight repository checks.

## Heavy Tests

Full `pytest` was not run because existing tests may require GPUs, model checkpoints, external repositories, datasets, or API-backed tools.

## Remaining Setup Risks

- The local login environment exposes `python3` as Python 3.6.8 and does not expose a `python` command; project metadata requires Python >=3.10.
- Heavy ML/audio/RL dependencies are intentionally kept out of `pyproject.toml` and are split into optional requirements files.
- Some extracted modules may still need import cleanup before full package-wide imports or full tests are expected to pass.

## Submodules

Submodules have been added under `third_party/` for `Audio-Maestro`, `DeSTA2.5-Audio`, and `ToolRL`. The setup checker treats these registered submodule paths as gitlinks for parent-repo safety checks and still scans normal parent-repo files outside submodules.
