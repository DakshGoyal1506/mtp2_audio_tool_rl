# Quickstart

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/DakshGoyal1506/mtp2_audio_tool_rl.git
cd mtp2_audio_tool_rl
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the minimal base setup:

```bash
python3 -m pip install -e .
python3 -m pip install -r requirements/base.txt
```

Run the lightweight checks:

```bash
python3 scripts/setup/check_repo_setup.py
python3 scripts/setup/check_imports.py
python3 scripts/data/validate_manifest.py manifests/example_manifest.jsonl
bash scripts/setup/apply_patches_preview.sh
```

Optional heavier dependency sets:

- `requirements/audio.txt`
- `requirements/rl.txt`

Those optional sets are not required for the default repo checks and may pull in substantial ML/audio dependencies.
