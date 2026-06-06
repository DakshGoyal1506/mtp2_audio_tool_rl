# CI Status

GitHub Actions runs a lightweight repository check on `push` and `pull_request`.

The workflow checks:

- repository safety rules
- lightweight package imports
- the tiny example manifest
- lightweight pytest tests for manifest validation and repository setup helpers
- patch dry-runs against checked-out submodules
- parent-repo large files above 10 MB

Heavy ML/audio dependencies such as `torch`, `torchaudio`, `transformers`, `datasets`, `deepspeed`, `trl`, and `wandb` are intentionally excluded.

Full `pytest` is intentionally not run yet because existing tests may require GPUs, model checkpoints, external datasets, submodules, or heavyweight dependencies. CI currently runs only:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_validate_manifest.py tests/test_repo_setup_checks.py
```

Submodules are checked out only so patch dry-run validation can run. CI should not contain datasets, audio, checkpoints, generated results, logs, or model weights.
