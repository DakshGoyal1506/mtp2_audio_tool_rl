#!/usr/bin/env python3
"""Lightweight repository safety check for the clean MTP2 repo."""

import sys
from pathlib import Path


MAX_FILE_BYTES = 10 * 1024 * 1024

EXPECTED_DIRS = [
    "src/mtp2_audio_tool_rl",
    "scripts",
    "configs",
    "tests",
    "docs",
    "patches",
    "manifests",
    "third_party",
]

BLOCKED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".tar",
    ".zip",
    ".sif",
    ".log",
    ".out",
    ".err",
}

BLOCKED_DIR_NAMES = {
    "wandb",
    "runs",
    "outputs",
    "results",
    "logs",
    "old_logs",
    "training_logs",
    "judge_logs",
    "checkpoints",
    "precomputed_embeds",
    "test-mini-audios",
    "__pycache__",
    ".venv",
    ".ipynb_checkpoints",
}

ALLOWED_DIRS = {
    Path("scripts/data"),
    Path("scripts/data/cache_tools"),
    Path("src/mtp2_audio_tool_rl/datasets"),
    Path("configs/datasets"),
}

SKIP_DIR_NAMES = {".git"}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def rel(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def is_allowed_dir(path: Path, root: Path) -> bool:
    relative = rel(path, root)
    return relative in ALLOWED_DIRS or any(
        relative == allowed or allowed in relative.parents for allowed in ALLOWED_DIRS
    )


def main() -> int:
    root = repo_root()
    failures = []

    for expected in EXPECTED_DIRS:
        if not (root / expected).is_dir():
            failures.append(f"missing expected directory: {expected}")

    for path in root.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue

        if path.is_dir():
            if path.name in BLOCKED_DIR_NAMES and not is_allowed_dir(path, root):
                failures.append(f"blocked directory present: {rel(path, root)}")
            continue

        if not path.is_file():
            continue

        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file above 10 MB: {rel(path, root)}")

        if path.suffix.lower() in BLOCKED_EXTENSIONS:
            failures.append(f"blocked artifact extension: {rel(path, root)}")

    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PASS")
    print(f"repo root: {root}")
    print("expected directories present")
    print("no files above 10 MB")
    print("no blocked artifact extensions or directories found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
