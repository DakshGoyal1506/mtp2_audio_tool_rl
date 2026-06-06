#!/usr/bin/env python3
"""Lightweight repository safety check for the clean MTP2 repo."""

import configparser
import os
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


def load_submodule_paths(root: Path):
    gitmodules = root / ".gitmodules"
    if not gitmodules.is_file():
        return []

    parser = configparser.ConfigParser()
    parser.read(str(gitmodules))

    submodules = []
    for section in parser.sections():
        if not section.startswith("submodule "):
            continue
        if not parser.has_option(section, "path"):
            continue
        raw_path = parser.get(section, "path").strip()
        if raw_path:
            submodules.append(Path(raw_path))
    return sorted(set(submodules), key=lambda item: str(item))


def is_submodule_path(relative: Path, submodule_paths) -> bool:
    return relative in submodule_paths


def main() -> int:
    root = repo_root()
    failures = []
    submodule_paths = load_submodule_paths(root)

    for expected in EXPECTED_DIRS:
        if not (root / expected).is_dir():
            failures.append(f"missing expected directory: {expected}")

    for submodule_path in submodule_paths:
        if not (root / submodule_path).is_dir():
            failures.append(f".gitmodules references missing submodule path: {submodule_path}")

    for current_root, dirs, files in os.walk(str(root)):
        current = Path(current_root)
        relative_current = rel(current, root)

        for dirname in list(dirs):
            child = current / dirname
            relative_child = rel(child, root)

            if dirname in SKIP_DIR_NAMES:
                dirs.remove(dirname)
                continue

            if is_submodule_path(relative_child, submodule_paths):
                print(f"Skipping submodule working tree: {relative_child}")
                dirs.remove(dirname)
                continue

            if child.name in BLOCKED_DIR_NAMES and not is_allowed_dir(child, root):
                failures.append(f"blocked directory present: {relative_child}")
                dirs.remove(dirname)

        if relative_current in submodule_paths:
            continue

        for filename in files:
            path = current / filename

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
    if submodule_paths:
        print("registered submodule paths exist")
    print("no files above 10 MB")
    print("no blocked artifact extensions or directories found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
