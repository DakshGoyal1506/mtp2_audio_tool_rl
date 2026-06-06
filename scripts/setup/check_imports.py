#!/usr/bin/env python3
"""Lightweight import check that avoids model, GPU, dataset, and API loading."""

import importlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True


SAFE_IMPORTS = [
    "mtp2_audio_tool_rl",
    "mtp2_audio_tool_rl.datasets",
    "mtp2_audio_tool_rl.evaluation",
    "mtp2_audio_tool_rl.inference",
    "mtp2_audio_tool_rl.prompts",
    "mtp2_audio_tool_rl.rewards",
    "mtp2_audio_tool_rl.tool_calling",
    "mtp2_audio_tool_rl.tool_execution",
    "mtp2_audio_tool_rl.tools",
    "mtp2_audio_tool_rl.utils",
]

SKIPPED_HEAVY = [
    "mtp2_audio_tool_rl.grpo",
    "mtp2_audio_tool_rl.grpo_audio_maestro",
    "mtp2_audio_tool_rl.grpo_single_phase",
    "mtp2_audio_tool_rl.grpo_llm_decoupled",
    "mtp2_audio_tool_rl.desta_vllm",
]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def main() -> int:
    root = repo_root()
    src = root / "src"
    sys.path.insert(0, str(src))

    try:
        importlib.import_module("mtp2_audio_tool_rl")
    except Exception as exc:
        print("FAIL")
        print(f"base package import failed: {exc}")
        return 1

    print("PASS")
    print("base package import succeeded")

    for module in SAFE_IMPORTS:
        try:
            importlib.import_module(module)
            print(f"import ok: {module}")
        except Exception as exc:
            print(f"import warning: {module}: {exc}")

    for module in SKIPPED_HEAVY:
        print(f"skip heavy import: {module} (may load torch, models, GPUs, external repos, or APIs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
