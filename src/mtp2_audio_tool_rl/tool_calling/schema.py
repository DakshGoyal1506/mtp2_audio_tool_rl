"""Canonical tool-call schema helpers for debug GRPO experiments."""

import os
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple


DEFAULT_ALLOWED_TOOLS = (
    "speaker_diarization",
    "emotion_recognition",
    "chord_recognition",
    "audio_captioning",
    "audio_event_detection",
)

WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class ToolCall(NamedTuple):
    """Validated tool call with a normalized tool name."""

    tool_name: str
    arguments: Dict[str, Any]


class ToolCallValidationResult(NamedTuple):
    """Structured validation result for model-emitted tool calls."""

    valid: bool
    error_code: Optional[str] = None
    message: str = ""
    tool_call: Optional[ToolCall] = None
    errors: Tuple[str, ...] = ()
    data: Optional[Dict[str, Any]] = None


def normalize_tool_name(tool_name: str) -> str:
    """Normalize model-emitted tool names to the canonical snake_case form."""

    return tool_name.strip().lower().replace("-", "_").replace(" ", "_")


def is_relative_safe_path(value: Any) -> bool:
    """Return True when a path-like value is relative and has no parent traversal."""

    if not isinstance(value, str):
        return False

    candidate = value.strip()
    if not candidate:
        return False
    if "://" in candidate or os.path.isabs(candidate) or WINDOWS_DRIVE_RE.match(candidate):
        return False

    normalized = candidate.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(candidate)

    if posix_path.is_absolute() or windows_path.is_absolute():
        return False
    if any(part == ".." for part in posix_path.parts):
        return False

    return True


def normalized_allowed_tools(allowed_tools: Optional[Iterable[str]] = None) -> set:
    """Normalize an allowed-tool collection for membership checks."""

    tools = DEFAULT_ALLOWED_TOOLS if allowed_tools is None else allowed_tools
    return {normalize_tool_name(tool) for tool in tools}


def unsafe_path_fields(arguments: Mapping[str, Any]) -> List[str]:
    """Return argument keys with unsafe path-like values."""

    unsafe = []
    for key, value in arguments.items():
        if "path" in str(key).lower() and not is_relative_safe_path(value):
            unsafe.append(str(key))
    return unsafe
