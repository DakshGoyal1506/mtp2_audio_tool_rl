"""Validate canonical tool calls emitted by audio-language models."""

import json
from typing import Any, Iterable, Optional

from mtp2_audio_tool_rl.tool_calling.schema import (
    ToolCall,
    ToolCallValidationResult,
    normalize_tool_name,
    normalized_allowed_tools,
    unsafe_path_fields,
)


def parse_tool_call(raw: Any) -> ToolCallValidationResult:
    """Parse a Python dict or JSON string into an object-like value."""

    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolCallValidationResult(
                valid=False,
                error_code="malformed_json",
                message=f"Tool call is not valid JSON: {exc}",
                errors=("malformed_json",),
            )
        if not isinstance(parsed, dict):
            return ToolCallValidationResult(
                valid=False,
                error_code="not_object",
                message="Tool call must be a JSON object.",
                errors=("not_object",),
            )
        return ToolCallValidationResult(valid=True, message="parsed", data=parsed)

    if isinstance(raw, dict):
        return ToolCallValidationResult(valid=True, message="parsed", data=raw)

    return ToolCallValidationResult(
        valid=False,
        error_code="not_object",
        message="Tool call must be a JSON object or Python dict.",
        errors=("not_object",),
    )


def validate_tool_call(raw: Any, allowed_tools: Optional[Iterable[str]] = None) -> ToolCallValidationResult:
    """Validate a JSON string or dict tool call."""

    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            return ToolCallValidationResult(
                valid=False,
                error_code="malformed_json",
                message=f"Tool call is not valid JSON: {exc}",
                errors=("malformed_json",),
            )
    else:
        data = raw

    return validate_tool_call_dict(data, allowed_tools=allowed_tools)


def validate_tool_call_dict(
    data: Any,
    allowed_tools: Optional[Iterable[str]] = None,
) -> ToolCallValidationResult:
    """Validate a dict against the canonical tool-call schema."""

    if not isinstance(data, dict):
        return ToolCallValidationResult(
            valid=False,
            error_code="not_object",
            message="Tool call must be an object.",
            errors=("not_object",),
        )

    if "tool_name" not in data:
        return ToolCallValidationResult(
            valid=False,
            error_code="missing_tool_name",
            message="Missing required field: tool_name.",
            errors=("missing_tool_name",),
        )

    raw_tool_name = data["tool_name"]
    if not isinstance(raw_tool_name, str):
        return ToolCallValidationResult(
            valid=False,
            error_code="invalid_tool_name_type",
            message="tool_name must be a string.",
            errors=("invalid_tool_name_type",),
        )

    tool_name = normalize_tool_name(raw_tool_name)
    if tool_name not in normalized_allowed_tools(allowed_tools):
        return ToolCallValidationResult(
            valid=False,
            error_code="unknown_tool",
            message=f"Unknown tool: {raw_tool_name!r}.",
            errors=("unknown_tool",),
        )

    if "arguments" not in data:
        return ToolCallValidationResult(
            valid=False,
            error_code="missing_arguments",
            message="Missing required field: arguments.",
            errors=("missing_arguments",),
        )

    arguments = data["arguments"]
    if not isinstance(arguments, dict):
        return ToolCallValidationResult(
            valid=False,
            error_code="invalid_arguments_type",
            message="arguments must be an object.",
            errors=("invalid_arguments_type",),
        )

    unsafe_fields = unsafe_path_fields(arguments)
    if unsafe_fields:
        return ToolCallValidationResult(
            valid=False,
            error_code="unsafe_path",
            message=f"Unsafe path argument(s): {', '.join(sorted(unsafe_fields))}.",
            errors=("unsafe_path",),
        )

    return ToolCallValidationResult(
        valid=True,
        message="Tool call is valid.",
        tool_call=ToolCall(tool_name=tool_name, arguments=dict(arguments)),
    )
