#!/usr/bin/env python3
"""Validate lightweight audio manifest JSONL files without extra dependencies."""

import argparse
import json
import os
import re
import sys
from pathlib import Path


REQUIRED_FIELDS = [
    "sample_id",
    "source_dataset",
    "split",
    "audio_path",
    "task_type",
    "prompt",
    "answer",
]

OPTIONAL_PATH_FIELDS = ["tool_result_path"]
NORMAL_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def is_absolute_or_machine_path(value):
    return (
        os.path.isabs(value)
        or WINDOWS_DRIVE_RE.match(value) is not None
        or "://" in value
    )


def validate_path_field(row, field, line_no, errors, warnings):
    value = row.get(field)
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        errors.append(f"line {line_no}: {field} must be a non-empty string")
        return
    if is_absolute_or_machine_path(value):
        errors.append(f"line {line_no}: {field} must be relative, got {value!r}")
        return
    if field == "audio_path":
        suffix = Path(value).suffix.lower()
        if suffix not in NORMAL_AUDIO_EXTENSIONS:
            warnings.append(
                f"line {line_no}: audio_path extension {suffix!r} is not a common audio extension"
            )


def validate_row(row, line_no):
    errors = []
    warnings = []

    if not isinstance(row, dict):
        return [f"line {line_no}: row must be a JSON object"], warnings

    for field in REQUIRED_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"line {line_no}: required field {field!r} must be a non-empty string")

    validate_path_field(row, "audio_path", line_no, errors, warnings)
    for field in OPTIONAL_PATH_FIELDS:
        validate_path_field(row, field, line_no, errors, warnings)

    if "duration_sec" in row and not isinstance(row["duration_sec"], (int, float)):
        errors.append(f"line {line_no}: duration_sec must be numeric when present")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        errors.append(f"line {line_no}: metadata must be an object when present")

    return errors, warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an MTP2 audio manifest JSONL file.")
    parser.add_argument("manifest", help="Path to JSONL manifest.")
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional schema path. Parsed for existence/JSON validity only; no jsonschema dependency is required.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)

    if args.schema:
        schema_path = Path(args.schema)
        if not schema_path.is_file():
            print(f"ERROR: schema not found: {schema_path}", file=sys.stderr)
            return 1
        try:
            json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"ERROR: schema is not valid JSON: {schema_path}: {exc}", file=sys.stderr)
            return 1

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    errors = []
    warnings = []
    valid_rows = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue

            row_errors, row_warnings = validate_row(row, line_no)
            errors.extend(row_errors)
            warnings.extend(row_warnings)
            if not row_errors:
                valid_rows += 1

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("PASS")
    print(f"valid rows: {valid_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
