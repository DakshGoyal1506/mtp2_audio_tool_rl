import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "data" / "validate_manifest.py"


def write_manifest(path: Path, rows):
    path.write_text(
        "\n".join(json.dumps(row) if not isinstance(row, str) else row for row in rows) + "\n",
        encoding="utf-8",
    )


def run_validator(path: Path):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )


def valid_row(**overrides):
    row = {
        "sample_id": "example-001",
        "source_dataset": "example",
        "split": "test",
        "audio_path": "external_audio/example_001.wav",
        "task_type": "qa",
        "prompt": "What is happening?",
        "answer": "A placeholder answer.",
    }
    row.update(overrides)
    return row


def test_valid_jsonl_manifest_passes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row(), valid_row(sample_id="example-002")])

    result = run_validator(manifest)

    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "valid rows: 2" in result.stdout


def test_missing_required_field_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    row = valid_row()
    del row["answer"]
    write_manifest(manifest, [row])

    result = run_validator(manifest)

    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "required field 'answer'" in result.stderr


def test_absolute_audio_path_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row(audio_path="/data/example_001.wav")])

    result = run_validator(manifest)

    assert result.returncode != 0
    assert "audio_path must be relative" in result.stderr


def test_absolute_tool_result_path_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row(tool_result_path="/outputs/tool_result.json")])

    result = run_validator(manifest)

    assert result.returncode != 0
    assert "tool_result_path must be relative" in result.stderr


def test_invalid_json_line_fails(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row(), "{not valid json"])

    result = run_validator(manifest)

    assert result.returncode != 0
    assert "invalid JSON" in result.stderr


def test_non_audio_extension_warns_but_passes(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row(audio_path="external_audio/example_001.txt")])

    result = run_validator(manifest)

    assert result.returncode == 0
    assert "PASS" in result.stdout
    assert "WARNING" in result.stderr
    assert "not a common audio extension" in result.stderr
