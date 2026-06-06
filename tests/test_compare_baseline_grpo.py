import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "eval" / "compare_baseline_grpo.py"


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compare_baseline_grpo_outputs(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    grpo = tmp_path / "grpo.jsonl"
    out_dir = tmp_path / "compare"

    write_jsonl(
        manifest,
        [
            {"sample_id": "s1", "answer": "two speakers"},
            {"sample_id": "s2", "answer": "happy"},
        ],
    )
    write_jsonl(
        baseline,
        [
            {"sample_id": "s1", "prediction": "one speaker"},
            {"sample_id": "s2", "prediction": "happy"},
        ],
    )
    write_jsonl(
        grpo,
        [
            {
                "sample_id": "s1",
                "final_answer": "two speakers",
                "tool_call_present": True,
                "tool_call_valid": True,
                "tool_result_used": True,
            },
            {
                "sample_id": "s2",
                "final_answer": "sad",
                "tool_call_present": True,
                "tool_call_valid": False,
                "tool_result_used": False,
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--manifest",
            str(manifest),
            "--baseline",
            str(baseline),
            "--grpo",
            str(grpo),
            "--out-dir",
            str(out_dir),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "comparison.csv").is_file()
    assert (out_dir / "failure_examples.jsonl").is_file()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_samples"] == 2
    assert metrics["delta_exact_match"] == 0.0
    assert metrics["grpo_tool_call_rate"] == 1.0
    assert metrics["grpo_valid_tool_call_rate"] == 0.5
    assert metrics["grpo_invalid_tool_call_rate"] == 0.5
    assert metrics["grpo_tool_result_used_rate"] == 0.5
