#!/usr/bin/env python3
"""Compare baseline and GRPO JSONL outputs with lightweight metrics."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtp2_audio_tool_rl.rewards.tool_use_reward import exact_match_reward


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def index_by_sample_id(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            raise ValueError(f"{label}: row missing sample_id")
        indexed[str(sample_id)] = row
    return indexed


def prediction_text(row: Dict[str, Any]) -> Any:
    return row.get("prediction", row.get("final_answer", ""))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare(manifest_path: Path, baseline_path: Path, grpo_path: Path, out_dir: Path) -> Dict[str, Any]:
    manifest = index_by_sample_id(read_jsonl(manifest_path), "manifest")
    baseline = index_by_sample_id(read_jsonl(baseline_path), "baseline")
    grpo = index_by_sample_id(read_jsonl(grpo_path), "grpo")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    failure_rows = []

    baseline_scores = []
    grpo_scores = []
    tool_call_present_values = []
    valid_tool_values = []
    invalid_tool_values = []
    tool_result_used_values = []
    missing_baseline = []
    missing_grpo = []

    for sample_id, gold_row in manifest.items():
        gold_answer = gold_row.get("answer", "")
        baseline_row = baseline.get(sample_id)
        grpo_row = grpo.get(sample_id)

        if baseline_row is None:
            missing_baseline.append(sample_id)
        if grpo_row is None:
            missing_grpo.append(sample_id)

        baseline_prediction = prediction_text(baseline_row or {})
        grpo_prediction = prediction_text(grpo_row or {})
        baseline_em = exact_match_reward(baseline_prediction, gold_answer)
        grpo_em = exact_match_reward(grpo_prediction, gold_answer)

        baseline_scores.append(baseline_em)
        grpo_scores.append(grpo_em)

        tool_call_present = as_bool((grpo_row or {}).get("tool_call_present", False))
        tool_call_valid_raw = (grpo_row or {}).get("tool_call_valid")
        tool_result_used_raw = (grpo_row or {}).get("tool_result_used")

        tool_call_present_values.append(1.0 if tool_call_present else 0.0)
        if tool_call_present and tool_call_valid_raw is not None:
            tool_call_valid = as_bool(tool_call_valid_raw)
            valid_tool_values.append(1.0 if tool_call_valid else 0.0)
            invalid_tool_values.append(0.0 if tool_call_valid else 1.0)
        if tool_result_used_raw is not None:
            tool_result_used_values.append(1.0 if as_bool(tool_result_used_raw) else 0.0)

        row = {
            "sample_id": sample_id,
            "gold_answer": gold_answer,
            "baseline_prediction": baseline_prediction,
            "grpo_prediction": grpo_prediction,
            "baseline_exact_match": baseline_em,
            "grpo_exact_match": grpo_em,
            "tool_call_present": tool_call_present,
            "tool_call_valid": tool_call_valid_raw,
            "tool_result_used": tool_result_used_raw,
        }
        comparison_rows.append(row)

        if baseline_em != grpo_em or baseline_row is None or grpo_row is None:
            failure_rows.append(row)

    metrics = {
        "num_samples": len(manifest),
        "baseline_exact_match": mean(baseline_scores),
        "grpo_exact_match": mean(grpo_scores),
        "delta_exact_match": mean(grpo_scores) - mean(baseline_scores),
        "grpo_tool_call_rate": mean(tool_call_present_values),
        "grpo_valid_tool_call_rate": mean(valid_tool_values),
        "grpo_invalid_tool_call_rate": mean(invalid_tool_values),
        "missing_baseline_predictions": len(missing_baseline),
        "missing_grpo_predictions": len(missing_grpo),
    }
    if tool_result_used_values:
        metrics["grpo_tool_result_used_rate"] = mean(tool_result_used_values)

    write_outputs(out_dir, metrics, comparison_rows, failure_rows)
    return metrics


def write_outputs(
    out_dir: Path,
    metrics: Dict[str, Any],
    comparison_rows: List[Dict[str, Any]],
    failure_rows: List[Dict[str, Any]],
) -> None:
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "sample_id",
        "gold_answer",
        "baseline_prediction",
        "grpo_prediction",
        "baseline_exact_match",
        "grpo_exact_match",
        "tool_call_present",
        "tool_call_valid",
        "tool_result_used",
    ]
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (out_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in failure_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and GRPO JSONL outputs.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--grpo", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = compare(args.manifest, args.baseline, args.grpo, args.out_dir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
