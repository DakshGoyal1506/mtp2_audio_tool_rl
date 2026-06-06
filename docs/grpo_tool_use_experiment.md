# GRPO Tool-Use Smoke Experiment

This experiment layer provides a minimal, reproducible path for testing tool-call formatting, deterministic reward breakdowns, and baseline-vs-GRPO comparison before wiring a full training run.

## Purpose

The goal is to make tool-use experiments debuggable without loading models, GPUs, datasets, or audio. The new code validates model-emitted tool calls, scores simple answer/tool-use behavior, and compares external prediction files.

## Tool-Call Schema

The canonical tool-call format is:

```json
{
  "tool_name": "speaker_diarization",
  "arguments": {
    "audio_path": "audio/sample_001.wav"
  }
}
```

Validation checks:

- `tool_name` is present, string-valued, normalized, and allowed.
- `arguments` is present and object-valued.
- path-like arguments such as `audio_path` and `tool_result_path` are relative paths.
- absolute paths, URLs, Windows drive paths, and parent traversal paths are rejected.

The default allowed tools are `speaker_diarization`, `emotion_recognition`, `chord_recognition`, `audio_captioning`, and `audio_event_detection`.

## Reward Breakdown

The debug reward module measures:

- exact answer match after normalization
- valid tool-call bonus when a tool call is present
- invalid tool-call penalty
- helpfulness reward when a needed tool result is used
- unnecessary tool-use penalty when a tool was not needed

This is intentionally deterministic and simple. It is for smoke testing and reproducible scoring, not the final research reward.

## Debug Dry-Run

Prepare an external manifest outside Git, then run:

```bash
bash scripts/train/grpo/run_grpo_debug.sh
```

Useful environment variables:

- `MTP2_REPO_ROOT`
- `MTP2_EXTERNAL_ROOT`
- `MTP2_MANIFEST_ROOT`
- `MTP2_OUTPUT_ROOT`
- `GRPO_TOOL_USE_MANIFEST`
- `GRPO_TOOL_USE_OUTPUT_DIR`
- `GRPO_TOOL_USE_LOG_DIR`

Default behavior is dry-run only. It validates the manifest, runs a small tool-call/reward sanity check, and prints the training command placeholder. It does not start a large training run.

## Compare Baseline And GRPO Outputs

Use external JSONL prediction files:

```bash
python3 scripts/eval/compare_baseline_grpo.py \
  --manifest /path/to/manifest.jsonl \
  --baseline /path/to/baseline_outputs.jsonl \
  --grpo /path/to/grpo_outputs.jsonl \
  --out-dir /path/to/output/compare
```

The comparison script writes:

- `metrics.json`
- `comparison.csv`
- `failure_examples.jsonl`

Outputs must stay outside Git unless they are tiny documentation examples.

## Remaining Training Work

- choose the canonical GRPO training entrypoint
- connect `RUN_ACTUAL_GRPO=1` to that entrypoint
- generate baseline and GRPO prediction JSONL files using external manifests
- record results using [results_table_template.md](results_table_template.md)
