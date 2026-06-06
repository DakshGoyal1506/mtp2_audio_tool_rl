#!/usr/bin/env python3
"""
Convert synth_cached.jsonl to the format expected by the GRPO trainer.

Input fields:  correct_answer, options, tool, relative_audio_path, tool_outputs
Output fields: answer, choices, task, audio_id, tool_outputs (+ all original fields)

Mapping:
  correct_answer -> answer
  options        -> choices
  tool           -> tool (kept), task (derived from tool)
  relative_audio_path -> audio_id (prefixed with "dataset/")
"""

import json
import os
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

INPUT_JSONL = _this_dir / "synth_cached.jsonl"
OUTPUT_JSON = _this_dir / "synth_cached_train.json"

# Map tool name -> task category (sound / speech / music)
TOOL_TO_TASK = {
    "sound_classification":     "sound",
    "sound_duration_analysis":  "sound",
    "speech_recognition":       "speech",
    "speaker_diarization":      "speech",
    "emotion_recognition":      "speech",
    "speech_to_noise_ratio":    "speech",
    "stressed_analysis":        "speech",
    "genre_analysis":           "music",
    "get_audio_features":       "music",
    "chord_recognition":        "music",
}


def main():
    print(f"Loading {INPUT_JSONL}...")
    entries = []
    with open(INPUT_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries")

    converted = []
    for entry in entries:
        item = dict(entry)

        # Field mappings
        item["answer"] = item.pop("correct_answer", "")
        item["choices"] = item.pop("options", [])

        # Derive task from tool
        tool = item.get("tool", "")
        item["task"] = TOOL_TO_TASK.get(tool, "sound")

        # audio_id: path relative to project root (e.g. "dataset/audio/Anispeech/xxx.wav")
        rel = item.get("relative_audio_path", "")
        item["audio_id"] = os.path.join("dataset", rel) if rel else ""

        converted.append(item)

    # Verify
    missing = [c for c in converted if not c.get("task") or not c.get("question") or not c.get("answer")]
    if missing:
        print(f"WARNING: {len(missing)} items missing task/question/answer")

    # Task distribution
    tasks = {}
    for c in converted:
        t = c["task"]
        tasks[t] = tasks.get(t, 0) + 1
    print(f"Task distribution: {tasks}")

    # Save as JSON array
    with open(OUTPUT_JSON, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(converted)} items to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
