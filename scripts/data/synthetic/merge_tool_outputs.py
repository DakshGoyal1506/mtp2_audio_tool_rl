#!/usr/bin/env python3
"""
Merge tool outputs from the three processing steps into a single cached JSONL.

Reads:
  - synth_stress.jsonl  (has speech_recognition, diarization, emotion, sound_classification,
                          sound_duration_analysis, genre_analysis, audio_features,
                          speech_to_noise_ratio, stressed_analysis)
  - synth_chord.jsonl   (has chord_recognition)

Produces:
  - synth_cached.jsonl  (all tool_outputs merged per entry)
"""

import json
from pathlib import Path

_this_dir = Path(__file__).resolve().parent

STRESS_JSONL = _this_dir / "synth_stress.jsonl"
CHORD_JSONL = _this_dir / "synth_chord.jsonl"
OUTPUT_JSONL = _this_dir / "synth_cached.jsonl"


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    print(f"Loading {STRESS_JSONL}...")
    stress_data = load_jsonl(STRESS_JSONL)
    print(f"  {len(stress_data)} entries")

    print(f"Loading {CHORD_JSONL}...")
    chord_data = load_jsonl(CHORD_JSONL)
    print(f"  {len(chord_data)} entries")

    # Build chord lookup by row_index
    chord_by_row = {}
    for entry in chord_data:
        row = entry.get("row_index", None)
        chord_output = entry.get("tool_outputs", {}).get("chord_recognition", {})
        if row is not None:
            chord_by_row[row] = chord_output

    merged = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for entry in stress_data:
            row = entry.get("row_index", None)
            tool_outputs = entry.get("tool_outputs", {}).copy()

            # Merge chord recognition
            if row in chord_by_row:
                tool_outputs["chord_recognition"] = chord_by_row[row]
                merged += 1

            entry_out = entry.copy()
            entry_out["tool_outputs"] = tool_outputs
            fout.write(json.dumps(entry_out, ensure_ascii=False) + "\n")

    print(f"\nMerged {merged} chord results into {len(stress_data)} entries")
    print(f"Output saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
