#!/usr/bin/env python3
"""
Process all audio files from grpo_tool_dataset with autochord and create
a JSONL with chord recognition outputs.
"""

import json
import asyncio
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import sys

_cache_tools_dir = Path(__file__).resolve().parent.parent.parent / "grpo_dataset" / "cache_tools"
sys.path.insert(0, str(_cache_tools_dir / "Audio-Maestro"))

import autochord

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
DATASET_DIR = _project_dir / "dataset"
INPUT_JSONL = DATASET_DIR / "grpo_tool_dataset.with_relative_audio.jsonl"
OUTPUT_JSONL = _this_dir / "synth_chord.jsonl"


def resolve_audio_path(entry: dict) -> str:
    rel = entry["relative_audio_path"]
    return str(DATASET_DIR / rel)


async def process_single_audio_autochord(audio_path: str) -> dict:
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    try:
        print(f"  Running autochord...")
        base_name = Path(audio_path).stem
        lab_file = Path(audio_path).parent / f"{base_name}_chords.lab"

        if lab_file.exists():
            print(f"  Using cached chords: {lab_file}")
        else:
            await asyncio.to_thread(autochord.recognize, audio_path, lab_fn=str(lab_file))

        chord_changes = []
        with open(lab_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, chord = parts
                    chord_changes.append({
                        "time": float(start),
                        "end_time": float(end),
                        "chord": chord,
                    })

        chord_changes_str = [
            f"{c['time']:.2f}s-{c['end_time']:.2f}s: {c['chord']}" for c in chord_changes[:100]
        ]

        return {
            "function": "chord_recognition",
            "chord_changes": chord_changes_str,
            "total_chords": len(chord_changes),
        }

    except Exception as e:
        print(f"  Error in autochord: {str(e)}")
        return {
            "function": "chord_recognition",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


async def main():
    print(f"Loading {INPUT_JSONL}...")
    data = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries")

    processed_audios = {}

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for i, entry in enumerate(tqdm(data, desc="Processing autochord")):
            audio_path = resolve_audio_path(entry)

            print(f"\n[{i+1}/{len(data)}] Processing: {entry['relative_audio_path']}")

            if audio_path in processed_audios:
                print(f"  Using cached results")
                tool_outputs = processed_audios[audio_path]
            else:
                tool_outputs = await process_single_audio_autochord(audio_path)
                processed_audios[audio_path] = tool_outputs

            extended_entry = entry.copy()
            extended_entry["tool_outputs"] = {"chord_recognition": tool_outputs}
            fout.write(json.dumps(extended_entry, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone! Processed {len(data)} entries.")
    print(f"Unique audio files processed: {len(processed_audios)}")
    print(f"Output saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
