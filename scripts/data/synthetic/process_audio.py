#!/usr/bin/env python3
"""
Process all audio files from grpo_tool_dataset.with_relative_audio.jsonl with
AudioCopilot tools and create an extended JSONL with all tool outputs.

Audio files are in dataset/audio/<subdir>/<file>.{wav,mp3}
"""

import json
import asyncio
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import sys

# Audio-Maestro lives in the sibling cache_tools dir
_cache_tools_dir = Path(__file__).resolve().parent.parent.parent / "grpo_dataset" / "cache_tools"
sys.path.insert(0, str(_cache_tools_dir / "Audio-Maestro"))
os.environ["PYANNOTE_AUDIO_DISABLE_TORCHCODEC"] = "1"

from audio_maestro.audio_copilot import AudioCopilotTool

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
DATASET_DIR = _project_dir / "dataset"
AUDIO_DIR = DATASET_DIR / "audio"
INPUT_JSONL = DATASET_DIR / "grpo_tool_dataset.with_relative_audio.jsonl"
OUTPUT_JSONL = _this_dir / "synth_extended.jsonl"


def resolve_audio_path(entry: dict) -> str:
    """Resolve audio path from relative_audio_path field."""
    rel = entry["relative_audio_path"]
    full = DATASET_DIR / rel
    return str(full)


async def process_single_audio(tool: AudioCopilotTool, audio_path: str) -> dict:
    """
    Process a single audio file with all AudioCopilot tools
    (except chord_recognition and stressed_analysis — handled separately).
    """
    results = {}

    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    tools_to_run = [
        ("speech_recognition", tool.speech_recognition),
        ("speaker_diarization", tool.speaker_diarization),
        ("emotion_recognition", tool.emotion_recognition),
        ("sound_classification", tool.sound_classification),
        ("sound_duration_analysis", tool.sound_duration_analysis),
        ("genre_analysis", tool.genre_analysis),
        ("audio_features", tool.get_audio_features),
        ("speech_to_noise_ratio", tool.speech_to_noise_ratio),
    ]

    for tool_name, tool_func in tools_to_run:
        try:
            print(f"  Running {tool_name}...")
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(audio_path)
            else:
                result = tool_func(audio_path)
            results[tool_name] = result
        except Exception as e:
            print(f"  Error in {tool_name}: {str(e)}")
            results[tool_name] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    return results


async def main():
    print(f"Loading {INPUT_JSONL}...")
    data = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries")

    # Initialize AudioCopilotTool
    print("Initializing AudioCopilotTool...")
    tool = AudioCopilotTool()

    # Track unique audio files to avoid processing duplicates
    processed_audios = {}

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as fout:
        for i, entry in enumerate(tqdm(data, desc="Processing audio files")):
            audio_path = resolve_audio_path(entry)

            print(f"\n[{i+1}/{len(data)}] Processing: {entry['relative_audio_path']} -> {audio_path}")

            if audio_path in processed_audios:
                print(f"  Using cached results")
                tool_outputs = processed_audios[audio_path]
            else:
                tool_outputs = await process_single_audio(tool, audio_path)
                processed_audios[audio_path] = tool_outputs

            extended_entry = entry.copy()
            extended_entry["tool_outputs"] = tool_outputs
            fout.write(json.dumps(extended_entry, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone! Processed {len(data)} entries.")
    print(f"Unique audio files processed: {len(processed_audios)}")
    print(f"Output saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
