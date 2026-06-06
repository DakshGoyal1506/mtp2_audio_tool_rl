#!/usr/bin/env python3
"""
Process all audio files from wavecaps grpo_tool_dataset.jsonl with AudioCopilot tools
and create an extended JSONL file with all tool outputs.
"""

import json
import asyncio
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Audio-Maestro"))
os.environ["PYANNOTE_AUDIO_DISABLE_TORCHCODEC"] = "1"

from audio_maestro.audio_copilot import AudioCopilotTool


def resolve_audio_path(audio_id: str, audio_dir: Path) -> str:
    """
    Resolve audio ID like 'WavCaps_Freesound30s/101253.flac' to actual wav path.
    Audio files are stored as audio/101253.wav
    """
    basename = Path(audio_id).stem  # e.g. '101253'
    wav_path = audio_dir / f"{basename}.wav"
    if wav_path.exists():
        return str(wav_path)
    # Try flac as fallback
    flac_path = audio_dir / f"{basename}.flac"
    if flac_path.exists():
        return str(flac_path)
    return str(wav_path)  # Return wav path even if missing (will error later)


async def process_single_audio(tool: AudioCopilotTool, audio_path: str) -> dict:
    """
    Process a single audio file with all AudioCopilot tools (except chord_recognition and stressed_analysis).
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
    """Main function to process all audio files"""

    base_dir = Path(__file__).parent
    audio_dir = base_dir.parent / "wavecaps" / "wavecaps_freesound" / "audio"
    input_jsonl = base_dir.parent / "wavecaps" / "wavecaps_freesound" / "output" / "grpo_tool_dataset.jsonl"
    output_jsonl = base_dir / "wavecaps_extended.jsonl"

    # Load input JSONL
    print(f"Loading {input_jsonl}...")
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
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

    # Open output file for streaming writes
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for i, entry in enumerate(tqdm(data, desc="Processing audio files")):
            audio_id = entry.get("id", "")
            audio_path = resolve_audio_path(audio_id, audio_dir)

            print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} -> {audio_path}")

            if audio_path in processed_audios:
                print(f"  Using cached results for {audio_id}")
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
    print(f"Output saved to: {output_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
