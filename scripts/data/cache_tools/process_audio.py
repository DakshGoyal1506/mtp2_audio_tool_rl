#!/usr/bin/env python3
"""
Process all audio files in mmau-test-mini.json with AudioCopilot tools
and create an extended JSON file with all tool outputs.
"""

import json
import asyncio
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import traceback
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from audio_maestro.audio_copilot import AudioCopilotTool


async def process_single_audio(tool: AudioCopilotTool, audio_path: str, task_type: str = None) -> dict:
    """
    Process a single audio file with all AudioCopilot tools (except chord_recognition).

    Args:
        tool: AudioCopilotTool instance
        audio_path: Path to the audio file
        task_type: Optional task type to determine which tools to run

    Returns:
        Dictionary containing results from all tools
    """
    results = {}

    # Check if file exists
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    # Tool functions to execute (excluding chord_recognition)
    tools_to_run = [
        ("speech_recognition", tool.speech_recognition),
        ("speaker_diarization", tool.speaker_diarization),
        ("emotion_recognition", tool.emotion_recognition),
        # ("stressed_analysis", tool.stressed_analysis),
        ("sound_classification", tool.sound_classification),
        ("sound_duration_analysis", tool.sound_duration_analysis),
        ("genre_analysis", tool.genre_analysis),
        ("audio_features", tool.get_audio_features),
        ("speech_to_noise_ratio", tool.speech_to_noise_ratio),
    ]

    for tool_name, tool_func in tools_to_run:
        try:
            print(f"  Running {tool_name}...")

            # Most tools are async
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

    # Paths
    base_dir = Path(__file__).parent
    input_json = base_dir / "mmau-test-mini.json"
    output_json = base_dir / "mmau-test-mini-extended.json"

    # Load input JSON
    print(f"Loading {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")

    # Initialize AudioCopilotTool
    print("Initializing AudioCopilotTool...")
    tool = AudioCopilotTool()

    # Track unique audio files to avoid processing duplicates
    processed_audios = {}

    # Process each entry
    extended_data = []

    for i, entry in enumerate(tqdm(data, desc="Processing audio files")):
        audio_id = entry.get("audio_id", "")

        # Resolve audio path
        if audio_id.startswith("./"):
            audio_path = str(base_dir / audio_id[2:])
        else:
            audio_path = str(base_dir / audio_id)

        print(f"\n[{i+1}/{len(data)}] Processing: {audio_id}")

        # Check if we've already processed this audio
        if audio_path in processed_audios:
            print(f"  Using cached results for {audio_id}")
            tool_outputs = processed_audios[audio_path]
        else:
            # Process the audio with all tools
            tool_outputs = await process_single_audio(
                tool,
                audio_path,
                task_type=entry.get("task")
            )
            processed_audios[audio_path] = tool_outputs

        # Create extended entry
        extended_entry = entry.copy()
        extended_entry["tool_outputs"] = tool_outputs
        extended_data.append(extended_entry)

        # Save intermediate results every 10 entries
        if (i + 1) % 10 == 0:
            print(f"\nSaving intermediate results ({i+1} entries)...")
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(extended_data, f, indent=2, ensure_ascii=False)

    # Save final results
    print(f"\nSaving final results to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extended_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Processed {len(data)} entries.")
    print(f"Unique audio files processed: {len(processed_audios)}")
    print(f"Output saved to: {output_json}")


if __name__ == "__main__":
    asyncio.run(main())
