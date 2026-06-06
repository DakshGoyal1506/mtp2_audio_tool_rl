#!/usr/bin/env python3
"""
Process all audio files in mmau-test-mini-extended.json with stress analysis
and create an extended JSON file with stress analysis outputs.
Uses pre-generated Whisper transcriptions from the extended file.
Sequential execution - one audio at a time.
"""

import json
import asyncio
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
os.environ["PYANNOTE_AUDIO_DISABLE_TORCHCODEC"] = "1"

from audio_maestro.audio_copilot import AudioCopilotTool


async def process_single_audio_stress(tool: AudioCopilotTool, audio_path: str, transcript: str = None) -> dict:
    """
    Process a single audio file with stress analysis.

    Args:
        tool: AudioCopilotTool instance
        audio_path: Path to the audio file
        transcript: Optional pre-generated transcript from extended JSON

    Returns:
        Dictionary containing stress analysis results
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    try:
        print(f"  Running stressed_analysis...")

        # Check if stressed_analysis is async
        if asyncio.iscoroutinefunction(tool.stressed_analysis):
            result = await tool.stressed_analysis(audio_path, transcript=transcript)
        else:
            result = tool.stressed_analysis(audio_path, transcript=transcript)

        return result

    except Exception as e:
        print(f"  Error in stressed_analysis: {str(e)}")
        return {
            "function": "stressed_analysis",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


async def main():
    """Main function to process all audio files with stress analysis"""

    # Paths
    base_dir = Path(__file__).parent
    input_json = base_dir / "mmau-test-mini-extended.json"  # Use extended file with transcriptions
    output_json = base_dir / "mmau-test-mini-stress.json"

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

    for i, entry in enumerate(tqdm(data, desc="Processing audio files with stress analysis")):
        audio_id = entry.get("audio_id", "")

        # Resolve audio path
        if audio_id.startswith("./"):
            audio_path = str(base_dir / audio_id[2:])
        else:
            audio_path = str(base_dir / audio_id)

        # Extract pre-generated transcript from the extended JSON
        transcript = None
        tool_outputs = entry.get("tool_outputs", {})
        speech_recognition = tool_outputs.get("speech_recognition", {})
        if speech_recognition and "text" in speech_recognition:
            transcript = speech_recognition.get("text", "")
            if transcript:
                print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (using pre-generated transcript)")
            else:
                print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (no transcript available)")
        else:
            print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (no speech_recognition in tool_outputs)")

        # Check if we've already processed this audio
        if audio_path in processed_audios:
            print(f"  Using cached results for {audio_id}")
            stress_result = processed_audios[audio_path]
        else:
            # Process the audio with stress analysis, passing the pre-generated transcript
            stress_result = await process_single_audio_stress(tool, audio_path, transcript=transcript)
            processed_audios[audio_path] = stress_result

        # Create extended entry
        extended_entry = entry.copy()
        extended_entry["tool_outputs"] = entry.get("tool_outputs", {}).copy()
        extended_entry["tool_outputs"]["stressed_analysis"] = stress_result
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