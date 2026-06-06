#!/usr/bin/env python3
"""
Process all audio files in mmau-test-mini.json with autochord
and create an extended JSON file with chord recognition outputs.
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

import autochord


async def process_single_audio_autochord(audio_path: str) -> dict:
    """
    Process a single audio file with autochord for chord recognition.

    Args:
        audio_path: Path to the audio file

    Returns:
        Dictionary containing chord recognition results
    """
    # Check if file exists
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

        # Parse lab file
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
    """Main function to process all audio files with autochord"""

    # Paths
    base_dir = Path(__file__).parent
    input_json = base_dir / "mmau-test-mini.json"
    output_json = base_dir / "mmau-test-mini-autochord.json"

    # Load input JSON
    print(f"Loading {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")

    # Track unique audio files to avoid processing duplicates
    processed_audios = {}

    # Process each entry
    extended_data = []

    for i, entry in enumerate(tqdm(data, desc="Processing audio files with autochord")):
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
            # Process the audio with autochord
            tool_outputs = await process_single_audio_autochord(audio_path)
            processed_audios[audio_path] = tool_outputs

        # Create extended entry
        extended_entry = entry.copy()
        extended_entry["tool_outputs"] = {"chord_recognition": tool_outputs}
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
