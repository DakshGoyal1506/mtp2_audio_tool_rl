#!/usr/bin/env python3
"""
Process all audio files from wavecaps_extended.jsonl with stress analysis.
Uses pre-generated Whisper transcriptions from the extended file.
"""

import json
import asyncio
import os
from pathlib import Path
from tqdm import tqdm
import traceback
import sys

sys.path.insert(0, str(Path(__file__).parent / "Audio-Maestro"))
os.environ["PYANNOTE_AUDIO_DISABLE_TORCHCODEC"] = "1"

from audio_maestro.audio_copilot import AudioCopilotTool


def resolve_audio_path(audio_id: str, audio_dir: Path) -> str:
    basename = Path(audio_id).stem
    wav_path = audio_dir / f"{basename}.wav"
    if wav_path.exists():
        return str(wav_path)
    flac_path = audio_dir / f"{basename}.flac"
    if flac_path.exists():
        return str(flac_path)
    return str(wav_path)


async def process_single_audio_stress(tool: AudioCopilotTool, audio_path: str, transcript: str = None) -> dict:
    if not os.path.exists(audio_path):
        return {"error": f"Audio file not found: {audio_path}"}

    try:
        print(f"  Running stressed_analysis...")
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
    base_dir = Path(__file__).parent
    audio_dir = base_dir.parent / "wavecaps" / "wavecaps_freesound" / "audio"
    input_jsonl = base_dir / "wavecaps_extended.jsonl"
    output_jsonl = base_dir / "wavecaps_stress.jsonl"

    print(f"Loading {input_jsonl}...")
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} entries")

    print("Initializing AudioCopilotTool...")
    tool = AudioCopilotTool()

    processed_audios = {}

    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for i, entry in enumerate(tqdm(data, desc="Processing stress analysis")):
            audio_id = entry.get("id", "")
            audio_path = resolve_audio_path(audio_id, audio_dir)

            # Extract pre-generated transcript
            transcript = None
            tool_outputs = entry.get("tool_outputs", {})
            speech_rec = tool_outputs.get("speech_recognition", {})
            if speech_rec and "text" in speech_rec:
                transcript = speech_rec.get("text", "")
                if transcript:
                    print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (using pre-generated transcript)")
                else:
                    print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (no transcript available)")
            else:
                print(f"\n[{i+1}/{len(data)}] Processing: {audio_id} (no speech_recognition in tool_outputs)")

            if audio_path in processed_audios:
                print(f"  Using cached results for {audio_id}")
                stress_result = processed_audios[audio_path]
            else:
                stress_result = await process_single_audio_stress(tool, audio_path, transcript=transcript)
                processed_audios[audio_path] = stress_result

            extended_entry = entry.copy()
            extended_entry["tool_outputs"] = entry.get("tool_outputs", {}).copy()
            extended_entry["tool_outputs"]["stressed_analysis"] = stress_result
            fout.write(json.dumps(extended_entry, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"\nDone! Processed {len(data)} entries.")
    print(f"Unique audio files processed: {len(processed_audios)}")
    print(f"Output saved to: {output_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
