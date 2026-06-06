import json
import os
import subprocess
import sys
from pathlib import Path


def require_env(name):
    value = os.environ.get(name)
    if not value:
        print(f"Error: set {name} before running this script.", file=sys.stderr)
        sys.exit(1)
    return value


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_ROOT = Path(os.environ.get("MTP2_MANIFEST_ROOT", REPO_ROOT / "manifests"))
AUDIO_ROOT = Path(require_env("MTP2_AUDIO_ROOT"))
OUTPUT_ROOT = Path(require_env("MTP2_OUTPUT_ROOT"))

jsonl_file = MANIFEST_ROOT / "grpo_tool_dataset.jsonl"
source_base_dir = AUDIO_ROOT
target_dir = OUTPUT_ROOT / "final_audio"

target_dir.mkdir(parents=True, exist_ok=True)

with open(jsonl_file, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        id_str = data["id"]

        # Extract filename (e.g., "Y-0OkG0fLLkc.flac" from "WavCaps_AudioSetSL/Y-0OkG0fLLkc.flac")
        basename = os.path.basename(id_str)
        filename_no_ext = os.path.splitext(basename)[0]

        source_path = source_base_dir / basename
        target_path = target_dir / f"{filename_no_ext}.wav"

        if not os.path.exists(source_path):
            print(f"Warning: source file not found: {source_path}")
            continue

        print(f"Processing {basename} -> {target_path}")

        # Use ffmpeg to resample to 16kHz and convert to wav
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output if it exists
            "-i", str(source_path),
            "-ar", "16000",
            str(target_path)
        ]

        # Suppress ffmpeg output to keep logs clean
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Done extracting and converting audios.")
