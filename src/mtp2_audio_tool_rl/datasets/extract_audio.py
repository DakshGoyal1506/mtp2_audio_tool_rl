import json
import os
import subprocess

jsonl_file = "/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/output/grpo_tool_dataset.jsonl"
source_base_dir = "/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/audioset/mnt/fast/nobackup/scratch4weeks/xm00178/WavCaps/data/waveforms/AudioSet_SL_flac"
target_dir = "/home/speech-nlp-cse/24m0756/abhishek/grpo_dataset/final_audio"

os.makedirs(target_dir, exist_ok=True)

with open(jsonl_file, "r") as f:
    for line in f:
        data = json.loads(line.strip())
        id_str = data["id"]

        # Extract filename (e.g., "Y-0OkG0fLLkc.flac" from "WavCaps_AudioSetSL/Y-0OkG0fLLkc.flac")
        basename = os.path.basename(id_str)
        filename_no_ext = os.path.splitext(basename)[0]

        source_path = os.path.join(source_base_dir, basename)
        target_path = os.path.join(target_dir, f"{filename_no_ext}.wav")

        if not os.path.exists(source_path):
            print(f"Warning: source file not found: {source_path}")
            continue

        print(f"Processing {basename} -> {target_path}")

        # Use ffmpeg to resample to 16kHz and convert to wav
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output if it exists
            "-i", source_path,
            "-ar", "16000",
            target_path
        ]

        # Suppress ffmpeg output to keep logs clean
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Done extracting and converting audios.")
