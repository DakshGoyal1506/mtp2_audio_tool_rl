"""
check_audio.py — Verify synthetic dataset audio files: sampling rates, duration
stats, coverage against the JSONL, and precomputed embed coverage.

Run standalone:
    python synthetic_dataset/check_audio.py
"""

import os
import sys
import json
import wave
import glob
import struct
import statistics

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)

DATASET_DIR = os.path.join(_project_dir, "dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
JSONL_PATH = os.path.join(DATASET_DIR, "grpo_tool_dataset.with_relative_audio.jsonl")
EMBED_DIR = os.path.join(_this_dir, "precomputed_embeds")


def check_audio():
    # ── 1. Discover all audio files recursively ──
    audio_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.endswith((".wav", ".mp3", ".flac")):
                audio_files.append(os.path.join(root, f))
    audio_files.sort()
    print(f"Found {len(audio_files)} audio files under {AUDIO_DIR}")

    # Extension breakdown
    ext_counts = {}
    for p in audio_files:
        ext = os.path.splitext(p)[1]
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("\n--- Extension Distribution ---")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count} files")

    # Duration / SR stats for wav files only (wave module can't read mp3)
    wav_files = [p for p in audio_files if p.endswith(".wav")]
    sr_counts = {}
    ch_counts = {}
    durations = []
    bad_files = []

    for path in wav_files:
        try:
            with wave.open(path, "r") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                dur = wf.getnframes() / sr
                sr_counts[sr] = sr_counts.get(sr, 0) + 1
                ch_counts[ch] = ch_counts.get(ch, 0) + 1
                durations.append((os.path.basename(path), sr, ch, dur))
        except Exception:
            # Fallback: parse header manually (handles IEEE float WAVs)
            try:
                with open(path, 'rb') as fh:
                    fh.read(4)  # RIFF
                    fh.read(4)  # size
                    fh.read(4)  # WAVE
                    fh.read(4)  # fmt
                    fmt_size = struct.unpack('<I', fh.read(4))[0]
                    fmt_data = fh.read(fmt_size)
                    ch = struct.unpack('<H', fmt_data[2:4])[0]
                    sr = struct.unpack('<I', fmt_data[4:8])[0]
                    bits = struct.unpack('<H', fmt_data[14:16])[0]
                    # Skip to data chunk
                    while True:
                        chunk_id = fh.read(4)
                        if not chunk_id or len(chunk_id) < 4:
                            raise ValueError("data chunk not found")
                        chunk_size = struct.unpack('<I', fh.read(4))[0]
                        if chunk_id == b'data':
                            n_frames = chunk_size // (ch * (bits // 8))
                            dur = n_frames / sr
                            break
                        fh.seek(chunk_size, 1)
                    sr_counts[sr] = sr_counts.get(sr, 0) + 1
                    ch_counts[ch] = ch_counts.get(ch, 0) + 1
                    durations.append((os.path.basename(path), sr, ch, dur))
            except Exception as e:
                bad_files.append((os.path.basename(path), str(e)))

    if wav_files:
        print(f"\n--- WAV Sampling Rate Distribution ---")
        for sr, count in sorted(sr_counts.items()):
            print(f"  {sr} Hz: {count} files")

        print(f"\n--- WAV Channel Distribution ---")
        for ch, count in sorted(ch_counts.items()):
            print(f"  {ch} channel(s): {count} files")

        durs = [d[3] for d in durations]
        if durs:
            print(f"\n--- WAV Duration Stats ---")
            print(f"  Total wav files: {len(durs)}")
            print(f"  Min duration:    {min(durs):.2f}s")
            print(f"  Max duration:    {max(durs):.2f}s")
            print(f"  Mean duration:   {statistics.mean(durs):.2f}s")
            print(f"  Median:          {statistics.median(durs):.2f}s")
            print(f"  Total audio:     {sum(durs)/60:.1f} min")

    if bad_files:
        print(f"\n--- Bad/Corrupt Files ({len(bad_files)}) ---")
        for name, err in bad_files:
            print(f"  {name}: {err}")

    # ── 2. Check JSONL coverage ──
    print(f"\n--- JSONL Coverage ---")
    jsonl_entries = []
    with open(JSONL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jsonl_entries.append(json.loads(line))

    rel_paths = {e["relative_audio_path"] for e in jsonl_entries}
    print(f"  JSONL entries: {len(jsonl_entries)}")
    print(f"  Unique audio paths: {len(rel_paths)}")

    # Check each JSONL path exists on disk
    disk_rels = set()
    for p in audio_files:
        disk_rels.add(os.path.relpath(p, DATASET_DIR))

    missing_on_disk = sorted(rel_paths - disk_rels)
    extra_on_disk = sorted(disk_rels - rel_paths)

    print(f"  Matched to disk: {len(rel_paths) - len(missing_on_disk)}")
    if missing_on_disk:
        print(f"  Missing on disk for {len(missing_on_disk)} paths:")
        for m in missing_on_disk[:10]:
            print(f"    {m}")
    if extra_on_disk:
        print(f"  Extra on disk (not in JSONL): {len(extra_on_disk)}")
        for e in extra_on_disk[:10]:
            print(f"    {e}")

    # ── 3. Check precomputed embeds ──
    if os.path.isdir(EMBED_DIR):
        embed_files = glob.glob(os.path.join(EMBED_DIR, "*_embed.pt"))
        print(f"\n--- Precomputed Embeds ---")
        print(f"  Found {len(embed_files)} embed files in {EMBED_DIR}")

        # Build expected embed names from JSONL relative_audio_path basenames
        expected_basenames = set()
        for e in jsonl_entries:
            bn = os.path.splitext(os.path.basename(e["relative_audio_path"]))[0]
            expected_basenames.add(bn)

        embed_basenames = {os.path.basename(f).replace("_embed.pt", "") for f in embed_files}
        covered = expected_basenames & embed_basenames
        missing_embeds = expected_basenames - embed_basenames
        print(f"  Covered: {len(covered)}/{len(expected_basenames)}")
        if missing_embeds:
            print(f"  Missing embeds for {len(missing_embeds)} audios:")
            for m in sorted(missing_embeds)[:10]:
                print(f"    {m}")
    else:
        print(f"\n  No precomputed_embeds/ dir yet.")

    # ── Summary ──
    non_16k = sum(c for sr, c in sr_counts.items() if sr != 16000)
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Audio files: {len(audio_files)}, JSONL paths: {len(rel_paths)}")
    print(f"  Missing on disk: {len(missing_on_disk)}, Extra on disk: {len(extra_on_disk)}")
    print(f"  Non-16kHz wav files: {non_16k}")
    print(f"  Bad files: {len(bad_files)}")
    if non_16k == 0 and len(bad_files) == 0 and len(missing_on_disk) == 0:
        print(f"  STATUS: ALL CHECKS PASSED")
    else:
        print(f"  STATUS: ISSUES FOUND — see above")


if __name__ == "__main__":
    check_audio()
