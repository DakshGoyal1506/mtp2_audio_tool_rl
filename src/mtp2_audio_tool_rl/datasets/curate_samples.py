import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path

from prompts import ALL_TOOLS

# Path to locally cached dataset files
DESTA_CACHE_DIR = Path.home() / ".cache/huggingface/hub/datasets--DeSTA-ntu--DeSTA-AQA5M-FROM-Llama3.1-8B-Instruct/snapshots"

DEFAULT_OUTPUT_FILE = "curated_tool_candidates.jsonl"
DEFAULT_TARGET_PER_TOOL = 1000

# Datasets to ignore based on the MMAU benchmark overlap.
# Matching is done on normalized names so aliases such as "voxceleb1" and
# "VoxCeleb-1" resolve to the same key.
IGNORE_DATASETS = {
    "Audioset",
    "AudioSet",
    "AudioSet-20K",
    "WavCaps_AudioSetSL",
    "AudioSet Strong",
    "Mustard",
    "MELD",
    "VoxCeleb-1",
    "voxceleb1",
    "IEMOCAP",
    "MusicBench",
    "Jamendo",
    "SDD",
    "MusicCaps",
    "GuitarSet",
    "MUSDB18",
    "Synthetic",
}

# Prefer explicit dataset-to-tool mappings. These are much more reliable than
# prompt text because DeSTA prompts can be intentionally mismatched with the
# actual audio content.
DATASET_TO_TOOL = {
    "AliMeeting": ["speaker_diarization"],
    "Libricount": ["speaker_diarization", "sound_duration_analysis"],
    "VCTK_augmented2": ["speaker_diarization", "speech_to_noise_ratio"],
    "GtzanGenre": ["genre_analysis"],
    "FMA_medium": ["genre_analysis"],
    "OpenSinger": ["genre_analysis", "get_audio_features"],
    "Mridangam": ["chord_recognition", "get_audio_features"],
    "Nsynth": ["get_audio_features", "chord_recognition"],
    "TMHINT-QI": ["speech_to_noise_ratio"],
    "LibriTTS_R": ["speech_to_noise_ratio", "get_audio_features"],
    "dynamic-superb-noise-reverb": ["speech_to_noise_ratio"],
    "Anispeech": ["emotion_recognition", "stressed_analysis"],
    "CAFE": ["emotion_recognition"],
    "CREMA-D": ["emotion_recognition"],
    "Dusha": ["emotion_recognition"],
    "EMNS": ["emotion_recognition"],
    "EMOVO": ["emotion_recognition"],
    "EmoV_DB": ["emotion_recognition"],
    "Emotion_Speech_Dataset": ["emotion_recognition"],
    "MSP-PODCAST-Publish-1.12": ["emotion_recognition"],
    "MSP_IMPROV": ["emotion_recognition"],
    "expresso": ["emotion_recognition", "stressed_analysis"],
    "release_BIIC-Podcast-v1-01": ["emotion_recognition"],
    "common_voice_en": ["speech_recognition"],
    "common_voice_zh": ["speech_recognition"],
    "cszs_es_en": ["speech_recognition"],
    "cszs_fr_en": ["speech_recognition"],
    "cszs_zh_en": ["speech_recognition"],
    "meta_fair_asr": ["speech_recognition"],
    "Speech_Command": ["speech_recognition"],
    "KeSpeech": ["speech_recognition"],
    "Voxlingual_Top10": ["speech_recognition"],
    "ESC50": ["sound_classification"],
    "FSD50K": ["sound_classification"],
    "Clotho": ["sound_classification"],
    "AudioCaps": ["sound_classification"],
    "Audiocaps2": ["sound_classification"],
    "VocalSound": ["sound_classification", "sound_duration_analysis"],
    "ASVspoof2015": ["stressed_analysis", "speech_to_noise_ratio"],
    "ASVspoof5": ["stressed_analysis", "speech_to_noise_ratio"],
    "ASVspoofing2019": ["stressed_analysis", "speech_to_noise_ratio"],
}

# Optional fallback heuristics on metadata only. These are disabled by default
# because the DeSTA prompt field is not a trustworthy signal for tool choice.
TOOL_HEURISTICS = {
    "emotion_recognition": ["emotion:", "sentiment:", "valence:", "arousal:", "amused", "angry", "happy", "sad"],
    "genre_analysis": ["genre:", "song:", "reggae", "rock", "jazz", "bass"],
    "speaker_diarization": ["speaker:", "number of speakers", "speakers talking"],
    "speech_recognition": ["accent:", "dialect:", "language:", "code-switching", "command:"],
    "sound_classification": ["audio category:", "type:", "sound of", "animal", "water", "knock"],
    "get_audio_features": ["pitch:", "volume:", "tempo", "timbre", "midi velocity", "midi note"],
    "sound_duration_analysis": ["duration:", "[00:00-", "[0.00-"],
    "speech_to_noise_ratio": ["snr", "noise level", "reverberation", "speech_quality", "fidelity", "quality"],
    "stressed_analysis": ["intonation", "stress", "prosod", "speaking speed", "activation:"],
    "chord_recognition": ["tonic:", "harmony", "chord", "musical structure"],
}

def normalize_name(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


NORMALIZED_IGNORE_DATASETS = {normalize_name(name) for name in IGNORE_DATASETS}
NORMALIZED_DATASET_TO_TOOL = {
    normalize_name(dataset_name): tools for dataset_name, tools in DATASET_TO_TOOL.items()
}


def iter_jsonl_files(snapshot_dir):
    """Return shard files in round-robin order: audio.0, speech.0, audio.1, speech.1, ..."""
    grouped = defaultdict(list)

    for jsonl_file in sorted(snapshot_dir.glob("*.jsonl")):
        shard_prefix = jsonl_file.stem.split(".", 1)[0]
        grouped[shard_prefix].append(jsonl_file)

    ordered_files = []
    ordered_groups = [grouped[name] for name in sorted(grouped)]

    for group in zip_longest(*ordered_groups):
        for file_path in group:
            if file_path is not None:
                ordered_files.append(file_path)

    return ordered_files


def extract_audio_text(item):
    messages = item.get("messages") or []
    for message in messages:
        for audio_info in message.get("audios") or []:
            text = (audio_info.get("text") or "").strip()
            if text:
                return text
    return ""


def build_metadata_text(item):
    return " ".join(
        part
        for part in [
            item.get("dataset", ""),
            item.get("seed_description", ""),
            extract_audio_text(item),
        ]
        if part
    )


def iter_string_fields(value, path="root"):
    if isinstance(value, str):
        yield path, value
        return

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_path = "{0}.{1}".format(path, key)
            for item in iter_string_fields(child_value, child_path):
                yield item
        return

    if isinstance(value, list):
        for index, child_value in enumerate(value):
            child_path = "{0}[{1}]".format(path, index)
            for item in iter_string_fields(child_value, child_path):
                yield item


def is_english_text(text):
    for char in text:
        if not char.isascii():
            # Allow common typographical non-ASCII characters like punctuation (smart quotes, em-dashes, etc.)
            if unicodedata.category(char).startswith('P'):
                continue
            return False
    return True

def is_english_only_row(item):
    for key in ("prompt", "seed_description"):
        text = item.get(key)
        if text and not is_english_text(text):
            return False, f"non_english_in_{key}", key
    return True, "", ""


def get_possible_tools(item, allow_heuristics=False):
    dataset_name = item.get("dataset", "")
    dataset_key = normalize_name(dataset_name)

    direct_tools = NORMALIZED_DATASET_TO_TOOL.get(dataset_key, [])
    if direct_tools:
        return list(direct_tools), "dataset_mapping"

    if not allow_heuristics:
        return [], "none"

    metadata_text = build_metadata_text(item).lower()
    heuristic_tools = []
    for tool, keywords in TOOL_HEURISTICS.items():
        if any(keyword in metadata_text for keyword in keywords):
            heuristic_tools.append(tool)

    return list(dict.fromkeys(heuristic_tools)), "metadata_heuristic"


def build_dataset_cap_map(target_per_tool, explicit_tool_to_datasets):
    cap_map = {}
    for tool in ALL_TOOLS:
        supporting_datasets = explicit_tool_to_datasets.get(tool, set())
        if supporting_datasets:
            cap_map[tool] = math.ceil(target_per_tool / len(supporting_datasets))
        else:
            cap_map[tool] = target_per_tool
    return cap_map


def choose_tool(
    candidate_tools,
    dataset_name,
    tool_counts,
    dataset_tool_counts,
    target_per_tool,
    dataset_caps,
):
    eligible_tools = []
    for tool in candidate_tools:
        if tool_counts[tool] >= target_per_tool:
            continue
        if dataset_tool_counts[tool][dataset_name] >= dataset_caps[tool]:
            continue
        eligible_tools.append(tool)

    if not eligible_tools:
        return None

    # Prefer the most under-filled tool first, then the one that has used this
    # dataset the least, which keeps the curated set diverse.
    eligible_tools.sort(
        key=lambda tool: (
            tool_counts[tool] / target_per_tool,
            dataset_tool_counts[tool][dataset_name],
            tool_counts[tool],
            tool,
        )
    )
    return eligible_tools[0]


def all_targets_met(tool_counts, target_per_tool):
    return all(tool_counts[tool] >= target_per_tool for tool in ALL_TOOLS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate balanced DeSTA candidate samples for each MMAU-style tool."
    )
    parser.add_argument(
        "--target-per-tool",
        type=int,
        default=DEFAULT_TARGET_PER_TOOL,
        help=f"Number of candidate samples to collect per tool (default: {DEFAULT_TARGET_PER_TOOL})",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help=f"Where to write the curated JSONL output (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--allow-heuristics",
        action="store_true",
        help="Also use metadata-only keyword heuristics for datasets that lack an explicit tool mapping.",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Keep only rows whose string fields look English using heuristic filtering.",
    )
    args = parser.parse_args()

    snapshot_dirs = sorted(DESTA_CACHE_DIR.iterdir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No cached dataset found at {DESTA_CACHE_DIR}.")

    snapshot_dir = snapshot_dirs[0]
    jsonl_files = iter_jsonl_files(snapshot_dir)

    explicit_tool_to_datasets = defaultdict(set)
    for dataset_name, tools in DATASET_TO_TOOL.items():
        dataset_key = normalize_name(dataset_name)
        if dataset_key in NORMALIZED_IGNORE_DATASETS:
            continue
        for tool in tools:
            explicit_tool_to_datasets[tool].add(dataset_name)

    dataset_caps = build_dataset_cap_map(args.target_per_tool, explicit_tool_to_datasets)
    tool_counts: Counter = Counter({tool: 0 for tool in ALL_TOOLS})
    dataset_tool_counts = {tool: Counter() for tool in ALL_TOOLS}
    curated_samples = []

    scanned_rows = 0
    ignored_overlap_rows = 0
    skipped_unmapped_rows = 0
    skipped_non_english_rows = 0
    skipped_non_english_prompt = 0
    skipped_non_english_seed = 0
    processed_files = []

    print("Curating DeSTA candidates with explicit dataset mappings first...")
    if args.english_only:
        print("English filter mode: heuristic (unicodedata/ASCII check)")

    for jsonl_file in jsonl_files:
        if all_targets_met(tool_counts, args.target_per_tool):
            break

        processed_files.append(jsonl_file.name)

        with open(jsonl_file, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                if all_targets_met(tool_counts, args.target_per_tool):
                    break

                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                scanned_rows += 1

                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                dataset_name = item.get("dataset", "")
                if normalize_name(dataset_name) in NORMALIZED_IGNORE_DATASETS:
                    ignored_overlap_rows += 1
                    continue

                if args.english_only:
                    is_english_only, reason, _ = is_english_only_row(item)
                    if not is_english_only:
                        skipped_non_english_rows += 1
                        if "prompt" in reason:
                            skipped_non_english_prompt += 1
                        elif "seed_description" in reason:
                            skipped_non_english_seed += 1
                        continue

                possible_tools, inference_source = get_possible_tools(
                    item, allow_heuristics=args.allow_heuristics
                )
                if not possible_tools:
                    skipped_unmapped_rows += 1
                    continue

                chosen_tool = choose_tool(
                    possible_tools,
                    dataset_name,
                    tool_counts,
                    dataset_tool_counts,
                    args.target_per_tool,
                    dataset_caps,
                )
                if chosen_tool is None:
                    continue

                curated_item = dict(item)
                curated_item["candidate_tool"] = chosen_tool
                curated_item["candidate_tool_source"] = inference_source
                curated_samples.append(curated_item)

                tool_counts[chosen_tool] += 1
                dataset_tool_counts[chosen_tool][dataset_name] += 1

    output_path = Path(args.output_file)
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in curated_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nCurated samples saved to {output_path}")
    print(f"Rows scanned: {scanned_rows}")
    print(f"Overlap rows skipped: {ignored_overlap_rows}")
    if args.english_only:
        print(f"Non-English rows skipped: {skipped_non_english_rows} (Prompt: {skipped_non_english_prompt}, Seed: {skipped_non_english_seed})")
    print(f"Rows without a usable mapping: {skipped_unmapped_rows}")
    print(f"Shard files touched: {', '.join(processed_files)}")

    print("\nTool fulfillment breakdown:")
    for tool in ALL_TOOLS:
        count = tool_counts[tool]
        status = "✓" if count >= args.target_per_tool else "⚠"
        print(f"[{status}] {tool}: {count}/{args.target_per_tool}")

    print("\nDataset spread per tool:")
    for tool in ALL_TOOLS:
        spread = dataset_tool_counts[tool].most_common()
        if not spread:
            print(f"- {tool}: none")
            continue
        summary = ", ".join(f"{dataset}={count}" for dataset, count in spread)
        print(f"- {tool}: {summary}")


if __name__ == "__main__":
    main()
