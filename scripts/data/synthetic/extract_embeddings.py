"""
extract_embeddings.py — Precompute Qformer embeddings + VAD for the synthetic
GRPO tool dataset.

Reads dataset/grpo_tool_dataset.with_relative_audio.jsonl, loads audio from
dataset/audio/<subdir>/<file>, extracts Qformer embeddings via DeSTA2.5
perception encoder, and saves each as {basename}_embed.pt in
synthetic_dataset/precomputed_embeds/.

Handles both .wav and .mp3 files via torchaudio.

Saved format per audio:
  {
    "qformer": Tensor(prompt_size, llm_hidden_dim),  # fp16
    "vad":     bool,                                 # True if speech detected
  }
"""

import os
import sys
import json
import argparse
import wave
import glob

import numpy as np
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_desta_dir = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")
for _p in [_project_dir, _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desta.models.modeling_desta25 import DeSTA25AudioModel  # type: ignore

TARGET_SR = 16000


def load_audio_as_numpy(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load a wav/mp3/flac file as float32 mono numpy array at target_sr."""
    data, sr = sf.read(path, dtype="float32")
    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample if needed
    if sr != target_sr:
        waveform = torch.from_numpy(data).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        data = waveform.squeeze(0).numpy()
    return data


# ── VAD helper ───────────────────────────────────────────────────────────────

def _load_vad():
    """Load silero-vad from local hub cache."""
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    except Exception:
        local_vad = os.path.join(
            os.path.expanduser("~"), ".cache", "torch", "hub",
            "snakers4_silero-vad_master",
        )
        vad_model, utils = torch.hub.load(
            repo_or_dir=local_vad, model="silero_vad", source="local"
        )
    get_speech_timestamps = utils[0]
    return vad_model, get_speech_timestamps


def _run_vad(samples: torch.Tensor, vad_model, get_speech_timestamps) -> bool:
    """Return True if any speech timestamps are detected."""
    try:
        vad_device = next(vad_model.parameters()).device
    except StopIteration:
        vad_device = torch.device("cpu")
    ts = get_speech_timestamps(samples.to(vad_device), vad_model, sampling_rate=16000)
    return len(ts) > 0


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Qformer embeddings + VAD for synthetic GRPO tool dataset."
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path or HF repo of DeSTA2.5-Audio model")
    parser.add_argument("--audio_dir", type=str,
                        default=os.path.join(_project_dir, "dataset", "audio"),
                        help="Root audio directory (default: dataset/audio)")
    parser.add_argument("--data_jsonl", type=str,
                        default=os.path.join(_project_dir, "dataset",
                                             "grpo_tool_dataset.with_relative_audio.jsonl"),
                        help="Path to the JSONL dataset")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_this_dir, "precomputed_embeds"),
                        help="Output dir for .pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip already-extracted files (default: on)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Re-extract even if output exists")
    parser.add_argument("--check_only", action="store_true", default=False,
                        help="Only run audio check, skip extraction")
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main():
    args = parse_args()

    # ── Step 1: Collect audio paths from JSONL ──
    print("=" * 60)
    print("Step 1: Loading dataset from JSONL ...")
    print("=" * 60)

    entries = []
    with open(args.data_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries from JSONL")

    # Build mapping: relative_audio_path -> full disk path
    # relative_audio_path is relative to dataset/ dir (e.g. audio/Anispeech/xxx.wav)
    dataset_dir = os.path.dirname(args.data_jsonl)
    path_to_full = {}
    missing = []
    seen_paths = set()

    for entry in entries:
        rel = entry["relative_audio_path"]
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        full = os.path.join(dataset_dir, rel)
        if os.path.exists(full):
            path_to_full[rel] = full
        else:
            missing.append(rel)

    print(f"Matched {len(path_to_full)} unique audio files, {len(missing)} missing")
    if missing:
        print(f"  Missing examples: {missing[:5]}")
        sys.exit(1)

    if args.check_only:
        print("--check_only mode: all audio files found. Exiting.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 2: Load model ──
    print("\n" + "=" * 60)
    print("Step 2: Loading DeSTA2.5 model ...")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    model = DeSTA25AudioModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    # Delete Whisper decoder — not needed, saves VRAM
    if hasattr(model.perception.whisper.model, "decoder"):
        del model.perception.whisper.model.decoder
    if hasattr(model.perception.whisper, "proj_out"):
        del model.perception.whisper.proj_out

    if not hasattr(model, "processor"):
        from transformers import AutoProcessor
        model.processor = AutoProcessor.from_pretrained(
            model.config.encoder_model_id,
            cache_dir=os.getenv("HF_HOME"),
        )
    print("Model loaded.")

    # ── Step 3: Load VAD ──
    print("Loading silero-VAD ...")
    vad_model, get_speech_timestamps = _load_vad()
    vad_model = vad_model.to(args.device)
    print("VAD loaded.")

    # ── Step 4: Extract embeddings ──
    print("\n" + "=" * 60)
    print("Step 4: Extracting embeddings ...")
    print("=" * 60)

    skipped = extracted = failed = 0

    for rel_path, full_path in tqdm(path_to_full.items(), desc="Extracting"):
        basename = os.path.splitext(os.path.basename(rel_path))[0]
        out_name = f"{basename}_embed.pt"
        out_path = os.path.join(args.out_dir, out_name)

        if not args.overwrite and args.skip_existing and os.path.exists(out_path):
            try:
                existing = torch.load(out_path, map_location="cpu", weights_only=False)
                if isinstance(existing, dict) and "qformer" in existing and "vad" in existing:
                    skipped += 1
                    continue
                print(f"  [INFO] Re-extracting {out_name} (old format)")
            except Exception:
                pass

        try:
            # 1. Load raw audio (float32, 16 kHz, mono)
            samples = load_audio_as_numpy(full_path, target_sr=TARGET_SR)
            samples_t = torch.from_numpy(samples).float()

            # 2. Run VAD
            vad = _run_vad(samples_t, vad_model, get_speech_timestamps)

            # 3. Processor -> mel features (1, 80, T_mel)
            batch_features = model.processor(
                [samples], sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_features.to(args.device, dtype=torch.float16)

            # 4. Qformer forward
            dummy_trans = torch.zeros(
                (1, model.config.llm_config.hidden_size),
                device=args.device, dtype=torch.float16,
            )
            batch_audio_features, _ = model.perception(
                input_features=batch_features,
                transcription_embeddings_list=[dummy_trans],
            )
            qformer_out = batch_audio_features[0].cpu().detach()  # (prompt_size, llm_hidden)

            # 5. Save
            torch.save({"qformer": qformer_out, "vad": vad}, out_path)
            extracted += 1

        except Exception as exc:
            print(f"  [ERROR] {rel_path}: {exc}")
            failed += 1

    print(
        f"\nDone — extracted: {extracted}, skipped: {skipped}, failed: {failed}"
        f"\nEmbeds saved to: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
