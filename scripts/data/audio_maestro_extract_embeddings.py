"""
extract_embeddings.py — Precompute Qformer embeddings + VAD decisions for GRPO training.

Saved format (per audio file):
  {
    "qformer": Tensor(prompt_size, llm_hidden_dim),  # Qformer output, fp16
    "vad":     bool,                                 # True if speech detected
  }

At training/inference time:
  - If vad=True  → embed the speech_recognition transcript and concat with qformer output.
  - If vad=False → use only the qformer tensor (no transcription appended).

This separation means transcription is injected at runtime from the cached
speech_recognition tool output, keeping the embedding independent of the ASR text.
"""

import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_scripts_dir)
_desta_dir   = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")
for _p in [_project_dir, _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desta.models.modeling_desta25 import DeSTA25AudioModel  # type: ignore
from desta.utils.audio import AudioSegment                   # type: ignore


# ── VAD helper ───────────────────────────────────────────────────────────────

def _load_vad():
    """Load silero-vad from local hub cache (no network required after first use)."""
    try:
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    except Exception:
        # Fallback: local clone if hub cache is unavailable
        local_vad = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                                 "snakers4_silero-vad_master")
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
        description="Extract Qformer embeddings + VAD decisions (no transcription baked in)."
    )
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Path or HF repo of DeSTA2.5-Audio model")
    parser.add_argument("--data_json",   type=str, required=True,
                        help="Path to dataset JSON (e.g., mmau-test-mini-cached.json)")
    parser.add_argument("--out_dir",     type=str, required=True,
                        help="Directory to save precomputed .pt embed dicts")
    parser.add_argument("--audio_root",  type=str, default="",
                        help="Prefix prepended to relative audio paths in the JSON")
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip already-extracted files (default: on)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Re-extract even if output already exists (overrides --skip_existing)")
    return parser.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model (Whisper encoder + Qformer only; decoder not needed) ──
    print(f"Loading DeSTA2.5-Audio model from {args.model_path} …")
    model = DeSTA25AudioModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    # Delete Whisper decoder — not needed and saves VRAM
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

    # ── load VAD ──
    print("Loading silero-VAD …")
    vad_model, get_speech_timestamps = _load_vad()
    vad_model = vad_model.to(args.device)
    print("VAD loaded.")

    # ── collect unique audio IDs ──
    print(f"Loading dataset from {args.data_json} …")
    with open(args.data_json) as f:
        data = json.load(f)

    audio_ids = list({item["audio_id"] for item in data if item.get("audio_id")})
    print(f"Found {len(audio_ids):,} unique audio files.")

    skipped = extracted = failed = 0

    for audio_id in tqdm(audio_ids, desc="Extracting"):
        out_name = os.path.splitext(os.path.basename(audio_id))[0] + "_embed.pt"
        out_path = os.path.join(args.out_dir, out_name)

        if not args.overwrite and args.skip_existing and os.path.exists(out_path):
            # Smart-skip: if existing file is old flat-tensor format, re-extract
            try:
                existing = torch.load(out_path, map_location="cpu", weights_only=False)
                if isinstance(existing, dict) and "qformer" in existing and "vad" in existing:
                    skipped += 1
                    continue
                # Old format detected — fall through to re-extract
                print(f"  [INFO] Re-extracting {out_name} (old format detected)")
            except Exception:
                pass  # Corrupt file — re-extract

        audio_path = (
            os.path.join(args.audio_root, audio_id.lstrip("./"))
            if args.audio_root else audio_id
        )
        if not os.path.exists(audio_path):
            print(f"  [WARN] missing audio: {audio_path}")
            failed += 1
            continue

        try:
            # 1. Load raw audio samples (float32, 16 kHz, mono)
            samples = AudioSegment.from_file(
                audio_path, target_sr=16000, channel_selector="average"
            ).samples                                             # np.ndarray (T,)
            samples_t = torch.from_numpy(samples).float()

            # 2. Run VAD
            vad = _run_vad(samples_t, vad_model, get_speech_timestamps)

            # 3. Processor → mel features  (1, 80, T_mel)
            batch_features = model.processor(
                [samples], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(args.device, dtype=torch.float16)

            # 4. Qformer forward (transcription not needed — Qformer is
            #    purely audio-encoder → cross-attention → query tokens)
            #    We pass a minimal dummy so the interface is satisfied.
            dummy_trans = torch.zeros(
                (1, model.config.llm_config.hidden_size),
                device=args.device, dtype=torch.float16
            )
            batch_audio_features, _ = model.perception(
                input_features=batch_features,
                transcription_embeddings_list=[dummy_trans],
            )
            qformer_out = batch_audio_features[0].cpu().detach()  # (prompt_size, llm_hidden)

            # 5. Save as dict
            torch.save({"qformer": qformer_out, "vad": vad}, out_path)
            extracted += 1

        except Exception as exc:
            print(f"  [ERROR] {audio_id}: {exc}")
            failed += 1

    print(
        f"\nDone — extracted: {extracted}, skipped: {skipped}, failed: {failed}"
        f"\nEmbeds saved to: {args.out_dir}"
    )


if __name__ == "__main__":
    main()

