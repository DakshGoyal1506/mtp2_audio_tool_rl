"""
test_embed_pipeline.py — End-to-end test of the new offline-embed pipeline.

Pipeline simulated:
  1. Load audio → run Whisper encoder + Qformer → save {"qformer": T, "vad": bool}
  2. If VAD=True → run Whisper decoder to get speech transcript (SR)
  3. At "inference" time: load embed, embed SR transcript with LLM tokenizer, concat
  4. Feed combined embedding to LLM → generate answer for the given question

This mirrors exactly what happens during GRPO training:
  - precomputed_embed supplies the qformer tensor + vad flag
  - speech_recognition tool output supplies the transcript text (injected at training time)

Usage:
    python scripts/test_embed_pipeline.py \
        --model_path DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B \
        --audio test-mini-audios/some_audio.flac \
        --question "What is the primary mood conveyed by the audio?" \
        --choices "Peaceful" "Energetic" "Sad" "Tense" \
        [--embed_out /tmp/test_embed.pt]   # optional: save/re-use embed
"""

import argparse
import os
import sys
import json
import tempfile

import torch

# ── path setup ───────────────────────────────────────────────────────────────
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_scripts_dir)
_desta_dir   = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")
for _p in [_project_dir, _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from desta.utils.audio import AudioSegment  # type: ignore


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Test the offline-embed + SR pipeline end-to-end")
    p.add_argument("--model_path", required=True,
                   help="HF repo or local path to DeSTA2.5-Audio model")
    p.add_argument("--audio",      required=True,
                   help="Path to audio file (.wav / .flac / .mp3)")
    p.add_argument("--question",   required=True,
                   help="Natural-language question about the audio")
    p.add_argument("--choices",    nargs="+", default=[],
                   help="Answer choices (optional, for MCQ-style questions)")
    p.add_argument("--embed_out",  default=None,
                   help="Path to save/load the precomputed embed dict (.pt). "
                        "If the file already exists it is loaded instead of recomputed.")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="0 = greedy decoding")
    return p.parse_args()


# ── Step 1 & 2: extract Qformer + VAD + transcript ───────────────────────────

def extract_embed(model, audio_path: str, device: str) -> dict:
    """
    Run VAD + Qformer on audio.  If speech detected, also run Whisper decoder
    to get the transcript.

    Returns:
        {
          "qformer":      Tensor(prompt_size, hidden_dim),  cpu fp16
          "vad":          bool,
          "transcript":   str | None   (None if vad=False)
        }
    """
    from transformers import AutoProcessor

    print(f"  Loading audio: {audio_path}")
    samples = AudioSegment.from_file(
        audio_path, target_sr=16000, channel_selector="average"
    ).samples
    samples_t = torch.from_numpy(samples).float()

    # ── VAD ──────────────────────────────────────────────────────────────────
    print("  Running VAD …")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
        force_reload=False, trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    vad_model = vad_model.to(device)
    ts  = get_speech_timestamps(samples_t, vad_model, sampling_rate=16000)
    vad = len(ts) > 0
    print(f"  VAD → speech_detected={vad}")

    # ── Processor ────────────────────────────────────────────────────────────
    if not hasattr(model, "processor") or model.processor is None:
        model.processor = AutoProcessor.from_pretrained(
            model.config.encoder_model_id, cache_dir=os.getenv("HF_HOME")
        )

    batch_features = model.processor(
        [samples], sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype=torch.float16)

    # ── Qformer forward (transcription not used in Qformer itself) ───────────
    print("  Running Qformer …")
    dummy_trans = torch.zeros(
        (1, model.config.llm_config.hidden_size), device=device, dtype=torch.float16
    )
    with torch.inference_mode():
        batch_audio_features, _ = model.perception(
            input_features=batch_features,
            transcription_embeddings_list=[dummy_trans],
        )
    qformer_out = batch_audio_features[0].cpu().detach()   # (P, H)

    # ── Speech recognition (only if VAD=True) ───────────────────────────────
    transcript = None
    if vad:
        print("  Running Whisper ASR …")
        with torch.inference_mode():
            asr_ids = model.perception.whisper.generate(
                input_features=batch_features,
                attention_mask=None,
                max_new_tokens=128,
            )
        transcript = model.processor.batch_decode(asr_ids, skip_special_tokens=True)[0].strip()
        print(f"  Transcript: {transcript!r}")

    return {"qformer": qformer_out, "vad": vad, "transcript": transcript}


# ── Step 3 & 4: generate answer using the precomputed embed ─────────────────

def run_inference(model, embed: dict, audio_path: str,
                  question: str, choices: list,
                  device: str, max_new_tokens: int, temperature: float) -> str:
    """
    Build messages, load qformer + (optional) SR transcript, run generation.
    Mirrors the training-time path in modeling_grpo.py.
    """
    import tempfile

    # Save embed dict to a tempfile so the model's generate() path can load it
    with tempfile.NamedTemporaryFile(suffix="_embed.pt", delete=False) as tf:
        embed_tmp_path = tf.name
    torch.save({"qformer": embed["qformer"], "vad": embed["vad"]}, embed_tmp_path)

    # Build MCQ or open-ended prompt
    if choices:
        choices_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
        user_content = f"<|AUDIO|>\n{question}\n\nChoices:\n{choices_str}"
    else:
        user_content = f"<|AUDIO|>\n{question}"

    # transcript comes from the "speech_recognition tool" — injected at inference time
    sr_text = embed.get("transcript") if embed["vad"] else None

    messages = [
        {"role": "system", "content": "You are an expert audio analysis assistant."},
        {
            "role": "user",
            "content": user_content,
            "audios": [{"audio": audio_path, "text": sr_text, "embed_path": embed_tmp_path}],
        },
    ]

    print("  Running LLM generation …")
    do_sample = temperature > 0.0
    with torch.inference_mode():
        out = model.generate(
            messages=messages,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )

    os.unlink(embed_tmp_path)
    return out.text[0] if isinstance(out.text, list) else out.text


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_path}")
    print(f"{'='*60}")

    # Use the GRPO model class so we aren't burdened by silero-vad at model load
    # (VAD is loaded separately below for the extraction step)
    try:
        from grpo.modeling_grpo import GRPODeSTA25AudioModel
        # Don't skip perception — we need Whisper encoder + Qformer + decoder for extraction
        GRPODeSTA25AudioModel.skip_perception = False
        model_cls = GRPODeSTA25AudioModel
    except ImportError:
        from desta.models.modeling_desta25 import DeSTA25AudioModel
        model_cls = DeSTA25AudioModel

    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    if not hasattr(model, "tokenizer"):
        model._setup_generation()

    print(f"Model loaded on {args.device}.")

    # ── Step 1 & 2: extract or load embed ───────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 1 & 2: Qformer extraction + VAD + ASR")
    print(f"{'='*60}")

    embed_path = args.embed_out or os.path.join(
        tempfile.gettempdir(),
        os.path.splitext(os.path.basename(args.audio))[0] + "_embed.pt",
    )

    if os.path.exists(embed_path):
        print(f"  Loading existing embed from: {embed_path}")
        raw = torch.load(embed_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "transcript" in raw:
            embed = raw
        else:
            # old embed or no transcript key — re-extract
            print("  Embed missing 'transcript' key — re-extracting …")
            embed = extract_embed(model, args.audio, args.device)
    else:
        embed = extract_embed(model, args.audio, args.device)

    if args.embed_out:
        torch.save(embed, args.embed_out)
        print(f"  Embed saved to: {args.embed_out}")

    # ── Step 3 & 4: inference ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 3 & 4: Inference with precomputed embed + SR transcript")
    print(f"{'='*60}")
    print(f"  Question : {args.question}")
    if args.choices:
        print(f"  Choices  : {args.choices}")
    print(f"  VAD      : {embed['vad']}")
    print(f"  Transcript: {embed.get('transcript')!r}")

    answer = run_inference(
        model       = model,
        embed       = embed,
        audio_path  = args.audio,
        question    = args.question,
        choices     = args.choices,
        device      = args.device,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
    )

    print(f"\n{'='*60}")
    print("Answer:")
    print(f"{'='*60}")
    print(answer)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
