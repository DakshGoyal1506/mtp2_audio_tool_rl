"""
Embedding utilities for DeSTA2.5-Audio + vLLM integration.

Handles loading precomputed QFormer embeddings and building the full
prompt_embeds tensor that vLLM consumes in place of token IDs.

Precomputed embed format (.pt file):
  {"qformer": Tensor(prompt_size, hidden_dim), "vad": bool}
  - qformer: output of QFormer connector, typically shape (64, 4096)
  - vad: whether speech was detected; if True, transcription tokens are appended

The core idea: we take the Llama-3.1 tokenizer + embed_tokens layer to convert
the text portions of the prompt into embeddings, then splice in the precomputed
QFormer embeddings at the <|AUDIO|> position — producing one contiguous
(seq_len, hidden_dim) tensor that vLLM uses for prefill.
"""

import os
import glob
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ── DeSTA2.5 constants ────────────────────────────────────────────────────────
AUDIO_LOCATOR = "<|AUDIO|>"
PLACEHOLDER_TOKEN = "<|reserved_special_token_87|>"
DEFAULT_PROMPT_SIZE = 64  # QFormer output length
DEFAULT_LLM_MODEL_ID = "DeSTA-ntu/Llama-3.1-8B-Instruct"


def load_embed_cache(
    embed_dir: str,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """
    Build a lazy embed cache that loads files on first access.

    Args:
        embed_dir: Directory containing *_embed.pt files.
        device: Target device for tensors ("cpu" recommended for cache).

    Returns:
        LazyEmbedCache (dict-like) mapping absolute file path → loaded dict/tensor.
    """
    paths = sorted(glob.glob(os.path.join(embed_dir, "*_embed.pt")))
    if not paths:
        logger.warning(f"No *_embed.pt files found in {embed_dir}")
        return {}

    logger.info(f"Embed cache: {len(paths)} files (lazy loading)")
    return LazyEmbedCache(paths, device)


class LazyEmbedCache(dict):
    """Dict-like cache that loads .pt files on first access."""

    def __init__(self, paths: List[str], device: str = "cpu"):
        super().__init__()
        self._device = device
        # Map absolute path → itself (so `path in cache` works)
        self._known = {os.path.abspath(p) for p in paths}

    def __contains__(self, key):
        abs_key = os.path.abspath(key) if isinstance(key, str) else key
        return abs_key in self._known

    def __getitem__(self, key):
        abs_key = os.path.abspath(key) if isinstance(key, str) else key
        if abs_key not in self.keys():
            if abs_key in self._known or os.path.exists(abs_key):
                super().__setitem__(
                    abs_key,
                    torch.load(abs_key, map_location=self._device, weights_only=True),
                )
            else:
                raise KeyError(abs_key)
        return super().__getitem__(abs_key)

    def __len__(self):
        return len(self._known)


def resolve_embed_path(
    audio_id: str,
    embed_dir: str,
) -> Optional[str]:
    """
    Resolve an audio_id to its precomputed embed file path.

    Args:
        audio_id: Audio identifier (e.g. "dataset/test-mini-audios/abc.wav"
                  or just "abc" UUID).
        embed_dir: Directory with *_embed.pt files.

    Returns:
        Absolute path to the embed file, or None if not found.
    """
    # Strip directory and extension to get the base UUID
    base = os.path.splitext(os.path.basename(audio_id))[0]
    candidate = os.path.join(embed_dir, f"{base}_embed.pt")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return None


def _resolve_model_path(model_id: str) -> str:
    """
    Resolve a HuggingFace model ID to a local directory path.

    Checks (in order):
      1. If model_id is already a local directory → use as-is.
      2. If it exists in the HF hub cache → return the snapshot path.
      3. Otherwise return model_id unchanged (will need network).
    """
    if os.path.isdir(model_id):
        return model_id

    # Try HF cache layout: $HF_HOME/hub/models--<org>--<name>/snapshots/<hash>/
    hf_home = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    cache_name = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(hf_home, "hub", cache_name, "snapshots")
    if os.path.isdir(snapshots_dir):
        # Pick the first (usually only) snapshot
        snaps = sorted(os.listdir(snapshots_dir))
        if snaps:
            resolved = os.path.join(snapshots_dir, snaps[0])
            logger.info(f"Resolved {model_id} → {resolved}")
            return resolved

    return model_id


def _setup_tokenizer(
    llm_model_id: str = DEFAULT_LLM_MODEL_ID,
) -> AutoTokenizer:
    """
    Initialize the Llama-3.1 tokenizer with DeSTA's custom tokens.
    """
    resolved = _resolve_model_path(llm_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_tokens([AUDIO_LOCATOR])
    return tokenizer


def _prepare_audio_context_and_start_positions(
    token_list: List[str],
    audio_locator: str,
    audio_size_list: List[int],
    transcription_size_list: List[int],
    placeholder_token: str,
) -> Tuple[List[str], List[int]]:
    """
    Expand <|AUDIO|> tokens into placeholder runs.

    Mirrors DeSTA's _prepare_audio_context_and_start_positions exactly.
    Each <|AUDIO|> token is replaced with (audio_size + transcription_size)
    copies of placeholder_token.

    Returns:
        (expanded_token_list, start_positions)
    """
    result: List[str] = []
    start_positions: List[int] = []
    audio_idx = 0

    for token in token_list:
        if token == audio_locator:
            start_positions.append(len(result))
            total = audio_size_list[audio_idx] + transcription_size_list[audio_idx]
            result.extend([placeholder_token] * total)
            audio_idx += 1
        else:
            result.append(token)

    return result, start_positions


def build_prompt_embeds(
    embed_path: str,
    transcription: Optional[str],
    system_prompt: str,
    user_prompt: str,
    tokenizer: AutoTokenizer,
    embed_layer: torch.nn.Embedding,
    embed_cache: Optional[Dict[str, Dict]] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, int]:
    """
    Build a full (1, seq_len, hidden_dim) prompt_embeds tensor for vLLM.

    This replicates what DeSTA's _prepare_inputs_for_llm does but produces
    a standalone tensor suitable for vLLM's prompt_embeds input.

    Args:
        embed_path: Path to the precomputed *_embed.pt file.
        transcription: Speech transcription text (or None / " ").
        system_prompt: System message content.
        user_prompt: User message content.
        tokenizer: Llama-3.1 tokenizer with <|AUDIO|> added.
        embed_layer: The LLM's embed_tokens layer.
        embed_cache: Optional pre-loaded cache (avoids repeated disk reads).
        dtype: Target dtype.
        device: Target device.

    Returns:
        (prompt_embeds, prompt_length) — the tensor and its sequence length.
    """
    # Load precomputed embed
    if embed_cache and embed_path in embed_cache:
        raw = embed_cache[embed_path]
    else:
        raw = torch.load(embed_path, map_location="cpu", weights_only=True)

    if isinstance(raw, dict):
        qformer = raw["qformer"]  # (P, H)
        vad = bool(raw.get("vad", False))
    else:
        # Legacy: flat tensor
        qformer = raw
        vad = False

    audio_size = qformer.size(0)  # typically 64

    # Transcription handling
    if vad and transcription and transcription.strip():
        trans_token_count = len(
            tokenizer.tokenize(transcription, add_special_tokens=False)
        )
    else:
        trans_token_count = 0
        transcription = " "

    # Build chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{AUDIO_LOCATOR}\n{user_prompt}",
        },
    ]
    audio_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audio_context = audio_context.replace(
        AUDIO_LOCATOR,
        f"<start_audio>{AUDIO_LOCATOR}<end_audio>",
    )

    # Expand <|AUDIO|> → placeholder tokens
    tokens = tokenizer.tokenize(audio_context)
    expanded_tokens, start_positions = _prepare_audio_context_and_start_positions(
        token_list=tokens,
        audio_locator=AUDIO_LOCATOR,
        audio_size_list=[audio_size],
        transcription_size_list=[trans_token_count],
        placeholder_token=PLACEHOLDER_TOKEN,
    )

    # Re-encode to IDs
    text = tokenizer.convert_tokens_to_string(expanded_tokens)
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    # Build full embedding sequence
    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids).clone()  # (1, seq_len, H)

    # Prepare audio embeddings to splice in
    qformer = qformer.to(device=device, dtype=dtype)
    if trans_token_count > 0:
        trans_ids = tokenizer.encode(
            transcription, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        trans_emb = embed_layer(trans_ids).squeeze(0)  # (T, H)
        audio_emb = torch.cat([qformer, trans_emb], dim=0)  # (P+T, H)
    else:
        audio_emb = qformer  # (P, H)

    # Splice audio embeddings into the placeholder region
    assert len(start_positions) == 1, "Expected exactly one audio slot"
    sp = start_positions[0]
    inputs_embeds[0, sp : sp + audio_emb.size(0)] = audio_emb.to(
        dtype=inputs_embeds.dtype
    )

    return inputs_embeds.squeeze(0), inputs_embeds.size(1)


def build_prompt_embeds_continuation(
    embed_path: str,
    transcription: Optional[str],
    system_prompt: str,
    user_prompt: str,
    assistant_phase1: str,
    injected_tool_output: str,
    tokenizer: AutoTokenizer,
    embed_layer: torch.nn.Embedding,
    embed_cache: Optional[Dict[str, Dict]] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, int]:
    """
    Build prompt_embeds for Phase 2 generation (tool continuation).

    The conversation is:
      system → user(audio + question) → assistant(phase1) → user(tool_output) → assistant(...)

    Returns:
        (prompt_embeds, prompt_length)
    """
    # Load precomputed embed
    if embed_cache and embed_path in embed_cache:
        raw = embed_cache[embed_path]
    else:
        raw = torch.load(embed_path, map_location="cpu", weights_only=True)

    if isinstance(raw, dict):
        qformer = raw["qformer"]
        vad = bool(raw.get("vad", False))
    else:
        qformer = raw
        vad = False

    audio_size = qformer.size(0)

    if vad and transcription and transcription.strip():
        trans_token_count = len(
            tokenizer.tokenize(transcription, add_special_tokens=False)
        )
    else:
        trans_token_count = 0
        transcription = " "

    # Build multi-turn conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{AUDIO_LOCATOR}\n{user_prompt}"},
        {"role": "assistant", "content": assistant_phase1},
        {"role": "user", "content": injected_tool_output},
    ]
    audio_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    audio_context = audio_context.replace(
        AUDIO_LOCATOR,
        f"<start_audio>{AUDIO_LOCATOR}<end_audio>",
    )

    tokens = tokenizer.tokenize(audio_context)
    expanded_tokens, start_positions = _prepare_audio_context_and_start_positions(
        token_list=tokens,
        audio_locator=AUDIO_LOCATOR,
        audio_size_list=[audio_size],
        transcription_size_list=[trans_token_count],
        placeholder_token=PLACEHOLDER_TOKEN,
    )

    text = tokenizer.convert_tokens_to_string(expanded_tokens)
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids).clone()

    qformer = qformer.to(device=device, dtype=dtype)
    if trans_token_count > 0:
        trans_ids = tokenizer.encode(
            transcription, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        trans_emb = embed_layer(trans_ids).squeeze(0)
        audio_emb = torch.cat([qformer, trans_emb], dim=0)
    else:
        audio_emb = qformer

    assert len(start_positions) == 1
    sp = start_positions[0]
    inputs_embeds[0, sp : sp + audio_emb.size(0)] = audio_emb.to(
        dtype=inputs_embeds.dtype
    )

    return inputs_embeds.squeeze(0), inputs_embeds.size(1)
