"""
modeling_grpo.py — Decoder-free DeSTA2.5-Audio model for GRPO training.

Why a separate file?
--------------------
During GRPO rollouts each sample is already paired with a *cached*
speech_recognition transcript (pre-computed from Whisper at dataset build
time).  Re-running the Whisper decoder + VAD to re-transcribe the same audio
on every forward pass is therefore:

  1. Pure wasted compute   (~30 % of generate() wall-clock).
  2. Unnecessary GPU memory  (Whisper decoder weights + KV-cache).
  3. A fragile dependency on silero-vad (requires internet on first load).

This class subclasses DeSTA25AudioModel and overrides the two methods that
touch the decoder/VAD, leaving every other behaviour — Whisper encoder,
QFormer connector, LLM backbone, LoRA, log-prob scoring — untouched.

How pl_trainer.py avoided the decoder
--------------------------------------
The original PTL trainer trains on pre-aligned (audio, text) pairs and simply
deletes the decoder immediately after loading:

    del self.model.perception.whisper.model.decoder
    del self.model.perception.whisper.proj_out

We do the same in from_pretrained(), and additionally skip VAD in generate().

Usage
-----
    from grpo2.modeling_grpo import GRPODeSTA25AudioModel

    model = GRPODeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-...")
    model.generate(
        messages=[
            {"role": "system", "content": "..."},
            {
                "role": "user",
                "content": "<|AUDIO|>\\n...",
                # Pass transcript as "text" → skips decoder entirely
                "audios": [{"audio": "/path/to/audio.wav", "text": "hello world"}],
            },
        ],
        ...
    )

If `text` is provided for every audio in the message, the Whisper decoder is
never called.  If `text` is None for some audio, the fallback is an empty
string `" "` (silent / non-speech audio handling), consistent with the VAD
behaviour in the original code when no speech timestamps are detected.
"""

import os
import logging
import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flash-attention helper
# ---------------------------------------------------------------------------

def _best_attn_implementation() -> str:
    """
    Return the best available attention implementation:
      - "flash_attention_2"  if flash-attn package is installed
      - "sdpa"               otherwise (torch 2.0+ built-in; still uses
                              FlashAttention kernels on A100 via SDPA)
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


# Resolve once at import time so every process logs the same value.
_ATTN_IMPL: str = _best_attn_implementation()

_attn_status = (
    f"[ATTN] flash_attention_2 (flash-attn {__import__('flash_attn').__version__})"
    if _ATTN_IMPL == "flash_attention_2"
    else "[ATTN] sdpa (flash-attn not installed — using PyTorch SDPA)"
)
print(_attn_status, flush=True)
logger.info(_attn_status)


# ---------------------------------------------------------------------------
# Import base class — prefer local desta install, fall back to package root
# ---------------------------------------------------------------------------
try:
    from desta.models.modeling_desta25 import (   # type: ignore
        DeSTA25AudioModel,
        DeSTA25Config,
        GenerationOutput,
        _prepare_audio_context_and_start_positions,
    )
except ImportError:
    from desta import DeSTA25AudioModel  # type: ignore  # noqa: F401
    DeSTA25Config = None
    GenerationOutput = None
    _prepare_audio_context_and_start_positions = None


# ---------------------------------------------------------------------------
# GRPODeSTA25AudioModel
# ---------------------------------------------------------------------------

class GRPODeSTA25AudioModel(DeSTA25AudioModel):
    """
    DeSTA2.5-Audio with Whisper decoder and VAD removed for GRPO training.

    Differences from the base class:
      - LLM loaded with flash_attention_2 (or sdpa fallback) for faster generation.
      - Whisper decoder weights are deleted on load  →  lower GPU memory.
      - VAD (silero-vad) is never loaded             →  no internet + faster startup.
      - generate() uses provided "text" transcripts; falls back to " " when absent.
      - No ASR call ever takes place during GRPO rollouts.
      - When skip_perception=True, the entire Whisper encoder + Qformer are
        never loaded at all, saving ~1.5 GB VRAM.
    """

    # Set to True *before* calling from_pretrained() to completely skip
    # loading Whisper + Qformer (requires precomputed embeddings).
    skip_perception: bool = False
    supports_gradient_checkpointing = True

    # ------------------------------------------------------------------
    # Overridden: inject flash-attn when loading the LLM backbone
    # ------------------------------------------------------------------

    def __init__(self, config, cache_dir=None, token=None, **kwargs):
        """
        Same as parent __init__ but injects attn_implementation into the
        AutoModelForCausalLM.from_pretrained call via a scoped monkey-patch
        so we don't have to duplicate the full parent constructor.

        When cls.skip_perception is True, also monkey-patches
        WhisperPerception.__init__ to be a bare nn.Module (no Whisper load).
        """
        from desta.models.modeling_desta25 import WhisperPerception  # type: ignore

        _orig_llm = AutoModelForCausalLM.from_pretrained
        _orig_wp  = WhisperPerception.__init__

        def _patched_llm(*a, **kw):
            kw.setdefault("attn_implementation", _ATTN_IMPL)
            return _orig_llm(*a, **kw)

        def _noop_wp(self_wp, config_wp):
            """Minimal init — just an empty nn.Module, no Whisper loaded."""
            import torch.nn as nn
            nn.Module.__init__(self_wp)
            self_wp.config = config_wp

        AutoModelForCausalLM.from_pretrained = _patched_llm
        if self.__class__.skip_perception:
            WhisperPerception.__init__ = _noop_wp
            logger.info("GRPODeSTA25AudioModel: WhisperPerception.__init__ monkey-patched → Whisper NOT loaded.")

        try:
            super().__init__(config, cache_dir=cache_dir, token=token, **kwargs)
        finally:
            AutoModelForCausalLM.from_pretrained = _orig_llm  # always restore
            WhisperPerception.__init__ = _orig_wp              # always restore

        # Cache for precomputed embeddings — preloaded via load_embed_cache()
        self._embed_cache = {}

        logger.info(
            f"GRPODeSTA25AudioModel: LLM loaded with attn_implementation='{_ATTN_IMPL}'"
        )

    def load_embed_cache(self, embed_dir: str):
        """Preload all precomputed embeddings into CPU memory at startup."""
        import glob
        from tqdm import tqdm

        paths = sorted(glob.glob(os.path.join(embed_dir, "*_embed.pt")))
        if not paths:
            logger.warning(f"No embed files found in {embed_dir}")
            return

        for p in tqdm(paths, desc="Loading embeddings", unit="file"):
            self._embed_cache[p] = torch.load(p, map_location="cpu", weights_only=True)

        total_mb = sum(
            v["qformer"].nelement() * v["qformer"].element_size() if isinstance(v, dict)
            else v.nelement() * v.element_size()
            for v in self._embed_cache.values()
        ) / (1024 ** 2)
        logger.info(f"Embed cache: {len(self._embed_cache)} files, {total_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Overridden: skip perception.whisper references when it doesn't exist
    # ------------------------------------------------------------------

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm_model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        if gradient_checkpointing_kwargs is not None:
            kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs
        self.llm_model.config.use_cache = False
        self.llm_model.gradient_checkpointing_enable(**kwargs)
        
    def gradient_checkpointing_disable(self):
        self.llm_model.gradient_checkpointing_disable()
        self.llm_model.config.use_cache = True

    def state_dict(self, *args, **kwargs):
        # Base DeSTA25AudioModel.state_dict() doesn't accept the standard
        # PyTorch kwargs (destination, prefix, keep_vars) that nn.Module
        # passes when recursing.  Bypass it and use nn.Module's version
        # so PEFT's save_pretrained / get_peft_model_state_dict works.
        return torch.nn.Module.state_dict(self, *args, **kwargs)

    def configure_trainable_parameters(self):
        """
        Same as parent but gracefully handles missing perception.whisper
        when skip_perception is True (precomputed embeddings mode).
        """
        known_parameters = []
        # Freeze LLM parameters
        for name, params in self.llm_model.named_parameters():
            params.requires_grad = False
            known_parameters.append(f"llm_model.{name}")

        # Freeze encoder parameters (only if whisper actually exists)
        if hasattr(self, "perception") and hasattr(self.perception, "whisper"):
            for name, params in self.perception.whisper.named_parameters():
                params.requires_grad = False
                known_parameters.append(f"perception.whisper.{name}")

        # Make other parameters or lora parameters trainable
        self.trainable_parameter_names = []
        trainable_parameters = []
        for name, params in self.named_parameters():
            if name not in known_parameters or "lora" in name:
                params.requires_grad = True
                self.trainable_parameter_names.append(name)
                trainable_parameters.append(params)

    # ------------------------------------------------------------------
    # Overridden: drop decoder weights immediately after load
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load weights then conditionally strip perception components.

        If cls.skip_perception is True (precomputed embeds mode):
            → delete the entire perception module (Whisper + Qformer).
        Otherwise:
            → only delete the Whisper decoder (original behaviour).
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if cls.skip_perception:
            # Nuke the entire perception stack — never needed with precomputed embeds.
            if hasattr(model, "perception"):
                del model.perception
                import torch.nn as nn
                model.perception = nn.Identity()  # placeholder so hasattr checks don't crash
                logger.info(
                    "GRPODeSTA25AudioModel: Entire perception module (Whisper encoder + Qformer) "
                    "deleted — using precomputed embeddings."
                )
        else:
            # Only delete Whisper decoder sub-modules to free GPU/CPU memory
            if hasattr(model, "perception") and hasattr(model.perception, "whisper"):
                if hasattr(model.perception.whisper.model, "decoder"):
                    del model.perception.whisper.model.decoder
                    logger.info("GRPODeSTA25AudioModel: Whisper decoder deleted (not needed for GRPO).")

                if hasattr(model.perception.whisper, "proj_out"):
                    del model.perception.whisper.proj_out
                    logger.info("GRPODeSTA25AudioModel: Whisper proj_out deleted.")

        return model

    # ------------------------------------------------------------------
    # Overridden: skip VAD during _setup_generation
    # ------------------------------------------------------------------

    def _setup_generation(self):
        """
        Same as base class but WITHOUT loading silero-vad.
        We never need VAD because transcripts are always pre-supplied.
        When skip_perception is True, also skip the Whisper AutoProcessor.
        """
        from transformers import AutoTokenizer, AutoProcessor  # already imported in base

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_id, cache_dir=os.getenv("HF_HOME")
        )
        self.tokenizer.pad_token        = self.tokenizer.eos_token
        self.tokenizer.pad_token_id     = self.tokenizer.eos_token_id
        self.tokenizer.padding_side     = "left"
        self.tokenizer.add_tokens([self.audio_locator])

        if not self.__class__.skip_perception:
            self.processor = AutoProcessor.from_pretrained(
                self.config.encoder_model_id, cache_dir=os.getenv("HF_HOME")
            )
        else:
            self.processor = None
            logger.info("GRPODeSTA25AudioModel: Whisper AutoProcessor skipped (precomputed embeds mode).")

        assert len(self.tokenizer.tokenize(self.audio_locator)) == 1, \
            "audio_locator must be a single token"
        assert len(self.tokenizer.tokenize(self.placeholder_token)) == 1, \
            "placeholder_token must be a single token in the tokenizer"

        # NOTE: silero-vad is intentionally NOT loaded here.
        logger.info("GRPODeSTA25AudioModel: _setup_generation done (VAD skipped).")

    # ------------------------------------------------------------------
    # Overridden: generate() without VAD or decoder
    # ------------------------------------------------------------------


    def forward(self, input_ids,
                attention_mask,
                batch_features,
                batch_transcription_ids,
                batch_start_positions,
                labels=None,
                precomputed_embeds=None,
                **kwargs):

        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=batch_features,
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
            precomputed_embeds=precomputed_embeds,
        )

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def _prepare_inputs_for_llm(self,
                               input_ids,
                               attention_mask,
                               batch_features,
                               batch_transcription_ids,
                               batch_start_positions,
                               precomputed_embeds=None,
        ):
        
        N_audio = len(batch_start_positions)
        inputs_embeds = self.llm_model.model.embed_tokens(input_ids).clone()
        
        if precomputed_embeds is not None and N_audio > 0:
            for audio_batch_idx in range(N_audio):
                start_position    = batch_start_positions[audio_batch_idx]
                text_batch_idx    = start_position[0]
                audio_start_pos   = start_position[1]

                emb_path = precomputed_embeds[audio_batch_idx]
                raw = self._embed_cache[emb_path]

                if isinstance(raw, dict):
                    # ── new format: {"qformer": Tensor, "vad": bool} ──
                    qformer = raw["qformer"].to(self.device, dtype=inputs_embeds.dtype)  # (P, H)
                    vad     = bool(raw["vad"])
                    if vad and batch_transcription_ids is not None:
                        trans_ids = batch_transcription_ids[audio_batch_idx]
                        if trans_ids.dim() == 2:
                            trans_ids = trans_ids.squeeze(0)   # (T,)
                        if trans_ids.numel() > 0:
                            trans_emb = self.llm_model.model.embed_tokens(
                                trans_ids.to(self.device)
                            )                                  # (T, H)
                            audio_embeddings = torch.cat([qformer, trans_emb], dim=0)
                        else:
                            audio_embeddings = qformer
                    else:
                        audio_embeddings = qformer
                else:
                    # ── legacy format: flat concatenated tensor ──
                    audio_embeddings = raw.to(self.device, dtype=inputs_embeds.dtype)

                target_slice = slice(audio_start_pos, audio_start_pos + audio_embeddings.size(0))
                inputs_embeds[text_batch_idx, target_slice] = audio_embeddings

            return inputs_embeds
            
        # fallback to original
        transcription_embeddings_list = []
        with torch.no_grad():
            for audio_batch_idx in range(N_audio):
                trans_emb = self.llm_model.model.embed_tokens(
                    batch_transcription_ids[audio_batch_idx].squeeze(0)
                ) 
                transcription_embeddings_list.append(trans_emb)
                
        batch_audio_features, _ = self.perception(
            input_features=batch_features, transcription_embeddings_list=transcription_embeddings_list
        )
        
        for audio_batch_idx in range(N_audio):
            start_position = batch_start_positions[audio_batch_idx] 
            text_batch_idx = start_position[0]
            audio_start_position = start_position[1]
            
            audio_features = batch_audio_features[audio_batch_idx]
            trans_emb = transcription_embeddings_list[audio_batch_idx] 
            audio_embeddings = torch.cat([audio_features, trans_emb], dim=0)

            target_slice = slice(audio_start_position, audio_start_position + audio_embeddings.size(0))
            inputs_embeds[text_batch_idx, target_slice] = audio_embeddings
            
        return inputs_embeds

    def _generate_step(self, inputs, pad_token_id, temperature=0.7, top_p=0.9, max_new_tokens=512, do_sample=True, **kwargs):
        input_ids = inputs["context_input_ids"] # only context inputs
        attention_mask = inputs["context_attention_mask"] # only context attention mask
        batch_start_positions = inputs["context_batch_start_positions"]

        batch_transcription_ids = inputs["batch_transcription_ids"]

        # get the generated text
        inputs_embeds = self._prepare_inputs_for_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            batch_features=inputs["batch_features"],
            batch_transcription_ids=batch_transcription_ids,
            batch_start_positions=batch_start_positions,
            precomputed_embeds=inputs.get("precomputed_embeds")
        )

        if do_sample is False:
            top_p = None
            temperature = None

        # Ensure generation stops at Llama-3.1 stop tokens
        eos_token_id = kwargs.pop("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.llm_model.config, "eos_token_id", None)
            if eos_token_id is None:
                eos_token_id = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        messages,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        max_new_tokens=512,
        **kwargs,
    ):
        """
        Identical to base-class generate() **except**:

          1. VAD is never run.
          2. The Whisper decoder is never called.
          3. When `audio["text"]` is provided it is used as-is as the transcript.
          4. When `audio["text"]` is None, the transcript defaults to " " (as the
             base class does for non-speech audio after VAD returns no timestamps).

        All other behaviour (Whisper encoder → QFormer → LLM) is unchanged.
        """
        if not hasattr(self, "tokenizer"):
            self._setup_generation()

        # ---- normalise to list-of-conversation-lists ----
        if isinstance(messages, list):
            if isinstance(messages[0], dict):
                messages_list = [messages]
            else:
                messages_list = messages
        else:
            raise ValueError("messages should be a list of dicts or a list of lists.")

        # ---- collect audio paths + transcriptions ----
        all_audios: list = []
        all_transcriptions: list = []
        all_embed_paths: list = []

        for conv in messages_list:
            for message in conv:
                content = message["content"]
                audios  = message.get("audios", [])
                assert len(audios) == content.count(self.audio_locator), \
                    "audio count does not match <|AUDIO|> count in content"
                for audio in audios:
                    all_audios.append(audio["audio"])
                    # Use provided text; fall back to " " (mirrors base-class
                    # behaviour for non-speech audio where VAD returns nothing).
                    transcript = audio.get("text")
                    all_embed_paths.append(audio.get("embed_path"))
                    all_transcriptions.append(transcript if transcript is not None else " ")

        # ---- audio path guard ----
        for audio_path in all_audios:
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found: {audio_path}")

        if len(all_audios) > 0:
            from desta.utils.audio import AudioSegment   # type: ignore

            all_precomputed = len(all_embed_paths) > 0 and all(
                p and os.path.exists(p) for p in all_embed_paths
            )

            # Resolve encoder dtype safely (perception may have been deleted)
            _enc_dtype = (
                self.perception.whisper.model.encoder.conv1.weight.dtype
                if hasattr(self.perception, "whisper")
                else torch.bfloat16
            )

            if all_precomputed:
                 batch_features = torch.zeros((len(all_audios), 1, 1), device=self.device, dtype=_enc_dtype)
            else:
                 raw_features = []
                 for audio_path in all_audios:
                     feature = AudioSegment.from_file(
                         audio_path, target_sr=16000, channel_selector="average"
                     ).samples
                     raw_features.append(feature)

                 batch_features = self.processor(
                     raw_features, sampling_rate=16000, return_tensors="pt"
                 ).input_features
                 batch_features = batch_features.to(self.device)
                 batch_features = batch_features.to(dtype=_enc_dtype)

            if all_precomputed:
                # Determine placeholder sizes from the stored embed dict (or legacy tensor)
                audio_size_list        = []
                transcription_size_list = []
                for ep, text in zip(all_embed_paths, all_transcriptions):
                    raw = self._embed_cache[ep]
                    if isinstance(raw, dict):
                        # new format
                        audio_size_list.append(raw["qformer"].size(0))  # prompt_size (e.g. 64)
                        if raw["vad"] and text.strip():
                            transcription_size_list.append(
                                len(self.tokenizer.tokenize(text, add_special_tokens=False))
                            )
                        else:
                            transcription_size_list.append(0)
                    else:
                        # legacy: full embed already contains both parts
                        audio_size_list.append(raw.size(0))
                        transcription_size_list.append(0)
            else:
                audio_size_list = [self.config.prompt_size] * len(all_audios)

                # NO ASR — transcriptions are already ready from above

                transcription_size_list = [
                    len(self.tokenizer.tokenize(text, add_special_tokens=False))
                    for text in all_transcriptions
                ]

            # ---- build tokenised audio context ----
            audio_context_list       = []
            start_positions_list     = []

            for conv in messages_list:
                audio_context = self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                audio_context = audio_context.replace(
                    self.audio_locator,
                    f"<start_audio>{self.audio_locator}<end_audio>",
                )
                audio_context, start_positions = _prepare_audio_context_and_start_positions(
                    token_list=self.tokenizer.tokenize(audio_context),
                    audio_locator=self.audio_locator,
                    audio_size_list=audio_size_list,
                    transcription_size_list=transcription_size_list,
                    placeholder_token=self.placeholder_token,
                )
                audio_context = self.tokenizer.convert_tokens_to_string(audio_context)
                audio_context_list.append(audio_context)
                start_positions_list.append(start_positions)

            audio_context_inputs = self.tokenizer(
                audio_context_list,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_length=True,
                add_special_tokens=False,
            )

            audio_context_batch_start_positions = []
            for i in range(audio_context_inputs["length"].size(0)):
                total_length = audio_context_inputs["length"][i]
                pad_length   = total_length - audio_context_inputs["attention_mask"][i].sum()
                for sp in start_positions_list[i]:
                    audio_context_batch_start_positions.append((i, sp + pad_length))

            batch_transcription_ids = [
                self.tokenizer.encode(
                    text, add_special_tokens=False, return_tensors="pt"
                ).long().to(self.device)
                for text in all_transcriptions
            ]

            inputs = {
                "batch_features":                  batch_features,
                "batch_transcription_ids":         batch_transcription_ids,
                "context_input_ids":               audio_context_inputs["input_ids"],
                "context_attention_mask":          audio_context_inputs["attention_mask"],
                "context_batch_start_positions":   audio_context_batch_start_positions,
                "precomputed_embeds":              all_embed_paths,
            }
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            generated_ids = self._generate_step(
                inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True),
                audios=[(a, t) for a, t in zip(all_audios, all_transcriptions)],
                generated_ids=generated_ids.tolist(),
            )

        else:
            # ---- text-only fallback (no audio in messages) ----
            inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            generated_ids = self.llm_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                eos_token_id=terminators,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **kwargs,
            )

            generated_ids_list = [
                generated_ids[i][inputs["input_ids"].shape[1]:].tolist()
                for i in range(len(generated_ids))
            ]

            return GenerationOutput(
                text=self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True),
                audios=[],
                generated_ids=generated_ids_list,
            )
