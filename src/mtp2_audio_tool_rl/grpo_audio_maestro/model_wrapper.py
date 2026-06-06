"""
DeSTA25GRPOModel — wraps DeSTA25AudioModel for GRPO training.

Responsibilities:
  1. generate_completions(): sample G completions per question (two-phase for tool calls)
  2. compute_log_probs():    token-level log-probs under the policy model
  3. reference_log_probs():  token-level log-probs under the frozen reference model

Two-phase generation for tool-calling completions:
  Phase 1 — model generates <think>...<tool>...</tool>
  Phase 2 — cached <tool_output> is injected, then model generates <think>...<answer>
  Log-probs span ONLY the model-generated tokens; injected <tool_output> is masked out.
"""

import json
import logging
import re
import copy
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_tool_call(text: str) -> Optional[str]:
    """
    Return the tool function name if a well-formed <tool>...</tool> block is found.
    Expects JSON array: [{"function": "...", "parameters": {...}}]
    Returns None if no valid tool call found.
    """
    m = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(1).strip())
        if isinstance(arr, list) and arr:
            return arr[0].get("function")
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _format_tool_output(tool_name: str, tool_result: Any) -> str:
    """Render a <tool_output> block from cached tool results."""
    result_str = json.dumps(tool_result, indent=2) if not isinstance(tool_result, str) else tool_result
    return f"\n<tool_output>\n{result_str}\n</tool_output>\n"


# ---------------------------------------------------------------------------
# DeSTA25GRPOModel
# ---------------------------------------------------------------------------

class DeSTA25GRPOModel:
    """
    Thin wrapper around DeSTA25AudioModel that adds GRPO-specific methods.

    Args:
        model:   Loaded DeSTA25AudioModel instance (with use_lora=True).
        device:  torch.device for inference/training.
    """

    # Class-level flag so trainer knows which model family this is
    model_family = "desta"

    def __init__(self, model, device: torch.device):
        self.model  = model
        self.device = device

    # ------------------------------------------------------------------
    # Message format helpers (called by trainer for model-agnostic code)
    # ------------------------------------------------------------------

    def build_messages_phase1(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> List[Dict]:
        """Build phase-1 messages in DeSTA format."""
        transcription = kwargs.get("transcription")
        embed_path = kwargs.get("precomputed_embed")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{user_prompt}",
                "audios": [{"audio": audio_path, "text": transcription, "embed_path": embed_path}],
            },
        ]

    def build_messages_phase2(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> List[Dict]:
        """Build phase-2 (follow-up) messages in DeSTA format."""
        transcription = kwargs.get("transcription")
        embed_path = kwargs.get("precomputed_embed")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{user_prompt}",
                "audios": [{"audio": audio_path, "text": transcription, "embed_path": embed_path}],
            },
        ]

    def get_llm_backbone(self):
        """Return the core LLM module (for gradient checkpointing control)."""
        return self.model.llm_model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_completions(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        cached_tool_outputs: Dict[str, Any],
        followup_system_prompt: str,
        followup_user_template,   # callable(question, choices, tool_output_str) -> (sys, user)
        question: str,
        choices: List[str],
        G: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        transcription: Optional[str] = None,  # cached ASR text — skips Whisper decoder/VAD
        precomputed_embed: Optional[str] = None, # Path to .pt tensor for audio embeddings
    ) -> List[Dict]:
        """
        Sample G completions for one question.

        Uses ``num_return_sequences=G`` to batch all phase-1 generations into a
        single ``model.generate()`` call, which shares the KV-cache of the prompt
        and decodes G sequences in parallel — ~G× faster than sequential calls.

        Phase-2 follow-ups (for completions that called a tool) are also batched
        per-tool group.

        Each completion is a dict:
          {
            "text":          str,   # full final text (phase1 + injected + phase2 OR direct)
            "phase1":        str,   # first model output (think + optional tool)
            "phase2":        str,   # second model output (think + answer) — empty if no tool
            "called_tools":  bool,
            "tool_name":     str | None,
            "injected_output": str | None,  # the <tool_output> block that was injected
          }
        """
        completions = []

        messages_phase1 = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{user_prompt}",
                "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
            },
        ]

        # ---- Phase 1: batch-generate G completions in one call ----
        with torch.inference_mode():
            out = self.model.generate(
                messages=messages_phase1,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=G,
            )
        phase1_texts = out.text if isinstance(out.text, list) else [out.text]

        # ---- Classify: which completions called a tool? ----
        # Group tool-calling completions by tool_name for batched phase-2
        tool_groups: Dict[str, List[int]] = {}   # tool_name -> [indices]
        for idx, p1 in enumerate(phase1_texts):
            tool_name = _parse_tool_call(p1)
            if tool_name and tool_name in cached_tool_outputs:
                tool_groups.setdefault(tool_name, []).append(idx)
            else:
                completions.append({
                    "text":             p1,
                    "phase1":           p1,
                    "phase2":           "",
                    "called_tools":     False,
                    "tool_name":        None,
                    "injected_output":  None,
                    "_order":           idx,
                })

        # ---- Phase 2: batch-generate follow-ups per tool group ----
        for tool_name, indices in tool_groups.items():
            tool_result = cached_tool_outputs[tool_name]
            injected    = _format_tool_output(tool_name, tool_result)

            f_sys, f_user = followup_user_template(
                question, choices, json.dumps(tool_result, indent=2)
            )
            messages_phase2 = [
                {"role": "system", "content": f_sys},
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{f_user}",
                    "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
                },
            ]

            n_p2 = len(indices)
            with torch.inference_mode():
                out2 = self.model.generate(
                    messages=messages_phase2,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=n_p2,
                )
            phase2_texts = out2.text if isinstance(out2.text, list) else [out2.text]

            for sub_idx, orig_idx in enumerate(indices):
                p2 = phase2_texts[sub_idx] if sub_idx < len(phase2_texts) else ""
                p1 = phase1_texts[orig_idx]
                completions.append({
                    "text":             p1 + injected + p2,
                    "phase1":           p1,
                    "phase2":           p2,
                    "called_tools":     True,
                    "tool_name":        tool_name,
                    "injected_output":  injected,
                    "_order":           orig_idx,
                })

        # Restore original order and strip the helper key
        completions.sort(key=lambda c: c.pop("_order", 0))
        return completions

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    def _compute_logprobs_for_messages(
        self,
        messages: List[Dict],
        completion_text: str,
        no_grad: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute mean per-token log-prob for completion_text given messages context.

        The prompt is constructed EXACTLY as in generate():
          1. apply_chat_template → text string
          2. Replace <|AUDIO|> → <start_audio><|AUDIO|><end_audio>
          3. _prepare_audio_context_and_start_positions expands <|AUDIO|> into
             audio_size + transcription_size placeholder tokens
          4. Tokenize the expanded prompt → prompt_ids (matches context_input_ids in generate)
          5. Append completion tokens → full_ids
          6. Forward through model with correct batch_start_positions
          7. Slice logits at S_prompt to score completion tokens

        This ensures S_prompt and batch_start_positions match what the model saw
        during generation — any mismatch would bias the log-probs.
        """
        tokenizer = self.model.tokenizer

        # --- locate audio in messages ---
        audio_path = None
        precomputed_embed_path = None
        transcription_text = None
        for msg in messages:
            if "audios" in msg and msg["audios"]:
                audio_path             = msg["audios"][0]["audio"]
                precomputed_embed_path = msg["audios"][0].get("embed_path")
                transcription_text     = msg["audios"][0].get("text")  # may be None
                break

        if audio_path is None:
            raise ValueError("No audio path found in messages for log-prob computation.")

        # --- resolve audio_size and transcription_size (mirrors generate()) ---
        if precomputed_embed_path:
            raw = torch.load(precomputed_embed_path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict):
                audio_size = raw["qformer"].size(0)       # e.g. 64
                vad        = bool(raw.get("vad", False))
                if vad and transcription_text and transcription_text.strip():
                    transcription_size = len(tokenizer.tokenize(
                        transcription_text, add_special_tokens=False
                    ))
                else:
                    transcription_size = 0
            else:
                # legacy flat tensor: audio + transcription already merged
                audio_size         = raw.size(0)
                transcription_size = 0
            batch_features = torch.zeros(
                (1, 1, 1), device=self.device, dtype=torch.bfloat16
            )
        else:
            # live Whisper path - use config prompt_size
            audio_size         = self.model.config.prompt_size
            transcription_size = len(tokenizer.tokenize(
                transcription_text or " ", add_special_tokens=False
            ))
            from desta.utils.audio import AudioSegment  # type: ignore
            audio_array = AudioSegment.from_file(
                audio_path, target_sr=16000, channel_selector="average"
            ).samples
            batch_features = self.model.processor(
                [audio_array], sampling_rate=16000, return_tensors="pt"
            ).input_features.to(self.device)

        audio_size_list         = [audio_size]
        transcription_size_list = [transcription_size]

        # --- build expanded prompt_ids exactly as generate() does ---
        from grpo.modeling_grpo import _prepare_audio_context_and_start_positions  # type: ignore

        audio_context = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio_context = audio_context.replace(
            self.model.audio_locator,
            f"<start_audio>{self.model.audio_locator}<end_audio>",
        )
        audio_context_tokens, start_positions = _prepare_audio_context_and_start_positions(
            token_list=tokenizer.tokenize(audio_context),
            audio_locator=self.model.audio_locator,
            audio_size_list=audio_size_list,
            transcription_size_list=transcription_size_list,
            placeholder_token=self.model.placeholder_token,
        )
        audio_context_str = tokenizer.convert_tokens_to_string(audio_context_tokens)

        prompt_enc = tokenizer(
            audio_context_str,
            return_tensors="pt",
            add_special_tokens=False,
            return_length=True,
        )
        prompt_ids  = prompt_enc["input_ids"].to(self.device)       # (1, S_prompt)
        S_prompt    = prompt_ids.shape[1]

        # Adjust start_position for left-padding (no padding here since batch=1)
        batch_start_positions = [(0, start_positions[0])]

        # --- completion tokens ---
        completion_ids = tokenizer(
            completion_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)                                  # (1, S_comp)

        full_ids  = torch.cat([prompt_ids, completion_ids], dim=1)  # (1, S_full)
        attn_mask = torch.ones_like(full_ids)

        # --- batch_transcription_ids: mirrors generate() ---
        trans_str = transcription_text if (transcription_size > 0 and transcription_text) else " "
        batch_transcription_ids = [
            tokenizer.encode(
                trans_str, add_special_tokens=False, return_tensors="pt"
            ).long().to(self.device)
        ]

        # --- forward pass ---
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            outputs = self.model.forward(
                input_ids=full_ids,
                attention_mask=attn_mask,
                batch_features=batch_features,
                batch_transcription_ids=batch_transcription_ids,
                batch_start_positions=batch_start_positions,
                precomputed_embeds=[precomputed_embed_path] if precomputed_embed_path else None,
            )

        # logits[i] predicts token i+1
        # completion token at position S_prompt+j is predicted by logits[S_prompt+j-1]
        logits       = outputs.logits[0]                             # (S_full, vocab)
        comp_logits  = logits[S_prompt - 1: S_prompt - 1 + completion_ids.shape[1]]
        comp_targets = completion_ids[0]

        log_probs       = F.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs.gather(1, comp_targets.unsqueeze(1)).squeeze(1)

        n_tokens      = token_log_probs.shape[0]
        mean_log_prob = token_log_probs.sum() / max(n_tokens, 1)
        return mean_log_prob, n_tokens

    def compute_log_probs(
        self,
        messages: List[Dict],
        completion_text: str,
    ) -> Tuple[torch.Tensor, int]:
        """Log-probs under the policy (trainable) model. Gradients enabled."""
        return self._compute_logprobs_for_messages(messages, completion_text, no_grad=False)

    def reference_log_probs(
        self,
        messages: List[Dict],
        completion_text: str,
    ) -> Tuple[torch.Tensor, int]:
        """
        Log-probs under the *reference* (base) model. No gradients.

        With LoRA we never deepcopy the model — instead we temporarily disable
        all LoRA adapter layers so the forward pass runs through the frozen
        base weights only, then restore them.
        """
        # _lora_base_model is the PEFT LoraModel (peft_model.base_model), which
        # exposes disable_adapter_layers() / enable_adapter_layers().  This is
        # set by _enable_lora() alongside model.llm_model = peft_model.base_model.model.
        lora_base = getattr(self.model, "_lora_base_model", None)
        has_adapters = lora_base is not None and hasattr(lora_base, "disable_adapter_layers")
        try:
            if has_adapters:
                lora_base.disable_adapter_layers()
            return self._compute_logprobs_for_messages(messages, completion_text, no_grad=True)
        finally:
            if has_adapters:
                lora_base.enable_adapter_layers()
