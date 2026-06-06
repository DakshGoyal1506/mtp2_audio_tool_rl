"""
QwenOmniGRPOModel — wraps Qwen2.5-Omni for GRPO training.

Drop-in replacement interface for DeSTA25GRPOModel so the trainer can work
with either model transparently.

Responsibilities:
  1. generate_completions(): sample G completions per question (two-phase for tool calls)
  2. compute_log_probs():    mean per-token log-prob under the policy model
  3. reference_log_probs():  mean per-token log-prob under the frozen reference model
  4. build_messages():       construct Qwen-format messages (used by trainer)
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _SuppressQwenSysPromptWarning(logging.Filter):
    """
    Filter out the noisy 'System prompt modified' warning emitted by
    Qwen2.5-Omni when a non-default system prompt is used.

    This warning is about audio (speech) output quality. Since the Talker
    module is disabled (text-only mode) the warning is completely irrelevant.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        return "System prompt modified" not in record.getMessage()


def _install_qwen_warning_filter() -> None:
    """Install the filter on the root logger (once is enough)."""
    root = logging.getLogger()
    for f in root.filters:
        if isinstance(f, _SuppressQwenSysPromptWarning):
            return  # already installed
    root.addFilter(_SuppressQwenSysPromptWarning())


# ---------------------------------------------------------------------------
# Parsing helpers (shared with DeSTA wrapper)
# ---------------------------------------------------------------------------

def _parse_tool_call(text: str) -> Optional[str]:
    """Return the tool function name if a well-formed <tool>...</tool> block is found."""
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
    result_str = json.dumps(tool_result, indent=2)
    return f"\n<tool_output>\n{result_str}\n</tool_output>\n"


class QwenOmniGRPOModel:
    """
    Thin wrapper around Qwen2.5-Omni that adds GRPO-specific methods.

    Interface is identical to DeSTA25GRPOModel so GRPOTrainer works with both.

    Args:
        model:      Loaded Qwen2_5OmniForConditionalGeneration instance.
        processor:  Qwen2_5OmniProcessor instance.
        device:     torch.device for inference/training.
    """

    # Class-level flag so trainer knows which model family this is
    model_family = "qwen"

    def __init__(self, model, processor, device: torch.device):
        self.model     = model
        self.processor = processor
        self.device    = device
        # Suppress the harmless "System prompt modified" warning — the Talker
        # is disabled so audio output quality warnings are irrelevant.
        _install_qwen_warning_filter()

    def get_llm_backbone(self):
        """Return the core LLM module (for gradient checkpointing control)."""
        return self.model.thinker if hasattr(self.model, 'thinker') else self.model

    # ------------------------------------------------------------------
    # Qwen message format helpers
    # ------------------------------------------------------------------

    def _build_qwen_messages(
        self,
        system_prompt: str,
        user_text: str,
        audio_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build Qwen2.5-Omni chat-format messages.

        Qwen's format uses typed content items:
          {"type": "text", "text": "..."} and {"type": "audio", "audio": "path"}
        """
        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
        user_content = []
        if audio_path:
            user_content.append({"type": "audio", "audio": audio_path})
        user_content.append({"type": "text", "text": user_text})
        user_msg = {"role": "user", "content": user_content}
        return [system_msg, user_msg]

    def build_messages_phase1(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> List[Dict]:
        """Build phase-1 messages in Qwen format. Used by trainer for log-prob computation."""
        return self._build_qwen_messages(system_prompt, user_prompt, audio_path)

    def build_messages_phase2(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> List[Dict]:
        """Build phase-2 messages in Qwen format. Used by trainer for log-prob computation."""
        return self._build_qwen_messages(system_prompt, user_prompt, audio_path)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_text(
        self,
        messages: List[Dict],
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text completions from Qwen2.5-Omni given chat messages.

        Uses the Thinker (text-only output) — Talker is disabled.
        """
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError(
                "qwen-omni-utils not installed. Install with: pip install qwen-omni-utils[decord] -U"
            )

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        results = []
        # Qwen's generate with return_audio=False returns text_ids directly
        for _ in range(num_return_sequences):
            text_ids = self.model.generate(
                **inputs,
                return_audio=False,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
            )
            decoded = self.processor.batch_decode(
                text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            results.append(decoded[0] if decoded else "")

        return results

    def generate_completions(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        cached_tool_outputs: Dict[str, Any],
        followup_system_prompt: str,
        followup_user_template,
        question: str,
        choices: List[str],
        G: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        transcription: Optional[str] = None,
        precomputed_embed: Optional[str] = None,
    ) -> List[Dict]:
        """
        Sample G completions for one question.

        Each completion dict matches the DeSTA schema:
          {
            "text":             full final text
            "phase1":           first model output
            "phase2":           second model output (empty if no tool)
            "called_tools":     bool
            "tool_name":        str | None
            "injected_output":  str | None
          }
        """
        completions = []

        messages_phase1 = self._build_qwen_messages(system_prompt, user_prompt, audio_path)

        # ---- Phase 1: generate G completions ----
        with torch.inference_mode():
            phase1_texts = self._generate_text(
                messages_phase1,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=G,
            )

        # ---- Classify: which called a tool? ----
        tool_groups: Dict[str, List[int]] = {}
        for idx, p1 in enumerate(phase1_texts):
            tool_name = _parse_tool_call(p1)
            if tool_name and tool_name in cached_tool_outputs:
                tool_groups.setdefault(tool_name, []).append(idx)
            else:
                completions.append({
                    "text":            p1,
                    "phase1":          p1,
                    "phase2":          "",
                    "called_tools":    False,
                    "tool_name":       None,
                    "injected_output": None,
                    "_order":          idx,
                })

        # ---- Phase 2: follow-up for tool-calling completions ----
        for tool_name, indices in tool_groups.items():
            tool_result = cached_tool_outputs[tool_name]
            injected = _format_tool_output(tool_name, tool_result)

            f_sys, f_user = followup_user_template(
                question, choices, json.dumps(tool_result, indent=2)
            )
            messages_phase2 = self._build_qwen_messages(f_sys, f_user, audio_path)

            with torch.inference_mode():
                phase2_texts = self._generate_text(
                    messages_phase2,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=len(indices),
                )

            for sub_idx, orig_idx in enumerate(indices):
                p2 = phase2_texts[sub_idx] if sub_idx < len(phase2_texts) else ""
                p1 = phase1_texts[orig_idx]
                completions.append({
                    "text":            p1 + injected + p2,
                    "phase1":          p1,
                    "phase2":          p2,
                    "called_tools":    True,
                    "tool_name":       tool_name,
                    "injected_output": injected,
                    "_order":          orig_idx,
                })

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
        Compute mean per-token log-prob for completion_text given messages.

        Steps:
          1. Tokenize prompt (from chat template) and completion separately.
          2. Forward the full input_ids through the thinker.
          3. Extract log-probs only for completion token positions.

        Returns:
            (mean_log_prob, n_tokens)
        """
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError("qwen-omni-utils not installed.")

        tokenizer = self.processor.tokenizer

        # Build prompt-only token IDs via chat template
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Process multimodal info (audio features)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

        # Tokenize prompt only to get prompt length
        prompt_inputs = self.processor(
            text=prompt_text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        )
        prompt_inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                         for k, v in prompt_inputs.items()}

        S_prompt = prompt_inputs["input_ids"].shape[1]

        # Tokenize completion
        completion_ids = tokenizer(
            completion_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.model.device)  # (1, S_comp)

        # Concatenate prompt + completion input_ids
        full_ids = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)

        # Extend attention mask
        if "attention_mask" in prompt_inputs:
            comp_mask = torch.ones_like(completion_ids)
            full_mask = torch.cat([prompt_inputs["attention_mask"], comp_mask], dim=1)
        else:
            full_mask = torch.ones_like(full_ids)

        # Build kwargs for forward — pass audio features/embeddings from processor
        forward_kwargs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
        }

        # Copy over any audio/image/video embedding keys from processor output
        for k, v in prompt_inputs.items():
            if k not in ("input_ids", "attention_mask") and hasattr(v, 'to'):
                # Pad/extend if needed — for features that correspond to prompt only
                forward_kwargs[k] = v.to(self.model.device)

        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            # Use the thinker for forward pass (text generation backbone)
            thinker = self.model.thinker if hasattr(self.model, 'thinker') else self.model
            outputs = thinker(**forward_kwargs)

        logits = outputs.logits[0]  # (S_full, vocab)

        # Completion log-probs: logits[i] predicts token i+1
        comp_logits = logits[S_prompt - 1: S_prompt - 1 + completion_ids.shape[1]]
        comp_targets = completion_ids[0]

        log_probs = F.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=1, index=comp_targets.unsqueeze(1)
        ).squeeze(1)

        n_tokens = token_log_probs.shape[0]
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
        Log-probs under the reference (base) model. No gradients.

        Temporarily disables LoRA adapter layers, then re-enables them.
        """
        lora_base = getattr(self.model, "_lora_base_model", None)
        has_adapters = lora_base is not None and hasattr(lora_base, "disable_adapter_layers")
        try:
            if has_adapters:
                lora_base.disable_adapter_layers()
            return self._compute_logprobs_for_messages(messages, completion_text, no_grad=True)
        finally:
            if has_adapters:
                lora_base.enable_adapter_layers()
