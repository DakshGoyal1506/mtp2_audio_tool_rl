"""
DeSTA25GRPOModel — wraps DeSTA25AudioModel for GRPO training.

Single-episode architecture:
  The model generates a CONTINUOUS sequence. If it calls a tool, cached tool
  output is injected and generation CONTINUES from the same context — the model
  sees its own Phase 1 reasoning when producing the Phase 2 answer.

  Episode format:
    <think>reasoning</think>
    <tool>[{"function": "...", "parameters": {...}}]</tool>
    <tool_output>cached result</tool_output>          ← injected, mask=0
    <think>interpret tool result</think>
    <answer>exact option text</answer>

  OR (direct answer, no tool):
    <think>reasoning</think>
    <answer>exact option text</answer>

  Log-probs span ONLY model-generated tokens; injected <tool_output> is masked.
"""

import json
import logging
import random
import re
from typing import Dict, List, Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_tool_call(text: str) -> Optional[str]:
    """
    Return the tool function name if a well-formed <tool>...</tool> block is found.
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
# DeSTA25GRPOModel — single-episode generation
# ---------------------------------------------------------------------------

class DeSTA25GRPOModel:
    """
    Thin wrapper around DeSTA25AudioModel for single-episode GRPO generation.

    Key difference from the two-phase architecture:
      Phase 2 is generated as a CONTINUATION of the same conversation, not a
      separate prompt. The model sees its own Phase 1 output + injected tool
      output when generating the Phase 2 answer.
    """

    model_family = "desta"

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def get_llm_backbone(self):
        """Return the core LLM module (for gradient checkpointing control)."""
        return self.model.llm_model

    # ------------------------------------------------------------------
    # Generation — single continuous episode
    # ------------------------------------------------------------------

    def generate_completions(
        self,
        audio_path: str,
        system_prompt: str,
        user_prompt: str,
        cached_tool_outputs: Dict[str, Any],
        question: str,
        choices: List[str],
        G: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        transcription: Optional[str] = None,
        precomputed_embed: Optional[str] = None,
        negative_tool_pool: Optional[List[Dict[str, Any]]] = None,
        negative_tool_ratio: float = 0.0,
    ) -> List[Dict]:
        """
        Sample G completions for one question using single-episode generation.

        Flow per completion:
          1. Generate Phase 1 with batch G (until model stops)
          2. For each completion that called a tool with cached output:
             - Build continuation: [system, user, assistant: phase1 + tool_output]
             - Generate Phase 2 as continuation (model sees its own Phase 1)
          3. Direct-answer completions are returned as-is

        Returns list of dicts with keys:
          text, phase1, phase2, called_tools, tool_name, injected_output
        """
        completions = []

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<|AUDIO|>\n{user_prompt}",
                "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
            },
        ]

        # ---- Phase 1: batch-generate G completions ----
        with torch.inference_mode():
            out = self.model.generate(
                messages=messages,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=G,
            )
        phase1_texts = out.text if isinstance(out.text, list) else [out.text]

        # ---- Classify: tool call or direct answer? ----
        needs_phase2 = []  # (index, tool_name) pairs
        for idx, p1 in enumerate(phase1_texts):
            tool_name = _parse_tool_call(p1)
            if tool_name and tool_name in cached_tool_outputs:
                needs_phase2.append((idx, tool_name))
            else:
                completions.append({
                    "text":            p1,
                    "phase1":          p1,
                    "phase2":          "",
                    "called_tools":    False,
                    "tool_name":       None,
                    "injected_output": None,
                    "is_negative_tool": False,
                    "_order":          idx,
                })

        # ---- Deterministic balancing of negative-tool injections ----
        # Previously this was a per-completion Bernoulli draw, which on small
        # `needs_phase2` groups led to either 0/all-negative groups and noisy
        # advantage estimates. We now poison EXACTLY floor(N * ratio) of the
        # tool-calling completions per (prompt, GPU) — same expected count but
        # zero between-group variance in the negative count.
        if (negative_tool_ratio > 0.0
                and negative_tool_pool
                and len(needs_phase2) > 0):
            n_neg = int(len(needs_phase2) * negative_tool_ratio)
            if n_neg > 0:
                neg_positions = set(random.sample(range(len(needs_phase2)), n_neg))
            else:
                neg_positions = set()
        else:
            neg_positions = set()

        # ---- Phase 2: CONTINUE generation with same context ----
        # Build full conversation: system + user + assistant(phase1+tool_output)
        # Then generate continuation — model sees its own Phase 1 reasoning.
        for k, (idx, tool_name) in enumerate(needs_phase2):
            p1 = phase1_texts[idx]

            # Negative tool injection: deterministic count (see above). When
            # this slot is chosen, swap with the SAME tool's output from a
            # different audio sample so the model must cross-check the tool
            # against its own audio perception rather than blindly trusting.
            is_negative = False
            if k in neg_positions:
                candidates = [
                    other[tool_name]
                    for other in negative_tool_pool
                    if tool_name in other and other is not cached_tool_outputs
                ]
                if candidates:
                    tool_result = random.choice(candidates)
                    is_negative = True

            if not is_negative:
                tool_result = cached_tool_outputs[tool_name]

            injected = _format_tool_output(tool_name, tool_result)

            # Split into proper roles:
            # - assistant: the model's Phase 1 output (think + tool call)
            # - user: the injected tool output (external system response)
            # This way the model learns "I called a tool → system gave me results → I interpret"
            continuation_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{user_prompt}",
                    "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
                },
                {
                    "role": "assistant",
                    "content": p1,
                },
                {
                    "role": "user",
                    "content": injected,
                },
            ]

            with torch.inference_mode():
                out2 = self.model.generate(
                    messages=continuation_messages,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                )
            p2 = out2.text if isinstance(out2.text, str) else out2.text[0]

            completions.append({
                "text":            p1 + injected + p2,
                "phase1":          p1,
                "phase2":          p2,
                "called_tools":    True,
                "tool_name":       tool_name,
                "injected_output": injected,
                "is_negative_tool": is_negative,
                "_order":          idx,
            })

        # Restore original generation order
        completions.sort(key=lambda c: c.pop("_order", 0))
        return completions
