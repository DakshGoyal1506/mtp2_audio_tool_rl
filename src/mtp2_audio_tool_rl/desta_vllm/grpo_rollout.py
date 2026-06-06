"""
grpo_rollout.py — GRPO rollout adapter using DeSTAVLLMEngine.

Provides a generate_completions() function with the same signature and
return format as DeSTA25GRPOModel.generate_completions() from
grpo_single_phase/model_wrapper.py, but backed by vLLM for ~5-10x faster
sampling during GRPO training.

Integration point:
    In AudioTRLGRPOTrainer._generate(), replace:
        comps = desta_wrapper.generate_completions(...)
    with:
        from desta_vllm.grpo_rollout import VLLMRolloutEngine
        rollout = VLLMRolloutEngine(engine)
        comps = rollout.generate_completions(...)

The return format is identical: list of dicts with keys
  {text, phase1, phase2, called_tools, tool_name, injected_output, is_negative_tool}
"""

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

from desta_vllm.engine import (
    DeSTAVLLMEngine,
    GenerationResult,
    _format_tool_output,
    _parse_tool_call,
)

logger = logging.getLogger(__name__)


class VLLMRolloutEngine:
    """
    Drop-in replacement for DeSTA25GRPOModel's generation logic, backed by vLLM.

    This class wraps DeSTAVLLMEngine and provides generate_completions()
    with the same signature as grpo_single_phase/model_wrapper.py, so it
    can be swapped into AudioTRLGRPOTrainer._generate() with minimal changes.
    """

    def __init__(self, engine: DeSTAVLLMEngine):
        self.engine = engine

    def generate_completions(
        self,
        embed_path: str,
        transcription: Optional[str],
        system_prompt: str,
        user_prompt: str,
        cached_tool_outputs: Dict[str, Any],
        question: str,
        choices: List[str],
        G: int = 4,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        negative_tool_pool: Optional[List[Dict[str, Any]]] = None,
        negative_tool_ratio: float = 0.0,
        **kwargs,
    ) -> List[Dict]:
        """
        Generate G completions with optional two-phase tool continuation.

        Returns list of dicts matching DeSTA25GRPOModel.generate_completions():
          - text: full generated text
          - phase1: Phase 1 output
          - phase2: Phase 2 output (empty if no tool call)
          - called_tools: bool
          - tool_name: str or None
          - injected_output: str or None
          - is_negative_tool: bool
        """
        # Phase 1: generate G completions
        phase1_results = self.engine.generate(
            embed_path=embed_path,
            transcription=transcription,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=G,
        )

        # Classify: tool call or direct answer
        completions = []
        needs_phase2 = []  # (index, tool_name)

        for idx, res in enumerate(phase1_results):
            tool_name = _parse_tool_call(res.text)
            if tool_name and tool_name in cached_tool_outputs:
                needs_phase2.append((idx, tool_name))
            else:
                completions.append({
                    "text": res.text,
                    "phase1": res.text,
                    "phase2": "",
                    "called_tools": False,
                    "tool_name": None,
                    "injected_output": None,
                    "is_negative_tool": False,
                    "_order": idx,
                })

        # Deterministic negative-tool balancing
        neg_positions = set()
        if (
            negative_tool_ratio > 0.0
            and negative_tool_pool
            and len(needs_phase2) > 0
        ):
            n_neg = int(len(needs_phase2) * negative_tool_ratio)
            if n_neg > 0:
                neg_positions = set(random.sample(range(len(needs_phase2)), n_neg))

        # Phase 2: batch continuations via vLLM
        if needs_phase2:
            from desta_vllm.embed_utils import build_prompt_embeds_continuation

            p2_prompts = []
            p2_meta = []

            for k, (idx, tool_name) in enumerate(needs_phase2):
                p1_text = phase1_results[idx].text

                # Negative tool injection
                is_negative = False
                if k in neg_positions:
                    candidates = [
                        other[tool_name]
                        for other in (negative_tool_pool or [])
                        if tool_name in other and other is not cached_tool_outputs
                    ]
                    if candidates:
                        tool_result = random.choice(candidates)
                        is_negative = True

                if not is_negative:
                    tool_result = cached_tool_outputs[tool_name]

                injected = _format_tool_output(tool_name, tool_result)

                embeds, _ = build_prompt_embeds_continuation(
                    embed_path=embed_path,
                    transcription=transcription,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    assistant_phase1=p1_text,
                    injected_tool_output=injected,
                    tokenizer=self.engine.tokenizer,
                    embed_layer=self.engine.embed_layer,
                    embed_cache=self.engine.embed_cache,
                    dtype=self.engine._dtype,
                    device="cpu",
                )
                p2_prompts.append({"prompt_embeds": embeds})
                p2_meta.append((idx, tool_name, injected, p1_text, is_negative))

            p2_sampling = self.engine.SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=1,
                stop_token_ids=self.engine._stop_token_ids,
            )

            p2_outputs = self.engine.llm.generate(
                prompts=p2_prompts,
                sampling_params=p2_sampling,
                lora_request=self.engine._lora_request,
            )

            for j, output in enumerate(p2_outputs):
                idx, tool_name, injected, p1_text, is_negative = p2_meta[j]
                p2_text = output.outputs[0].text

                completions.append({
                    "text": p1_text + injected + p2_text,
                    "phase1": p1_text,
                    "phase2": p2_text,
                    "called_tools": True,
                    "tool_name": tool_name,
                    "injected_output": injected,
                    "is_negative_tool": is_negative,
                    "_order": idx,
                })

        # Restore original order
        completions.sort(key=lambda c: c.pop("_order", 0))
        return completions
