"""
diversity_sampler.py — Diversity-aware GRPO completion sampler.

Problem
-------
When the cold policy model is untrained, it collapses to a single output
pattern (e.g. always emitting only "<think>" with no answer/tool tags).
All G completions then receive the same reward ⇒ std(r) ≈ 0 ⇒
advantages ≈ 0 ⇒ the entire batch is skipped with a "No non-zero advantage
terms" warning.  No gradient flows, training never starts.

Root cause: zero within-group reward *variance*, not wrong rewards per se.

Fix
---
Split the G rollout slots into two categories:

  G_free   — sampled freely from the policy (normal GRPO).
  G_forced — forced to execute a tool call via a SYNTHETIC phase-1 prefix,
             followed by a normally-sampled phase-2 completion.

For forced slots we:
  1. Construct a deterministic phase-1 text that calls a tool
     (round-robined across available cached tools).
  2. Inject the cached <tool_output> exactly as the normal two-phase path does.
  3. Run the policy model on the follow-up conversation to produce phase-2
     (the actual model-generated tokens that carry the gradient).

Effect on rewards
-----------------
  Free-path completions: currently get format≈0.30, correctness≈0.00 (total≈0.12).
  Forced-path completions: phase-2 produces <think>+<answer> tags, which scores:
    format  = 0.15+0.15 (<think>) + 0.20+0.20 (<answer>) + 0.15+0.15 (<tool>)
            = 1.0  ×  format_weight   →  ≈0.40
    correctness  = 0 or 1,  ×  correctness_weight   →  0 or ≈0.60
    total ∈ {0.40, ~1.0}

  Even if all answers are wrong, free=0.12 vs forced=0.40 gives non-zero
  variance ⇒ non-zero advantages ⇒ GRPO loss flows.

Schema
------
Every completion dict returned has the normal model_wrapper schema plus one
optional diagnostic key:

  {
    "text":             str,    # full text (phase1 + injected + phase2 or direct)
    "phase1":           str,    # first model output / synthetic prefix
    "phase2":           str,    # second model output (empty for direct answers)
    "called_tools":     bool,
    "tool_name":        str | None,
    "injected_output":  str | None,
    "forced_tool":      bool,   # True for forced-path completions (diagnostic)
  }

Usage
-----
    from grpo.diversity_sampler import diverse_generate_completions

    completions = diverse_generate_completions(
        policy,
        audio_path=..., system_prompt=..., user_prompt=...,
        cached_tool_outputs=..., followup_user_template=...,
        question=..., choices=...,
        G=16, temperature=0.9, top_p=0.95, max_new_tokens=2048,
        forced_tool_fraction=0.5,   # half forced, half free
    )

Configuration
-------------
Add to your YAML config:

    forced_tool_fraction: 0.5   # fraction of G slots that use forced-tool path
                                # 0.0 → pure free sampling (original behaviour)
                                # 1.0 → all forced (not recommended)
"""

import json
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .model_wrapper import DeSTA25GRPOModel, _format_tool_output
from .model_wrapper_qwen import QwenOmniGRPOModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic phase-1 construction
# ---------------------------------------------------------------------------

_FORCED_THINK = (
    "<think>\n"
    "I need to gather more information from the audio before I can answer. "
    "Let me use a tool to extract the relevant features.\n"
    "</think>"
)


def _make_forced_phase1(tool_name: str) -> str:
    """
    Build a deterministic phase-1 string that calls *tool_name*.

    The text is constructed — not sampled — so it is perfectly well-formed
    and will reliably trigger the two-phase path in the trainer.

    Example output::

        <think>
        I need to gather more information from the audio before I can answer.
        Let me use a tool to extract the relevant features.
        </think>
        <tool>
        [{"function": "chord_recognition", "parameters": {"audio_path": "<audio>"}}]
        </tool>
    """
    tool_call_json = json.dumps(
        [{"function": tool_name, "parameters": {"audio_path": "<audio>"}}],
        ensure_ascii=False,
    )
    return f"{_FORCED_THINK}\n<tool>\n{tool_call_json}\n</tool>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diverse_generate_completions(
    policy,   # DeSTA25GRPOModel or QwenOmniGRPOModel
    *,
    audio_path: str,
    system_prompt: str,
    user_prompt: str,
    cached_tool_outputs: Dict[str, Any],
    followup_user_template: Callable,   # (question, choices, tool_output_str) -> (sys, user)
    question: str,
    choices: List[str],
    G: int = 4,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
    forced_tool_fraction: float = 0.5,
    transcription: Optional[str] = None,  # cached ASR text — skips Whisper decoder/VAD
    precomputed_embed: Optional[str] = None, # precomputed
) -> List[Dict]:
    """
    Sample G completions with diversity enforcement.

    Parameters
    ----------
    policy : DeSTA25GRPOModel
        The policy model wrapper.
    audio_path, system_prompt, user_prompt :
        Same inputs forwarded to ``policy.generate_completions``.
    cached_tool_outputs : dict
        Mapping tool_name → cached result (from the dataset item).
        When empty, all G slots fall back to free sampling.
    followup_user_template : callable
        ``(question, choices, tool_output_str) -> (sys_prompt, user_prompt)``
    G : int
        Total completions per question.
    temperature, top_p, max_new_tokens :
        Sampling hyper-parameters.
    forced_tool_fraction : float
        Fraction of G slots dedicated to forced-tool completions.
        Set 0.0 to disable (identical to the original behaviour).
        Clamped to [0, 1]; at least 1 forced slot is created when > 0 and
        ``cached_tool_outputs`` is non-empty.

    Returns
    -------
    list of dict
        G completion dicts in randomised order (forced/free interleaved to
        avoid positional bias in advantage normalisation).
    """
    if not cached_tool_outputs or forced_tool_fraction <= 0.0:
        # Fallback: pure free sampling — identical to original behaviour
        return policy.generate_completions(
            audio_path=audio_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cached_tool_outputs=cached_tool_outputs,
            followup_system_prompt=None,
            followup_user_template=followup_user_template,
            question=question,
            choices=choices,
            G=G,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            transcription=transcription,
            precomputed_embed=precomputed_embed,
        )

    forced_tool_fraction = max(0.0, min(1.0, forced_tool_fraction))
    G_forced = max(1, round(G * forced_tool_fraction))
    G_free   = G - G_forced

    available_tools = list(cached_tool_outputs.keys())
    completions: List[Dict] = []

    # ------------------------------------------------------------------ #
    # 1. Free completions — normal policy sampling                        #
    # ------------------------------------------------------------------ #
    if G_free > 0:
        free_comps = policy.generate_completions(
            audio_path=audio_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            cached_tool_outputs=cached_tool_outputs,
            followup_system_prompt=None,
            followup_user_template=followup_user_template,
            question=question,
            choices=choices,
            G=G_free,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            transcription=transcription,
            precomputed_embed=precomputed_embed,
        )
        for c in free_comps:
            c.setdefault("forced_tool", False)
        completions.extend(free_comps)
        logger.debug(
            "[diversity] %d free completions sampled (tool calls: %d/%d)",
            G_free,
            sum(c["called_tools"] for c in free_comps),
            G_free,
        )

    # ------------------------------------------------------------------ #
    # 2. Forced-tool completions — synthetic phase-1 + real phase-2       #
    #    Batched per-tool: one generate() call per unique tool.            #
    # ------------------------------------------------------------------ #

    # Group forced slots by tool name
    forced_plan: Dict[str, List[int]] = {}  # tool_name -> [slot indices]
    for i in range(G_forced):
        tool_name = available_tools[i % len(available_tools)]
        forced_plan.setdefault(tool_name, []).append(i)

    forced_successes = 0
    for tool_name, slot_indices in forced_plan.items():
        phase1_text = _make_forced_phase1(tool_name)
        tool_result = cached_tool_outputs[tool_name]
        injected    = _format_tool_output(tool_name, tool_result)

        f_sys, f_user = followup_user_template(
            question, choices, json.dumps(tool_result, indent=2)
        )
        # Build phase-2 messages via the wrapper (model-agnostic)
        messages_phase2 = policy.build_messages_phase2(
            audio_path=audio_path,
            system_prompt=f_sys,
            user_prompt=f_user,
            transcription=transcription,
            precomputed_embed=precomputed_embed,
        )

        n_slots = len(slot_indices)
        try:
            with torch.inference_mode():
                if getattr(policy, 'model_family', 'desta') == 'qwen':
                    # Qwen: use the wrapper's generate helper
                    phase2_texts = policy._generate_text(
                        messages_phase2,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=n_slots,
                    )
                else:
                    # DeSTA: use model.generate() directly
                    out2 = policy.model.generate(
                        messages=messages_phase2,
                        do_sample=(temperature > 0.0),
                        temperature=temperature if temperature > 0.0 else 1.0,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=n_slots,
                    )
                    phase2_texts = out2.text if isinstance(out2.text, list) else [out2.text]

            for sub_idx in range(n_slots):
                phase2_text = phase2_texts[sub_idx] if sub_idx < len(phase2_texts) else ""
                full_text   = phase1_text + injected + phase2_text
                completions.append({
                    "text":            full_text,
                    "phase1":          phase1_text,
                    "phase2":          phase2_text,
                    "called_tools":    True,
                    "tool_name":       tool_name,
                    "injected_output": injected,
                    "forced_tool":     True,
                })
                forced_successes += 1
            logger.debug(
                "[diversity] forced tool=%r done (%d sequences) | preview: %r",
                tool_name, n_slots,
                phase2_texts[0][:80] if phase2_texts else "",
            )
        except Exception as exc:
            logger.warning(
                "[diversity] forced phase-2 failed for tool=%r (%s). "
                "Falling back to free completion for %d slot(s).",
                tool_name, exc, n_slots,
            )
            fallback = policy.generate_completions(
                audio_path=audio_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                cached_tool_outputs=cached_tool_outputs,
                followup_system_prompt=None,
                followup_user_template=followup_user_template,
                question=question,
                choices=choices,
                G=n_slots,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                transcription=transcription,
                precomputed_embed=precomputed_embed,
            )
            for c in fallback:
                c["forced_tool"] = False
            completions.extend(fallback)

    logger.debug(
        "[diversity] G=%d | free=%d | forced=%d/%d succeeded",
        G, G_free, forced_successes, G_forced,
    )

    # Shuffle so forced and free completions are interspersed — avoids any
    # positional bias in advantage normalisation (which normalises within the
    # full group, not by position).
    random.shuffle(completions)
    return completions
