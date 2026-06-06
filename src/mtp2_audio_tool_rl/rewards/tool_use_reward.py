"""Deterministic debug reward breakdown for tool-use GRPO experiments."""

import re
from typing import Any, Dict, Mapping, Optional


DEFAULT_WEIGHTS = {
    "answer_weight": 1.0,
    "tool_validity_weight": 0.2,
    "tool_helpfulness_weight": 0.3,
    "unnecessary_tool_penalty": -0.2,
    "invalid_tool_penalty": -0.4,
}


def normalize_answer(text: Any) -> str:
    """Normalize answer text for exact-match debug scoring."""

    if text is None:
        return ""
    normalized = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized.strip()


def exact_match_reward(prediction: Any, gold: Any) -> float:
    """Return 1.0 when normalized prediction and gold are identical."""

    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def _weights(overrides: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if overrides:
        for key, value in overrides.items():
            if key in weights:
                weights[key] = float(value)
    return weights


def compute_tool_use_reward(
    prediction: Any = None,
    gold_answer: Any = None,
    *,
    final_answer: Any = None,
    tool_call_valid: Optional[bool] = None,
    tool_call_present: bool = False,
    tool_was_needed: bool = False,
    tool_result_used: Optional[bool] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Compute a transparent reward breakdown for smoke experiments."""

    active_weights = _weights(weights)
    answer_text = final_answer if final_answer is not None else prediction

    answer_reward = exact_match_reward(answer_text, gold_answer) * active_weights["answer_weight"]
    tool_validity_reward = 0.0
    tool_helpfulness_reward = 0.0
    unnecessary_tool_penalty = 0.0
    invalid_tool_penalty = 0.0

    if tool_call_present:
        if tool_call_valid is True:
            tool_validity_reward = active_weights["tool_validity_weight"]
        elif tool_call_valid is False:
            invalid_tool_penalty = active_weights["invalid_tool_penalty"]

        if not tool_was_needed:
            unnecessary_tool_penalty = active_weights["unnecessary_tool_penalty"]

    if tool_was_needed and tool_result_used is True:
        tool_helpfulness_reward = active_weights["tool_helpfulness_weight"]

    components = {
        "answer_reward": answer_reward,
        "tool_validity_reward": tool_validity_reward,
        "tool_helpfulness_reward": tool_helpfulness_reward,
        "unnecessary_tool_penalty": unnecessary_tool_penalty,
        "invalid_tool_penalty": invalid_tool_penalty,
    }
    total_reward = sum(components.values())
    return {"total_reward": total_reward, **components}
