
import re
import json
import math
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content of first occurrence of <tag>...</tag>.
    Falls back to capturing from <tag> until the next </ if closing tag is malformed."""
    # Try exact match first
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: capture from <tag> until next </
    fallback = rf"<{tag}>(.*?)</"
    m = re.search(fallback, text, re.DOTALL)
    return m.group(1).strip() if m else None

def format_reward(completion: str, called_tools: bool) -> float:
    """
    Tag-presence penalty — returns a value in [0, 1].

    Expected structure WITHOUT tool call:
      <think>...</think><answer>...</answer>

    Expected structure WITH tool call:
      <think>...</think><tool>...</tool>  (tool output injected)  <think>...</think><answer>...</answer>

    Penalties:
      <answer> entirely absent              → -0.5
      <answer> opened but </answer> missing → -0.2
      <think>  count mismatch               → -0.1 per missing occurrence
      <tool>   opened but </tool> missing   → -0.2
    """
    # penalty = 0.0

    # # --- <answer> block ---
    # if "<answer>" not in completion:
    #     penalty -= 0.5
    # elif "</answer>" not in completion:
    #     penalty -= 0.2

    # # --- <think> block ---
    # # With tool calls we expect two <think>...</think> pairs (before and after tool output),
    # # without tool calls we expect one <think>...</think> pair
    # has_tool_tag = "<tool>" in completion
    # expected_thinks = 2 if has_tool_tag else 1
    # actual_open = completion.count("<think>")
    # actual_close = completion.count("</think>")
    # missing_open = max(0, expected_thinks - actual_open)
    # missing_close = max(0, expected_thinks - actual_close)
    # penalty -= 0.1 * (missing_open + missing_close)

    # # --- <tool> block ---
    # if '<tool>' in completion and '</tool>' not in completion:
    #     penalty -= 0.2

    # return round(penalty, 4)

    ### NOTE: Below code will casuse reward hacking avoid that ###
    # reward = 0.0
    # pathA_total_tags = 4 # <think>, </think>, <answer>, </answer>
    # pathB_total_tags = 8 # <think>, </think>, <tool>, </tool>, <think>, </think>, <answer>, </answer>

    # reward = completion.count("<think>") + completion.count("</think>") + completion.count("<answer>") + completion.count("</answer>") + completion.count("<tool>") + completion.count("</tool>")
    # if called_tools:
    #     reward = reward / pathB_total_tags
    # else:
    #     reward = reward / pathA_total_tags
    # return round(reward, 4)
    ### ##################################

    actual_tags = re.findall(r'</?(?:think|tool|answer)>', completion)

    if called_tools:
        expected_tags = ['<think>', '</think>', '<tool>', '</tool>', '<think>', '</think>', '<answer>', '</answer>']
    else:
        expected_tags = ['<think>', '</think>', '<answer>', '</answer>']

    # Calculate sequence-based reward
    score = 0.0
    for i, tag in enumerate(actual_tags):
        if i < len(expected_tags) and tag == expected_tags[i]:
            score += 1.0
        else:
            score -= 1.0

    reward = max(0.0, min(1.0, score / len(expected_tags)))
    return round(reward, 4)



def extract_answer(completion: str) -> Optional[str]:
    """
    Extract the answer from <answer>...</answer>.

    Handles both plain text and JSON-wrapped formats.
    """
    content = _extract_tag(completion, "answer")
    return content if content else None

def _normalize(text: str) -> str:
    """Lowercase + strip for comparison."""
    return text.lower().strip()


def _tokenize(text: str) -> set:
    """Extract word tokens from text (lowercase)."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def _token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score (SQuAD-style) for soft partial credit."""
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _soft_correctness(completion_text: str, gold: str, choices: List[str]) -> float:
    """
    Exact-match correctness only.

    Returns:
      1.0   extracted answer exactly matches gold (case-insensitive, stripped)
      0.0   otherwise
    """
    predicted = extract_answer(completion_text)

    if predicted is not None:
        pred_norm = _normalize(predicted)
        gold_norm = _normalize(gold)

        if pred_norm == gold_norm:
            return 1.0

    return 0.0


def _option_mention_score(text: str, gold: str, choices: List[str]) -> float:
    """
    Partial credit when the completion mentions answer options in the body
    (even without proper <answer> tags).  Differentiates completions that
    are "thinking about the right answer" from those that are pure gibberish.

    Returns:
      0.5  if gold answer appears in text (case-insensitive)
      0.1  if any other option appears
      0.0  otherwise
    """
    text_lower = text.lower()
    if gold.lower() in text_lower:
        return 0.5
    for opt in choices:
        if opt.lower() in text_lower:
            return 0.1
    return 0.0


def _count_tool_calls(completion_text: str) -> int:
    """
    Count tool calls from JSON arrays inside <tool>...</tool> blocks.

    Expected format is a JSON array per <tool> block, e.g.:
      <tool>[{"function": "x", "parameters": {...}}]</tool>

    Fallback behavior for robustness:
      - malformed/non-JSON payload inside a present <tool> block counts as 1 call
      - non-list JSON payload inside a present <tool> block counts as 1 call
    """
    tool_blocks = re.findall(r"<tool>(.*?)</tool>", completion_text, re.DOTALL)
    if not tool_blocks:
        return 0

    total_calls = 0
    for payload in tool_blocks:
        payload = payload.strip()
        if not payload:
            continue

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            total_calls += 1
            continue

        if isinstance(parsed, list):
            total_calls += len(parsed)
        else:
            total_calls += 1

    return total_calls


# ──────────────────────────────────────────────────────────────────────────────
# ToolRL-style two-term reward
# ──────────────────────────────────────────────────────────────────────────────
#
# Reference: Qian et al., "ToolRL: Reward is All Tool Learning Needs",
# arXiv 2504.13958 (2025).
#
# Following the paper's recipe and three core takeaways:
#   1. Length rewards do not help and can hurt.
#   2. Smooth dynamic reward scale (format ↓ / correctness ↑ over training)
#      outperforms abrupt phase switches.
#   3. Fine-grained correctness > coarse exact-match.
#
# We use exactly two terms:
#
#     reward = w_correct * c_score + w_format * f_score
#
# where
#   c_score ∈ [0, 1]   token-F1 soft correctness on the extracted <answer>
#   f_score ∈ [0, 1]   tag-sequence match (path-aware: A vs. B)
#
# By design we do NOT add tool-usage bonuses, per-call taxes, group
# difficulty terms, exploration baselines, counterfactual helpers, or any
# other shaping. The model learns when to use tools via the correctness
# signal alone — tools that genuinely help raise c_score, tools that hurt
# lower it, and GRPO's group-relative advantage does the rest. Three prior
# iterations of shaped rewards on this setup collapsed (B→0) or spammed
# (B→G), matching documented failure modes in Search-R1 (arXiv 2503.09516)
# §5.1.
#
# Dynamic scaling (ToolRL Takeaway 2): if a trainer_state is passed via
# kwargs we linearly interpolate the format/correctness weights from
# (w_format=high, w_correct=low) at step 0 toward (w_format=low,
# w_correct=high) at the final step. When no trainer_state is available
# (e.g. unit tests, smoke runs) we fall back to the static endpoint values.
# This is a no-op if `dynamic_scaling` is set to False.


_TOOL_REWARD_PARAMS = {
    # Static endpoint weights (used when dynamic_scaling=False or no trainer state).
    "w_correct":        0.90,
    "w_format":         0.10,

    # Dynamic scaling endpoints (ToolRL Takeaway 2).
    # Linear interpolation from start → end as training progresses.
    "dynamic_scaling":  True,
    "w_correct_start":  0.50,   # start: format/correctness balanced
    "w_format_start":   0.50,
    "w_correct_end":    0.90,   # end: correctness dominates
    "w_format_end":     0.10,

    # Threshold used only for logging / diagnostics.
    "correct_threshold": 0.7,

    # ── Tool-quality robustness shaping ───────────────────────────────────
    # Applied ONLY to completions flagged is_negative_tool=True. The signal
    # encourages the policy to keep its own audio-grounded reasoning when the
    # tool output is unreliable, rather than blindly copying it.
    #   • Resisted a bad tool (answer correct)  → +robustness_bonus
    #   • Followed a bad tool to a wrong answer → −gullibility_penalty
    # Magnitudes are small relative to the correctness term so they shape
    # behavior without dominating it.
    "robustness_bonus":     0.20,
    "gullibility_penalty":  0.10,
}

# Keys that are accepted but ignored (legacy YAML compatibility). We log a
# DEBUG message so old configs with shaping params don't crash training.
_OBSOLETE_TOOL_REWARD_KEYS = {
    "tool_help_bonus",
    "efficiency_bonus",
    "efficiency_extra",
    "missed_tool_cost",
    "failed_tool_cost",
    "tool_call_cost",
    "spam_penalty",
    "exploration_baseline",
    "hard_group_penalty",
    "hard_group_threshold",
}


def set_tool_reward_params(**kwargs):
    """Override tool-reward parameters from config / training script.

    Unknown / obsolete keys (from legacy shaped-reward configs) are silently
    dropped with a DEBUG log line so existing YAML files keep working.
    """
    for k, v in kwargs.items():
        if k in _TOOL_REWARD_PARAMS:
            # Preserve bool type for dynamic_scaling, cast numerics to float.
            if isinstance(_TOOL_REWARD_PARAMS[k], bool):
                _TOOL_REWARD_PARAMS[k] = bool(v)
            else:
                _TOOL_REWARD_PARAMS[k] = float(v)
        elif k in _OBSOLETE_TOOL_REWARD_KEYS:
            logger.debug(f"Ignoring obsolete tool reward param '{k}' (ToolRL two-term reward).")
        else:
            logger.warning(f"Unknown tool reward param '{k}' — ignoring")
    logger.info(f"Tool reward params updated: {_TOOL_REWARD_PARAMS}")


def _current_weights(trainer_state) -> tuple:
    """Return (w_correct, w_format) for the current training step.

    Linearly interpolates from (_start) to (_end) when dynamic_scaling is on
    and trainer_state exposes global_step / max_steps. Falls back to the
    static (w_correct, w_format) otherwise.
    """
    P = _TOOL_REWARD_PARAMS
    if not P.get("dynamic_scaling", False):
        return P["w_correct"], P["w_format"]

    if trainer_state is None:
        return P["w_correct"], P["w_format"]

    cur = getattr(trainer_state, "global_step", None)
    tot = getattr(trainer_state, "max_steps", None)
    if cur is None or tot is None or tot <= 0:
        return P["w_correct"], P["w_format"]

    p = min(1.0, max(0.0, float(cur) / float(tot)))
    w_correct = P["w_correct_start"] + p * (P["w_correct_end"] - P["w_correct_start"])
    w_format  = P["w_format_start"]  + p * (P["w_format_end"]  - P["w_format_start"])
    return w_correct, w_format


def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
    """ToolRL-style two-term reward: w_correct * c_score + w_format * f_score.

    No tool-usage bonuses, no per-call taxes, no exploration baselines, no
    group statistics. The policy learns *when* to use tools via the
    correctness signal alone; GRPO group-relative advantage does the rest.
    """
    P = _TOOL_REWARD_PARAMS
    threshold = P["correct_threshold"]
    robust_bonus = float(P.get("robustness_bonus", 0.0))
    gullible_pen = float(P.get("gullibility_penalty", 0.0))

    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")
    trainer_state = kwargs.get("trainer_state")
    # is_negative_tool is plumbed in from _generate via TRL's extra_fields.
    # Defaults to False per-completion when the feature is disabled.
    is_neg_list = kwargs.get("is_negative_tool") or [False] * len(completions)

    w_correct, w_format = _current_weights(trainer_state)

    n = len(completions)
    rewards: List[float] = []
    n_a = 0
    n_b = 0
    sum_c = 0.0
    sum_f = 0.0
    n_correct = 0

    # Robustness telemetry (negative-tool subgroup).
    n_neg = 0
    n_neg_correct = 0      # resisted the bad tool
    n_neg_wrong = 0        # followed the bad tool (or otherwise wrong)
    n_clean_b = 0          # Path-B completions with a clean tool output
    n_clean_b_correct = 0

    for i in range(n):
        completion = completions[i]
        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
            text = completion[-1].get("content", "")
        else:
            text = str(completion)

        gold = gold_answers[i]
        choices = choices_list[i]
        is_neg = bool(is_neg_list[i]) if i < len(is_neg_list) else False

        used_tool = _count_tool_calls(text) > 0
        c_score = _soft_correctness(text, gold, choices)
        f_score = format_reward(text, used_tool)

        reward = w_correct * c_score + w_format * f_score

        # Tool-quality robustness shaping — only on poisoned tool outputs.
        # The model can only earn this signal on Path B (a tool was actually
        # used and we swapped its output); is_neg is False on Path A.
        if is_neg:
            if c_score >= threshold:
                reward += robust_bonus
            else:
                reward -= gullible_pen

        rewards.append(float(reward))

        if used_tool:
            n_b += 1
            if is_neg:
                n_neg += 1
                if c_score >= threshold:
                    n_neg_correct += 1
                else:
                    n_neg_wrong += 1
            else:
                n_clean_b += 1
                if c_score >= threshold:
                    n_clean_b_correct += 1
        else:
            n_a += 1
        sum_c += c_score
        sum_f += f_score
        if c_score >= threshold:
            n_correct += 1

    avg_c = sum_c / n if n else 0.0
    avg_f = sum_f / n if n else 0.0
    acc = n_correct / n if n else 0.0
    # Robustness summary — printed only when negatives were injected.
    if n_neg > 0:
        robust_rate = n_neg_correct / n_neg
        clean_rate = (n_clean_b_correct / n_clean_b) if n_clean_b > 0 else float("nan")
        gap = clean_rate - robust_rate if n_clean_b > 0 else float("nan")
        neg_str = (
            f" | neg={n_neg} robust={robust_rate:.2f} "
            f"clean_b={clean_rate:.2f} gap={gap:.2f}"
        )
    else:
        neg_str = ""
    print(
        f"(A={n_a}, B={n_b}) acc={acc:.2f} c={avg_c:.2f} f={avg_f:.2f} "
        f"| w_correct={w_correct:.2f} w_format={w_format:.2f}{neg_str}"
    )
    return rewards
