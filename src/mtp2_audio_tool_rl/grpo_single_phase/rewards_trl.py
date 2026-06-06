
import re
import json
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
    penalty = 0.0

    # --- <answer> block ---
    if "<answer>" not in completion:
        penalty -= 0.5
    elif "</answer>" not in completion:
        penalty -= 0.2

    # --- <think> block ---
    # With tool calls we expect two <think>...</think> pairs (before and after tool output),
    # without tool calls we expect one <think>...</think> pair
    has_tool_tag = "<tool>" in completion
    expected_thinks = 2 if has_tool_tag else 1
    actual_open = completion.count("<think>")
    actual_close = completion.count("</think>")
    missing_open = max(0, expected_thinks - actual_open)
    missing_close = max(0, expected_thinks - actual_close)
    penalty -= 0.1 * (missing_open + missing_close)

    # --- <tool> block ---
    if '<tool>' in completion and '</tool>' not in completion:
        penalty -= 0.2

    return round(penalty, 4)


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
    Soft correctness with token F1 for partial credit.

    Returns:
      1.0   exact match with gold
      0.7   extracted answer has high token F1 (≥0.5) with gold
      0.3   extracted answer matches a wrong choice exactly (structured but wrong)
      0.15  gold answer mentioned anywhere in completion body
      0.05  any choice mentioned in completion body
      0.0   no answer extracted and no relevant mention
    """
    predicted = extract_answer(completion_text)

    if predicted is not None:
        pred_norm = _normalize(predicted)
        gold_norm = _normalize(gold)

        # Exact match
        if pred_norm == gold_norm:
            return 1.0

        # High token F1 with gold (near-miss partial credit)
        f1 = _token_f1(predicted, gold)
        if f1 >= 0.5:
            return 0.7

    # No valid <answer> tag or no match — check body mentions
    text_lower = completion_text.lower()
    if gold.lower() in text_lower:
        return 0.15

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


_REWARD_WEIGHTS = {
    "format": 0.10,
    "correctness": 0.60,
    "option_mention": 0.15,
    "tool_bonus": 0.15,
}

def set_reward_weights(fw: float, cw: float, mw: float, tw: float):
    """Override reward weights from config."""
    _REWARD_WEIGHTS["format"] = fw
    _REWARD_WEIGHTS["correctness"] = cw
    _REWARD_WEIGHTS["option_mention"] = mw
    _REWARD_WEIGHTS["tool_bonus"] = tw
    logger.info(f"Reward weights set: format={fw}, correctness={cw}, mention={mw}, tool={tw}")


def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
    """
    TRL-compatible reward function with soft correctness scoring and anti-spam tool logic.
    """
    rewards = []

    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")

    fw = _REWARD_WEIGHTS["format"]
    cw = _REWARD_WEIGHTS["correctness"]
    mw = _REWARD_WEIGHTS["option_mention"]
    tw = _REWARD_WEIGHTS["tool_bonus"]

    for i in range(len(completions)):
        completion = completions[i]
        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
            completion_text = completion[-1].get("content", "")
        else:
            completion_text = str(completion)

        gold = gold_answers[i]
        choices = choices_list[i]

        # tool calls from JSON arrays inside <tool>...</tool>
        tool_count = _count_tool_calls(completion_text)
        called_tools = tool_count > 0

        tag_penalty = format_reward(completion_text, called_tools)
        f_score = max(0.0, 1.0 + tag_penalty)

        c_score = _soft_correctness(completion_text, gold, choices)

        # option-mention score (0, 0.1, or 0.5)
        m_score = _option_mention_score(completion_text, gold, choices) if c_score < 0.5 else 1.0

        # --- tool usage score ---
        tool_score = 0.0

        if c_score >= 0.7:
            # Got the answer right
            if not called_tools:
                tool_score = 1.0    # correct without tool (highly efficient)
            else:
                # correct with tool
                if tool_count == 1:
                    tool_score = 0.8
                else:
                    # spamming tools before getting it right
                    tool_score = max(0.0, 0.8 - (0.2 * (tool_count - 1)))
        else:
            # Got the answer WRONG
            if called_tools:
                # Penalty for calling tools but failing. Capped at -1.0 to
                # prevent extreme negative advantages
                tool_score = max(-1.0, -0.5 * tool_count)
            else:
                # Wrong answer, no tools used. Normal baseline.
                tool_score = 0.0

        total = (fw * f_score) + (cw * c_score) + (mw * m_score) + (tw * tool_score)
        # Clamp reward to [0, 1] to bound advantage magnitude and prevent extreme policy gradients that cause KL/entropy divergence.
        total = max(0.0, min(1.0, total))
        rewards.append(float(total))

    return rewards

# def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
#     """
#     TRL-compatible reward function with soft correctness scoring.

#     Reward components (weights set via set_reward_weights or defaults):
#       format   — tag structure quality
#       correct  — soft correctness (0.0 to 1.0 with partial credit)
#       mention  — partial credit for mentioning options (cold-start helper)
#       tool     — bonus for well-formed tool calls
#     """
#     rewards = []

#     gold_answers = kwargs.get("gold_answer")
#     choices_list = kwargs.get("choices")

#     fw = _REWARD_WEIGHTS["format"]
#     cw = _REWARD_WEIGHTS["correctness"]
#     mw = _REWARD_WEIGHTS["option_mention"]
#     tw = _REWARD_WEIGHTS["tool_bonus"]

#     for i in range(len(completions)):
#         completion = completions[i]
#         if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
#             completion_text = completion[-1].get("content", "")
#         else:
#             completion_text = str(completion)

#         gold = gold_answers[i]
#         choices = choices_list[i]

#         called_tools = "<tool>" in completion_text

#         # --- format score (0..1) ---
#         tag_penalty = format_reward(completion_text, called_tools)
#         f_score = max(0.0, 1.0 + tag_penalty)

#         # --- soft correctness score (0.0 to 1.0 with partial credit) ---
#         c_score = _soft_correctness(completion_text, gold, choices)

#         # --- option-mention score (0, 0.1, or 0.5) ---
#         # Only matters when correctness < 0.5 (cold-start differentiation)
#         m_score = _option_mention_score(completion_text, gold, choices) if c_score < 0.5 else 1.0

#         # --- tool usage score ---
#         # Incentive: correct without tool > correct with tool > incorrect with tool
#         # This prevents the model from always calling tools when it can answer by listening.
#         tool_score = 0.0
#         if c_score >= 0.7:
#             # Got the answer right
#             if not called_tools:
#                 tool_score = 1.0    # best: correct without tool (efficient)
#             else:
#                 tool_score = 0.5    # good: correct with tool (but tool wasn't needed)
#         else:
#             # Got it wrong
#             if called_tools and tag_penalty > -0.2:
#                 tool_score = 0.3    # tried with tool, still wrong but structured attempt

#         total = (fw * f_score) + (cw * c_score) + (mw * m_score) + (tw * tool_score)
#         rewards.append(float(total))

#     return rewards
