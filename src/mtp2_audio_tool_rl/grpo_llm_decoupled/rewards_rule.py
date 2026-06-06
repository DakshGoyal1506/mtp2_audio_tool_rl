
import re
import json
import math
import logging
import os
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
    # text_lower = completion_text.lower()
    # if gold.lower() in text_lower:
    #     return 0.15

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
    "correctness": 0.90,
    "option_mention": 0.15, # ignore
    "tool_bonus": 0.10,
}

# Only trajectories above this correctness threshold are eligible for tool
# efficiency learning in the decoupled channel.
_TOOL_CONDITION_CORRECTNESS = 0.7

def set_reward_weights(fw: float, cw: float, mw: float, tw: float):
    """Override reward weights from config."""
    _REWARD_WEIGHTS["format"] = fw
    _REWARD_WEIGHTS["correctness"] = cw
    _REWARD_WEIGHTS["option_mention"] = mw
    _REWARD_WEIGHTS["tool_bonus"] = tw
    logger.info(f"Reward weights set: format={fw}, correctness={cw}, mention={mw}, tool={tw}")


def _score_single_completion(completion, gold: str, choices: List[str]):
    """Compute reusable heuristic components for one completion."""
    if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
        completion_text = completion[-1].get("content", "")
    else:
        completion_text = str(completion)

    tool_count = _count_tool_calls(completion_text)
    called_tools = tool_count > 0
    tag_penalty = format_reward(completion_text, called_tools)
    f_score = max(0.0, 1.0 + tag_penalty)
    c_score = _soft_correctness(completion_text, gold, choices)
    return f_score, c_score, tool_count, called_tools


def trl_reward_acc_function(prompts, completions, **kwargs) -> List[float]:
    """Accuracy channel reward: format + correctness, independent of tool efficiency."""
    rewards = []
    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")
    path_b_count = 0

    fw = _REWARD_WEIGHTS["format"]
    cw = _REWARD_WEIGHTS["correctness"]

    for i in range(len(completions)):
        gold = gold_answers[i]
        choices = choices_list[i]
        f_score, c_score, _, called_tools = _score_single_completion(completions[i], gold, choices)
        if called_tools:
            path_b_count += 1
        rewards.append(float((fw * f_score) + (cw * c_score)))

    rank = os.environ.get("LOCAL_RANK", "0")
    logger.info(
        f"[Rank {rank}] Heuristic batch: {len(completions)} completions "
        f"(A={len(completions) - path_b_count}, B={path_b_count})"
    )

    return rewards


def trl_reward_tool_function(prompts, completions, **kwargs) -> List[float]:
    """
    Tool channel reward with dense per-sample signal.

    Why dense (no None values): TRL uses NaN-aware std for normalization. If a
    prompt-group has <=1 non-NaN item in this channel, channel std becomes NaN and
    this channel contributes no gradient for that group.

        We therefore provide a value for every sample:
            - Correct trajectories: 1 / (tool_count + 1)
                    * no-tool correct => 1.0 (best)
                    * one-tool correct => 0.5
            - Incorrect A-path: 0.0
            - Incorrect B-path: tiny early exploration bonus with per-call decay

        This keeps exploration alive without introducing a persistent incentive to call
        one tool on every sample.
    """
    rewards: List[float] = []
    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")
    trainer_state = kwargs.get("trainer_state")

    # Cosine decay exploration bonus over first 10% of training steps.
    explore_weight = 0.0
    if trainer_state is not None:
        current_step = getattr(trainer_state, "global_step", 0)
        total_steps = getattr(trainer_state, "max_steps", 0)
        if total_steps and total_steps > 0:
            horizon = max(1, int(0.1 * total_steps))
            scaled = min(1.0, float(current_step) / float(horizon))
            explore_weight = 0.5 * (1.0 + math.cos(math.pi * scaled))

    early_bonus = 0.05 * explore_weight

    for i in range(len(completions)):
        gold = gold_answers[i]
        choices = choices_list[i]
        _, c_score, tool_count, called_tools = _score_single_completion(completions[i], gold, choices)

        if c_score >= _TOOL_CONDITION_CORRECTNESS:
            rewards.append(float(1.0 / (tool_count + 1.0)))
        elif not called_tools:
            rewards.append(0.0)
        else:
            rewards.append(float(early_bonus / (tool_count + 1.0)))

    return rewards



def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
    """
    Legacy mixed reward (kept for backward compatibility).
    Prefer trl_reward_acc_function + trl_reward_tool_function for decoupled runs.
    """
    rewards = []

    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")

    fw = _REWARD_WEIGHTS["format"]
    cw = _REWARD_WEIGHTS["correctness"]
    mw = _REWARD_WEIGHTS["option_mention"]
    tw = _REWARD_WEIGHTS["tool_bonus"]
    path_b_count = 0

    for i in range(len(completions)):
        gold = gold_answers[i]
        choices = choices_list[i]

        f_score, c_score, tool_count, called_tools = _score_single_completion(completions[i], gold, choices)
        if called_tools:
            path_b_count += 1

        # option-mention score (0, 0.1, or 0.5)
        m_score = 1.0

        # --- tool usage score ---
        tool_score = 0.0

        if c_score >= 0.7:
            if called_tools:
                tool_score = -0.05 * tool_count
            else:
                tool_score = 0.0
        else:
            tool_score = 0.0

        total = (fw * f_score) + (cw * c_score) + tool_score
        # Clamp reward to [0, 1] to bound advantage magnitude and prevent extreme policy gradients that cause KL/entropy divergence.
        # total = max(0.0, min(1.0, total))
        rewards.append(float(total))

    print(f"(A={len(completions) - path_b_count}, B={path_b_count})")
    return rewards


# def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
#     """
#     Correctness-driven tool reward. No efficiency bonus for avoiding tools.
#     Tool usage has a near-zero cost when correct (tiebreaker only),
#     letting the correctness signal teach the model WHEN tools help.
#     """
#     rewards = []

#     gold_answers = kwargs.get("gold_answer")
#     choices_list = kwargs.get("choices")

#     trainer_state = kwargs.get("trainer_state")
#     current_step = None
#     total_steps = None

#     if trainer_state is not None:
#         current_step = getattr(trainer_state, "global_step", current_step)
#         total_steps = getattr(trainer_state, "max_steps", total_steps)

#     if current_step is None or total_steps is None:
#         raise ValueError(
#             "trl_reward_function requires trainer_state with "
#             "global_step and max_steps."
#         )

#     fw = _REWARD_WEIGHTS["format"]
#     cw = _REWARD_WEIGHTS["correctness"]
#     tw = _REWARD_WEIGHTS["tool_bonus"]

#     # Short exploration window (first 10%) — tiny nudge to discover tools
#     H = 0.1 * total_steps
#     scaled_p = min(1.0, current_step / H)
#     exploration_bonus = 0.5 * (1.0 + math.cos(math.pi * scaled_p))  # 1→0

#     for i in range(len(completions)):
#         completion = completions[i]
#         completion_text = (
#             completion[-1].get("content", "")
#             if isinstance(completion, list)
#             else str(completion)
#         )

#         gold = gold_answers[i]
#         choices = choices_list[i]

#         t = _count_tool_calls(completion_text)
#         used_tool = t > 0

#         f_score = format_reward(completion_text, used_tool)
#         c = _soft_correctness(completion_text, gold, choices)
#         STEP_PENALTY = 0.05
#         tool_score = 0.0

#         # ---- Tool score: correctness-driven, no efficiency bonus ----
#         # if used_tool:
#         #     if c >= 0.7:
#         #         # Correct with tool: near-zero cost (tiebreaker only)
#         #         tool_score = -0.02 * t
#         #     else:
#         #         # Wrong with tool: moderate penalty
#         #         tool_score = -0.15 * t
#         # else:
#         #     # No tool used: NEUTRAL regardless of correctness
#         #     # (this is the key difference — no bonus here)
#         #     tool_score = 0.0

#         # Early exploration: tiny nudge so model discovers tools exist
#         # Decays to 0 by 20% of training, never becomes negative
#         if used_tool and exploration_bonus > 0:
#             tool_score += exploration_bonus * 0.1

#         tool_tax = -1 * (t * STEP_PENALTY)

#         total = (fw * f_score) + (cw * c) + tool_tax + tool_score
#         rewards.append(float(total))

#     return rewards

# full cosine decay
# def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
#     """
#     reward function with cosine decay tool exploration.
#     Uses `trainer_state` when available to calculate training progress,
#     with fallback to `step` / `total_steps` kwargs.
#     """
#     rewards = []

#     gold_answers = kwargs.get("gold_answer")
#     choices_list = kwargs.get("choices")

#     trainer_state = kwargs.get("trainer_state")
#     current_step = None
#     total_steps = None

#     if trainer_state is not None:
#         current_step = getattr(trainer_state, "global_step", current_step)
#         total_steps = getattr(trainer_state, "max_steps", total_steps)

#     if current_step is None or total_steps is None:
#         raise ValueError("trl_reward_function requires trainer_state with global_step and max_steps, or step and total_steps kwargs.")


#     fw = _REWARD_WEIGHTS["format"]
#     cw = _REWARD_WEIGHTS["correctness"]
#     mw = _REWARD_WEIGHTS["option_mention"]
#     tw = _REWARD_WEIGHTS["tool_bonus"]


#     H = 0.4*total_steps # exploration cutoff at 40% progress
#     progress = min(1.0, current_step / total_steps)
#     scaled_p = min(1.0, current_step / H)
#     exploration_weight = 0.5 * (1.0 + math.cos(math.pi * scaled_p))
#     print(f"Progress: {progress:.4f}, Scaled Progress: {scaled_p:.4f}, Exploration weight: {exploration_weight:.4f}")

#     for i in range(len(completions)):
#         completion = completions[i]
#         completion_text = completion[-1].get("content", "") if isinstance(completion, list) else str(completion)

#         gold = gold_answers[i]
#         choices = choices_list[i]

#         t = _count_tool_calls(completion_text)
#         I = 1.0 if t > 0 else 0.0

#         f_score = format_reward(completion_text, I > 0)
#         c = _soft_correctness(completion_text, gold, choices)

#         if c >= 0.7:
#             tool_score = exploration_weight * I - 0.1 * t
#         else:
#             tool_score = exploration_weight * I - 0.3 * t

#         total = (fw * f_score) + (cw * c) + (tw * tool_score) # max=1.06
#         # total = max(0.0, min(1.5, total))
#         rewards.append(float(total))

#     return rewards


# def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
#     """
#     reward function with cosine decay tool exploration.
#     Uses `trainer_state` when available to calculate training progress,
#     with fallback to `step` / `total_steps` kwargs.
#     """
#     rewards = []

#     gold_answers = kwargs.get("gold_answer")
#     choices_list = kwargs.get("choices")

#     trainer_state = kwargs.get("trainer_state")
#     current_step = None
#     total_steps = None

#     if trainer_state is not None:
#         current_step = getattr(trainer_state, "global_step", current_step)
#         total_steps = getattr(trainer_state, "max_steps", total_steps)

#     if current_step is None or total_steps is None:
#         raise ValueError("trl_reward_function requires trainer_state with global_step and max_steps, or step and total_steps kwargs.")

#     progress = min(1.0, current_step / total_steps)
#     exploration_weight = 0.5 * (1.0 + math.cos(math.pi * progress))

#     print(f"Progress: {progress:.4f}, Exploration weight: {exploration_weight:.4f}")

#     fw = _REWARD_WEIGHTS["format"]
#     cw = _REWARD_WEIGHTS["correctness"]
#     mw = _REWARD_WEIGHTS["option_mention"]
#     tw = _REWARD_WEIGHTS["tool_bonus"]

#     for i in range(len(completions)):
#         completion = completions[i]
#         completion_text = completion[-1].get("content", "") if isinstance(completion, list) else str(completion)

#         gold = gold_answers[i]
#         choices = choices_list[i]

#         tool_count = _count_tool_calls(completion_text)
#         called_tools = tool_count > 0

#         tag_penalty = format_reward(completion_text, called_tools)
#         f_score = max(0.0, 1.0 + tag_penalty)
#         c_score = _soft_correctness(completion_text, gold, choices)
#         m_score = _option_mention_score(completion_text, gold, choices) if c_score < 0.5 else 1.0

#         tool_score = 0.0


#         if c_score >= 0.7:
#             # CORRECT ANSWER
#             if not called_tools:
#                 tool_score = 1.0
#             else:
#                 efficiency = 0.8 if tool_count == 1 else max(0.0, 0.8 - (0.2 * (tool_count - 1)))
#                 # During exploration, don't penalize tool usage for correct answers at all
#                 tool_score = (exploration_weight * 1.0) + ((1.0 - exploration_weight) * efficiency)
#         else:
#             # INCORRECT ANSWER
#             if called_tools:
#                 # The 'Switch': Early on, give +0.2 exploration bonus for trying a tool.
#                 # Later, apply the -0.5 penalty per tool.
#                 exploration_bonus = 0.2
#                 hard_penalty = max(-1.0, -0.5 * tool_count)

#                 tool_score = (exploration_weight * exploration_bonus) + ((1.0 - exploration_weight) * hard_penalty)
#             else:
#                 tool_score = 0.0

#         total = (fw * f_score) + (cw * c_score) + (mw * m_score) + (tw * tool_score)
#         total = max(0.0, min(1.0, total))
#         rewards.append(float(total))

#     return rewards

# v2
# def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
#     """
#     TRL-compatible reward function with soft correctness scoring and anti-spam tool logic.
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

#         # tool calls from JSON arrays inside <tool>...</tool>
#         tool_count = _count_tool_calls(completion_text)
#         called_tools = tool_count > 0

#         tag_penalty = format_reward(completion_text, called_tools)
#         f_score = max(0.0, 1.0 + tag_penalty)

#         c_score = _soft_correctness(completion_text, gold, choices)

#         # option-mention score (0, 0.1, or 0.5)
#         m_score = _option_mention_score(completion_text, gold, choices) if c_score < 0.5 else 1.0

#         # --- tool usage score ---
#         tool_score = 0.0

#         if c_score >= 0.7:
#             # Got the answer right
#             if not called_tools:
#                 tool_score = 1.0    # correct without tool (highly efficient)
#             else:
#                 # correct with tool
#                 if tool_count == 1:
#                     tool_score = 0.8
#                 else:
#                     # spamming tools before getting it right
#                     tool_score = max(0.0, 0.8 - (0.2 * (tool_count - 1)))
#         else:
#             # Got the answer WRONG
#             if called_tools:
#                 # Penalty for calling tools but failing. Capped at -1.0 to
#                 # prevent extreme negative advantages
#                 tool_score = max(-1.0, -0.5 * tool_count)
#             else:
#                 # Wrong answer, no tools used. Normal baseline.
#                 tool_score = 0.0

#         total = (fw * f_score) + (cw * c_score) + (mw * m_score) + (tw * tool_score)
#         # Clamp reward to [0, 1] to bound advantage magnitude and prevent extreme policy gradients that cause KL/entropy divergence.
#         total = max(0.0, min(1.0, total))
#         rewards.append(float(total))

#     return rewards

# v1
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
