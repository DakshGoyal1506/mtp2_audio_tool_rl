"""
LLM-augmented reward function for GRPO training.

Path-dependent scoring:
  Path A (no tool):   0.6*answer + 0.3*think + 0.1*format
  Path B (with tool): 0.5*answer + 0.15*think1 + 0.1*tool + 0.15*think2 + 0.1*format

Each dimension scored 1-5 by the LLM judge, normalized to [0,1] by dividing by 5.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from rewards_rule import (
    format_reward,
    extract_answer,
    _extract_tag,
    _count_tool_calls,
    _normalize,
    _soft_correctness,
    _option_mention_score,
    trl_reward_function as _heuristic_reward_function,
)

logger = logging.getLogger(__name__)

# ── Global judge reference (set by training script) ───────────────────────────

_JUDGE = None  # LLMJudge instance, set by set_judge()

def set_judge(judge):
    """Set the global LLMJudge instance used by the reward function."""
    global _JUDGE
    _JUDGE = judge
    logger.info(f"LLM judge set: {judge.base_url if judge else 'None'}")


# ── Reward weights per path ──────────────────────────────────────────────────

PATH_A_WEIGHTS = {
    "answer": 0.60,
    "think":  0.30,
    "format": 0.10,
}

PATH_B_WEIGHTS = {
    "answer": 0.50,
    "think1": 0.15,
    "tool":   0.10,
    "think2": 0.15,
    "format": 0.10,
}

def set_path_a_weights(answer: float, think: float, fmt: float):
    """Override Path A weights from config."""
    PATH_A_WEIGHTS["answer"] = answer
    PATH_A_WEIGHTS["think"] = think
    PATH_A_WEIGHTS["format"] = fmt
    logger.info(f"Path A weights: answer={answer}, think={think}, format={fmt}")

def set_path_b_weights(answer: float, think1: float, tool: float, think2: float, fmt: float):
    """Override Path B weights from config."""
    PATH_B_WEIGHTS["answer"] = answer
    PATH_B_WEIGHTS["think1"] = think1
    PATH_B_WEIGHTS["tool"] = tool
    PATH_B_WEIGHTS["think2"] = think2
    PATH_B_WEIGHTS["format"] = fmt
    logger.info(
        f"Path B weights: answer={answer}, think1={think1}, tool={tool}, "
        f"think2={think2}, format={fmt}"
    )


# ── Path detection ────────────────────────────────────────────────────────────

def _is_path_b(text: str) -> bool:
    """Detect if the completion used a tool call (Path B)."""
    return "<tool>" in text


# ── Extract tool info for judge context ───────────────────────────────────────

def _extract_tool_info(completion_text: str) -> tuple:
    """Extract tool name and output from completion for judge context."""
    tool_name = None
    tool_output = None

    tool_block = _extract_tag(completion_text, "tool")
    if tool_block:
        try:
            arr = json.loads(tool_block.strip())
            if isinstance(arr, list) and arr:
                tool_name = arr[0].get("function")
        except (json.JSONDecodeError, AttributeError):
            tool_name = "malformed"

    tool_out_block = _extract_tag(completion_text, "tool_output")
    if tool_out_block:
        tool_output = tool_out_block[:1500]

    return tool_name, tool_output


def _is_default_scores(scores: Dict[str, float]) -> bool:
    """Check if judge returned all-default scores (every value == 1.0), i.e. it failed."""
    return all(v == 1.0 for v in scores.values())


# ── Compute weighted reward from judge scores ────────────────────────────────

def _compute_reward(scores: Dict[str, float], path_b: bool) -> float:
    """
    Compute weighted reward from raw 1-5 judge scores.
    Each score is normalized to [0,1] by (score - 1) / 4, then weighted.
    In case of bad output (missing score), it defaults to 1.0 (reward 0).
    """
    weights = PATH_B_WEIGHTS if path_b else PATH_A_WEIGHTS
    total = 0.0
    n_answer = scores.get("answer", 1.0)//4
    for key, weight in weights.items():
        if key not in scores:
            print(f"Missing judge score for '{key}', defaulting to 1.0")
            raw = 1.0
        else:
            raw = scores[key]

        normalized_score = (raw - 1.0) / 4.0
        if key == 'tool':
            # penalize or reward only and only when answer is correct
            normalized_score = (raw -1) / 4.0
            tool_score = weight * normalized_score if n_answer == 1.0 else 0.0
            total += tool_score
            # print(f"#######  Tool score: {tool_score}, n_answer: {n_answer} #######")
        else:
            total += weight * normalized_score

    return max(0.0, min(1.0, total))


# ── Main reward function ──────────────────────────────────────────────────────

def trl_reward_function(prompts, completions, **kwargs) -> List[float]:
    """
    TRL-compatible reward function with LLM judge, path-dependent scoring.

    Path A (no tool):   0.6*answer + 0.3*think + 0.1*format  (all /5)
    Path B (with tool): 0.5*answer + 0.15*think1 + 0.1*tool + 0.15*think2 + 0.1*format  (all /5)
    """
    # bypass judge for prototting
    return _heuristic_reward_function(prompts, completions, **kwargs)

    gold_answers = kwargs.get("gold_answer")
    choices_list = kwargs.get("choices")

    # ── Prepare batch data ────────────────────────────────────────────────
    batch_texts = []
    batch_questions = []
    batch_choices = []
    batch_golds = []
    batch_tool_names = []
    batch_tool_outputs = []
    batch_predicted = []
    batch_is_path_b = []

    for i in range(len(completions)):
        completion = completions[i]
        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[-1], dict):
            text = completion[-1].get("content", "")
        else:
            text = str(completion)

        batch_texts.append(text)

        # Extract question from prompt (last user message)
        prompt = prompts[i]
        if isinstance(prompt, list):
            question = ""
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    # TRL dict items often keep raw keys, so we check msg["question"] first
                    if "question" in msg and msg["question"]:
                        question = msg["question"]
                        break

                    # Fallback to parsing from content
                    content = msg.get("content", "")
                    q_match = re.search(r'Question:\s*(.+?)(?:\n|$)', content)
                    question = q_match.group(1) if q_match else content[:200]
                    break
            batch_questions.append(question)
        else:
            batch_questions.append(str(prompt)[:200])

        gold = gold_answers[i]
        choices = choices_list[i]
        batch_choices.append(choices if isinstance(choices, list) else [])
        batch_golds.append(gold)

        tool_name, tool_output = _extract_tool_info(text)
        batch_tool_names.append(tool_name)
        batch_tool_outputs.append(tool_output)
        batch_predicted.append(extract_answer(text))
        batch_is_path_b.append(_is_path_b(text))

    # ── Call LLM judge in concurrent batch ────────────────────────────────
    if _JUDGE is not None:
        judge_scores = _JUDGE.score_batch(
            completions=batch_texts,
            questions=batch_questions,
            choices_list=batch_choices,
            gold_answers=batch_golds,
            predicted_answers=batch_predicted,
            tool_names=batch_tool_names,
            tool_outputs=batch_tool_outputs,
            is_path_b=batch_is_path_b,
        )
    else:
        # No judge — use default 1.0 (zero reward) scores
        judge_scores = []
        for b in batch_is_path_b:
            if b:
                judge_scores.append({"think1": 1.0, "tool": 1.0, "think2": 1.0,
                                     "answer": 1.0, "format": 1.0})
            else:
                judge_scores.append({"think": 1.0, "answer": 1.0, "format": 1.0})

    # ── Compute per-completion rewards (with heuristic fallback) ─────────
    rewards = []
    n_fallback = 0
    for i in range(len(completions)):
        if _is_default_scores(judge_scores[i]):
            # Judge failed for this completion — use heuristic reward
            n_fallback += 1
            fb = _heuristic_reward_function(
                [prompts[i]], [completions[i]],
                gold_answer=[batch_golds[i]],
                choices=[batch_choices[i]],
            )
            rewards.append(fb[0])
        else:
            reward = _compute_reward(judge_scores[i], batch_is_path_b[i])
            rewards.append(float(reward))

    # ── Log batch summary ─────────────────────────────────────────────────
    n_a = sum(1 for b in batch_is_path_b if not b)
    n_b = sum(1 for b in batch_is_path_b if b)
    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
    fallback_str = f", fallback={n_fallback}" if n_fallback else ""
    rank = os.environ.get("LOCAL_RANK", "0")
    logger.info(
        f"[Rank {rank}] Judge batch: {len(rewards)} completions (A={n_a}, B={n_b}{fallback_str}), "
        f"avg_reward={avg_r:.3f}, min={min(rewards):.3f}, max={max(rewards):.3f}"
    )
    for i in range(len(rewards)):
        path = "B" if batch_is_path_b[i] else "A"
        logger.debug(
            f"  [{i}] Path {path} | q={batch_questions[i][:60]!r} "
            f"gold={batch_golds[i]!r} pred={batch_predicted[i]!r} "
            f"tool={batch_tool_names[i]} | scores={judge_scores[i]} | reward={rewards[i]:.3f}"
        )

    return rewards
