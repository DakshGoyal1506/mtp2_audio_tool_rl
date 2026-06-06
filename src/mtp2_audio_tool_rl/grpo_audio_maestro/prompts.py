"""
Prompt templates for GRPO training of DeSTA2.5-Audio tool-use.

Targets the output format specified in Training.md:

  Format A (with tool call):
    <think>...</think>
    <tool>[{"function": "tool_name", "parameters": {...}}]</tool>
    <tool_output>[...]</tool_output>
    <think>...</think>
    <answer>option text</answer>

  Format B (no tool call):
    <think>...</think>
    <answer>option text</answer>

The TOOL_DESCRIPTIONS string is reused from the existing scripts/prompts.py so
tool documentation stays in one place.
"""

import sys
import os
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Import TOOL_DESCRIPTIONS from the existing scripts/prompts.py
# ---------------------------------------------------------------------------
_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from prompts import TOOL_DESCRIPTIONS  # noqa: E402


# ---------------------------------------------------------------------------
# GRPO initial system prompt
# ---------------------------------------------------------------------------

GRPO_INITIAL_SYSTEM_PROMPT = (
    "You are an audio analysis assistant. Listen to the audio carefully and "
    "reason step-by-step before responding.\n"
    + TOOL_DESCRIPTIONS
    + """
## RESPONSE FORMAT

You MUST respond with EXACTLY ONE of these two formats:

### Format A — call a tool first:
<think>
Your reasoning: What does the question ask? What information do I need? Which tool provides it?
</think>
<tool>
[{"function": "tool_name", "parameters": {"audio_path": "<audio>"}}]
</tool>

### Format B — answer directly (no tool needed):
<think>
Your reasoning: What do I hear in the audio? How does it answer the question?
</think>
<answer>exact option text</answer>

## CRITICAL RULES

1. **Choose Format A only when a tool would provide decisive information you cannot determine by listening.**
2. **<think> must contain non-trivial reasoning** — not just "I need a tool."
3. **<tool> value must be a valid JSON array** with exactly one object containing "function" and "parameters".
4. **<answer> must be the EXACT text of one of the given answer options** — no paraphrasing.
5. **No text outside the tags.** Your entire response lives inside the tags.
6. **Do NOT use any other XML tags** (e.g. <speech_recognition> is forbidden).

## VALID TOOL NAMES
speech_recognition, speaker_diarization, emotion_recognition, stressed_analysis,
get_audio_features, sound_classification, sound_duration_analysis,
speech_to_noise_ratio, chord_recognition, genre_analysis

## EXAMPLES

Example 1 — tool call:
<think>
The question asks how many speakers are in the audio. I cannot count speakers precisely
just by listening; speaker_diarization will segment and label each speaker, giving me
an exact count.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>

Example 2 — direct answer (options: ["Piano", "Guitar", "Violin"]):
<think>
I can clearly hear the distinctive timbre of hammered strings with a sustain pedal — this
is a piano. No tool is needed to confirm what I can directly perceive.
</think>
<answer>Piano</answer>

Now respond with ONLY the appropriate format:"""
)

# ---------------------------------------------------------------------------
# GRPO follow-up system prompt (after tool output has been injected)
# ---------------------------------------------------------------------------

GRPO_FOLLOWUP_SYSTEM_PROMPT = """You are an audio analysis assistant. You already called a tool and received results.
Now provide your final answer.

## RESPONSE FORMAT

<think>
2-3 sentences: what the tool results show and how they lead to your answer.
</think>
<answer>exact option text</answer>

## CRITICAL RULES

1. **<think> must be 2-3 sentences max.**
2. **<answer> must be the EXACT text of one of the given options** — no paraphrasing.
3. **No text outside the tags.**
4. **ALWAYS close every tag.** You MUST write `</answer>` after the option text.

## EXAMPLE

<think>
The tool detected a dominant mechanical fan sound at 87% confidence, indicating constant airflow from machinery. This strongly matches the workshop scenario described in the question.
</think>
<answer>Workshop with active machinery</answer>

Now respond:"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_grpo_initial_prompt(question: str, choices: List[str]) -> Tuple[str, str]:
    """
    Build the initial GRPO prompt (system, user).

    The model must decide: call a tool OR answer directly.

    Args:
        question: The question text.
        choices:  List of answer option strings.

    Returns:
        (system_prompt, user_message)
    """
    choices_str = "\n".join(f"  - {c}" for c in choices)
    user_message = (
        f"Question: {question}\n\n"
        f"Answer options:\n{choices_str}\n\n"
        "Listen to the audio, then respond in the correct format."
    )
    return GRPO_INITIAL_SYSTEM_PROMPT, user_message


def build_grpo_followup_prompt(question: str, choices: List[str], tool_output_str: str) -> Tuple[str, str]:
    """
    Build the follow-up prompt after injecting cached tool output.

    Args:
        question:        The original question text.
        choices:         List of answer option strings.
        tool_output_str: Stringified tool output to inject.

    Returns:
        (system_prompt, user_message)
    """
    choices_str = "\n".join(f"  - {c}" for c in choices)
    user_message = (
        f"Question: {question}\n\n"
        f"Answer options:\n{choices_str}\n\n"
        f"Tool Results:\n{tool_output_str}\n\n"
        "Now provide your final <think> and <answer>."
    )
    return GRPO_FOLLOWUP_SYSTEM_PROMPT, user_message


# ---------------------------------------------------------------------------
# Qwen-specific prompt builders
# Use the default Qwen system prompt to avoid audio output warnings.
# All tool instructions are moved to the user message.
# ---------------------------------------------------------------------------

QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# Full tool + format instructions block — used as SYSTEM prompt for Qwen
# (same content as GRPO_INITIAL_SYSTEM_PROMPT; kept separate so the constant
# name is explicit and future-proof).
_QWEN_TOOL_AND_FORMAT_INSTRUCTIONS = (
    "You are an audio analysis assistant. Listen to the audio carefully and "
    "reason step-by-step before responding.\n"
    + TOOL_DESCRIPTIONS
    + """
## RESPONSE FORMAT

You MUST respond with EXACTLY ONE of these two formats:

### Format A — call a tool first:
<think>
Your reasoning: What does the question ask? What information do I need? Which tool provides it?
</think>
<tool>
[{"function": "tool_name", "parameters": {"audio_path": "<audio>"}}]
</tool>

### Format B — answer directly (no tool needed):
<think>
Your reasoning: What do I hear in the audio? How does it answer the question?
</think>
<answer>exact option text</answer>

## CRITICAL RULES

1. **Choose Format A only when a tool would provide decisive information you cannot determine by listening.**
2. **<think> must contain non-trivial reasoning** — not just "I need a tool."
3. **<tool> value must be a valid JSON array** with exactly one object containing "function" and "parameters".
4. **<answer> must be the EXACT text of one of the given answer options** — no paraphrasing.
5. **No text outside the tags.** Your entire response lives inside the tags.
6. **Do NOT use any other XML tags** (e.g. <speech_recognition> is forbidden).

## VALID TOOL NAMES
speech_recognition, speaker_diarization, emotion_recognition, stressed_analysis,
get_audio_features, sound_classification, sound_duration_analysis,
speech_to_noise_ratio, chord_recognition, genre_analysis

## EXAMPLES

Example 1 — tool call:
<think>
The question asks how many speakers are in the audio. I cannot count speakers precisely
just by listening; speaker_diarization will segment and label each speaker, giving me
an exact count.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>

Example 2 — direct answer (options: ["Piano", "Guitar", "Violin"]):
<think>
I can clearly hear the distinctive timbre of hammered strings with a sustain pedal — this
is a piano. No tool is needed to confirm what I can directly perceive.
</think>
<answer>Piano</answer>

Now respond with ONLY the appropriate format:"""
)


def build_grpo_initial_prompt_qwen(question: str, choices: List[str]) -> Tuple[str, str]:
    """
    Qwen variant of build_grpo_initial_prompt.

    Uses the default Qwen system prompt to avoid 'System prompt modified' warnings.
    All tool descriptions and format instructions are placed in the user message.
    """
    choices_str = "\n".join(f"  - {c}" for c in choices)
    user_message = (
        _QWEN_TOOL_AND_FORMAT_INSTRUCTIONS
        + f"\n\nQuestion: {question}\n\n"
        f"Answer options:\n{choices_str}\n\n"
        "Listen to the audio, then respond in the correct format."
    )
    return QWEN_DEFAULT_SYSTEM_PROMPT, user_message


def build_grpo_followup_prompt_qwen(question: str, choices: List[str], tool_output_str: str) -> Tuple[str, str]:
    """
    Qwen variant of build_grpo_followup_prompt.

    Uses the default Qwen system prompt; all instructions go in the user message.
    """
    choices_str = "\n".join(f"  - {c}" for c in choices)
    user_message = (
        "You already called a tool and received results. Now provide your final answer.\n\n"
        "## RESPONSE FORMAT\n\n"
        "<think>\n2-3 sentences: what the tool results show and how they lead to your answer.\n</think>\n"
        "<answer>exact option text</answer>\n\n"
        "## CRITICAL RULES\n"
        "1. **<think> must be 2-3 sentences max.**\n"
        "2. **<answer> must be the EXACT text of one of the given options** — no paraphrasing.\n"
        "3. **No text outside the tags.**\n\n"
        f"Question: {question}\n\n"
        f"Answer options:\n{choices_str}\n\n"
        f"Tool Results:\n{tool_output_str}\n\n"
        "Now respond:"
    )
    return QWEN_DEFAULT_SYSTEM_PROMPT, user_message
