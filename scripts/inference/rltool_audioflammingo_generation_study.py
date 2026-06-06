"""
AudioFlamingo3 generation study: Compare two prompt styles.

Style 1 - "Think+Tool" (GRPO format from rlTool/prompts.py):
  System prompt instructs model to use <think>...<tool>... or <think>...<answer>...

Style 2 - "Direct" (plain MCQ format from mmau_infer.py):
  Simple "Answer with the correct option text only" prompt.

Sends 10 samples from mmau-test-mini-cached.json to AudioFlamingo3 vLLM server
and logs full outputs to a log file for qualitative comparison.

Usage:
    python audioflammingo_generation_study.py \
        --host 192.168.1.44 --port 8000 \
        --input mmau-test-mini-cached.json \
        --audio-dir ../skills/test-mini-audios \
        --log-file generation_study.log
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Tool list (kept short for small models)
# ---------------------------------------------------------------------------
TOOL_LIST = """Tools: speech_recognition, speaker_diarization, emotion_recognition, stress_analysis, get_audio_features, sound_classification, sound_duration_analysis, speech_to_noise_ratio, chord_recognition, genre_analysis"""

# ---------------------------------------------------------------------------
# Style 1: Think+Tool in system prompt (simplified)
# ---------------------------------------------------------------------------
GRPO_SYSTEM_PROMPT = """You are an audio assistant with tools.
""" + TOOL_LIST + """

Respond in ONE of these formats:

Format A (use a tool):
<think>why you need the tool</think>
<tool>[{"function": "tool_name", "parameters": {"audio_path": "<audio>"}}]</tool>

Format B (answer directly):
<think>what you hear and why</think>
<answer>exact option text</answer>

Rules:
- <answer> must be EXACT text from the options.
- Only use a tool if listening alone cannot answer the question."""


THINK_SUFFIX = "\nPlease think and reason about the input audio before you respond."


def build_grpo_initial_prompt(question: str, choices: List[str], think: bool = False) -> Tuple[str, str]:
    """Build Think+Tool prompt (system, user)."""
    choices_str = " | ".join(choices)
    user_message = f"{question}\nOptions: {choices_str}"
    if think:
        user_message += THINK_SUFFIX
    return GRPO_SYSTEM_PROMPT, user_message


def get_model_name(host: str, port: int) -> str:
    """Query the /v1/models endpoint to get the served model name."""
    url = f"http://{host}:{port}/v1/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["data"][0]["id"]


def encode_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_model(host: str, port: int, model_name: str, messages: list,
               max_tokens: int = 1024, temperature: float = 0.7) -> str:
    """Send a chat completion request and return the content string."""
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"http://{host}:{port}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp_data = json.loads(resp.read().decode("utf-8"))
    return resp_data["choices"][0]["message"]["content"]


def build_direct_prompt(question: str, choices: List[str], think: bool = False) -> Tuple[str, str]:
    """Style 2: Simple direct MCQ prompt."""
    system = "You are an audio assistant. Answer the question about the audio."
    choices_str = " | ".join(choices)
    user = f"{question}\nOptions: {choices_str}\n\nAnswer with the exact option text only."
    if think:
        user += THINK_SUFFIX
    return system, user


# ---------------------------------------------------------------------------
# Style 3: Think+Tool instructions in USER role (not system)
# ---------------------------------------------------------------------------

def build_user_role_think_prompt(question: str, choices: List[str], think: bool = False) -> Tuple[str, str]:
    """Style 3: Think+Tool instructions in user message, minimal system prompt."""
    system = "You are an audio assistant."
    choices_str = " | ".join(choices)
    user = (
        f"{TOOL_LIST}\n\n"
        f"Respond in ONE of these formats:\n"
        f"Format A (use a tool): <think>reasoning</think><tool>[{{\"function\": \"tool_name\", \"parameters\": {{\"audio_path\": \"<audio>\"}}}}]</tool>\n"
        f"Format B (answer directly): <think>reasoning</think><answer>exact option text</answer>\n\n"
        f"{question}\nOptions: {choices_str}"
    )
    if think:
        user += THINK_SUFFIX
    return system, user


def run_sample(sample: dict, host: str, port: int, model_name: str,
               audio_dir: str, think: bool = False) -> dict:
    """Run all prompt styles on a single sample and return results."""
    # Resolve audio path
    audio_id = sample["audio_id"]
    if audio_id.startswith("./"):
        audio_id = audio_id[2:]
    audio_path = os.path.join(audio_dir, os.path.basename(audio_id))

    if not os.path.isfile(audio_path):
        return {"error": f"Audio not found: {audio_path}"}

    audio_b64 = encode_audio(audio_path)
    question = sample["question"]
    choices = sample["choices"]
    gold = sample["answer"]

    results = {
        "id": sample["id"],
        "question": question,
        "choices": choices,
        "gold_answer": gold,
    }

    # Audio content block (shared)
    audio_content = {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": "wav"},
    }

    max_tok = 1024 if think else 256

    # --- Style 1: Think+Tool (GRPO prompt) ---
    sys_prompt_1, user_msg_1 = build_grpo_initial_prompt(question, choices, think=think)
    messages_1 = [
        {"role": "system", "content": sys_prompt_1},
        {"role": "user", "content": [
            audio_content,
            {"type": "text", "text": user_msg_1},
        ]},
    ]
    try:
        output_1 = call_model(host, port, model_name, messages_1, max_tokens=max_tok)
        results["style1_think_tool"] = output_1
    except Exception as e:
        results["style1_think_tool"] = f"ERROR: {e}"

    # --- Style 2: Direct (simple MCQ) ---
    sys_prompt_2, user_msg_2 = build_direct_prompt(question, choices, think=think)
    messages_2 = [
        {"role": "system", "content": sys_prompt_2},
        {"role": "user", "content": [
            audio_content,
            {"type": "text", "text": user_msg_2},
        ]},
    ]
    try:
        output_2 = call_model(host, port, model_name, messages_2, max_tokens=max_tok)
        results["style2_direct"] = output_2
    except Exception as e:
        results["style2_direct"] = f"ERROR: {e}"

    # --- Style 3: Think+Tool in USER role ---
    sys_prompt_3, user_msg_3 = build_user_role_think_prompt(question, choices, think=think)
    messages_3 = [
        {"role": "system", "content": sys_prompt_3},
        {"role": "user", "content": [
            audio_content,
            {"type": "text", "text": user_msg_3},
        ]},
    ]
    try:
        output_3 = call_model(host, port, model_name, messages_3, max_tokens=max_tok)
        results["style3_user_role"] = output_3
    except Exception as e:
        results["style3_user_role"] = f"ERROR: {e}"

    return results


def extract_answer(text: str, choices: List[str]) -> str:
    """Extract answer from model output for evaluation."""
    # Try <answer> tags
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # Fallback: match a choice in the text
    for c in choices:
        if c.lower() in text.lower():
            return c
    # Last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def format_log_entry(idx: int, result: dict) -> str:
    """Format a single result into a readable log block."""
    sep = "=" * 80
    lines = [
        sep,
        f"  SAMPLE {idx + 1}: {result['id']}",
        sep,
        f"  Question: {result['question']}",
        f"  Choices:  {result['choices']}",
        f"  Gold:     {result['gold_answer']}",
        "",
    ]

    # Style 1
    lines.append("-" * 40 + " STYLE 1: Think+Tool " + "-" * 40)
    s1 = result.get("style1_think_tool", "N/A")
    pred1 = extract_answer(s1, result["choices"]) if "ERROR" not in s1 else "ERROR"
    correct1 = "CORRECT" if pred1.lower() == result["gold_answer"].lower() else "WRONG"
    lines.append(f"  Predicted: {pred1} [{correct1}]")
    lines.append(f"  Raw output:")
    for line in s1.split("\n"):
        lines.append(f"    | {line}")
    lines.append("")

    # Style 2
    lines.append("-" * 40 + " STYLE 2: Direct     " + "-" * 40)
    s2 = result.get("style2_direct", "N/A")
    pred2 = extract_answer(s2, result["choices"]) if "ERROR" not in s2 else "ERROR"
    correct2 = "CORRECT" if pred2.lower() == result["gold_answer"].lower() else "WRONG"
    lines.append(f"  Predicted: {pred2} [{correct2}]")
    lines.append(f"  Raw output:")
    for line in s2.split("\n"):
        lines.append(f"    | {line}")
    lines.append("")

    # Style 3
    lines.append("-" * 40 + " STYLE 3: Think+Tool in User Role " + "-" * 27)
    s3 = result.get("style3_user_role", "N/A")
    pred3 = extract_answer(s3, result["choices"]) if "ERROR" not in s3 else "ERROR"
    correct3 = "CORRECT" if pred3.lower() == result["gold_answer"].lower() else "WRONG"
    lines.append(f"  Predicted: {pred3} [{correct3}]")
    lines.append(f"  Raw output:")
    for line in s3.split("\n"):
        lines.append(f"    | {line}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AudioFlamingo3 two-prompt-style generation study"
    )
    parser.add_argument("--host", default="192.168.1.44")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input", default="mmau-test-mini-cached.json")
    parser.add_argument("--audio-dir", default="../skills/test-mini-audios")
    parser.add_argument("--log-file", default="generation_study.log")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--think", action="store_true",
                        help="Add 'think step by step' instruction to all styles")
    args = parser.parse_args()

    # Load samples
    with open(args.input, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.input}")

    # Select first N samples (diverse subset)
    selected = samples[: args.num_samples]
    mode_label = "think" if args.think else "direct"
    print(f"Running {len(selected)} samples through 3 prompt styles (mode={mode_label})...")

    # Get model name
    if args.model:
        model_name = args.model
    else:
        model_name = get_model_name(args.host, args.port)
    print(f"Model: {model_name}")

    # Run inference
    all_results = []
    style1_correct = 0
    style2_correct = 0
    style3_correct = 0

    for i, sample in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {sample['id'][:12]}... ", end="", flush=True)
        result = run_sample(sample, args.host, args.port, model_name, args.audio_dir,
                            think=args.think)
        all_results.append(result)

        if "error" in result:
            print(f"SKIP ({result['error']})")
            continue

        # Quick accuracy check
        s1 = result.get("style1_think_tool", "")
        s2 = result.get("style2_direct", "")
        s3 = result.get("style3_user_role", "")
        pred1 = extract_answer(s1, sample["choices"]) if "ERROR" not in s1 else ""
        pred2 = extract_answer(s2, sample["choices"]) if "ERROR" not in s2 else ""
        pred3 = extract_answer(s3, sample["choices"]) if "ERROR" not in s3 else ""
        c1 = pred1.lower() == sample["answer"].lower()
        c2 = pred2.lower() == sample["answer"].lower()
        c3 = pred3.lower() == sample["answer"].lower()
        if c1:
            style1_correct += 1
        if c2:
            style2_correct += 1
        if c3:
            style3_correct += 1
        print(f"S1:{'OK' if c1 else 'X'} S2:{'OK' if c2 else 'X'} S3:{'OK' if c3 else 'X'}")

    # Write log file
    n_valid = len([r for r in all_results if "error" not in r])
    with open(args.log_file, "w") as f:
        f.write(f"AudioFlamingo3 Generation Study\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Server: {args.host}:{args.port}\n")
        f.write(f"Samples: {n_valid}\n")
        f.write(f"\n")
        f.write(f"SUMMARY\n")
        f.write(f"  Style 1 (Think+Tool in System): {style1_correct}/{n_valid} correct\n")
        f.write(f"  Style 2 (Direct):               {style2_correct}/{n_valid} correct\n")
        f.write(f"  Style 3 (Think+Tool in User):   {style3_correct}/{n_valid} correct\n")
        f.write(f"\n{'=' * 80}\n")
        f.write(f"  DETAILED OUTPUTS\n")
        f.write(f"{'=' * 80}\n\n")

        for i, result in enumerate(all_results):
            if "error" not in result:
                f.write(format_log_entry(i, result))
                f.write("\n\n")

    print(f"\nResults written to: {args.log_file}")
    print(f"Style 1 (Think+Tool in System): {style1_correct}/{n_valid}")
    print(f"Style 2 (Direct):               {style2_correct}/{n_valid}")
    print(f"Style 3 (Think+Tool in User):   {style3_correct}/{n_valid}")


if __name__ == "__main__":
    main()
