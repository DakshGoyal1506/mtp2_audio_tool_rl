"""
Empirical Tool Helpfulness Judge
=================================
For each (question, choices), give Gemma each tool's ACTUAL output one at a time,
ask it to reason (CoT) and pick an answer. Compare with gold answer.

A tool is marked helpful if Gemma gets the right answer using that tool's output.

Input:  synthetic_dataset/synth_cached_train.json
Output: tool_empirical_results.json

Usage:
  python judge_tool_empirical.py --host <vllm_host> --port 8000
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# tool name in gold_tool -> key in tool_outputs
TOOL_ALIAS = {"get_audio_features": "audio_features"}

ALL_TOOLS = [
    "speech_recognition", "speaker_diarization", "emotion_recognition",
    "sound_classification", "sound_duration_analysis", "genre_analysis",
    "audio_features", "speech_to_noise_ratio", "stressed_analysis",
    "chord_recognition",
]

SYSTEM_PROMPT = """\
You are an expert audio analysis assistant.
You will be given a multiple-choice question about an audio clip, the answer choices, \
and the output of one audio-analysis tool that was run on that audio.

Your task: use the tool output to reason step-by-step, then pick the correct answer.

Rules:
- First, explain your reasoning in 2-3 sentences based on the tool output.
- Then, state your final answer as EXACTLY one of the given choices.
- Format your response as JSON (no markdown fences):
  {"reasoning": "...", "answer": "..."}
"""


def build_user_prompt(question, choices, tool_name, tool_output):
    choices_str = "\n".join(f"  - {c}" for c in choices)
    tool_json = json.dumps(tool_output, indent=2, ensure_ascii=False)
    return f"""\
Question: {question}

Choices:
{choices_str}

Tool used: {tool_name}
Tool output:
{tool_json}

Reason step-by-step using the tool output, then give your final answer. Return JSON: {{"reasoning": "...", "answer": "..."}}"""


def query_gemma(host, port, system_prompt, user_prompt, max_tokens=400):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-26B-A4B-it",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def parse_response(raw_text):
    """Extract {"reasoning": ..., "answer": ...} from response."""
    # 1) Direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    # 2) Find outermost { }
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end+1])
        except json.JSONDecodeError:
            pass
    # 3) Try to fix common issues: unescaped quotes inside strings
    if start != -1 and end != -1:
        block = raw_text[start:end+1]
        # Attempt: extract reasoning and answer with regex
        r_match = re.search(r'"reasoning"\s*:\s*"(.*?)",\s*"answer"\s*:\s*"(.*?)"',
                            block, re.DOTALL)
        if r_match:
            return {"reasoning": r_match.group(1), "answer": r_match.group(2)}
        # Reversed order
        r_match = re.search(r'"answer"\s*:\s*"(.*?)",\s*"reasoning"\s*:\s*"(.*?)"',
                            block, re.DOTALL)
        if r_match:
            return {"answer": r_match.group(1), "reasoning": r_match.group(2)}
    return None


# Retry prompt for when parsing fails
RETRY_PROMPT = """\
Your previous response could not be parsed. Please answer again.
Reply with ONLY a JSON object, no other text:
{{"reasoning": "<your 1-2 sentence reasoning>", "answer": "<exact choice text>"}}
"""


def match_answer(predicted, gold, choices):
    """Check if predicted answer matches gold, handling letter labels & fuzzy matching."""
    if predicted is None:
        return False
    pred = str(predicted).strip()
    gold_norm = gold.strip().lower()

    # 1) Direct match (case-insensitive)
    if pred.lower() == gold_norm:
        return True

    # 2) Predicted is a letter like "B" or "B." -> map to choice
    letter_match = re.match(r'^([A-Da-d])[\.\)\s]?$', pred)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord('A')
        if 0 <= idx < len(choices) and choices[idx].strip().lower() == gold_norm:
            return True

    # 3) Predicted starts with letter prefix like "A. happy" -> strip prefix
    prefix_match = re.match(r'^[A-Da-d][\.\)]\s*(.+)$', pred)
    if prefix_match:
        stripped = prefix_match.group(1).strip().lower()
        if stripped == gold_norm:
            return True

    # 4) Gold is contained in predicted or vice versa (fuzzy)
    if gold_norm in pred.lower() or pred.lower() in gold_norm:
        return True

    return False


VERIFY_SYSTEM = """\
You are a strict auditor. You will see a question, the tool output that was provided, \
and the reasoning+answer a model gave. Your job: decide if the model's answer was \
genuinely derived from information in the tool output, or if the model guessed/speculated \
despite the tool output being irrelevant or insufficient.

Reply ONLY with JSON (no markdown):
{"verified": true/false, "explanation": "one sentence why"}

Set verified=true ONLY if the tool output contains concrete evidence that logically \
leads to the answer. Set verified=false if:
- The reasoning says the tool is insufficient/irrelevant but still picks an answer.
- The model guessed based on general knowledge, not the tool output.
- The reasoning contradicts the tool output.
"""


def build_verify_prompt(question, choices, tool_name, tool_output, reasoning, predicted):
    choices_str = "\n".join(f"  - {c}" for c in choices)
    tool_json = json.dumps(tool_output, indent=2, ensure_ascii=False)
    return f"""\
Question: {question}
Choices:
{choices_str}

Tool: {tool_name}
Tool output:
{tool_json}

Model's reasoning: {reasoning}
Model's answer: {predicted}

Was this answer genuinely derived from the tool output, or was it a guess? Return JSON only."""


def verify_result(sample, tool_name, tool_result, host, port):
    """2nd layer: check if a 'correct' answer was actually derived from tool output."""
    tool_key = TOOL_ALIAS.get(tool_name, tool_name)
    tool_output = sample["tool_outputs"].get(tool_key, {})
    prompt = build_verify_prompt(
        sample["question"], sample["choices"], tool_name, tool_output,
        tool_result["reasoning"], tool_result["predicted"],
    )
    raw = query_gemma(host, port, VERIFY_SYSTEM, prompt, max_tokens=200)
    parsed = parse_response(raw)
    if parsed and "verified" in parsed:
        return parsed["verified"], parsed.get("explanation", "")
    # Fallback: look for true/false in text
    lower = raw.lower()
    if '"verified": true' in lower or '"verified":true' in lower:
        return True, raw
    return False, raw


def process_one(sample, tool_name, host, port):
    """Ask Gemma to answer question using one tool's output. Retry on parse failure."""
    question = sample["question"]
    choices  = sample["choices"]
    gold     = sample["answer"]
    tool_outputs = sample["tool_outputs"]

    tool_key = TOOL_ALIAS.get(tool_name, tool_name)
    tool_output = tool_outputs.get(tool_key, {})

    user_prompt = build_user_prompt(question, choices, tool_name, tool_output)

    # First attempt
    raw = query_gemma(host, port, SYSTEM_PROMPT, user_prompt)
    parsed = parse_response(raw)

    # Retry if parse failed
    if parsed is None:
        retry_raw = query_gemma(host, port, SYSTEM_PROMPT,
                                user_prompt + "\n\n" + RETRY_PROMPT)
        parsed = parse_response(retry_raw)
        if parsed is None:
            # Last resort: try to extract answer by matching choices in raw text
            raw_lower = raw.lower()
            for c in choices:
                if c.lower() in raw_lower:
                    parsed = {"answer": c, "reasoning": raw[:500]}
                    break

    if parsed:
        predicted = parsed.get("answer", "")
        reasoning = parsed.get("reasoning", "")
        correct = match_answer(predicted, gold, choices)
    else:
        predicted = raw
        reasoning = ""
        correct = False

    result = {
        "tool": tool_name,
        "predicted": predicted,
        "reasoning": reasoning,
        "correct": correct,
    }

    # 2nd layer verification for correct answers
    if correct:
        verified, explanation = verify_result(sample, tool_name, result, host, port)
        result["verified"] = verified
        result["verify_explanation"] = explanation
    else:
        result["verified"] = False

    return result


def process_sample(sample, host, port):
    """Process all 10 tools for one sample."""
    results_per_tool = {}
    for tool_name in ALL_TOOLS:
        results_per_tool[tool_name] = process_one(sample, tool_name, host, port)

    # correct = right answer
    helpful = sorted([t for t, r in results_per_tool.items() if r["correct"]])
    unhelpful = sorted([t for t, r in results_per_tool.items() if not r["correct"]])
    # verified = right answer AND not speculated
    verified_helpful = sorted([t for t, r in results_per_tool.items()
                               if r["correct"] and r.get("verified")])

    gold_tool = sample.get("tool", "")
    gold_key = TOOL_ALIAS.get(gold_tool, gold_tool)

    return {
        "id":       sample.get("id", ""),
        "question": sample["question"],
        "choices":  sample["choices"],
        "answer":   sample["answer"],
        "gold_tool": gold_tool,
        "task":     sample.get("task", ""),
        "helpful_tools":          helpful,
        "verified_helpful_tools": verified_helpful,
        "unhelpful_tools":        unhelpful,
        "num_helpful":            len(helpful),
        "num_verified_helpful":   len(verified_helpful),
        "gold_tool_marked_helpful":  gold_key in helpful,
        "gold_tool_verified":        gold_key in verified_helpful,
        "tool_results": results_per_tool,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="synthetic_dataset/synth_cached_train.json")
    parser.add_argument("--output", default="tool_empirical_results.json")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel requests to vLLM")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    total_calls = len(data) * len(ALL_TOOLS)
    print(f"Processing {len(data)} samples × {len(ALL_TOOLS)} tools = {total_calls} calls")
    print(f"Judge: http://{args.host}:{args.port}")

    results = [None] * len(data)
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_sample, s, args.host, args.port): i
            for i, s in enumerate(data)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {
                    "id": data[idx].get("id", ""),
                    "question": data[idx]["question"],
                    "error": str(e),
                }
            done += 1
            if done % 20 == 0 or done == len(data):
                print(f"  {done}/{len(data)} samples done")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid = [r for r in results if "tool_results" in r]
    errors = len(results) - len(valid)

    print(f"\n{'='*60}")
    print(f"Total samples: {len(results)},  Errors: {errors}")
    print()

    # Per-tool accuracy (raw correct vs verified)
    tool_correct = Counter()
    tool_verified = Counter()
    tool_total = Counter()
    for r in valid:
        for t, tr in r["tool_results"].items():
            tool_total[t] += 1
            if tr["correct"]:
                tool_correct[t] += 1
            if tr.get("verified"):
                tool_verified[t] += 1

    print(f"{'Tool':<30s}  {'Raw':>5s}  {'Verified':>8s}  {'Total':>5s}  {'Raw%':>6s}  {'Ver%':>6s}")
    print("-" * 70)
    for t in ALL_TOOLS:
        c, v, tot = tool_correct[t], tool_verified[t], tool_total[t]
        print(f"  {t:<28s}  {c:>5d}  {v:>8d}  {tot:>5d}  {100*c/tot:5.1f}%  {100*v/tot:5.1f}%")

    # Gold tool accuracy
    gold_hit = sum(1 for r in valid if r["gold_tool_marked_helpful"])
    gold_ver = sum(1 for r in valid if r.get("gold_tool_verified"))
    print(f"\nGold tool correct (raw):     {gold_hit}/{len(valid)} "
          f"({100*gold_hit/len(valid):.1f}%)")
    print(f"Gold tool correct (verified): {gold_ver}/{len(valid)} "
          f"({100*gold_ver/len(valid):.1f}%)")

    # Per gold-tool breakdown
    gold_stats = defaultdict(lambda: {"total": 0, "hit": 0, "ver": 0})
    for r in valid:
        gt = r["gold_tool"]
        gold_stats[gt]["total"] += 1
        if r["gold_tool_marked_helpful"]:
            gold_stats[gt]["hit"] += 1
        if r.get("gold_tool_verified"):
            gold_stats[gt]["ver"] += 1
    print(f"\n{'Gold Tool':<28s}  {'Raw':>7s}  {'Verified':>10s}")
    print("-" * 55)
    for t in sorted(gold_stats):
        s = gold_stats[t]
        print(f"  {t:<26s}  {s['hit']:3d}/{s['total']:3d} ({100*s['hit']/s['total']:5.1f}%)  "
              f"{s['ver']:3d}/{s['total']:3d} ({100*s['ver']/s['total']:5.1f}%)")

    # Negative samples
    negatives = [r for r in valid if r["num_helpful"] == 0]
    neg_verified = [r for r in valid if r["num_verified_helpful"] == 0]
    print(f"\nSamples where NO tool helped (raw):      {len(negatives)}")
    print(f"Samples where NO tool helped (verified):  {len(neg_verified)}")

    # Speculation stats
    total_correct = sum(tool_correct.values())
    total_verified = sum(tool_verified.values())
    speculated = total_correct - total_verified
    print(f"\nSpeculation filtered out: {speculated}/{total_correct} "
          f"({100*speculated/total_correct:.1f}% of 'correct' answers were guesses)"
          if total_correct else "")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
