"""
Tool Helpfulness Judge
======================
For each (question, choices, answer) triple, ask Gemma which of the 10 tools
can help answer the question (positive) and which cannot (negative).

Input:  synthetic_dataset/synth_cached_train.json
Output: tool_helpfulness_results.json  (clean, only needed fields)

Usage:
  python judge_tool_helpfulness.py --host <vllm_host> --port 8000
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── All 10 tools with short descriptions ──────────────────────────────────────
TOOL_DESCRIPTIONS = {
    "speech_recognition":      "Transcribes spoken words in the audio to text.",
    "speaker_diarization":     "Identifies distinct speakers and their time segments.",
    "emotion_recognition":     "Detects the emotional tone (happy, angry, sad …) of each speaker.",
    "sound_classification":    "Classifies environmental / non-speech sound events (cough, siren …).",
    "sound_duration_analysis": "Measures duration of each sound event in the audio.",
    "genre_analysis":          "Identifies the music genre or instrument in the audio.",
    "audio_features":          "Extracts low-level features: volume, pitch, timbre, tempo.",
    "speech_to_noise_ratio":   "Estimates the signal-to-noise ratio of speech.",
    "stressed_analysis":       "Detects stressed vs unstressed phonemes / syllables.",
    "chord_recognition":       "Identifies musical chords and their timestamps.",
}

TOOL_LIST = sorted(TOOL_DESCRIPTIONS.keys())

SYSTEM_PROMPT = """\
You are an expert audio analysis judge.
You will be given a multiple-choice question about an audio clip, the answer choices, and the correct answer.
You also have a list of 10 audio-analysis tools with descriptions.

Your task: decide which tools could help answer the question correctly, and explain why.

Rules:
- A tool is HELPFUL if its output would contain information directly useful to determine the correct answer.
- A tool is NOT HELPFUL if its output is irrelevant to the question.
- You must evaluate EVERY tool and give a short reason (1 sentence) for each.

Respond ONLY with a JSON object (no markdown fences). For each tool, give an object with "helpful" (bool) and "reason" (string).
Example:
{"speech_recognition": {"helpful": true, "reason": "The question asks what was said, so transcription is needed."}, "speaker_diarization": {"helpful": false, "reason": "The question is not about identifying speakers."}, ...}
"""


def build_user_prompt(question, choices, answer):
    tools_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in sorted(TOOL_DESCRIPTIONS.items())
    )
    choices_str = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
    return f"""\
Question: {question}

Choices:
{choices_str}

Correct Answer: {answer}

Available Tools:
{tools_block}

Which tools are helpful for answering this question? For each tool give {{"helpful": bool, "reason": "..."}}. Return JSON only."""


def query_gemma(host, port, system_prompt, user_prompt, max_tokens=1200):
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


def parse_tool_verdict(raw_text):
    """Extract the JSON dict from Gemma's response (with reasoning)."""
    # Try direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    # Find outermost JSON block: first { to last }
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start:end+1])
        except json.JSONDecodeError:
            pass
    return None


def process_sample(sample, host, port):
    question = sample["question"]
    choices  = sample["choices"]
    answer   = sample["answer"]

    user_prompt = build_user_prompt(question, choices, answer)
    raw = query_gemma(host, port, SYSTEM_PROMPT, user_prompt)
    verdict = parse_tool_verdict(raw)

    # Build clean result
    result = {
        "id":       sample.get("id", ""),
        "question": question,
        "choices":  choices,
        "answer":   answer,
        "gold_tool": sample.get("tool", ""),
        "task":     sample.get("task", ""),
    }

    if verdict:
        helpful   = sorted([t for t, v in verdict.items()
                             if (v.get("helpful") if isinstance(v, dict) else v)])
        unhelpful = sorted([t for t, v in verdict.items()
                             if not (v.get("helpful") if isinstance(v, dict) else v)])
        # Build per-tool reasoning dict
        tool_reasoning = {}
        for t, v in verdict.items():
            if isinstance(v, dict):
                tool_reasoning[t] = {
                    "helpful": v.get("helpful", False),
                    "reason":  v.get("reason", ""),
                }
            else:
                tool_reasoning[t] = {"helpful": bool(v), "reason": ""}
        result["helpful_tools"]   = helpful
        result["unhelpful_tools"] = unhelpful
        result["tool_reasoning"]  = tool_reasoning
        result["num_helpful"]     = len(helpful)
        result["gold_tool_marked_helpful"] = sample.get("tool", "") in helpful or (
            # handle get_audio_features -> audio_features alias
            sample.get("tool", "") == "get_audio_features" and "audio_features" in helpful
        )
    else:
        result["parse_error"] = True
        result["raw_response"] = raw

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="synthetic_dataset/synth_cached_train.json")
    parser.add_argument("--output", default="synthetic_dataset/synth_tool_helpfulness_results.json")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel requests to vLLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N samples (for testing)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.limit:
        data = data[: args.limit]

    print(f"Processing {len(data)} samples with {args.workers} workers …")
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
            if done % 50 == 0 or done == len(data):
                print(f"  {done}/{len(data)} done")

    # ── Summary stats ─────────────────────────────────────────────────────────
    total = len(results)
    errors = sum(1 for r in results if "error" in r or r.get("parse_error"))
    valid  = [r for r in results if "helpful_tools" in r]
    gold_hit = sum(1 for r in valid if r["gold_tool_marked_helpful"])

    print(f"\n{'='*60}")
    print(f"Total samples:  {total}")
    print(f"Parse errors:   {errors}")
    print(f"Valid verdicts:  {len(valid)}")
    print(f"Gold tool marked helpful: {gold_hit}/{len(valid)} "
          f"({100*gold_hit/len(valid):.1f}%)" if valid else "")
    print()

    # Per-tool stats: how often each tool is marked helpful
    from collections import Counter
    helpful_counts = Counter()
    for r in valid:
        for t in r["helpful_tools"]:
            helpful_counts[t] += 1
    print("Tool helpfulness frequency (across all questions):")
    for t in TOOL_LIST:
        cnt = helpful_counts.get(t, 0)
        print(f"  {t:30s}  {cnt:4d}/{len(valid)}  ({100*cnt/len(valid):5.1f}%)")

    # Per gold-tool: how often the gold tool was actually marked helpful
    from collections import defaultdict
    gold_stats = defaultdict(lambda: {"total": 0, "hit": 0})
    for r in valid:
        gt = r["gold_tool"]
        gold_stats[gt]["total"] += 1
        if r["gold_tool_marked_helpful"]:
            gold_stats[gt]["hit"] += 1
    print("\nGold tool recall (was gold tool marked helpful?):")
    for t in sorted(gold_stats):
        s = gold_stats[t]
        print(f"  {t:30s}  {s['hit']:3d}/{s['total']:3d}  "
              f"({100*s['hit']/s['total']:5.1f}%)")

    # Negative-only samples (no tool can help)
    negatives = [r for r in valid if r["num_helpful"] == 0]
    print(f"\nSamples where NO tool was marked helpful: {len(negatives)}")

    # ── Write clean output ────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
