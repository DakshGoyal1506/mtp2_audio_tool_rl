"""
MMAU inference with structured output via vLLM json_schema.

Two-step pipeline per sample:
  1. Tool selection: model picks a tool (or NONE) from enum + reasoning
  2. Final answer: model picks answer from choices enum + reasoning

If tool != NONE, injects cached tool_output into step 2 context.

Usage:
    python mmau_structured_infer.py \
        --host 192.168.1.44 --port 8000 \
        --input mmau-test-mini-cached.json \
        --output mmau-structured-results.json \
        --audio-dir ./test-mini-audios \
        --model af3-think
"""

import argparse
import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


TOOLS_ENUM = [
    "speech_recognition", "speaker_diarization", "emotion_recognition",
    "sound_classification", "get_audio_features", "sound_duration_analysis",
    "speech_to_noise_ratio", "chord_recognition", "genre_analysis",
    "stress_analysis", "NONE"
]

TOOL_DESCRIPTIONS = """Tools:
speech_recognition=transcribe speech | speaker_diarization=who speaks when, speaker count | emotion_recognition=detect emotions | sound_classification=classify non-speech sounds | get_audio_features=pitch/tempo/loudness | sound_duration_analysis=timing of events | speech_to_noise_ratio=SNR | chord_recognition=musical chords | genre_analysis=music genre | stress_analysis=stressed phonemes | NONE=no tool needed"""

TOOL_SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool": {"type": "string", "enum": TOOLS_ENUM}
    },
    "required": ["reasoning", "tool"]
}


def get_model_name(host, port):
    url = f"http://{host}:{port}/v1/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    return data["data"][0]["id"]


def encode_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_api(host, port, payload, timeout=120):
    url = f"http://{host}:{port}/v1/chat/completions"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def infer_sample(sample, host, port, model_name, audio_dir, max_retries=3):
    """Run structured 2-step inference on a single sample."""
    result = dict(sample)
    if "answer" in result and "gold_answer" not in result:
        result["gold_answer"] = result["answer"]

    # Resolve audio
    audio_id = sample["audio_id"]
    if audio_id.startswith("./"):
        audio_id = audio_id[2:]
    audio_path = os.path.join(audio_dir, os.path.basename(audio_id))

    if not os.path.isfile(audio_path):
        result["model_prediction"] = ""
        result["error"] = f"Audio not found: {audio_path}"
        return result

    audio_b64 = encode_audio(audio_path)
    question = sample["question"]
    choices = sample["choices"]

    audio_content = {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}

    # --- Step 1: Tool selection ---
    tool_msg = [{
        "role": "user",
        "content": [
            audio_content,
            {"type": "text", "text": (
                f"Question: {question}\n"
                f"Options: {' | '.join(choices)}\n\n"
                f"{TOOL_DESCRIPTIONS}\n\n"
                "Pick a tool ONLY if needed. Pick NONE if you can answer by listening alone.\n"
                "Please think and reason about the input audio before you respond."
            )},
        ]
    }]

    tool_payload = {
        "model": model_name,
        "messages": tool_msg,
        "max_tokens": 256,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "tool_select", "schema": TOOL_SELECT_SCHEMA}
        },
    }

    tool_choice = "NONE"
    tool_reasoning = ""
    for attempt in range(max_retries):
        try:
            resp = call_api(host, port, tool_payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            tool_choice = parsed.get("tool", "NONE")
            tool_reasoning = parsed.get("reasoning", "")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                result["tool_select_error"] = str(e)
            else:
                time.sleep(1)

    result["tool_selected"] = tool_choice
    result["tool_reasoning"] = tool_reasoning

    # --- Step 2: Answer (with tool output if tool was selected) ---
    answer_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "answer": {"type": "string", "enum": choices}
        },
        "required": ["reasoning", "answer"]
    }

    # Build context for step 2
    context_text = (
        f"{'='*40}\n"
        f"QUESTION: {question}\n"
        f"OPTIONS: {' | '.join(choices)}\n"
        f"{'='*40}\n\n"
    )

    # Pass step 1 reasoning
    if tool_reasoning:
        context_text += (
            f"--- Your Initial Analysis ---\n"
            f"{tool_reasoning}\n"
            f"-----------------------------\n\n"
        )

    # Inject tool output if tool was selected and available
    if tool_choice != "NONE" and "tool_outputs" in sample:
        tool_out = sample["tool_outputs"].get(tool_choice)
        if tool_out:
            tool_out_str = json.dumps(tool_out, indent=2)
            context_text += (
                f"--- Tool Result: {tool_choice} ---\n"
                f"{tool_out_str}\n"
                f"{'-'*35}\n\n"
            )

    context_text += (
        "Based on all the above, pick the correct answer. "
        "Tool output may be wrong — trust your own listening."
    )

    answer_msg = [{
        "role": "user",
        "content": [
            audio_content,
            {"type": "text", "text": context_text},
        ]
    }]

    answer_payload = {
        "model": model_name,
        "messages": answer_msg,
        "max_tokens": 256,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer_select", "schema": answer_schema}
        },
    }

    for attempt in range(max_retries):
        try:
            resp = call_api(host, port, answer_payload)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            result["model_prediction"] = parsed.get("answer", "")
            result["answer_reasoning"] = parsed.get("reasoning", "")
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                result["model_prediction"] = ""
                result["error"] = str(e)
            else:
                time.sleep(1)

    return result


def main():
    parser = argparse.ArgumentParser(description="MMAU structured inference")
    parser.add_argument("--input", default="mmau-test-mini-cached.json")
    parser.add_argument("--output", default="mmau-structured-results.json")
    parser.add_argument("--host", default="192.168.1.44")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--audio-dir", default="./test-mini-audios")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default="af3-think")
    args = parser.parse_args()

    with open(args.input) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {args.input}")
    print(f"Model: {args.model} @ {args.host}:{args.port}")

    results = [None] * len(samples)
    correct = 0
    tool_used = 0
    done = 0
    total = len(samples)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for idx, sample in enumerate(samples):
            fut = pool.submit(infer_sample, sample, args.host, args.port,
                              args.model, args.audio_dir)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            res = fut.result()
            results[idx] = res
            done += 1
            if res.get("tool_selected", "NONE") != "NONE":
                tool_used += 1
            if res.get("model_prediction", "").lower() == res.get("gold_answer", "").lower():
                correct += 1
            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] acc={correct}/{done} "
                      f"({100*correct/done:.1f}%) tools={tool_used}", flush=True)

    n_valid = sum(1 for r in results if r and "error" not in r)
    print(f"\nResults: {correct}/{n_valid} correct ({100*correct/max(n_valid,1):.1f}%)")
    print(f"Tool used: {tool_used}/{n_valid} ({100*tool_used/max(n_valid,1):.1f}%)")

    # Tool distribution
    from collections import Counter
    tool_dist = Counter(r.get("tool_selected", "NONE") for r in results if r)
    print("\nTool distribution:")
    for tool, cnt in tool_dist.most_common():
        print(f"  {tool}: {cnt}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
