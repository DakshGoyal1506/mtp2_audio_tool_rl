"""
Quick throughput benchmark: 8K tokens × 64 requests (16×4 batches).
Run against an already-running SGLang server.

Usage:
    python grpo_single_phase_llm/benchmark/bench_throughput.py \
        --port 8000 --total-requests 64 --concurrency 16
"""

import argparse
import concurrent.futures
import json
import time

import requests

# ~600 token prompt; we'll ask the model to generate up to 7400 tokens
# to approximate 8K total tokens per request
JUDGE_PROMPT = """\
You are evaluating the reasoning quality of an AI assistant that answers audio-related multiple-choice questions.

## Completion to evaluate:
<think>
The question asks about the number of speakers in the audio. I can hear multiple voices speaking, but it's hard to count exact speakers just by listening. The audio contains what sounds like a panel discussion with several participants. I notice at least two distinct male voices and one female voice. There seems to be some overlap in the conversation making it harder to distinguish individual speakers. Let me consider the speaker_diarization tool to get a more precise count. Wait, actually, I think I can hear four distinct voices based on pitch, tone, and speaking patterns. The first voice is a deep male voice, the second is a higher-pitched male voice, the third is a female voice with a slight accent, and there might be a fourth speaker who speaks briefly near the end.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>
<tool_output>
{"speakers": [{"id": "SPEAKER_00", "start": 0.5, "end": 12.3}, {"id": "SPEAKER_01", "start": 3.2, "end": 15.7}, {"id": "SPEAKER_02", "start": 8.1, "end": 20.4}, {"id": "SPEAKER_03", "start": 18.2, "end": 25.0}]}
</tool_output>
<think>
The speaker diarization results confirm there are 4 distinct speakers in the audio. SPEAKER_00 speaks from 0.5 to 12.3 seconds, SPEAKER_01 from 3.2 to 15.7, SPEAKER_02 from 8.1 to 20.4, and SPEAKER_03 briefly from 18.2 to 25.0. This aligns with my initial observation of hearing four distinct voices with some overlap.
</think>
<answer>4</answer>

## Question:
How many speakers are present in the audio clip?

## Answer choices:
  - 2
  - 3
  - 4
  - 5

## Rate the THINKING QUALITY on a scale of 0-10:

Consider:
- Does the <think> block show substantive reasoning about the audio content?
- If a tool was called, does the post-tool <think> block correctly interpret the tool output?
- Is the reasoning logically coherent and relevant to the question?
- Does the assistant connect audio observations to the chosen answer?

Respond with ONLY a JSON object: {"coherence": <0-10>}"""


def single_request(api_url, model, prompt, max_tokens, req_id):
    """Send one request, return stats."""
    t0 = time.time()
    try:
        resp = requests.post(
            api_url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a precise scoring assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": 0.7,
            },
            timeout=600,
        )
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return {
            "req_id": req_id,
            "ok": True,
            "latency_s": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        return {
            "req_id": req_id,
            "ok": False,
            "latency_s": time.time() - t0,
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def run_batch(api_url, model, prompt, max_tokens, concurrency, total_requests):
    """Run total_requests with given concurrency, return results."""
    results = []
    batches = (total_requests + concurrency - 1) // concurrency

    print(f"\n{'='*60}")
    print(f"  Benchmark: {total_requests} requests, concurrency={concurrency}")
    print(f"  Max tokens per request: ~{max_tokens} completion")
    print(f"  Batches: {batches} (last may be partial)")
    print(f"{'='*60}\n")

    wall_start = time.time()
    completed = 0

    for batch_idx in range(batches):
        batch_start = completed
        batch_end = min(completed + concurrency, total_requests)
        batch_size = batch_end - batch_start

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(single_request, api_url, model, prompt, max_tokens, i)
                for i in range(batch_start, batch_end)
            ]
            batch_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        t1 = time.time()

        results.extend(batch_results)
        completed = batch_end

        ok_count = sum(1 for r in batch_results if r["ok"])
        batch_tokens = sum(r["total_tokens"] for r in batch_results)
        batch_comp_tokens = sum(r["completion_tokens"] for r in batch_results)

        print(f"  Batch {batch_idx+1}/{batches}: "
              f"{ok_count}/{batch_size} ok, "
              f"{batch_tokens} total tok, "
              f"{batch_comp_tokens} gen tok, "
              f"{t1-t0:.1f}s wall, "
              f"{batch_tokens/(t1-t0):.0f} tok/s")

    wall_end = time.time()
    return results, wall_end - wall_start


def print_summary(results, wall_time, concurrency):
    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]

    total_prompt = sum(r["prompt_tokens"] for r in ok_results)
    total_comp = sum(r["completion_tokens"] for r in ok_results)
    total_tok = sum(r["total_tokens"] for r in ok_results)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Requests:      {len(results)} total, {len(ok_results)} ok, {len(fail_results)} failed")
    print(f"  Concurrency:   {concurrency}")
    print(f"  Wall time:     {wall_time:.1f}s")
    print()
    if ok_results:
        avg_lat = sum(r["latency_s"] for r in ok_results) / len(ok_results)
        p50 = sorted(r["latency_s"] for r in ok_results)[len(ok_results)//2]
        p99 = sorted(r["latency_s"] for r in ok_results)[int(len(ok_results)*0.99)]
        avg_prompt = total_prompt / len(ok_results)
        avg_comp = total_comp / len(ok_results)

        print(f"  Avg prompt tokens:     {avg_prompt:.0f}")
        print(f"  Avg completion tokens: {avg_comp:.0f}")
        print(f"  Avg total tokens:      {avg_prompt+avg_comp:.0f}")
        print()
        print(f"  Avg latency:     {avg_lat:.2f}s")
        print(f"  P50 latency:     {p50:.2f}s")
        print(f"  P99 latency:     {p99:.2f}s")
        print()
        print(f"  Total tokens:        {total_tok}")
        print(f"  Total gen tokens:    {total_comp}")
        print(f"  Throughput (total):  {total_tok/wall_time:.1f} tok/s")
        print(f"  Throughput (gen):    {total_comp/wall_time:.1f} tok/s")
        print(f"  Requests/s:          {len(ok_results)/wall_time:.2f}")

    if fail_results:
        print(f"\n  FAILURES ({len(fail_results)}):")
        for r in fail_results[:5]:
            print(f"    req {r['req_id']}: {r.get('error','unknown')}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--total-requests", type=int, default=64,
                        help="Total requests (default 64 = 16x4)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Concurrent requests per batch")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max completion tokens per request")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # Health check
    print(f"Checking server at {args.host}:{args.port}...")
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        print(f"  Health: HTTP {r.status_code}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        print("  Proceeding anyway...\n")

    # Warmup
    print("Warmup (2 requests)...")
    for i in range(2):
        res = single_request(api_url, args.model, JUDGE_PROMPT, args.max_tokens, f"warmup-{i}")
        status = "ok" if res["ok"] else f"FAIL: {res.get('error')}"
        print(f"  warmup-{i}: {status}, {res['total_tokens']} tok, {res['latency_s']:.2f}s")

    # Run benchmark
    results, wall_time = run_batch(
        api_url, args.model, JUDGE_PROMPT,
        args.max_tokens, args.concurrency, args.total_requests,
    )

    print_summary(results, wall_time, args.concurrency)


if __name__ == "__main__":
    main()
