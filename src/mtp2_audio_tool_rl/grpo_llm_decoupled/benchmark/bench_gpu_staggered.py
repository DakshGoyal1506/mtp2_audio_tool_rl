"""
Simulate realistic GRPO judge workload:
  - 4 GPUs, each with G=16 × batch=8 = 128 completions = 256 judge calls
  - GPUs finish generation at staggered times (0s, 2s, 5s, 8s offsets)
  - Each GPU's 256 requests arrive as one concurrent batch
  - ~2K input tokens (worst case), 64 max completion tokens

Usage:
    python bench_gpu_staggered.py --port 8000 --model "Qwen/Qwen3.5-27B"
"""

import argparse
import concurrent.futures
import threading
import time

import requests

# ~2K token prompt (padded with realistic judge content)
PROMPT_2K = """\
You are evaluating the reasoning quality of an AI assistant that answers audio-related multiple-choice questions.

## Completion to evaluate:
<think>
The question asks about the number of speakers in the audio. I can hear multiple voices speaking, but it's hard to count exact speakers just by listening. The audio contains what sounds like a panel discussion with several participants. I notice at least two distinct male voices and one female voice. There seems to be some overlap in the conversation making it harder to distinguish individual speakers. Let me consider the speaker_diarization tool to get a more precise count. Wait, actually, I think I can hear four distinct voices based on pitch, tone, and speaking patterns. The first voice is a deep male voice, the second is a higher-pitched male voice, the third is a female voice with a slight accent, and there might be a fourth speaker who speaks briefly near the end.

Let me analyze this more carefully. The first speaker has a deep baritone voice and speaks with a measured pace, discussing what appears to be economic policy. The second speaker interrupts occasionally with a higher-pitched voice, offering counterarguments. The third speaker, a woman, mediates between the two and introduces new topics. There's definitely a fourth voice that appears near timestamp 18 seconds - younger sounding, possibly a student asking a question.

The audio quality is relatively clear despite some background noise. I can hear ambient room noise suggesting this is recorded in a large hall or auditorium. The microphone seems to be positioned centrally, capturing all speakers roughly equally. There's a brief moment of feedback around the 10-second mark but it doesn't significantly impact comprehension.

Considering the acoustic properties - the reverb pattern suggests a medium-sized room, possibly a conference room or small lecture hall. The speakers seem to be seated relatively close together based on the similar volume levels. The fourth speaker's voice comes from slightly further away, consistent with an audience member using a separate microphone.

I should use the speaker diarization tool to confirm my analysis. This will give precise timestamps for each speaker segment and help verify the total speaker count. My initial estimate of 4 speakers seems reasonable based on the distinct vocal characteristics I've identified, but tool verification would provide more confidence.

Additionally, I notice some non-speech audio events: what sounds like papers shuffling around 7 seconds, a glass being set down at 14 seconds, and general audience murmur in the background. These shouldn't be counted as separate speakers but are worth noting for completeness.

The prosodic patterns also support 4 speakers - each has a distinct intonation pattern. Speaker 1 uses falling intonation at statement ends, Speaker 2 often uses rising intonation suggesting questions or challenges, Speaker 3 has a distinctive rhythm with longer pauses between phrases, and Speaker 4's brief contribution has the typical rising intonation of someone asking a question.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>
<tool_output>
{"speakers": [{"id": "SPEAKER_00", "start": 0.5, "end": 12.3, "segments": [{"start": 0.5, "end": 3.1}, {"start": 5.2, "end": 8.4}, {"start": 10.1, "end": 12.3}]}, {"id": "SPEAKER_01", "start": 3.2, "end": 15.7, "segments": [{"start": 3.2, "end": 5.0}, {"start": 8.5, "end": 10.0}, {"start": 13.1, "end": 15.7}]}, {"id": "SPEAKER_02", "start": 8.1, "end": 20.4, "segments": [{"start": 8.1, "end": 8.4}, {"start": 12.4, "end": 16.2}, {"start": 17.0, "end": 20.4}]}, {"id": "SPEAKER_03", "start": 18.2, "end": 25.0, "segments": [{"start": 18.2, "end": 19.5}, {"start": 21.0, "end": 25.0}]}], "total_speakers": 4, "total_duration": 25.0, "overlap_ratio": 0.12}
</tool_output>
<think>
The speaker diarization results confirm there are 4 distinct speakers in the audio. SPEAKER_00 speaks from 0.5 to 12.3 seconds across three segments, SPEAKER_01 from 3.2 to 15.7 across three segments, SPEAKER_02 from 8.1 to 20.4 across three segments, and SPEAKER_03 from 18.2 to 25.0 across two segments. The overlap ratio of 0.12 indicates some simultaneous speech but not excessive. This aligns perfectly with my initial observation of hearing four distinct voices with some overlap. The segmented timestamps also match my perception of turn-taking patterns in the discussion. SPEAKER_00 and SPEAKER_01 alternate frequently in the first half, while SPEAKER_02 becomes more prominent in the middle section, and SPEAKER_03 contributes mainly toward the end. The total duration of 25 seconds matches the audio length I perceived.
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
    t0 = time.time()
    try:
        resp = requests.post(
            api_url,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a precise scoring assistant. Respond only with JSON."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_tokens,
                "temperature": 0.0,
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


def gpu_worker(gpu_id, delay_s, num_calls, api_url, model, prompt, max_tokens):
    """Simulate one GPU: wait delay_s, then fire all judge calls concurrently."""
    time.sleep(delay_s)
    t0 = time.time()
    print(f"  [GPU {gpu_id}] Starting {num_calls} concurrent judge calls at +{delay_s}s offset")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_calls) as pool:
        futures = [
            pool.submit(single_request, api_url, model, prompt, max_tokens,
                        f"gpu{gpu_id}-{i}")
            for i in range(num_calls)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    t1 = time.time()
    ok = sum(1 for r in results if r["ok"])
    total_tok = sum(r["total_tokens"] for r in results)
    gen_tok = sum(r["completion_tokens"] for r in results)
    avg_lat = sum(r["latency_s"] for r in results) / len(results) if results else 0

    print(f"  [GPU {gpu_id}] Done: {ok}/{num_calls} ok, "
          f"{total_tok} total tok, {gen_tok} gen tok, "
          f"wall={t1-t0:.1f}s, avg_lat={avg_lat:.1f}s, "
          f"{total_tok/(t1-t0):.0f} tok/s")

    return {
        "gpu_id": gpu_id,
        "delay_s": delay_s,
        "wall_time": t1 - t0,
        "results": results,
        "finish_time": t1,
        "start_time": t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--G", type=int, default=16, help="Generations per prompt")
    parser.add_argument("--scores-per-completion", type=int, default=2,
                        help="Judge calls per completion (coherence + tool)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-delays", type=str, default="0,2,5,8",
                        help="Staggered arrival delays in seconds for each GPU")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    delays = [float(x) for x in args.gpu_delays.split(",")]
    calls_per_gpu = args.batch_size * args.G * args.scores_per_completion

    # Pad delays if fewer than num_gpus
    while len(delays) < args.num_gpus:
        delays.append(delays[-1] + 3)

    print("=" * 65)
    print("  GRPO Judge Staggered GPU Benchmark")
    print("=" * 65)
    print(f"  Model:              {args.model}")
    print(f"  GPUs:               {args.num_gpus}")
    print(f"  batch_size:         {args.batch_size}")
    print(f"  G (generations):    {args.G}")
    print(f"  Completions/GPU:    {args.batch_size * args.G}")
    print(f"  Scores/completion:  {args.scores_per_completion}")
    print(f"  Judge calls/GPU:    {calls_per_gpu}")
    print(f"  Total judge calls:  {calls_per_gpu * args.num_gpus}")
    print(f"  GPU arrival delays: {delays[:args.num_gpus]}s")
    print(f"  Max tokens:         {args.max_tokens}")
    print(f"  Input: ~2K tokens (worst case)")
    print("=" * 65)

    # Health check
    print(f"\nChecking server at {args.host}:{args.port}...")
    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
        print(f"  Health: HTTP {r.status_code}")
    except Exception as e:
        print(f"  Health check failed: {e}")

    # Warmup
    print("\nWarmup (2 requests)...")
    for i in range(2):
        res = single_request(api_url, args.model, PROMPT_2K, args.max_tokens, f"warmup-{i}")
        status = "ok" if res["ok"] else f"FAIL: {res.get('error')}"
        print(f"  warmup-{i}: {status}, "
              f"prompt={res['prompt_tokens']} tok, "
              f"comp={res['completion_tokens']} tok, "
              f"{res['latency_s']:.2f}s")

    # Launch all GPUs in parallel threads with staggered delays
    print(f"\n--- Launching {args.num_gpus} GPU workers (staggered) ---\n")
    wall_start = time.time()

    gpu_futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        for gpu_id in range(args.num_gpus):
            f = pool.submit(
                gpu_worker, gpu_id, delays[gpu_id], calls_per_gpu,
                api_url, args.model, PROMPT_2K, args.max_tokens,
            )
            gpu_futures.append(f)

        gpu_results = [f.result() for f in gpu_futures]

    wall_end = time.time()
    total_wall = wall_end - wall_start

    # Summary
    all_results = []
    for gr in gpu_results:
        all_results.extend(gr["results"])

    ok_results = [r for r in all_results if r["ok"]]
    fail_results = [r for r in all_results if not r["ok"]]
    total_tok = sum(r["total_tokens"] for r in ok_results)
    total_gen = sum(r["completion_tokens"] for r in ok_results)
    total_prompt = sum(r["prompt_tokens"] for r in ok_results)

    print(f"\n{'='*65}")
    print(f"  STAGGERED GPU RESULTS")
    print(f"{'='*65}")
    print(f"  Total requests:  {len(all_results)} ({len(ok_results)} ok, {len(fail_results)} failed)")
    print(f"  Total wall time: {total_wall:.1f}s ({total_wall/60:.1f}min)")
    print()

    for gr in gpu_results:
        gpu_ok = [r for r in gr["results"] if r["ok"]]
        gpu_tok = sum(r["total_tokens"] for r in gpu_ok)
        gpu_gen = sum(r["completion_tokens"] for r in gpu_ok)
        gpu_lats = [r["latency_s"] for r in gpu_ok]
        avg_lat = sum(gpu_lats) / len(gpu_lats) if gpu_lats else 0
        p50 = sorted(gpu_lats)[len(gpu_lats)//2] if gpu_lats else 0
        p99 = sorted(gpu_lats)[int(len(gpu_lats)*0.99)] if gpu_lats else 0
        print(f"  GPU {gr['gpu_id']} (delay={gr['delay_s']}s):")
        print(f"    Wall time:    {gr['wall_time']:.1f}s")
        print(f"    Requests:     {len(gpu_ok)}/{len(gr['results'])}")
        print(f"    Total tokens: {gpu_tok} (gen: {gpu_gen})")
        print(f"    Throughput:   {gpu_tok/gr['wall_time']:.0f} tok/s")
        print(f"    Avg latency:  {avg_lat:.1f}s  |  P50: {p50:.1f}s  |  P99: {p99:.1f}s")
        print()

    if ok_results:
        avg_prompt = total_prompt / len(ok_results)
        avg_comp = total_gen / len(ok_results)
        all_lats = sorted(r["latency_s"] for r in ok_results)
        overall_avg_lat = sum(all_lats) / len(all_lats)
        overall_p50 = all_lats[len(all_lats)//2]
        overall_p99 = all_lats[int(len(all_lats)*0.99)]

        # DDP sync point = time from wall_start to last GPU finishing
        last_gpu_finish = max(gr["finish_time"] for gr in gpu_results)
        ddp_sync_time = last_gpu_finish - wall_start

        print(f"  --- Aggregate ---")
        print(f"  Avg prompt tokens:     {avg_prompt:.0f}")
        print(f"  Avg completion tokens: {avg_comp:.0f}")
        print(f"  Overall avg latency:   {overall_avg_lat:.1f}s")
        print(f"  Overall P50 latency:   {overall_p50:.1f}s")
        print(f"  Overall P99 latency:   {overall_p99:.1f}s")
        print(f"  Aggregate throughput:  {total_tok/total_wall:.0f} tok/s")
        print(f"  Aggregate gen tok/s:   {total_gen/total_wall:.0f} tok/s")
        print()
        print(f"  --- Training Step Impact ---")
        print(f"  DDP sync time (last GPU done):  {ddp_sync_time:.1f}s ({ddp_sync_time/60:.1f}min)")
        print(f"  This is the ACTUAL judge overhead per training step")

    if fail_results:
        print(f"\n  FAILURES ({len(fail_results)}):")
        for r in fail_results[:5]:
            print(f"    {r['req_id']}: {r.get('error','?')}")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
