"""
Realistic GRPO judge benchmark — simulates 3 GPUs finishing at staggered times,
each dumping their completions as a concurrent batch onto a single judge server.

Per GPU: batch_size × num_generations = 8 × 16 = 128 completions → 128 judge calls
3 GPUs arrive with offsets (0s, 2s, 5s) to simulate real DDP stagger.
1 judge call per completion (combined scoring).
~2K prompt tokens (worst case), ~1K max completion tokens.
"""

import argparse
import concurrent.futures
import threading
import time
import requests

# ~4K token prompt (worst case)
COMBINED_JUDGE_PROMPT = """\
You are evaluating the quality of an AI assistant that answers audio-related multiple-choice questions.
The assistant can optionally call one audio analysis tool to help answer. The available tools are:
- speaker_diarization: Identifies distinct speakers and their time segments in the audio
- sound_classification: Classifies sounds/instruments/events in the audio
- speech_recognition: Transcribes spoken content from the audio
- audio_tagging: Tags audio with high-level labels (music, speech, noise, etc.)
- emotion_recognition: Detects emotional tone from speech patterns

Each tool returns structured JSON output. The assistant should choose the most appropriate tool for the question type, or skip tool use if the question can be answered from direct audio perception alone.

## Audio context:
- Duration: 25.0 seconds
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Format: WAV
- Audio description: A panel discussion recording from a conference room with moderate background noise including occasional paper rustling and distant HVAC system hum. The recording quality is generally clear but some overlapping speech segments reduce intelligibility.

## Completion to evaluate:
<think>
The question asks about the number of speakers in the audio. I can hear multiple voices speaking, but it's hard to count exact speakers just by listening. The audio contains what sounds like a panel discussion with several participants. I notice at least two distinct male voices and one female voice. There seems to be some overlap in the conversation making it harder to distinguish individual speakers. Let me consider the speaker_diarization tool to get a more precise count. Wait, actually, I think I can hear four distinct voices based on pitch, tone, and speaking patterns. The first voice is a deep male voice, the second is a higher-pitched male voice, the third is a female voice with a slight accent, and there might be a fourth speaker who speaks briefly near the end. This is a complex scenario that requires careful analysis of the audio characteristics.

Let me also consider the acoustic environment. The recording appears to be from a conference room setting based on the slight reverberation I can detect. There's some background noise that could potentially mask quieter speakers. The panel discussion format typically involves a moderator plus several panelists, which would be consistent with 3-5 speakers. Given the overlapping speech patterns I'm hearing, especially around the 8-12 second mark, it seems like multiple people are trying to speak simultaneously, which further complicates manual speaker counting.

I should use the speaker_diarization tool here because: (1) counting speakers precisely by ear is unreliable especially with overlapping speech, (2) the overlapping segments around 8-12 seconds make it particularly difficult to distinguish speakers auditorily, (3) diarization can identify speakers even when they speak briefly or quietly, and (4) the background noise adds additional complexity that algorithmic analysis handles better than human perception in edge cases.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>
<tool_output>
{"speakers": [{"id": "SPEAKER_00", "start": 0.5, "end": 12.3, "confidence": 0.94, "segments": [{"start": 0.5, "end": 5.2, "text": "Welcome everyone to today's panel discussion on climate change policy and its implications for the next decade of environmental governance", "confidence": 0.96}, {"start": 8.1, "end": 12.3, "text": "Let me start by asking our first panelist about their research findings on carbon emission trends in developing nations", "confidence": 0.93}], "speaker_embedding_distance": 0.12, "gender_estimate": "male", "pitch_range_hz": [95, 180]}, {"id": "SPEAKER_01", "start": 3.2, "end": 15.7, "confidence": 0.91, "segments": [{"start": 3.2, "end": 7.8, "text": "Thank you for having me. Our latest study published in Nature Climate Change shows a dramatic increase in industrial emissions across Southeast Asian manufacturing sectors", "confidence": 0.89}, {"start": 13.1, "end": 15.7, "text": "The data clearly supports intervention at the policy level, particularly through carbon pricing mechanisms and cap-and-trade systems", "confidence": 0.92}], "speaker_embedding_distance": 0.15, "gender_estimate": "male", "pitch_range_hz": [110, 195]}, {"id": "SPEAKER_02", "start": 8.1, "end": 20.4, "confidence": 0.88, "segments": [{"start": 8.1, "end": 11.5, "text": "I'd like to add that from an economic perspective, the cost-benefit analysis of immediate action versus delayed response shows a clear advantage for early intervention", "confidence": 0.87}, {"start": 16.2, "end": 20.4, "text": "We need to balance environmental concerns with economic growth, especially for nations still in the early stages of industrialization where emissions per capita remain low", "confidence": 0.90}], "speaker_embedding_distance": 0.18, "gender_estimate": "female", "pitch_range_hz": [165, 280]}, {"id": "SPEAKER_03", "start": 18.2, "end": 25.0, "confidence": 0.86, "segments": [{"start": 18.2, "end": 22.1, "text": "As someone working in renewable energy deployment across sub-Saharan Africa, I can tell you that the transition is not just environmentally necessary but economically viable", "confidence": 0.85}, {"start": 23.0, "end": 25.0, "text": "The transition is already underway and solar costs have dropped 89 percent since 2010", "confidence": 0.88}], "speaker_embedding_distance": 0.20, "gender_estimate": "male", "pitch_range_hz": [100, 170]}], "overlap_segments": [{"start": 3.2, "end": 5.2, "speakers": ["SPEAKER_00", "SPEAKER_01"], "type": "transition"}, {"start": 8.1, "end": 11.5, "speakers": ["SPEAKER_00", "SPEAKER_02"], "type": "interjection"}, {"start": 18.2, "end": 20.4, "speakers": ["SPEAKER_02", "SPEAKER_03"], "type": "transition"}], "total_speech_duration": 22.8, "total_silence_duration": 2.2, "num_speakers_detected": 4, "diarization_model": "pyannote/speaker-diarization-3.1", "processing_time_seconds": 1.23}
</tool_output>
<think>
The speaker diarization results confirm there are 4 distinct speakers in the audio. Let me analyze each speaker in detail:

SPEAKER_00 (confidence: 0.94) speaks from 0.5 to 12.3 seconds and appears to be the moderator based on welcoming remarks ("Welcome everyone to today's panel discussion") and question framing ("Let me start by asking our first panelist"). The speaker has a male voice with pitch range 95-180 Hz and low embedding distance (0.12), indicating consistent voice characteristics throughout their segments.

SPEAKER_01 (confidence: 0.91) speaks from 3.2 to 15.7 seconds and seems to be a researcher discussing study findings published in Nature Climate Change. They discuss carbon emission trends and policy interventions. Male voice with pitch range 110-195 Hz. There is an overlap with SPEAKER_00 from 3.2-5.2s during a transition period, which is natural in panel discussions.

SPEAKER_02 (confidence: 0.88) speaks from 8.1 to 20.4 seconds and brings an economic perspective, discussing cost-benefit analysis and balancing environmental concerns with economic growth. Female voice with pitch range 165-280 Hz. She interjects during SPEAKER_00's remarks at 8.1-11.5s and overlaps with SPEAKER_03 near the end.

SPEAKER_03 (confidence: 0.86) speaks briefly from 18.2 to 25.0 seconds and mentions renewable energy deployment in sub-Saharan Africa, citing that solar costs have dropped 89% since 2010. Male voice with pitch range 100-170 Hz. This speaker has the lowest confidence score, possibly due to the brief speaking duration and overlap with SPEAKER_02.

The diarization reveals three overlap segments, all classified as either "transition" or "interjection" types, which are common in panel discussions. The total speech duration is 22.8s out of 25s, with only 2.2s of silence, indicating an active discussion.

This aligns with my initial observation of hearing four distinct voices with some overlap. The tool was appropriate for this question since exact speaker counting benefits from diarization analysis rather than relying solely on auditory perception. The diarization also reveals the conversational dynamics with overlapping speech segments, confirming this is indeed a multi-speaker panel discussion format. The tool choice was essential here — while I initially guessed 4 speakers, the overlapping segments at 3.2-5.2s, 8.1-11.5s, and 18.2-20.4s could have led to miscounting without algorithmic analysis. The confidence scores (0.86-0.94) are all above typical thresholds, lending reliability to the 4-speaker conclusion.
</think>
<answer>4</answer>

## Question:
How many speakers are present in the audio clip?

## Answer choices:
  - 2
  - 3
  - 4
  - 5

## Gold (correct) answer:
4

## Tool called:
speaker_diarization

## Tool was available from:
[speaker_diarization, sound_classification, speech_recognition, audio_tagging, emotion_recognition]

## Assistant's final answer:
4

## Previous completions from this batch (for context on consistency):
- Completion 1: Used speaker_diarization, answered 4 (correct), coherence=8, tool_quality=9. Reasoning: "I can hear what sounds like a panel discussion. Using speaker_diarization to get precise count." Post-tool: "Confirmed 4 speakers with overlapping segments at transitions." Tool choice was appropriate for speaker counting task. Strong pre-tool hypothesis matched tool output. Good use of confidence scores in interpretation.
- Completion 2: No tool used, answered 3 (incorrect), coherence=4, tool_quality=2. Reasoning: "I hear three voices - two male and one female." Failed to detect SPEAKER_03 who speaks briefly at the end. Should have used speaker_diarization given the overlapping speech segments that make manual counting unreliable. The reasoning was superficial and didn't consider that brief speakers might be missed by ear.
- Completion 3: Used speech_recognition, answered 4 (correct), coherence=6, tool_quality=5. Reasoning: "Let me transcribe the audio to count speakers." Chose speech_recognition instead of speaker_diarization. While the transcription revealed content from 4 different contexts, the tool was suboptimal for speaker counting. The reasoning correctly inferred 4 speakers from distinct content topics but the approach was indirect and less reliable.
- Completion 4: Used speaker_diarization, answered 4 (correct), coherence=9, tool_quality=10. Reasoning: "Multiple overlapping voices detected. Speaker counting requires diarization for precision." Excellent pre-tool analysis noting overlapping segments at 8-12s mark. Post-tool reasoning analyzed each speaker's role and cross-referenced overlap patterns with initial observations. Best completion in this batch.
- Completion 5: Used audio_tagging, answered 2 (incorrect), coherence=2, tool_quality=1. Reasoning: "Let me tag the audio to find speakers." Used completely wrong tool - audio_tagging returns high-level labels not speaker information. The reasoning showed fundamental misunderstanding of available tools. Final answer of 2 was incorrect and not supported by tool output.
- Completion 6: No tool used, answered 4 (correct), coherence=7, tool_quality=6. Reasoning: "I can distinguish four voices: a moderator with a deep voice, a researcher with higher pitch, a female economist, and a brief speaker at the end discussing renewables." Good auditory analysis but could have been more certain with tool verification. The reasoning was solid even without tool use, showing strong audio perception skills.
- Completion 7: Used speaker_diarization, answered 5 (incorrect), coherence=3, tool_quality=7. Reasoning: "Multiple speakers detected, using diarization." Despite correct tool choice, misinterpreted the output by counting an overlap segment as a separate speaker. The post-tool reasoning failed to properly analyze the overlap_segments field which would have clarified the actual speaker count.

## Scoring rubric examples for calibration:
- A completion that uses the right tool, shows good pre-tool reasoning, correctly interprets output, and reaches the right answer should score coherence=8-9, tool_quality=8-9
- A completion that skips tool use but still reasons well about audio content and gets the right answer deserves coherence=7-8, tool_quality=5-6 (tool would have added confidence)
- A completion that uses the wrong tool or misinterprets tool output, even if getting the right answer by luck, should have tool_quality=3-5 and coherence=4-6
- A completion that shows no reasoning or copies the question back as reasoning should score coherence=0-2 regardless of correctness
- A completion that calls a tool unnecessarily (e.g., speaker_diarization for "what language is spoken?") should have tool_quality=2-4 even if reasoning is otherwise good

## Additional evaluation guidelines:
- Pre-tool reasoning quality: The <think> block before any tool call should demonstrate genuine engagement with the audio content. The assistant should describe what it actually perceives (voices, sounds, patterns) rather than generic statements. Look for specific observations like pitch differences, accent detection, temporal patterns, or background noise characterization.
- Tool parameter correctness: Verify that tool parameters make sense for the task. For speaker_diarization, the audio_path parameter should reference the provided audio. For sound_classification, check if confidence thresholds are appropriate.
- Post-tool synthesis: After receiving tool output, the assistant should integrate algorithmic results with its initial perceptual observations. A high-quality completion cross-references tool output with pre-tool hypotheses and notes confirmations or surprises. The assistant should also acknowledge limitations in both its own perception and the tool's output (e.g., low confidence scores, short speaker segments).
- Answer consistency: The final <answer> tag must be consistent with the reasoning. If the reasoning clearly supports one answer but the <answer> tag contains a different one, coherence should be penalized significantly (subtract 3-4 points).

## Rate the overall completion quality as a JSON object with TWO scores (0-10 each):

1. **coherence**: Quality of <think> reasoning - is it substantive, logically coherent, correctly interprets tool output, connects observations to the answer?
   - 9-10: Exceptional reasoning with detailed audio observations, correct tool interpretation, clear logical chain
   - 7-8: Good reasoning with some audio observations, mostly correct interpretation
   - 5-6: Adequate reasoning but shallow observations or minor interpretation errors
   - 3-4: Weak reasoning with significant gaps or errors in logic
   - 1-2: Minimal or nonsensical reasoning
   - 0: No reasoning provided or completely irrelevant

2. **tool_quality**: Was the tool necessary? Was the right tool chosen? If no tool, should one have been used?
   - 9-10: Perfect tool choice, clearly necessary, correct parameters, output well-utilized
   - 7-8: Good tool choice, mostly necessary, reasonable parameters
   - 5-6: Acceptable tool choice but questionable necessity or suboptimal tool selection
   - 3-4: Poor tool choice, wrong tool for the task, or unnecessary tool use
   - 1-2: Actively harmful tool choice that confused the reasoning
   - 0: Tool should have been used but wasn't, or used a nonexistent tool

Consider:
- Does the <think> block show substantive reasoning about the audio content before calling a tool?
- If a tool was called, does the post-tool <think> block correctly interpret ALL fields of the tool output?
- Is the reasoning logically coherent and relevant to the question?
- Does the assistant connect audio observations to the chosen answer with explicit evidence?
- Was calling this tool NECESSARY? Could the question be answered just by listening?
- Was the RIGHT tool selected for this question type? (e.g., speaker_diarization for counting speakers, sound_classification for instrument identification)
- Did the assistant handle overlap segments and confidence scores appropriately?
- Is the final answer consistent with both the reasoning and tool output?

Respond with ONLY a JSON object: {"coherence": <0-10>, "tool_quality": <0-10>}"""


def single_request(api_url, model, prompt, max_tokens, req_id):
    """Send one request, return stats."""
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
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=600,
        )
        t1 = time.time()
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        choices = data.get("choices", [])
        content = (choices[0]["message"].get("content") or
                   choices[0]["message"].get("reasoning_content") or
                   "(empty)") if choices else "(empty)"
        return {
            "req_id": req_id,
            "ok": True,
            "latency_s": t1 - t0,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "content": content[:200],
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
            "content": "",
        }


def gpu_worker(gpu_id, delay, api_url, model, prompt, max_tokens,
               completions_per_gpu, results_dict, timing_dict):
    """Simulate a single GPU's judge workload: wait for stagger delay, then fire all at once."""
    time.sleep(delay)
    arrive_time = time.time()
    print(f"  [GPU {gpu_id}] Arrived (delay={delay}s), sending {completions_per_gpu} concurrent judge calls...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=completions_per_gpu) as pool:
        futures = [
            pool.submit(single_request, api_url, model, prompt, max_tokens,
                        f"gpu{gpu_id}-{i}")
            for i in range(completions_per_gpu)
        ]
        gpu_results = [f.result() for f in concurrent.futures.as_completed(futures)]

    done_time = time.time()
    ok = sum(1 for r in gpu_results if r["ok"])
    total_tok = sum(r["total_tokens"] for r in gpu_results)
    comp_tok = sum(r["completion_tokens"] for r in gpu_results)
    wall = done_time - arrive_time

    print(f"  [GPU {gpu_id}] Done: {ok}/{completions_per_gpu} ok, "
          f"{total_tok} total tok, {comp_tok} gen tok, "
          f"{wall:.1f}s wall, {total_tok/wall:.0f} tok/s")

    results_dict[gpu_id] = gpu_results
    timing_dict[gpu_id] = {
        "arrive": arrive_time,
        "done": done_time,
        "wall": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="192.168.1.48")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-generations", "-G", type=int, default=16)
    parser.add_argument("--num-gpus", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max completion tokens for judge response (~1K for reasoning)")
    parser.add_argument("--gpu-delays", type=str, default="0,2,5",
                        help="Comma-separated stagger delays in seconds per GPU")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    completions_per_gpu = args.batch_size * args.num_generations
    delays = [float(d) for d in args.gpu_delays.split(",")]
    while len(delays) < args.num_gpus:
        delays.append(delays[-1] + 3)

    total_calls = completions_per_gpu * args.num_gpus

    print(f"\n{'='*65}")
    print(f"  GRPO Judge Benchmark — Realistic Multi-GPU Scenario")
    print(f"{'='*65}")
    print(f"  Model:            {args.model}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Num generations:  {args.num_generations}")
    print(f"  Completions/GPU:  {completions_per_gpu}")
    print(f"  Num GPUs:         {args.num_gpus}")
    print(f"  Total judge calls: {total_calls}")
    print(f"  Judge calls/completion: 1 (combined scoring)")
    print(f"  Max completion tokens: {args.max_tokens}")
    print(f"  GPU stagger delays: {delays[:args.num_gpus]}")
    print(f"  Input tokens: ~2K (worst case)")
    print(f"{'='*65}")

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
        res = single_request(api_url, args.model, COMBINED_JUDGE_PROMPT, args.max_tokens, f"warmup-{i}")
        status = "ok" if res["ok"] else f"FAIL: {res.get('error')}"
        print(f"  warmup-{i}: {status}, {res['prompt_tokens']} prompt + {res['completion_tokens']} gen = {res['total_tokens']} tok, {res['latency_s']:.2f}s")
        if res["ok"]:
            print(f"    Response: {res['content']}")

    # Run staggered multi-GPU simulation
    print(f"\n{'='*65}")
    print(f"  Starting {args.num_gpus}-GPU staggered simulation...")
    print(f"{'='*65}\n")

    results_dict = {}
    timing_dict = {}
    global_start = time.time()

    threads = []
    for gpu_id in range(args.num_gpus):
        t = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, delays[gpu_id], api_url, args.model,
                  COMBINED_JUDGE_PROMPT, args.max_tokens,
                  completions_per_gpu, results_dict, timing_dict),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    global_end = time.time()
    global_wall = global_end - global_start

    # ── Summary ──
    all_results = []
    for gpu_id in range(args.num_gpus):
        all_results.extend(results_dict[gpu_id])

    ok_results = [r for r in all_results if r["ok"]]
    fail_results = [r for r in all_results if not r["ok"]]

    total_prompt = sum(r["prompt_tokens"] for r in ok_results)
    total_comp = sum(r["completion_tokens"] for r in ok_results)
    total_tok = sum(r["total_tokens"] for r in ok_results)

    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY — GRPO Training Step Judge Overhead")
    print(f"{'='*65}")
    print(f"  Requests:    {len(all_results)} total, {len(ok_results)} ok, {len(fail_results)} failed")
    print(f"  Global wall: {global_wall:.1f}s")
    print()

    # Per-GPU breakdown
    for gpu_id in range(args.num_gpus):
        t = timing_dict[gpu_id]
        gpu_res = [r for r in results_dict[gpu_id] if r["ok"]]
        gpu_tok = sum(r["total_tokens"] for r in gpu_res)
        gpu_comp = sum(r["completion_tokens"] for r in gpu_res)
        lats = sorted(r["latency_s"] for r in gpu_res) if gpu_res else [0]
        print(f"  GPU {gpu_id}: wall={t['wall']:.1f}s, "
              f"{len(gpu_res)}/{completions_per_gpu} ok, "
              f"{gpu_tok} total tok, {gpu_comp} gen tok, "
              f"avg_lat={sum(lats)/len(lats):.1f}s, "
              f"p50={lats[len(lats)//2]:.1f}s, "
              f"p99={lats[int(len(lats)*0.99)]:.1f}s")

    print()
    if ok_results:
        avg_prompt = total_prompt / len(ok_results)
        avg_comp = total_comp / len(ok_results)
        all_lats = sorted(r["latency_s"] for r in ok_results)

        print(f"  Avg prompt tokens:     {avg_prompt:.0f}")
        print(f"  Avg completion tokens: {avg_comp:.0f}")
        print(f"  Avg total tokens:      {avg_prompt+avg_comp:.0f}")
        print()
        print(f"  Avg latency:   {sum(all_lats)/len(all_lats):.1f}s")
        print(f"  P50 latency:   {all_lats[len(all_lats)//2]:.1f}s")
        print(f"  P99 latency:   {all_lats[int(len(all_lats)*0.99)]:.1f}s")
        print()
        print(f"  Total tokens:      {total_tok}")
        print(f"  Total gen tokens:  {total_comp}")
        print(f"  Throughput:        {total_tok/global_wall:.0f} tok/s")
        print(f"  Gen throughput:    {total_comp/global_wall:.0f} tok/s")
        print(f"  Requests/s:        {len(ok_results)/global_wall:.2f}")

    # DDP gate time = max GPU done time
    latest_done = max(timing_dict[g]["done"] for g in range(args.num_gpus))
    ddp_gate = latest_done - global_start
    print()
    print(f"  ┌───────────────────────────────────────────────┐")
    print(f"  │ DDP-gated judge time (slowest GPU): {ddp_gate:.1f}s     │")
    print(f"  │ This is the overhead added to each GRPO step  │")
    print(f"  └───────────────────────────────────────────────┘")

    if fail_results:
        print(f"\n  FAILURES ({len(fail_results)}):")
        for r in fail_results[:5]:
            print(f"    {r['req_id']}: {r.get('error','?')}")

    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
