"""
Benchmark vLLM throughput with gpt-oss-20b on a single GPU.

Simulates the judge workload: 2.5K tokens per request (prompt + completion),
measures tokens/sec and requests/sec to estimate overhead per training step.

Usage:
    # On a DGX node with 1 GPU:
    python grpo_single_phase_llm/benchmark/bench_judge.py

    # Or via SLURM:
    sbatch grpo_single_phase_llm/benchmark/bench_judge.sh
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import IO
from typing import List
from typing import Optional

import requests

# Add project root
_bench_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(os.path.dirname(_bench_dir))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ── Sample prompts that mimic real judge workloads ────────────────────────────

SAMPLE_COHERENCE_PROMPT = """\
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

SAMPLE_TOOL_PROMPT = """\
You are evaluating whether an AI assistant made good tool-use decisions when answering an audio question.

## Question:
What instrument is being played in this audio clip?

## Answer choices:
  - Piano
  - Guitar
  - Violin
  - Drums

## Gold (correct) answer:
Piano

## Tool called:
sound_classification

## Tool output (truncated):
{"classifications": [{"label": "Piano", "confidence": 0.92}, {"label": "Keyboard", "confidence": 0.85}]}

## Assistant's final answer:
Piano

## Rate the TOOL USAGE on a scale of 0-10:

Consider:
- Was calling this tool NECESSARY? Could the question be answered just by listening?
- Was the RIGHT tool selected for this question type?
- If the tool output contradicts the audio, did the assistant handle it well?
- Penalize: calling tools for trivially answerable questions, selecting irrelevant tools

Respond with ONLY a JSON object: {"tool_quality": <0-10>}"""


def start_server(
    model: str,
    port: int,
    gpu_id: int = 0,
    log_fp: Optional[IO[str]] = None,
) -> subprocess.Popen:
    """Start vLLM server as subprocess."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--max-model-len", "4096",
        "--dtype", "auto",
        "--trust-remote-code",
        "--disable-log-requests",
    ]
    print(f"Starting vLLM: {' '.join(cmd)}")
    if log_fp is not None:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    else:
        # Fallback for local runs: inherit parent stdout/stderr.
        proc = subprocess.Popen(cmd, env=env)
    return proc


def stream_process_output(proc: subprocess.Popen, log_fp: IO[str]) -> threading.Thread:
    """Mirror child process output to stdout and log file in real time."""
    def _worker() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(line, end="")
            log_fp.write(line)
            log_fp.flush()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def check_gpu_access(gpu_id: int) -> tuple[bool, str]:
    """Check whether the requested GPU appears accessible on this node."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "nvidia-smi not found. This node likely has no NVIDIA GPU access."

    try:
        res = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as e:
        return False, f"failed to run nvidia-smi -L: {e}"

    if res.returncode != 0:
        msg = res.stderr.strip() or res.stdout.strip() or "unknown nvidia-smi failure"
        return False, f"nvidia-smi failed ({res.returncode}): {msg}"

    gpu_lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip().startswith("GPU ")]
    if not gpu_lines:
        return False, "nvidia-smi reports no visible GPUs on this node."
    if gpu_id >= len(gpu_lines):
        return False, f"requested GPU {gpu_id}, but only {len(gpu_lines)} GPU(s) visible."

    return True, f"GPU preflight OK: {gpu_lines[gpu_id]}"


def print_log_tail(log_path: str, n_lines: int = 200) -> None:
    """Print the tail of a log file to stdout for easier SLURM debugging."""
    print(f"\n--- vLLM startup log tail: {log_path} (last {n_lines} lines) ---")
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        tail = lines[-n_lines:]
        if tail:
            print("".join(tail).rstrip())
        else:
            print("[vLLM log file is empty]")
    except FileNotFoundError:
        print("[vLLM log file not found]")
    except Exception as e:
        print(f"[failed to read vLLM log: {e}]")
    print("--- end vLLM startup log tail ---\n")


def _run_diag_cmd(cmd: List[str], title: str) -> None:
    """Run a short diagnostic command and print output."""
    print(f"--- {title} ---")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        if out:
            print(out)
        if err:
            print(f"[stderr] {err}")
        if not out and not err:
            print("[no output]")
    except Exception as e:
        print(f"[diagnostic failed: {e}]")


def print_process_diagnostics(proc: Optional[subprocess.Popen], port: int) -> None:
    """Print useful process/runtime diagnostics when vLLM fails to become ready."""
    print("\n=== vLLM runtime diagnostics ===")
    if proc is None:
        print("No process handle available.")
        print("=== end diagnostics ===\n")
        return

    pid = proc.pid
    _run_diag_cmd(["ps", "-p", str(pid), "-o", "pid,ppid,etime,state,pcpu,pmem,cmd"], "process status")
    _run_diag_cmd(["bash", "-lc", f"ss -ltnp | grep ':{port} ' || true"], f"port {port} listeners")
    _run_diag_cmd(["bash", "-lc", "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true"], "GPU compute processes")
    _run_diag_cmd(["bash", "-lc", f"ls -l /proc/{pid}/fd 2>/dev/null | head -n 30 || true"], "process fd snapshot")
    print("=== end diagnostics ===\n")


def wait_ready(base_url: str, timeout: int = 300, proc: Optional[subprocess.Popen] = None) -> bool:
    """Wait for server health endpoint, or fail early if process exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                print(f"vLLM process exited before readiness check passed (return code {rc}).")
                return False
        try:
            r = requests.get(f"{base_url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def call_judge(api_url: str, model: str, prompt: str, max_tokens: int = 64) -> dict:
    """Single judge call, returns timing info."""
    t0 = time.time()
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
        timeout=120,
    )
    t1 = time.time()
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "latency_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def run_benchmark(api_url: str, model: str, num_requests: int = 50, batch_sizes: List[int] = None):
    """Run throughput benchmark."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    prompts = [SAMPLE_COHERENCE_PROMPT, SAMPLE_TOOL_PROMPT]

    print("\n" + "=" * 70)
    print("vLLM Judge Throughput Benchmark")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"API: {api_url}")
    print(f"Target: ~2.5K tokens per request (prompt + response)")
    print()

    # ── Warmup ────────────────────────────────────────────────────────────
    print("Warmup (2 requests)...")
    for p in prompts:
        call_judge(api_url, model, p)
    print("Warmup done.\n")

    # ── Sequential throughput ─────────────────────────────────────────────
    print("--- Sequential Throughput (1 request at a time) ---")
    results = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        r = call_judge(api_url, model, prompt)
        results.append(r)

    total_time = sum(r["latency_s"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    avg_prompt_tokens = sum(r["prompt_tokens"] for r in results) / len(results)
    avg_comp_tokens = sum(r["completion_tokens"] for r in results) / len(results)
    avg_latency = total_time / len(results)

    print(f"  Requests:         {num_requests}")
    print(f"  Avg prompt tokens:  {avg_prompt_tokens:.0f}")
    print(f"  Avg completion tokens: {avg_comp_tokens:.0f}")
    print(f"  Avg latency/req:  {avg_latency:.3f}s")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Throughput:       {total_tokens / total_time:.1f} tokens/s")
    print(f"  Requests/s:       {num_requests / total_time:.2f}")
    print()

    # ── Concurrent throughput (simulated via rapid fire) ──────────────────
    import concurrent.futures

    for batch_size in batch_sizes:
        print(f"--- Concurrent batch_size={batch_size} ---")
        batch_prompts = [prompts[i % len(prompts)] for i in range(batch_size)]

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [pool.submit(call_judge, api_url, model, p) for p in batch_prompts]
            batch_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        t1 = time.time()

        batch_tokens = sum(r["total_tokens"] for r in batch_results)
        batch_time = t1 - t0
        print(f"  Requests:    {batch_size}")
        print(f"  Wall time:   {batch_time:.2f}s")
        print(f"  Tokens:      {batch_tokens}")
        print(f"  Throughput:  {batch_tokens / batch_time:.1f} tokens/s")
        print(f"  Requests/s:  {batch_size / batch_time:.2f}")
        print()

    # ── Training step estimate ────────────────────────────────────────────
    print("=" * 70)
    print("TRAINING STEP ESTIMATE")
    print("=" * 70)
    # batch=8, G=16, 3 GPUs → 8*16*3 = 384 completions per step (but gathered on main)
    # Each completion needs 2 judge calls (coherence + tool_quality)
    completions_per_step = 8 * 16  # per-GPU, then gathered
    judge_calls = completions_per_step * 2  # 2 scores per completion
    seq_time = judge_calls * avg_latency

    # With concurrent batching (conservative estimate)
    concurrent_batch = 32
    concurrent_rounds = judge_calls / concurrent_batch
    # Estimate concurrent latency as ~1.5x single request (diminishing with batch)
    concurrent_avg = avg_latency * 1.5
    concurrent_time = concurrent_rounds * concurrent_avg

    print(f"  Completions per step (batch={8} × G={16}): {completions_per_step}")
    print(f"  Judge calls (2 per completion):           {judge_calls}")
    print(f"  Sequential estimate:                      {seq_time:.1f}s ({seq_time/60:.1f}min)")
    print(f"  Concurrent (batch={concurrent_batch}):            ~{concurrent_time:.1f}s ({concurrent_time/60:.1f}min)")
    print(f"  Current training step time:               ~150s (2.5min)")
    print()
    print(f"  ⚡ Recommended: concurrent batching with {concurrent_batch} workers")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM judge throughput")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--gpu", type=int, default=0, help="GPU to run vLLM on")
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--no-start-server", action="store_true",
                        help="Don't start vLLM server (assume already running)")
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32",
                        help="Comma-separated concurrent batch sizes to test")
    parser.add_argument("--startup-timeout", type=int, default=900,
                        help="Seconds to wait for vLLM /health during startup")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    api_url = f"{base_url}/v1/chat/completions"
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    vllm_log_path = os.path.join("logs", f"bench_judge_vllm_{job_id}.log")

    proc = None
    vllm_log_fp = None
    log_thread = None
    if not args.no_start_server:
        gpu_ok, gpu_msg = check_gpu_access(args.gpu)
        print(gpu_msg)
        if not gpu_ok:
            print("ERROR: GPU preflight failed. Run this benchmark on a GPU compute node (e.g., via sbatch).")
            sys.exit(2)

        os.makedirs("logs", exist_ok=True)
        vllm_log_fp = open(vllm_log_path, "w", encoding="utf-8", buffering=1)
        print(f"vLLM detailed log: {vllm_log_path}")
        proc = start_server(args.model, args.port, args.gpu, log_fp=vllm_log_fp)
        if proc.stdout is not None and vllm_log_fp is not None:
            log_thread = stream_process_output(proc, vllm_log_fp)
            print("vLLM output streaming enabled (line-by-line to SLURM and detailed log).")
        print(f"Waiting for server at {base_url} (timeout={args.startup_timeout}s)...")
        if not wait_ready(base_url, timeout=args.startup_timeout, proc=proc):
            if proc:
                rc = proc.poll()
                if rc is not None:
                    print(f"vLLM process exited early with return code {rc}.")
                else:
                    print("vLLM process is still running but /health never became ready.")
                    print_process_diagnostics(proc, args.port)
            if vllm_log_fp:
                vllm_log_fp.flush()
            print_log_tail(vllm_log_path, n_lines=300)
            print("ERROR: Server didn't start. Exiting.")
            if proc:
                proc.kill()
            if log_thread:
                log_thread.join(timeout=2)
            if vllm_log_fp:
                vllm_log_fp.close()
            sys.exit(1)
        print("Server ready!\n")

    try:
        run_benchmark(api_url, args.model, args.num_requests, batch_sizes)
    finally:
        if proc:
            print("\nShutting down server...")
            proc.terminate()
            proc.wait(timeout=30)
            print("Done.")
        if log_thread:
            log_thread.join(timeout=2)
        if vllm_log_fp:
            vllm_log_fp.close()


if __name__ == "__main__":
    main()
