"""
run_benchmark.py — Fast MMAU benchmark using vLLM + precomputed embeddings.

Drop-in replacement for:
    python scripts/tool_execute.py --model desta-8b --data mmau-test-mini-cached.json

Usage:
    python -m desta_vllm.run_benchmark \
        --data mmau-test-mini-cached.json \
        --embed-dir precomputed_embeds \
        --output results/vllm/tools-benchmark.json \
        [--direct]  # skip tool selection, answer directly
        [--lora-path checkpoints/grpo-desta-XXXX/adapter_model]
        [--tensor-parallel 1]
        [--gpu-memory 0.85]
        [--max-model-len 4096]

Outputs:
    - results JSON with predictions + accuracy per task
    - JSONL prompt log for debugging
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from desta_vllm.engine import DeSTAVLLMEngine
from desta_vllm.tool_execute_vllm import (
    VLLMToolExecutor,
    extract_model_prediction,
    format_question_with_choices,
    load_cached_tools,
)

logger = logging.getLogger(__name__)


def _string_match(answer: str, prediction: str, choices: list) -> bool:
    """MMAU official string_match metric."""
    def tokenize(text):
        return set(re.findall(r"\b\w+\b", text.lower()))

    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(answer)
    if not pred_tokens:
        return False
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != gold_tokens:
            incorrect_tokens.update(choice_tokens - gold_tokens)
    return gold_tokens.issubset(pred_tokens) and pred_tokens.isdisjoint(
        incorrect_tokens
    )


def save_result_iterative(result_entry: dict, output_file: str) -> None:
    """Append one result entry to JSONL."""
    jsonl_file = output_file.replace(".json", ".jsonl")
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")


def finalize_results(output_file: str) -> str:
    """Convert JSONL → JSON array."""
    jsonl_file = output_file.replace(".json", ".jsonl")
    if not os.path.exists(jsonl_file):
        return output_file
    results = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Finalized {len(results)} results to {output_file}")
    return output_file


def _save_result_entry(
    item: dict,
    prediction: str,
    is_correct: bool,
    result: dict,
    output_file: str,
) -> None:
    """Build and append one result entry to JSONL."""
    result_entry = {
        "id": item.get("id", ""),
        "audio_id": item.get("audio_id", ""),
        "question": item.get("question", ""),
        "choices": item.get("choices", []),
        "gold_answer": item.get("answer", ""),
        "model_prediction": prediction,
        "is_correct": is_correct,
        "result_type": result.get("type", "unknown"),
        "tools_selected": [c["function"] for c in result.get("tool_calls", [])],
        "tool_results": result.get("tool_results", {}),
        "think": result.get("parsed_answer", {}).get("think", ""),
        "raw_initial_response": result.get("initial_response", ""),
        "raw_final_response": result.get("final_response", ""),
        "task": item.get("task", "unknown"),
        "category": item.get("category", ""),
        "sub-category": item.get("sub-category", ""),
        "difficulty": item.get("difficulty", ""),
    }
    save_result_iterative(result_entry, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="MMAU benchmark with vLLM + precomputed DeSTA embeddings"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="mmau-test-mini-cached.json",
        help="Path to evaluation data JSON (with cached tool_outputs)",
    )
    parser.add_argument(
        "--embed-dir",
        type=str,
        default="precomputed_embeds",
        help="Directory with *_embed.pt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output results file (auto-generated if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DeSTA-ntu/Llama-3.1-8B-Instruct",
        help="Llama backbone model ID",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Direct mode: answer without tool selection",
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.85,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model sequence length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens per generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for faster startup",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batched tool-assisted evaluation",
    )
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        questions_data = json.load(f)
    questions_data = [
        item
        for item in questions_data
        if item.get("question") and item.get("answer")
    ]
    logger.info(f"Loaded {len(questions_data)} questions")

    cached_tools = load_cached_tools(questions_data)
    logger.info(f"Cached tool outputs for {len(cached_tools)} audio files")

    # ── Output path ───────────────────────────────────────────────────────
    if args.output is None:
        results_dir = os.path.join(_project_dir, "results", "vllm")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "direct" if args.direct else "tools"
        lora_tag = ""
        if args.lora_path:
            lora_tag = f"-lora"
        args.output = os.path.join(
            results_dir, f"{mode}{lora_tag}-{timestamp}.json"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    logger.info(f"Results will be saved to {args.output}")

    # ── Initialize engine ─────────────────────────────────────────────────
    t0 = time.time()
    engine = DeSTAVLLMEngine(
        llm_model_id=args.model,
        embed_dir=args.embed_dir,
        lora_path=args.lora_path,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        enable_lora=bool(args.lora_path),
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )
    logger.info(f"Engine initialized in {time.time() - t0:.1f}s")

    executor = VLLMToolExecutor(
        engine=engine,
        cached_tools=cached_tools,
        embed_dir=args.embed_dir,
        max_new_tokens=args.max_new_tokens,
    )

    # ── Check for resume ──────────────────────────────────────────────────
    processed_ids = set()
    jsonl_file = args.output.replace(".json", ".jsonl")
    if os.path.exists(jsonl_file):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    processed_ids.add(entry.get("id", ""))
        logger.info(f"Resuming: {len(processed_ids)} already processed")

    # ── Run evaluation ────────────────────────────────────────────────────
    total = len(questions_data)
    correct = 0
    attempted = 0
    task_stats: Dict[str, Dict[str, int]] = {}

    t_start = time.time()

    # Filter out already-processed items
    pending_items = [
        (i, item)
        for i, item in enumerate(questions_data)
        if item.get("id", f"q_{i+1}") not in processed_ids
    ]
    logger.info(f"Processing {len(pending_items)} items (skipping {total - len(pending_items)} already done)")

    # Unified batch loop — works for both direct and tool-assisted modes
    batch_size = args.batch_size
    pending_only = [item for _, item in pending_items]

    for batch_start in range(0, len(pending_only), batch_size):
        batch = pending_only[batch_start : batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(pending_only))
        logger.info(
            f"=== Batch {batch_start // batch_size + 1}: "
            f"items {batch_start+1}-{batch_end} of {len(pending_only)} ==="
        )

        t_batch = time.time()
        try:
            batch_results = executor.process_batch(batch, direct=args.direct)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            batch_results = [
                {"type": "error", "error": str(e)} for _ in batch
            ]

        for j, (item, result) in enumerate(zip(batch, batch_results)):
            gold = item.get("answer", "")
            choices = item.get("choices", [])

            prediction = ""
            if result.get("type") != "error":
                prediction = result.get("parsed_answer", {}).get(
                    "answer", ""
                ) or extract_model_prediction(
                    result.get("final_response", "")
                )

            is_correct = _string_match(gold, prediction, choices)

            task = item.get("task", "unknown")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            task_stats[task]["total"] += 1
            if is_correct:
                task_stats[task]["correct"] += 1
                correct += 1
            attempted += 1

            _save_result_entry(item, prediction, is_correct, result, args.output)

        batch_elapsed = time.time() - t_batch
        acc = correct / attempted * 100 if attempted > 0 else 0
        logger.info(
            f"Batch done in {batch_elapsed:.1f}s "
            f"({batch_elapsed/len(batch):.2f}s/item) | "
            f"Running acc={acc:.1f}% ({correct}/{attempted})"
        )

    elapsed = time.time() - t_start

    # ── Finalize ──────────────────────────────────────────────────────────
    finalize_results(args.output)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"MMAU Benchmark Results (vLLM)")
    print(f"=" * 70)
    print(f"Mode:       {'Direct' if args.direct else 'Tool-assisted'}")
    print(f"Model:      {args.model}")
    if args.lora_path:
        print(f"LoRA:       {args.lora_path}")
    print(f"Total:      {attempted}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {correct/attempted*100:.1f}%" if attempted > 0 else "N/A")
    print(f"Time:       {elapsed:.1f}s ({elapsed/max(attempted,1):.2f}s/question)")
    print()
    print("Per-task breakdown:")
    for task_name in sorted(task_stats.keys()):
        s = task_stats[task_name]
        task_acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {task_name:30s}: {s['correct']:3d}/{s['total']:3d} = {task_acc:5.1f}%")
    print(f"=" * 70)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
