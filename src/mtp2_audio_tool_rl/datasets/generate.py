"""
Generate GRPO training dataset from DeSTA-AQA5M.

Generates samples by streaming the DeSTA-AQA5M dataset and asking Gemini to
dynamically pick an appropriate tool to use for each audio clip based on its description,
then generate a multiple choice question. Saves output as JSONL.

Usage:
    # Test run (few samples per tool target)
    python generate.py --samples-per-tool 1

    # Full generation (200 samples per tool)
    python generate.py --samples-per-tool 200

    # Resume interrupted generation
    python generate.py --samples-per-tool 200 --resume
"""

import os
import sys
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Set, Tuple
from datasets import load_dataset
from openai_client import LocalLLMClient
from prompts import build_prompt, ALL_TOOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "grpo_tool_dataset.jsonl"

# Path to already curated candidates
FILTERED_FILE = Path("filtered_curated_candidates.jsonl")

def get_existing_state() -> Tuple[Dict, Set]:
    """Return dictionary of existing tool counts and a set of processed audio IDs."""
    counts = {t: 0 for t in ALL_TOOLS}
    processed_ids = set()

    if not OUTPUT_FILE.exists():
        return counts, processed_ids

    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                t = sample.get("tool")
                if t and t in counts:
                    counts[t] += 1
                audio_id = sample.get("id")
                if audio_id:
                    processed_ids.add(audio_id)
            except json.JSONDecodeError:
                continue
    return counts, processed_ids


def main():
    parser = argparse.ArgumentParser(description="Generate GRPO tool dataset from DeSTA-AQA5M dynamically")
    parser.add_argument(
        "--samples-per-tool", type=int, default=200,
        help="Target number of samples to collect for each tool (default: 200)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from where we left off (skip existing samples)"
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="Path to filtered candidates JSONL file (default: filtered_curated_candidates.jsonl)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write output JSONL and logs (default: output/)"
    )
    args = parser.parse_args()

    # Override paths if provided
    global OUTPUT_DIR, OUTPUT_FILE, FILTERED_FILE
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE = OUTPUT_DIR / "grpo_tool_dataset.jsonl"
    if args.input_file:
        FILTERED_FILE = Path(args.input_file)

    # Setup LLM trace logger (after output dir is resolved)
    llm_logger = logging.getLogger("llm_logger")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False
    fh = logging.FileHandler(OUTPUT_DIR / "llm_trace.log")
    fh.setFormatter(logging.Formatter("-" * 80 + "\n%(asctime)s\n%(message)s\n"))
    llm_logger.addHandler(fh)

    client = LocalLLMClient()

    # Determine counts and already processed IDs
    if args.resume:
        counts, processed_ids = get_existing_state()
    else:
        counts = {t: 0 for t in ALL_TOOLS}
        processed_ids = set()

    logger.info("=" * 60)
    logger.info("GRPO Dynamic Dataset Generation")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)

    for t, c in counts.items():
        logger.info(f"  {t:<25} : {c}")

    def all_done():
        return False  # we'll just process all lines in the filtered file

    if not FILTERED_FILE.exists():
        raise FileNotFoundError(f"No filtered file found at {FILTERED_FILE}.")

    logger.info(f"\nProcessing {FILTERED_FILE} to generate...")

    scanned = 0
    generated = 0
    skipped = 0
    failed = 0

    try:
        from concurrent.futures import FIRST_COMPLETED, wait

        def do_gen(item, target_tool):
            prompt = build_prompt(item, target_tool)
            try:
                res, pr, out = client.generate(prompt)
                return item, res, pr, out, None
            except RuntimeError as e:
                return item, None, prompt, None, e

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = set()

            with open(FILTERED_FILE) as f:
                for line in f:
                    if all_done():
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if item["id"] in processed_ids:
                        continue

                    scanned += 1

                    target_tool = item.get("candidate_tool")
                    if not target_tool or target_tool not in ALL_TOOLS:
                        skipped += 1
                        continue

                    # Submit to executor
                    future = executor.submit(do_gen, item, target_tool)
                    futures.add(future)

                    # Process completed when pool is full
                    while len(futures) >= 10:
                        done, futures = wait(futures, return_when=FIRST_COMPLETED)

                        for completed_future in done:
                            item, result, prompt, content, err = completed_future.result()

                            if err:
                                failed += 1
                                logger.error(f"Generate fail: {err}")
                                continue

                            llm_logger.info(f"PROMPT:\n{prompt}\n\nOUTPUT:\n{content}")

                            t = result.tool if result else None
                            if t not in ALL_TOOLS:
                                # Overwrite if generation got the tool name wrong
                                t = item.get("candidate_tool")

                            counts[t] += 1
                            generated += 1
                            processed_ids.add(item["id"])

                            out_data = result.model_dump() if result else {}
                            if "tool" not in out_data or out_data["tool"] not in ALL_TOOLS:
                                out_data["tool"] = t
                            out_data["seed_description"] = item.get("seed_description")
                            out_data["prompt"] = item.get("prompt")
                            with open(OUTPUT_FILE, "a") as out_f:
                                out_f.write(json.dumps(out_data) + "\n")

                            logger.info(f"[{t}] ✓ {counts[t]} total (audio: {item['id'][:30]}...)")

            # Flush remaining
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for completed_future in done:
                    item, result, prompt, content, err = completed_future.result()

                    if err:
                        failed += 1
                        logger.error(f"Generate fail: {err}")
                        continue

                    llm_logger.info(f"PROMPT:\n{prompt}\n\nOUTPUT:\n{content}")

                    t = result.tool if result else None
                    if t not in ALL_TOOLS:
                        # Overwrite if generation got the tool name wrong
                        t = item.get("candidate_tool")

                    counts[t] += 1
                    generated += 1
                    processed_ids.add(item["id"])

                    out_data = result.model_dump() if result else {}
                    if "tool" not in out_data or out_data["tool"] not in ALL_TOOLS:
                        out_data["tool"] = t
                    out_data["seed_description"] = item.get("seed_description")
                    out_data["prompt"] = item.get("prompt")
                    with open(OUTPUT_FILE, "a") as out_f:
                        out_f.write(json.dumps(out_data) + "\n")
                    logger.info(f"[{t}] ✓ {counts[t]} total (audio: {item['id'][:30]}...)")

    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user!")

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION SUMMMARY")
    logger.info(f"Scanned chunks : {scanned}")
    logger.info(f"Generated      : {generated}")
    logger.info(f"Skipped        : {skipped} (invalid target tool)")
    logger.info(f"Failed API     : {failed}")
    logger.info("\nFinal Per-tool counts:")
    for t in ALL_TOOLS:
        logger.info(f"  ✓ {t:<25} {counts[t]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
