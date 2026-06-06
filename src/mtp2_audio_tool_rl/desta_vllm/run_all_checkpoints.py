#!/usr/bin/env python3
"""
Submit SLURM inference jobs for all checkpoints using vLLM + Apptainer.

Uses desta_vllm/run_benchmark.py inside the gemma4.sif container for fast
batched inference with precomputed embeddings.

Default mode submits exactly 3 jobs:
  Job 1: base model, direct (no tools, no LoRA)
  Job 2: base model, tools (no LoRA)
  Job 3: all checkpoints with tools (sequential in one job)

Usage:
    python desta_vllm/run_all_checkpoints.py 117768
    python desta_vllm/run_all_checkpoints.py 117768 --max-jobs 5
    python desta_vllm/run_all_checkpoints.py 117768 --direct
    python desta_vllm/run_all_checkpoints.py 117768 --base-tools
    python desta_vllm/run_all_checkpoints.py 117768 --ckpts-only
    python desta_vllm/run_all_checkpoints.py 117768 --steps 50,100,200
    python desta_vllm/run_all_checkpoints.py 117768 --dry-run
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(PROJECT_DIR, "checkpoints")
DEFAULT_VLLM_CACHE_ROOT = os.environ.get(
    "VLLM_CACHE_ROOT", os.path.join(PROJECT_DIR, ".cache", "vllm")
)

SBATCH_HEADER = """\
#!/bin/bash
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --exclude=cn14-dgx,cn22-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --job-name={job_name}
#SBATCH --output=results/vllm/{log_name}_slurm_%j.log

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "${{HOME}}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${{HOME}}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda is not available. Load conda or install it under $HOME/anaconda3."
    exit 1
fi
conda activate gemma

set -eo pipefail

cd {project_dir}

IFILE="./gemma4.sif"
if [ ! -f "$IFILE" ]; then
    echo "ERROR: $IFILE not found!"
    exit 1
fi

export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"

RESULTS_DIR="{results_dir}"
mkdir -p "$RESULTS_DIR"

SCRATCH_DIR="${{RESULTS_DIR}}/scratch_${{SLURM_JOB_ID:-$$}}"
mkdir -p "$SCRATCH_DIR"
trap "rm -rf '$SCRATCH_DIR'" EXIT
"""

# Each run block launches apptainer exec with the benchmark command
SINGLE_RUN_BLOCK = """
echo "=============================================="
echo "{description}"
echo "=============================================="

PYTHON_CMD="python3 -m desta_vllm.run_benchmark \\
    --data '{data}' \\
    --embed-dir '{embed_dir}' \\
    --output '{output}' \\
    --model DeSTA-ntu/Llama-3.1-8B-Instruct \\
    --max-model-len {max_model_len} \\
    --max-new-tokens {max_new_tokens} \\
    --gpu-memory {gpu_memory} \\
    --tensor-parallel 1 \\
    --batch-size {batch_size}{lora_flag}{direct_flag}"

echo ">>> $PYTHON_CMD"

apptainer exec --cleanenv --nv \\
    -B /dev/shm \\
    -B "${{HF_HOME}}:${{HF_HOME}}" \\
    -B "{project_dir}:{project_dir}" \\
    -B "${{HOME}}:${{HOME}}" \\
    -B "$SCRATCH_DIR:/scratch" \\
    --env HOME="/scratch" \\
    --env TMPDIR="/scratch" \\
    --env HF_HOME="${{HF_HOME}}" \\
    --env HF_HUB_OFFLINE_MODE=1 \\
    --env TRANSFORMERS_OFFLINE=1 \\
    --env PYTHONPATH="{project_dir}" \\
    --env PYTHONUNBUFFERED=1 \\
    --env USER=user \\
    --env VLLM_LOGGING_LEVEL=INFO \\
    --env VLLM_CONFIGURE_LOGGING=1 \\
    --env VLLM_CACHE_ROOT="{vllm_cache_root}" \\
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \\
    "$IFILE" \\
    bash -c "mkdir -p '{vllm_cache_root}' && cd '{project_dir}' && stdbuf -oL -eL $PYTHON_CMD"

echo "Exit code: $?"
echo ""

# Quick accuracy summary
if [ -f "{output}" ]; then
    python3 -c "
import json
with open('{output}') as f:
    data = json.load(f)
total = len(data)
correct = sum(1 for d in data if d.get('is_correct', False))
print(f'  Total: {{total}}, Correct: {{correct}}, Accuracy: {{correct/total*100:.1f}}%' if total else '  No results')
tasks = {{}}
for d in data:
    t = d.get('task', 'unknown')
    tasks.setdefault(t, [0, 0])
    tasks[t][1] += 1
    if d.get('is_correct'):
        tasks[t][0] += 1
for t in sorted(tasks):
    c, n = tasks[t]
    print(f'  {{t:30s}}: {{c:3d}}/{{n:3d}} = {{c/n*100:5.1f}}%')
" 2>/dev/null || true
fi
"""


def find_checkpoints(job_id: str) -> list:
    run_dir = os.path.join(CHECKPOINT_ROOT, f"grpo-desta-{job_id}")
    if not os.path.isdir(run_dir):
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    ckpt_dirs = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
    )

    if not ckpt_dirs:
        print(f"ERROR: No checkpoint-* directories found in {run_dir}")
        sys.exit(1)

    return ckpt_dirs


def _build_single_run(
    *,
    description,
    data,
    embed_dir,
    output,
    lora_path=None,
    direct=False,
    max_model_len=4096,
    max_new_tokens=2048,
    gpu_memory=0.85,
    batch_size=64,
):
    return SINGLE_RUN_BLOCK.format(
        description=description,
        data=data,
        embed_dir=embed_dir,
        output=output,
        project_dir=PROJECT_DIR,
        vllm_cache_root=DEFAULT_VLLM_CACHE_ROOT,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        gpu_memory=gpu_memory,
        batch_size=batch_size,
        lora_flag=f" \\\n    --lora-path '{lora_path}'" if lora_path else "",
        direct_flag=" \\\n    --direct" if direct else "",
    )


def _build_script(*, job_name, log_name, results_dir, body_blocks):
    header = SBATCH_HEADER.format(
        job_name=job_name,
        log_name=log_name,
        project_dir=PROJECT_DIR,
        results_dir=results_dir,
    )
    return header + "\n".join(body_blocks) + "\n"


def _chunk_list(lst, n_chunks):
    k = math.ceil(len(lst) / n_chunks)
    return [lst[i : i + k] for i in range(0, len(lst), k)]


def _submit_or_print(script_content, script_path, label, dry_run):
    if dry_run:
        print(f"=== {label} ===")
        print(script_content)
        print()
        return None

    with open(script_path, "w") as f:
        f.write(script_content)

    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(script_path),
    )

    os.unlink(script_path)

    if result.returncode == 0:
        slurm_job_id = result.stdout.strip().split()[-1]
        print(f"  {label} -> SLURM job {slurm_job_id}")
        return slurm_job_id
    else:
        print(f"  {label} -> FAILED: {result.stderr.strip()}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Submit vLLM inference for all checkpoints of a training run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("job_id", type=str, help="Training run job ID (e.g. 117768)")
    parser.add_argument(
        "--data",
        type=str,
        default="mmau-test-mini-cached.json",
        help="Evaluation data file",
    )
    parser.add_argument(
        "--embed-dir",
        type=str,
        default="precomputed_embeds",
        help="Precomputed embeddings directory",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=1,
        help="Max SLURM jobs to submit (default: 1, all ckpts sequential)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model sequence length",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.85,
        help="GPU memory utilization",
    )
    parser.add_argument("--direct", action="store_true", help="Base model direct only")
    parser.add_argument("--base-tools", action="store_true", help="Base model tools only")
    parser.add_argument("--ckpts-only", action="store_true", help="Checkpoints only (no base)")
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated checkpoint steps (e.g. --steps 50,100,200)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print scripts without submitting")
    args = parser.parse_args()

    results_dir = os.path.join(PROJECT_DIR, "results", "inference", f"{args.job_id}_tools")
    os.makedirs(results_dir, exist_ok=True)

    common = dict(
        data=args.data,
        embed_dir=args.embed_dir,
        max_model_len=args.max_model_len,
        gpu_memory=args.gpu_memory,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Single-mode: --direct or --base-tools
    # ------------------------------------------------------------------
    if args.direct or args.base_tools:
        tag = "direct" if args.direct else "tools"
        print(f"Run: base model (no LoRA), mode={tag}")
        print(f"Data: {args.data}")
        print()

        output = os.path.join(results_dir, f"{args.job_id}_{tag}_base.json")
        block = _build_single_run(
            description=f"vLLM Inference: job={args.job_id} mode={tag} (base, no LoRA)",
            output=output,
            direct=(tag == "direct"),
            **common,
        )
        script = _build_script(
            job_name=f"vllm-{args.job_id}-{tag}-base",
            log_name=f"vllm_{args.job_id}_{tag}_base",
            results_dir=results_dir,
            body_blocks=[block],
        )
        script_path = os.path.join(PROJECT_DIR, f".tmp_vllm_{args.job_id}_{tag}_base.sh")
        _submit_or_print(script, script_path, f"base {tag} (no LoRA)", args.dry_run)
        print(f"\nResults dir: {results_dir}/")
        return

    # ------------------------------------------------------------------
    # Default: base-direct + base-tools + all checkpoints
    # ------------------------------------------------------------------
    ckpt_dirs = find_checkpoints(args.job_id)

    if args.steps:
        requested = set(args.steps.split(","))
        ckpt_dirs = [
            d
            for d in ckpt_dirs
            if re.search(r"checkpoint-(\d+)", d).group(1) in requested
        ]
        if not ckpt_dirs:
            print(f"ERROR: None of the requested steps ({args.steps}) found.")
            sys.exit(1)

    base_jobs = 0 if args.ckpts_only else 2
    ckpt_slots = max(1, args.max_jobs - base_jobs)

    print(f"Run: grpo-desta-{args.job_id} (vLLM)")
    print(f"Data: {args.data}")
    print(f"Max SLURM jobs: {args.max_jobs}  (base: {base_jobs}, ckpt batches: {ckpt_slots})")
    print(f"Checkpoints: {len(ckpt_dirs)}")
    for d in ckpt_dirs:
        print(f"  - {os.path.basename(d)}")
    print()

    submitted = []

    # --- Base model jobs ---
    if not args.ckpts_only:
        for tag, is_direct in [("direct", True), ("tools", False)]:
            output = os.path.join(results_dir, f"{args.job_id}_{tag}_base.json")
            block = _build_single_run(
                description=f"vLLM Inference: job={args.job_id} mode={tag} (base, no LoRA)",
                output=output,
                direct=is_direct,
                **common,
            )
            script = _build_script(
                job_name=f"vllm-{args.job_id}-{tag}-base",
                log_name=f"vllm_{args.job_id}_{tag}_base",
                results_dir=results_dir,
                body_blocks=[block],
            )
            script_path = os.path.join(PROJECT_DIR, f".tmp_vllm_{args.job_id}_{tag}_base.sh")
            jid = _submit_or_print(script, script_path, f"base {tag}", args.dry_run)
            if jid:
                submitted.append((f"base-{tag}", jid))

    # --- Checkpoint LoRA jobs ---
    batches = _chunk_list(ckpt_dirs, ckpt_slots)
    for batch_idx, batch in enumerate(batches):
        steps = []
        blocks = []
        for ckpt_path in batch:
            step = re.search(r"checkpoint-(\d+)", ckpt_path).group(1)
            steps.append(step)
            output = os.path.join(results_dir, f"{args.job_id}_tools_step{step}.json")
            blocks.append(
                _build_single_run(
                    description=f"vLLM Inference: job={args.job_id} tools step-{step} (LoRA)",
                    output=output,
                    lora_path=ckpt_path,
                    **common,
                )
            )

        step_range = steps[0] if len(steps) == 1 else f"{steps[0]}-{steps[-1]}"
        label = f"ckpts steps {step_range} ({len(steps)} checkpoints)"
        script = _build_script(
            job_name=f"vllm-{args.job_id}-batch{batch_idx}",
            log_name=f"vllm_{args.job_id}_batch{batch_idx}",
            results_dir=results_dir,
            body_blocks=blocks,
        )
        script_path = os.path.join(PROJECT_DIR, f".tmp_vllm_{args.job_id}_batch{batch_idx}.sh")
        jid = _submit_or_print(script, script_path, label, args.dry_run)
        if jid:
            submitted.append((label, jid))

    if submitted:
        print(f"\nSubmitted {len(submitted)} SLURM jobs:")
        for label, jid in submitted:
            print(f"  {label} -> job {jid}")

    print(f"\nResults dir: {results_dir}/")


if __name__ == "__main__":
    main()
