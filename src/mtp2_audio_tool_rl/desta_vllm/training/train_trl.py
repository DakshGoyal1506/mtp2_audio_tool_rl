"""
GRPO training with vLLM-accelerated rollouts (colocate mode).

Same as grpo_single_phase_llm/train_trl.py but uses DeSTAVLLMEngine for
fast batched rollout generation during GRPO training.

Architecture (TRL colocate pattern):
  - N GPUs total, one accelerate process per GPU (DDP)
  - Each process runs BOTH:
      * its own vLLM rollout engine (distributed_executor_backend="external_launcher")
      * its share of the HF model + LoRA training (gradient computation)
  - vLLM sleep mode swaps weights+cache to CPU during the gradient step,
    leaving room for backward pass. Wake before rollout, sleep after.
  - LoRA weights synced HF → vLLM per-rank inside _generate (after wake_up,
    before generation). Each rank has its own vLLM, so no cross-rank file IO.
"""

import os
import sys
import logging
import datetime
import json

# When running inside an Apptainer container, conda site-packages are passed
# via CONDA_SITE_PKGS env var and appended to the END of sys.path so that
# container packages (vLLM, torch, transformers) take priority. Conda only
# provides packages missing from the container (TRL, PEFT, accelerate, etc.).
# This MUST happen before importing any conda-only packages (omegaconf, trl, peft).
_conda_site = os.environ.get("CONDA_SITE_PKGS", "")
if _conda_site and _conda_site not in sys.path:
    sys.path.append(_conda_site)

import torch
from omegaconf import OmegaConf

# Path setup
_training_dir = os.path.dirname(os.path.abspath(__file__))
_vllm_dir = os.path.dirname(_training_dir)
_project_dir = os.path.dirname(_vllm_dir)
_desta_dir = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")

for _p in [
    _project_dir,
    os.path.join(_project_dir, "scripts"),
    _desta_dir,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force-import desta package from DeSTA2.5-Audio source tree BEFORE anything
# else touches it.  In envs where desta is not pip-installed (e.g. gemma),
# Python 3.12's stricter namespace-package resolution can fail with
# KeyError: 'desta' if another desta* directory (desta_vllm) is found first.
import importlib
if "desta" not in sys.modules:
    _desta_spec = importlib.util.spec_from_file_location(
        "desta", os.path.join(_desta_dir, "desta", "__init__.py"),
        submodule_search_locations=[os.path.join(_desta_dir, "desta")],
    )
    if _desta_spec:
        _desta_mod = importlib.util.module_from_spec(_desta_spec)
        sys.modules["desta"] = _desta_mod
        _desta_spec.loader.exec_module(_desta_mod)

# ── Reuse from grpo_single_phase ──────────────────────────────────────────────
from grpo_single_phase.train_trl import (
    AudioTRLGRPOTrainer,
    GreedyEvalCallback,
    make_trl_dataset,
)
from grpo_single_phase.dataset import split_mmau_dataset
from grpo_single_phase.modeling_grpo import GRPODeSTA25AudioModel

# ── Local reward functions (self-contained, no grpo_single_phase_llm deps) ───
from desta_vllm.training.rewards_llm import (
    trl_reward_function,
    set_path_a_weights,
    set_path_b_weights,
    set_judge,
    set_tool_reward_params,
)
from grpo_single_phase_llm.judge import LLMJudge

# ── vLLM training components ─────────────────────────────────────────────────
from desta_vllm.training.trainer import create_vllm_grpo_trainer, LoRASyncCallback
from desta_vllm.engine import DeSTAVLLMEngine

from trl import GRPOConfig
from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "GRPO_CONFIG", "desta_vllm/training/configs/vllm_grpo_v1.yaml"
    )
    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # ── Initialize vLLM engine for rollouts ───────────────────────────────
    precomputed_embed_dir = cfg.get("precomputed_embed_dir", None)
    if precomputed_embed_dir and not os.path.isabs(precomputed_embed_dir):
        precomputed_embed_dir = os.path.join(_project_dir, precomputed_embed_dir)

    # ── Colocate mode: each accelerate process gets one GPU, runs both
    # vLLM (rollout) and HF model (training) on that GPU. vLLM uses
    # ``distributed_executor_backend="external_launcher"`` so it binds to
    # the current CUDA device (set by accelerate via LOCAL_RANK) instead of
    # spawning its own subprocess. With sleep mode, vLLM weights+KV cache
    # are swapped to CPU during the gradient step so the training model has
    # room to compute backward. This is the standard TRL pattern used for
    # multimodal GRPO (see trl examples/scripts/grpo_vlm.py).
    #
    # No CUDA_VISIBLE_DEVICES manipulation — accelerate sets it correctly.
    # Each rank reads LOCAL_RANK from env and binds via torch.cuda.set_device.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Bind this process to its local GPU BEFORE anything touches CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    logger.info(
        f"[rank={rank}/{world_size} local_rank={local_rank}] "
        f"using GPU {torch.cuda.current_device()} "
        f"({torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else 'CPU'})"
    )

    # ── CRITICAL: initialize torch.distributed BEFORE vLLM ────────────────
    # vLLM's `external_launcher` backend calls `init_world_group` which
    # asserts that the current rank exists in the WORLD process group. If
    # `torch.distributed` is not initialized yet, vLLM calls
    # `init_process_group(world_size=1, rank=$RANK)` — invalid when RANK>0.
    # TRL avoids this by creating vLLM AFTER the trainer's super().__init__
    # has already initialized the Accelerator's process group. We init
    # explicitly here using env vars set by `accelerate launch`:
    # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    import torch.distributed as dist
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")  # uses env://
        logger.info(
            f"[rank={rank}] torch.distributed initialized "
            f"(backend={dist.get_backend()}, world_size={dist.get_world_size()})"
        )

    # vLLM colocate settings — each process gets its own engine on its GPU
    vllm_memory_util = float(cfg.get("vllm_gpu_memory_utilization", 0.45))
    vllm_enable_sleep_mode = bool(cfg.get("vllm_enable_sleep_mode", True))
    vllm_enforce_eager = bool(cfg.get("vllm_enforce_eager", True))

    vllm_engine = DeSTAVLLMEngine(
        embed_dir=precomputed_embed_dir,
        tensor_parallel_size=1,  # one vLLM instance per GPU (colocate)
        gpu_memory_utilization=vllm_memory_util,
        max_model_len=cfg.get("vllm_max_model_len", 4096),
        seed=cfg.get("seed", 42) + rank,  # diversify rollouts across ranks
        enable_lora=True,
        max_lora_rank=cfg.get("lora_rank", 64),
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=vllm_enable_sleep_mode,
        enforce_eager=vllm_enforce_eager,
    )
    logger.info(
        f"[rank={rank}] vLLM rollout engine initialized "
        f"(colocate, sleep_mode={vllm_enable_sleep_mode}, "
        f"enforce_eager={vllm_enforce_eager}, gpu_mem={vllm_memory_util})"
    )

    # ── Connect to external judge server ──────────────────────────────────
    judge_model = cfg.get("judge_model", "Qwen/Qwen3.5-27B")
    judge_host = cfg.get("judge_host", "localhost")
    judge_port = cfg.get("judge_port", 8000)
    judge_base_url = f"http://{judge_host}:{judge_port}"

    judge = LLMJudge(
        base_url=judge_base_url,
        model=judge_model,
        timeout=cfg.get("judge_timeout", 300.0),
        max_completion_tokens=cfg.get("judge_max_tokens", 256),
        max_workers=cfg.get("judge_max_workers", 32),
        reasoning_effort=cfg.get("judge_reasoning_effort", "low"),
    )
    set_judge(judge)

    if judge.is_available():
        logger.info(f"Judge server reachable at {judge_base_url}")
    else:
        logger.warning(f"Judge server NOT reachable at {judge_base_url} — will use fallback scores")

    # ── Set path-dependent reward weights ─────────────────────────────────
    pa = cfg.get("path_a_weights", {})
    set_path_a_weights(
        answer=pa.get("answer", 0.60),
        think=pa.get("think", 0.30),
        fmt=pa.get("format", 0.10),
    )
    pb = cfg.get("path_b_weights", {})
    set_path_b_weights(
        answer=pb.get("answer", 0.50),
        think1=pb.get("think1", 0.15),
        tool=pb.get("tool", 0.10),
        think2=pb.get("think2", 0.15),
        fmt=pb.get("format", 0.10),
    )

    trp = cfg.get("tool_reward_params", None)
    if trp is not None:
        set_tool_reward_params(**dict(trp))

    # ── Data setup (identical to grpo_single_phase_llm) ───────────────────
    _sp_dir = os.path.join(_project_dir, "grpo_single_phase")
    data_file = os.path.join(_sp_dir, cfg.data_file)
    with open(data_file, "r") as f:
        all_data = json.load(f)
    all_data = [item for item in all_data if item.get("task") and item.get("question") and item.get("answer")]

    disable_eval = cfg.get("disable_eval", False)

    splits_dir = os.path.join(_sp_dir, cfg.splits_dir)
    train_items, eval_items, test_items = split_mmau_dataset(
        data_file=data_file, splits_dir=splits_dir,
        train_ratio=cfg.train_ratio, eval_ratio=cfg.eval_ratio, seed=cfg.data_seed,
    )

    if disable_eval and eval_items:
        logger.info(f"Eval disabled — merging {len(eval_items)} eval items into training set")
        train_items.extend(eval_items)
        eval_items = []

    from tool_execute import load_cached_tools
    cached_tools = load_cached_tools(train_items + eval_items)
    for item in train_items:
        item["cached_tool_outputs"] = cached_tools.get(item.get("audio_id", ""), {})
    for item in eval_items:
        item["cached_tool_outputs"] = cached_tools.get(item.get("audio_id", ""), {})

    audio_root = _project_dir
    train_ds = make_trl_dataset(train_items, audio_root, precomputed_embed_dir)
    eval_ds = make_trl_dataset(eval_items, audio_root, precomputed_embed_dir) if eval_items else None

    # ── Model setup (HF model for gradient computation) ───────────────────
    GRPODeSTA25AudioModel.skip_perception = bool(
        precomputed_embed_dir and os.path.isdir(precomputed_embed_dir)
    )
    # Resolve HF model ID to local cache path (no internet on compute nodes)
    from desta_vllm.embed_utils import _resolve_model_path
    _desta_model_path = _resolve_model_path(cfg.model_hf_name)
    logger.info(f"Resolved DeSTA model: {cfg.model_hf_name} → {_desta_model_path}")

    # Monkey-patch AutoModelForCausalLM.from_pretrained to resolve HF model IDs
    # to local cache paths. DeSTA's __init__ calls this with config.llm_model_id
    # ("DeSTA-ntu/Llama-3.1-8B-Instruct") which fails offline. The patch in
    # modeling_grpo.py only adds attn_implementation, so we wrap it further.
    from transformers import AutoModelForCausalLM as _AutoLLM
    _real_llm_from_pretrained = _AutoLLM.from_pretrained

    @staticmethod
    def _offline_llm_from_pretrained(pretrained_model_name_or_path, *a, **kw):
        resolved = _resolve_model_path(pretrained_model_name_or_path)
        if resolved != pretrained_model_name_or_path:
            logger.info(f"Resolved LLM: {pretrained_model_name_or_path} → {resolved}")
        return _real_llm_from_pretrained(resolved, *a, **kw)

    _AutoLLM.from_pretrained = _offline_llm_from_pretrained

    model = GRPODeSTA25AudioModel.from_pretrained(_desta_model_path, torch_dtype=torch.bfloat16)

    # Restore original from_pretrained
    _AutoLLM.from_pretrained = _real_llm_from_pretrained

    # Resolve model IDs in config so downstream calls (tokenizer, processor)
    # use local paths instead of trying to reach huggingface.co
    model.config.llm_model_id = _resolve_model_path(model.config.llm_model_id)
    if hasattr(model.config, "encoder_model_id"):
        model.config.encoder_model_id = _resolve_model_path(model.config.encoder_model_id)

    model._setup_generation()

    if precomputed_embed_dir and os.path.isdir(precomputed_embed_dir):
        model.load_embed_cache(precomputed_embed_dir)

    # Propagate token IDs from inner LLM config
    _llm_cfg = model.llm_model.config
    _llm_gen = getattr(model.llm_model, "generation_config", None)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        val = getattr(_llm_gen, attr, None) if _llm_gen else None
        if val is None:
            val = getattr(_llm_cfg, attr, None)
        if val is None and attr == "pad_token_id":
            val = model.tokenizer.pad_token_id
        if val is not None:
            setattr(model.config, attr, val)
            if getattr(model, "generation_config", None) is not None:
                setattr(model.generation_config, attr, val)

    if not hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = lambda *args, **kwargs: {}

    for name, param in model.named_parameters():
        if "perception" in name:
            param.requires_grad_(False)

    peft_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Training config ───────────────────────────────────────────────────
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))

    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-vllm-{slurm_id}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-vllm-{timestamp}")

    os.makedirs(dynamic_output_dir, exist_ok=True)
    judge.set_trace_dir(dynamic_output_dir)

    training_args = GRPOConfig(
        output_dir=dynamic_output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.get("eval_batch_size", cfg.G),
        num_generations=cfg.G,
        num_iterations=cfg.get("num_iterations", 1),
        max_completion_length=cfg.max_new_tokens,
        beta=cfg.beta,
        epsilon=cfg.get("epsilon", 0.2),
        logging_steps=cfg.log_every_n_steps,
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        num_train_epochs=cfg.epochs,
        eval_strategy="no" if disable_eval else "epoch",
        eval_on_start=False,
        save_strategy="epoch",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,
        log_completions=True,
        num_completions_to_print=0,
        temperature=cfg.temperature,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
    )

    # ── Create vLLM-backed trainer ────────────────────────────────────────
    VLLMAudioGRPOTrainer = create_vllm_grpo_trainer(AudioTRLGRPOTrainer)

    callbacks = []
    if not disable_eval and eval_items:
        callbacks.append(
            GreedyEvalCallback(
                eval_items=eval_items,
                audio_root=_project_dir,
                precomputed_embed_dir=precomputed_embed_dir,
                max_new_tokens=cfg.max_new_tokens,
            )
        )

    trainer_kwargs = dict(
        model=model,
        reward_funcs=[trl_reward_function],
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config,
        processing_class=model.tokenizer,
        kl_threshold=cfg.get("kl_threshold", 0.0),
        negative_tool_ratio=cfg.get("negative_tool_ratio", 0.0),
        vllm_engine=vllm_engine,
        callbacks=callbacks,
    )
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds

    trainer = VLLMAudioGRPOTrainer(**trainer_kwargs)

    # Add LoRA sync callback (sync weights after each step)
    lora_sync_cb = LoRASyncCallback(trainer)
    trainer.add_callback(lora_sync_cb)

    logger.info("VLLMAudioGRPOTrainer initialized with vLLM rollouts + LLM-as-judge rewards.")

    # Rich console redirect
    completions_log = os.path.join(dynamic_output_dir, "completions.log")
    try:
        from rich.console import Console
        _comp_file = open(completions_log, "w")
        trainer._console = Console(file=_comp_file, width=200, force_terminal=False)
        logger.info(f"TRL completion rollouts will be logged to {completions_log}")
    except Exception:
        pass

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model()

    logger.info(f"Training complete. Model saved to {dynamic_output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
