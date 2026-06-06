"""
GRPO training entrypoint for decoupled heuristic rewards.

Reuses grpo_single_phase trainer/model infrastructure but switches reward
computation to two heuristic channels:
    1. Accuracy channel (format + correctness)
    2. Tool-efficiency channel (conditioned on correctness)
"""

import os
import sys
import logging
import datetime
import json
import torch
from omegaconf import OmegaConf

# Path setup — same as grpo_single_phase/train_trl.py
_llm_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_llm_dir)
_desta_dir = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")

for _p in [_project_dir, os.path.join(_project_dir, "scripts"), _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Reuse everything from grpo_single_phase ───────────────────────────────────
from grpo_single_phase.train_trl import (
    AudioTRLGRPOTrainer,
    GreedyEvalCallback,
    make_trl_dataset,
)
from grpo_single_phase.dataset import split_mmau_dataset
from grpo_single_phase.modeling_grpo import GRPODeSTA25AudioModel

# ── Decoupled heuristic rewards ───────────────────────────────────────────────
from grpo_single_phase_llm_decoupled.rewards_rule import (
    trl_reward_acc_function,
    trl_reward_tool_function,
    set_reward_weights,
)

from trl import GRPOConfig
from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "GRPO_CONFIG", "grpo_single_phase_llm_decoupled/configs/optimized_v4.yaml"
    )
    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # ── Set heuristic reward weights ──────────────────────────────────────
    set_reward_weights(
        fw=cfg.get("format_reward_weight", 0.10),
        cw=cfg.get("correctness_reward_weight", 0.90),
        mw=cfg.get("option_mention_weight", 0.0),
        tw=cfg.get("tool_bonus_weight", 0.10),
    )

    # ── Data setup (identical to grpo_single_phase) ───────────────────────
    _sp_dir = os.path.join(_project_dir, "grpo_single_phase")
    data_file = os.path.join(_sp_dir, cfg.data_file)
    with open(data_file, "r") as f:
        all_data = json.load(f)
    all_data = [item for item in all_data if item.get("task") and item.get("question") and item.get("answer")]

    splits_dir = os.path.join(_sp_dir, cfg.splits_dir)
    train_items, eval_items, test_items = split_mmau_dataset(
        data_file=data_file, splits_dir=splits_dir,
        train_ratio=cfg.train_ratio, eval_ratio=cfg.eval_ratio, seed=cfg.data_seed,
    )

    from tool_execute import load_cached_tools
    cached_tools = load_cached_tools(train_items + eval_items)
    for item in train_items:
        item["cached_tool_outputs"] = cached_tools.get(item.get("audio_id", ""), {})
    for item in eval_items:
        item["cached_tool_outputs"] = cached_tools.get(item.get("audio_id", ""), {})

    audio_root = _project_dir
    precomputed_embed_dir = cfg.get("precomputed_embed_dir", None)
    train_ds = make_trl_dataset(train_items, audio_root, precomputed_embed_dir)
    eval_ds = make_trl_dataset(eval_items, audio_root, precomputed_embed_dir)

    # ── Model setup (identical to grpo_single_phase) ──────────────────────
    GRPODeSTA25AudioModel.skip_perception = bool(
        precomputed_embed_dir and os.path.isdir(precomputed_embed_dir)
    )
    model = GRPODeSTA25AudioModel.from_pretrained(cfg.model_hf_name, torch_dtype=torch.bfloat16)
    model._setup_generation()

    if precomputed_embed_dir and os.path.isdir(precomputed_embed_dir):
        model.load_embed_cache(precomputed_embed_dir)

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
    num_gpus = int(os.environ.get("WORLD_SIZE", cfg.get("num_gpus", 3)))

    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-{slurm_id}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-{timestamp}")

    os.makedirs(dynamic_output_dir, exist_ok=True)

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
        eval_strategy="epoch",
        eval_on_start=False,
        save_strategy="epoch",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,  # 60 min — tolerate slow judge HTTP calls
        log_completions=True,
        num_completions_to_print=0,
        temperature=cfg.temperature,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        reward_weights=[cfg.get("w_acc", 1.0), cfg.get("w_tool", 0.15)],
        multi_objective_aggregation=cfg.get("multi_objective_aggregation", "normalize_then_sum"),
        scale_rewards=cfg.get("scale_rewards", "group"),
    )

    # ── Create trainer (reuses AudioTRLGRPOTrainer from grpo_single_phase) ─
    trainer = AudioTRLGRPOTrainer(
        model=model,
        reward_funcs=[trl_reward_acc_function, trl_reward_tool_function],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=model.tokenizer,
        kl_threshold=cfg.get("kl_threshold", 0.0),
        callbacks=[
            GreedyEvalCallback(
                eval_items=eval_items,
                audio_root=_project_dir,
                precomputed_embed_dir=precomputed_embed_dir,
                max_new_tokens=cfg.max_new_tokens,
            ),
        ],
    )

    logger.info("TRL GRPOTrainer initialized with decoupled heuristic rewards.")

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


if __name__ == "__main__":
    main()
