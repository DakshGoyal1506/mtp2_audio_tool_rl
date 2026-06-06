"""
train_grpo.py — Main entry point for GRPO training of DeSTA2.5-Audio tool-use.

Usage:
    # Single GPU
    python grpo/train_grpo.py --config grpo/configs/default.yaml

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=4 grpo/train_grpo.py --config grpo/configs/default.yaml

    # Dry run (one batch, no backward)
    python grpo/train_grpo.py --config grpo/configs/default.yaml --dry-run
"""

import argparse
import logging
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DeepSpeedPlugin
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup

# ---------------------------------------------------------------------------
# Path setup — allow running from Audio-Maestro/ root or grpo/ subdir
# ---------------------------------------------------------------------------
_grpo_dir    = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_grpo_dir)           # Audio-Maestro/
_desta_dir   = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")

for _p in [_project_dir, os.path.join(_project_dir, "scripts"), _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grpo.dataset       import split_mmau_dataset, GRPOAudioDataset
from grpo.model_wrapper import DeSTA25GRPOModel
from grpo.model_wrapper_qwen import QwenOmniGRPOModel
from grpo.trainer       import GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
# get_logger from accelerate gates all .info() calls to main process only
logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for DeSTA2.5-Audio")
    parser.add_argument("--config",  type=str, required=True,  help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true",       help="Process one batch, no backward pass")
    parser.add_argument("--skip-baseline-eval", action="store_true",
                        help="Skip the pre-training baseline eval (saves ~10 min startup time)")
    parser.add_argument(
        "overrides", nargs="*",
        help="OmegaConf dot-notation overrides, e.g. num_gpus=2 G=8"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_desta_model(cfg, device):
    """Load DeSTA25AudioModel with LoRA enabled, freeze encoder + QFormer."""
    try:
        from grpo.modeling_grpo import GRPODeSTA25AudioModel  # type: ignore
    except ImportError:
        # Fallback: original class (no decoder removal)
        try:
            from desta.models.modeling_desta25 import DeSTA25AudioModel as GRPODeSTA25AudioModel  # type: ignore
        except ImportError:
            from desta import DeSTA25AudioModel as GRPODeSTA25AudioModel  # type: ignore

    # When precomputed embeddings are available, skip loading Whisper + Qformer
    # entirely — saves ~1.5 GB VRAM and removes the audio-model load time.
    precomputed_embed_dir = cfg.get("precomputed_embed_dir", None)
    if precomputed_embed_dir and os.path.isdir(precomputed_embed_dir):
        GRPODeSTA25AudioModel.skip_perception = True
        logger.info(
            f"Precomputed embeddings found at {precomputed_embed_dir} — "
            "Whisper encoder + Qformer will NOT be loaded."
        )
    else:
        GRPODeSTA25AudioModel.skip_perception = False

    logger.info(f"Loading DeSTA25 model: {cfg.model_hf_name}")

    model = GRPODeSTA25AudioModel.from_pretrained(
        cfg.model_hf_name,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)

    # Enable LoRA on the LLM backbone
    _enable_lora(model, cfg)

    # Freeze speech encoder (Whisper) and QFormer connector (if they still exist)
    for name, param in model.named_parameters():
        if "perception" in name:   # Whisper encoder + QFormer
            param.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {n_trainable:,} / {n_total:,} "
        f"({100*n_trainable/n_total:.2f}%)"
    )
    return model


def _is_qwen_model(cfg) -> bool:
    """Check if the config specifies a Qwen model."""
    name = cfg.get("model_name", "").lower()
    hf   = cfg.get("model_hf_name", "").lower()
    return "qwen" in name or "qwen" in hf


def _enable_lora(model, cfg):
    """Attach LoRA adapters to the LLM's q/k/v projection layers.

    Mirrors the official DeSTA training approach: store the unwrapped
    LlamaForCausalLM (with LoRA layers grafted in) on model.llm_model so that
    model.llm_model.model.embed_tokens works as expected.  We also stash the
    PeftModel's inner LoraModel as model._lora_base_model, which exposes
    disable_adapter_layers() / enable_adapter_layers() for reference_log_probs.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

        lora_cfg = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model.llm_model, lora_cfg)
        # Official DeSTA style: unwrap to the base LlamaForCausalLM so that
        # .model.embed_tokens access in modeling_desta25.py keeps working.
        model._peft_model      = peft_model                  # PeftModel — has save_pretrained()
        model._lora_base_model = peft_model.base_model        # LoraModel — has disable/enable_adapter_layers()
        model.llm_model        = peft_model.base_model.model  # LlamaForCausalLM with LoRA layers

        logger.info(
            f"LoRA enabled: rank={cfg.lora_rank}, alpha={cfg.lora_alpha}, "
            f"targets={list(cfg.lora_target_modules)}"
        )

        if cfg.get("gradient_checkpointing", False):
            # Recompute activations during backward instead of storing them —
            # trades ~30% extra compute for ~40% less activation memory.
            # use_reentrant=False is required with LoRA (avoids PEFT/autograd conflict).
            model.llm_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Suppress the per-step `use_cache incompatible` warning.
            model.llm_model.config.use_cache = False
            logger.info("Gradient checkpointing enabled (use_reentrant=False)")
        else:
            logger.info("Gradient checkpointing disabled (gradient_checkpointing: false in config)")
    except ImportError:
        logger.error("peft not installed — cannot enable LoRA. Install with: pip install peft")
        raise


# ---------------------------------------------------------------------------
# Accelerator / DeepSpeed setup
# ---------------------------------------------------------------------------

def build_accelerator(cfg) -> Accelerator:
    ds_stage = cfg.get("deepspeed_stage", 0)
    if ds_stage and ds_stage > 0:
        ds_plugin = DeepSpeedPlugin(
            zero_stage=ds_stage,
            gradient_accumulation_steps=cfg.grad_accumulation_steps,
        )
        ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.batch_size
        accel = Accelerator(
            gradient_accumulation_steps=cfg.grad_accumulation_steps,
            deepspeed_plugin=ds_plugin,
        )
        logger.info(f"Accelerator with DeepSpeed ZeRO-{ds_stage}")
    else:
        accel = Accelerator(
            gradient_accumulation_steps=cfg.grad_accumulation_steps,
        )
        logger.info("Accelerator without DeepSpeed")
    return accel


# ---------------------------------------------------------------------------
# Build optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, cfg, total_steps: int):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    cfg   = OmegaConf.load(args.config)

    # Apply CLI overrides (e.g. num_gpus=2 G=8)
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Resolve env-var judge credentials into config
    if not cfg.get("judge_url"):
        cfg.judge_url = os.getenv("LLM_JUDGE_URL", None)
    if not cfg.get("judge_model"):
        cfg.judge_model = os.getenv("LLM_JUDGE_MODEL", None)

    # ---- Accelerator ----
    accel  = build_accelerator(cfg)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}", main_process_only=True)
    device = accel.device

    # ---- Dataset splits ----
    # Resolve data_file relative to grpo/ dir
    data_file  = os.path.join(_grpo_dir, cfg.data_file)
    splits_dir = os.path.join(_grpo_dir, cfg.splits_dir)

    train_items, eval_items, test_items = split_mmau_dataset(
        data_file=data_file,
        splits_dir=splits_dir,
        train_ratio=cfg.train_ratio,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.data_seed,
    )

    # Audio files are relative to the mmau-test-mini-cached.json location,
    # which lives in Audio-Maestro/
    audio_root    = _project_dir
    precomputed_embed_dir = cfg.get("precomputed_embed_dir", None)
    train_dataset = GRPOAudioDataset(train_items, audio_root=audio_root, precomputed_embed_dir=precomputed_embed_dir)
    eval_dataset  = GRPOAudioDataset(eval_items,  audio_root=audio_root, precomputed_embed_dir=precomputed_embed_dir)

    if accel.is_main_process:
        logger.info(
            f"Dataset ready — train: {len(train_dataset)}, eval: {len(eval_dataset)}, "
            f"test: {len(test_items)} (test set not used during training)"
        )


    # ---- Policy model ----
    use_qwen = _is_qwen_model(cfg)

    if use_qwen:
        from grpo.modeling_qwen_grpo import load_qwen_model, enable_lora_qwen
        policy_model, qwen_processor = load_qwen_model(cfg, device)
        policy_model = enable_lora_qwen(policy_model, cfg)
        policy_wrapper = QwenOmniGRPOModel(policy_model, qwen_processor, device)
    else:
        policy_model = load_desta_model(cfg, device)
        policy_wrapper = DeSTA25GRPOModel(policy_model, device)
        qwen_processor = None  # not used

    # ---- Optimizer & scheduler ----
    steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size)
    total_steps     = steps_per_epoch * cfg.epochs
    optimizer, scheduler = build_optimizer(policy_model, cfg, total_steps)

    # Prepare optimizer + policy model with accelerator
    policy_model, optimizer = accel.prepare(policy_model, optimizer)
    # Re-wrap after prepare (accelerator may wrap the model)
    if use_qwen:
        policy_wrapper = QwenOmniGRPOModel(policy_model, qwen_processor, device)
    else:
        policy_wrapper = DeSTA25GRPOModel(policy_model, device)

    # ---- Trainer ----
    trainer = GRPOTrainer(
        policy_wrapper=policy_wrapper,
        accelerator=accel,
        cfg=cfg,
    )
    trainer.set_optimizer(optimizer, scheduler)

    # ---- Run training ----
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dry_run=args.dry_run,
        skip_baseline_eval=args.skip_baseline_eval,
    )

    if accel.is_main_process and not args.dry_run:
        logger.info(f"Training complete. Checkpoints saved to {cfg.output_dir}")

    # ---- Cleanup distributed process group ----
    accel.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
