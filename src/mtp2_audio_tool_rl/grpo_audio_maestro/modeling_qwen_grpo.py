"""
modeling_qwen_grpo.py — Load Qwen2.5-Omni for GRPO training.

Handles:
  - Loading the model with flash_attention_2 for speed
  - Disabling the Talker module (saves ~2 GB, we only need text output)
  - Providing a unified load_qwen_model() entry point for train_grpo.py
"""

import logging
import torch

logger = logging.getLogger(__name__)


def load_qwen_model(cfg, device):
    """
    Load Qwen2.5-Omni model + processor for GRPO training.

    Returns:
        (model, processor) tuple.
    """
    try:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    except ImportError:
        raise ImportError(
            "Qwen2.5-Omni requires transformers>=4.51.3. Install with:\n"
            "  pip install transformers>=4.52.3 accelerate qwen-omni-utils[decord]"
        )

    model_hf = cfg.model_hf_name
    logger.info(f"Loading Qwen2.5-Omni model: {model_hf}")

    # Determine attention implementation
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        logger.info(f"Using flash_attention_2 (flash-attn {flash_attn.__version__})")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash-attn not available, falling back to SDPA")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_hf,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=None,  # we handle device placement ourselves
    )
    model.to(device)

    # Disable the Talker (speech synthesis) module — saves ~2 GB VRAM.
    # We only need text output for GRPO training.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
        logger.info("Qwen2.5-Omni: Talker module disabled (text-only mode, saves ~2 GB)")

    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_hf)

    logger.info("Qwen2.5-Omni model loaded successfully")
    return model, processor


def enable_lora_qwen(model, cfg):
    """
    Attach LoRA adapters to the Qwen2.5-Omni Thinker (LLM backbone).

    For Qwen2.5-Omni, the thinker is the main language model.
    We apply LoRA to its attention projection layers.

    Args:
        model: Qwen2_5OmniForConditionalGeneration instance.
        cfg:   OmegaConf config with lora_rank, lora_alpha, etc.

    Returns:
        The model with LoRA enabled on the thinker.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # The thinker is the LLM backbone of Qwen2.5-Omni
    thinker = model.thinker if hasattr(model, "thinker") else model

    peft_model = get_peft_model(thinker, lora_cfg)

    # Store references for saving and reference log-probs
    model._peft_model      = peft_model                    # PeftModel — has save_pretrained()
    model._lora_base_model = peft_model.base_model          # LoraModel — has disable/enable_adapter_layers()

    # Replace thinker with the LoRA-wrapped version's inner model
    # so that .model.embed_tokens etc. still work
    if hasattr(model, "thinker"):
        model.thinker = peft_model.base_model.model
    model._lora_thinker = peft_model  # keep reference to the full PeftModel

    logger.info(
        f"LoRA enabled on Qwen thinker: rank={cfg.lora_rank}, "
        f"alpha={cfg.lora_alpha}, targets={list(cfg.lora_target_modules)}"
    )

    # Gradient checkpointing
    thinker_inner = peft_model.base_model.model
    if cfg.get("gradient_checkpointing", False):
        thinker_inner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(thinker_inner, "config"):
            thinker_inner.config.use_cache = False
        logger.info("Gradient checkpointing enabled on Qwen thinker (use_reentrant=False)")

    # Freeze everything except LoRA params
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)
    # Re-enable LoRA params (get_peft_model already does this, but be safe)
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {n_trainable:,} / {n_total:,} "
        f"({100*n_trainable/n_total:.2f}%)"
    )

    return model
