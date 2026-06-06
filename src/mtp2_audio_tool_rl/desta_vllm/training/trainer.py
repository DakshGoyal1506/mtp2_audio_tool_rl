"""
VLLMAudioGRPOTrainer — AudioTRLGRPOTrainer with vLLM-accelerated rollouts.

Architecture:
  - vLLM engine: fast batched sampling for GRPO rollouts (no grad)
  - HF model (DeSTA + LoRA): gradient computation for policy optimization
  - After each optimization step, LoRA weights are synced HF → vLLM

This class overrides _generate() to use VLLMRolloutEngine from
desta_vllm/grpo_rollout.py instead of the sequential HF generate.
Everything else (logprobs, loss, rewards, eval) is inherited from
AudioTRLGRPOTrainer.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class VLLMAudioGRPOTrainer:
    """
    Mixin-style class that wraps AudioTRLGRPOTrainer to replace the rollout
    generation with vLLM-backed VLLMRolloutEngine.

    Usage:
        from desta_vllm.training.trainer import create_vllm_grpo_trainer
        TrainerClass = create_vllm_grpo_trainer(AudioTRLGRPOTrainer)
        trainer = TrainerClass(model=..., vllm_engine=engine, ...)
    """
    pass


def create_vllm_grpo_trainer(base_class):
    """
    Factory that creates a VLLMAudioGRPOTrainer class extending the given
    base AudioTRLGRPOTrainer with vLLM rollout support.

    This avoids import-time circular dependencies since AudioTRLGRPOTrainer
    lives in grpo_single_phase/ which may not be on sys.path yet.
    """

    class VLLMAudioGRPOTrainerImpl(base_class):
        """
        AudioTRLGRPOTrainer with vLLM-accelerated rollouts.

        Overrides:
          - _generate(): uses VLLMRolloutEngine instead of HF model.generate()
          - Adds LoRA weight sync (HF → vLLM) after each optimization step.
        """

        def __init__(self, *args, vllm_engine=None, **kwargs):
            """
            Args:
                vllm_engine: DeSTAVLLMEngine instance (already initialized).
                All other args passed to base AudioTRLGRPOTrainer.
            """
            super().__init__(*args, **kwargs)
            if vllm_engine is None:
                raise ValueError("vllm_engine is required for VLLMAudioGRPOTrainer")

            from desta_vllm.grpo_rollout import VLLMRolloutEngine
            self._vllm_engine = vllm_engine
            self._vllm_rollout = VLLMRolloutEngine(vllm_engine)
            self._lora_sync_dir = None
            logger.info("VLLMAudioGRPOTrainer initialized with vLLM rollout engine")

        def _sync_lora_to_vllm(self):
            """
            Save current LoRA weights from HF model and load them into vLLM.

            This is called after each optimization step to keep the rollout
            policy in sync with the training policy.
            """
            unwrapped = self.accelerator.unwrap_model(self.model)

            # Save LoRA adapter to temp dir
            if self._lora_sync_dir is None:
                self._lora_sync_dir = tempfile.mkdtemp(prefix="vllm_lora_sync_")
                logger.info(f"LoRA sync directory: {self._lora_sync_dir}")

            try:
                # PEFT models have save_pretrained on the PeftModel wrapper
                if hasattr(unwrapped, "save_pretrained"):
                    unwrapped.save_pretrained(self._lora_sync_dir)
                elif hasattr(unwrapped, "base_model") and hasattr(unwrapped.base_model, "save_pretrained"):
                    unwrapped.base_model.save_pretrained(self._lora_sync_dir)
                else:
                    logger.warning("Cannot find save_pretrained on model — skipping LoRA sync")
                    return

                self._vllm_engine.update_lora(self._lora_sync_dir)
                logger.debug("LoRA weights synced HF → vLLM")
            except Exception as e:
                logger.error(f"LoRA sync failed: {e}", exc_info=True)

        def _generate(self, prompts: List[Any]):
            """
            Override: use vLLM for rollout generation instead of HF model.

            The prompt format from TRL is a list of message dicts. We extract
            audio metadata and use VLLMRolloutEngine.generate_completions().

            Returns the same tuple format as AudioTRLGRPOTrainer._generate():
                (prompt_ids, completion_ids, tool_mask, completions,
                 num_items_in_batch, None, extra_fields)
            """
            device = self.accelerator.device
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            tokenizer = unwrapped_model.tokenizer

            # Wake vLLM (no-op if sleep mode disabled). Sync LoRA before generation.
            self._vllm_engine.wake_up()
            self._sync_lora_to_vllm()

            # Mirror base: disable grad checkpointing for generation
            was_gc = unwrapped_model.is_gradient_checkpointing
            was_training = unwrapped_model.training
            if was_gc:
                unwrapped_model.gradient_checkpointing_disable()
            if was_training:
                unwrapped_model.eval()

            from collections import OrderedDict
            from grpo_single_phase.prompts import build_grpo_initial_prompt
            from grpo_single_phase.modeling_grpo import _prepare_audio_context_and_start_positions

            # Group by audio_path (same as base)
            unique_groups = OrderedDict()
            for idx, prompt in enumerate(prompts):
                audio_path = prompt[-1]["audio_path"]
                if audio_path not in unique_groups:
                    unique_groups[audio_path] = []
                unique_groups[audio_path].append(idx)

            N = len(prompts)
            prompt_ids_list = [None] * N
            completion_ids_list = [None] * N
            tool_mask_list = [None] * N
            completions_list = [None] * N
            is_negative_tool_list = [False] * N

            batch_features_list = [None] * N
            batch_transcription_ids_list = [None] * N
            batch_start_positions_list = [None] * N
            precomputed_embeds_list = [None] * N

            # Build negative tool pool
            negative_tool_pool = []
            for _ap in unique_groups:
                _sample_idx = unique_groups[_ap][0]
                _pool_tools = prompts[_sample_idx][-1].get("cached_tool_outputs", {})
                if _pool_tools:
                    negative_tool_pool.append(_pool_tools)

            negative_tool_ratio = getattr(self, "_negative_tool_ratio", 0.0)

            for audio_path, indices in unique_groups.items():
                local_G = len(indices)
                user_msg = prompts[indices[0]][-1]
                precomputed_embed = user_msg["precomputed_embed"]
                question = user_msg["question"]
                choices = user_msg["choices"]
                cached_tools = user_msg["cached_tool_outputs"]

                sr_output = cached_tools.get("speech_recognition", {})
                transcription = sr_output.get("text") if isinstance(sr_output, dict) else None

                sys_p, user_p = build_grpo_initial_prompt(question, choices)

                # ── vLLM rollout instead of HF model.generate() ──
                embed_path = precomputed_embed
                if embed_path is None:
                    raise ValueError(f"Precomputed embed required for vLLM rollout: {audio_path}")

                comps = self._vllm_rollout.generate_completions(
                    embed_path=embed_path,
                    transcription=transcription,
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    cached_tool_outputs=cached_tools,
                    question=question,
                    choices=choices,
                    G=local_G,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_new_tokens=self.args.max_completion_length,
                    negative_tool_pool=negative_tool_pool,
                    negative_tool_ratio=negative_tool_ratio,
                )

                # ── Build audio context for logprob computation (same as base) ──
                messages_p1 = [
                    {"role": "system", "content": sys_p},
                    {
                        "role": "user",
                        "content": f"<|AUDIO|>\n{user_p}",
                        "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
                    },
                ]

                raw = torch.load(precomputed_embed, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    audio_size = raw["qformer"].size(0)
                    vad = bool(raw.get("vad", False))
                    transcription_size = (
                        len(tokenizer.tokenize(transcription, add_special_tokens=False))
                        if vad and transcription and transcription.strip()
                        else 0
                    )
                else:
                    raise ValueError(f"Unexpected format in precomputed embed: {precomputed_embed}")
                batch_feature = torch.zeros((1, 1, 1), device=device, dtype=torch.bfloat16)

                audio_context = tokenizer.apply_chat_template(
                    messages_p1, tokenize=False, add_generation_prompt=True
                )
                audio_context = audio_context.replace(
                    unwrapped_model.audio_locator,
                    f"<start_audio>{unwrapped_model.audio_locator}<end_audio>",
                )

                audio_context_tokens, start_positions = _prepare_audio_context_and_start_positions(
                    token_list=tokenizer.tokenize(audio_context),
                    audio_locator=unwrapped_model.audio_locator,
                    audio_size_list=[audio_size],
                    transcription_size_list=[transcription_size],
                    placeholder_token=unwrapped_model.placeholder_token,
                )
                audio_context_str = tokenizer.convert_tokens_to_string(audio_context_tokens)
                prompt_ids = tokenizer(
                    audio_context_str, return_tensors="pt", add_special_tokens=False
                ).input_ids[0].tolist()
                start_pos = start_positions[0]

                trans_str = transcription if (transcription_size > 0 and transcription) else " "
                trans_ids = tokenizer.encode(
                    trans_str, add_special_tokens=False, return_tensors="pt"
                ).long().to(device)

                # ── Tokenize completions for logprob computation (same as base) ──
                for j, idx in enumerate(indices):
                    comp = comps[j]
                    if comp["called_tools"]:
                        msgs_base = [
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": user_p},
                        ]
                        ids_prompt_text = tokenizer.apply_chat_template(
                            msgs_base, tokenize=True, add_generation_prompt=True
                        )
                        ids_through_p1 = tokenizer.apply_chat_template(
                            msgs_base + [{"role": "assistant", "content": comp["phase1"]}],
                            tokenize=True, add_generation_prompt=False,
                        )
                        ids_through_inj = tokenizer.apply_chat_template(
                            msgs_base + [
                                {"role": "assistant", "content": comp["phase1"]},
                                {"role": "user", "content": comp["injected_output"]},
                            ],
                            tokenize=True, add_generation_prompt=True,
                        )
                        ids_full = tokenizer.apply_chat_template(
                            msgs_base + [
                                {"role": "assistant", "content": comp["phase1"]},
                                {"role": "user", "content": comp["injected_output"]},
                                {"role": "assistant", "content": comp["phase2"]},
                            ],
                            tokenize=True, add_generation_prompt=False,
                        )

                        completion_ids = ids_full[len(ids_prompt_text):]
                        p1_len = len(ids_through_p1) - len(ids_prompt_text)
                        masked_len = len(ids_through_inj) - len(ids_through_p1)
                        p2_len = len(ids_full) - len(ids_through_inj)
                        tool_mask = [1] * p1_len + [0] * masked_len + [1] * p2_len
                    else:
                        completion_ids = tokenizer(
                            comp["phase1"], return_tensors="pt", add_special_tokens=False
                        ).input_ids[0].tolist()
                        tool_mask = [1] * len(completion_ids)

                    # Enforce max length
                    if len(completion_ids) > self.args.max_completion_length:
                        completion_ids = completion_ids[: self.args.max_completion_length]
                        tool_mask = tool_mask[: self.args.max_completion_length]

                    prompt_ids_list[idx] = prompt_ids
                    completion_ids_list[idx] = completion_ids
                    tool_mask_list[idx] = tool_mask
                    completions_list[idx] = [{"role": "assistant", "content": comp["text"]}]
                    is_negative_tool_list[idx] = bool(comp.get("is_negative_tool", False))

                    batch_features_list[idx] = batch_feature
                    batch_transcription_ids_list[idx] = trans_ids
                    precomputed_embeds_list[idx] = precomputed_embed
                    batch_start_positions_list[idx] = start_pos

            # Store for logprob calculation
            self._current_audio_kwargs = {
                "batch_features": batch_features_list,
                "batch_transcription_ids": batch_transcription_ids_list,
                "precomputed_embeds": precomputed_embeds_list,
                "raw_start_positions": batch_start_positions_list,
            }

            num_items_in_batch = torch.tensor(
                sum(len(c) for c in completion_ids_list), device=device
            )

            # Restore model state
            if was_gc:
                unwrapped_model.gradient_checkpointing_enable()
            if was_training:
                unwrapped_model.train()

            # Put vLLM back to sleep to free GPU memory for the gradient step
            # (no-op if sleep mode disabled). Must be after we collect all
            # completions but before backward pass on the HF model.
            self._vllm_engine.sleep(level=2)

            return (
                prompt_ids_list,
                completion_ids_list,
                tool_mask_list,
                completions_list,
                num_items_in_batch,
                None,
                {"is_negative_tool": is_negative_tool_list},
                None,  # images  (TRL 1.4.0)
                None,  # tool_images  (TRL 1.4.0)
            )

    # Give it a proper name for debugging
    VLLMAudioGRPOTrainerImpl.__name__ = "VLLMAudioGRPOTrainer"
    VLLMAudioGRPOTrainerImpl.__qualname__ = "VLLMAudioGRPOTrainer"
    return VLLMAudioGRPOTrainerImpl


class LoRASyncCallback(TrainerCallback):
    """
    Callback that syncs LoRA weights from HF model → vLLM engine.

    In colocate mode with sleep, the sync happens inside ``_generate`` (right
    after vLLM is woken up and before rollout), so this callback is a no-op
    by default. It is kept for back-compat and to allow forcing a sync at
    arbitrary points (e.g. after checkpoint resume).
    """

    def __init__(self, trainer, sync_on_step_end: bool = False):
        self._trainer = trainer
        self._sync_on_step_end = sync_on_step_end

    def on_step_end(self, args, state, control, **kwargs):
        """Optionally sync LoRA weights after each gradient step.

        Disabled by default in colocate+sleep mode: vLLM is asleep at this
        point, so the sync would happen while weights are offloaded. Sync is
        done inside ``_generate`` instead.
        """
        if self._sync_on_step_end and hasattr(self._trainer, "_sync_lora_to_vllm"):
            self._trainer._sync_lora_to_vllm()
