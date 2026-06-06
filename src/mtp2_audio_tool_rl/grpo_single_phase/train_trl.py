import os
import sys
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import OmegaConf

# Path setup
_grpo_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_grpo_dir)
_desta_dir = os.path.join(os.path.dirname(_project_dir), "DeSTA2.5-Audio")

for _p in [_project_dir, os.path.join(_project_dir, "scripts"), _desta_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from grpo_single_phase.dataset import split_mmau_dataset
from grpo_single_phase.model_wrapper import DeSTA25GRPOModel
from grpo_single_phase.modeling_grpo import GRPODeSTA25AudioModel, _prepare_audio_context_and_start_positions
from grpo_single_phase.prompts import build_grpo_initial_prompt
from grpo_single_phase.rewards_trl import trl_reward_function, set_reward_weights

from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl.trainer.utils import selective_log_softmax, entropy_from_logits
from transformers import TrainerCallback
import json
import re

logger = logging.getLogger(__name__)

def make_trl_dataset(items: List[Dict], audio_root: str, precomputed_embed_dir: str) -> Dataset:
    trl_items = []
    for item in items:
        audio_id = item.get("audio_id", "")
        audio_path = audio_id
        if audio_root:
            audio_path = os.path.join(audio_root, audio_id.lstrip("./"))

        precomputed_embed_path = None
        if precomputed_embed_dir and audio_id:
            filename = os.path.basename(audio_id)
            embed_name = os.path.splitext(filename)[0] + "_embed.pt"
            candidate_path = os.path.join(precomputed_embed_dir, embed_name)
            if os.path.exists(candidate_path):
                precomputed_embed_path = candidate_path

        trl_items.append({
            "prompt": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": "",
                    "audio_path": audio_path,
                    "precomputed_embed": precomputed_embed_path,
                    "cached_tool_outputs": item.get("cached_tool_outputs", {}),
                    "question": item.get("question", ""),
                    "choices": item.get("choices", [])
                }
            ],
            "gold_answer": item.get("answer", ""),
            "choices": item.get("choices", []),
            "id": item.get("id", "")
        })
    return Dataset.from_list(trl_items)


def _string_match(answer: str, prediction: str, choices: list) -> bool:
    """MMAU official string_match metric (mirrors evaluation.py)."""
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(answer)
    if not pred_tokens:
        return False
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != gold_tokens:
            incorrect_tokens.update(choice_tokens - gold_tokens)
    return gold_tokens.issubset(pred_tokens) and pred_tokens.isdisjoint(incorrect_tokens)


class GreedyEvalCallback(TrainerCallback):
    """
    Runs greedy-decoding evaluation on the eval set at end of each epoch.
    Logs task-wise and overall accuracy to the SLURM log.
    """
    def __init__(self, eval_items, audio_root, precomputed_embed_dir, max_new_tokens=2048):
        self.eval_items = eval_items
        self.audio_root = audio_root
        self.precomputed_embed_dir = precomputed_embed_dir
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not kwargs.get("accelerator", None):
            # Try to get accelerator from the trainer stored in kwargs
            pass
        # Only run on main process
        accelerator = kwargs.get("accelerator", None)
        if accelerator and not accelerator.is_main_process:
            return

        logger.info("=" * 60)
        logger.info("  GREEDY EVAL — Epoch %d, Step %d", int(state.epoch or 0), state.global_step)
        logger.info("=" * 60)

        try:
            unwrapped = model
            if hasattr(model, "module"):
                unwrapped = model.module
            if hasattr(unwrapped, "base_model"):
                # PEFT wrapped — use it as-is, LoRA weights are active
                pass

            was_training = unwrapped.training
            was_gc = getattr(unwrapped, "is_gradient_checkpointing", False)
            if was_gc:
                unwrapped.gradient_checkpointing_disable()
            unwrapped.eval()

            device = next(unwrapped.parameters()).device
            desta_wrapper = DeSTA25GRPOModel(unwrapped, device)

            task_metrics = {}
            total_correct, total_count = 0, 0
            eval_results = []

            for item in self.eval_items:
                audio_id = item.get("audio_id", "")
                question = item.get("question", "")
                choices = item.get("choices", [])
                gold = item.get("answer", "")
                task = item.get("task", "unknown")
                cached_tools = item.get("cached_tool_outputs", {})

                audio_path = os.path.join(self.audio_root, audio_id.lstrip("./")) if audio_id else ""
                precomputed_embed = None
                if self.precomputed_embed_dir and audio_id:
                    embed_name = os.path.splitext(os.path.basename(audio_id))[0] + "_embed.pt"
                    candidate = os.path.join(self.precomputed_embed_dir, embed_name)
                    if os.path.exists(candidate):
                        precomputed_embed = candidate

                sr_output = cached_tools.get("speech_recognition", {})
                transcription = sr_output.get("text") if isinstance(sr_output, dict) else None

                sys_p, user_p = build_grpo_initial_prompt(question, choices)

                with torch.no_grad():
                    comps = desta_wrapper.generate_completions(
                        audio_path=audio_path,
                        system_prompt=sys_p,
                        user_prompt=user_p,
                        cached_tool_outputs=cached_tools,
                        question=question,
                        choices=choices,
                        G=1,
                        temperature=0.0,
                        top_p=1.0,
                        max_new_tokens=self.max_new_tokens,
                        transcription=transcription,
                        precomputed_embed=precomputed_embed,
                    )

                comp = comps[0]
                from grpo_single_phase.rewards_trl import extract_answer
                predicted = extract_answer(comp["text"]) or ""
                matched = _string_match(gold, predicted, choices)

                if task not in task_metrics:
                    task_metrics[task] = [0, 0]
                task_metrics[task][1] += 1
                total_count += 1
                if matched:
                    task_metrics[task][0] += 1
                    total_correct += 1

                eval_results.append({
                    "id": item.get("id", ""),
                    "question": question,
                    "gold": gold,
                    "predicted": predicted,
                    "matched": matched,
                    "task": task,
                    "used_tool": comp["called_tools"],
                })

            # Log results
            logger.info("-" * 40)
            logger.info("Task-wise Accuracy:")
            for task_name, (correct, count) in sorted(task_metrics.items()):
                acc = (correct / count * 100) if count > 0 else 0
                logger.info("  %s: %.2f%% (%d/%d)", task_name, acc, correct, count)
            overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0
            logger.info("-" * 40)
            logger.info("Overall: %.2f%% (%d/%d)", overall_acc, total_correct, total_count)
            logger.info("=" * 60)

            # Save eval results to JSONL
            eval_dir = os.path.join(args.output_dir, "greedy_eval")
            os.makedirs(eval_dir, exist_ok=True)
            eval_path = os.path.join(eval_dir, f"eval_step_{state.global_step:05d}.jsonl")
            with open(eval_path, "w") as f:
                for r in eval_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("Greedy eval results saved to %s", eval_path)

            # Restore model state
            if was_gc:
                unwrapped.gradient_checkpointing_enable()
            if was_training:
                unwrapped.train()

        except Exception as e:
            logger.error("Greedy eval failed: %s", e, exc_info=True)


class AudioTRLGRPOTrainer(GRPOTrainer):
    """
    Extending TRL GRPOTrainer to handle DeSTA-specific two-stage generation and pre-computed embeddings.
    """
    def __init__(self, *args, kl_threshold: float = 0.0, negative_tool_ratio: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)

        # Adding KL threshold to prevent sudden spikes during training
        self.kl_threshold = kl_threshold
        self._kl_skips = 0
        if kl_threshold > 0:
            logger.info(f"KL safety threshold enabled: will zero loss when batch KL > {kl_threshold}")

        # Negative tool injection: fraction of tool-calling completions that
        # receive a swapped tool output (same tool, different audio).
        self._negative_tool_ratio = negative_tool_ratio
        if negative_tool_ratio > 0:
            logger.info(f"Negative tool injection enabled: ratio={negative_tool_ratio}")

    def _generate(self, prompts: List[Any]):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # We unwrap the model to access DeSTA methods cleanly, but keep parameters on device
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Mirror TRL's _generate_single_turn: disable gradient checkpointing and
        # switch to eval mode during generation so that KV-cache works correctly.
        # Without this, gradient checkpointing forces use_cache=False and
        # past_key_values=None, producing garbage tokens.
        was_gc = unwrapped_model.is_gradient_checkpointing
        was_training = unwrapped_model.training
        if was_gc:
            unwrapped_model.gradient_checkpointing_disable()
            logger.info("_generate: disabled gradient checkpointing for generation (was_gc=True)")
        if was_training:
            unwrapped_model.eval()

        desta_wrapper = DeSTA25GRPOModel(unwrapped_model, device)
        tokenizer = unwrapped_model.tokenizer

        # TRL's RepeatSampler already repeats each prompt num_generations times and
        # distributes them across GPUs.  _generate must return exactly len(prompts)
        # completions (one per item).  We group by unique audio_path so we only
        # encode audio once, then generate the local count of completions needed.
        from collections import OrderedDict
        unique_groups = OrderedDict()  # audio_path -> list of indices
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
        # Parallel array of negative-tool flags — must reach the reward
        # function via TRL's `extra_fields` so robustness shaping can fire.
        is_negative_tool_list = [False] * N

        batch_features_list = [None] * N
        batch_transcription_ids_list = [None] * N
        batch_start_positions_list = [None] * N
        precomputed_embeds_list = [None] * N

        # Build negative tool pool from all unique samples in this batch.
        # Each entry is one sample's cached_tool_outputs dict.
        negative_tool_pool = []
        for _ap in unique_groups:
            _sample_idx = unique_groups[_ap][0]
            _pool_tools = prompts[_sample_idx][-1].get("cached_tool_outputs", {})
            if _pool_tools:
                negative_tool_pool.append(_pool_tools)

        # Read negative_tool_ratio from config (default 0 = disabled)
        negative_tool_ratio = getattr(self, "_negative_tool_ratio", 0.0)

        for audio_path, indices in unique_groups.items():
            local_G = len(indices)  # how many copies of this prompt on this GPU
            user_msg = prompts[indices[0]][-1]
            precomputed_embed = user_msg["precomputed_embed"]
            question = user_msg["question"]
            choices = user_msg["choices"]
            cached_tools = user_msg["cached_tool_outputs"]

            sr_output = cached_tools.get("speech_recognition", {})
            transcription = sr_output.get("text") if isinstance(sr_output, dict) else None

            sys_p, user_p = build_grpo_initial_prompt(question, choices)

            # Phase 1 & 2: generate with custom wrapper (pure LLM uses vLLM for fast and efficient generation)
            comps = desta_wrapper.generate_completions(
                audio_path=audio_path,
                system_prompt=sys_p,
                user_prompt=user_p,
                cached_tool_outputs=cached_tools,
                question=question,
                choices=choices,
                G=local_G,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_completion_length,
                transcription=transcription,
                precomputed_embed=precomputed_embed,
                negative_tool_pool=negative_tool_pool,
                negative_tool_ratio=negative_tool_ratio,
            )

            # Prepare audio features exactly as expected by logprobs
            messages_p1 = [
                {"role": "system", "content": sys_p},
                {
                    "role": "user",
                    "content": f"<|AUDIO|>\n{user_p}",
                    "audios": [{"audio": audio_path, "text": transcription, "embed_path": precomputed_embed}],
                },
            ]

            if precomputed_embed:
                raw = torch.load(precomputed_embed, map_location="cpu", weights_only=False)
                if isinstance(raw, dict):
                    audio_size = raw["qformer"].size(0)
                    vad = bool(raw.get("vad", False))
                    transcription_size = len(tokenizer.tokenize(transcription, add_special_tokens=False)) if vad and transcription and transcription.strip() else 0
                else:
                    raise ValueError(f"Unexpected format in precomputed embed file: {precomputed_embed}")
                batch_feature = torch.zeros((1, 1, 1), device=device, dtype=torch.bfloat16)
            else:
                raise ValueError(f"Precomputed embed is required but not found for audio: {audio_path}")
                ## Disable this flow bcz this is superrrrrrr slow!! Maybe we can optimize it later. Don't remove it!!!

                # audio_size = unwrapped_model.config.prompt_size
                # transcription_size = len(tokenizer.tokenize(transcription or " ", add_special_tokens=False))
                # from desta.utils.audio import AudioSegment
                # audio_array = AudioSegment.from_file(audio_path, target_sr=16000, channel_selector="average").samples
                # batch_feature = unwrapped_model.processor([audio_array], sampling_rate=16000, return_tensors="pt").input_features.to(device)

            audio_context = tokenizer.apply_chat_template(messages_p1, tokenize=False, add_generation_prompt=True)
            audio_context = audio_context.replace(unwrapped_model.audio_locator, f"<start_audio>{unwrapped_model.audio_locator}<end_audio>")

            audio_context_tokens, start_positions = _prepare_audio_context_and_start_positions(
                token_list=tokenizer.tokenize(audio_context),
                audio_locator=unwrapped_model.audio_locator,
                audio_size_list=[audio_size],
                transcription_size_list=[transcription_size],
                placeholder_token=unwrapped_model.placeholder_token,
            )
            audio_context_str = tokenizer.convert_tokens_to_string(audio_context_tokens)
            prompt_ids = tokenizer(audio_context_str, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
            start_pos = start_positions[0]

            trans_str = transcription if (transcription_size > 0 and transcription) else " "
            trans_ids = tokenizer.encode(trans_str, add_special_tokens=False, return_tensors="pt").long().to(device)

            for j, idx in enumerate(indices):
                comp = comps[j]
                if comp["called_tools"]:
                    # Use apply_chat_template for correct turn-boundary tokens
                    # between phase1, injected tool output, and phase2.
                    # This fixes the Phase 2 context mismatch: the old code
                    # concatenated raw-tokenized pieces without the <|eot_id|>
                    # and turn-header tokens the model actually saw at generation
                    # time, so logprobs were computed on a wrong sequence.
                    # Text-only messages (no audio) — audio only affects the
                    # prompt which cancels out when we take differences.
                    msgs_base = [
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p},
                    ]
                    ids_prompt_text = tokenizer.apply_chat_template(
                        msgs_base, tokenize=True, add_generation_prompt=True)
                    ids_through_p1 = tokenizer.apply_chat_template(
                        msgs_base + [{"role": "assistant", "content": comp["phase1"]}],
                        tokenize=True, add_generation_prompt=False)
                    ids_through_inj = tokenizer.apply_chat_template(
                        msgs_base + [
                            {"role": "assistant", "content": comp["phase1"]},
                            {"role": "user", "content": comp["injected_output"]},
                        ], tokenize=True, add_generation_prompt=True)
                    ids_full = tokenizer.apply_chat_template(
                        msgs_base + [
                            {"role": "assistant", "content": comp["phase1"]},
                            {"role": "user", "content": comp["injected_output"]},
                            {"role": "assistant", "content": comp["phase2"]},
                        ], tokenize=True, add_generation_prompt=False)

                    completion_ids = ids_full[len(ids_prompt_text):]
                    p1_len = len(ids_through_p1) - len(ids_prompt_text)
                    masked_len = len(ids_through_inj) - len(ids_through_p1)
                    p2_len = len(ids_full) - len(ids_through_inj)
                    tool_mask = [1] * p1_len + [0] * masked_len + [1] * p2_len
                else:
                    completion_ids = tokenizer(comp["phase1"], return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
                    tool_mask = [1] * len(completion_ids)

                # Enforce max length
                if len(completion_ids) > self.args.max_completion_length:
                    completion_ids = completion_ids[:self.args.max_completion_length]
                    tool_mask = tool_mask[:self.args.max_completion_length]

                prompt_ids_list[idx] = prompt_ids
                completion_ids_list[idx] = completion_ids
                tool_mask_list[idx] = tool_mask
                completions_list[idx] = [{"role": "assistant", "content": comp["text"]}]
                is_negative_tool_list[idx] = bool(comp.get("is_negative_tool", False))

                batch_features_list[idx] = batch_feature
                batch_transcription_ids_list[idx] = trans_ids
                precomputed_embeds_list[idx] = precomputed_embed
                batch_start_positions_list[idx] = start_pos

        # Store for the logprob calculation
        self._current_audio_kwargs = {
            "batch_features": batch_features_list,
            "batch_transcription_ids": batch_transcription_ids_list,
            "precomputed_embeds": precomputed_embeds_list,
            "raw_start_positions": batch_start_positions_list,
        }

        num_items_in_batch = torch.tensor(
            sum(len(c) for c in completion_ids_list), device=device
        )

        # Restore model state after generation
        if was_gc:
            unwrapped_model.gradient_checkpointing_enable()
        if was_training:
            unwrapped_model.train()

        return (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions_list,
            num_items_in_batch,
            None,
            # Forward per-completion flags to reward functions. TRL merges
            # extra_fields entries into inputs[i][key] and then _calculate_rewards
            # exposes them as a list kwarg of the same name.
            {"is_negative_tool": is_negative_tool_list},
        )

    def _generate_and_score_completions(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Capture per-question metadata before super() repeats & scatters them.
        # Each GPU only sees its local slice of prompts, so we capture locally
        # then gather across GPUs after the super call.
        local_extras = [
            {
                "gold_answer": inp.get("gold_answer", ""),
                "choices": str(inp.get("choices", [])),
                "id": inp.get("id", ""),
            }
            for inp in inputs
        ]

        output = super()._generate_and_score_completions(inputs)

        # Barrier: ensure all ranks finished reward scoring before gather.
        # Without this, ranks that finish judge HTTP calls faster race ahead
        # to the gather collective while slow ranks are still scoring,
        # causing NCCL timeout deadlocks.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Gather metadata from all GPUs so main process has the full set.
        # gather_object returns a flat list: [gpu0_item0, gpu1_item0, gpu2_item0, ...]
        from accelerate.utils import gather_object
        self._parquet_extras_per_question = gather_object(local_extras)

        # We need to extract the audio features from self._current_audio_kwargs
        # and store them in the output batch so that when TRL splits the batch
        # across multiple optimization steps, the features get sliced correctly.
        if hasattr(self, "_current_audio_kwargs"):
            # Pad batch_transcription_ids to avoid DDP crashes
            trans_ids_list = self._current_audio_kwargs["batch_transcription_ids"]
            if len(trans_ids_list) > 0:
                max_len = max(t.size(1) for t in trans_ids_list) if trans_ids_list[0].dim() == 2 else max(t.size(0) for t in trans_ids_list)
                if max_len == 0:
                    max_len = 1

                padded_trans_ids = []
                trans_lengths = []
                for t in trans_ids_list:
                    if t.dim() == 2:
                        t = t.squeeze(0)
                    length = t.size(0)
                    trans_lengths.append(length)
                    if length < max_len:
                        pad_tensor = torch.zeros(max_len - length, dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad_tensor], dim=0)
                    elif length > max_len:
                        t = t[:max_len]
                    padded_trans_ids.append(t)

                output["batch_transcription_ids"] = torch.stack(padded_trans_ids, dim=0)
                output["batch_transcription_lengths"] = torch.tensor(trans_lengths, device=self.accelerator.device)
            else:
                output["batch_transcription_ids"] = torch.empty((0, 1), dtype=torch.long, device=self.accelerator.device)
                output["batch_transcription_lengths"] = torch.empty((0,), dtype=torch.long, device=self.accelerator.device)

            output["batch_features"] = self._current_audio_kwargs["batch_features"]
            output["precomputed_embeds"] = self._current_audio_kwargs["precomputed_embeds"]
            output["raw_start_positions"] = self._current_audio_kwargs["raw_start_positions"]

        return output

    def _compute_loss(self, model, inputs):
        self._current_sliced_inputs = inputs
        try:
            loss = super()._compute_loss(model, inputs)
        finally:
            self._current_sliced_inputs = None

        # Smooth KL-adaptive loss scaling: instead of a hard zero/full binary,
        # linearly scale down the loss as KL approaches & exceeds the threshold.
        # This provides a gentle braking effect rather than sudden loss of signal.
        #   KL < 0.5*threshold → full loss (scale=1.0)
        #   KL = threshold     → half loss (scale=0.5)
        #   KL > 2*threshold   → zero loss (scale=0.0)
        if self.kl_threshold > 0 and self.model.training:
            mode = "train"
            kl_vals = self._metrics.get(mode, {}).get("kl", [])
            if kl_vals:
                latest_kl = kl_vals[-1]
                soft_start = self.kl_threshold * 0.5
                hard_stop = self.kl_threshold * 2.0
                if latest_kl > soft_start:
                    # Linear ramp from 1.0 at soft_start to 0.0 at hard_stop
                    scale = max(0.0, 1.0 - (latest_kl - soft_start) / (hard_stop - soft_start))
                    loss = loss * scale
                    if scale == 0.0:
                        self._kl_skips += 1
                    if latest_kl > self.kl_threshold:
                        logger.warning(
                            f"KL braking: batch KL={latest_kl:.2f} > threshold={self.kl_threshold}. "
                            f"Loss scaled by {scale:.2f} (total zeros: {self._kl_skips})."
                        )

        return loss

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, **kwargs):
        """
        Override to pass audio features to the DeSTA model forward pass correctly.
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        # Determine if we are in _compute_loss (sliced) or _generate_and_score_completions (unsliced)
        if hasattr(self, "_current_sliced_inputs") and self._current_sliced_inputs is not None:
            audio_inputs = self._current_sliced_inputs
            features_list = audio_inputs["batch_features"]
            trans_ids_tensor = audio_inputs["batch_transcription_ids"]
            trans_lengths = audio_inputs["batch_transcription_lengths"]
            embeds_list = audio_inputs["precomputed_embeds"]
            start_pos_list = audio_inputs["raw_start_positions"]

            # Reconstruct list of unpadded 1D tensors (DeSTA expects 1D or 1xT)
            trans_ids_list = []
            for i in range(trans_ids_tensor.size(0)):
                length = trans_lengths[i].item()
                # Keep it as 2D (1, T) to match what DeSTA's fallback expects
                trans_ids_list.append(trans_ids_tensor[i, :length].unsqueeze(0))
        else:
            audio_inputs = self._current_audio_kwargs
            features_list = audio_inputs["batch_features"]
            trans_ids_list = audio_inputs["batch_transcription_ids"]
            embeds_list = audio_inputs["precomputed_embeds"]
            start_pos_list = audio_inputs["raw_start_positions"]

        prompt_length = input_ids.size(1) - logits_to_keep
        prompt_mask = attention_mask[:, :prompt_length]
        # Pad length is exactly the number of zeros in the prompt mask (since we use padding_side="left" for prompts)
        pad_lengths = (prompt_length - prompt_mask.sum(dim=1)).tolist()

        for start in range(0, input_ids.size(0), batch_size):
            end = start + batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]
            actual_bsz = input_ids_batch.size(0)

            # Prepare batch features
            batch_feats = features_list[start:end]
            if len(batch_feats) > 0 and batch_feats[0].dim() == 3:
                batch_features_tensor = torch.cat(batch_feats, dim=0)
            else:
                batch_features_tensor = None

            batch_trans_ids = trans_ids_list[start:end]
            batch_embeds = embeds_list[start:end]

            # Adjust audio starting positions by considering the padding applied
            batch_start_positions = []
            for i in range(actual_bsz):
                global_idx = start + i
                pad_len = pad_lengths[global_idx]
                raw_sp = start_pos_list[global_idx]
                batch_start_positions.append((i, raw_sp + int(pad_len)))

            # Call DeSTA forward
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                batch_features=batch_features_tensor,
                batch_transcription_ids=batch_trans_ids,
                batch_start_positions=batch_start_positions,
                precomputed_embeds=batch_embeds,
            ).logits

            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.args.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]

            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                all_entropies.append(entropy_from_logits(logits))

        return torch.cat(all_logps, dim=0), torch.cat(all_entropies, dim=0) if compute_entropy else None

    def log(self, logs, *args, **kwargs):
        """Override to enrich parquet with gold_answer/choices/id + GRPO metrics,
        and catch Rich rendering crashes from gibberish Unicode."""
        # Before super().log() writes the parquet, inject extra columns into _logs
        # so they end up in the parquet file automatically.
        if hasattr(self, "_parquet_extras_per_question") and self._parquet_extras_per_question:
            n_prompts = len(self._logs["prompt"])  # already gathered, total across GPUs
            extras = self._parquet_extras_per_question  # already gathered across GPUs
            n_unique = len(extras)  # number of unique prompts across all GPUs
            G = self.num_generations
            # Each unique prompt has G completions; repeat each entry G times
            if n_unique > 0 and n_prompts > 0:
                expanded = [e for e in extras for _ in range(G)]
                # Trim or pad to exact length in case of mismatch
                if len(expanded) < n_prompts:
                    expanded = expanded + [extras[-1]] * (n_prompts - len(expanded))
                expanded = expanded[:n_prompts]
                self._logs["gold_answer"] = [e["gold_answer"] for e in expanded]
                self._logs["choices"] = [e["choices"] for e in expanded]
                self._logs["id"] = [e["id"] for e in expanded]
            self._parquet_extras_per_question = []

        # Inject GRPO aggregate metrics into _logs so they appear in parquet
        mode = "train" if self.model.training else "eval"
        if self._metrics[mode]:
            metrics_snapshot = {key: sum(val) / len(val) for key, val in self._metrics[mode].items() if len(val) > 0}
            self._logs["_metrics"] = metrics_snapshot  # stash for _write_enriched_parquet

        # Snapshot logs before super().log() clears them
        self._snapshot_logs_for_jsonl()

        try:
            # Let TRL write the standard parquet + wandb
            super().log(logs, *args, **kwargs)
        except RuntimeError as e:
            if "StopIteration" in str(e):
                logger.warning("Skipped completion table render (Rich crashed on Unicode). Metrics still logged.")
            else:
                raise
        finally:
            # Write JSONL completions (easier to read than parquet)
            self._write_completions_jsonl()

    def _snapshot_logs_for_jsonl(self):
        """Capture completion data before super().log() might clear it."""
        self._jsonl_snapshot = {}
        if not hasattr(self, "_logs") or "prompt" not in self._logs:
            return
        self._jsonl_snapshot = {
            "prompts": list(self._logs.get("prompt", [])),
            "completions": list(self._logs.get("completion", [])),
            "advantages": list(self._logs.get("advantages", [])),
            "rewards": {k: list(v) for k, v in self._logs.get("rewards", {}).items()},
        }
        for col in ("gold_answer", "choices", "id"):
            if col in self._logs:
                self._jsonl_snapshot[col] = list(self._logs[col])
        if "_metrics" in self._logs:
            self._jsonl_snapshot["metrics"] = self._logs["_metrics"]

    def _write_completions_jsonl(self):
        """Write completions as JSONL — one JSON object per completion, human-readable."""
        if not self.accelerator.is_main_process or not self.log_completions:
            return
        snapshot = getattr(self, "_jsonl_snapshot", {})
        if not snapshot or not snapshot.get("prompts"):
            return

        completions_dir = os.path.join(self.args.output_dir, "completions")
        os.makedirs(completions_dir, exist_ok=True)
        jsonl_path = os.path.join(completions_dir, f"completions_{self.state.global_step:05d}.jsonl")

        try:
            n = len(snapshot["prompts"])
            metrics = snapshot.get("metrics", {})
            with open(jsonl_path, "w") as f:
                for i in range(n):
                    # Extract just the question from the full prompt to save space
                    full_prompt = snapshot["prompts"][i]
                    question = full_prompt
                    q_start = full_prompt.rfind("Question: ")
                    if q_start >= 0:
                        question = full_prompt[q_start:].strip()
                    record = {
                        "step": self.state.global_step,
                        "prompt": question,
                        "completion": snapshot["completions"][i],
                        "advantage": snapshot["advantages"][i] if i < len(snapshot["advantages"]) else None,
                    }
                    # Add per-reward-function scores
                    for rk, rv in snapshot.get("rewards", {}).items():
                        if i < len(rv):
                            record[rk] = rv[i]
                    # Add metadata
                    for col in ("gold_answer", "choices", "id"):
                        if col in snapshot and i < len(snapshot[col]):
                            record[col] = snapshot[col][i]
                    # Add aggregate metrics (same for all rows in this step)
                    if metrics:
                        record["metrics"] = metrics
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL completions: {e}")

        # Clean up
        self._jsonl_snapshot = {}
        for col in ("gold_answer", "choices", "id", "_metrics"):
            self._logs.pop(col, None)

def main():
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRPO_CONFIG", "grpo/configs/optimized.yaml")
    logger.info(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    data_file = os.path.join(_grpo_dir, cfg.data_file)
    with open(data_file, "r") as f:
        all_data = json.load(f)

    # Filter valid items
    all_data = [item for item in all_data if item.get("task") and item.get("question") and item.get("answer")]

    splits_dir = os.path.join(_grpo_dir, cfg.splits_dir)
    train_items, eval_items, test_items = split_mmau_dataset(
        data_file=data_file, splits_dir=splits_dir, train_ratio=cfg.train_ratio, eval_ratio=cfg.eval_ratio, seed=cfg.data_seed
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

    GRPODeSTA25AudioModel.skip_perception = bool(precomputed_embed_dir and os.path.isdir(precomputed_embed_dir))
    model = GRPODeSTA25AudioModel.from_pretrained(cfg.model_hf_name, torch_dtype=torch.bfloat16)
    model._setup_generation()

    if precomputed_embed_dir and os.path.isdir(precomputed_embed_dir):
        model.load_embed_cache(precomputed_embed_dir)

    # Propagate token IDs from the inner Llama config to the outer DeSTA config
    # so TRL / generate() see proper eos/bos/pad without hardcoded values.
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

    # Add dummy prepare_inputs_for_generation to avoid PEFT error during initialization
    if not hasattr(model, "prepare_inputs_for_generation"):
        model.prepare_inputs_for_generation = lambda *args, **kwargs: {}

    # Freeze perception
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

    import math
    import datetime
    num_gpus = int(os.environ.get("WORLD_SIZE", cfg.get("num_gpus", 1)))

    slurm_id = os.environ.get("SLURM_JOB_ID")
    if slurm_id:
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-{slurm_id}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dynamic_output_dir = os.path.join(os.path.dirname(cfg.output_dir), f"grpo-desta-{timestamp}")

    training_args = GRPOConfig(
        output_dir=dynamic_output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=max(1, cfg.G // num_gpus),
        num_generations=cfg.G,
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
        log_completions=True,
        num_completions_to_print=0,  # suppress Rich table in SLURM log; all G completions saved to JSONL
        temperature=cfg.temperature,
        max_grad_norm=cfg.max_grad_norm,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
    )

    # Set reward weights from config
    set_reward_weights(
        fw=cfg.get("format_reward_weight", 0.10),
        cw=cfg.get("correctness_reward_weight", 0.60),
        mw=cfg.get("option_mention_weight", 0.15),
        tw=cfg.get("tool_bonus_weight", 0.15),
    )

    trainer = AudioTRLGRPOTrainer(
        model=model,
        reward_funcs=[trl_reward_function],
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

    logger.info("TRL GRPOTrainer initialized for DeSTA tool-use tasks.")

    # Redirect TRL's rich-table completion output to a separate log file.
    # TRL prints via rich Console; we redirect its file to keep SLURM log clean.
    completions_log = os.path.join(dynamic_output_dir, "completions.log")
    os.makedirs(dynamic_output_dir, exist_ok=True)
    try:
        from rich.console import Console
        _comp_file = open(completions_log, "w")
        trainer._console = Console(file=_comp_file, width=200, force_terminal=False)
        logger.info(f"TRL completion rollouts will be logged to {completions_log}")
    except Exception:
        pass

    trainer.train()

if __name__ == "__main__":
    main()