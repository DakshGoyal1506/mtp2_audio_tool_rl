"""
DeSTAVLLMEngine — core vLLM-backed inference engine for DeSTA2.5-Audio.

Accepts precomputed QFormer embeddings, builds prompt_embeds tensors, and
generates via vLLM's offline LLM API with continuous batching + paged KV.

This module is backend-agnostic at the public API level: callers pass
embed_path + transcription + prompts and get back generated text.

Supports:
  - Single and batched generation with n=G completions per prompt
  - LoRA adapter hot-swap (for GRPO training rollouts)
  - Two-phase tool-calling generation (Phase 1 → inject tool output → Phase 2)
  - Both vLLM offline (colocated) and OpenAI-compatible server modes
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from desta_vllm.embed_utils import (
    AUDIO_LOCATOR,
    DEFAULT_LLM_MODEL_ID,
    PLACEHOLDER_TOKEN,
    _resolve_model_path,
    _setup_tokenizer,
    build_prompt_embeds,
    build_prompt_embeds_continuation,
    load_embed_cache,
    resolve_embed_path,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Result of a single generation request."""
    text: str
    token_ids: List[int] = field(default_factory=list)
    phase1: str = ""
    phase2: str = ""
    called_tools: bool = False
    tool_name: Optional[str] = None
    injected_output: Optional[str] = None


@dataclass
class BatchGenerationResult:
    """Result of a batch generation request (one prompt, G completions)."""
    results: List[GenerationResult] = field(default_factory=list)


# ── Tool-call parsing (reused from model_wrapper.py) ─────────────────────────

def _parse_tool_call(text: str) -> Optional[str]:
    """Return tool function name if <tool>...</tool> block found."""
    m = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
    if not m:
        return None
    try:
        arr = json.loads(m.group(1).strip())
        if isinstance(arr, list) and arr:
            return arr[0].get("function")
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _format_tool_output(tool_name: str, tool_result: Any) -> str:
    """Render a <tool_output> block."""
    result_str = (
        json.dumps(tool_result, indent=2)
        if not isinstance(tool_result, str)
        else tool_result
    )
    return f"\n<tool_output>\n{result_str}\n</tool_output>\n"


# ── Engine ────────────────────────────────────────────────────────────────────

class DeSTAVLLMEngine:
    """
    vLLM-backed generation engine for DeSTA2.5-Audio.

    Initializes a vLLM LLM instance with the Llama-3.1 backbone, loads
    precomputed embeddings, and provides generate / generate_batch methods
    that accept audio embed paths instead of raw audio.

    Args:
        llm_model_id: HuggingFace model ID for the Llama-3.1 backbone.
        embed_dir: Directory containing precomputed *_embed.pt files.
        lora_path: Optional initial LoRA adapter path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        max_model_len: Maximum sequence length.
        dtype: Data type for the model.
        seed: Random seed.
        enforce_eager: Disable CUDA graphs (useful for debugging).
        enable_lora: Whether to enable LoRA adapter support.
        max_lora_rank: Maximum LoRA rank.
        trust_remote_code: Trust remote code in model config.
        quantization: Quantization method (e.g. "awq", "gptq", None).
    """

    def __init__(
        self,
        llm_model_id: str = DEFAULT_LLM_MODEL_ID,
        embed_dir: Optional[str] = None,
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        dtype: str = "bfloat16",
        seed: int = 42,
        enforce_eager: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        trust_remote_code: bool = True,
        quantization: Optional[str] = None,
        distributed_executor_backend: Optional[str] = None,
        enable_sleep_mode: bool = False,
    ):
        # Resolve HF hub ID → local snapshot path (works offline)
        self.llm_model_id = _resolve_model_path(llm_model_id)
        self.embed_dir = embed_dir
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = _setup_tokenizer(self.llm_model_id)

        # ── Embed cache ───────────────────────────────────────────────────
        self.embed_cache: Dict[str, Dict] = {}
        if embed_dir and os.path.isdir(embed_dir):
            self.embed_cache = load_embed_cache(embed_dir, device="cpu")

        # ── vLLM engine (load model weights ONCE) ────────────────────────
        from vllm import LLM, SamplingParams  # noqa: E402

        self.SamplingParams = SamplingParams

        engine_kwargs: Dict[str, Any] = dict(
            model=self.llm_model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            seed=seed,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            enable_prompt_embeds=True,
        )
        if distributed_executor_backend:
            engine_kwargs["distributed_executor_backend"] = distributed_executor_backend
        if enable_sleep_mode:
            engine_kwargs["enable_sleep_mode"] = True
        if quantization:
            engine_kwargs["quantization"] = quantization
        if enable_lora:
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_lora_rank"] = max_lora_rank

        logger.info(f"Initializing vLLM engine: {engine_kwargs}")
        self.llm = LLM(**engine_kwargs)
        self._sleep_mode_enabled = enable_sleep_mode

        # ── Extract embed_tokens from the already-loaded vLLM model ──────
        # This avoids loading the 4 safetensors shards a second time.
        logger.info("Extracting embed_tokens from vLLM model...")

        def _extract_embed_tokens(model: torch.nn.Module) -> torch.Tensor:
            """Extract embed_tokens weight from the loaded vLLM model."""
            # Walk the model to find the embedding layer
            for name, module in model.named_modules():
                if "embed_tokens" in name and isinstance(module, torch.nn.Embedding):
                    return module.weight.detach().cpu().clone()
            # Fallback: try common attribute paths
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                return model.model.embed_tokens.weight.detach().cpu().clone()
            raise RuntimeError("Could not find embed_tokens in vLLM model")

        try:
            results = self.llm.apply_model(_extract_embed_tokens)
            embed_weight = results[0]  # First (and only) worker result
        except Exception as e:
            logger.warning(f"apply_model extraction failed ({e}), "
                           "falling back to safetensors loading")
            embed_weight = self._load_embed_from_safetensors()

        _num_embeddings = len(self.tokenizer)
        _tmp_model = torch.nn.Embedding(_num_embeddings, embed_weight.size(1))
        if embed_weight.size(0) < _num_embeddings:
            _tmp_model.weight.data[:embed_weight.size(0)] = embed_weight
        else:
            _tmp_model.weight.data = embed_weight[:_num_embeddings]

        self.embed_layer = _tmp_model.to("cpu")
        self.embed_layer.eval()
        self.embed_layer.requires_grad_(False)
        logger.info(f"embed_tokens loaded: {self.embed_layer.weight.shape}")

        # ── LoRA state ────────────────────────────────────────────────────
        self._lora_path = lora_path
        self._lora_request = None
        if lora_path:
            self.update_lora(lora_path)

        # ── Stop tokens for Llama-3.1 ─────────────────────────────────────
        self._stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # ── Sleep mode: offload to CPU so HF training model has GPU room ─
        # The trainer wakes us up before each rollout and sleeps us after.
        if self._sleep_mode_enabled:
            logger.info("Putting vLLM engine to sleep (CPU offload) for HF model load")
            try:
                self.llm.sleep(level=2)
            except Exception as e:
                logger.warning(f"Initial vLLM sleep failed: {e}")

        logger.info("DeSTAVLLMEngine initialized successfully.")

    def _load_embed_from_safetensors(self) -> torch.Tensor:
        """Fallback: load embed_tokens directly from the safetensors index."""
        from safetensors import safe_open
        model_dir = self.llm_model_id
        index_file = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                idx = json.load(f)
            for key, shard in idx.get("weight_map", {}).items():
                if "embed_tokens.weight" in key:
                    with safe_open(os.path.join(model_dir, shard),
                                   framework="pt", device="cpu") as sf:
                        return sf.get_tensor(key)
        raise FileNotFoundError("embed_tokens.weight not found in safetensors")

    # ── LoRA management ───────────────────────────────────────────────────

    @staticmethod
    def _remap_lora_if_needed(lora_path: str) -> str:
        """
        If the LoRA adapter was trained on the full DeSTA model, its weight
        keys contain an extra ``llm_model.`` segment:

            base_model.model.llm_model.model.layers.0.self_attn.q_proj.lora_A.weight

        vLLM loads the bare Llama backbone, so it expects:

            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

        This method detects the mismatch and creates a *sibling* directory
        ``<checkpoint>_vllm/`` with remapped safetensors + patched config.
        Subsequent calls reuse the cached directory.
        """
        import shutil
        import struct as _struct

        adapter_st = os.path.join(lora_path, "adapter_model.safetensors")
        config_json = os.path.join(lora_path, "adapter_config.json")
        if not os.path.isfile(adapter_st) or not os.path.isfile(config_json):
            return lora_path  # nothing to remap

        # Quick check: read safetensors header for key prefix
        with open(adapter_st, "rb") as f:
            hdr_size = _struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_size).decode("utf-8"))
        keys = [k for k in hdr if k != "__metadata__"]
        if not keys:
            return lora_path

        DESTA_PREFIX = "base_model.model.llm_model."
        needs_remap = any(k.startswith(DESTA_PREFIX) for k in keys)
        if not needs_remap:
            return lora_path  # already correct for vLLM

        # Target directory next to the original checkpoint
        remapped_dir = lora_path.rstrip("/") + "_vllm"
        remapped_st = os.path.join(remapped_dir, "adapter_model.safetensors")
        if os.path.isfile(remapped_st):
            logger.info(f"Using cached remapped LoRA: {remapped_dir}")
            return remapped_dir

        logger.info(f"Remapping DeSTA LoRA keys (llm_model.* → *) → {remapped_dir}")
        os.makedirs(remapped_dir, exist_ok=True)

        # Remap safetensors: reload with torch, rename keys, re-save
        from safetensors.torch import load_file, save_file

        state = load_file(adapter_st)
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("llm_model.", "", 1) if "llm_model." in k else k
            new_state[new_key] = v
        save_file(new_state, remapped_st)

        # Patch adapter_config.json: change base_model_name_or_path
        with open(config_json) as f:
            cfg = json.load(f)
        cfg["base_model_name_or_path"] = "meta-llama/Llama-3.1-8B-Instruct"
        with open(os.path.join(remapped_dir, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # Copy tokenizer files if present (vLLM may need them)
        for fname in ("tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json"):
            src = os.path.join(lora_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, remapped_dir)

        logger.info(f"LoRA remapped: {len(new_state)} weights → {remapped_dir}")
        return remapped_dir

    def update_lora(self, lora_path: str, lora_name: str = "desta_lora"):
        """
        Hot-swap LoRA adapter weights.  Call this between GRPO gradient
        steps to keep the rollout policy in sync with the training policy.

        Automatically remaps DeSTA-style LoRA keys (``llm_model.*``) to
        the bare Llama naming expected by vLLM.
        """
        from vllm.lora.request import LoRARequest

        resolved = self._remap_lora_if_needed(lora_path)
        self._lora_path = resolved
        self._lora_request = LoRARequest(lora_name, 1, resolved)
        logger.info(f"LoRA updated: {resolved} (original: {lora_path})")

    # ── Sleep mode helpers (colocate mode memory management) ─────────────

    def wake_up(self):
        """Wake vLLM engine — moves weights+cache back to GPU. No-op if sleep mode disabled."""
        if self._sleep_mode_enabled:
            try:
                self.llm.wake_up()
            except Exception as e:
                logger.warning(f"vLLM wake_up failed: {e}")

    def sleep(self, level: int = 2):
        """Put vLLM engine to sleep — offloads weights+cache to CPU. No-op if sleep mode disabled."""
        if self._sleep_mode_enabled:
            try:
                self.llm.sleep(level=level)
            except Exception as e:
                logger.warning(f"vLLM sleep failed: {e}")

    # ── Single generation ─────────────────────────────────────────────────

    def generate(
        self,
        embed_path: str,
        transcription: Optional[str],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
        n: int = 1,
        do_sample: bool = True,
    ) -> List[GenerationResult]:
        """
        Generate n completions for a single audio prompt.

        Args:
            embed_path: Path to precomputed *_embed.pt file.
            transcription: Speech transcription (or None).
            system_prompt: System message.
            user_prompt: User message.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            max_new_tokens: Max tokens to generate.
            n: Number of completions to sample.
            do_sample: Whether to sample (False = greedy).

        Returns:
            List of n GenerationResult objects.
        """
        # Build prompt_embeds on CPU, then let vLLM handle device placement
        prompt_embeds, seq_len = build_prompt_embeds(
            embed_path=embed_path,
            transcription=transcription,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=self.tokenizer,
            embed_layer=self.embed_layer,
            embed_cache=self.embed_cache,
            dtype=self._dtype,
            device="cpu",
        )

        sampling_params = self.SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            n=n,
            stop_token_ids=self._stop_token_ids,
        )

        # vLLM generate with prompt_embeds
        outputs = self.llm.generate(
            prompts=[{"prompt_embeds": prompt_embeds}],
            sampling_params=sampling_params,
            lora_request=self._lora_request,
        )

        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append(
                    GenerationResult(
                        text=completion.text,
                        token_ids=list(completion.token_ids),
                    )
                )
        return results

    # ── Batch generation ──────────────────────────────────────────────────

    def generate_batch(
        self,
        items: List[Dict[str, Any]],
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
    ) -> List[List[GenerationResult]]:
        """
        Generate completions for a batch of audio prompts.

        Each item in `items` should have:
          - embed_path: str
          - transcription: Optional[str]
          - system_prompt: str
          - user_prompt: str

        Returns:
            List of lists, one per input item, each containing n results.
        """
        prompts = []
        for item in items:
            embeds, _ = build_prompt_embeds(
                embed_path=item["embed_path"],
                transcription=item.get("transcription"),
                system_prompt=item["system_prompt"],
                user_prompt=item["user_prompt"],
                tokenizer=self.tokenizer,
                embed_layer=self.embed_layer,
                embed_cache=self.embed_cache,
                dtype=self._dtype,
                device="cpu",
            )
            prompts.append({"prompt_embeds": embeds})

        sampling_params = self.SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
            n=n,
            stop_token_ids=self._stop_token_ids,
        )

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
        )

        all_results = []
        for output in outputs:
            group = []
            for completion in output.outputs:
                group.append(
                    GenerationResult(
                        text=completion.text,
                        token_ids=list(completion.token_ids),
                    )
                )
            all_results.append(group)
        return all_results

    # ── Two-phase tool-calling generation ─────────────────────────────────

    def generate_with_tools(
        self,
        embed_path: str,
        transcription: Optional[str],
        system_prompt: str,
        user_prompt: str,
        cached_tool_outputs: Dict[str, Any],
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 2048,
    ) -> List[GenerationResult]:
        """
        Generate completions with automatic tool-call detection and
        two-phase continuation.

        Phase 1: Generate initial response (may contain <tool>...</tool>).
        Phase 2: If tool was called and output is cached, inject the result
                 and continue generation.

        Args:
            embed_path: Path to *_embed.pt file.
            transcription: Speech transcription.
            system_prompt: System prompt.
            user_prompt: User prompt.
            cached_tool_outputs: Dict mapping tool_name → cached result.
            n: Number of completions per prompt.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling.
            max_new_tokens: Max new tokens per phase.

        Returns:
            List of GenerationResult with tool-call metadata populated.
        """
        # Phase 1
        phase1_results = self.generate(
            embed_path=embed_path,
            transcription=transcription,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n,
        )

        final_results = []
        # Collect phase-2 candidates
        phase2_items = []  # (index, tool_name)

        for i, res in enumerate(phase1_results):
            tool_name = _parse_tool_call(res.text)
            if tool_name and tool_name in cached_tool_outputs:
                phase2_items.append((i, tool_name))
            else:
                final_results.append(
                    GenerationResult(
                        text=res.text,
                        token_ids=res.token_ids,
                        phase1=res.text,
                        phase2="",
                        called_tools=False,
                        tool_name=None,
                        injected_output=None,
                    )
                )

        # Phase 2: batch all continuations
        if phase2_items:
            p2_prompts = []
            p2_meta = []

            for idx, tool_name in phase2_items:
                p1_text = phase1_results[idx].text
                tool_result = cached_tool_outputs[tool_name]
                injected = _format_tool_output(tool_name, tool_result)

                embeds, _ = build_prompt_embeds_continuation(
                    embed_path=embed_path,
                    transcription=transcription,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    assistant_phase1=p1_text,
                    injected_tool_output=injected,
                    tokenizer=self.tokenizer,
                    embed_layer=self.embed_layer,
                    embed_cache=self.embed_cache,
                    dtype=self._dtype,
                    device="cpu",
                )
                p2_prompts.append({"prompt_embeds": embeds})
                p2_meta.append((idx, tool_name, injected, p1_text))

            p2_sampling = self.SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=1,
                stop_token_ids=self._stop_token_ids,
            )

            p2_outputs = self.llm.generate(
                prompts=p2_prompts,
                sampling_params=p2_sampling,
                lora_request=self._lora_request,
            )

            for j, output in enumerate(p2_outputs):
                idx, tool_name, injected, p1_text = p2_meta[j]
                p2_text = output.outputs[0].text
                final_results.append(
                    GenerationResult(
                        text=p1_text + injected + p2_text,
                        token_ids=(
                            phase1_results[idx].token_ids
                            + list(output.outputs[0].token_ids)
                        ),
                        phase1=p1_text,
                        phase2=p2_text,
                        called_tools=True,
                        tool_name=tool_name,
                        injected_output=injected,
                    )
                )

        return final_results

    # ── Utility ───────────────────────────────────────────────────────────

    def get_embed_layer(self) -> torch.nn.Embedding:
        """Return the embedding layer (for external users building custom embeds)."""
        return self.embed_layer

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def resolve_embed(self, audio_id: str) -> Optional[str]:
        """Resolve an audio_id to an embed path using the configured embed_dir."""
        if self.embed_dir:
            return resolve_embed_path(audio_id, self.embed_dir)
        return None
