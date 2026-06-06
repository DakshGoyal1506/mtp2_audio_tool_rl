"""
tool_execute_vllm.py — vLLM-based AudioToolExecutor for fast benchmarking.

Drop-in replacement for scripts/tool_execute.py that uses DeSTAVLLMEngine
instead of the native DeSTA25AudioModel.generate() for ~5-10x faster
inference on MMAU benchmarks.

Key differences from the original:
  - No Whisper encoder / QFormer / VAD at runtime
  - All audio processing is precomputed (embed files)
  - vLLM handles LLM generation with continuous batching
  - Same prompt templates and parsing logic as the original
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from desta_vllm.engine import DeSTAVLLMEngine

# Import prompt templates — use the GRPO format (<think>, <tool>, <answer> tags)
from grpo.prompts import (
    GRPO_INITIAL_SYSTEM_PROMPT,
    GRPO_DIRECT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Parsing helpers (same as scripts/tool_execute.py) ─────────────────────────

def _extract_think(text: str) -> str:
    """Extract content from <think>...</think> block."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_tool_calls(text: str) -> Tuple[List[Dict], Optional[str]]:
    """Extract function calls from GRPO-format <tool> tag.

    Expected format:
        <think>reasoning</think>
        <tool>[{"function": "tool_name", "parameters": {...}}]</tool>
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    if hasattr(text, "text"):
        text = text.text
    text = str(text).strip()
    function_calls = []

    tool_match = re.search(r"<tool>\s*(.*?)\s*</tool>", text, re.DOTALL)
    if tool_match:
        raw = tool_match.group(1).strip()
        try:
            parsed = json.loads(raw)
            # New format: [{"function": "tool_name", "parameters": {...}}]
            if isinstance(parsed, list):
                for entry in parsed:
                    func_name = entry.get("function", "").lower().strip()
                    if func_name:
                        function_calls.append({
                            "function": func_name,
                            "args": ["path"],
                            "raw_call": f'{func_name}("path")',
                        })
                if function_calls:
                    return function_calls, None
            # Old format fallback: {"tools": ["name1", ...]} or {"function": "name"}
            elif isinstance(parsed, dict):
                tools_list = parsed.get("tools", [])
                if tools_list:
                    for t in tools_list:
                        t = t.lower().strip() if isinstance(t, str) else ""
                        if t:
                            function_calls.append({"function": t, "args": ["path"], "raw_call": f'{t}("path")'})
                    return function_calls, None
                func_name = parsed.get("function", parsed.get("name", "")).lower().strip()
                if func_name:
                    function_calls.append({"function": func_name, "args": ["path"], "raw_call": f'{func_name}("path")'})
                    return function_calls, None
        except json.JSONDecodeError as e:
            return [], f"Invalid JSON in <tool> tag: {e}"

    # Check for <answer> tag (direct answer, no tools)
    if re.search(r"<answer>", text):
        return [], None

    return [], None


def parse_answer(text: str) -> Tuple[Dict[str, str], Optional[str]]:
    """Extract answer from GRPO-format response.

    Expected format:
        <think>reasoning</think>
        <answer>exact option text</answer>

    Returns ({"think": ..., "answer": ...}, error_or_None).
    """
    if isinstance(text, list):
        text = text[0] if text else ""
    if hasattr(text, "text"):
        text = text.text
    text = str(text).strip()

    # Require <think> tag
    think = _extract_think(text)
    if not think:
        return (
            {"think": "", "answer": ""},
            f"Missing <think> tag: {text[:150]}",
        )

    # Require <answer> tag
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
        return {"think": think, "answer": content}, None

    if "<answer>" in text and "</answer>" not in text:
        return (
            {"think": think, "answer": ""},
            f"Unclosed <answer> tag: {text[:150]}",
        )

    if "<tool>" in text:
        return (
            {"think": think, "answer": ""},
            f"Contains <tool> but no <answer>: {text[:150]}",
        )

    return (
        {"think": think, "answer": ""},
        f"Missing <answer> tag: {text[:150]}",
    )


def extract_model_prediction(response: str) -> str:
    """Extract clean answer from LLM response."""
    if not response:
        return ""
    response = str(response).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return response


def format_question_with_choices(question: str, choices: List[str]) -> str:
    """Format question with multiple choice options (GRPO format)."""
    if not choices:
        return question
    choices_str = "\n".join(f"  - {c}" for c in choices)
    return f"{question}\n\nAnswer options:\n{choices_str}"


# ── VLLMToolExecutor ──────────────────────────────────────────────────────────


class VLLMToolExecutor:
    """
    Unified batched inference for MMAU evaluation.

    One method — process_batch(items, direct) — handles both modes:
      direct=True  → GRPO_DIRECT_SYSTEM_PROMPT (no tool descriptions)
      direct=False → GRPO_INITIAL_SYSTEM_PROMPT (with tool descriptions)

    If the model outputs a <tool> tag, cached tool results are looked up
    and a Phase 2 continuation is generated using single-episode architecture.

    Every phase retries parse failures up to max_retries times.
    """

    def __init__(
        self,
        engine: DeSTAVLLMEngine,
        cached_tools: Optional[Dict[str, Dict]] = None,
        embed_dir: Optional[str] = None,
        max_new_tokens: int = 2048,
        max_retries: int = 3,
    ):
        self.engine = engine
        self.cached_tools = cached_tools or {}
        self.embed_dir = embed_dir or engine.embed_dir
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries

    # ── helpers ───────────────────────────────────────────────────────────

    def _resolve_embed(self, audio_id: str) -> Optional[str]:
        if self.embed_dir:
            from desta_vllm.embed_utils import resolve_embed_path
            return resolve_embed_path(audio_id, self.embed_dir)
        return self.engine.resolve_embed(audio_id)

    def _get_transcription(self, audio_id: str) -> Optional[str]:
        if audio_id in self.cached_tools:
            sr = self.cached_tools[audio_id].get("speech_recognition", {})
            if isinstance(sr, dict):
                return sr.get("text")
        return None

    def _build_embeds(self, embed_path, transcription, system_prompt, user_prompt):
        from desta_vllm.embed_utils import build_prompt_embeds
        embeds, _ = build_prompt_embeds(
            embed_path=embed_path,
            transcription=transcription,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tokenizer=self.engine.tokenizer,
            embed_layer=self.engine.embed_layer,
            embed_cache=self.engine.embed_cache,
            dtype=self.engine._dtype,
            device="cpu",
        )
        return embeds

    def _build_continuation_embeds(
        self, embed_path, transcription, system_prompt, user_prompt,
        assistant_phase1, injected_tool_output,
    ):
        """Build prompt_embeds for Phase 2 (single-episode continuation).

        Conversation: system → user(audio+question) → assistant(phase1) → user(tool_output) → ...
        This matches the sequential inference architecture exactly.
        """
        from desta_vllm.embed_utils import build_prompt_embeds_continuation
        embeds, _ = build_prompt_embeds_continuation(
            embed_path=embed_path,
            transcription=transcription,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            assistant_phase1=assistant_phase1,
            injected_tool_output=injected_tool_output,
            tokenizer=self.engine.tokenizer,
            embed_layer=self.engine.embed_layer,
            embed_cache=self.engine.embed_cache,
            dtype=self.engine._dtype,
            device="cpu",
        )
        return embeds

    def _sampling_params(self):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            n=1,
            stop_token_ids=self.engine._stop_token_ids,
        )

    def _batch_generate(self, prompts, sampling=None):
        if sampling is None:
            sampling = self._sampling_params()
        return self.engine.llm.generate(
            prompts=prompts,
            sampling_params=sampling,
            lora_request=self.engine._lora_request,
        )

    def _lookup_tools(self, audio_id: str, function_calls: List[Dict]) -> Dict:
        tool_results = {}
        for k, call in enumerate(function_calls):
            func_name = call["function"]
            cached = self.cached_tools.get(audio_id, {}).get(func_name)
            if cached is not None:
                tool_results[f"{func_name}_{k}"] = cached
            else:
                tool_results[f"{func_name}_{k}"] = {
                    "error": f"No cached result for {func_name}"
                }
        return tool_results

    # ── unified batch method ─────────────────────────────────────────────

    def process_batch(
        self,
        items: List[Dict[str, Any]],
        direct: bool = False,
    ) -> List[Dict]:
        """
        Process a batch of queries.

        Args:
            items:  List of dicts with audio_id, question, choices, ...
            direct: If True, use direct prompt (no tool descriptions).
                    If False, use tool prompt; items that output <tool>
                    get a Phase 2 call.

        Returns:
            List of result dicts aligned with items.
        """
        n = len(items)
        system_prompt = GRPO_DIRECT_SYSTEM_PROMPT if direct else GRPO_INITIAL_SYSTEM_PROMPT
        results: List[Optional[Dict]] = [None] * n
        sampling = self._sampling_params()

        # ── Prepare Phase 1 ──────────────────────────────────────────────
        p1_prompts = []
        p1_meta = []  # (orig_idx, audio_id, question, embed_path, transcription)

        for i, item in enumerate(items):
            audio_id = item.get("audio_id", "")
            question = format_question_with_choices(
                item.get("question", ""), item.get("choices", [])
            )
            embed_path = self._resolve_embed(audio_id)
            if not embed_path:
                results[i] = {"type": "error", "error": f"No embed for {audio_id}"}
                continue

            transcription = self._get_transcription(audio_id)
            user_msg = question + "\n\nListen to the audio, then respond in the correct format."

            embeds = self._build_embeds(embed_path, transcription, system_prompt, user_msg)
            p1_prompts.append({"prompt_embeds": embeds})
            p1_meta.append((i, audio_id, question, embed_path, transcription))

        if not p1_prompts:
            return results

        # ── Phase 1: generate ────────────────────────────────────────────
        logger.info(f"Phase 1: {len(p1_prompts)} items (mode={'direct' if direct else 'tools'})")
        p1_outputs = self._batch_generate(p1_prompts, sampling)

        # ── Phase 1: parse + collect retries ─────────────────────────────
        # Track items needing retry: {p1_idx: response_text}
        p1_retry = {}

        for j, output in enumerate(p1_outputs):
            orig_idx, audio_id, question, embed_path, transcription = p1_meta[j]
            response = output.outputs[0].text

            ok = self._handle_p1_response(
                results, orig_idx, audio_id, question,
                embed_path, transcription, response, direct,
            )
            if not ok:
                p1_retry[j] = response

        # ── Phase 1: retry loop ──────────────────────────────────────────
        for attempt in range(self.max_retries):
            if not p1_retry:
                break
            logger.info(f"Phase 1 retry {attempt+1}/{self.max_retries}: {len(p1_retry)} items")

            retry_prompts = []
            retry_map = []
            for j, prev_resp in p1_retry.items():
                orig_idx, audio_id, question, embed_path, transcription = p1_meta[j]
                corrective = (
                    question
                    + "\n\nListen to the audio, then respond in the correct format."
                    + f"\n\nYour previous response was malformed: {prev_resp[:200]}"
                    + "\n\nRespond ONLY with:\n<think>your reasoning</think>\n<answer>exact option text</answer>"
                )
                embeds = self._build_embeds(embed_path, transcription, system_prompt, corrective)
                retry_prompts.append({"prompt_embeds": embeds})
                retry_map.append(j)

            retry_outputs = self._batch_generate(retry_prompts, sampling)
            new_retry = {}
            for k, output in enumerate(retry_outputs):
                j = retry_map[k]
                orig_idx, audio_id, question, embed_path, transcription = p1_meta[j]
                response = output.outputs[0].text

                ok = self._handle_p1_response(
                    results, orig_idx, audio_id, question,
                    embed_path, transcription, response, direct,
                )
                if not ok:
                    new_retry[j] = response
            p1_retry = new_retry

        # Finalize remaining Phase 1 failures
        for j, prev_resp in p1_retry.items():
            orig_idx = p1_meta[j][0]
            parsed, _ = parse_answer(prev_resp)
            if not parsed.get("answer"):
                parsed = {"think": "Parse failed after retries", "answer": str(prev_resp)}
            results[orig_idx] = {
                "type": "direct_answer",
                "response": prev_resp,
                "tool_calls": [], "tool_results": {},
                "parsed_answer": parsed,
                "initial_response": prev_resp,
                "final_response": prev_resp,
            }

        # ── Phase 2: tool-calling items ──────────────────────────────────
        # Collect items that were marked _needs_p2 by _handle_p1_response
        p2_indices = [i for i in range(n) if results[i] is not None and results[i].get("_needs_p2")]
        if p2_indices:
            self._run_phase2(results, p2_indices, sampling)

        # Final safety fill
        for i in range(n):
            if results[i] is None:
                results[i] = {"type": "error", "error": "Unknown processing error"}

        return results

    def _handle_p1_response(
        self, results, orig_idx, audio_id, question,
        embed_path, transcription, response, direct,
    ) -> bool:
        """
        Parse a Phase 1 response and populate results[orig_idx].

        Returns True if successfully parsed (direct answer or valid tool call
        queued for Phase 2). Returns False if parsing failed (needs retry).
        """
        # Require <think> tag in all responses
        think = _extract_think(response)
        if not think:
            return False  # needs retry — missing <think>

        # Check for <tool> tag (only meaningful when not in direct mode)
        function_calls, tool_error = parse_tool_calls(response)

        if function_calls and not direct:
            # Valid tool call → queue for Phase 2
            tool_results = self._lookup_tools(audio_id, function_calls)
            # Format tool output like sequential: <tool_output>...</tool_output>
            tool_output_parts = []
            for key, val in tool_results.items():
                func_name = key.rsplit("_", 1)[0]  # strip _0 suffix
                val_str = json.dumps(val, indent=2) if not isinstance(val, str) else val
                tool_output_parts.append(f"\n<tool_output>\n{val_str}\n</tool_output>\n")
            injected_tool_output = "".join(tool_output_parts)
            results[orig_idx] = {
                "_needs_p2": True,
                "_audio_id": audio_id,
                "_question": question,
                "_embed_path": embed_path,
                "_transcription": transcription,
                "_initial_response": response,
                "_function_calls": function_calls,
                "_tool_results": tool_results,
                "_injected_tool_output": injected_tool_output,
            }
            return True

        if tool_error:
            return False  # needs retry — malformed <tool> JSON

        # Try to parse as direct answer
        parsed, parse_err = parse_answer(response)
        if parse_err or not parsed.get("answer"):
            return False  # needs retry

        results[orig_idx] = {
            "type": "direct_answer",
            "response": response,
            "tool_calls": [], "tool_results": {},
            "parsed_answer": parsed,
            "initial_response": response,
            "final_response": response,
        }
        return True

    def _run_phase2(self, results, p2_indices, sampling):
        """Run Phase 2 generation using single-episode continuation.

        Matches sequential architecture: the model sees its own Phase 1 output
        and tool output as a conversation continuation, NOT a fresh prompt.

        Conversation: system → user(audio+question) → assistant(phase1) → user(tool_output) → ...
        """
        system_prompt = GRPO_INITIAL_SYSTEM_PROMPT  # same system prompt as Phase 1

        # Build Phase 2 continuation prompts
        p2_prompts = []
        p2_meta = []  # orig_idx list

        for orig_idx in p2_indices:
            r = results[orig_idx]
            user_msg = r["_question"] + "\n\nListen to the audio, then respond in the correct format."
            embeds = self._build_continuation_embeds(
                r["_embed_path"], r["_transcription"],
                system_prompt, user_msg,
                r["_initial_response"],       # assistant Phase 1 output
                r["_injected_tool_output"],    # <tool_output>...</tool_output>
            )
            p2_prompts.append({"prompt_embeds": embeds})
            p2_meta.append(orig_idx)

        logger.info(f"Phase 2: {len(p2_prompts)} tool-calling items")
        p2_outputs = self._batch_generate(p2_prompts, sampling)

        # Parse results, track failures
        p2_retry = {}  # p2_idx → response
        for j, output in enumerate(p2_outputs):
            orig_idx = p2_meta[j]
            response = output.outputs[0].text
            parsed, parse_err = parse_answer(response)

            if parse_err or not parsed.get("answer"):
                p2_retry[j] = response
            else:
                r = results[orig_idx]
                results[orig_idx] = {
                    "type": "tool_assisted",
                    "initial_response": r["_initial_response"],
                    "tool_calls": r["_function_calls"],
                    "tool_results": r["_tool_results"],
                    "parsed_answer": parsed,
                    "final_response": response,
                }

        # Retry loop — rebuild continuation with corrective suffix
        for attempt in range(self.max_retries):
            if not p2_retry:
                break
            logger.info(f"Phase 2 retry {attempt+1}/{self.max_retries}: {len(p2_retry)} items")

            retry_prompts = []
            retry_map = []
            for j, prev_resp in p2_retry.items():
                orig_idx = p2_meta[j]
                r = results[orig_idx]
                # Append corrective hint to the tool output
                corrective_tool_output = (
                    r["_injected_tool_output"]
                    + f"\nYour previous response was malformed: {prev_resp[:200]}"
                    + "\nRespond ONLY with:\n<think>your reasoning</think>\n<answer>exact option text</answer>"
                )
                user_msg = r["_question"] + "\n\nListen to the audio, then respond in the correct format."
                embeds = self._build_continuation_embeds(
                    r["_embed_path"], r["_transcription"],
                    system_prompt, user_msg,
                    r["_initial_response"],
                    corrective_tool_output,
                )
                retry_prompts.append({"prompt_embeds": embeds})
                retry_map.append(j)

            retry_outputs = self._batch_generate(retry_prompts, sampling)
            new_retry = {}
            for k, output in enumerate(retry_outputs):
                j = retry_map[k]
                orig_idx = p2_meta[j]
                response = output.outputs[0].text
                parsed, parse_err = parse_answer(response)

                if parse_err or not parsed.get("answer"):
                    new_retry[j] = response
                else:
                    r = results[orig_idx]
                    results[orig_idx] = {
                        "type": "tool_assisted",
                        "initial_response": r["_initial_response"],
                        "tool_calls": r["_function_calls"],
                        "tool_results": r["_tool_results"],
                        "parsed_answer": parsed,
                        "final_response": response,
                    }
            p2_retry = new_retry

        # Finalize remaining failures
        for j, prev_resp in p2_retry.items():
            orig_idx = p2_meta[j]
            r = results[orig_idx]
            parsed, _ = parse_answer(prev_resp)
            if not parsed.get("answer"):
                parsed = {"think": "Parse failed after retries", "answer": str(prev_resp)}
            results[orig_idx] = {
                "type": "tool_assisted",
                "initial_response": r["_initial_response"],
                "tool_calls": r["_function_calls"],
                "tool_results": r["_tool_results"],
                "parsed_answer": parsed,
                "final_response": prev_resp,
            }


# ── Cached tools loader ─────────────────────────────────────────────────────

def load_cached_tools(data: List[Dict]) -> Dict[str, Dict]:
    """Extract cached tool outputs from dataset."""
    cached_tools = {}

    def _clean_tool_output(obj):
        if isinstance(obj, str):
            return re.sub(r"[\u4e00-\u9fff]+/([a-zA-Z]+)", r"\1", obj)
        elif isinstance(obj, list):
            return [_clean_tool_output(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _clean_tool_output(v) for k, v in obj.items()}
        return obj

    for item in data:
        audio_id = item.get("audio_id", "")
        tool_outputs = item.get("tool_outputs", {})
        if audio_id and tool_outputs:
            tool_outputs = _clean_tool_output(tool_outputs)
            get_aud_feat = tool_outputs.pop("audio_features", None)
            cached_tools[audio_id] = tool_outputs
            cached_tools[audio_id]["get_audio_features"] = get_aud_feat
    return cached_tools
