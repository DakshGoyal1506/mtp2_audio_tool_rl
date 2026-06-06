"""
LLM Judge client — connects to an SGLang/vLLM server via OpenAI-compatible API.

Scores GRPO rollouts using path-specific prompts:
  Path A (no tool): think → answer
  Path B (with tool): think → tool_call → tool_output → think → answer

Single judge call per completion with concurrent batching.
"""

import json
import logging
import os
import threading
import concurrent.futures
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic Schemas ─────────────────────────────────────────────────────────

class PathAScore(BaseModel):
    think: float = Field(default=1.0, ge=1.0, le=5.0)
    answer: float = Field(default=1.0, ge=1.0, le=5.0)
    format: float = Field(default=1.0, ge=1.0, le=5.0)

    model_config = {"extra": "ignore"}

class PathBScore(BaseModel):
    think1: float = Field(default=1.0, ge=1.0, le=5.0)
    tool: float = Field(default=1.0, ge=1.0, le=5.0)
    think2: float = Field(default=1.0, ge=1.0, le=5.0)
    answer: float = Field(default=1.0, ge=1.0, le=5.0)
    format: float = Field(default=1.0, ge=1.0, le=5.0)

    model_config = {"extra": "ignore"}


# ── Path A prompt: think → answer (no tool call) ─────────────────────────────

# PATH_A_PROMPT = """\
# You are scoring an AI assistant that answers audio multiple-choice questions.
# The assistant answered WITHOUT using any audio analysis tool.

# ## Question:
# {question}

# ## Answer choices:
# {choices}

# ## Gold (correct) answer:
# {gold}

# ## Completion:
# {completion}

# ## Assistant's final answer:
# {predicted}

# Score each dimension 1-5. Use the FULL range; scores of 2 and 4 are expected when appropriate.

# 1. **think** — Reasoning quality in the <think> block.
#    - 1: Empty, placeholder text, or completely unrelated to the audio/question.
#    - 2: Mentions the topic but reasoning is vague, generic, or shows no real audio understanding (e.g., "I think the answer is X").
#    - 3: References some audio features (e.g., sounds heard, speech content, musical elements) but the reasoning is shallow, partially wrong, or does not clearly lead to the chosen answer.
#    - 4: Solid reasoning that identifies relevant audio cues (e.g., speaker characteristics, instruments, environmental sounds) and logically connects them to the answer, with only minor gaps.
#    - 5: Thorough reasoning that accurately describes specific audio observations, systematically considers the choices, and clearly deduces the answer.
#    Additional: If the <think> conclusion contradicts the <answer>, cap this score at 2 (inconsistency).

# 2. **answer** — Correctness of the final answer.
#    - 1: Wrong answer that is not even close to the gold answer.
#    - 2: Wrong answer but in a related category or adjacent option.
#    - 3: Partially correct (e.g., correct category but wrong specific, or matches a plausible distractor).
#    - 4: Correct answer but phrased differently, or correct with minor formatting issues.
#    - 5: Exactly matches the gold answer.

# 3. **format** — Proper use of the required XML structure: <think>...</think><answer>...</answer>.
#    - 1: Missing both tags or output is unstructured text.
#    - 2: Only one tag present (e.g., has <answer> but no <think>), or tags are severely broken.
#    - 3: Both tags present but malformed (e.g., unclosed, nested wrongly, extra content outside tags).
#    - 4: Tags are correct and properly closed, with only minor issues (e.g., whitespace, extra newlines).
#    - 5: Clean, properly structured <think>...</think><answer>...</answer> with no extra content outside tags.

# Respond with ONLY a JSON object: {{"think": <1-5>, "answer": <1-5>, "format": <1-5>}}"""

PATH_A_PROMPT = """\
You are an expert evaluator scoring an AI assistant's performance on audio-based multiple-choice questions.

## Context
Question: {question}
Choices: {choices}
Gold Answer: {gold}

## Assistant Completion:
{completion}
Final Predicted Answer: {predicted}

## Scoring Instructions
Score 1-5. Avoid "3" (Neutral) unless the performance is strictly mediocre. If it is "good but not perfect," use 4. If it is "poor but has content," use 2.

1. **think (Reasoning)**
   - 1: Placeholder, empty, or ignores the audio context.
   - 2: Generic reasoning that doesn't reference specific audio cues (e.g., "based on the audio, I pick A").
   - 3: References audio features but reasoning is shallow or fails to explain why distractors were rejected.
   - 4: Detailed reasoning citing specific audio cues (pitch, timbre, speech patterns) that logically lead to the gold answer.
   - 5: Exceptional reasoning that not only identifies the correct answer but explicitly explains why the other choices are incorrect based on the audio.

2. **answer (Correctness)**
   - 1: Incorrect.
   - 5: Exactly matches the gold answer (including logic and label).

3. **format (Structure)**
   - 1: Missing <think> or <answer> tags.
   - 3: Tags present but contains text outside of the XML structure.
   - 5: Perfect XML compliance.

Respond with ONLY a JSON object:
{{
  "think": <1-5>,
  "answer": <1-5>,
  "format": <1-5>
}}"""


# ── Path B prompt: think → tool → think → answer ────────────────────────────

# PATH_B_PROMPT = """\
# You are scoring an AI assistant that answers audio multiple-choice questions.
# The assistant used an audio analysis TOOL: think → tool_call → tool_output → think → answer.

# ## Question:
# {question}

# ## Answer choices:
# {choices}

# ## Gold (correct) answer:
# {gold}

# ## Completion:
# {completion}

# ## Tool called:
# {tool_name}

# ## Tool output (truncated):
# {tool_output}

# ## Assistant's final answer:
# {predicted}

# Score each dimension 1-5. Use the FULL range; scores of 2 and 4 are expected when appropriate.

# 1. **think1** — Quality of the FIRST <think> block (before the tool call).
#    - 1: Empty, placeholder, or unrelated to the audio/question.
#    - 2: Mentions the topic but gives no concrete audio observations and no justification for calling a tool.
#    - 3: Some audio observations present, but the reasoning for why a tool is needed is weak or generic (e.g., "let me use a tool to help").
#    - 4: Identifies specific audio features relevant to the question and gives a reasonable explanation for why the chosen tool would help.
#    - 5: Clearly describes what was heard in the audio, explains what information is still missing, and articulates exactly why the specific tool is needed to answer.

# 2. **tool** — Tool selection and usage.
#    - 1: Called a completely wrong tool for this question type, or the tool call is malformed/unparseable.
#    - 2: Called a somewhat relevant tool but it is clearly not the best fit (e.g., using transcription when the question is about music tempo).
#    - 3: Reasonable tool choice but either the question could have been answered without it, or the parameters are suboptimal.
#    - 4: Good tool choice that is well-suited to the question; parameters are correct; the tool provides useful information.
#    - 5: Optimal tool for this question type; the tool is genuinely necessary (question cannot be reliably answered by listening alone); parameters are correct.

# 3. **think2** — Quality of the SECOND <think> block (after receiving tool output).
#    - 1: Ignores the tool output entirely, or just repeats the first think block.
#    - 2: Acknowledges the tool output but does not interpret or analyze it (e.g., "the tool says X so the answer is Y" with no reasoning).
#    - 3: References specific parts of the tool output but the interpretation is shallow or partially incorrect.
#    - 4: Correctly interprets the tool output, connects it to the question, and uses it to reason toward the answer with only minor gaps.
#    - 5: Accurately interprets the tool output, integrates it with initial audio observations, and provides a clear, well-supported deduction of the answer.
#    Additional: If the think2 conclusion contradicts the <answer>, cap this score at 2 (inconsistency).

# 4. **answer** — Correctness of the final answer.
#    - 1: Wrong answer that is not even close to the gold answer.
#    - 2: Wrong answer but in a related category or adjacent option.
#    - 3: Partially correct (e.g., correct category but wrong specific, or matches a plausible distractor).
#    - 4: Correct answer but phrased differently, or correct with minor formatting issues.
#    - 5: Exactly matches the gold answer.

# 5. **format** — Proper use of the required XML structure: <think>...<tool>...<tool_output>...<think>...<answer>.
#    - 1: Missing most tags or output is unstructured text.
#    - 2: Some tags present but major structural issues (e.g., missing second <think>, wrong tag order).
#    - 3: All required tags present but malformed (e.g., unclosed, nested wrongly, extra content outside tags).
#    - 4: Tags are correct and in proper order, with only minor issues (e.g., whitespace, extra newlines).
#    - 5: Clean, properly structured tags in correct order with no extra content outside tags.

# Respond with ONLY a JSON object: {{"think1": <1-5>, "tool": <1-5>, "think2": <1-5>, "answer": <1-5>, "format": <1-5>}}"""

PATH_B_PROMPT = """\
You are an expert evaluator scoring an AI assistant that uses audio tools to answer questions.

## Evaluation Data
Question: {question}
Choices: {choices}
Gold Answer: {gold}
Completion: {completion}
Tool Used: {tool_name}
Tool Output: {tool_output}
Assistant's Final Answer: {predicted}

## Scoring Rubric (1-5)

1. **think1 (Strategy)**
   - 1-2: No clear plan; calls tool without explaining what it hopes to find.
   - 5: Clearly identifies a specific audio ambiguity and explains why {tool_name} is the best way to resolve it.

2. **tool (Efficiency & Accuracy)**
   - 1: Wrong tool for the task OR the tool output is incorrect/poorly formatted and the assistant should have known better.
   - 3: Correct tool, but the parameters were slightly off or the tool was not strictly necessary to answer the question.
   - 5: Perfect tool selection with optimal parameters; the tool provides the critical "missing link" for the answer.

3. **think2 (Synthesis & Skepticism)**
   - 1-2: Blindly follows tool output even if it contradicts the audio or Mentions tool output but doesn't integrate it with the original <think1> observations.
   - 5: Effectively uses tool output to confirm or deny initial hypotheses or If the tool output was incorrect or unhelpful, give a 5 if the assistant explicitly identifies the tool's error and relies on its own audio reasoning instead.

4. **answer (Correctness)**
   - 1: Wrong.
   - 5: Correct.

5. **format (Structure)**
   - 1: Broken XML.
   - 5: Clean <think>...<tool>...<tool_output>...<think>...<answer> structure.

## Scoring Guidance
If the assistant is "lazy" (uses a tool for a very simple question), cap the 'tool' score at 3.
If the assistant is "gullible" (follows a wrong tool to a wrong answer), cap 'think2' at 1.

Respond with ONLY a JSON object:
{{
  "think1": <1-5>,
  "tool": <1-5>,
  "think2": <1-5>,
  "answer": <1-5>,
  "format": <1-5>
}}"""


# ── LLMJudge class ─────────────────────────────────────────────────────────────

class LLMJudge:
    """Concurrent batch-scoring client for a judge LLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8899",
        model: str = "Qwen/Qwen3.5-27B",
        timeout: float = 60.0,
        max_completion_tokens: int = 2048,
        max_workers: int = 32,
        reasoning_effort: str = "low",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_completion_tokens = max_completion_tokens
        self.max_workers = max_workers
        self.reasoning_effort = reasoning_effort
        self._available = None

        # Judge trace logging — written to JSONL inside output_dir/judge_trace/
        self._trace_dir = None
        self._trace_lock = threading.Lock()
        self._trace_batch_idx = 0

    # ── Trace logging ─────────────────────────────────────────────────────

    def set_trace_dir(self, output_dir: str):
        """Enable judge trace logging into <output_dir>/judge_trace/."""
        self._trace_dir = os.path.join(output_dir, "judge_trace")
        os.makedirs(self._trace_dir, exist_ok=True)
        logger.info(f"Judge trace logging enabled → {self._trace_dir}")

    def _write_trace(self, record: dict):
        """Append one JSON record to the current batch trace file (thread-safe)."""
        if self._trace_dir is None:
            return
        path = os.path.join(self._trace_dir, f"judge_batch_{self._trace_batch_idx:05d}.jsonl")
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._trace_lock:
            with open(path, "a") as f:
                f.write(line)

    def is_available(self) -> bool:
        if self._available:
            return True
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5, proxies={"http": None, "https": None})
            if r.status_code == 200:
                self._available = True
                return True
        except Exception as e:
            logger.debug(f"Judge health check failed: {e}")
        return False

    # Multi-channel delimiter used by gpt-oss models:
    # analysis channel has reasoning, final channel has clean JSON.
    _FINAL_CHANNEL_DELIM = "<|channel|>final<|message|>"

    def _call(self, prompt: str, retries: int = 2) -> Optional[str]:
        """Single chat completion call with retry. Returns raw text or None."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise scoring assistant. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": 0.0,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout,
                    proxies={"http": None, "https": None}
                )
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return None
                content = choices[0].get("message", {}).get("content", "")
                return content if content else None
            except Exception as e:
                logger.warning(f"Judge call failed (attempt {attempt+1}/{retries}): {e}")
        return None

    @staticmethod
    def _find_last_json(text: str) -> Optional[str]:
        """Find the last complete JSON object in text by scanning from the end."""
        last_close = text.rfind('}')
        while last_close >= 0:
            depth = 0
            for i in range(last_close, -1, -1):
                if text[i] == '}':
                    depth += 1
                elif text[i] == '{':
                    depth -= 1
                if depth == 0:
                    candidate = text[i:last_close + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
            last_close = text.rfind('}', 0, last_close)
        return None

    def _parse_scores(self, text: Optional[str], is_path_b: bool) -> Dict[str, float]:
        """Extract numeric scores from JSON response using Pydantic."""
        model = PathBScore if is_path_b else PathAScore
        default_scores = model().model_dump()

        if text is None:
            return default_scores

        # 1) If multi-channel output, extract the final channel (clean JSON)
        if self._FINAL_CHANNEL_DELIM in text:
            text = text.split(self._FINAL_CHANNEL_DELIM)[-1]
        text = text.strip()

        if not text:
            return default_scores

        # 2) Try direct parse (fast path — works when content is pure JSON)
        try:
            parsed = model.model_validate_json(text)
            return parsed.model_dump()
        except Exception:
            pass

        # 3) Find the last complete JSON object (fallback for any stray text)
        json_str = self._find_last_json(text)
        if json_str:
            try:
                parsed = model.model_validate_json(json_str)
                return parsed.model_dump()
            except Exception as e:
                logger.warning(f"JSON Validation Error: {e} | extracted: {json_str[:300]}")
        else:
            logger.warning(f"No JSON found in judge response: {text[:300]}")

        return default_scores

    def _score_one(self, completion: str, question: str, choices_str: str,
                   gold: str, predicted: str, tool_name: Optional[str],
                   tool_output: Optional[str], is_path_b: bool) -> Dict[str, float]:
        """Score a single completion, returns raw 1-5 scores."""
        comp_trunc = completion[:4000]
        path = "B" if is_path_b else "A"

        if is_path_b:
            prompt = PATH_B_PROMPT.format(
                completion=comp_trunc, question=question, choices=choices_str,
                gold=gold, tool_name=tool_name or "unknown",
                tool_output=(tool_output or "N/A")[:1500], predicted=predicted or "N/A",
            )
        else:
            prompt = PATH_A_PROMPT.format(
                completion=comp_trunc, question=question, choices=choices_str,
                gold=gold, predicted=predicted or "N/A",
            )

        raw = self._call(prompt)
        scores = self._parse_scores(raw, is_path_b)

        logger.debug(
            f"Judge [Path {path}] q={question[:80]!r} gold={gold!r} pred={predicted!r}"
            f" tool={tool_name} | raw={raw!r} | scores={scores}"
        )

        # Persist full prompt + response to trace JSONL
        self._write_trace({
            "path": path,
            "question": question,
            "choices": choices_str,
            "gold": gold,
            "predicted": predicted,
            "tool_name": tool_name,
            "judge_prompt": prompt,
            "judge_raw_response": raw,
            "parsed_scores": scores,
        })

        return scores

    def score_batch(
        self,
        completions: List[str],
        questions: List[str],
        choices_list: List[List[str]],
        gold_answers: List[str],
        predicted_answers: List[Optional[str]],
        tool_names: List[Optional[str]],
        tool_outputs: List[Optional[str]],
        is_path_b: List[bool],
    ) -> List[Dict[str, float]]:
        """Score a batch concurrently. Returns list of raw 1-5 score dicts."""
        if not self.is_available():
            logger.warning("Judge unavailable — returning default scores (1.0)")
            defaults = []
            for b in is_path_b:
                if b:
                    defaults.append({"think1": 1.0, "tool": 1.0, "think2": 1.0,
                                     "answer": 1.0, "format": 1.0})
                else:
                    defaults.append({"think": 1.0, "answer": 1.0, "format": 1.0})
            return defaults

        n = len(completions)
        self._trace_batch_idx += 1
        choices_strs = ["\n".join(f"  - {c}" for c in ch) for ch in choices_list]

        results = [None] * n
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {}
            for i in range(n):
                f = pool.submit(
                    self._score_one,
                    completions[i], questions[i], choices_strs[i],
                    gold_answers[i], predicted_answers[i],
                    tool_names[i], tool_outputs[i], is_path_b[i],
                )
                future_to_idx[f] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Judge scoring failed for idx {idx}: {e}")
                    if is_path_b[idx]:
                        results[idx] = {"think1": 1.0, "tool": 1.0, "think2": 1.0,
                                        "answer": 1.0, "format": 1.0}
                    else:
                        results[idx] = {"think": 1.0, "answer": 1.0, "format": 1.0}

        return results
