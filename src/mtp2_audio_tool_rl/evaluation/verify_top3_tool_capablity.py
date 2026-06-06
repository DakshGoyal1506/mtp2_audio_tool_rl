#!/usr/bin/env python3
"""
Verify whether wrong no-tool questions are answerable by one of cached top-3 tools.

For each incorrect direct-answer sample:
1) Load its top_3_tools and corresponding cached tool_outputs.
2) Ask a judge LLM whether one of the top-3 tools provides enough evidence.
3) Save verdicts, CoT reasoning, and aggregates.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests


FINAL_CHANNEL_DELIM = "<|channel|>final<|message|>"


TOOL_INFO = {
    "speech_recognition": "Transcribe speech to text with timestamps.",
    "speaker_diarization": "Detect speakers and their speaking segments.",
    "emotion_recognition": "Detect emotion labels over speech segments.",
    "stressed_analysis": "Analyze lexical/phoneme stress patterns in speech.",
    "get_audio_features": "Extract low-level features like volume, pitch, tempo.",
    "audio_features": "Extract low-level features like volume, pitch, tempo.",
    "sound_classification": "Classify dominant sound events in audio.",
    "sound_duration_analysis": "Estimate duration for sound events.",
    "speech_to_noise_ratio": "Estimate speech-to-noise ratio.",
    "chord_recognition": "Detect chord progression in music.",
    "genre_analysis": "Estimate music genre/instrument profile.",
}


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _tokenize(text: Any) -> set:
    return set(re.findall(r"\b\w+\b", str(text or "").lower()))


def string_match(answer: Any, prediction: Any, choices: List[Any]) -> bool:
    """Match logic reused from evaluation.py for consistency."""
    prediction_tokens = _tokenize(prediction)
    answer_tokens = _tokenize(answer)

    if not prediction_tokens:
        return False

    incorrect_tokens = set()
    for choice in choices or []:
        choice_tokens = _tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    cond1 = answer_tokens.issubset(prediction_tokens)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    return cond1 and cond2


def _is_correct(sample: Dict[str, Any]) -> bool:
    gold = sample.get("gold_answer", sample.get("answer", ""))
    pred = sample.get("model_prediction", "")
    choices = sample.get("choices", [])

    if isinstance(choices, list) and choices:
        return string_match(gold, pred, choices)

    return _normalize_text(gold) == _normalize_text(pred)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        if isinstance(data.get("items"), list):
            return data["items"]

    raise ValueError(f"Unsupported JSON structure in {path}")


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _find_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    if FINAL_CHANNEL_DELIM in text:
        text = text.split(FINAL_CHANNEL_DELIM)[-1]
    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    last_close = text.rfind("}")
    while last_close >= 0:
        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
            if depth == 0:
                candidate = text[i:last_close + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    break
        last_close = text.rfind("}", 0, last_close)

    return None


def _canonicalize_choice(predicted: str, choices: List[Any]) -> str:
    if not predicted or predicted == "NONE":
        return "NONE"

    for c in choices:
        if _normalize_text(c) == _normalize_text(predicted):
            return str(c)

    return "NONE"


class JudgeClient:
    def __init__(
        self,
        host: str,
        port: int,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        retries: int,
    ):
        self.host = host
        self.port = port
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/v1/chat/completions"

    def check_health(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/health",
                timeout=5,
                proxies={"http": None, "https": None},
            )
            return r.status_code == 200
        except Exception:
            return False

    def _post(self, payload: Dict[str, Any]) -> str:
        resp = requests.post(
            self.api_url,
            json=payload,
            timeout=self.timeout,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def call(self, prompt: str) -> str:
        base_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evidence auditor. "
                        "Accept only explicit evidence from provided tool outputs. "
                        "Reject speculative reasoning. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }

        payload_variants = []
        payload_max_tokens = dict(base_payload)
        payload_max_tokens["max_tokens"] = self.max_tokens
        payload_variants.append(payload_max_tokens)

        payload_max_completion = dict(base_payload)
        payload_max_completion["max_completion_tokens"] = self.max_tokens
        payload_variants.append(payload_max_completion)

        last_err = None
        for _ in range(self.retries):
            for payload in payload_variants:
                try:
                    return self._post(payload)
                except Exception as exc:
                    last_err = exc

        return (
            '{"can_be_answered_with_top3": false, '
            '"best_tool": "NONE", '
            '"predicted_answer_from_best_tool": "NONE", '
            '"confidence": 0.0, '
            f'"cot_reasoning": "Judge call failed: {str(last_err)}", '
            '"tool_analysis": []}'
        )


def _serialize_tool_output(obj: Any, char_limit: int) -> str:
    raw = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(raw) > char_limit:
        raw = raw[:char_limit] + "\n... (truncated)"
    return raw


def _get_top3_tools(cached_item: Dict[str, Any]) -> List[str]:
    top3 = cached_item.get("top_3_tools", [])
    if not isinstance(top3, list):
        top3 = []
    top3 = [str(t) for t in top3 if t]
    if top3:
        return top3[:3]

    tool_outputs = cached_item.get("tool_outputs", {})
    if isinstance(tool_outputs, dict):
        return list(tool_outputs.keys())[:3]

    return []


def _build_prompt(
    wrong_row: Dict[str, Any],
    cached_row: Dict[str, Any],
    top3: List[str],
    per_tool_char_limit: int,
) -> Tuple[str, List[str], str]:
    question = wrong_row.get("question", cached_row.get("question", ""))
    choices = wrong_row.get("choices", cached_row.get("choices", []))
    gold = wrong_row.get("gold_answer", cached_row.get("answer", ""))
    wrong_pred = wrong_row.get("model_prediction", "")
    wrong_completion = wrong_row.get("raw_final_response", wrong_row.get("phase1", ""))
    wrong_completion = str(wrong_completion or "")[:2000]

    tool_outputs = cached_row.get("tool_outputs", {})
    if not isinstance(tool_outputs, dict):
        tool_outputs = {}

    tool_sections = []
    for tool_name in top3:
        desc = TOOL_INFO.get(tool_name, "Audio analysis tool output.")
        out_obj = tool_outputs.get(tool_name, None)
        if out_obj is None:
            out_str = "No cached output found for this tool."
        else:
            out_str = _serialize_tool_output(out_obj, per_tool_char_limit)

        tool_sections.append(
            "Tool: " + tool_name + "\n"
            + "Description: " + desc + "\n"
            + "Output:\n" + out_str
        )

    if not tool_sections:
        tool_sections = ["No top-3 tool outputs available."]

    choices_block = "\n".join([f"- {c}" for c in choices])
    tools_block = "\n\n".join(tool_sections)

    prompt = f"""You are a strict evidence auditor for an incorrect no-tool answer on an audio multiple-choice question.

Question:
{question}

Choices:
{choices_block}

Gold answer:
{gold}

Wrong no-tool prediction:
{wrong_pred}

Wrong no-tool completion:
{wrong_completion}

Top-3 cached tool outputs:
{tools_block}

Task:
Determine whether at least one of these top-3 tool outputs contains enough evidence to correctly answer the question.

**Gold answer is always correct and Tool output might be incorrect.***

Decision policy:
1) Start from can_be_answered_with_top3 = false by default.
2) Switch to true only if one tool provides explicit, choice-discriminative evidence that directly supports exactly one choice.
3) If reasoning requires interpretation, plausibility, scenario assumptions, or outside knowledge, keep false.

Hard constraints:
1) Use only evidence explicitly present in the provided tool outputs.
2) Do not use outside knowledge, plausibility, or scenario assumptions.
3) The following are NOT sufficient evidence: "could", "might", "possibly", "likely", "consistent with", "in the context", "when interpreted".
4) If the tool evidence is ambiguous or can fit multiple choices, set can_be_answered_with_top3 to false.
5) If no direct and choice-discriminative evidence exists, set best_tool to NONE and predicted_answer_from_best_tool to NONE.
6) Do not treat the gold answer as evidence; it is shown only for audit reference.

Positive decision requirements (must all hold when can_be_answered_with_top3=true):
1) best_tool is one of the provided top-3 tools.
2) predicted_answer_from_best_tool exactly matches one listed choice.
3) tool_analysis for best_tool includes direct quote snippets copied from tool output.
4) cot_reasoning cites explicit evidence and explains why alternatives are not supported.
5) Every quote in evidence_quotes must be an exact contiguous substring from that tool output.

If can_be_answered_with_top3 is false:
1) Set best_tool to NONE.
2) Set predicted_answer_from_best_tool to NONE.
3) Set confidence <= 0.49.

Return only one JSON object (no extra text) using this schema:
{{
  "can_be_answered_with_top3": true or false,
  "best_tool": "tool_name_or_NONE",
  "predicted_answer_from_best_tool": "exact_choice_or_NONE",
  "confidence": 0.0_to_1.0,
  "cot_reasoning": "detailed reasoning",
  "tool_analysis": [
    {{
      "tool": "tool_name",
      "can_support_gold": true or false,
            "evidence": "short evidence summary with direct quote snippets; use NO_DIRECT_EVIDENCE if none",
            "evidence_quotes": ["exact snippet 1", "exact snippet 2"]
    }}
  ]
}}
"""
    return prompt, choices, str(gold)


def _parse_judge_output(raw: str, choices: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    obj = _find_last_json_object(raw)
    if not isinstance(obj, dict):
        return None, "parse_error"

    can_ans = obj.get("can_be_answered_with_top3", False)
    if isinstance(can_ans, str):
        can_ans = can_ans.strip().lower() in {"true", "yes", "1"}

    best_tool = str(obj.get("best_tool", "NONE") or "NONE")
    pred_from_tool = str(obj.get("predicted_answer_from_best_tool", "NONE") or "NONE")
    pred_from_tool = _canonicalize_choice(pred_from_tool, choices)

    conf = obj.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    parsed = {
        "can_be_answered_with_top3": bool(can_ans),
        "best_tool": best_tool,
        "predicted_answer_from_best_tool": pred_from_tool,
        "confidence": conf,
        "cot_reasoning": str(obj.get("cot_reasoning", "")),
        "tool_analysis": obj.get("tool_analysis", []),
    }
    return parsed, "ok"


def _build_cached_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = row.get("id")
        if qid and qid not in out:
            out[qid] = row
    return out


def _is_wrong_no_tool(sample: Dict[str, Any], require_no_tool: bool = True) -> bool:
    if _is_correct(sample):
        return False

    if not require_no_tool:
        return True

    result_type = sample.get("result_type")
    tools_selected = sample.get("tools_selected", [])

    no_tools_list = isinstance(tools_selected, list) and len(tools_selected) == 0
    result_direct = result_type == "direct_answer" if result_type is not None else True

    return no_tools_list and result_direct


def _evaluate_one(
    idx: int,
    wrong_row: Dict[str, Any],
    cached_by_id: Dict[str, Dict[str, Any]],
    judge: JudgeClient,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    qid = wrong_row.get("id", "")
    result: Dict[str, Any] = {
        "id": qid,
        "question": wrong_row.get("question", ""),
        "gold_answer": wrong_row.get("gold_answer", ""),
        "model_prediction": wrong_row.get("model_prediction", ""),
        "status": "ok",
    }

    if not qid:
        result["status"] = "missing_id"
        return idx, result

    cached = cached_by_id.get(qid)
    if cached is None:
        result["status"] = "missing_cached_item"
        return idx, result

    top3 = _get_top3_tools(cached)
    if not top3:
        result["status"] = "missing_top3_tools"
        return idx, result

    prompt, choices, gold = _build_prompt(wrong_row, cached, top3, args.per_tool_char_limit)
    raw = judge.call(prompt)
    parsed, parse_status = _parse_judge_output(raw, choices)

    result["top_3_tools"] = top3
    result["gold_answer"] = gold

    if parse_status != "ok" or parsed is None:
        result["status"] = "judge_parse_failed"
        if args.keep_raw:
            result["judge_raw_response"] = raw
        return idx, result

    result.update(
        {
            "judge_can_be_answered_with_top3": parsed["can_be_answered_with_top3"],
            "judge_best_tool": parsed["best_tool"],
            "judge_predicted_answer_from_best_tool": parsed["predicted_answer_from_best_tool"],
            "judge_confidence": parsed["confidence"],
            "judge_cot_reasoning": parsed["cot_reasoning"],
            "judge_tool_analysis": parsed["tool_analysis"],
        }
    )

    pred_from_tool = parsed["predicted_answer_from_best_tool"]
    if pred_from_tool != "NONE":
        result["judge_predicted_matches_gold"] = string_match(gold, pred_from_tool, choices)
    else:
        result["judge_predicted_matches_gold"] = False

    if args.keep_raw:
        result["judge_raw_response"] = raw

    return idx, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify whether incorrect no-tool questions can be answered by one of cached top-3 tools "
            "using a judge LLM."
        )
    )
    parser.add_argument(
        "--direct-results",
        type=str,
        default="results/inference/101490_tools/101490_direct_base.jsonl",
        help="Path to no-tool inference results (.json or .jsonl).",
    )
    parser.add_argument(
        "--cached-top3",
        type=str,
        default="mmau-test-mini-cached-top3.json",
        help="Path to cached dataset with top_3_tools and tool_outputs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference/101490_tools/verify_top3_tool_capablity.jsonl",
        help="Output jsonl for all judged wrong questions.",
    )
    parser.add_argument(
        "--output-yes",
        type=str,
        default="results/inference/101490_tools/verify_top3_tool_capablity.answerable.jsonl",
        help="Output jsonl for subset judged answerable with top-3.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/inference/101490_tools/verify_top3_tool_capablity.summary.json",
        help="Output JSON summary.",
    )

    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Evaluate all rows; by default only wrong rows are evaluated.",
    )
    parser.add_argument(
        "--allow-tool-rows",
        action="store_true",
        help="By default we require no-tool rows; set this to include tool rows too.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--per-tool-char-limit", type=int, default=3000)
    parser.add_argument("--keep-raw", action="store_true")

    parser.add_argument("--judge-model", type=str, default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--judge-host", type=str, default="192.168.1.19")
    parser.add_argument("--judge-port", type=int, default=8000)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)

    args = parser.parse_args()

    direct_rows = load_json_or_jsonl(args.direct_results)
    cached_rows = load_json_or_jsonl(args.cached_top3)
    cached_by_id = _build_cached_index(cached_rows)

    require_no_tool = not args.allow_tool_rows
    if args.include_correct:
        selected_rows = direct_rows
    else:
        selected_rows = [
            row for row in direct_rows if _is_wrong_no_tool(row, require_no_tool=require_no_tool)
        ]

    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]

    print(f"Loaded direct rows: {len(direct_rows)}")
    print(f"Loaded cached rows: {len(cached_rows)}")
    print(f"Rows selected for judge evaluation: {len(selected_rows)}")

    judge = JudgeClient(
        host=args.judge_host,
        port=args.judge_port,
        model=args.judge_model,
        max_tokens=args.judge_max_tokens,
        temperature=args.judge_temperature,
        timeout=args.timeout,
        retries=args.retries,
    )

    healthy = judge.check_health()
    print(f"Judge health ({judge.base_url}/health): {'OK' if healthy else 'NOT READY'}")

    results: List[Optional[Dict[str, Any]]] = [None] * len(selected_rows)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(_evaluate_one, i, row, cached_by_id, judge, args)
            for i, row in enumerate(selected_rows)
        ]

        done = 0
        for future in as_completed(futures):
            idx, row_result = future.result()
            results[idx] = row_result
            done += 1
            if done % args.progress_every == 0 or done == len(futures):
                print(f"Progress: {done}/{len(futures)}")

    final_results = [r for r in results if r is not None]
    answerable = [
        r
        for r in final_results
        if r.get("status") == "ok" and r.get("judge_can_be_answered_with_top3") is True
    ]

    write_jsonl(args.output, final_results)
    write_jsonl(args.output_yes, answerable)

    status_counts: Dict[str, int] = {}
    for r in final_results:
        st = r.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    summary = {
        "direct_rows_total": len(direct_rows),
        "rows_evaluated_by_judge": len(final_results),
        "rows_answerable_with_top3": len(answerable),
        "rows_answerable_with_top3_and_pred_match_gold": sum(
            1 for r in answerable if r.get("judge_predicted_matches_gold") is True
        ),
        "status_counts": status_counts,
        "judge_model": args.judge_model,
        "judge_host": args.judge_host,
        "judge_port": args.judge_port,
        "judge_max_tokens": args.judge_max_tokens,
        "input_direct_results": args.direct_results,
        "input_cached_top3": args.cached_top3,
        "output_all": args.output,
        "output_answerable_only": args.output_yes,
    }

    out_dir = os.path.dirname(args.summary_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
