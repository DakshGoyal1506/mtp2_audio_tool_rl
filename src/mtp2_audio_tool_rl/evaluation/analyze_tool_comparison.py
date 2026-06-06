#!/usr/bin/env python3
"""
Compare no-tool (direct) vs forced-tool results in a 2×2 matrix.

Categories:
  1. No-tool ✓, Tool ✓  — Easy (model handles both)
  2. No-tool ✓, Tool ✗  — Adversarial (tool hurts; minimize this)
  3. No-tool ✗, Tool ✓  — Helpful tool (tool rescues)
  4. No-tool ✗, Tool ✗  — Very hard (neither works)

Usage:
  python analyze_tool_comparison.py \
    --direct-results results/inference/101490_tools/101490_direct_base.jsonl \
    --forced-results results/inference/101490_tools/101490_forced_tool.jsonl \
    --best-tool-map results/inference/101490_tools/best_tool_selection.jsonl \
    --output results/inference/101490_tools/comparison_analysis.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


# ── Matching (same as evaluation.py / verify script) ─────────────────────────

def _tokenize(text: Any) -> set:
    return set(re.findall(r"\b\w+\b", str(text or "").lower()))


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def string_match(answer: Any, prediction: Any, choices: List[Any]) -> bool:
    prediction_tokens = _tokenize(prediction)
    answer_tokens = _tokenize(answer)
    if not prediction_tokens:
        return False
    incorrect_tokens = set()
    for choice in choices or []:
        choice_tokens = _tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    return answer_tokens.issubset(prediction_tokens) and prediction_tokens.isdisjoint(incorrect_tokens)


def is_correct(sample: Dict[str, Any]) -> bool:
    gold = sample.get("gold_answer", sample.get("answer", ""))
    pred = sample.get("model_prediction", "")
    choices = sample.get("choices", [])
    if isinstance(choices, list) and choices:
        return string_match(gold, pred, choices)
    return _normalize_text(gold) == _normalize_text(pred)


# ── IO ───────────────────────────────────────────────────────────────────────

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    rows.append(json.loads(s))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="2×2 comparison: no-tool vs forced-tool")
    parser.add_argument("--direct-results", type=str,
                        default="results/inference/101490_tools/101490_direct_base.jsonl")
    parser.add_argument("--forced-results", type=str,
                        default="results/inference/101490_tools/101490_forced_tool.jsonl")
    parser.add_argument("--best-tool-map", type=str, default=None,
                        help="Optional: best_tool_selection.jsonl for tool metadata")
    parser.add_argument("--output", type=str,
                        default="results/inference/101490_tools/comparison_analysis.json")
    args = parser.parse_args()

    direct_rows = load_json_or_jsonl(args.direct_results)
    forced_rows = load_json_or_jsonl(args.forced_results)

    # Index by id
    direct_by_id = {r["id"]: r for r in direct_rows if "id" in r}
    forced_by_id = {r["id"]: r for r in forced_rows if "id" in r}

    # Optional tool map
    tool_map: Dict[str, Dict[str, Any]] = {}
    if args.best_tool_map and os.path.exists(args.best_tool_map):
        tm_rows = load_json_or_jsonl(args.best_tool_map)
        tool_map = {r["id"]: r for r in tm_rows if "id" in r}

    # Common ids
    common_ids = sorted(set(direct_by_id.keys()) & set(forced_by_id.keys()))
    print(f"Direct results: {len(direct_rows)}")
    print(f"Forced-tool results: {len(forced_rows)}")
    print(f"Common IDs for comparison: {len(common_ids)}")

    # ── 2×2 classification ───────────────────────────────────────────
    categories = {
        "easy": [],           # No-tool ✓, Tool ✓
        "adversarial": [],    # No-tool ✓, Tool ✗
        "helpful_tool": [],   # No-tool ✗, Tool ✓
        "very_hard": [],      # No-tool ✗, Tool ✗
    }

    # Breakdown by metadata
    by_task = defaultdict(lambda: {"easy": 0, "adversarial": 0, "helpful_tool": 0, "very_hard": 0})
    by_category = defaultdict(lambda: {"easy": 0, "adversarial": 0, "helpful_tool": 0, "very_hard": 0})
    by_difficulty = defaultdict(lambda: {"easy": 0, "adversarial": 0, "helpful_tool": 0, "very_hard": 0})
    by_tool = defaultdict(lambda: {"easy": 0, "adversarial": 0, "helpful_tool": 0, "very_hard": 0})
    by_tool_confidence = defaultdict(lambda: {"easy": [], "adversarial": [], "helpful_tool": [], "very_hard": []})

    for qid in common_ids:
        d = direct_by_id[qid]
        f = forced_by_id[qid]

        d_correct = is_correct(d)
        f_correct = is_correct(f)

        if d_correct and f_correct:
            cat = "easy"
        elif d_correct and not f_correct:
            cat = "adversarial"
        elif not d_correct and f_correct:
            cat = "helpful_tool"
        else:
            cat = "very_hard"

        forced_tool = f.get("forced_tool", "unknown")
        tool_conf = f.get("tool_selection_confidence", tool_map.get(qid, {}).get("confidence", 0.0))

        entry = {
            "id": qid,
            "question": d.get("question", ""),
            "gold_answer": d.get("gold_answer", d.get("answer", "")),
            "direct_prediction": d.get("model_prediction", ""),
            "forced_prediction": f.get("model_prediction", ""),
            "direct_correct": d_correct,
            "forced_correct": f_correct,
            "forced_tool": forced_tool,
            "tool_confidence": tool_conf,
            "task": d.get("task", ""),
            "category": d.get("category", ""),
            "sub-category": d.get("sub-category", ""),
            "difficulty": d.get("difficulty", ""),
        }
        categories[cat].append(entry)

        task = d.get("task", "unknown")
        cat_name = d.get("category", "unknown")
        diff = d.get("difficulty", "unknown")
        # Ensure hashable keys (some fields may be lists in the JSON)
        if isinstance(task, list):
            task = ", ".join(str(x) for x in task)
        if isinstance(cat_name, list):
            cat_name = ", ".join(str(x) for x in cat_name)
        if isinstance(diff, list):
            diff = ", ".join(str(x) for x in diff)
        by_task[task][cat] += 1
        by_category[cat_name][cat] += 1
        by_difficulty[diff][cat] += 1
        by_tool[forced_tool][cat] += 1
        by_tool_confidence[forced_tool][cat].append(tool_conf)

    # ── Summary ──────────────────────────────────────────────────────
    total = len(common_ids)
    direct_acc = sum(1 for qid in common_ids if is_correct(direct_by_id[qid])) / max(total, 1)
    forced_acc = sum(1 for qid in common_ids if is_correct(forced_by_id[qid])) / max(total, 1)

    # Oracle: correct if EITHER is correct
    oracle_acc = sum(
        1 for qid in common_ids
        if is_correct(direct_by_id[qid]) or is_correct(forced_by_id[qid])
    ) / max(total, 1)

    matrix = {
        "easy (both ✓)": len(categories["easy"]),
        "adversarial (direct ✓, tool ✗)": len(categories["adversarial"]),
        "helpful_tool (direct ✗, tool ✓)": len(categories["helpful_tool"]),
        "very_hard (both ✗)": len(categories["very_hard"]),
    }

    # Tool-level analysis
    tool_analysis = {}
    for tool_name, counts in sorted(by_tool.items()):
        t_total = sum(counts.values())
        conf_all = []
        for cat_key in ["easy", "adversarial", "helpful_tool", "very_hard"]:
            conf_all.extend(by_tool_confidence[tool_name][cat_key])
        avg_conf = sum(conf_all) / max(len(conf_all), 1)

        tool_analysis[tool_name] = {
            "total": t_total,
            **counts,
            "tool_accuracy": round((counts["easy"] + counts["helpful_tool"]) / max(t_total, 1), 4),
            "hurt_rate": round(counts["adversarial"] / max(t_total, 1), 4),
            "rescue_rate": round(counts["helpful_tool"] / max(t_total, 1), 4),
            "avg_confidence": round(avg_conf, 3),
        }

    summary = {
        "total_compared": total,
        "direct_accuracy": round(direct_acc, 4),
        "forced_tool_accuracy": round(forced_acc, 4),
        "oracle_accuracy": round(oracle_acc, 4),
        "net_tool_impact": round(forced_acc - direct_acc, 4),
        "matrix": matrix,
        "matrix_pct": {k: round(v / max(total, 1) * 100, 2) for k, v in matrix.items()},
        "by_task": dict(by_task),
        "by_category": dict(by_category),
        "by_difficulty": dict(by_difficulty),
        "tool_analysis": tool_analysis,
    }

    # ── Save outputs ─────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # Save per-category detail files
    base = os.path.splitext(args.output)[0]
    for cat_name, entries in categories.items():
        detail_path = f"{base}.{cat_name}.jsonl"
        with open(detail_path, "w", encoding="utf-8") as fp:
            for e in entries:
                fp.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ── Print report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2×2 COMPARISON: No-Tool vs Forced-Tool")
    print("=" * 70)
    print(f"Total samples compared: {total}")
    print(f"Direct (no-tool) accuracy:  {direct_acc:.2%}")
    print(f"Forced-tool accuracy:       {forced_acc:.2%}")
    print(f"Oracle accuracy (best of):  {oracle_acc:.2%}")
    print(f"Net tool impact:            {forced_acc - direct_acc:+.2%}")
    print()

    print("┌─────────────────────┬──────────┬──────────┐")
    print("│                     │ Tool ✓   │ Tool ✗   │")
    print("├─────────────────────┼──────────┼──────────┤")
    print(f"│ No-tool ✓           │ {len(categories['easy']):>5}    │ {len(categories['adversarial']):>5}    │")
    print(f"│ No-tool ✗           │ {len(categories['helpful_tool']):>5}    │ {len(categories['very_hard']):>5}    │")
    print("└─────────────────────┴──────────┴──────────┘")
    print()

    print("Interpretation:")
    print(f"  Easy (both correct):           {len(categories['easy']):>4} ({len(categories['easy'])/max(total,1):.1%})")
    print(f"  Adversarial (tool hurts):      {len(categories['adversarial']):>4} ({len(categories['adversarial'])/max(total,1):.1%})")
    print(f"  Helpful tool (tool rescues):   {len(categories['helpful_tool']):>4} ({len(categories['helpful_tool'])/max(total,1):.1%})")
    print(f"  Very hard (neither correct):   {len(categories['very_hard']):>4} ({len(categories['very_hard'])/max(total,1):.1%})")
    print()

    # Tool breakdown
    print("Per-tool analysis:")
    print(f"  {'Tool':<25} {'Total':>5} {'Acc':>6} {'Hurt':>6} {'Rescue':>6} {'Conf':>5}")
    print(f"  {'─'*25} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
    for tool_name in sorted(tool_analysis, key=lambda t: -tool_analysis[t]["total"]):
        ta = tool_analysis[tool_name]
        print(f"  {tool_name:<25} {ta['total']:>5} {ta['tool_accuracy']:>5.1%} "
              f"{ta['hurt_rate']:>5.1%} {ta['rescue_rate']:>5.1%} {ta['avg_confidence']:>5.2f}")
    print()

    # By difficulty
    print("By difficulty:")
    for diff in sorted(by_difficulty):
        d = by_difficulty[diff]
        d_total = sum(d.values())
        if d_total == 0:
            continue
        print(f"  {diff:<12}: easy={d['easy']:>3} adversarial={d['adversarial']:>3} "
              f"helpful={d['helpful_tool']:>3} hard={d['very_hard']:>3}  "
              f"(direct={100*(d['easy']+d['adversarial'])/d_total:.0f}% "
              f"tool={100*(d['easy']+d['helpful_tool'])/d_total:.0f}%)")

    # By task
    print("\nBy task:")
    for task in sorted(by_task):
        d = by_task[task]
        d_total = sum(d.values())
        if d_total == 0:
            continue
        print(f"  {task:<12}: easy={d['easy']:>3} adversarial={d['adversarial']:>3} "
              f"helpful={d['helpful_tool']:>3} hard={d['very_hard']:>3}  "
              f"(direct={100*(d['easy']+d['adversarial'])/d_total:.0f}% "
              f"tool={100*(d['easy']+d['helpful_tool'])/d_total:.0f}%)")

    print(f"\nResults saved to: {args.output}")
    print(f"Detail files: {base}.{{easy,adversarial,helpful_tool,very_hard}}.jsonl")


if __name__ == "__main__":
    main()
