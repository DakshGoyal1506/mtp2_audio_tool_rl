# Reward Function Design

> **TL;DR:** Previously the reward was binary (correct=1, wrong=0). Now we use a **soft, multi-component reward** that gives partial credit — token F1 for near-correct answers (0.7), credit for mentioning the gold answer in reasoning (0.15), and a separate format score penalizing missing tags. Tool usage is rewarded **only when needed** — correct answers without tools score higher than correct answers with tools, preventing unnecessary tool calls.

## Total Reward

```
R = fw × format + cw × correctness + mw × mention + tw × tool_bonus
```

Default weights (configurable via YAML):

| Component | Weight | Key |
|-----------|--------|-----|
| Format | 0.10 | `format_reward_weight` |
| Correctness | 0.60 | `correctness_reward_weight` |
| Mention | 0.15 | `option_mention_weight` |
| Tool bonus | 0.15 | `tool_bonus_weight` |

---

## 1. Format Score

Starts at 0.0, penalizes missing/malformed tags. Can go negative.

| Condition | Penalty |
|-----------|---------|
| `<answer>` absent | -0.5 |
| `<answer>` present, `</answer>` missing | -0.2 |
| Each missing `<think>` (expect 1, or 2 if `<tool>` present) | -0.1 |
| Each missing `</think>` (same count rule) | -0.1 |
| `<tool>` present, `</tool>` missing | -0.2 |

Final: `max(0, 1 + penalty)` → used as `f_score` in [0, 1].

---

## 2. Correctness Score

Extracted from `<answer>...</answer>` (with fallback to `<tag>...</` for malformed closings).

| Condition | Score |
|-----------|-------|
| Exact match (lowercased) with gold | 1.0 |
| Token F1 ≥ 0.5 with gold | 0.7 |
| Gold mentioned in body (no `<answer>` tag) | 0.15 |
| Otherwise | 0.0 |

---

## 3. Mention Score

Only active when correctness < 0.5 (cold-start helper). When correctness ≥ 0.5, defaults to 1.0.

| Condition | Score |
|-----------|-------|
| Gold answer in body text | 0.5 |
| Any choice option in body text | 0.1 |
| Nothing relevant | 0.0 |

---

## 4. Tool Usage Score

Incentivizes efficient tool use: correct without tool > correct with tool > wrong with tool.

| Condition | Score |
|-----------|-------|
| Correct (c_score ≥ 0.7) WITHOUT tool call | **1.0** (best — efficient) |
| Correct (c_score ≥ 0.7) WITH tool call | 0.5 (good but tool wasn't needed) |
| Wrong, but well-formed tool call | 0.3 (structured attempt) |
| Wrong, no tool | 0.0 |

---

## Example Rewards

| Scenario | Format | Correct | Mention | Tool | Total |
|----------|--------|---------|---------|------|-------|
| Perfect + tool | 0.0 (→1.0) | 1.0 | 1.0 | 1.0 | 1.0 |
| Perfect, no tool | 0.0 (→1.0) | 1.0 | 1.0 | 0.0 | 0.85 |
| Right answer, missing `<think>` | -0.2 (→0.8) | 1.0 | 1.0 | 0.0 | 0.83 |
| Wrong answer, good format | 0.0 (→1.0) | 0.0 | 0.0 | 0.0 | 0.10 |
| No tags, gibberish | -0.6 (→0.4) | 0.0 | 0.0 | 0.0 | 0.04 |
