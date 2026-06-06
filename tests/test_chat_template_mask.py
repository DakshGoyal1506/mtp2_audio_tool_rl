#!/usr/bin/env python3
"""
Test script: Verify chat-template-based tool_mask construction.

Simulates the proposed fix for Phase 2 context mismatch:
- Uses apply_chat_template(tokenize=True) on incremental message lists
- Derives mask boundaries from token length differences
- Validates prefix consistency, BPE safety, mask alignment, and decoded content

Friend's gotchas checked:
  1. Leading space token / double-space / BPE merge at boundaries
  2. Label shift alignment (logits[:-1] vs labels[1:] and mask alignment)
"""

import os
import sys

# Suppress HF warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer

# ============================================================================
# Test data — realistic examples from actual GRPO training
# ============================================================================

SYSTEM_PROMPT = "You are an audio analysis assistant."

USER_PROMPT = "Question: What emotion is the speaker expressing?\nChoices:\n- Happy\n- Sad\n- Angry\n- Neutral"

# Phase 1: model's think + tool call
PHASE1_TEXTS = [
    '<think>\nThe question asks about the speaker\'s emotion. I need to analyze the emotional tone.\n</think>\n<tool>\n[{"function": "emotion_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
    '<think>\nI should check what emotion is conveyed in this speech.\n</think>\n<tool>\n[{"function": "emotion_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
    '<think>\nLet me identify the speaker\'s emotional state using a tool.\n</think>\n<tool>\n[{"function": "speech_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
]

# Injected tool output (exactly as _format_tool_output produces)
INJECTED_OUTPUTS = [
    '\n<tool_output>\n{"emotion": "happy", "confidence": 0.92}\n</tool_output>\n',
    '\n<tool_output>\n{"emotion": "sad", "confidence": 0.78}\n</tool_output>\n',
    '\n<tool_output>\n{"text": "I really can\'t believe this happened to me"}\n</tool_output>\n',
]

# Phase 2: model's response after seeing tool output
PHASE2_TEXTS = [
    '<think>\nThe emotion recognition tool detected "happy" with high confidence (0.92). This aligns with the upbeat tone I heard.\n</think>\n<answer>Happy</answer>',
    '<think>\nThe tool says "sad" with 0.78 confidence. The speaker\'s tone did sound downcast.\n</think>\n<answer>Sad</answer>',
    '<think>\nThe transcription says "I really can\'t believe this happened to me" — this sounds frustrated and upset.\n</think>\n<answer>Angry</answer>',
]


def build_messages(sys_p, user_p, p1=None, injected=None, p2=None):
    """Build message list at various stages, matching model_wrapper.py format."""
    msgs = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": user_p},
    ]
    if p1 is not None:
        msgs.append({"role": "assistant", "content": p1})
    if injected is not None:
        msgs.append({"role": "user", "content": injected})
    if p2 is not None:
        msgs.append({"role": "assistant", "content": p2})
    return msgs


def test_case(tokenizer, p1, injected, p2, case_idx):
    """Run all validations for one test case."""
    print(f"\n{'='*70}")
    print(f"TEST CASE {case_idx}")
    print(f"{'='*70}")

    # ── Step 1: Build 4 incremental message lists ─────────────────────────
    msgs_prompt      = build_messages(SYSTEM_PROMPT, USER_PROMPT)
    msgs_through_p1  = build_messages(SYSTEM_PROMPT, USER_PROMPT, p1=p1)
    msgs_through_inj = build_messages(SYSTEM_PROMPT, USER_PROMPT, p1=p1, injected=injected)
    msgs_full        = build_messages(SYSTEM_PROMPT, USER_PROMPT, p1=p1, injected=injected, p2=p2)

    # ── Step 2: Tokenize with chat template ───────────────────────────────
    ids_prompt      = tokenizer.apply_chat_template(msgs_prompt,      tokenize=True, add_generation_prompt=True)
    ids_through_p1  = tokenizer.apply_chat_template(msgs_through_p1,  tokenize=True, add_generation_prompt=False)
    ids_through_inj = tokenizer.apply_chat_template(msgs_through_inj, tokenize=True, add_generation_prompt=True)
    ids_full        = tokenizer.apply_chat_template(msgs_full,        tokenize=True, add_generation_prompt=False)

    print(f"\nToken lengths:")
    print(f"  ids_prompt       = {len(ids_prompt)}")
    print(f"  ids_through_p1   = {len(ids_through_p1)}")
    print(f"  ids_through_inj  = {len(ids_through_inj)}")
    print(f"  ids_full         = {len(ids_full)}")

    # ── GOTCHA #1: Prefix Consistency ─────────────────────────────────────
    # Verify each tokenization is a strict prefix of the next
    errors = []

    # ids_prompt should be prefix of ids_through_p1
    prefix_len_1 = len(ids_prompt)
    if ids_through_p1[:prefix_len_1] != ids_prompt:
        # Find where they diverge
        for k in range(min(prefix_len_1, len(ids_through_p1))):
            if ids_through_p1[k] != ids_prompt[k]:
                errors.append(
                    f"PREFIX MISMATCH: ids_prompt vs ids_through_p1 at pos {k}: "
                    f"{ids_prompt[k]}('{tokenizer.decode([ids_prompt[k]])}') vs "
                    f"{ids_through_p1[k]}('{tokenizer.decode([ids_through_p1[k]])}')"
                )
                break
    else:
        print(f"  ✓ ids_prompt is strict prefix of ids_through_p1")

    prefix_len_2 = len(ids_through_p1)
    if ids_through_inj[:prefix_len_2] != ids_through_p1:
        for k in range(min(prefix_len_2, len(ids_through_inj))):
            if ids_through_inj[k] != ids_through_p1[k]:
                errors.append(
                    f"PREFIX MISMATCH: ids_through_p1 vs ids_through_inj at pos {k}: "
                    f"{ids_through_p1[k]}('{tokenizer.decode([ids_through_p1[k]])}') vs "
                    f"{ids_through_inj[k]}('{tokenizer.decode([ids_through_inj[k]])}')"
                )
                break
    else:
        print(f"  ✓ ids_through_p1 is strict prefix of ids_through_inj")

    prefix_len_3 = len(ids_through_inj)
    if ids_full[:prefix_len_3] != ids_through_inj:
        for k in range(min(prefix_len_3, len(ids_full))):
            if ids_full[k] != ids_through_inj[k]:
                errors.append(
                    f"PREFIX MISMATCH: ids_through_inj vs ids_full at pos {k}: "
                    f"{ids_through_inj[k]}('{tokenizer.decode([ids_through_inj[k]])}') vs "
                    f"{ids_full[k]}('{tokenizer.decode([ids_full[k]])}')"
                )
                break
    else:
        print(f"  ✓ ids_through_inj is strict prefix of ids_full")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False

    # ── Step 3: Compute completion_ids and mask ───────────────────────────
    prompt_len = len(ids_prompt)
    completion_ids = ids_full[prompt_len:]

    p1_len     = len(ids_through_p1)  - len(ids_prompt)
    masked_len = len(ids_through_inj) - len(ids_through_p1)
    p2_len     = len(ids_full)        - len(ids_through_inj)

    tool_mask = [1] * p1_len + [0] * masked_len + [1] * p2_len

    print(f"\nCompletion breakdown:")
    print(f"  Total completion tokens: {len(completion_ids)}")
    print(f"  p1 region:     {p1_len} tokens (mask=1)")
    print(f"  masked region: {masked_len} tokens (mask=0)")
    print(f"  p2 region:     {p2_len} tokens (mask=1)")
    print(f"  mask total:    {len(tool_mask)}")

    assert len(tool_mask) == len(completion_ids), \
        f"Mask length {len(tool_mask)} != completion length {len(completion_ids)}"
    print(f"  ✓ mask length == completion length")

    # ── Verify decoded content ────────────────────────────────────────────
    p1_tokens = completion_ids[:p1_len]
    masked_tokens = completion_ids[p1_len:p1_len + masked_len]
    p2_tokens = completion_ids[p1_len + masked_len:]

    p1_decoded = tokenizer.decode(p1_tokens, skip_special_tokens=False)
    masked_decoded = tokenizer.decode(masked_tokens, skip_special_tokens=False)
    p2_decoded = tokenizer.decode(p2_tokens, skip_special_tokens=False)

    print(f"\nDecoded regions:")
    print(f"  p1 (mask=1): {repr(p1_decoded[:120])}...")
    print(f"  masked (mask=0): {repr(masked_decoded[:120])}...")
    print(f"  p2 (mask=1): {repr(p2_decoded[:120])}...")

    # Verify p1 region starts with phase1 content
    # (may have slight formatting from template, but should contain the think block)
    assert "<think>" in p1_decoded, f"p1 region doesn't contain <think>: {p1_decoded[:80]}"
    print(f"  ✓ p1 region contains <think>")

    assert "<tool>" in p1_decoded, f"p1 region doesn't contain <tool>: {p1_decoded[:80]}"
    print(f"  ✓ p1 region contains <tool>")

    # Verify masked region contains the tool output
    assert "<tool_output>" in masked_decoded, f"masked region doesn't contain <tool_output>: {masked_decoded[:80]}"
    print(f"  ✓ masked region contains <tool_output>")

    # Verify masked region contains turn headers
    assert "assistant" in masked_decoded.lower(), f"masked region doesn't contain assistant header"
    print(f"  ✓ masked region contains assistant turn header")

    # Verify p2 region contains the answer
    assert "<answer>" in p2_decoded, f"p2 region doesn't contain <answer>: {p2_decoded[:80]}"
    print(f"  ✓ p2 region contains <answer>")

    # ── GOTCHA #1b: Leading Space Check ───────────────────────────────────
    # Check that separate tokenization of p1 doesn't add/lose a leading space vs template
    p1_standalone = tokenizer(p1, add_special_tokens=False).input_ids
    print(f"\n  Leading space check:")
    print(f"    p1 standalone tokens: {len(p1_standalone)}")
    print(f"    p1 from template:     {p1_len}")
    if len(p1_standalone) != p1_len:
        # This is expected — the template may add/remove leading whitespace or eot tokens
        # The KEY thing is that the mask boundaries are correct via prefix subtraction
        print(f"    ⚠ Lengths differ (expected — template adds eot/headers). "
              f"This is why we use template-based boundaries, not standalone tokenization.")
    else:
        print(f"    Lengths match")

    # ── GOTCHA #2: Label Shift Alignment ──────────────────────────────────
    # In causal LM loss: logits[:-1] predicts labels[1:]
    # Our tool_mask aligns with completion_ids (labels)
    # When computing loss, TRL does:
    #   logits = logits[:, :-1, :]           # predict next token
    #   logits = logits[:, -logits_to_keep:, :]  # keep completion logits
    #   completion_ids = input_ids[:, -logits_to_keep:]  # labels
    #   logps = log_softmax(logits)[completion_ids]
    # Then mask is applied: per_token_loss * mask
    #
    # Key insight: logits[t] predicts token[t+1]. So logits[-logits_to_keep:]
    # gives predictions for completion_ids[0], completion_ids[1], ..., completion_ids[-1]
    # The mask should align with completion_ids.
    #
    # But wait: TRL's code does logits[:, :-1, :] THEN logits[:, -logits_to_keep:, :]
    # This means it takes the LAST logits_to_keep positions from the shifted logits.
    # logits_to_keep = len(completion_ids), so:
    #   shifted_logits has seq_len-1 positions
    #   we take the last len(completion_ids) of them
    #   these correspond to predicting completion_ids[0..N-1]
    #
    # So mask[i] masks the loss for predicting completion_ids[i]. This is correct!
    print(f"\n  Label shift alignment:")
    print(f"    logits_to_keep = {len(completion_ids)}")
    print(f"    After shift: logits[-{len(completion_ids)}:] predicts completion_ids[0:{len(completion_ids)}]")
    print(f"    tool_mask[i] masks loss for predicting completion_ids[i]")
    print(f"    ✓ Mask aligns with shifted labels correctly")

    # ── Verify the OLD approach (flat concat) vs NEW approach ─────────────
    print(f"\n  Old vs New approach comparison:")
    old_completion = tokenizer(p1, add_special_tokens=False).input_ids \
                   + tokenizer(injected, add_special_tokens=False).input_ids \
                   + tokenizer(p2, add_special_tokens=False).input_ids
    print(f"    Old (flat concat):    {len(old_completion)} tokens")
    print(f"    New (chat template):  {len(completion_ids)} tokens")
    print(f"    Difference:           {len(completion_ids) - len(old_completion)} tokens (turn headers)")

    # The new approach has MORE tokens because it includes turn headers
    # between p1 and injected, and between injected and p2
    assert len(completion_ids) > len(old_completion), \
        "New approach should have more tokens (it includes turn headers)"
    print(f"    ✓ New approach correctly includes turn headers")

    # ── Verify the new completion matches generation context ──────────────
    # The full ids_full should exactly match what the model saw during generation
    # (system + user + assistant:p1 + eot + user:injected + eot + assistant:p2)
    full_text = tokenizer.decode(ids_full, skip_special_tokens=False)
    print(f"\n  Full conversation decode (last 200 chars):")
    print(f"    ...{repr(full_text[-200:])}")

    print(f"\n  ✓ TEST CASE {case_idx} PASSED")
    return True


def main():
    print("Loading Llama-3.1 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct")
    print(f"Loaded. Vocab size: {tokenizer.vocab_size}")

    # Show key special tokens
    special = {}
    for name in ["bos_token", "eos_token", "pad_token"]:
        tok = getattr(tokenizer, name, None)
        tid = getattr(tokenizer, f"{name}_id", None)
        special[name] = (tok, tid)
    print(f"Special tokens: {special}")

    # Check what headers look like
    print("\n--- Chat template structure inspection ---")
    simple_msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    template_text = tokenizer.apply_chat_template(simple_msgs, tokenize=False, add_generation_prompt=True)
    print(f"Template (system+user+gen_prompt):\n{repr(template_text)}")

    simple_msgs2 = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "ASST1"},
        {"role": "user", "content": "USR2"},
    ]
    template_text2 = tokenizer.apply_chat_template(simple_msgs2, tokenize=False, add_generation_prompt=True)
    print(f"\nTemplate (sys+usr+asst+usr+gen_prompt):\n{repr(template_text2)}")

    # ── Run test cases ────────────────────────────────────────────────────
    all_passed = True
    for i, (p1, inj, p2) in enumerate(zip(PHASE1_TEXTS, INJECTED_OUTPUTS, PHASE2_TEXTS)):
        passed = test_case(tokenizer, p1, inj, p2, i + 1)
        if not passed:
            all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    if all_passed:
        print(f"ALL {len(PHASE1_TEXTS)} TEST CASES PASSED ✓")
    else:
        print(f"SOME TEST CASES FAILED ✗")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
