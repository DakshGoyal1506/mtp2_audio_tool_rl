"""
Test: Does BPE boundary mismatch cause tool_mask misalignment?

Simulates the exact tokenization logic from train_trl.py:
  completion_ids = tokenizer(full_text)          
  p1_len         = len(tokenizer(phase1))
  p1_inj_len     = len(tokenizer(phase1 + injected))
  inj_len        = p1_inj_len - p1_len
  mask           = [1]*p1_len + [0]*inj_len + [1]*(len(completion_ids) - p1_inj_len)
"""

import os, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("DeSTA-ntu/Llama-3.1-8B-Instruct")

def make_injected(tool_result):
    result_str = json.dumps(tool_result, indent=2) if not isinstance(tool_result, str) else tool_result
    return f"\n<tool_output>\n{result_str}\n</tool_output>\n"

test_cases = [
    {
        "phase1": '<think>\nThe question asks about the content of the speech. I need to transcribe it.\n</think>\n<tool>\n[{"function": "speech_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
        "injected": make_injected({"text": "Hello world, this is a test."}),
        "phase2": '\n<think>\nThe transcription says hello world.\n</think>\n<answer>Emergency broadcast test</answer>',
    },
    {
        "phase1": '<think>\nI need to determine the emotion.\n</think>\n<tool>\n[{"function": "emotion_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
        "injected": make_injected({"emotion": "happy", "confidence": 0.95}),
        "phase2": '\n<think>\nThe emotion is happy.\n</think>\n<answer>Happy</answer>',
    },
    {   # BPE tricky: no newline before injected
        "phase1": '<think>\nAnalyzing</think>\n<tool>\n[{"function": "get_audio_features", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
        "injected": make_injected({"duration": 5.2}),
        "phase2": '<think>\nDuration is 5.2s.\n</think>\n<answer>5 seconds</answer>',
    },
    {   # Longer tool output  
        "phase1": '<think>\nCounting speakers.\n</think>\n<tool>\n[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]\n</tool>',
        "injected": make_injected([{"speaker": "S0", "start": 0.0, "end": 2.5}, {"speaker": "S1", "start": 2.5, "end": 5.0}]),
        "phase2": '\n<think>\nTwo speakers.\n</think>\n<answer>2 speakers</answer>',
    },
    {   # Phase1 ends with extra newline  
        "phase1": '<think>\nNeed chord info\n</think>\n<tool>\n[{"function": "chord_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>\n',
        "injected": make_injected({"chords": ["C", "Am", "F", "G"]}),
        "phase2": '<think>\nPop progression.\n</think>\n<answer>C major</answer>',
    },
]

print("=" * 70)
print("BPE BOUNDARY MISMATCH TEST")
print("=" * 70)

mismatches = 0
for i, case in enumerate(test_cases):
    p1, inj, p2 = case["phase1"], case["injected"], case["phase2"]
    full = p1 + inj + p2

    # === EXACT logic from train_trl.py lines 367-375 ===
    comp_ids = tokenizer(full, add_special_tokens=False).input_ids
    p1_len = len(tokenizer(p1, add_special_tokens=False).input_ids)
    p1_inj_len = len(tokenizer(p1 + inj, add_special_tokens=False).input_ids)
    inj_len = p1_inj_len - p1_len
    remaining = len(comp_ids) - p1_inj_len

    # === Ground truth: tokenize separately ===
    gt_p1 = tokenizer(p1, add_special_tokens=False).input_ids
    gt_inj = tokenizer(inj, add_special_tokens=False).input_ids
    gt_p2 = tokenizer(p2, add_special_tokens=False).input_ids
    
    # Check if tokens at boundaries line up
    full_p1_region = comp_ids[:p1_len]
    full_inj_region = comp_ids[p1_len:p1_inj_len]
    
    p1_match = (full_p1_region == gt_p1)
    inj_match = (full_inj_region == gt_inj)
    len_ok = (p1_len + inj_len + remaining == len(comp_ids))

    has_issue = not p1_match or not inj_match or not len_ok
    if has_issue:
        mismatches += 1

    print(f"\nCase {i+1}: {'!! MISMATCH !!' if has_issue else 'OK'}")
    print(f"  full={len(comp_ids)}  p1={p1_len} inj={inj_len} rem={remaining}  sum={p1_len+inj_len+remaining}")
    print(f"  ground truth: p1={len(gt_p1)} inj={len(gt_inj)} p2={len(gt_p2)} sum={len(gt_p1)+len(gt_inj)+len(gt_p2)}")
    print(f"  p1 tokens match: {p1_match}  inj tokens match: {inj_match}")
    
    if has_issue:
        # Where do they diverge?
        for j in range(min(len(full_p1_region), len(gt_p1))):
            if full_p1_region[j] != gt_p1[j]:
                print(f"  p1 diverges at pos {j}: full={tokenizer.decode([full_p1_region[j]])!r} vs gt={tokenizer.decode([gt_p1[j]])!r}")
                break
        for j in range(min(len(full_inj_region), len(gt_inj))):
            if full_inj_region[j] != gt_inj[j]:
                print(f"  inj diverges at pos {j}: full={tokenizer.decode([full_inj_region[j]])!r} vs gt={tokenizer.decode([gt_inj[j]])!r}")
                break

print(f"\n{'=' * 70}")
print(f"RESULT: {mismatches}/{len(test_cases)} mismatches")
if mismatches == 0:
    print("Hypothesis NOT confirmed for Llama-3 tokenizer with these patterns.")
else:
    print("BUG CONFIRMED — mask boundaries shift due to BPE merging.")
print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX VERIFICATION: tokenize each piece separately, concatenate token IDs
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FIX VERIFICATION — separate tokenization + concat")
print("=" * 70)

fix_failures = 0
for i, case in enumerate(test_cases):
    p1, inj, p2 = case["phase1"], case["injected"], case["phase2"]

    # === NEW (fixed) approach: tokenize pieces separately, concat IDs ===
    p1_ids = tokenizer(p1, add_special_tokens=False).input_ids
    inj_ids = tokenizer(inj, add_special_tokens=False).input_ids
    p2_ids = tokenizer(p2, add_special_tokens=False).input_ids

    completion_ids = p1_ids + inj_ids + p2_ids
    tool_mask = [1] * len(p1_ids) + [0] * len(inj_ids) + [1] * len(p2_ids)

    # Verify: mask length == completion length
    len_ok = len(tool_mask) == len(completion_ids)
    # Verify: decode each masked region back to original text
    decoded_p1 = tokenizer.decode(completion_ids[:len(p1_ids)])
    decoded_inj = tokenizer.decode(completion_ids[len(p1_ids):len(p1_ids)+len(inj_ids)])
    decoded_p2 = tokenizer.decode(completion_ids[len(p1_ids)+len(inj_ids):])
    p1_ok = decoded_p1 == p1
    inj_ok = decoded_inj == inj
    p2_ok = decoded_p2 == p2
    # Verify: mask=0 region maps exactly to injected text
    masked_tokens = [completion_ids[j] for j in range(len(tool_mask)) if tool_mask[j] == 0]
    mask0_ok = masked_tokens == inj_ids
    # Verify: full decoded text is the same as the original
    full_ok = tokenizer.decode(completion_ids) == (p1 + inj + p2)

    all_ok = len_ok and p1_ok and inj_ok and p2_ok and mask0_ok and full_ok
    if not all_ok:
        fix_failures += 1

    print(f"\nCase {i+1}: {'FAIL' if not all_ok else 'PASS'}")
    print(f"  lens:        mask={len(tool_mask)} comp={len(completion_ids)} match={len_ok}")
    print(f"  decode p1:   {p1_ok}  inj: {inj_ok}  p2: {p2_ok}")
    print(f"  mask0==inj:  {mask0_ok}")
    print(f"  full decode: {full_ok}")
    if not full_ok:
        orig = p1 + inj + p2
        recon = tokenizer.decode(completion_ids)
        # Find first difference
        for j in range(min(len(orig), len(recon))):
            if orig[j] != recon[j]:
                print(f"  first diff at char {j}: orig={orig[j:j+20]!r} vs recon={recon[j:j+20]!r}")
                break

print(f"\n{'=' * 70}")
print(f"FIX RESULT: {fix_failures}/{len(test_cases)} failures")
if fix_failures == 0:
    print("FIX VERIFIED: separate tokenization produces exact mask boundaries.")
else:
    print("FIX HAS ISSUES — needs further investigation.")
print(f"{'=' * 70}")
