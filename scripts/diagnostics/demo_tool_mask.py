"""
Demo: How completion_ids and tool_mask are computed in _generate().

This script simulates the logic from grpo_single_phase/train_trl.py
(AudioTRLGRPOTrainer._generate) that builds:
  - completion_ids : the token IDs the model is scored on (everything after the prompt)
  - tool_mask      : a 0/1 mask over completion_ids where
                       1 = model-generated token (counts toward the GRPO loss)
                       0 = injected tool output   (masked out of the loss)

We walk through 3 sample completions:
  (A) The model called a tool  → mask has a 0-region for the injected output
  (B) The model answered directly (no tool) → mask is all 1s
  (C) Tool call with longer reasoning       → same logic, different lengths

────────────────────────────────────────────────────────────
WHY CAN'T WE JUST NAIVELY CONCATENATE TOKENS?

In Llama-3 (and similar chat models), apply_chat_template inserts
special turn-boundary tokens between messages:
  <|eot_id|>                                  ← end of the assistant's turn
  <|start_header_id|>user<|end_header_id|>    ← start of user's turn
  ...etc...

If we just tokenized phase1, injected_output, phase2 separately and
concatenated the IDs, we'd MISS these boundary tokens. The resulting
sequence wouldn't match what the model actually saw during generation,
so the log-probabilities would be wrong. By using apply_chat_template
at progressive stages and taking length differences, we get the exact
token sequence including all boundary tokens.
────────────────────────────────────────────────────────────

Uses the real DeSTA/Llama-3 tokenizer.
Run with: conda activate slm && python demo_tool_mask.py
"""

from transformers import AutoTokenizer

# ═══════════════════════════════════════════════════════════════════════
# Load real tokenizer — same Llama-3.1 tokenizer DeSTA uses internally
# ═══════════════════════════════════════════════════════════════════════
# DeSTA loads its tokenizer from config.llm_model_id which points to:
TOKENIZER_NAME = "DeSTA-ntu/Llama-3.1-8B-Instruct"
print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}\n")


# ═══════════════════════════════════════════════════════════════════════
# Helper: pretty-print token IDs with their text and mask
# ═══════════════════════════════════════════════════════════════════════

def show_tokens(label, ids, mask=None):
    """Print token IDs alongside their decoded text and optional mask values."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Total tokens: {len(ids)}")
    if mask:
        ones  = sum(m for m in mask)
        zeros = len(mask) - ones
        print(f"  Mask: {ones} trainable (1), {zeros} masked-out (0)")
    print(f"{'='*80}")
    print(f"  {'idx':>4s}  {'id':>6s}  {'token':<45s}  {'mask':>4s}")
    print(f"  {'----':>4s}  {'------':>6s}  {'-----':<45s}  {'----':>4s}")
    for i, tid in enumerate(ids):
        token_str = repr(tokenizer.decode([tid]))
        m_str = str(mask[i]) if mask else ""
        flag = "  <<<< MASKED (injected)" if mask and mask[i] == 0 else ""
        print(f"  [{i:3d}]  {tid:6d}  {token_str:<45s}  {m_str:>4s}{flag}")


# ═══════════════════════════════════════════════════════════════════════
# SAMPLE DATA — mimicking what generate_completions() returns
# ═══════════════════════════════════════════════════════════════════════

system_prompt = "You are an audio analysis assistant."
user_prompt   = "How many speakers are in the audio?\nOptions: 1, 2, 3"

# Each "comp" dict mirrors what DeSTA25GRPOModel.generate_completions returns.
# phase1 = model's first generation (thinking + optional tool call)
# injected_output = tool result inserted by environment (NOT model-generated)
# phase2 = model's second generation after seeing the tool result

samples = [
    # ── Sample A: Model called a tool ────────────────────────────────
    {
        "name": "Sample A -- tool call (speaker_diarization)",
        "called_tools": True,
        "phase1": (
            '<think>\nI need to count the speakers precisely.\n</think>\n'
            '<tool>\n[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]\n</tool>'
        ),
        "injected_output": (
            '<tool_output>\n[{"speaker_diarization": {"speakers": ["Speaker 1", "Speaker 2"]}}]\n</tool_output>'
        ),
        "phase2": (
            '<think>\nThe tool found 2 speakers.\n</think>\n'
            '<answer>2</answer>'
        ),
    },
    # ── Sample B: Direct answer, no tool ─────────────────────────────
    {
        "name": "Sample B -- direct answer (no tool)",
        "called_tools": False,
        "phase1": (
            '<think>\nI can clearly hear two distinct voices.\n</think>\n'
            '<answer>2</answer>'
        ),
    },
    # ── Sample C: Tool call with longer reasoning ────────────────────
    {
        "name": "Sample C -- tool call (emotion_recognition), longer",
        "called_tools": True,
        "phase1": (
            '<think>\nThe question asks about emotion. Let me use emotion_recognition '
            'to get a precise classification rather than guessing from tone alone.\n</think>\n'
            '<tool>\n[{"function": "emotion_recognition", "parameters": {"audio_path": "<audio>"}}]\n</tool>'
        ),
        "injected_output": (
            '<tool_output>\n[{"emotion_recognition": {"emotion": "happy", "confidence": 0.92}}]\n</tool_output>'
        ),
        "phase2": (
            '<think>\nThe tool says the emotion is happy with 92% confidence. '
            'That matches the upbeat tone I heard.\n</think>\n'
            '<answer>happy</answer>'
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════
# CORE LOGIC — replicated from train_trl.py _generate()
# ═══════════════════════════════════════════════════════════════════════

for comp in samples:
    print(f"\n\n{'#'*72}")
    print(f"#  {comp['name']}")
    print(f"{'#'*72}")

    if comp["called_tools"]:
        # ──────────────────────────────────────────────────────────────
        # TOOL-CALL PATH
        #
        # The model produced two generation phases:
        #   phase1: thinking + tool call         (model-generated)
        #   injected_output: tool result         (NOT model-generated)
        #   phase2: thinking + final answer      (model-generated)
        #
        # We need completion_ids = everything after the prompt, including
        # the injected tool output in the middle.
        # But we ONLY want to train on what the model actually generated
        # (phase1 & phase2), NOT the injected tool output.
        #
        # Strategy: use apply_chat_template at 4 progressive stages
        # and take length differences to find segment boundaries.
        # ──────────────────────────────────────────────────────────────

        # Base messages = the prompt (system + user)
        msgs_base = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        # ── Stage 1: Prompt only (with generation prompt) ─────────
        # This is the input the model sees before it generates anything.
        # Ends with: <|start_header_id|>assistant<|end_header_id|>\n\n
        ids_prompt_text = tokenizer.apply_chat_template(
            msgs_base, tokenize=True, add_generation_prompt=True
        )

        # ── Stage 2: Prompt + phase1 (assistant's first turn) ─────
        # Phase1 is the model's first generation (thinking + tool call).
        # add_generation_prompt=False because this is a completed turn
        # (it ends with <|eot_id|>).
        ids_through_p1 = tokenizer.apply_chat_template(
            msgs_base + [{"role": "assistant", "content": comp["phase1"]}],
            tokenize=True, add_generation_prompt=False
        )

        # ── Stage 3: Prompt + phase1 + injected tool output ──────
        # The injected tool result arrives as a new user message.
        # add_generation_prompt=True because the model will generate phase2.
        # This adds the assistant header for the next generation turn.
        ids_through_inj = tokenizer.apply_chat_template(
            msgs_base + [
                {"role": "assistant", "content": comp["phase1"]},
                {"role": "user",      "content": comp["injected_output"]},
            ],
            tokenize=True, add_generation_prompt=True
        )

        # ── Stage 4: Full conversation ────────────────────────────
        # Everything: prompt + phase1 + injected + phase2
        ids_full = tokenizer.apply_chat_template(
            msgs_base + [
                {"role": "assistant", "content": comp["phase1"]},
                {"role": "user",      "content": comp["injected_output"]},
                {"role": "assistant", "content": comp["phase2"]},
            ],
            tokenize=True, add_generation_prompt=False
        )

        # ── Slice: completion = everything after the prompt ─────────
        # This is what we compute log-probabilities on.
        completion_ids = ids_full[len(ids_prompt_text):]

        # ── Compute segment lengths by subtraction ──────────────────
        #
        # Visually, the stages look like this (growing left to right):
        #
        #   ids_prompt_text:  [PROMPT ........................ asst_header]
        #   ids_through_p1:   [PROMPT ... asst_header  PHASE1  <|eot_id|>]
        #   ids_through_inj:  [PROMPT ... PHASE1  <|eot_id|>  user_header  INJECTED  <|eot_id|>  asst_header]
        #   ids_full:         [PROMPT ... PHASE1  <|eot_id|>  ... INJECTED ... <|eot_id|>  asst_header  PHASE2  <|eot_id|>]
        #
        #   p1_len     = ids_through_p1  - ids_prompt_text    <- phase1 content + <|eot_id|>
        #   masked_len = ids_through_inj - ids_through_p1     <- user turn headers + tool output + asst header
        #   p2_len     = ids_full        - ids_through_inj    <- phase2 content + <|eot_id|>
        #
        p1_len     = len(ids_through_p1)  - len(ids_prompt_text)
        masked_len = len(ids_through_inj) - len(ids_through_p1)
        p2_len     = len(ids_full)        - len(ids_through_inj)

        # ── Build the mask ──────────────────────────────────────────
        # 1 = trainable (model-generated), 0 = masked (injected + turn boundaries)
        tool_mask = [1] * p1_len + [0] * masked_len + [1] * p2_len

        # ── Sanity check ────────────────────────────────────────────
        assert len(tool_mask) == len(completion_ids), (
            f"Mask length {len(tool_mask)} != completion length {len(completion_ids)}"
        )

        # ── Print the 4-stage breakdown ─────────────────────────────
        print(f"\n  Token counts at each progressive stage:")
        print(f"  +-----------------------------------------------------+")
        print(f"  | Stage 1: ids_prompt_text  = {len(ids_prompt_text):4d} tokens           |")
        print(f"  |          (system + user + assistant gen header)      |")
        print(f"  | Stage 2: ids_through_p1   = {len(ids_through_p1):4d} tokens           |")
        print(f"  |          (+ phase1 assistant turn + eot)            |")
        print(f"  | Stage 3: ids_through_inj  = {len(ids_through_inj):4d} tokens           |")
        print(f"  |          (+ injected tool output as user turn)      |")
        print(f"  | Stage 4: ids_full         = {len(ids_full):4d} tokens           |")
        print(f"  |          (+ phase2 assistant turn + eot)            |")
        print(f"  +-----------------------------------------------------+")
        print()
        print(f"  Derived segment lengths (by subtraction):")
        print(f"    p1_len     = {len(ids_through_p1):3d} - {len(ids_prompt_text):3d} = {p1_len:3d}  [mask=1] TRAINABLE")
        print(f"    masked_len = {len(ids_through_inj):3d} - {len(ids_through_p1):3d} = {masked_len:3d}  [mask=0] NOT trained (injected)")
        print(f"    p2_len     = {len(ids_full):3d} - {len(ids_through_inj):3d} = {p2_len:3d}  [mask=1] TRAINABLE")
        print(f"    total      = {p1_len} + {masked_len} + {p2_len} = {len(completion_ids)}")

        # Show compact mask visualization
        print(f"\n  Mask visualization (each char = 1 token):")
        mask_str = "".join("#" if m == 1 else "." for m in tool_mask)
        print(f"    {mask_str}")
        print(f"    # = trainable (model-generated)   . = masked (injected)")

        show_tokens("Full Completion Tokens + Tool Mask", completion_ids, tool_mask)

    else:
        # ──────────────────────────────────────────────────────────────
        # NO-TOOL PATH
        #
        # The model answered directly in a single turn (phase1 only).
        # There's no injected output, so every token is model-generated
        # and the mask is all 1s.
        # ──────────────────────────────────────────────────────────────

        completion_ids = tokenizer(comp["phase1"], return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        tool_mask = [1] * len(completion_ids)

        print(f"\n  No tool called -> single-phase response, mask is all 1s")
        print(f"  Completion length: {len(completion_ids)} tokens (all trainable)")

        # Show compact mask visualization
        print(f"\n  Mask visualization (each char = 1 token):")
        mask_str = "".join("#" for _ in tool_mask)
        print(f"    {mask_str}")
        print(f"    All # = every token is trainable")

        show_tokens("Full Completion Tokens + Tool Mask", completion_ids, tool_mask)


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY — WHY THE MASK MATTERS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*72}")
print("SUMMARY -- WHY THE TOOL MASK MATTERS FOR GRPO")
print("=" * 72)
print("""
In GRPO, the training loss for each completion token is approximately:

    per_token_loss = advantage * log_prob_ratio * tool_mask

Where:
  - advantage      = reward - baseline  (from the G completions for this prompt)
  - log_prob_ratio = log pi(token) - log pi_ref(token)
  - tool_mask      = 1 for model-generated, 0 for injected

The injected tool output was NOT generated by the model -- it was
deterministically inserted by the environment after the tool call.

WITHOUT the mask (mask=1 everywhere):
  X  We'd reward/penalize tokens the model never chose
  X  Log-probabilities on forced tokens are meaningless
  X  The policy gradient would be corrupted with noise

WITH the mask (mask=0 for injected region):
  OK  Only the model's own decisions contribute to the gradient
  OK  Phase 1 (reasoning + tool selection)  -> rewarded/penalized
  OK  Phase 2 (interpretation + answer)     -> rewarded/penalized
  OK  Injected tool output passes through without affecting gradients

Note: the masked region includes not just the tool output text but
also the chat-template turn-boundary tokens (<|eot_id|>, turn headers)
that surround it. These are all part of the "injected" context.
""")
