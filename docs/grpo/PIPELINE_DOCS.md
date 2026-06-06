# GRPO Training & Evaluation Pipeline — Full Documentation

> **Group Relative Policy Optimization (GRPO)** for DeSTA2.5-Audio tool-use on the MMAU benchmark.
>
> Reference: [DeepSeekMath GRPO (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300)

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architecture & Loading](#4-model-architecture--loading)
5. [Two-Phase Generation (Tool-Augmented Rollouts)](#5-two-phase-generation-tool-augmented-rollouts)
6. [Prompt Templates](#6-prompt-templates)
7. [Reward System](#7-reward-system)
8. [GRPO Training Algorithm](#8-grpo-training-algorithm)
9. [Training Loop & Trainer](#9-training-loop--trainer)
10. [TRL-Based Training Pipeline (Alternative)](#10-trl-based-training-pipeline-alternative)
11. [Evaluation Pipeline](#11-evaluation-pipeline)
12. [Inference Pipeline](#12-inference-pipeline)
13. [Configuration Reference](#13-configuration-reference)
14. [Distributed Training & Infrastructure](#14-distributed-training--infrastructure)
15. [Checkpoint Management](#15-checkpoint-management)
16. [Logging & Monitoring](#16-logging--monitoring)

---

## 1. High-Level Overview

This codebase fine-tunes audio-language models (DeSTA2.5-Audio, Qwen2.5-Omni) to answer multiple-choice audio questions using **tool-augmented reasoning**. The model learns when and how to call audio analysis tools (speech recognition, speaker diarization, emotion detection, etc.) through GRPO — a reinforcement learning algorithm that optimizes the policy by comparing groups of sampled completions against each other.

### Core Training Loop (Simplified)

```
For each question in the batch:
  1. Sample G completions from the policy model
  2. For completions that called a tool → inject cached tool outputs → generate phase 2
  3. Score each completion: format_reward + correctness_reward
  4. Normalize advantages within the group of G completions
  5. Compute policy and reference log-probs
  6. GRPO loss = -advantage * log_pi(completion) + beta * KL(pi || pi_ref)
  7. Backward pass (one completion at a time for memory efficiency)
  8. Optimizer step with gradient clipping
```

### Two Entry Points

| Entry Point | File | Framework | Description |
|---|---|---|---|
| **Custom GRPO** | `train_grpo.py` | Accelerate + custom trainer | Hand-written GRPO loop with full control over two-phase generation, per-completion backward, DDP padding |
| **TRL GRPO** | `train_trl.py` | HuggingFace TRL `GRPOTrainer` | Overrides TRL's built-in GRPO with DeSTA-specific `_generate` and `_get_per_token_logps_and_entropies` |

---

## 2. Directory Structure

```
grpo/
├── __init__.py
├── configs/
│   ├── default.yaml          # Conservative config (G=16, beta=0.01, lr=1e-5)
│   ├── optimized.yaml        # Tuned config (G=16, beta=0.5, lr=5e-5, gradient_checkpointing)
│   └── qwen_omni.yaml        # Qwen2.5-Omni-7B config (G=8, lr=2e-6)
├── splits/
│   ├── train.json            # Stratified training split
│   ├── eval.json             # Stratified eval split
│   └── test.json             # Held-out test split
├── dataset.py                # Dataset splitting & PyTorch Dataset class
├── model_wrapper.py          # DeSTA25GRPOModel — generation + log-prob computation
├── modeling_grpo.py          # GRPODeSTA25AudioModel — Whisper-decoder-free model class
├── prompts.py                # System/user prompt templates for DeSTA and Qwen
├── rewards.py                # format_reward, correctness_reward, compute_reward
├── rewards_trl.py            # TRL-compatible reward function wrapper
├── trainer.py                # GRPOTrainer — custom training loop + eval + checkpoint
├── train_grpo.py             # Main entry point (custom trainer)
├── train_trl.py              # Main entry point (TRL-based trainer)
├── trl_grpo_trainer_orig.py  # Reference copy of TRL's GRPOTrainer source
├── run_grpo2.sh              # SLURM batch script for custom trainer
├── run_trl.sh                # SLURM batch script for TRL trainer
└── run_debug.sh              # Single-GPU debug launch script
```

---

## 3. Data Pipeline

### 3.1 Source Data

The dataset is **MMAU (Massive Multi-task Audio Understanding)** — a multiple-choice audio Q&A benchmark. Each item contains:

```json
{
  "id": "uuid",
  "task": "sound | speech | music",
  "question": "What instrument is playing?",
  "choices": ["Piano", "Guitar", "Violin", "Drums"],
  "answer": "Piano",
  "audio_id": "test-mini-audios/002ebba8.wav",
  "difficulty": "easy | medium | hard",
  "sub-category": "instrument_identification",
  "cached_tool_outputs": {
    "speech_recognition": {"text": "..."},
    "sound_classification": {...},
    ...
  }
}
```

**Key insight:** Tool outputs are **pre-cached** at dataset build time. During training, when the model calls a tool, the cached result is injected — no actual tool execution happens. This makes training deterministic and fast.

### 3.2 Dataset Splitting (`dataset.py`)

**`split_mmau_dataset()`** creates a stratified split by task category:

| Split | Default Ratio | Purpose |
|---|---|---|
| Train | 10-30% | GRPO optimization |
| Eval | 10% | Validation / best-checkpoint selection |
| Test | 60-80% | Held-out evaluation |

- **Stratified by task** (sound, speech, music) so each split has balanced representation
- **Deterministic** with `seed=42` — splits are saved to `splits/` directory and reused across runs
- **Idempotent** — if split files already exist, they're loaded directly

### 3.3 `GRPOAudioDataset` (PyTorch Dataset)

Each `__getitem__` returns:

| Field | Type | Description |
|---|---|---|
| `audio_path` | str | Absolute path to the .wav file |
| `precomputed_embed` | str or None | Path to `{uuid}_embed.pt` containing pre-extracted Whisper+QFormer embeddings |
| `question` | str | Question text |
| `choices` | list[str] | Answer options |
| `gold_answer` | str | Correct answer text |
| `cached_tool_outputs` | dict | Pre-computed tool results keyed by tool name |
| `task` | str | `sound`, `speech`, or `music` |

### 3.4 Precomputed Embeddings

To save GPU memory and compute, audio embeddings can be pre-extracted:

- Stored as `.pt` files in `precomputed_embeds/` directory
- Each file contains either:
  - A flat tensor (legacy): merged audio + transcription embeddings
  - A dict (new format): `{"qformer": tensor, "vad": bool}` — QFormer output + VAD flag
- When available, the Whisper encoder and QFormer are **not loaded at all** (`skip_perception=True`), saving ~1.5 GB VRAM

---

## 4. Model Architecture & Loading

### 4.1 `GRPODeSTA25AudioModel` (`modeling_grpo.py`)

A subclass of `DeSTA25AudioModel` optimized for GRPO training:

| Optimization | Savings | How |
|---|---|---|
| Whisper decoder removal | ~500 MB VRAM | Deleted after load; cached transcripts used instead |
| VAD (silero-vad) removal | Startup time + no internet required | Never loaded; transcripts come from cache |
| Skip entire perception | ~1.5 GB VRAM | When `skip_perception=True`, Whisper encoder + QFormer are replaced with a no-op |
| Flash Attention | ~2x faster generation | Auto-detects `flash_attention_2` or falls back to PyTorch SDPA |

**Loading sequence:**
1. `skip_perception` class flag is set based on whether precomputed embeddings exist
2. `from_pretrained()` monkey-patches `AutoModelForCausalLM.from_pretrained` to inject `attn_implementation`
3. If `skip_perception=True`, `WhisperPerception.__init__` is replaced with a no-op
4. After loading, `_setup_generation()` initializes the tokenizer/processor

### 4.2 LoRA Configuration

LoRA adapters are attached to the LLM backbone (LlamaForCausalLM) via PEFT:

```yaml
lora_rank: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: [q_proj, v_proj, k_proj]
```

**Model hierarchy after LoRA:**
```
model._peft_model       → PeftModel (has save_pretrained())
model._lora_base_model  → LoraModel (has disable/enable_adapter_layers())
model.llm_model         → LlamaForCausalLM with LoRA layers grafted in
```

The speech encoder (Whisper) and QFormer connector are **frozen** — only LoRA weights on the LLM are trainable. Typical trainable params: ~0.1% of total.

### 4.3 Gradient Checkpointing

When enabled (`gradient_checkpointing: true`):
- Recomputes activations during backward instead of storing them
- Saves ~40% activation memory at ~30% extra compute cost
- Uses `use_reentrant=False` (required for LoRA/PEFT compatibility)
- Automatically disabled during generation (KV-cache needs it off), re-enabled for backward

### 4.4 Multi-Model Support

The codebase supports two model families:

| Model | Wrapper | Config |
|---|---|---|
| DeSTA2.5-Audio (Llama 3.1 8B backbone) | `DeSTA25GRPOModel` | `default.yaml`, `optimized.yaml` |
| Qwen2.5-Omni-7B | `QwenOmniGRPOModel` | `qwen_omni.yaml` |

Model family is auto-detected by checking for "qwen" in `model_name` or `model_hf_name`.

---

## 5. Two-Phase Generation (Tool-Augmented Rollouts)

The core innovation: completions can involve **tool calls**, handled as a two-phase generation process.

### Phase 1: Initial Generation

The model receives the audio + question and generates one of:

**Format A (Tool Call):**
```xml
<think>
I need to identify the speakers. Speaker diarization will give me an exact count.
</think>
<tool>
[{"function": "speaker_diarization", "parameters": {"audio_path": "<audio>"}}]
</tool>
```

**Format B (Direct Answer):**
```xml
<think>
I can clearly hear a piano playing. No tool needed.
</think>
<answer>Piano</answer>
```

### Phase 2: Tool Output Injection + Follow-Up

For Format A completions:
1. The tool name is extracted from the `<tool>` block via `_parse_tool_call()`
2. The cached tool output is looked up and formatted as `<tool_output>...</tool_output>`
3. A **new** follow-up prompt is constructed with the tool results
4. The model generates a second response with `<think>` reasoning + `<answer>`

### Batched Generation

Generation is optimized with batching:
- **Phase 1**: All G completions for one question are generated in a single `model.generate()` call using `num_return_sequences=G`
- **Phase 2**: Tool-calling completions are grouped by tool name and batch-generated per group
- Audio encoding is done **once per unique audio**, then reused across all G completions

### Final Completion Structure

```python
{
    "text":             "full concatenated text",
    "phase1":           "model output from phase 1",
    "phase2":           "model output from phase 2 (empty if no tool)",
    "called_tools":     True/False,
    "tool_name":        "speaker_diarization" or None,
    "injected_output":  "<tool_output>...</tool_output>" or None,
}
```

---

## 6. Prompt Templates

Defined in `prompts.py`. Two sets of builders exist — one for DeSTA, one for Qwen.

### 6.1 Initial Prompt (`build_grpo_initial_prompt`)

**System prompt** contains:
- Role definition ("audio analysis assistant")
- Complete tool descriptions (10 available tools)
- Exact response format specification (Format A and Format B)
- Critical rules (JSON format, exact option text, no text outside tags)
- Examples of both formats

**User message** contains:
- The question
- Answer options (bulleted list)
- Instruction to listen and respond

### 6.2 Follow-Up Prompt (`build_grpo_followup_prompt`)

After tool output injection:

**System prompt:** Concise instruction to analyze tool results and provide final answer with `<think>` + `<answer>` tags.

**User message:** Question + choices + tool results + "Now provide your final `<think>` and `<answer>`."

### 6.3 Available Tools

| Tool | Description |
|---|---|
| `speech_recognition` | Transcribe speech to text |
| `speaker_diarization` | Segment and label speakers |
| `emotion_recognition` | Detect emotions in speech |
| `stressed_analysis` | Analyze stress patterns |
| `get_audio_features` | Extract acoustic features (pitch, energy, etc.) |
| `sound_classification` | Classify environmental sounds |
| `sound_duration_analysis` | Analyze sound/silence durations |
| `speech_to_noise_ratio` | Compute SNR |
| `chord_recognition` | Identify musical chords |
| `genre_analysis` | Classify music genre |

---

## 7. Reward System

### 7.1 Format Reward (`rewards.py :: format_reward`)

A **penalty-based** system (range: [-0.8, 0.0], mapped to [0.2, 1.0] via `1.0 + penalty`):

| Condition | Penalty |
|---|---|
| `<answer>` entirely absent | -0.5 |
| `<answer>` opened but `</answer>` missing | -0.2 |
| `<think>` entirely absent | -0.1 |
| `<tool>` opened but `</tool>` missing | -0.2 |
| All tags present and well-formed | 0.0 (perfect) |

### 7.2 Correctness Reward (`rewards.py :: correctness_reward`)

- **Primary**: Exact string match (case-insensitive, stripped) against gold answer → 0.0 or 1.0
- **Optional LLM Judge**: If `LLM_JUDGE_URL` and `LLM_JUDGE_MODEL` are set, queries an OpenAI-compatible endpoint for semantic correctness, then averages with exact match

Answer extraction: Parses `<answer>...</answer>` tags, handles both plain text and legacy JSON `{"answer": "..."}` format.

### 7.3 Combined Reward (Custom Trainer)

```python
total = format_weight * (1.0 + format_penalty) + correctness_weight * correctness_score + tool_bonus
```

- **Tool bonus**: +0.1 if the model called a tool and it was well-formed (tag_penalty > -0.2)
- Default weights: `format_weight=0.2`, `correctness_weight=0.8`

### 7.4 TRL Reward Function (`rewards_trl.py :: trl_reward_function`)

A modified reward for the TRL pipeline with four components:

| Component | Weight | Description |
|---|---|---|
| Format | 0.10 | Tag structure quality |
| Correctness | 0.70 | Exact answer match |
| Option Mention | 0.10 | Partial credit for mentioning correct option (cold-start helper) |
| Tool Bonus | 0.10 | Well-formed tool call bonus |

The option-mention score provides gradient signal during cold-start when the model hasn't learned to produce proper `<answer>` tags yet:
- 0.5 if the gold answer appears anywhere in the completion text
- 0.1 if any other option is mentioned
- 0.0 otherwise

---

## 8. GRPO Training Algorithm

### 8.1 Algorithm Overview

GRPO (Group Relative Policy Optimization) avoids the need for a separate critic/value model by computing advantages **within each group** of sampled completions for the same question.

For a batch of Q questions, each with G completions:

1. **Rollout**: Sample G completions per question from the policy
2. **Reward**: Score each completion with `compute_reward()`
3. **Advantage Normalization** (per-group):
   ```
   A_{q,g} = (r_{q,g} - mean(r_q)) / (std(r_q) + eps)
   ```
4. **Loss**:
   ```
   L = -1/n_terms * sum_{q,g} A_{q,g} * log_pi(o_{q,g} | x_q) + beta * KL(pi_theta || pi_ref)
   ```

### 8.2 KL Divergence

Uses the **Schulman KL approximator** (always ≥ 0):

```python
log_ratio = log_pi_policy - log_pi_ref
KL ≈ exp(-log_ratio) + log_ratio - 1.0    # clamped at max=10.0
```

This avoids negative KL divergence that can occur with the naive `log_pi_policy - log_pi_ref` estimator.

### 8.3 Log-Probability Computation

For **tool-calling completions**, log-probs are computed in two parts:

```python
# Phase 1: log_probs of phase1 text given audio context
policy_lp_p1, n1 = policy.compute_log_probs(messages_p1, comp["phase1"])

# Phase 2: log_probs of phase2 text given follow-up context
policy_lp_p2, n2 = policy.compute_log_probs(messages_p2, comp["phase2"])

# Weighted average by token count
policy_lp = (policy_lp_p1 * n1 + policy_lp_p2 * n2) / (n1 + n2)
```

The **injected `<tool_output>` tokens are excluded** from log-prob computation — the model is only trained on tokens it actually generated.

### 8.4 Memory-Efficient Backward

Instead of accumulating all G computation graphs in memory, backward is called **per-completion**:

```python
for each completion with non-zero advantage:
    step_loss = (-advantage * policy_lp + beta * kl) / n_terms
    accelerator.backward(step_loss)
    del step_loss  # free graph immediately
    torch.cuda.empty_cache()
```

This ensures only ONE computation graph lives in GPU memory at a time.

### 8.5 DDP Rank Synchronization

A critical correctness detail: different ranks may have different numbers of non-zero advantage terms. DeepSpeed's gradient accumulation uses NCCL all-reduce at fixed boundaries (every N backward calls). If ranks disagree on the count, NCCL sequence numbers mismatch → deadlock.

**Solution**: Before the backward loop, all ranks agree on the global maximum `n_terms` via `all_reduce(MAX)`. Shorter ranks pad with zero-gradient dummy backward calls that touch all trainable parameters.

---

## 9. Training Loop & Trainer

### 9.1 `GRPOTrainer` (`trainer.py`)

The custom trainer manages the full lifecycle:

#### Initialization
- Receives policy + reference model wrappers, accelerator, config
- Auto-selects prompt builders based on model family (DeSTA vs Qwen)
- Creates a dedicated rollout logger (writes to `{output_dir}/run_logs/{slurm_id}/rollout.log`)

#### `train()` Method

```
1. Create DataLoader (shuffle=True, collate_fn returns list of dicts)
2. Initialize WandB (offline mode) and SystemMonitor
3. Run baseline evaluation (unless --skip-baseline-eval)
4. For each epoch:
   a. For each batch:
      - grpo_step(batch)  →  rollout → reward → advantage → log_probs → loss → backward → optimizer.step()
      - Log metrics to console and WandB
      - Mid-step eval if eval_every_n_steps > 0
   b. End-of-epoch eval (if eval_every_n_steps == 0)
   c. Save checkpoint every save_every_n_epochs
5. Save final checkpoint
6. Stop system monitor, finish WandB
```

#### `grpo_step()` Method

Returns a metrics dict:

| Metric | Description |
|---|---|
| `loss` | Accumulated GRPO loss |
| `mean_reward` | Average total reward across all completions |
| `mean_format` | Average format reward |
| `mean_correctness` | Average correctness reward |
| `mean_kl` | Average KL divergence |
| `mean_advantage` | Average absolute advantage |
| `tool_call_fraction` | Fraction of completions that called a tool |
| `lr` | Current learning rate |
| `mean_policy_logprob` | Average policy log-prob |
| `mean_ref_logprob` | Average reference log-prob |

#### `evaluate()` Method

- Uses greedy decoding (`temperature=0.0`, `G=1`)
- Runs over the full eval dataset with `torch.no_grad()`
- Returns `mean_reward`, `mean_format`, `mean_correctness`

### 9.2 `train_grpo.py` (Custom Trainer Entry Point)

**Flow:**
1. Parse args (config path, dry-run flag, CLI overrides)
2. Load config from YAML, merge CLI overrides via OmegaConf
3. Build Accelerator (optionally with DeepSpeed ZeRO-2)
4. Split dataset into train/eval/test
5. Load policy model (with LoRA) + reference model (frozen, no LoRA)
6. Build optimizer (AdamW) + cosine scheduler with warmup
7. `accelerator.prepare(model, optimizer)`
8. Create `GRPOTrainer` and call `trainer.train()`

**Optimizer:**
- AdamW with configurable `learning_rate` and `weight_decay`
- Cosine schedule with linear warmup
- Gradient clipping via `max_grad_norm`

---

## 10. TRL-Based Training Pipeline (Alternative)

### 10.1 `train_trl.py`

Uses HuggingFace TRL's `GRPOTrainer` with heavy customization for DeSTA's audio-specific needs.

### 10.2 `AudioTRLGRPOTrainer` (Custom Subclass)

Overrides three critical TRL methods:

#### `_generate()`
- Unwraps the accelerator model to access DeSTA methods
- Disables gradient checkpointing during generation
- Groups prompts by unique audio path (encode audio once, generate multiple completions)
- Runs two-phase generation via `DeSTA25GRPOModel.generate_completions()`
- Tokenizes completions with a **tool mask** (1 = model token, 0 = injected tool output)
- Stores audio features (`batch_features`, `batch_transcription_ids`, `precomputed_embeds`, `raw_start_positions`) for the log-prob computation

#### `_get_per_token_logps_and_entropies()`
- Overrides TRL's default to pass audio features through DeSTA's forward pass
- Reconstructs `batch_start_positions` accounting for left-padding
- Calls DeSTA's custom forward with `batch_features`, `batch_transcription_ids`, `precomputed_embeds`

#### `log()`
- Enriches TRL's parquet output with `gold_answer`, `choices`, `id` columns
- Injects GRPO aggregate metrics
- Catches Rich rendering crashes from gibberish Unicode in completions

### 10.3 `GRPOConfig` Settings

```python
GRPOConfig(
    num_generations=G,           # completions per question
    max_completion_length=2048,
    beta=0.5,                    # KL coefficient
    temperature=1.1,
    gradient_checkpointing=True,
    log_completions=True,        # save completions to parquet
    num_completions_to_print=0,  # suppress Rich console output
)
```

### 10.4 TRL Reward Interface

TRL expects a function with signature:
```python
def trl_reward_function(prompts, completions, **kwargs) -> List[float]
```

`kwargs` receives `gold_answer` and `choices` from the dataset columns.

---

## 11. Evaluation Pipeline

### 11.1 In-Training Evaluation (`trainer.py :: evaluate()`)

- **When**: Baseline (pre-training), mid-step (`eval_every_n_steps`), or end-of-epoch
- **Mode**: Greedy decoding, single completion per question
- **Metrics**: mean_reward, mean_format, mean_correctness
- **Best checkpoint**: Saved when eval reward exceeds previous best

### 11.2 Standalone Evaluation (`evaluation.py`)

A separate script for offline evaluation of model predictions:

```bash
python evaluation.py --input results/inference/run_tools_20260323.json
```

**Metrics computed:**
- **Task-wise accuracy**: sound, speech, music
- **Difficulty-wise accuracy**: easy, medium, hard
- **Sub-category accuracy**: per fine-grained category
- **Overall accuracy**: total correct / total

**Matching logic** (`string_match`):
1. Tokenize prediction and gold answer into word sets
2. Check: all gold tokens appear in prediction
3. Check: no tokens from incorrect choices appear (excluding shared tokens with gold)

---

## 12. Inference Pipeline

### 12.1 `inference/tool_execute.py`

Runs model inference on evaluation data with optional LoRA adapters.

**Modes:**
- **Tool mode** (default): Two-phase generation with tool calls
- **Direct mode** (`--direct`): Single-phase, no tool calls

**Usage:**
```bash
# Base model, tool mode
python inference/tool_execute.py --model desta-8b --data mmau-test-mini-cached.json

# LoRA model, tool mode
python inference/tool_execute.py --model desta-8b --data mmau-test-mini-cached.json \
    --lora-path checkpoints/grpo-desta-run2/best

# Direct mode (no tools)
python inference/tool_execute.py --model desta-8b --data mmau-test-mini-cached.json --direct
```

**LoRA loading:**
```python
peft_model = PeftModel.from_pretrained(model.llm_model, lora_path)
model.llm_model = peft_model.base_model.model
```

### 12.2 SLURM Submission (`inference/run.sh`)

```bash
# Positional args: MODEL DATA LORA_PATH PRECOMPUTED_DIR OUTPUT DIRECT
sbatch inference/run.sh desta-8b mmau-test-mini-cached.json checkpoints/best precomputed_embeds
```

---

## 13. Configuration Reference

### 13.1 `default.yaml` — Conservative Configuration

| Parameter | Value | Notes |
|---|---|---|
| G | 16 | Group size (completions per question) |
| beta | 0.01 | Low KL penalty |
| temperature | 0.9 | |
| learning_rate | 1e-5 | |
| batch_size | 2 | Per GPU |
| gradient_checkpointing | false | |
| deepspeed_stage | 2 | ZeRO-2 |
| train_ratio | 0.10 | 10% of data for training |

### 13.2 `optimized.yaml` — Tuned Configuration

| Parameter | Value | Notes |
|---|---|---|
| G | 16 | Same group size |
| beta | 0.5 | Higher KL penalty (prevents divergence) |
| temperature | 1.1 | More exploration |
| learning_rate | 5e-5 | 5x higher |
| batch_size | 4 | Per GPU |
| gradient_checkpointing | true | Memory savings enabled |
| train_ratio | 0.30 | 30% of data for training |
| eval_every_n_steps | 10 | Mid-training eval |
| max_grad_norm | 0.5 | Tighter gradient clipping |
| warmup_steps | 50 | Longer warmup |

### 13.3 `qwen_omni.yaml` — Qwen-Specific

| Parameter | Value | Notes |
|---|---|---|
| G | 8 | Lower (31 GB base model → less headroom) |
| batch_size | 1 | Minimal due to model size |
| grad_accumulation_steps | 8 | Compensates for tiny batch |
| learning_rate | 2e-6 | Very conservative |

---

## 14. Distributed Training & Infrastructure

### 14.1 Accelerate + DeepSpeed

- **Accelerator** from HuggingFace `accelerate` handles multi-GPU distribution
- **DeepSpeed ZeRO-2**: Partitions optimizer states and gradients across GPUs
- Policy model + optimizer are `accelerator.prepare()`'d; reference model stays on its original device

### 14.2 SLURM Scripts

Both `run_grpo2.sh` and `run_trl.sh` handle:

- Conda environment activation (`slm`)
- CUDA_HOME and LD_LIBRARY_PATH configuration
- Offline mode for HuggingFace and WandB (compute nodes have no internet)
- Dynamic master port selection (avoids conflicts)
- NCCL/Gloo socket binding to loopback for single-node jobs

**Custom trainer launch:**
```bash
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=<free_port> \
    grpo/train_grpo.py --config grpo/configs/optimized.yaml
```

**TRL trainer launch:**
```bash
accelerate launch --num_processes=4 --mixed_precision=bf16 grpo/train_trl.py
```

### 14.3 Environment Variables

| Variable | Purpose |
|---|---|
| `TRANSFORMERS_OFFLINE=1` | No HuggingFace Hub downloads |
| `HF_HUB_OFFLINE=1` | Same (belt + suspenders) |
| `WANDB_MODE=offline` | No WandB upload attempts |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces CUDA memory fragmentation |
| `LLM_JUDGE_URL` | OpenAI-compatible endpoint for judge scoring |
| `LLM_JUDGE_MODEL` | Model name for judge |

---

## 15. Checkpoint Management

### 15.1 What Gets Saved

Only **LoRA adapter weights** are saved (not the full 8B LLM):

```
checkpoint_dir/
├── adapter_config.json          # LoRA hyperparameters
├── adapter_model.safetensors    # LoRA delta weights (~18 MB for rank-16)
└── training_meta.json           # Base model name, epoch, step, best eval reward
```

### 15.2 Save Triggers

| Trigger | Tag |
|---|---|
| Every N epochs (`save_every_n_epochs`) | `epoch-{N}` |
| New best eval reward | `best` |
| End of training | `final` |

### 15.3 Loading Checkpoints

```python
from peft import PeftModel
model = GRPODeSTA25AudioModel.from_pretrained("DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B")
peft_model = PeftModel.from_pretrained(model.llm_model, "checkpoints/grpo-desta-run2/best")
model.llm_model = peft_model.base_model.model
```

---

## 16. Logging & Monitoring

### 16.1 WandB (Offline)

Logs to local `wandb/` directory. Metrics tracked:

**Training:**
- `train/loss`, `train/reward`, `train/reward_format`, `train/reward_correctness`
- `train/kl`, `train/advantage_abs_mean`, `train/tool_call_fraction`
- `train/lr`, `train/policy_logprob`, `train/ref_logprob`
- `gpu/{i}/mem_used_gb`, `gpu/{i}/gpu_util_pct`

**Evaluation:**
- `eval/reward`, `eval/reward_format`, `eval/reward_correctness`, `eval/epoch`

Sync to cloud later: `wandb sync wandb/<run_dir>`

### 16.2 Rollout Logs

Detailed per-completion rollout data is written to a dedicated file:
```
{output_dir}/run_logs/{slurm_job_id}/rollout.log
```

Contains: question text, gold answer, each completion's phase1/phase2 text, reward breakdown, advantage, and log-prob details.

### 16.3 Step Metadata (JSON)

Per-step data saved to:
```
{output_dir}/run_logs/{slurm_job_id}/epoch_{N}/step_{global_step}.json
```

Contains full question, completions, rewards, and advantages for post-hoc analysis.

### 16.4 TRL Parquet Logs

The TRL pipeline saves completions to:
```
{output_dir}/completions/completions_{step:05d}.parquet
```

Enriched with `gold_answer`, `choices`, `id`, and aggregate GRPO metrics.

### 16.5 System Monitor

A background thread (`SystemMonitor`) collects GPU memory, GPU utilization, CPU, and RAM stats at configurable intervals (default: 30s) and logs them to WandB.

### 16.6 GPU Stats

The trainer collects nvidia-smi statistics at each logging step for all visible GPUs:
- Memory used (GB) and utilization (%)
- GPU compute utilization (%)
- Logged both to console and WandB

### 16.7 SIGTERM Handling

WandB is properly finalized when SLURM cancels a job (SIGTERM → flush → exit), preventing data loss.
