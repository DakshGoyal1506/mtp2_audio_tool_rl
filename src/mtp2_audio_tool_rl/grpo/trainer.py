"""
GRPOTrainer — custom GRPO training loop for DeSTA2.5-Audio tool-use.

Algorithm (per batch of Q questions):
  1. For each question q, sample G completions from the policy model.
  2. Inject cached <tool_output> for tool-calling completions.
  3. Compute rewards r_{q,g} = w_f * format_reward + w_c * correctness_reward.
  4. Normalize advantages within each group of G completions:
       A_{q,g} = (r_{q,g} - mean(r_q)) / (std(r_q) + eps)
  5. Compute GRPO loss:
       L = -1/(G*Q) * sum_{q,g} A_{q,g} * log_pi(o_{q,g}|x_q)
           + beta * KL(pi_theta || pi_ref)
  6. Backward through policy model params only (LoRA weights on Llama backbone).

References:
  DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
"""

import json
import logging
import os
import signal
from typing import Dict, List, Optional, Any

from accelerate.logging import get_logger

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from .model_wrapper import DeSTA25GRPOModel
from .model_wrapper_qwen import QwenOmniGRPOModel
from .rewards import compute_reward
from .prompts import (
    build_grpo_initial_prompt,
    build_grpo_followup_prompt,
    build_grpo_initial_prompt_qwen,
    build_grpo_followup_prompt_qwen,
)
from .rewards import extract_answer
from .system_monitor import SystemMonitor

# get_logger gates .info() to main process only in distributed runs
logger = get_logger(__name__, log_level="INFO")


def _gpu_stats() -> dict:
    """
    Collect memory and utilization stats for all visible CUDA devices.
    Uses nvidia-smi for both memory and utilization so we get system-wide
    numbers across all DDP ranks, not just the current process's allocations.
    Returns a flat dict keyed by gpu<i>/<metric>.
    """
    import subprocess
    stats = {}
    if not torch.cuda.is_available():
        return stats
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            idx, gpu_util, mem_used_mib, mem_total_mib = parts
            i = int(idx)
            mem_used  = float(mem_used_mib)  / 1024
            mem_total = float(mem_total_mib) / 1024
            stats[f"gpu{i}/mem_used_gb"]  = round(mem_used,  2)
            stats[f"gpu{i}/mem_total_gb"] = round(mem_total, 2)
            stats[f"gpu{i}/mem_util_pct"] = round(100.0 * mem_used / mem_total, 1) if mem_total > 0 else 0.0
            stats[f"gpu{i}/gpu_util_pct"] = float(gpu_util)
    except Exception:
        # Fallback: per-process torch allocations (rank-0 only, but better than nothing)
        for i in range(torch.cuda.device_count()):
            try:
                props     = torch.cuda.get_device_properties(i)
                total_gb  = props.total_memory / 1024 ** 3
                alloc_gb  = torch.cuda.memory_allocated(i) / 1024 ** 3
                stats[f"gpu{i}/mem_used_gb"]  = round(alloc_gb, 2)
                stats[f"gpu{i}/mem_total_gb"] = round(total_gb, 2)
                stats[f"gpu{i}/mem_util_pct"] = round(100.0 * alloc_gb / total_gb, 1)
            except Exception:
                pass
    return stats

class GRPOTrainer:
    """
    GRPO Trainer for DeSTA2.5-Audio.

    Args:
        policy_wrapper:     DeSTA25GRPOModel wrapping the trainable model.
        ref_wrapper:        DeSTA25GRPOModel wrapping the frozen reference model.
        accelerator:        HuggingFace Accelerator instance (handles multi-GPU).
        cfg:                OmegaConf DictConfig with all hyperparameters.
    """

    def __init__(
        self,
        policy_wrapper,   # DeSTA25GRPOModel or QwenOmniGRPOModel
        ref_wrapper,      # DeSTA25GRPOModel or QwenOmniGRPOModel
        accelerator: Accelerator,
        cfg,
    ):
        self.policy  = policy_wrapper
        self.ref     = ref_wrapper
        self.accel   = accelerator
        self.cfg     = cfg
        self._is_qwen = getattr(policy_wrapper, 'model_family', '') == 'qwen'

        # Select prompt builders based on model family.
        # Qwen uses its default system prompt; all tool logic goes in user message.
        if self._is_qwen:
            self._initial_prompt_fn  = build_grpo_initial_prompt_qwen
            self._followup_prompt_fn = build_grpo_followup_prompt_qwen
        else:
            self._initial_prompt_fn  = build_grpo_initial_prompt
            self._followup_prompt_fn = build_grpo_followup_prompt

        # Optimiser + scheduler are set up externally and injected via set_optimizer()
        self.optimizer = None
        self.scheduler = None

        self._global_step = 0
        self._best_eval_reward = -float("inf")
        self._system_monitor: Optional[SystemMonitor] = None
        self._rollout_logger = self._make_rollout_logger()

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _make_rollout_logger(self) -> logging.Logger:
        """Create a dedicated file logger for verbose rollout details.
        Writes to {output_dir}/rollout_{SLURM_JOB_ID}.log on the main process;
        returns a no-op logger on worker processes.
        """
        rl = logging.getLogger("grpo.rollout")
        rl.setLevel(logging.INFO)
        rl.propagate = False  # don't leak into root logger / SLURM output
        if not self.accel.is_main_process or rl.handlers:
            return rl
        slurm_id  = os.environ.get("SLURM_JOB_ID", "local")
        log_dir   = os.path.join(self.cfg.output_dir, "run_logs", slurm_id)
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "rollout.log"), mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s — %(message)s"))
        rl.addHandler(fh)
        return rl

    # ------------------------------------------------------------------
    # WandB initialisation
    # ------------------------------------------------------------------

    def _init_wandb(self):
        """Initialise wandb in offline mode. Non-fatal — training continues if wandb fails."""
        if not WANDB_AVAILABLE or not self.accel.is_main_process:
            return
        try:
            self._init_wandb_inner()
        except Exception as e:
            logger.warning(f"wandb init failed ({type(e).__name__}: {e}). Training will continue WITHOUT wandb logging.")

    def _init_wandb_inner(self):
        if not WANDB_AVAILABLE or not self.accel.is_main_process:
            return
        try:
            from omegaconf import OmegaConf
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
        except Exception:
            cfg_dict = dict(self.cfg)

        wandb_cfg  = self.cfg.get("wandb", {})
        project    = wandb_cfg.get("project", "grpo-desta-audio")
        run_name   = wandb_cfg.get("run_name", None)
        wandb_dir  = wandb_cfg.get("dir", "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb.init(
            project=project,
            name=run_name,
            config=cfg_dict,
            dir=wandb_dir,
            resume="allow",
            settings=wandb.Settings(start_method="thread", init_timeout=15),
        )
        logger.info(f"wandb initialised (offline). Run dir: {wandb.run.dir}")

        # Register SIGTERM handler so wandb is properly finalised when SLURM
        # cancels the job (SIGTERM arrives before SIGKILL, giving a grace period).
        _prev_sigterm = signal.getsignal(signal.SIGTERM)
        def _sigterm_handler(signum, frame):
            logger.info("SIGTERM received — flushing wandb and exiting.")
            try:
                if wandb.run is not None:
                    wandb.finish(quiet=True)
            except Exception:
                pass
            if callable(_prev_sigterm):
                _prev_sigterm(signum, frame)
            else:
                raise SystemExit(1)
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # ------------------------------------------------------------------
    # Core GRPO step
    # ------------------------------------------------------------------

    def grpo_step(self, batch: List[dict], dry_run: bool = False, epoch: int = 0) -> Dict[str, float]:
        """
        Run one GRPO update step over a batch of questions.

        Args:
            batch:   List of dataset item dicts.
            dry_run: If True, skip backward pass (for smoke-testing).

        Returns:
            Dict with scalar metrics: loss, mean_reward, mean_format, mean_correctness.
        """
        G         = self.cfg.G
        beta      = self.cfg.beta
        eps       = self.cfg.eps
        fw        = self.cfg.format_reward_weight
        cw        = self.cfg.correctness_reward_weight
        judge_url = os.getenv("LLM_JUDGE_URL") or self.cfg.get("judge_url") or None
        judge_mdl = os.getenv("LLM_JUDGE_MODEL") or self.cfg.get("judge_model") or None

        all_advantages: List[torch.Tensor] = []   # one scalar per (q, g)
        all_policy_logprobs: List[torch.Tensor] = []
        all_ref_logprobs:    List[torch.Tensor] = []
        all_rewards:     List[float] = []
        all_formats:     List[float] = []
        all_correct:     List[float] = []
        all_kl:          List[float] = []
        all_advantages_f: List[float] = []
        n_tool_calls:    int = 0
        n_total_comps:   int = 0

        # ---- Rollout phase (no_grad, inference only) ----
        # Disable gradient checkpointing so KV-cache works during generation
        _llm_backbone = self.policy.get_llm_backbone()
        _gc_was_enabled = getattr(_llm_backbone, 'is_gradient_checkpointing', False)
        if _gc_was_enabled:
            _llm_backbone.gradient_checkpointing_disable()

        rollout_data: List[List[dict]] = []  # [q][g]

        for item in batch:
            audio_path    = item["audio_path"]
            question      = item["question"]
            choices       = item["choices"]
            gold          = item["gold_answer"]
            cached_tools  = item["cached_tool_outputs"]

            sys_p, user_p = self._initial_prompt_fn(question, choices)

            # Extract cached speech recognition transcript (produced by Whisper at
            # dataset-build time) so the model skips VAD + Whisper decoder on every
            # rollout call — saves ~30 % of generate() wall-clock time.
            sr_output = cached_tools.get("speech_recognition") or {}
            transcription = (
                sr_output.get("text") if isinstance(sr_output, dict) else None
            )

            completions = self.policy.generate_completions(
                audio_path=audio_path,
                system_prompt=sys_p,
                user_prompt=user_p,
                cached_tool_outputs=cached_tools,
                followup_system_prompt=None,
                followup_user_template=self._followup_prompt_fn,
                question=question,
                choices=choices,
                G=G,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                transcription=transcription,
                precomputed_embed=item.get("precomputed_embed"),
            )

            # Compute rewards
            group_rewards = []
            for comp in completions:
                r = compute_reward(
                    completion=comp["text"],
                    called_tools=comp["called_tools"],
                    gold=gold,
                    choices=choices,
                    format_weight=fw,
                    correctness_weight=cw,
                    judge_url=judge_url,
                    judge_model=judge_mdl,
                )
                comp["reward_dict"] = r
                group_rewards.append(r["total"])
                all_rewards.append(r["total"])
                all_formats.append(r["format"])
                all_correct.append(r["correctness"])

            # Normalize within group
            r_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mean_r   = r_tensor.mean()
            std_r    = r_tensor.std() if G > 1 else torch.tensor(1.0)
            std_r    = std_r.clamp(min=eps)
            advantages = (r_tensor - mean_r) / std_r  # (G,)

            for g_idx, comp in enumerate(completions):
                comp["advantage"] = advantages[g_idx].item()
                all_advantages_f.append(abs(advantages[g_idx].item()))
                if comp["called_tools"]:
                    n_tool_calls += 1
            n_total_comps += len(completions)

            rollout_data.append(completions)

            # ---- Per-group rollout log → dedicated rollout file only ----
            if self.accel.is_main_process:
                n_tools_in_group = sum(1 for c in completions if c["called_tools"])
                PAD = "  "
                lines = [
                    f"{'='*80}",
                    f"[Rollout] Q={question} | gold={gold}",
                    f"  G={len(completions)} | tools={n_tools_in_group}/{len(completions)} | "
                    f"reward: mean={mean_r:.3f} std={std_r:.3f} "
                    f"range=[{min(group_rewards):.3f}, {max(group_rewards):.3f}]",
                ]
                for g_idx, comp in enumerate(completions):
                    tag      = "T" if comp["called_tools"] else "D"
                    tool_info = f" | tool={comp['tool_name']}" if comp["called_tools"] else ""
                    lines.append(f"  {'-'*76}")
                    lines.append(
                        f"  g={g_idx}({tag}){tool_info} | "
                        f"rew={comp['reward_dict']['total']:.3f} "
                        f"(fmt={comp['reward_dict']['format']:.2f} "
                        f"corr={comp['reward_dict']['correctness']:.2f}) | "
                        f"adv={comp['advantage']:+.3f}"
                    )
                    # Phase 1
                    lines.append(f"  [phase1]")
                    for ln in comp["phase1"].splitlines():
                        lines.append(f"{PAD}    {ln}")
                    # Phase 2 — tool injection
                    if comp["called_tools"]:
                        lines.append(f"  [inject]")
                        for ln in (comp.get("injected_output") or "").splitlines():
                            lines.append(f"{PAD}    {ln}")
                        # Phase 3 — post-tool response
                        lines.append(f"  [phase2]")
                        for ln in (comp.get("phase2") or "").splitlines():
                            lines.append(f"{PAD}    {ln}")
                lines.append(f"{'='*80}")
                self._rollout_logger.info("\n".join(lines))

        if dry_run:
            metrics = {
                "loss":               0.0,
                "mean_reward":        float(torch.tensor(all_rewards).float().mean()),
                "mean_format":        float(torch.tensor(all_formats).float().mean()),
                "mean_correctness":   float(torch.tensor(all_correct).float().mean()),
                "mean_kl":            0.0,
                "mean_advantage":     float(torch.tensor(all_advantages_f).mean()) if all_advantages_f else 0.0,
                "tool_call_fraction": n_tool_calls / max(n_total_comps, 1),
            }
            self._save_step_meta(batch, rollout_data, metrics, epoch)
            return metrics

        # ---- Log-prob phase ----
        # Re-enable gradient checkpointing for the backward pass (saves activation memory)
        if _gc_was_enabled:
            _llm_backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # MEMORY-EFFICIENT: backward per-completion so only ONE computation graph
        # lives in GPU memory at a time (instead of accumulating all G graphs).
        # First pass: count terms to get the correct 1/n_terms scale factor.
        n_terms = 0
        for _item, _completions in zip(batch, rollout_data):
            for comp in _completions:
                if abs(comp["advantage"]) < 1e-6:
                    continue
                n_terms += 1

        # Synchronize backward-call count across DDP ranks BEFORE any early return.
        # This collective op must be called by every rank. With DeepSpeed gradient
        # accumulation (gradient_accumulation_steps > 1), NCCL all-reduce boundaries
        # fire every N backward() calls. Different batch items per rank can yield
        # different n_terms → different boundary crossings → NCCL sequence-number
        # mismatch → deadlock (seen as 10-min NCCL watchdog timeout on step 10).
        # Fix: agree on the global max here, then pad the shorter rank's backward
        # loop with zero-gradient dummy calls so every rank does the same count.
        if self.accel.num_processes > 1:
            import torch.distributed as dist
            _n_t = torch.tensor(n_terms, dtype=torch.long, device=self.accel.device)
            dist.all_reduce(_n_t, op=dist.ReduceOp.MAX)
            _n_terms_padded = int(_n_t.item())
        else:
            _n_terms_padded = n_terms

        if _n_terms_padded == 0:
            logger.warning("No non-zero advantage terms in batch — skipping update.")
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.learning_rate
            metrics = {
                "loss":                0.0,
                "mean_reward":         float(torch.tensor(all_rewards).float().mean()),
                "mean_format":         float(torch.tensor(all_formats).float().mean()),
                "mean_correctness":    float(torch.tensor(all_correct).float().mean()),
                "mean_kl":             0.0,
                "mean_advantage":      float(torch.tensor(all_advantages_f).mean()) if all_advantages_f else 0.0,
                "tool_call_fraction":  n_tool_calls / max(n_total_comps, 1),
                "lr":                  current_lr,
                "mean_policy_logprob": 0.0,
                "mean_ref_logprob":    0.0,
            }
            self._save_step_meta(batch, rollout_data, metrics, epoch)
            return metrics

        # Free generation-phase VRAM before log-prob computation
        torch.cuda.empty_cache()

        accumulated_loss = 0.0  # scalar for logging only
        _n_backward_calls = 0   # track actual backward calls for rank-padding

        for q_idx, (item, completions) in enumerate(zip(batch, rollout_data)):
            audio_path    = item["audio_path"]
            question      = item["question"]
            choices       = item["choices"]
            embed_path    = item.get("precomputed_embed")

            sys_p, user_p = self._initial_prompt_fn(question, choices)

            sr_output = item.get("cached_tool_outputs", {}).get("speech_recognition") or {}
            item_transcription = (
                sr_output.get("text") if isinstance(sr_output, dict) else None
            )

            messages_p1 = self.policy.build_messages_phase1(
                audio_path=audio_path,
                system_prompt=sys_p,
                user_prompt=user_p,
                transcription=item_transcription,
                precomputed_embed=embed_path,
            )

            for comp in completions:
                advantage = comp["advantage"]
                if abs(advantage) < 1e-6:
                    continue

                if comp["called_tools"]:
                    policy_lp_p1, n1 = self.policy.compute_log_probs(messages_p1, comp["phase1"])

                    f_sys, f_user = self._followup_prompt_fn(
                        question, choices,
                        json.dumps(item["cached_tool_outputs"].get(comp["tool_name"], {}), indent=2)
                    )
                    messages_p2 = self.policy.build_messages_phase2(
                        audio_path=audio_path,
                        system_prompt=f_sys,
                        user_prompt=f_user,
                        precomputed_embed=embed_path,
                        transcription=item_transcription,
                    )
                    policy_lp_p2, n2 = self.policy.compute_log_probs(messages_p2, comp["phase2"])

                    # Weighted average of per-token-mean log-probs by token count
                    total_tokens = max(n1 + n2, 1)
                    policy_lp = (policy_lp_p1 * n1 + policy_lp_p2 * n2) / total_tokens

                    ref_lp_p1, _ = self.ref.reference_log_probs(messages_p1, comp["phase1"])
                    ref_lp_p2, _ = self.ref.reference_log_probs(messages_p2, comp["phase2"])
                    ref_lp = (ref_lp_p1 * n1 + ref_lp_p2 * n2) / total_tokens

                else:
                    policy_lp, _ = self.policy.compute_log_probs(messages_p1, comp["phase1"])
                    ref_lp, _    = self.ref.reference_log_probs(messages_p1, comp["phase1"])

                # Schulman KL approximator: always >= 0, avoids negative KL divergence
                # KL ≈ exp(log_ref - log_policy) - (log_ref - log_policy) - 1
                log_ratio = policy_lp - ref_lp.detach()
                kl = (torch.exp(-log_ratio) + log_ratio - 1.0).clamp(max=10.0)
                step_loss = (-advantage * policy_lp + beta * kl) / n_terms

                # Backward IMMEDIATELY — frees this computation graph before the
                # next forward pass, so only 1 graph lives in VRAM at a time.
                self.accel.backward(step_loss)
                _n_backward_calls += 1

                _plp = policy_lp.detach().item()
                _rlp = ref_lp.detach().item()
                _kl  = kl.detach().item()
                _sl  = step_loss.detach().item()

                accumulated_loss += _sl
                all_policy_logprobs.append(_plp)
                all_ref_logprobs.append(_rlp)
                all_kl.append(_kl)

                # ---- Per-completion log-prob log → rollout file only ----
                if self.accel.is_main_process:
                    tag = "T" if comp["called_tools"] else "D"
                    self._rollout_logger.info(
                        f"    [LogProb] q={q_idx} g={tag} | "
                        f"adv={advantage:+.3f} | "
                        f"plp={_plp:.4f} rlp={_rlp:.4f} | "
                        f"kl={_kl:.2e} | loss={_sl:.4e}"
                    )

                # Eagerly free graph tensors
                del step_loss, policy_lp, ref_lp, kl
                torch.cuda.empty_cache()

        # Pad backward calls to the global max so every rank executes the same
        # number, keeping DeepSpeed's gradient-accumulation NCCL all-reduce
        # boundaries in lockstep across ranks (prevents NCCL timeout / SIGABRT).
        if _n_backward_calls < _n_terms_padded:
            trainable_params = [p for p in self.policy.model.parameters() if p.requires_grad]
            for _ in range(_n_terms_padded - _n_backward_calls):
                # Touch every trainable parameter so dummy backward calls produce
                # the same reduction buckets as real backward calls on other ranks.
                # Using a single parameter here can create NCCL size mismatches
                # (for example 65,536 vs 9,437,184 elements) and deadlock DDP.
                dummy_loss = sum(param.reshape(-1)[:1].sum() * 0.0 for param in trainable_params)
                self.accel.backward(dummy_loss)
                del dummy_loss
            del trainable_params

        # Grad clip + optimizer step
        if self.accel.sync_gradients:
            self.accel.clip_grad_norm_(
                [p for p in self.policy.model.parameters() if p.requires_grad],
                self.cfg.max_grad_norm,
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        self._global_step += 1

        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.cfg.learning_rate

        metrics = {
            "loss":               accumulated_loss,
            "mean_reward":        float(torch.tensor(all_rewards).float().mean()),
            "mean_format":        float(torch.tensor(all_formats).float().mean()),
            "mean_correctness":   float(torch.tensor(all_correct).float().mean()),
            "mean_kl":            float(torch.tensor(all_kl).mean()) if all_kl else 0.0,
            "mean_advantage":     float(torch.tensor(all_advantages_f).mean()) if all_advantages_f else 0.0,
            "tool_call_fraction": n_tool_calls / max(n_total_comps, 1),
            "lr":                 current_lr,
            "mean_policy_logprob": float(torch.tensor(all_policy_logprobs).mean()) if all_policy_logprobs else 0.0,
            "mean_ref_logprob":    float(torch.tensor(all_ref_logprobs).mean())    if all_ref_logprobs    else 0.0,
        }
        self._save_step_meta(batch, rollout_data, metrics, epoch)
        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        dry_run: bool = False,
        skip_baseline_eval: bool = False,
    ):
        """
        Full training loop across cfg.epochs.

        Args:
            train_dataset:       GRPOAudioDataset for training.
            eval_dataset:        GRPOAudioDataset for validation (optional).
            dry_run:             If True, run one batch with no backward pass and exit.
            skip_baseline_eval:  If True, skip the pre-training baseline eval (~10 min).
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # return list of dicts as-is
        )
        # Prepare with accelerator
        train_loader = self.accel.prepare(train_loader)

        # ---- WandB + system monitor (main process only) ----
        if self.accel.is_main_process:
            self._init_wandb()
            monitor_interval = self.cfg.get("wandb", {}).get("monitor_interval_s", 30)
            device_idx = self.accel.device.index if self.accel.device.type == "cuda" else 0
            self._system_monitor = SystemMonitor(
                interval_s=monitor_interval,
                device_idx=device_idx if device_idx is not None else 0,
            )
            self._system_monitor.start()

        if self.accel.is_main_process:
            logger.info(
                f"Starting GRPO training | epochs={self.cfg.epochs} | "
                f"G={self.cfg.G} | batch_size={self.cfg.batch_size} | "
                f"train_samples={len(train_dataset)}"
            )

        # ---- Baseline evaluation before training ----
        if eval_dataset is not None and not skip_baseline_eval:
            baseline_metrics = self.evaluate(eval_dataset)
            self._best_eval_reward = baseline_metrics["mean_reward"]
            if self.accel.is_main_process:
                logger.info(
                    f"=== Baseline Eval (pre-training) | "
                    f"reward={baseline_metrics['mean_reward']:.3f} | "
                    f"fmt={baseline_metrics['mean_format']:.3f} | "
                    f"corr={baseline_metrics['mean_correctness']:.3f} ==="
                )
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log(
                        {
                            "eval/reward":             baseline_metrics["mean_reward"],
                            "eval/reward_format":      baseline_metrics["mean_format"],
                            "eval/reward_correctness": baseline_metrics["mean_correctness"],
                            "eval/epoch":              0,
                        },
                        step=0,
                    )

        for epoch in range(self.cfg.epochs):
            self.policy.model.train()

            for step, batch in enumerate(train_loader):
                metrics = self.grpo_step(batch, dry_run=dry_run, epoch=epoch)

                if self.accel.is_main_process:
                    gpu = _gpu_stats()
                    # Build a compact GPU summary string for the log line
                    gpu_parts = [
                        f"gpu{i}: {gpu.get(f'gpu{i}/mem_used_gb','?')}GB "
                        f"({gpu.get(f'gpu{i}/gpu_util_pct','?')}%)"
                        for i in range(torch.cuda.device_count())
                        if f"gpu{i}/mem_used_gb" in gpu
                    ]
                    gpu_str = "  |  " + "  ".join(gpu_parts) if gpu_parts else ""
                    skipped = (metrics["loss"] == 0.0 and metrics["mean_kl"] == 0.0)
                    skip_tag = " [SKIPPED]" if skipped else ""
                    logger.info(
                        f"Epoch {epoch+1}/{self.cfg.epochs} | "
                        f"Step {self._global_step} | "
                        f"loss={metrics['loss']:.4f} | "
                        f"reward={metrics['mean_reward']:.3f} | "
                        f"fmt={metrics['mean_format']:.3f} | "
                        f"corr={metrics['mean_correctness']:.3f} | "
                        f"kl={metrics['mean_kl']:.2e} | "
                        f"plp={metrics['mean_policy_logprob']:.4f} | "
                        f"rlp={metrics['mean_ref_logprob']:.4f} | "
                        f"tool%={metrics['tool_call_fraction']*100:.1f}"
                        f"{gpu_str}{skip_tag}"
                    )
                    if WANDB_AVAILABLE and wandb.run is not None and step % self.cfg.log_every_n_steps == 0:
                        wandb.log(
                            {
                                "train/loss":               metrics["loss"],
                                "train/reward":             metrics["mean_reward"],
                                "train/reward_format":      metrics["mean_format"],
                                "train/reward_correctness": metrics["mean_correctness"],
                                "train/kl":                 metrics["mean_kl"],
                                "train/advantage_abs_mean": metrics["mean_advantage"],
                                "train/tool_call_fraction": metrics["tool_call_fraction"],
                                "train/lr":                 metrics["lr"],
                                "train/policy_logprob":     metrics["mean_policy_logprob"],
                                "train/ref_logprob":        metrics["mean_ref_logprob"],
                                "train/epoch":              epoch + 1,
                                **{f"gpu/{k}": v for k, v in gpu.items()},
                            },
                            step=self._global_step,
                        )

                # Mid-step evaluation
                eval_every = self.cfg.get("eval_every_n_steps", 0)
                if (
                    eval_every > 0
                    and eval_dataset is not None
                    and self._global_step % eval_every == 0
                ):
                    self._run_eval_and_maybe_save(eval_dataset, epoch, label=f"Step {self._global_step}")
                    self.policy.model.train()  # restore train mode

                if dry_run:
                    if self.accel.is_main_process:
                        logger.info("[DRY RUN] Single batch processed. Exiting.")
                    return

            # End-of-epoch validation (skipped if eval_every_n_steps > 0 to avoid double-eval
            # on the last step of an epoch — set eval_every_n_steps: 0 to restore epoch-only eval)
            eval_every = self.cfg.get("eval_every_n_steps", 0)
            if eval_dataset is not None and eval_every == 0:
                self._run_eval_and_maybe_save(eval_dataset, epoch, label=f"Epoch {epoch+1}")

            # Periodic checkpoint (always save every N epochs)
            if self.accel.is_main_process and (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

        if self.accel.is_main_process:
            logger.info("Training complete.")
            self._save_checkpoint(epoch, suffix="final")
            # Tear down system monitor and finish wandb
            if self._system_monitor is not None:
                self._system_monitor.stop()
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.finish()
                logger.info("wandb run finished. To sync: wandb sync <run_dir>")

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _run_eval_and_maybe_save(self, eval_dataset, epoch: int, label: str = ""):
        """Run evaluation, log results, update best checkpoint if improved."""
        eval_metrics = self.evaluate(eval_dataset)
        if self.accel.is_main_process:
            logger.info(
                f"=== Eval {label} | "
                f"reward={eval_metrics['mean_reward']:.3f} | "
                f"fmt={eval_metrics['mean_format']:.3f} | "
                f"corr={eval_metrics['mean_correctness']:.3f} ==="
            )
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log(
                    {
                        "eval/reward":             eval_metrics["mean_reward"],
                        "eval/reward_format":      eval_metrics["mean_format"],
                        "eval/reward_correctness": eval_metrics["mean_correctness"],
                        "eval/epoch":              epoch + 1,
                    },
                    step=self._global_step,
                )
            self._save_epoch_meta(epoch, eval_metrics)
            current_reward = eval_metrics["mean_reward"]
            if current_reward > self._best_eval_reward:
                logger.info(
                    f"New best eval reward: {current_reward:.4f} "
                    f"(prev best: {self._best_eval_reward:.4f})"
                )
                self._best_eval_reward = current_reward
                self._save_checkpoint(epoch, suffix="best")

    def evaluate(self, eval_dataset) -> Dict[str, float]:
        """Greedy-decode evaluation over eval_dataset, report mean rewards."""
        self.policy.model.eval()
        all_rewards, all_formats, all_correct = [], [], []

        fw = self.cfg.format_reward_weight
        cw = self.cfg.correctness_reward_weight
        judge_url = os.getenv("LLM_JUDGE_URL") or None
        judge_mdl = os.getenv("LLM_JUDGE_MODEL") or None

        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

        with torch.no_grad():
            for item_list in eval_loader:
                item = item_list[0]
                audio_path   = item["audio_path"]
                question     = item["question"]
                choices      = item["choices"]
                gold         = item["gold_answer"]
                cached_tools = item["cached_tool_outputs"]

                sys_p, user_p = self._initial_prompt_fn(question, choices)

                # Generate single greedy completion
                comps = self.policy.generate_completions(
                    audio_path=audio_path,
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    cached_tool_outputs=cached_tools,
                    followup_system_prompt=None,
                    followup_user_template=self._followup_prompt_fn,
                    question=question,
                    choices=choices,
                    G=1,
                    temperature=0.0,   # greedy
                    precomputed_embed=item.get("precomputed_embed"),
                    top_p=1.0,
                    max_new_tokens=self.cfg.max_new_tokens,
                )
                comp = comps[0]
                r = compute_reward(
                    completion=comp["text"],
                    called_tools=comp["called_tools"],
                    gold=gold,
                    choices=choices,
                    format_weight=fw,
                    correctness_weight=cw,
                    judge_url=judge_url,
                    judge_model=judge_mdl,
                )
                all_rewards.append(r["total"])
                all_formats.append(r["format"])
                all_correct.append(r["correctness"])

        return {
            "mean_reward":      sum(all_rewards) / max(len(all_rewards), 1),
            "mean_format":      sum(all_formats) / max(len(all_formats), 1),
            "mean_correctness": sum(all_correct) / max(len(all_correct), 1),
        }

    # ------------------------------------------------------------------
    # Step & epoch metadata saving
    # ------------------------------------------------------------------

    def _save_step_meta(
        self,
        batch: List[dict],
        rollout_data: List[List[dict]],
        metrics: Dict[str, float],
        epoch: int,
    ):
        """
        Persist per-step, per-instance, per-completion data to:
            {output_dir}/run_logs/{slurm_job_id}/epoch_{N}/step_{global_step}.json
        Main process only — non-fatal.
        """
        if not self.accel.is_main_process:
            return
        try:
            slurm_id = os.environ.get("SLURM_JOB_ID", "local")
            step_dir = os.path.join(
                self.cfg.output_dir, "run_logs", slurm_id, f"epoch_{epoch + 1}"
            )
            os.makedirs(step_dir, exist_ok=True)

            instances = []
            for item, completions in zip(batch, rollout_data):
                group_rewards = [c["reward_dict"]["total"] for c in completions]
                r_t = torch.tensor(group_rewards, dtype=torch.float32)
                inst: Dict[str, Any] = {
                    "question":          item["question"],
                    "gold":              item["gold_answer"],
                    "choices":           item.get("choices", []),
                    "audio_path":        item["audio_path"],
                    "group_mean_reward": float(r_t.mean()),
                    "group_std_reward":  float(r_t.std()) if len(completions) > 1 else 0.0,
                    "completions":       [],
                }
                for g_idx, comp in enumerate(completions):
                    inst["completions"].append({
                        "g":            g_idx,
                        "type":         "T" if comp["called_tools"] else "D",
                        "phase1":       comp.get("phase1", ""),
                        "phase2":       comp.get("phase2", ""),
                        "called_tools": bool(comp["called_tools"]),
                        "tool_name":    comp.get("tool_name"),
                        "reward":       comp["reward_dict"],
                        "advantage":    comp["advantage"],
                    })
                instances.append(inst)

            record = {
                "epoch":       epoch + 1,
                "global_step": self._global_step,
                "metrics":     metrics,
                "instances":   instances,
            }
            out_path = os.path.join(step_dir, f"step_{self._global_step}.json")
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save step meta: {e}")

    def _save_epoch_meta(self, epoch: int, eval_metrics: Dict[str, float]):
        """
        Persist epoch-level eval summary to:
            {output_dir}/run_logs/{slurm_job_id}/epoch_{N}/meta.json
        Main process only — non-fatal.
        """
        if not self.accel.is_main_process:
            return
        try:
            slurm_id  = os.environ.get("SLURM_JOB_ID", "local")
            epoch_dir = os.path.join(
                self.cfg.output_dir, "run_logs", slurm_id, f"epoch_{epoch + 1}"
            )
            os.makedirs(epoch_dir, exist_ok=True)
            record = {
                "epoch":        epoch + 1,
                "global_step":  self._global_step,
                "eval_metrics": eval_metrics,
            }
            with open(os.path.join(epoch_dir, "meta.json"), "w") as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save epoch meta: {e}")

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, suffix: str = ""):
        """Save ONLY LoRA adapter weights (not the full LLM).

        Saves via peft's LoraModel.save_pretrained() which writes:
          - adapter_config.json   (LoRA hyperparams)
          - adapter_model.safetensors  (LoRA delta weights, ~18 MB for rank-16)

        To load later:  PeftModel.from_pretrained(base_model, save_path)
        """
        try:
            tag = suffix or f"epoch-{epoch+1}"
            save_path = os.path.join(self.cfg.output_dir, tag)
            os.makedirs(save_path, exist_ok=True)

            # Unwrap model before saving (handles accelerate wrapping)
            unwrapped = self.accel.unwrap_model(self.policy.model)

            # _peft_model is the PeftModel which has save_pretrained()
            # that writes only adapter weights + config (not the full 8B LLM).
            # Note: _lora_base_model (LoraModel) does NOT have save_pretrained;
            # only PeftModel does.  That's why we store both.
            peft_model = getattr(unwrapped, "_peft_model", None)
            if peft_model is not None and hasattr(peft_model, "save_pretrained"):
                peft_model.save_pretrained(save_path)
                _adapter_files = list(__import__('pathlib').Path(save_path).glob('adapter_*'))
                _adapter_mb = sum(f.stat().st_size for f in _adapter_files) / 1e6
                logger.info(f"Saved LoRA adapter weights ({_adapter_mb:.1f} MB) → {[f.name for f in _adapter_files]}")
            else:
                # Fallback: save full model if PEFT wrapper not found
                logger.warning("PeftModel not found — saving full LLM weights (this is NOT expected with LoRA).")
                llm = self.policy.get_llm_backbone()
                llm.save_pretrained(save_path)

            # Also save a metadata file so we know which base model to load
            import json as _json
            meta = {
                "base_model": self.cfg.model_hf_name,
                "model_family": getattr(self.policy, 'model_family', 'desta'),
                "lora_rank": self.cfg.lora_rank,
                "lora_alpha": self.cfg.lora_alpha,
                "epoch": epoch + 1,
                "global_step": self._global_step,
                "best_eval_reward": self._best_eval_reward,
            }
            with open(os.path.join(save_path, "training_meta.json"), "w") as f:
                _json.dump(meta, f, indent=2)

            logger.info(f"LoRA adapter checkpoint saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
