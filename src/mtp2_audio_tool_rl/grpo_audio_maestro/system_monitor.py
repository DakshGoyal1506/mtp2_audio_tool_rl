"""
SystemMonitor — background thread that samples GPU, CPU, and RAM usage
and logs them to wandb at a fixed interval.

GPU stats:   torch.cuda (memory) + nvidia-smi subprocess (utilization %)
CPU stats:   psutil.cpu_percent()
RAM stats:   psutil.virtual_memory()
"""

import logging
import subprocess
import threading
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _nvml_utilization(device_idx: int = 0) -> Optional[float]:
    """
    Query GPU utilization % via nvidia-smi.
    Returns None if unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={device_idx}",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


class SystemMonitor:
    """
    Daemon thread that logs hardware metrics to wandb every `interval_s` seconds.

    Usage:
        monitor = SystemMonitor(interval_s=30, device_idx=0)
        monitor.start()
        ...training...
        monitor.stop()
    """

    def __init__(self, interval_s: int = 30, device_idx: int = 0):
        self.interval_s  = interval_s
        self.device_idx  = device_idx
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True, name="SystemMonitor")
        self._step       = 0

        # Lazy import — don't hard-fail if psutil missing
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not installed — CPU/RAM metrics will be skipped.")
            self._psutil = None

        try:
            import wandb as _wandb
            self._wandb = _wandb
        except ImportError:
            self._wandb = None

    def start(self):
        self._thread.start()
        logger.info(f"SystemMonitor started (interval={self.interval_s}s, gpu={self.device_idx})")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=self.interval_s + 2)

    def _run(self):
        while not self._stop_event.wait(timeout=self.interval_s):
            metrics = self._collect()
            if metrics and self._wandb is not None:
                try:
                    # commit=False so we don't advance wandb's internal step
                    # counter — training loop owns the step counter.
                    self._wandb.log(
                        {"system/" + k: v for k, v in metrics.items()},
                        commit=False,
                    )
                except Exception as e:
                    logger.debug(f"SystemMonitor wandb log failed: {e}")
            self._step += 1

    def _collect(self) -> dict:
        metrics = {}

        # --- GPU memory (via torch.cuda) ---
        if torch.cuda.is_available():
            try:
                props        = torch.cuda.get_device_properties(self.device_idx)
                total_gb     = props.total_memory / 1024 ** 3
                alloc_gb     = torch.cuda.memory_allocated(self.device_idx)  / 1024 ** 3
                reserved_gb  = torch.cuda.memory_reserved(self.device_idx)   / 1024 ** 3
                max_alloc_gb = torch.cuda.max_memory_allocated(self.device_idx) / 1024 ** 3

                metrics["gpu_mem_allocated_gb"]     = round(alloc_gb,     3)
                metrics["gpu_mem_reserved_gb"]       = round(reserved_gb,  3)
                metrics["gpu_mem_max_allocated_gb"]  = round(max_alloc_gb, 3)
                metrics["gpu_mem_total_gb"]          = round(total_gb,     3)
                metrics["gpu_mem_util_pct"]          = round(100.0 * alloc_gb / total_gb, 2)
            except Exception as e:
                logger.debug(f"GPU memory query failed: {e}")

            # --- GPU utilization (nvidia-smi) ---
            util = _nvml_utilization(self.device_idx)
            if util is not None:
                metrics["gpu_util_pct"] = util

        # --- CPU & RAM (psutil) ---
        if self._psutil is not None:
            try:
                metrics["cpu_util_pct"] = self._psutil.cpu_percent(interval=None)
                vm = self._psutil.virtual_memory()
                metrics["ram_used_gb"]  = round(vm.used  / 1024 ** 3, 3)
                metrics["ram_total_gb"] = round(vm.total / 1024 ** 3, 3)
                metrics["ram_util_pct"] = vm.percent
            except Exception as e:
                logger.debug(f"psutil query failed: {e}")

        return metrics
