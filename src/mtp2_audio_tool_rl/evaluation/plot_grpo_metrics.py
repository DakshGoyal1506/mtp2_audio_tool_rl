#!/usr/bin/env python3
"""Plot important GRPO metrics from training logs.

This script parses dict-style metric lines from logs, e.g.:
{'loss': 0.54, 'reward': 0.55, 'kl': 0.001, 'entropy': 1.02, 'epoch': 0.01}

It supports both train and eval keys commonly logged by GRPO/TRL trainers.
"""

import argparse
import ast
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _import_matplotlib_pyplot():
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        )

    # Headless-safe backend for clusters/servers.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


DEFAULT_METRIC_ORDER = [
    "reward",
    "kl",
    "entropy",
    "eval_reward",
    "eval_kl",
    "eval_entropy",
    "loss",
    "eval_loss",
    "grad_norm",
    "learning_rate",
    "reward_std",
    "eval_reward_std",
    "frac_reward_zero_std",
    "eval_frac_reward_zero_std",
    "step_time",
    "clip_ratio/low_mean",
    "clip_ratio/high_mean",
    "clip_ratio/region_mean",
    "eval_clip_ratio/low_mean",
    "eval_clip_ratio/high_mean",
    "eval_clip_ratio/region_mean",
]


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _extract_literal_dict(line: str) -> Optional[Dict[str, object]]:
    line = line.strip()
    if "{" not in line or "}" not in line:
        return None

    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return None

    snippet = line[start : end + 1]
    try:
        parsed = ast.literal_eval(snippet)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_metric_rows(log_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            parsed = _extract_literal_dict(raw_line)
            if parsed is None:
                continue

            numeric_items = {k: v for k, v in parsed.items() if _is_number(v)}
            if not numeric_items:
                continue
            rows.append(numeric_items)
    return rows


def available_metrics(rows: Sequence[Dict[str, object]]) -> List[str]:
    keys = set()
    for row in rows:
        for k, v in row.items():
            if _is_number(v):
                keys.add(k)
    return sorted(keys)


def choose_metrics(rows: Sequence[Dict[str, object]], requested: Optional[str]) -> List[str]:
    available = available_metrics(rows)
    available_set = set(available)

    if requested:
        out: List[str] = []
        for m in (x.strip() for x in requested.split(",")):
            if not m:
                continue
            if m in available_set:
                out.append(m)
            else:
                print(f"[warn] metric '{m}' not found in log; skipping")
        return out

    picked = [m for m in DEFAULT_METRIC_ORDER if m in available_set]

    # Add reward function-specific and other GRPO diagnostics if present.
    extras = [
        m
        for m in available
        if (
            m.startswith("rewards/")
            or m.startswith("eval_rewards/")
            or "clip_ratio" in m
        )
        and m not in picked
    ]
    picked.extend(extras)

    return picked


def series_for_metric(
    rows: Sequence[Dict[str, object]],
    metric: str,
    x_key: Optional[str],
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    point_idx = 0

    for row in rows:
        if metric not in row:
            continue
        value = row[metric]
        if not _is_number(value):
            continue

        if x_key and x_key in row and _is_number(row[x_key]):
            x = float(row[x_key])
        else:
            x = float(point_idx)
        point_idx += 1

        xs.append(x)
        ys.append(float(value))

    return xs, ys


def plot_metrics(
    rows: Sequence[Dict[str, object]],
    metrics: Sequence[str],
    output_path: Path,
    x_key: Optional[str],
    title: str,
    max_cols: int,
) -> None:
    plt = _import_matplotlib_pyplot()

    if not metrics:
        raise ValueError("No metrics to plot. Check your log file or --metrics argument.")

    cols = max(1, min(max_cols, len(metrics)))
    rows_n = math.ceil(len(metrics) / cols)

    fig, axes = plt.subplots(rows_n, cols, figsize=(6 * cols, 3.8 * rows_n), squeeze=False)
    flat_axes = [ax for row_ax in axes for ax in row_ax]

    for i, metric in enumerate(metrics):
        ax = flat_axes[i]
        x, y = series_for_metric(rows, metric, x_key)
        if not x:
            ax.set_visible(False)
            continue

        is_eval = metric.startswith("eval_") or metric.startswith("eval/")
        color = "tab:orange" if is_eval else "tab:blue"
        marker = "o" if len(x) <= 60 else None

        # --- OUTLIER CLIPPING LOGIC ---
        # If there are enough points, ignore extreme peaks by setting y-limits
        if len(y) > 20:
            sorted_y = sorted(y)
            # Find 1st and 99th percentiles
            p1 = sorted_y[int(len(sorted_y) * 0.01)]
            p99 = sorted_y[int(len(sorted_y) * 0.99)]

            y_range = p99 - p1
            if y_range > 0:
                # Allow a margin of 1.5x the range above the 99th and below the 1st percentile
                y_max_limit = p99 + 1.5 * y_range
                y_min_limit = p1 - 1.5 * y_range

                # If the max/min are extreme, clip the y-axis view
                actual_min, actual_max = sorted_y[0], sorted_y[-1]
                if actual_max > y_max_limit or actual_min < y_min_limit:
                    ax.set_ylim(
                        max(actual_min, y_min_limit),
                        min(actual_max, y_max_limit)
                    )
        # ------------------------------

        ax.plot(x, y, color=color, linewidth=1.3, marker=marker, markersize=2.6)
        ax.set_title(metric)
        ax.set_xlabel(x_key if x_key else "point_index")
        ax.grid(alpha=0.25)

    for j in range(len(metrics), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot GRPO train/eval metrics from a log file")
    p.add_argument("--log-file", type=Path, required=True, help="Path to GRPO training log")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <log_file_stem>_metrics.png)",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric keys to plot. If omitted, script picks important ones automatically.",
    )
    p.add_argument(
        "--x-key",
        type=str,
        default="epoch",
        help="Key to use as x-axis (default: epoch). Falls back to point index if missing.",
    )
    p.add_argument("--max-cols", type=int, default=3, help="Maximum subplot columns")
    p.add_argument("--title", type=str, default=None, help="Figure title")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.log_file.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    rows = parse_metric_rows(args.log_file)
    if not rows:
        raise RuntimeError(
            "No metric dict rows found in the log. Ensure the file contains lines with dict-like metrics."
        )

    metrics = choose_metrics(rows, args.metrics)
    if not metrics:
        raise RuntimeError("No requested metrics were found in the log.")

    output = args.output
    if output is None:
        output = args.log_file.with_name(f"{args.log_file.stem}_metrics.png")

    title = args.title if args.title else f"GRPO Metrics: {args.log_file.name}"
    x_key = args.x_key if args.x_key else None

    plot_metrics(rows, metrics, output, x_key=x_key, title=title, max_cols=args.max_cols)

    print(f"[ok] parsed rows: {len(rows)}")
    print(f"[ok] plotted metrics: {len(metrics)}")
    print(f"[ok] saved: {output}")
    print("[info] available metric keys:")
    for k in available_metrics(rows):
        print(f"  - {k}")


if __name__ == "__main__":
    main()
