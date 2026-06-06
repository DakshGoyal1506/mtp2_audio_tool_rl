#!/usr/bin/env python3
"""Generate a 2×2 matrix table image from comparison_analysis.json."""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def make_table(data, output):
    tool_analysis = data["tool_analysis"]

    # Sort tools by total count descending
    tools_sorted = sorted(tool_analysis.items(), key=lambda x: -x[1]["total"])

    # ── Build rows ────────────────────────────────────────────────────────
    headers = [
        "Tool", "N",
        "Easy\n(both ✓)", "Adversarial\n(direct ✓, forced ✗)",
        "Helpful\n(direct ✗, forced ✓)", "Very Hard\n(both ✗)",
        "Direct\nAcc", "Forced\nAcc", "Δ",
    ]

    rows = []
    # Overall row
    m = data["matrix"]
    total = data["total_compared"]
    easy = m["easy (both ✓)"]
    adv = m["adversarial (direct ✓, tool ✗)"]
    hlp = m["helpful_tool (direct ✗, tool ✓)"]
    hard = m["very_hard (both ✗)"]
    d_acc = data["direct_accuracy"]
    f_acc = data["forced_tool_accuracy"]
    delta = data["net_tool_impact"]

    rows.append([
        "OVERALL", str(total),
        f"{easy} ({easy/total*100:.1f}%)", f"{adv} ({adv/total*100:.1f}%)",
        f"{hlp} ({hlp/total*100:.1f}%)", f"{hard} ({hard/total*100:.1f}%)",
        f"{d_acc*100:.1f}%", f"{f_acc*100:.1f}%", f"{delta*100:+.1f}%",
    ])

    # Per-tool rows
    for name, t in tools_sorted:
        n = t["total"]
        e, a, h, v = t["easy"], t["adversarial"], t["helpful_tool"], t["very_hard"]
        d_a = (e + a) / max(n, 1)
        f_a = t["tool_accuracy"]
        d = f_a - d_a
        rows.append([
            name.replace("_", " "), str(n),
            f"{e} ({e/n*100:.1f}%)", f"{a} ({a/n*100:.1f}%)",
            f"{h} ({h/n*100:.1f}%)", f"{v} ({v/n*100:.1f}%)",
            f"{d_a*100:.1f}%", f"{f_a*100:.1f}%", f"{d*100:+.1f}%",
        ])

    # ── Render table ──────────────────────────────────────────────────────
    n_rows = len(rows)
    n_cols = len(headers)

    fig_width = 16
    fig_height = 0.42 * n_rows + 1.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)



    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    # ── Style ─────────────────────────────────────────────────────────────
    # Header styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_height(0.08)

    # Color maps for matrix columns
    green_light = "#d5f5e3"
    red_light = "#fadbd8"
    blue_light = "#d6eaf8"
    gray_light = "#f2f3f4"

    col_colors = {
        2: green_light,   # easy
        3: red_light,     # adversarial
        4: blue_light,    # helpful
        5: gray_light,    # very hard
    }

    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            # Overall row bold
            if i == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#ebedef")
            else:
                # Alternate row shading
                base = "#ffffff" if i % 2 == 0 else "#f9f9f9"
                cell.set_facecolor(base)

            # Matrix column coloring
            if j in col_colors:
                if i == 0:
                    cell.set_facecolor(col_colors[j])
                    cell.set_text_props(fontweight="bold")
                else:
                    # Lighter version for data rows
                    cell.set_facecolor(col_colors[j])
                    cell.set_alpha(0.7)

            # Delta column coloring
            if j == 8:
                val = rows[i][j]
                if val.startswith("+"):
                    cell.set_text_props(color="#27ae60", fontweight="bold")
                elif val.startswith("-"):
                    cell.set_text_props(color="#e74c3c", fontweight="bold")

    # Column widths
    col_widths = [0.14, 0.04, 0.11, 0.13, 0.13, 0.11, 0.07, 0.07, 0.06]
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[i, j].set_width(w)

    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close()
    print(f"Saved: {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/inference/101490_tools/comparison_analysis.json")
    ap.add_argument("--output", default="results/inference/101490_tools/comparison_matrix_table.png")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    make_table(data, args.output)


if __name__ == "__main__":
    main()
