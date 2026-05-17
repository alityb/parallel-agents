from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N = np.array([5, 10, 20, 50, 100])
ISOLATED_WALL = np.array([28.77, 53.56, 115.39, 286.72, 573.21])
SHARED_WALL = np.array([10.14, 19.86, 36.32, 98.00, 190.80])
ISOLATED_PREFILL = np.array([63_903, 128_182, 257_094, 643_918, 1_287_681])
SHARED_PREFILL = np.array([1_981, 4_920, 9_874, 23_605, 50_806])


def human_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def main() -> None:
    out = Path("docs/figures/opencode_h100_shared_sglang.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    isolated_color = "#df7968"
    shared_color = "#5bc5a4"
    text_color = "#2c2c2c"
    accent = "#087060"

    x = np.arange(len(N))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel A: prefill compute.
    ax = axes[0]
    bars_a = ax.bar(
        x - width / 2,
        ISOLATED_PREFILL,
        width,
        color=isolated_color,
        label="Isolated OpenCode",
    )
    bars_b = ax.bar(
        x + width / 2,
        SHARED_PREFILL,
        width,
        color=shared_color,
        label="Shared SGLang",
    )
    ax.set_yscale("log")
    ax.set_title("A. Prefill Compute", fontsize=22, pad=18)
    ax.set_ylabel("Prefill tokens computed (log scale)", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in N], fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars_a:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            human_tokens(value),
            ha="center",
            va="bottom",
            fontsize=12,
            color=text_color,
        )

    for i, bar in enumerate(bars_b):
        value = bar.get_height()
        reduction = 1 - SHARED_PREFILL[i] / ISOLATED_PREFILL[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.30,
            f"-{reduction:.0%}\n{human_tokens(value)}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=accent,
        )

    # Panel B: wall-clock.
    ax = axes[1]
    bars_c = ax.bar(
        x - width / 2,
        ISOLATED_WALL,
        width,
        color=isolated_color,
        label="Isolated OpenCode",
    )
    bars_d = ax.bar(
        x + width / 2,
        SHARED_WALL,
        width,
        color=shared_color,
        label="Shared SGLang",
    )
    ax.set_title("B. End-to-End Runtime", fontsize=22, pad=18)
    ax.set_ylabel("Wall-clock seconds", fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in N], fontsize=15)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars_c:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 10,
            f"{value:.0f}s",
            ha="center",
            va="bottom",
            fontsize=12,
            color=text_color,
        )

    for i, bar in enumerate(bars_d):
        value = bar.get_height()
        speedup = ISOLATED_WALL[i] / SHARED_WALL[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 10,
            f"{speedup:.2f}x\n{value:.1f}s",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=accent,
        )

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=17,
    )

    fig.suptitle(
        "BatchAgent + Shared SGLang vs Isolated OpenCode Sessions\n"
        "H100 · Qwen2.5-32B",
        fontsize=25,
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        "Output equivalence is audited separately via per-task finding-set comparison.",
        ha="center",
        fontsize=12,
        color="#555555",
    )

    fig.tight_layout(rect=[0.03, 0.07, 0.98, 0.86], w_pad=3.0)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
