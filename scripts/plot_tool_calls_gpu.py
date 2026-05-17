from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_PATH = Path("tests/benchmarks/results/backend_raw_vs_batchagent/qwen25_32b_h100_400ms_rerun.json")
GPU_CSV = Path("tests/benchmarks/results/backend_raw_vs_batchagent/qwen25_32b_h100_400ms_gpu_samples.csv")
OUT = Path("docs/figures/tool_calls_gpu_util.png")


def load_gpu_util() -> np.ndarray:
    values: list[float] = []
    for raw in GPU_CSV.read_text().splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 6:
            continue
        try:
            values.append(float(parts[-1]))
        except ValueError:
            continue
    return np.array(values, dtype=float)


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    raw = result["raw_endpoint_loop"]
    batch = result["batchagent"]
    gpu = load_gpu_util()

    isolated_calls = raw["tool_calls_executed"]
    batch_calls = batch["tool_calls_executed"]
    reduction = 1 - batch_calls / isolated_calls

    mean_gpu = float(np.mean(gpu)) if len(gpu) else 0.0
    peak_gpu = float(np.max(gpu)) if len(gpu) else 0.0
    busy_gpu = float(np.mean(gpu > 0) * 100) if len(gpu) else 0.0

    isolated_color = "#df7968"
    shared_color = "#5bc5a4"
    line_color = "#087060"
    text_color = "#2c2c2c"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    bars = ax.bar(
        [0, 1],
        [isolated_calls, batch_calls],
        color=[isolated_color, shared_color],
        width=0.58,
    )
    ax.set_title("A. Tool Calls Executed", fontsize=22, pad=16)
    ax.set_ylabel("Tool calls", fontsize=18)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Isolated", "BatchAgent"], fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(isolated_calls, batch_calls) * 1.22)

    ax.text(
        bars[0].get_x() + bars[0].get_width() / 2,
        isolated_calls + 4,
        f"{isolated_calls} calls",
        ha="center",
        va="bottom",
        fontsize=15,
        color=text_color,
    )
    ax.text(
        bars[1].get_x() + bars[1].get_width() / 2,
        batch_calls + 4,
        f"-{reduction:.0%}\n{batch_calls} calls",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color=line_color,
    )

    ax = axes[1]
    x = np.arange(len(gpu)) * 0.5
    ax.plot(x, gpu, color=line_color, linewidth=2.6)
    ax.fill_between(x, gpu, color=shared_color, alpha=0.22)
    ax.set_title("B. GPU Utilization Trace", fontsize=22, pad=16)
    ax.set_ylabel("GPU utilization (%)", fontsize=18)
    ax.set_xlabel("Elapsed seconds", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 108)
    ax.text(
        0.98,
        0.92,
        f"peak {peak_gpu:.0f}%\nmean {mean_gpu:.0f}%\nbusy {busy_gpu:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=text_color,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.35"},
    )

    fig.suptitle(
        "BatchAgent Tool Coalescing + GPU Utilization\nH100 · Qwen2.5-32B",
        fontsize=25,
        y=0.995,
    )
    fig.text(
        0.5,
        0.02,
        "GPU panel is the nvidia-smi sample trace for the H100 benchmark run.",
        ha="center",
        fontsize=12,
        color="#555555",
    )
    fig.tight_layout(rect=[0.03, 0.07, 0.98, 0.86], w_pad=3.0)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
