#!/usr/bin/env python3
"""Plot Base vs. Instruct Extraversion and Neuroticism slope contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
OUT_PATH = EXP_DIR / "figures" / "base_vs_instruct_contrasts.png"
CONCEPTS = ("happy", "sad", "angry", "hostile")
LABELS = ("happy", "sad", "angry", "hostile\n(rob.)")


def load_effect(filename: str, effect: str) -> tuple[list[float], list[float], list[bool]]:
    payload = json.loads((RESULTS_DIR / filename).read_text(encoding="utf-8"))
    by_axis = {
        row["axis"]: row
        for row in payload["inference"]
        if row["effect"] == effect
    }
    means = [by_axis[c]["mean"] * 1e4 for c in CONCEPTS]
    ses = [by_axis[c]["standard_error"] * 1e4 for c in CONCEPTS]
    sig = [
        (by_axis[c].get("holm_adjusted_p") is not None)
        and (by_axis[c]["holm_adjusted_p"] < 0.05)
        for c in CONCEPTS
    ]
    return means, ses, sig


def draw_panel(
    ax: plt.Axes,
    effect: str,
    title: str,
    *,
    show_legend: bool = False,
) -> None:
    x = np.arange(len(CONCEPTS))
    width = 0.34
    base_m, base_se, base_sig = load_effect(
        "base-replay-base-jeong-analysis.json", effect
    )
    inst_m, inst_se, inst_sig = load_effect(
        "base-replay-instruct-jeong-analysis.json", effect
    )
    ax.axhline(0, color="#666666", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(2.5, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)
    b1 = ax.bar(
        x - width / 2,
        base_m,
        width,
        yerr=base_se,
        capsize=3.5,
        label="Qwen2.5-1.5B (Base)",
        color="#3b6fb6",
        alpha=0.9,
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        inst_m,
        width,
        yerr=inst_se,
        capsize=3.5,
        label="Qwen2.5-1.5B-Instruct",
        color="#d96b43",
        alpha=0.9,
        zorder=3,
    )
    for bars, sigs, ses in ((b1, base_sig, base_se), (b2, inst_sig, inst_se)):
        for bar, is_sig, se in zip(bars, sigs, ses):
            if is_sig:
                h = bar.get_height()
                offset = se + 0.12 if h >= 0 else -(se + 0.20)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + offset,
                    "*",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
    ax.set_title(title)
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    if show_legend:
        ax.legend(frameon=False, loc="lower left")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    draw_panel(
        axes[0],
        "extraversion",
        "A. Extraversion (E): Preserved Dissociation",
        show_legend=False,
    )
    draw_panel(
        axes[1],
        "neuroticism",
        "B. Neuroticism (N): Selective Post-Training Re-routing",
        show_legend=True,
    )
    axes[0].set_ylabel(r"Slope contrast ($\times 10^{-4}$ cosine / round)")
    axes[0].set_ylim(-2.35, 1.95)

    for panel in axes:
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
