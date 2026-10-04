"""Figures for the technical report (in preparation), drawn from results/ only:

1. Same-seed divergence with prefix caching on, first vs later replicate pairs, by precision and arm
   (from 03_divergence_by_pair_position.csv).
2. Power to detect 3 pp vs number of questions, for 1 / 3 / 10 independent runs per question:
   a similar-condition comparison (cache off vs on) next to BF16 vs FP8 and BF16 vs INT4
   (from 04_dstudy_pairs.csv; independent seeds; question-level bootstrap bands).
3. Byte-level divergence of greedy first runs: within one cache-on server (replicate vs replicate)
   vs the same request on the cache-off and the cache-on server, by precision
   (from 08_cross_cache_divergence.csv).

Colours: categorical slots 1-2 (blue, orange) and an ordinal blue ramp for the number of runs,
validated with the dataviz palette checker (light mode, white surface).

Output: figures/paper_fig{1,2,3}_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RUNS_RAMP = {1: "#86b6ef", 3: "#2a78d6", 10: "#104281"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PRECISION = {"bf16": "BF16", "fp8_w8a8_block128": "FP8", "int4_awq_w4a16_g128": "INT4"}
ARM = {"greedy": "greedy", "qwen": "Qwen-rec.", "topp1": "temp.-only"}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 9,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
    }
)


def style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_first_pair() -> None:
    d = pd.read_csv(RESULTS / "03_divergence_by_pair_position.csv")
    d = d[d["experiment"].isin(["E01a", "E01b"]) & (d["prefix_cache"] == "on")]
    groups = [
        (p, a) for p in PRECISION for a in ARM if ((d["precision"] == p) & (d["arm"] == a)).any()
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    width = 0.36
    for j, (position, color, label) in enumerate(
        [("first", BLUE, "first pair (cache miss vs hit)"), ("later", ORANGE, "later pairs")]
    ):
        labelled = False
        for i, (p, a) in enumerate(groups):
            row = d[(d["precision"] == p) & (d["arm"] == a) & (d["pair_position"] == position)]
            if row.empty:
                continue
            r = row.iloc[0]
            x = i + (j - 0.5) * width
            ax.bar(x, 100 * r["rate"], width * 0.92, color=color, label=None if labelled else label)
            labelled = True
            ax.errorbar(
                x,
                100 * r["rate"],
                yerr=[[100 * (r["rate"] - r["ci_low"])], [100 * (r["ci_high"] - r["rate"])]],
                color=INK_2,
                linewidth=0.8,
                capsize=2,
            )
    ax.set_xticks(range(len(groups)), [f"{PRECISION[p]}\n{ARM[a]}" for p, a in groups])
    ax.set_ylabel("same-seed groups that differ (%)")
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_fig1_first_pair.png")
    plt.close(fig)


def fig_power() -> None:
    d = pd.read_csv(RESULTS / "04_dstudy_pairs.csv")
    panels = [
        ("E01a", "bf16|qwen|0.7|on|False", "(a) cache off vs on, BF16"),
        ("E01_precision", "fp8_w8a8_block128|qwen|0.7|off|False", "(b) BF16 vs FP8"),
        ("E01_precision", "int4_awq_w4a16_g128|qwen|0.7|off|False", "(c) BF16 vs INT4"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, (exp, other, title) in zip(axes, panels, strict=True):
        g = d[
            (d["experiment"] == exp)
            & (d["condition_1"] == "bf16|qwen|0.7|off|False")
            & (d["condition_2"] == other)
        ]
        for n_r, color in RUNS_RAMP.items():
            s = g[g["n_r"] == n_r].sort_values("n_q")
            ax.fill_between(
                s["n_q"],
                s["power_unpaired_lo"],
                s["power_unpaired_hi"],
                color=color,
                alpha=0.15,
                linewidth=0,
            )
            ax.plot(
                s["n_q"],
                s["power_unpaired"],
                color=color,
                linewidth=2,
                label=str(n_r),
            )
        ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 300, 1000], ["50", "100", "300", "1000"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left")
        ax.set_xlabel("questions")
        style(ax)
    axes[0].set_ylabel("power to detect 3 pp")
    axes[2].legend(
        frameon=False,
        loc="upper left",
        labelcolor=INK_2,
        title="runs per question",
        title_fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_fig2_power.png")
    plt.close(fig)


def fig_cross_cache() -> None:
    d = pd.read_csv(RESULTS / "08_cross_cache_divergence.csv")
    d = d[(d["arm"] == "greedy") & (d["position"] == "first")]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    width = 0.36
    for j, (comparison, color, label) in enumerate(
        [
            ("within cache-on server", BLUE, "within one cache-on server"),
            ("cache off vs on", ORANGE, "cache-off vs cache-on server"),
        ]
    ):
        for i, p in enumerate(PRECISION):
            r = d[(d["precision"] == p) & (d["comparison"] == comparison)].iloc[0]
            share = 100 * r["outputs_differ"] / r["pairs"]
            x = i + (j - 0.5) * width
            ax.bar(x, share, width * 0.92, color=color, label=label if i == 0 else None)
            ax.text(
                x, share + 1.5, f"{share:.0f}%", ha="center", va="bottom", color=INK_2, fontsize=8
            )
    ax.set_xticks(range(len(PRECISION)), list(PRECISION.values()))
    ax.set_ylabel("greedy first runs whose\noutputs differ (%)")
    ax.set_ylim(0, 90)
    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        labelcolor=INK_2,
        fontsize=8,
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "paper_fig3_cross_cache.png")
    plt.close(fig)


def main() -> None:
    needed = [
        "03_divergence_by_pair_position.csv",
        "04_dstudy_pairs.csv",
        "08_cross_cache_divergence.csv",
    ]
    if not all((RESULTS / f).exists() for f in needed):
        print("09_paper_figures: results missing, skipping")
        return
    fig_first_pair()
    fig_power()
    fig_cross_cache()
    print("09_paper_figures: paper_fig1-3 written")


if __name__ == "__main__":
    main()
