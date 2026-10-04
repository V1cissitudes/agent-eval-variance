"""Supplementary analyses for the report (descriptive, not pre-registered):
1. accuracy and run-to-run consistency by HotpotQA question type (bridge / comparison);
2. process metrics (searches, steps, format errors, share of runs ending at max_steps) by condition;
3. first divergence step of same-seed pairs that differ (which step the two runs part ways);
4. figure: same-seed divergence of the first vs later replicate pairs (from 03's breakdown);
5. format sensitivity of the precision comparison: accuracy difference to BF16 with the agent's
   parser (EM) and if the parser had also accepted an inline "Final Answer:" (em_inline_final, see
   01), paired by question, seed and replicate, with a question-level bootstrap interval.

Input : runs/_derived/runs_flat.csv, data/frozen/*.jsonl, results/03_divergence_by_pair_position.csv
Output: results/07_by_question_type.csv, results/07_process_metrics.csv,
        results/07_first_divergence_step.csv, results/07_precision_format_sensitivity.csv,
        figures/07_first_pair_effect.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agentrig.eval.dataset import load_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs" / "_derived" / "runs_flat.csv"
EVAL_SET = ROOT / "data" / "frozen" / "hotpotqa_distractor_val_300.jsonl"
COND = ["experiment", "model", "precision", "arm", "temperature", "prefix_cache"]


def by_question_type(df: pd.DataFrame) -> pd.DataFrame:
    types = {r["id"]: r["type"] for r in load_frozen(EVAL_SET)}
    df = df.assign(qtype=df["question_id"].map(types))
    per_q = df.groupby([*COND, "qtype", "question_id"]).agg(
        em=("em", "mean"), answers=("answer", "nunique"), actions=("actions", "nunique")
    )
    out = per_q.groupby([*COND, "qtype"]).agg(
        n_questions=("em", "size"),
        accuracy=("em", "mean"),
        share_multi_answer=("answers", lambda x: (x > 1).mean()),
        share_multi_actions=("actions", lambda x: (x > 1).mean()),
    )
    return out.reset_index()


def process_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(COND).agg(
        runs=("em", "size"),
        accuracy=("em", "mean"),
        f1=("f1", "mean"),
        searches=("n_searches", "mean"),
        steps=("n_steps", "mean"),
        format_errors_per_100_runs=("format_errors", lambda x: 100 * x.sum() / len(x)),
        format_error_run_share=("format_errors", lambda x: (x > 0).mean()),
        accuracy_inline_final=("em_inline_final", "mean"),
        no_search_share=("n_searches", lambda x: (x == 0).mean()),
        max_steps_share=("max_steps_reached", "mean"),
        latency_s=("latency_s", "mean"),
    )
    return out.reset_index()


def precision_format_sensitivity(df: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    """Quantized minus BF16 accuracy, strict and with inline final answers accepted (E01a's BF16
    greedy/qwen arms are the BF16 level of E01b, as in 06)."""
    cell = ["arm", "prefix_cache", "question_id", "seed_index", "replicate"]
    d = df[df["experiment"].isin(["E01a", "E01b"]) & df["arm"].isin(["greedy", "qwen"])]
    bf16 = d[d["precision"] == "bf16"].set_index(cell)[["em", "em_inline_final"]]
    rng = np.random.default_rng(20261004)
    rows = []
    for (prec, arm, cache), g in d[d["precision"] != "bf16"].groupby(
        ["precision", "arm", "prefix_cache"]
    ):
        m = g.set_index(cell)[["em", "em_inline_final"]].join(bf16, rsuffix="_bf16", how="inner")
        if m.empty:
            continue
        m = m.assign(
            d_strict=m["em"] - m["em_bf16"],
            d_inline=m["em_inline_final"] - m["em_inline_final_bf16"],
        )
        per_q = m.groupby(level="question_id")[["d_strict", "d_inline"]].mean().to_numpy()
        boot = per_q[rng.integers(0, len(per_q), (reps, len(per_q)))].mean(axis=1)
        rows.append(
            {
                "precision": prec,
                "arm": arm,
                "prefix_cache": cache,
                "matched_cells": len(m),
                "n_questions": len(per_q),
                "acc_bf16": m["em_bf16"].mean(),
                "acc_quantized": m["em"].mean(),
                "diff_strict": m["d_strict"].mean(),
                "diff_strict_ci_low": np.quantile(boot[:, 0], 0.025),
                "diff_strict_ci_high": np.quantile(boot[:, 0], 0.975),
                "acc_bf16_inline": m["em_inline_final_bf16"].mean(),
                "acc_quantized_inline": m["em_inline_final"].mean(),
                "diff_inline": m["d_inline"].mean(),
                "diff_inline_ci_low": np.quantile(boot[:, 1], 0.025),
                "diff_inline_ci_high": np.quantile(boot[:, 1], 0.975),
            }
        )
    return pd.DataFrame(rows)


def first_divergence(a: str, b: str) -> int:
    """1-based index of the first step where two action signatures differ (0 if identical)."""
    sa, sb = a.split(" > "), b.split(" > ")
    for i, (x, y) in enumerate(zip(sa, sb, strict=False), start=1):
        if x != y:
            return i
    return 0 if len(sa) == len(sb) else min(len(sa), len(sb)) + 1


def divergence_steps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby([*COND, "question_id", "seed_index"]):
        if len(g) < 2:
            continue
        step = first_divergence(g["actions"].iloc[0], g["actions"].iloc[1])
        if step:
            rows.append(
                {
                    **dict(zip([*COND, "question_id", "seed_index"], key, strict=True)),
                    "first_divergence_step": step,
                }
            )
    if not rows:
        return pd.DataFrame()
    steps = pd.DataFrame(rows)
    return steps.groupby([*COND, "first_divergence_step"]).size().rename("pairs").reset_index()


def plot_first_pair(path: Path) -> None:
    d = pd.read_csv(path)
    d = d[d["prefix_cache"] == "on"].copy()
    keys = ["experiment", "precision", "arm", "temperature"]
    d["label"] = (
        d["experiment"]
        + " "
        + d["precision"].astype(str)
        + "\n"
        + d["arm"].astype(str)
        + " T="
        + d["temperature"].astype(str)
    )
    labels = list(dict.fromkeys(d["label"]))
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2, 3.8))
    width = 0.38
    for j, (pos, color) in enumerate([("first", "#c0504d"), ("later", "#4f81bd")]):
        sub = d[d["pair_position"] == pos]
        if sub.duplicated(keys).any():
            raise SystemExit(
                "07_supplementary: duplicate condition rows in the first-pair breakdown"
            )
        sub = sub.set_index("label").reindex(labels)
        x = [i + (j - 0.5) * width for i in range(len(labels))]
        err = [sub["rate"] - sub["ci_low"], sub["ci_high"] - sub["rate"]]
        ax.bar(
            x, sub["rate"], width, yerr=err, capsize=3, color=color, label=f"{pos} replicate pair"
        )
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_ylabel("same-seed groups that differ\n(prefix cache on, 95% CI)")
    ax.set_title("Cache-induced divergence concentrates in the first (cache miss vs hit) pair")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / "07_first_pair_effect.png", dpi=150)
    plt.close()


def main() -> None:
    if not INPUT.exists():
        print("07_supplementary: no derived runs, skipping")
        return
    df = pd.read_csv(INPUT, keep_default_na=False)
    results = ROOT / "results"
    by_question_type(df).to_csv(results / "07_by_question_type.csv", index=False)
    process_metrics(df).to_csv(results / "07_process_metrics.csv", index=False)
    sensitivity = precision_format_sensitivity(df)
    if not sensitivity.empty:
        sensitivity.to_csv(results / "07_precision_format_sensitivity.csv", index=False)
    steps = divergence_steps(df)
    if not steps.empty:
        steps.to_csv(results / "07_first_divergence_step.csv", index=False)
    breakdown = results / "03_divergence_by_pair_position.csv"
    if breakdown.exists():
        plot_first_pair(breakdown)
    print("07_supplementary: question type, process metrics, divergence steps written")


if __name__ == "__main__":
    main()
