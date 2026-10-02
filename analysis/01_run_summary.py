"""Descriptive summary of every experiment under runs/: accuracy and run-to-run consistency per
question x condition. Writes results/01_run_summary.csv, one figure per experiment, and a flat
per-run table runs/_derived/runs_flat.csv (gitignored) that the R variance-component analysis reads.
Skips quietly when runs/ is empty (runs/ is not in git and lives mainly on the GPU machine)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agentrig.eval.metrics import normalize_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def action_signature(steps: list[dict]) -> str:
    """Sequence of (action, argument) a run took; comparable to 2602.11619's distinct sequences."""
    parts = []
    for st in steps:
        p = st["parsed"]
        if p["kind"] == "action":
            parts.append(f"{p['tool']}[{(p['argument'] or '').strip().lower()}]")
        else:
            parts.append(p["kind"])
    return " > ".join(parts)


def load_runs(include_dev: bool = False) -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "runs").glob("*/*.jsonl")):
        if path.parent.name.startswith("dev_") and not include_dev:  # development checks
            continue
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("error"):
                continue
            c, o = r["condition"], r["outcome"]
            rows.append(
                {
                    "experiment": r["experiment"],
                    "question_id": r["question_id"],
                    "repeat": r["repeat"],
                    "seed_index": r["meta"].get("seed_index", r["repeat"]),
                    "replicate": r["meta"].get("replicate", 0),
                    "endpoint": c["endpoint"],
                    "temperature": c["temperature"],
                    "thinking": c["enable_thinking"],
                    "prefix_cache": c["prefix_cache"],
                    "answer": normalize_answer(o["final_answer"] or ""),
                    "em": o["em"],
                    "f1": o["f1"],
                    "actions": action_signature(r["result"]["steps"]),
                    "n_steps": o["n_steps"],
                    "n_searches": o["n_searches"],
                    "format_errors": o["n_format_errors"],
                    "latency_s": o["total_latency_s"],
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "endpoint", "temperature", "thinking", "prefix_cache", "question_id"]

    def agg(g: pd.DataFrame) -> pd.Series:
        modal_share = g["answer"].value_counts(normalize=True).iloc[0]
        return pd.Series(
            {
                "n_runs": len(g),
                "em_mean": g["em"].mean(),
                "f1_mean": g["f1"].mean(),
                "f1_sd": g["f1"].std(ddof=1),
                "distinct_answers": g["answer"].nunique(),
                "modal_answer_share": modal_share,
                "distinct_action_seqs": g["actions"].nunique(),
                "searches_mean": g["n_searches"].mean(),
                "format_errors": g["format_errors"].sum(),
                "latency_mean_s": g["latency_s"].mean(),
            }
        )

    return df.groupby(keys).apply(agg, include_groups=False).reset_index()


def plot(summary: pd.DataFrame, experiment: str) -> None:
    s = summary[summary["experiment"] == experiment].copy()
    s["condition"] = s.apply(lambda r: f"T={r.temperature}\ncache={r.prefix_cache}", axis=1)
    conds = sorted(s["condition"].unique())
    data = [s.loc[s["condition"] == c, "modal_answer_share"] for c in conds]
    fig, ax = plt.subplots(figsize=(1.8 * len(conds) + 2, 3.5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(conds) + 1), conds)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of runs giving the modal answer\n(one point per question)")
    ax.set_title(f"{experiment}: run-to-run answer consistency")
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / f"01_{experiment}_consistency.png", dpi=150)
    plt.close()


def main() -> None:
    df = load_runs()
    if df.empty:
        (ROOT / "runs" / "_derived" / "runs_flat.csv").unlink(
            missing_ok=True
        )  # no stale input for 02
        print("01_run_summary: no experiment runs found, skipping")
        return
    summary = summarize(df)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "results" / "01_run_summary.csv", index=False)
    derived = ROOT / "runs" / "_derived"
    derived.mkdir(parents=True, exist_ok=True)
    df.to_csv(derived / "runs_flat.csv", index=False)
    for experiment in summary["experiment"].unique():
        plot(summary, experiment)
    print(f"01_run_summary: {len(df)} runs -> {len(summary)} question x condition rows")


if __name__ == "__main__":
    main()
