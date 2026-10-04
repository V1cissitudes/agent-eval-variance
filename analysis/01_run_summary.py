"""Descriptive summary of every experiment under runs/: accuracy and run-to-run consistency per
question x condition. Writes results/01_run_summary.csv, one figure per experiment, and a flat
per-run table runs/_derived/runs_flat.csv (gitignored) that the R variance-component analysis reads.
Skips quietly when runs/ is empty (runs/ is not in git and lives mainly on the GPU machine)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agentrig.eval.metrics import exact_match, normalize_answer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# Unanchored versions of the parser's patterns (src/agentrig/agent/parsing.py), for the sensitivity
# analysis below only; the agent itself requires "Final Answer:" / "Action:" at the start of a line.
_INLINE_FINAL = re.compile(r"Final\s+Answer\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_INLINE_ACTION = re.compile(r"Action\s*:\s*\w+\s*\[", re.IGNORECASE)


def inline_final_em(steps: list[dict], gold: str, em: float) -> float:
    """EM if the parser had also accepted a "Final Answer:" in the middle of a line (exploratory).
    Exact counterfactual: the episode is unchanged up to its first format error, and a lenient
    parser would have ended it there. If that step has an inline Action first, the episode would
    have continued differently, so the observed EM is kept."""
    for st in steps:
        if st["parsed"]["kind"] != "error":
            continue
        text = st["content"].replace("**", "").replace("__", "")
        final, action = _INLINE_FINAL.search(text), _INLINE_ACTION.search(text)
        if final and final.group(1) and (not action or final.start() < action.start()):
            return exact_match(final.group(1), gold)
        return em
    return em


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
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # truncated last line after a crash or shutdown
                print(f"01_run_summary: skipped an unreadable line in {path.name}")
                continue
            if r.get("error"):
                continue
            c, o = r["condition"], r["outcome"]
            rows.append(
                {
                    "run_id": r["run_id"],
                    "run_seed": r["meta"].get("run_seed"),
                    "experiment": r["experiment"],
                    "question_id": r["question_id"],
                    "repeat": r["repeat"],
                    "seed_index": r["meta"].get("seed_index", r["repeat"]),
                    "replicate": r["meta"].get("replicate", 0),
                    "endpoint": c["endpoint"],
                    "temperature": c["temperature"],
                    "thinking": c["enable_thinking"],
                    "prefix_cache": c["prefix_cache"],
                    "arm": c.get("arm", "default"),
                    "model": c.get("model", ""),
                    "precision": c.get("precision", ""),
                    "answer": normalize_answer(o["final_answer"] or ""),
                    "em": o["em"],
                    "em_inline_final": inline_final_em(r["result"]["steps"], o["gold"], o["em"]),
                    "f1": o["f1"],
                    "actions": action_signature(r["result"]["steps"]),
                    "n_steps": o["n_steps"],
                    "n_searches": o["n_searches"],
                    "format_errors": o["n_format_errors"],
                    "max_steps_reached": o["termination"] == "max_steps",
                    "latency_s": o["total_latency_s"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A resumed run can append a second successful record for the same run_id; keep the last one.
    duplicates = int(df.duplicated("run_id").sum())
    if duplicates:
        print(f"01_run_summary: dropped {duplicates} duplicate run_id records (kept the last)")
        df = df.drop_duplicates("run_id", keep="last")
    check_condition_identity(df)
    return df


def check_condition_identity(df: pd.DataFrame) -> None:
    """Downstream scripts group by a subset of the condition fields; make sure the omitted fields
    are constant within each experiment and that every precision maps to one checkpoint."""
    for exp, g in df.groupby("experiment"):
        for field in ("endpoint", "thinking"):
            if g[field].nunique() > 1:
                raise SystemExit(
                    f"{exp}: '{field}' varies within the experiment; analyses would mix it"
                )
        per_precision = g.groupby("precision")["model"].nunique()
        if (per_precision > 1).any():
            raise SystemExit(
                f"{exp}: a precision maps to several models: {per_precision.to_dict()}"
            )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "endpoint", "model", "precision", "arm", "temperature", "thinking"]
    keys += ["prefix_cache", "question_id"]

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
    s["condition"] = s.apply(
        lambda r: f"{r.precision} {r.arm}\nT={r.temperature}\ncache={r.prefix_cache}", axis=1
    )
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
