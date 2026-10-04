"""Same-seed divergence (E01 hypotheses H1/H3): how often two runs with the same seed and
settings differ, per experiment x model x arm x cache. Writes results/03_same_seed_divergence.csv.
Reads runs/_derived/runs_flat.csv from 01_run_summary.py."""

from __future__ import annotations

from math import exp, lgamma, log, log1p
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs" / "_derived" / "runs_flat.csv"
KEYS = [
    "experiment",
    "endpoint",
    "model",
    "precision",
    "arm",
    "temperature",
    "thinking",
    "prefix_cache",
]


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), in log space so large n cannot overflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = log(p), log1p(-p)
    terms = [
        lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(k + 1)
    ]
    m = max(terms)
    return min(1.0, exp(m) * sum(exp(t - m) for t in terms))


def solve_decreasing(f, target: float) -> float:
    """p in [0, 1] with f(p) = target, for f monotone decreasing in p (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval (no scipy dependency)."""
    lower = 0.0 if k == 0 else solve_decreasing(lambda p: binom_cdf(k - 1, n, p), 1 - alpha / 2)
    upper = 1.0 if k == n else solve_decreasing(lambda p: binom_cdf(k, n, p), alpha / 2)
    return lower, upper


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, question, seed) with at least two replicates."""
    grouped = df.groupby([*KEYS, "question_id", "seed_index"])
    pairs = grouped.agg(
        n=("answer", "size"),
        answers=("answer", "nunique"),
        actions=("actions", "nunique"),
    ).reset_index()
    pairs = pairs[pairs["n"] >= 2]
    pairs["answer_differs"] = pairs["answers"] > 1
    pairs["any_differs"] = (pairs["answers"] > 1) | (pairs["actions"] > 1)
    return pairs


def summarize(pairs: pd.DataFrame) -> pd.DataFrame:
    return summarize_by(pairs, KEYS)


def summarize_by(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in pairs.groupby(keys):
        n, k = len(g), int(g["any_differs"].sum())
        lo, hi = clopper_pearson(k, n)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "seed_groups": n,
                "differing": k,
                "rate": k / n,
                "ci_low": lo,
                "ci_high": hi,
                "answer_differing": int(g["answer_differs"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not INPUT.exists():
        print("03_same_seed_divergence: no derived runs, skipping")
        return
    df = pd.read_csv(INPUT, keep_default_na=False)
    pairs = pair_table(df)
    if pairs.empty:
        print("03_same_seed_divergence: no same-seed replicates, skipping")
        return
    out = summarize(pairs)
    out.to_csv(ROOT / "results" / "03_same_seed_divergence.csv", index=False)
    # Exploratory (not pre-registered, found in E01a): with prefix caching the first replicate
    # pair of a question in an arm (seed_index 0: cache miss vs hit) diverges far more often.
    pairs["pair_position"] = pairs["seed_index"].map(lambda i: "first" if i == 0 else "later")
    by_order = summarize_by(pairs, [*KEYS, "pair_position"])
    by_order.to_csv(ROOT / "results" / "03_divergence_by_pair_position.csv", index=False)
    print(f"03_same_seed_divergence: {len(out)} conditions, {len(pairs)} same-seed groups")


if __name__ == "__main__":
    main()
