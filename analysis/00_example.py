"""Placeholder showing the analysis convention; delete once real analysis scripts exist.

Reads the frozen eval set and writes a question-type count table and bar chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agentrig.eval.dataset import load_frozen  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "data" / "frozen" / "hotpotqa_distractor_val_300.jsonl"


def main() -> None:
    df = pd.DataFrame(load_frozen(EVAL_SET))
    counts = df["type"].value_counts().rename_axis("type").reset_index(name="n")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    counts.to_csv(ROOT / "results" / "00_eval_set_types.csv", index=False)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts["type"], counts["n"])
    ax.set_ylabel("questions")
    ax.set_title("Frozen eval set: question types")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "00_eval_set_types.png", dpi=150)


if __name__ == "__main__":
    main()
