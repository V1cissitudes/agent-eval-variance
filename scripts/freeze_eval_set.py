"""Sample and freeze the HotpotQA (distractor, validation) evaluation set. Run once.

Usage: uv run python scripts/freeze_eval_set.py [--force]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import datasets

from agentrig.eval.freeze import manifest_path_for, sample_ids, sha256_file, write_jsonl

DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"  # HF commit, pinned 2026-10-02
SEED = 20261001
N = 300
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "frozen" / "hotpotqa_distractor_val_300.jsonl"
FIELDS = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite an existing frozen set")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} already exists and is frozen. Use --force only if you mean it.")

    ds = datasets.load_dataset(
        DATASET, CONFIG, split=SPLIT, revision=REVISION, cache_dir=str(ROOT / "data" / "raw")
    )
    by_id = {row["id"]: row for row in ds}
    chosen = sample_ids(by_id.keys(), N, SEED)
    records = [{k: by_id[i][k] for k in FIELDS} for i in chosen]
    write_jsonl(records, OUT)

    manifest = {
        "file": OUT.name,
        "sha256": sha256_file(OUT),
        "n": N,
        "seed": SEED,
        "pool_size": len(by_id),
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "datasets_version": datasets.__version__,
        "sampling": "sort ids by sha256(f'{seed}:{id}'), take first n; order = hash order",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest_path_for(OUT).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {N} questions to {OUT}\nsha256={manifest['sha256']}")


if __name__ == "__main__":
    main()
