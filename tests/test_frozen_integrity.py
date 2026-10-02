from pathlib import Path

from agentrig.eval.dataset import load_frozen

EVAL_SET = Path(__file__).resolve().parents[1] / "data/frozen/hotpotqa_distractor_val_300.jsonl"


def test_frozen_set_matches_manifest() -> None:
    records = load_frozen(EVAL_SET)
    assert len(records) == 300
    assert len({r["id"] for r in records}) == 300
