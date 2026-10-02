from pathlib import Path

from agentrig.eval.freeze import sample_ids, sha256_file, write_jsonl

IDS = [f"q{i:04d}" for i in range(1000)]


def test_same_seed_same_sample() -> None:
    assert sample_ids(IDS, 50, seed=7) == sample_ids(IDS, 50, seed=7)


def test_input_order_does_not_matter() -> None:
    assert sample_ids(IDS, 50, seed=7) == sample_ids(list(reversed(IDS)), 50, seed=7)


def test_different_seed_different_sample() -> None:
    assert sample_ids(IDS, 50, seed=7) != sample_ids(IDS, 50, seed=8)


def test_jsonl_is_byte_identical(tmp_path: Path) -> None:
    records = [{"id": "a", "x": "é"}, {"id": "b", "x": 1}]
    write_jsonl(records, tmp_path / "a.jsonl")
    write_jsonl(records, tmp_path / "b.jsonl")
    assert sha256_file(tmp_path / "a.jsonl") == sha256_file(tmp_path / "b.jsonl")
