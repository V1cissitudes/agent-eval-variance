"""Exploratory analysis helpers: the inline-final-answer counterfactual (01) and the cross-launch
comparison (08)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, ROOT / "analysis" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summary = _load("01_run_summary")
cross = _load("08_cross_launch")


def step(content: str, kind: str) -> dict:
    return {"content": content, "parsed": {"kind": kind}}


def test_inline_final_rescores_the_first_format_error() -> None:
    steps = [
        step("Thought: look\nAction: search[x]", "action"),
        step("Thought: it was 2008. Final Answer: Paris.", "error"),
        step("Final Answer: London", "final"),
    ]
    assert summary.inline_final_em(steps, "Paris", em=0.0) == 1.0


def test_inline_final_keeps_em_without_format_errors() -> None:
    steps = [step("Final Answer: London", "final")]
    assert summary.inline_final_em(steps, "Paris", em=0.0) == 0.0


def test_inline_action_first_keeps_observed_em() -> None:
    steps = [step("Thought: check. Action: search[y] then Final Answer: Paris", "error")]
    assert summary.inline_final_em(steps, "Paris", em=0.0) == 0.0


def test_only_the_first_format_error_counts() -> None:
    steps = [
        step("I am not sure yet.", "error"),
        step("Thought: so Final Answer: Paris", "error"),
    ]
    assert summary.inline_final_em(steps, "Paris", em=0.0) == 0.0


def test_inline_final_strips_markdown() -> None:
    steps = [step("Thought: done **Final Answer:** Paris", "error")]
    assert summary.inline_final_em(steps, "Paris", em=0.0) == 1.0


def test_compare_counts_each_level() -> None:
    a = {"outputs": ["s1", "s2"], "actions": "search[x] > final", "answer": "paris"}
    same_text = dict(a)
    later_text = {**a, "outputs": ["s1", "s2 changed"]}
    new_answer = {"outputs": ["t1"], "actions": "final", "answer": "london"}
    stats = cross.compare([(a, same_text), (a, later_text), (a, new_answer)])
    assert stats == {
        "pairs": 3,
        "outputs_differ": 2,
        "actions_differ": 1,
        "answers_differ": 1,
        "first_step_differs": 1,
    }


def test_load_keys_and_skips_bad_lines(tmp_path: Path) -> None:
    record = {
        "question_id": "q1",
        "repeat": 1,
        "condition": {"temperature": 0.0},
        "meta": {"seed_index": 0, "replicate": 1},
        "result": {"steps": [{"content": "Final Answer: Paris", "parsed": {"kind": "final"}}]},
        "outcome": {"final_answer": "Paris"},
        "error": None,
    }
    path = tmp_path / "runs.jsonl"
    path.write_text(json.dumps(record) + "\n" + '{"truncated', encoding="utf-8")
    runs = cross.load(path)
    assert list(runs) == [("T=0.0", "q1", 0, 1)]
    assert runs[("T=0.0", "q1", 0, 1)]["answer"] == "paris"
