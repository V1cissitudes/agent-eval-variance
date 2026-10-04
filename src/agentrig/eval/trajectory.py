"""Run records and JSONL storage. One line per run (question x condition x repeat), with all
steps nested; files live under runs/ (gitignored) and are synced with scripts/sync_runs.sh."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentrig.agent.loop import AgentResult


@dataclass
class RunRecord:
    run_id: str
    experiment: str
    question_id: str
    repeat: int
    condition: dict[str, Any]  # endpoint, platform, model, precision, sampling, thinking, cache
    meta: dict[str, Any]  # git commit, prompt version, timestamps, package version
    result: dict[str, Any]  # AgentResult as a dict
    outcome: dict[str, Any]  # final answer, EM/F1, termination, step/search/token counts
    error: str | None = None


def summarize(result: AgentResult, gold: str, em: float, f1: float) -> dict[str, Any]:
    steps = result.steps
    return {
        "final_answer": result.final_answer,
        "gold": gold,
        "em": em,
        "f1": f1,
        "termination": result.termination,
        "n_steps": len(steps),
        "n_searches": sum(s.parsed["kind"] == "action" for s in steps),
        "n_format_errors": sum(s.parsed["kind"] == "error" for s in steps),
        "n_truncated": sum(s.finish_reason == "length" for s in steps),
        "max_prompt_tokens": max((s.prompt_tokens for s in steps), default=0),
        # context actually needed by the longest step (prompt + generated, incl. thinking)
        "max_step_tokens": max((s.prompt_tokens + s.completion_tokens for s in steps), default=0),
        "total_completion_tokens": sum(s.completion_tokens for s in steps),
        "total_latency_s": round(sum(s.latency_s for s in steps), 4),
    }


def append_jsonl(path: Path, record: RunRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def completed_run_ids(path: Path) -> set[str]:
    """Run ids already saved without an error, so an interrupted experiment can resume."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a crash or shutdown; that run is simply re-run
            if rec.get("error") is None:
                done.add(rec["run_id"])
    return done
