"""Repeated-trial runner: every question x condition x repeat runs the agent once and is saved as
one JSONL line. Re-running the same command resumes and skips runs that already succeeded."""

from __future__ import annotations

import hashlib
import itertools
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentrig import __version__
from agentrig.agent.loop import AgentResult, run_agent
from agentrig.agent.prompts import PROMPT_VERSION, STOP_SEQUENCES
from agentrig.config import CONFIG_DIR, Endpoint, load_endpoint, load_model, load_yaml
from agentrig.eval.dataset import load_frozen
from agentrig.eval.metrics import exact_match, f1_score
from agentrig.eval.trajectory import RunRecord, append_jsonl, completed_run_ids, summarize
from agentrig.llm.client import Sampling, build_request, chat, make_client
from agentrig.tools.search import ParagraphSearch

ROOT = CONFIG_DIR.parent


def derive_seed(*parts: object) -> int:
    """Stable 31-bit seed from any parts (same inputs -> same seed on every machine)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def expand_conditions(cfg: dict[str, Any], prefix_cache: str) -> list[dict[str, Any]]:
    """Cross temperatures x thinking settings; prefix cache is fixed per server run."""
    temps = cfg.get("temperatures", [0.0])
    thinking = cfg.get("enable_thinking", False)
    thinking_list = thinking if isinstance(thinking, list) else [thinking]
    return [
        {"temperature": t, "enable_thinking": th, "prefix_cache": prefix_cache}
        for t, th in itertools.product(temps, thinking_list)
    ]


def condition_key(cond: dict[str, Any]) -> str:
    think = "on" if cond["enable_thinking"] else "off"
    return f"T{cond['temperature']}_think-{think}_cache-{cond['prefix_cache']}"


def select_questions(cfg: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    records = load_frozen(ROOT / cfg["eval_set"])
    ids = cfg.get("question_ids") or []
    if ids:
        by_id = {r["id"]: r for r in records}
        records = [by_id[i] for i in ids]
    else:
        records = records[: cfg.get("n_questions", len(records))]
    return records[:limit] if limit else records


def run_one(
    question: dict[str, Any],
    cond: dict[str, Any],
    run_seed: int,
    ctx: dict[str, Any],
) -> AgentResult:
    endpoint: Endpoint = ctx["endpoint"]
    tool = ParagraphSearch(question["context"], top_k=ctx["cfg"].get("search_top_k", 2))

    def llm(messages: list[dict[str, str]], step: int):  # noqa: ANN202
        sampling = Sampling(
            temperature=cond["temperature"],
            top_p=ctx["cfg"].get("top_p", 1.0),
            top_k=ctx["cfg"].get("top_k"),
            max_tokens=ctx["cfg"].get("max_tokens_per_step", 1024),
            seed=run_seed + step,
        )
        request = build_request(
            endpoint, ctx["served"], messages, sampling, cond["enable_thinking"], STOP_SEQUENCES
        )
        return chat(ctx["client"], request)

    return run_agent(question["question"], {tool.name: tool}, llm, ctx["cfg"]["max_steps"])


def build_condition_record(cond: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
    cfg, endpoint = ctx["cfg"], ctx["endpoint"]
    return {
        **cond,
        "endpoint": endpoint.name,
        "platform": endpoint.platform,
        "model": cfg["model"],
        "served_model": ctx["served"],
        "precision": ctx["model_cfg"].get("precision", {}).get(endpoint.name),
        "top_p": cfg.get("top_p", 1.0),
        "top_k": cfg.get("top_k"),
        "max_tokens_per_step": cfg.get("max_tokens_per_step", 1024),
        "max_steps": cfg["max_steps"],
        "search_top_k": cfg.get("search_top_k", 2),
    }


def run_experiment(
    config_path: Path,
    prefix_cache: str,
    endpoint_name: str | None = None,
    limit: int | None = None,
) -> Path:
    cfg = load_yaml(config_path)
    endpoint = load_endpoint(endpoint_name or cfg["endpoint"])
    model_cfg = load_model(cfg["model"])
    ctx = {
        "cfg": cfg,
        "endpoint": endpoint,
        "model_cfg": model_cfg,
        "served": model_cfg["served_names"][endpoint.name],
        "client": make_client(endpoint),
    }
    out = ROOT / "runs" / cfg["name"] / f"{endpoint.name}__cache-{prefix_cache}.jsonl"
    done = completed_run_ids(out)
    meta_base = {"git": git_state(), "prompt_version": PROMPT_VERSION, "agentrig": __version__}
    questions = select_questions(cfg, limit)
    for cond in expand_conditions(cfg, prefix_cache):
        cond_record = build_condition_record(cond, ctx)
        key = condition_key(cond)
        for q, repeat in itertools.product(questions, range(cfg["n_repeats"])):
            run_id = f"{cfg['name']}/{endpoint.name}/{key}/{q['id']}/r{repeat:02d}"
            if run_id in done:
                continue
            # replicates_per_seed > 1 reuses a seed: differences between same-seed replicates
            # isolate serving-side nondeterminism from sampling variance (analysis_plan.md)
            seed_index, replicate = divmod(repeat, cfg.get("replicates_per_seed", 1))
            run_seed = derive_seed(cfg["seed"], q["id"], seed_index)
            meta = {**meta_base, "seed_index": seed_index, "replicate": replicate}
            record = execute(run_id, q, repeat, cond, cond_record, run_seed, ctx, meta)
            append_jsonl(out, record)
            print(f"{run_id}  em={record.outcome.get('em')}  err={record.error is not None}")
    return out


def execute(
    run_id: str,
    q: dict[str, Any],
    repeat: int,
    cond: dict[str, Any],
    cond_record: dict[str, Any],
    run_seed: int,
    ctx: dict[str, Any],
    meta_base: dict[str, Any],
) -> RunRecord:
    """Run one trial; an exception is saved as an error record instead of stopping everything."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    meta = {**meta_base, "started_at": started, "run_seed": run_seed}
    record = RunRecord(run_id, ctx["cfg"]["name"], q["id"], repeat, cond_record, meta, {}, {})
    try:
        result = run_one(q, cond, run_seed, ctx)
    except Exception as exc:  # noqa: BLE001 - keep the experiment running, log the failure
        record.error = f"{type(exc).__name__}: {exc}"
        return record
    em, f1 = (
        exact_match(result.final_answer, q["answer"]),
        f1_score(result.final_answer, q["answer"]),
    )
    record.result = asdict(result)
    record.outcome = summarize(result, q["answer"], em, f1)
    record.meta["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return record
