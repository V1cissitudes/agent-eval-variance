"""Connectivity check: send one message to the configured endpoint and print the reply.

Usage: uv run python scripts/check_endpoint.py [--endpoint NAME] [--model qwen3-4b] [--thinking]
                                               [--max-tokens N] [--save-raw PATH]
Also a regression check for the thinking switch. Exits non-zero if the reply is truncated or
empty, if thinking output appears while thinking is off, or if none appears while it is on.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agentrig.config import load_endpoint, load_model
from agentrig.llm.client import extract_reasoning, make_client, thinking_kwargs

PROMPT = "Reply with one short sentence: what is the capital of France?"


def find_problems(enable_thinking: bool, content: str, reasoning: str, finish: str) -> list[str]:
    problems = []
    if finish != "stop":
        problems.append(f"finish_reason={finish!r} (truncated or abnormal)")
    if not content.strip():
        problems.append("empty final answer")
    thought = bool(reasoning.strip()) or "<think>" in content
    if enable_thinking and not thought:
        problems.append("thinking is on, but no thinking output was found")
    if not enable_thinking and thought:
        problems.append("thinking is off, but thinking output was found")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--thinking", action="store_true", help="turn thinking mode on")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--save-raw", type=Path, default=None, help="write request+response JSON")
    args = parser.parse_args()

    endpoint = load_endpoint(args.endpoint)
    model_cfg = load_model(args.model)
    enable_thinking = args.thinking or model_cfg.get("enable_thinking", False)
    request = {
        "model": model_cfg["served_names"][endpoint.name],
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        **thinking_kwargs(endpoint, enable_thinking),
    }

    start = time.perf_counter()
    resp = make_client(endpoint).chat.completions.create(**request)
    elapsed = time.perf_counter() - start
    choice = resp.choices[0]
    content = choice.message.content or ""
    reasoning = extract_reasoning(choice.message)

    print(f"endpoint={endpoint.name} ({endpoint.platform}) model={request['model']}")
    print(f"enable_thinking={enable_thinking} finish_reason={choice.finish_reason}")
    print(f"latency={elapsed:.2f}s usage={resp.usage}")
    print(f"reply: {content!r}")
    print(f"reasoning: {len(reasoning)} chars | '<think>' in content: {'<think>' in content}")
    if args.save_raw:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        raw = {"request": request, "response": resp.model_dump()}
        args.save_raw.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"raw request/response saved to {args.save_raw}")

    problems = find_problems(enable_thinking, content, reasoning, choice.finish_reason)
    if problems:
        raise SystemExit("FAIL: " + "; ".join(problems))
    print("PASS")


if __name__ == "__main__":
    main()
