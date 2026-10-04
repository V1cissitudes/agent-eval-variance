"""Exploratory (not pre-registered), 2026-10-04: divergence *between* serving configurations and
launches, as opposed to 03's divergence between same-seed replicates *within* one server.

1. Cache off vs on: the same request (same question, arm and seed; replicate 0) on the cache-off and
   the cache-on server of the same precision. "first" = seed 0, the question's first run in the arm;
   "later" = seeds 1-2, which follow runs of the same question and so reuse its cached prompt. For
   contrast on the same byte-level scale, replicate 0 vs 1 of the same seed within the cache-on
   server ("within cache-on server"; 03 reports the answer-or-action version).
2. Launch reproducibility: the same configuration on separate server launches, run by run.
   BF16 greedy E00 vs E01a (T = 0, so the seed does not matter; see E00_VS_E01A for which of these
   are separate launches), and the dev_E01b_launch check (first 50 greedy questions per precision,
   re-run on a fresh server) when its runs exist.

Three levels: any model output differs (byte-level, any step), action sequence differs, normalized
final answer differs.

Input : runs/{E00_pilot,E01a,E01b,dev_E01b_launch}/*.jsonl
Output: results/08_cross_cache_divergence.csv, results/08_launch_reproducibility.csv
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from agentrig.eval.metrics import normalize_answer

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MAIN = {  # precision -> file pattern of the main runs ({} = cache setting)
    "bf16": "E01a/vllm_5070__qwen3-4b__cache-{}.jsonl",
    "fp8_w8a8_block128": "E01b/vllm_5070__qwen3-4b-fp8__cache-{}.jsonl",
    "int4_awq_w4a16_g128": "E01b/vllm_5070__qwen3-4b-awq__cache-{}.jsonl",
}
_spec = importlib.util.spec_from_file_location("summary", ROOT / "analysis" / "01_run_summary.py")
_run_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_summary)
action_signature = _run_summary.action_signature


def load(path: Path) -> dict[tuple, dict]:
    """(arm, question_id, seed_index, replicate) -> outputs, actions and answer of one run."""
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            continue
        steps = r["result"]["steps"]
        key = (
            r["condition"].get("arm", f"T={r['condition']['temperature']}"),
            r["question_id"],
            r["meta"].get("seed_index", r["repeat"]),
            r["meta"].get("replicate", 0),
        )
        out[key] = {
            "outputs": [s["content"] for s in steps],
            "actions": action_signature(steps),
            "answer": normalize_answer(r["outcome"]["final_answer"] or ""),
        }
    return out


def compare(pairs: list[tuple[dict, dict]]) -> dict:
    n = len(pairs)
    return {
        "pairs": n,
        "outputs_differ": sum(a["outputs"] != b["outputs"] for a, b in pairs),
        "actions_differ": sum(a["actions"] != b["actions"] for a, b in pairs),
        "answers_differ": sum(a["answer"] != b["answer"] for a, b in pairs),
        "first_step_differs": sum(a["outputs"][:1] != b["outputs"][:1] for a, b in pairs),
    }


def cross_cache() -> pd.DataFrame:
    rows = []
    for precision, pattern in MAIN.items():
        off, on = (load(RUNS / pattern.format(c)) for c in ("off", "on"))
        for arm in ("greedy", "qwen"):
            for position in ("first", "later"):
                keys = [
                    k
                    for k in off
                    if k[0] == arm
                    and k[3] == 0
                    and k in on
                    and ((k[2] == 0) == (position == "first"))
                ]
                if keys:
                    stats = compare([(off[k], on[k]) for k in keys])
                    rows.append(
                        {
                            "precision": precision,
                            "comparison": "cache off vs on",
                            "arm": arm,
                            "position": position,
                            **stats,
                        }
                    )
                # same scale for contrast: replicate 0 vs 1 of one seed within the cache-on server
                keys = [
                    k
                    for k in on
                    if k[0] == arm
                    and k[3] == 0
                    and k[:3] + (1,) in on
                    and ((k[2] == 0) == (position == "first"))
                ]
                if keys:
                    stats = compare([(on[k], on[k[:3] + (1,)]) for k in keys])
                    rows.append(
                        {
                            "precision": precision,
                            "comparison": "within cache-on server",
                            "arm": arm,
                            "position": position,
                            **stats,
                        }
                    )
    return pd.DataFrame(rows)


# From the vLLM launch logs on the 5070 (logs/vllm_*.log): E00 and E01a with caching off ran on the
# same server launch (01:15 on 2026-10-03); with caching on, on two launches (00:42 and 05:04) that
# loaded the same AOT-compiled graph (torch_compile_cache ab0f71ec). The dev_E01b_launch BF16
# servers loaded a different graph (b136b13b; the launch command had changed), FP8 and AWQ the same
# graphs as in E01b.
E00_VS_E01A = {"off": "E00 vs E01a (same launch)", "on": "E00 vs E01a (two launches, same graph)"}


def launches() -> pd.DataFrame:
    rows = []
    for cache in ("off", "on"):
        e00 = load(RUNS / "E00_pilot" / f"vllm_5070__cache-{cache}.jsonl")
        e01a = load(RUNS / MAIN["bf16"].format(cache))
        pairs = [
            (v, e01a[("greedy", k[1], 0, 0)])
            for k, v in e00.items()
            if k[0] == "T=0.0" and k[2] == 0 and k[3] == 0 and ("greedy", k[1], 0, 0) in e01a
        ]
        if pairs:
            label = E00_VS_E01A[cache]
            rows.append({"precision": "bf16", "cache": cache, "launches": label, **compare(pairs)})
        for precision, pattern in MAIN.items():
            model = Path(pattern).name.split("__")[1]
            check = load(RUNS / "dev_E01b_launch" / f"vllm_5070__{model}__cache-{cache}.jsonl")
            main = load(RUNS / pattern.format(cache))
            pairs = [(v, main[k]) for k, v in check.items() if k in main]
            if pairs:
                rows.append(
                    {
                        "precision": precision,
                        "cache": cache,
                        "launches": "dev_E01b_launch vs E01a/E01b",
                        **compare(pairs),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    results = ROOT / "results"
    cc = cross_cache()
    if cc.empty:
        print("08_cross_launch: no E01 runs, skipping")
        return
    cc.to_csv(results / "08_cross_cache_divergence.csv", index=False)
    launches().to_csv(results / "08_launch_reproducibility.csv", index=False)
    print("08_cross_launch: cross-cache divergence and launch reproducibility written")


if __name__ == "__main__":
    main()
