"""Run an experiment config. Prefix cache is a server-side setting, so say which one the server
was started with; each cache condition is a separate invocation.

Usage: uv run python scripts/run_experiment.py configs/experiments/dev_smoke.yaml \
           --prefix-cache on [--endpoint ollama_mac] [--limit 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentrig.eval.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prefix-cache", choices=["on", "off", "na"], required=True)
    parser.add_argument("--endpoint", default=None, help="override the config's endpoint")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()
    out = run_experiment(args.config, args.prefix_cache, args.endpoint, args.limit)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
