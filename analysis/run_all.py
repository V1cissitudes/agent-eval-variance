"""Run every analysis/NN_*.py and NN_*.R script in numeric order (used by `make figures`).

Convention: each script reads only runs/, data/frozen/ and results/, and writes only
results/ and figures/, so `make figures` can regenerate both directories from scratch.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT_PATTERN = re.compile(r"^\d{2}_.+\.(py|R)$")


def main() -> None:
    scripts = sorted(p for p in HERE.iterdir() if SCRIPT_PATTERN.match(p.name))
    for script in scripts:
        print(f"==> {script.name}")
        cmd = ["Rscript", str(script)] if script.suffix == ".R" else [sys.executable, str(script)]
        subprocess.run(cmd, check=True, cwd=HERE.parent)
    print(f"Done: {len(scripts)} script(s).")


if __name__ == "__main__":
    main()
