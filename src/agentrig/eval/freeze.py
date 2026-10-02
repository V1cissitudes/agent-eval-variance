"""Deterministic sampling and freezing of evaluation sets (JSONL + manifest with SHA256).

Sampling sorts items by sha256(f"{seed}:{id}") and takes the first n. Unlike
random.sample, this does not depend on Python's RNG implementation, so the same
seed gives the same subset on any machine and Python version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def hash_key(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def sample_ids(ids: Iterable[str], n: int, seed: int) -> list[str]:
    """Return n ids chosen deterministically by hash order."""
    unique = sorted(set(ids))
    if n > len(unique):
        raise ValueError(f"Requested {n} items but only {len(unique)} available")
    return sorted(unique, key=lambda i: hash_key(seed, i))[:n]


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path_for(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix(".manifest.json")
