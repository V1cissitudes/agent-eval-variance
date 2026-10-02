"""Load a frozen evaluation set, refusing to proceed if it does not match its manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentrig.eval.freeze import manifest_path_for, sha256_file


class FrozenSetError(RuntimeError):
    """The frozen file was modified or its manifest is missing."""


def load_manifest(jsonl_path: Path) -> dict[str, Any]:
    path = manifest_path_for(jsonl_path)
    if not path.exists():
        raise FrozenSetError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen(jsonl_path: Path) -> dict[str, Any]:
    """Check SHA256 and item count against the manifest; return the manifest."""
    manifest = load_manifest(jsonl_path)
    actual = sha256_file(jsonl_path)
    if actual != manifest["sha256"]:
        raise FrozenSetError(f"{jsonl_path} SHA256 {actual} != manifest {manifest['sha256']}")
    return manifest


def load_frozen(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """The only supported way to read evaluation questions."""
    path = Path(jsonl_path)
    manifest = verify_frozen(path)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if len(records) != manifest["n"]:
        raise FrozenSetError(f"{path} has {len(records)} records, manifest says {manifest['n']}")
    return records
