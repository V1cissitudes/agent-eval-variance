"""CPU-only shell integration; all service/GPU commands are forbidden or stubbed."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suffix", ["fp8", "awq"])
def test_dry_run_resolves_model_defaults_without_services(tmp_path: Path, suffix: str) -> None:
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    marker = tmp_path / "called"
    for name in ["tmux", "curl", "nvidia-smi", "ss"]:
        tool = forbidden / name
        tool.write_text(f"#!/bin/sh\necho called > '{marker}'\nexit 99\n")
        tool.chmod(0o755)
    env = {**os.environ, "PATH": f"{forbidden}:{os.environ['PATH']}", "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [
            "bash",
            "scripts/run_cache_pair.sh",
            "--dry-run",
            f"configs/experiments/E01b_{suffix}.yaml",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not marker.exists()
    assert "--kv-blocks 1361" in result.stdout
    revision = (
        "96b30dc13593a244a5e59e84687309f53c375cfa"
        if suffix == "fp8"
        else "74d4bd2bd4bff9cafc9345221320bffb08b406a3"
    )
    assert f"--revision {revision}" in result.stdout
    assert "HF_HUB_OFFLINE" in result.stdout
    assert result.stdout.count("python scripts/cache_pair_control.py server ") == 3


def test_serve_forwards_revision_kv_and_stable_name_to_fake_uv(tmp_path: Path) -> None:
    output = tmp_path / "argv.json"
    uv = tmp_path / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\nimport json, os, sys\n"
        "open(os.environ['CAPTURE'], 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    uv.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CAPTURE": str(output),
        "HF_HUB_OFFLINE": "1",
    }
    subprocess.run(
        [
            "bash",
            "scripts/serve_vllm.sh",
            "Qwen/Qwen3-4B-AWQ",
            "--revision",
            "abc123",
            "--kv-blocks",
            "1361",
            "--prefix-cache",
            "off",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    )
    argv = json.loads(output.read_text())
    assert argv[argv.index("--num-gpu-blocks-override") + 1] == "1361"
    assert argv[argv.index("--revision") + 1] == "abc123"
    assert argv[argv.index("--served-model-name") + 1] == "Qwen/Qwen3-4B-AWQ"
    assert "--no-enable-prefix-caching" in argv


def test_dry_run_explicit_values_override_defaults() -> None:
    result = subprocess.run(
        [
            "bash",
            "scripts/run_cache_pair.sh",
            "--dry-run",
            "configs/experiments/E01b_awq.yaml",
            "--revision",
            "other-commit",
            "--kv-blocks",
            "1200",
            "--dtype",
            "float16",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--kv-blocks 1200" in result.stdout
    assert "--revision other-commit" in result.stdout
    assert "--dtype float16" in result.stdout
