"""CPU-only checks for serving-state admission and completed trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "cache_pair_control", Path(__file__).resolve().parents[1] / "scripts/cache_pair_control.py"
)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def evidence() -> tuple[argparse.Namespace, dict, str, list[str]]:
    args = argparse.Namespace(
        model="Qwen/test",
        cache="on",
        dtype="bfloat16",
        gpu_mem=0.90,
        max_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        revision="abc123",
        kv_blocks=1361,
        port=8000,
        endpoint="vllm_5070",
    )
    models = {"data": [{"id": "Qwen/test", "root": "Qwen/test"}]}
    metrics = (
        "# HELP vllm:cache_config_info Configuration\n"
        'vllm:cache_config_info{enable_prefix_caching="True",gpu_memory_utilization="0.9",'
        'cache_dtype="auto",num_gpu_blocks="1361",num_gpu_blocks_override="1361",'
        'block_size="16",kv_cache_size_tokens="21776"} 1.0\n'
    )
    argv = [
        "/venv/bin/python3.12",
        "/venv/bin/vllm",
        "serve",
        "Qwen/test",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--served-model-name",
        "Qwen/test",
        "--dtype",
        "bfloat16",
        "--enable-prefix-caching",
        "--reasoning-parser",
        "qwen3",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.90",
        "--revision",
        "abc123",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--num-gpu-blocks-override",
        "1361",
    ]
    return args, models, metrics, argv


def test_actual_cache_and_serving_settings_must_match() -> None:
    args, models, metrics, argv = evidence()
    control.check_evidence(models, metrics, argv, args)
    with pytest.raises(ValueError, match="prefix caching"):
        control.check_evidence(models, metrics.replace('"True"', '"False"'), argv, args)
    args.cache = "off"
    argv[argv.index("--enable-prefix-caching")] = "--no-enable-prefix-caching"
    control.check_evidence(models, metrics.replace('"True"', '"False"'), argv, args)


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--revision", "wrong", "revision"),
        ("--max-model-len", "4096", "context"),
        ("--port", "9000", "port"),
        ("--dtype", "float16", "dtype"),
        ("--max-num-seqs", "2", "maximum sequences"),
        ("--max-num-batched-tokens", "1024", "batch token"),
        ("--num-gpu-blocks-override", "3661", "KV blocks"),
        ("--gpu-memory-utilization", "0.8", "GPU memory"),
    ],
)
def test_cmdline_mismatches_are_rejected(flag: str, value: str, message: str) -> None:
    args, models, metrics, argv = evidence()
    argv[argv.index(flag) + 1] = value
    with pytest.raises(ValueError, match=message):
        control.check_evidence(models, metrics, argv, args)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('num_gpu_blocks="1361"', 'num_gpu_blocks="3661"'),
        ('num_gpu_blocks_override="1361"', 'num_gpu_blocks_override="None"'),
        ('kv_cache_size_tokens="21776"', 'kv_cache_size_tokens="58576"'),
        ('cache_dtype="auto"', 'cache_dtype="fp8"'),
        ('block_size="16"', 'block_size="32"'),
    ],
)
def test_actual_kv_metrics_must_match(old: str, new: str) -> None:
    args, models, metrics, argv = evidence()
    with pytest.raises(ValueError):
        control.check_evidence(models, metrics.replace(old, new), argv, args)


@pytest.mark.parametrize(
    "metrics",
    [
        "",
        "vllm:cache_config_info{bad} 1.0",
        'vllm:cache_config_info{a="1",a="2"} 1.0',
        'vllm:cache_config_info{a="1"} 0.0',
        'vllm:cache_config_info{a="1"} 1.0\nvllm:cache_config_info{a="2"} 1.0',
    ],
)
def test_missing_malformed_or_multi_engine_metrics_fail_closed(metrics: str) -> None:
    with pytest.raises(ValueError):
        control.cache_labels(metrics)


def test_offline_snapshot_and_served_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    snapshot = "/cache/models--Qwen--test/snapshots/abc123"
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo, filename, revision: (
            snapshot + "/config.json"
            if (repo, filename, revision) == ("Qwen/test", "config.json", "abc123")
            else None
        ),
    )
    args, models, metrics, argv = evidence()
    models["data"][0]["root"] = snapshot
    argv[3] = snapshot
    control.check_evidence(models, metrics, argv, args)
    argv[3] = snapshot.replace("abc123", "badrev")
    with pytest.raises(ValueError, match="model path"):
        control.check_evidence(models, metrics, argv, args)
    models["data"][0]["id"] = "Wrong/model"
    with pytest.raises(ValueError, match="served model"):
        control.check_evidence(models, metrics, argv, args)


def test_command_rejects_indirect_settings_and_shell_wrappers() -> None:
    _, _, _, argv = evidence()
    for candidate in [
        ["bash", "-c", " ".join(argv)],
        ["uv", "run", *argv],
        [*argv, "--config", "uninspected.yaml"],
        [*argv, "--no-enable-prefix-caching"],
    ]:
        with pytest.raises(ValueError):
            control.serve_arguments(candidate)
    equals = ["/venv/bin/vllm", *argv[2:]]
    equals[equals.index("--revision") : equals.index("--revision") + 2] = ["--revision=abc123"]
    assert control.serve_arguments(equals).revision == "abc123"


def test_check_server_uses_only_public_read_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    args, models, metrics, argv = evidence()
    monkeypatch.setattr(control, "load_endpoint", lambda _: SimpleNamespace(api_key="EMPTY"))
    monkeypatch.setattr(control, "managed_listener_pids", lambda _: ({1, 2}, {2}))
    monkeypatch.setattr(control, "listener_command", lambda *a: argv)
    paths = []

    def fetch(path: str, port: int, api_key: str) -> bytes:
        paths.append(path)
        assert port == 8000
        return json.dumps(models).encode() if path == "/v1/models" else metrics.encode()

    monkeypatch.setattr(control, "fetch_local", fetch)
    control.check_server(args)
    assert paths == ["/v1/models", "/metrics"]


def test_owned_listener_cannot_be_replaced_mid_check(monkeypatch: pytest.MonkeyPatch) -> None:
    args, models, metrics, argv = evidence()
    monkeypatch.setattr(control, "load_endpoint", lambda _: SimpleNamespace(api_key="EMPTY"))
    trees = iter([({1, 2}, {2}), ({1, 3}, {3})])
    monkeypatch.setattr(control, "managed_listener_pids", lambda _: next(trees))
    monkeypatch.setattr(control, "listener_command", lambda *a: argv)
    monkeypatch.setattr(
        control,
        "fetch_local",
        lambda path, *a: json.dumps(models).encode() if path == "/v1/models" else metrics.encode(),
    )
    with pytest.raises(ValueError, match="process tree changed"):
        control.check_server(args)


def test_listener_command_uses_only_owned_ancestors(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, argv = evidence()
    monkeypatch.setattr(
        control, "process_arguments", lambda pid: argv if pid == 11 else ["VLLM::APIServer"]
    )
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: "PPid:\t11\n")
    assert control.listener_command(12, {11, 12}) == argv
    with pytest.raises(ValueError, match="cannot attribute"):
        control.listener_command(12, {12})


def test_listener_ownership_rejects_unrelated_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **k: "101\n")
    monkeypatch.setattr(Path, "iterdir", lambda _: iter([]))
    monkeypatch.setattr(control, "port_listeners", lambda _: 'users:(("python",pid=999,fd=3))')
    with pytest.raises(ValueError, match="not owned"):
        control.managed_listener_pids(8000)


def test_model_defaults_use_registered_revision_and_precision() -> None:
    root = Path(__file__).resolve().parents[1]
    for suffix, revision in [
        ("fp8", "96b30dc13593a244a5e59e84687309f53c375cfa"),
        ("awq", "74d4bd2bd4bff9cafc9345221320bffb08b406a3"),
    ]:
        info = control.config_info(root / f"configs/experiments/E01b_{suffix}.yaml")
        assert info[3:] == [revision, "bfloat16", "1361"]


def write_trajectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list]:
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "load_frozen", lambda _: [{"id": "q1"}])
    cfg = {
        "name": "E_TEST",
        "seed": 42,
        "model": "qwen-test",
        "eval_set": "data/frozen/hotpotqa_distractor_val_300.jsonl",
        "n_questions": 1,
        "n_repeats": 1,
        "temperatures": [0.7],
        "enable_thinking": [False, True],
        "top_p": 0.8,
        "top_k": 20,
        "max_steps": 8,
        "max_tokens_per_step": 1024,
        "search_top_k": 2,
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg))
    target = tmp_path / "runs/E_TEST/vllm_test__qwen-test__cache-on.jsonl"
    target.parent.mkdir(parents=True)
    rows = [
        {
            "question_id": "q1",
            "repeat": 0,
            "error": None,
            "condition": {
                "endpoint": "vllm_test",
                "prefix_cache": "on",
                "temperature": 0.7,
                "enable_thinking": thinking,
                "model": "qwen-test",
                "top_p": 0.8,
                "top_k": 20,
                "max_steps": 8,
                "max_tokens_per_step": 1024,
                "search_top_k": 2,
            },
        }
        for thinking in [False, True]
    ]
    for row in rows:
        row["condition"].update(arm="default", n_repeats=1, replicates_per_seed=1)
        think = "on" if row["condition"]["enable_thinking"] else "off"
        row["run_id"] = f"E_TEST/vllm_test/qwen-test/T0.7_think-{think}_cache-on/q1/r00"
        row["meta"] = {"run_seed": control.derive_seed(42, "q1", 0)}
    return path, target, rows


def test_resume_requires_all_thinking_conditions_and_no_latest_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, target, rows = write_trajectory(tmp_path, monkeypatch)
    target.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="missing or failed"):
        control.check_runs(path, "vllm_test", "on")
    error = {**rows[0], "error": "old failed attempt"}
    target.write_text("\n".join(map(json.dumps, [error, *rows])) + "\n")
    before = target.read_bytes()
    control.check_runs(path, "vllm_test", "on")
    assert target.read_bytes() == before
    target.write_text("\n".join(map(json.dumps, [*rows, error])) + "\n")
    with pytest.raises(ValueError, match="missing or failed"):
        control.check_runs(path, "vllm_test", "on")


def test_changed_sampling_config_cannot_silently_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, target, rows = write_trajectory(tmp_path, monkeypatch)
    rows[0]["condition"]["top_p"] = 1.0
    target.write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="different top_p"):
        control.check_runs(path, "vllm_test", "on", existing=True)


def test_memory_release_requires_no_compute_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "port_listeners", lambda _: "")
    monkeypatch.setattr(control, "gpu_pids", lambda: {999})
    assert not control.released(8000)
    monkeypatch.setattr(control, "gpu_pids", lambda: set())
    monkeypatch.setattr(control.subprocess, "check_output", lambda *a, **k: "1800\n")
    assert control.released(8000)  # Graphics memory is allowed to remain allocated.


def test_unsupported_gpu_query_cannot_be_treated_as_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="N/A\n"))
    with pytest.raises(ValueError, match="unavailable"):
        control.gpu_pids()


def test_arms_with_same_temperature_remain_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, target, rows = write_trajectory(tmp_path, monkeypatch)
    cfg = yaml.safe_load(path.read_text())
    cfg["enable_thinking"] = False
    cfg["arms"] = [
        {"name": "qwen", "temperature": 0.7, "n_repeats": 1},
        {"name": "topp1", "temperature": 0.7, "n_repeats": 2, "top_p": 1.0},
    ]
    path.write_text(yaml.safe_dump(cfg))
    complete = []
    for arm, count, top_p in [("qwen", 1, 0.8), ("topp1", 2, 1.0)]:
        for repeat in range(count):
            row = json.loads(json.dumps(rows[0]))
            row["repeat"] = repeat
            row["condition"].update(arm=arm, n_repeats=count, top_p=top_p)
            row["run_id"] = (
                f"E_TEST/vllm_test/qwen-test/{arm}_T0.7_think-off_cache-on/q1/r{repeat:02d}"
            )
            row["meta"]["run_seed"] = control.derive_seed(42, "q1", repeat)
            complete.append(row)
    target.write_text("\n".join(map(json.dumps, complete)) + "\n")
    control.check_runs(path, "vllm_test", "on")
    target.write_text("\n".join(map(json.dumps, complete[:-1])) + "\n")
    with pytest.raises(ValueError, match="missing or failed"):
        control.check_runs(path, "vllm_test", "on")


def test_legacy_path_is_not_silently_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, target, _ = write_trajectory(tmp_path, monkeypatch)
    legacy = target.parent / "vllm_test__cache-on.jsonl"
    legacy.write_text("existing legacy results\n")
    with pytest.raises(ValueError, match="legacy trajectory path"):
        control.check_runs(path, "vllm_test", "on", existing=True)
    assert legacy.read_text() == "existing legacy results\n"
