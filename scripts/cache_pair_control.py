"""Read-only checks used by run_cache_pair.sh; no model imports or inference requests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from agentrig.config import load_endpoint, load_model, load_yaml
from agentrig.eval.dataset import load_frozen
from agentrig.eval.runner import condition_key, derive_seed, expand_conditions

ROOT = Path(__file__).resolve().parents[1]


def no_active_runs() -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
            if any(Path(arg.decode()).name == "run_experiment.py" for arg in arguments if arg):
                raise ValueError("an experiment runner is active; refusing service changes")
        except (FileNotFoundError, ProcessLookupError):
            continue


def config_info(path: Path) -> list[str]:
    cfg = load_yaml(path)
    endpoint = cfg["endpoint"]
    endpoints = load_yaml(ROOT / "configs/endpoints.yaml")["endpoints"]
    if endpoints[endpoint]["thinking_control"] != "chat_template_kwargs":
        raise ValueError("cache pairs require a local vLLM endpoint")
    model = load_model(cfg["model"], ROOT / "configs")
    if model["hf_id"] != model["served_names"][endpoint]:
        raise ValueError("served model name must match the configured Hugging Face model")
    name = cfg["name"]
    if not name or any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for char in name
    ):
        raise ValueError(
            "experiment name must contain only letters, digits, underscores or hyphens"
        )
    precision = model["precision"][endpoint]
    # E01's approved policy: BF16 activations/KV and 21,776 tokens at every precision.
    if precision not in {"bf16", "fp8_w8a8_block128", "int4_awq_w4a16_g128"}:
        raise ValueError(f"no validated serving defaults for precision {precision!r}")
    revision = model.get("revision", "")
    if not isinstance(revision, str) or "\n" in revision:
        raise ValueError("model revision must be a single line")
    return [model["hf_id"], endpoint, name, revision, "bfloat16", "1361"]


def cache_labels(metrics: str) -> dict[str, str]:
    """Parse exactly one engine's cache gauge; malformed/ambiguous evidence is an error."""
    samples = [
        line for line in metrics.splitlines() if re.match(r"^(?:vllm:)?cache_config_info\{", line)
    ]
    if len(samples) != 1:
        raise ValueError("expected exactly one cache_config_info sample")
    match = re.fullmatch(r"[^{}]+\{(.*)\}\s+1(?:\.0+)?(?:\s+\d+)?", samples[0])
    if not match:
        raise ValueError("malformed cache_config_info sample")
    labels: dict[str, str] = {}
    remaining = match[1]
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)=("(?:[^"\\]|\\.)*")(?:,|$)')
    while remaining:
        item = pattern.match(remaining)
        if not item or item[1] in labels:
            raise ValueError("malformed or duplicate cache label")
        labels[item[1]] = json.loads(item[2])
        remaining = remaining[item.end() :]
    return labels


class CommandParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("unverifiable vLLM command: " + message)


def serve_arguments(argv: list[str]) -> argparse.Namespace:
    """Accept a vLLM CLI or Python module, never shell/uv command text."""
    executable = Path(argv[0]).name if argv else ""
    if executable == "vllm":
        tokens = argv[1:]
    elif executable.startswith("python") and len(argv) > 1 and Path(argv[1]).name == "vllm":
        tokens = argv[2:]
    elif executable.startswith("python") and argv[1:3] == ["-m", "vllm.entrypoints.cli.main"]:
        tokens = argv[3:]
    else:
        raise ValueError("not a vLLM CLI process")
    if not tokens or tokens[0] != "serve":
        raise ValueError("not a vLLM serve command")
    parser = CommandParser(add_help=False, allow_abbrev=False)
    parser.add_argument("model")
    for key in ("port", "max-model-len", "max-num-seqs", "max-num-batched-tokens"):
        parser.add_argument("--" + key, type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--num-gpu-blocks-override", type=int)
    parser.add_argument("--served-model-name", nargs="+")
    parser.add_argument("--host")
    parser.add_argument("--reasoning-parser")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable-prefix-caching", dest="cache", action="store_const", const="on")
    group.add_argument(
        "--no-enable-prefix-caching", dest="cache", action="store_const", const="off"
    )
    return parser.parse_args(tokens[1:])


def model_matches(actual: str, expected: str, revision: str) -> bool:
    if actual == expected:
        return True
    # This only resolves cached metadata; it cannot contact HF or load weights.
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(expected, "config.json", revision=revision or "main")
    return isinstance(cached, str) and Path(actual).resolve() == Path(cached).parent.resolve()


def check_evidence(
    models: dict[str, Any], metrics: str, argv: list[str], args: argparse.Namespace
) -> None:
    command = serve_arguments(argv)
    cache = cache_labels(metrics)
    served = models.get("data", [])
    model_ok = len(served) == 1 and served[0].get("id") == args.model
    if model_ok and served[0].get("root"):
        model_ok = model_matches(served[0]["root"], args.model, args.revision)
    checks = {
        "served model": model_ok,
        "model path": model_matches(command.model, args.model, args.revision),
        "served-model-name": (command.served_model_name or [command.model]) == [args.model],
        "revision": command.revision == args.revision,
        "port": command.port == args.port,
        "prefix caching": command.cache == args.cache
        and cache.get("enable_prefix_caching") == str(args.cache == "on"),
        "context length": command.max_model_len == args.max_len,
        "GPU memory fraction": command.gpu_memory_utilization == args.gpu_mem
        and float(cache.get("gpu_memory_utilization", "nan")) == args.gpu_mem,
        "maximum sequences": command.max_num_seqs == args.max_num_seqs,
        "batch token budget": command.max_num_batched_tokens == args.max_num_batched_tokens,
        "dtype": command.dtype == args.dtype and args.dtype in {"bfloat16", "float16"},
        "KV dtype": cache.get("cache_dtype") in {"auto", args.dtype},
        "KV blocks override": command.num_gpu_blocks_override == args.kv_blocks
        and cache.get("num_gpu_blocks_override") == str(args.kv_blocks),
        "KV allocation": int(cache.get("num_gpu_blocks", "0")) > 0,
        "block size": cache.get("block_size") == "16",
    }
    if args.kv_blocks is not None:
        checks["KV capacity"] = int(cache["num_gpu_blocks"]) == args.kv_blocks
        if "kv_cache_size_tokens" in cache:
            checks["KV tokens"] = int(cache["kv_cache_size_tokens"]) == 16 * args.kv_blocks
    mismatch = [field for field, correct in checks.items() if not correct]
    if mismatch:
        raise ValueError("server configuration mismatch: " + ", ".join(mismatch))


def process_arguments(pid: int) -> list[str]:
    return [
        part.decode()
        for part in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if part
    ]


def listener_command(listener: int, owned: set[int]) -> list[str]:
    """Walk only this listener's owned ancestors, excluding unrelated sibling servers."""
    pid = listener
    while pid in owned:
        argv = process_arguments(pid)
        try:
            serve_arguments(argv)
        except ValueError:
            status = (Path("/proc") / str(pid) / "status").read_text()
            parent = re.search(r"^PPid:\s+(\d+)", status, re.M)
            if not parent:
                break
            pid = int(parent[1])
        else:
            return argv
    raise ValueError("cannot attribute listener to a verifiable vLLM serve command")


def fetch_local(path: str, port: int, api_key: str) -> bytes:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {api_key}"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read()


def check_server(args: argparse.Namespace) -> None:
    endpoint = load_endpoint(args.endpoint)
    owned, listeners = managed_listener_pids(args.port)
    commands = [listener_command(pid, owned) for pid in listeners]
    models = json.loads(fetch_local("/v1/models", args.port, endpoint.api_key))
    metrics = fetch_local("/metrics", args.port, endpoint.api_key).decode()
    for argv in commands:
        check_evidence(models, metrics, argv, args)
    if managed_listener_pids(args.port) != (owned, listeners):
        raise ValueError("managed process tree changed during server verification")


def gpu_pids() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not line.isdecimal() for line in lines):
        raise ValueError("GPU process query unavailable; refusing to assume memory is released")
    return {int(line) for line in lines}


def port_listeners(port: int) -> str:
    return subprocess.check_output(["ss", "-H", "-ltnp", f"sport = :{port}"], text=True)


def managed_listener_pids(port: int) -> tuple[set[int], set[int]]:
    """Require one vllm pane and attribute every port listener to its process tree."""

    panes = subprocess.check_output(
        ["tmux", "list-panes", "-s", "-t", "=vllm", "-F", "#{pane_pid}"],
        text=True,
    ).splitlines()
    if len(panes) != 1:
        raise ValueError("vllm must contain exactly one pane")
    descendants = {int(panes[0])}
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            status = (entry / "status").read_text()
            match = re.search(r"^PPid:\s+(\d+)", status, re.M)
            if match:
                parents[int(entry.name)] = int(match.group(1))
        except (FileNotFoundError, ProcessLookupError):
            continue
    for _ in range(len(parents)):
        children = {pid for pid, parent in parents.items() if parent in descendants}
        if children <= descendants:
            break
        descendants |= children
    listeners = {int(pid) for pid in re.findall(r"pid=(\d+)", port_listeners(port))}
    if not listeners or not listeners <= descendants:
        raise ValueError("port listener is not owned by the vllm pane")
    return descendants, listeners


def managed_service(port: int) -> None:
    descendants, _ = managed_listener_pids(port)
    if not gpu_pids() <= descendants:
        raise ValueError("unrelated GPU compute processes are active")


def released(port: int) -> bool:
    """Windows graphics memory need not be zero; no CUDA compute process may remain."""
    if port_listeners(port) or gpu_pids():
        return False
    # Record the actual memory counter after compute processes have exited.
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if not memory or any(not line.strip().isdecimal() for line in memory.splitlines()):
        raise ValueError("GPU memory query unavailable")
    print(f"port released; no GPU compute processes; memory.used={memory} MiB (includes graphics)")
    return True


def check_runs(path: Path, endpoint: str, cache: str, existing: bool = False) -> None:
    cfg = load_yaml(path)
    questions = load_frozen(ROOT / cfg["eval_set"])
    if cfg.get("question_ids"):
        by_id = {question["id"]: question for question in questions}
        questions = [by_id[key] for key in cfg["question_ids"]]
    else:
        questions = questions[: cfg.get("n_questions", len(questions))]
    conditions = expand_conditions(cfg, cache)
    by_condition = {
        (cond["arm"], float(cond["temperature"]), cond["enable_thinking"]): cond
        for cond in conditions
    }
    expected = {
        (question["id"], cond["arm"], float(cond["temperature"]), cond["enable_thinking"], repeat)
        for question in questions
        for cond in conditions
        for repeat in range(cond["n_repeats"])
    }
    directory = ROOT / "runs" / cfg["name"]
    legacy = directory / f"{endpoint}__cache-{cache}.jsonl"
    if legacy.exists():
        raise ValueError(
            "legacy trajectory path detected; use a new experiment name "
            "or explicitly migrate records"
        )
    target = directory / f"{endpoint}__{cfg['model']}__cache-{cache}.jsonl"
    latest = {}
    if existing and not target.exists():
        return
    for line in target.read_text().splitlines():
        record = json.loads(line)
        condition = record["condition"]
        if condition["endpoint"] != endpoint or condition["prefix_cache"] != cache:
            raise ValueError("trajectory contains a different endpoint or cache condition")
        condition_id = (
            condition.get("arm", "default"),
            float(condition["temperature"]),
            condition["enable_thinking"],
        )
        if condition_id not in by_condition:
            raise ValueError("existing trajectory has a different experimental arm")
        expected_condition = by_condition[condition_id]
        for field in ["top_p", "top_k", "max_tokens_per_step", "n_repeats", "replicates_per_seed"]:
            if condition.get(field) != expected_condition[field]:
                raise ValueError(f"existing trajectory has different {field}; refusing to resume")
        for field, default in {"max_steps": None, "search_top_k": 2}.items():
            if condition.get(field) != cfg.get(field, default):
                raise ValueError(f"existing trajectory has different {field}; refusing to resume")
        if condition.get("model") != cfg["model"]:
            raise ValueError("existing trajectory has a different logical model")
        seed = derive_seed(
            cfg["seed"],
            record["question_id"],
            record["repeat"] // expected_condition["replicates_per_seed"],
        )
        if record.get("meta", {}).get("run_seed") != seed:
            raise ValueError("existing trajectory has a different seed scheme")
        identifier = (
            f"{cfg['name']}/{endpoint}/{cfg['model']}/"
            f"{condition_key(expected_condition)}/{record['question_id']}/r{record['repeat']:02d}"
        )
        if record.get("run_id") != identifier:
            raise ValueError("existing trajectory has a different run identifier format")
        key = (record["question_id"], *condition_id, record["repeat"])
        latest[key] = record
    if existing:
        return
    if (
        not expected
        or expected - latest.keys()
        or any(latest[key].get("error") is not None for key in expected & latest.keys())
    ):
        raise ValueError("experiment has missing or failed runs; keeping all existing records")
    print(f"verified {len(expected)} completed runs for cache={cache}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["config", "idle", "server", "owned", "released", "runs"]
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--cache", choices=["on", "off"])
    parser.add_argument("--model")
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--revision", default="")
    parser.add_argument("--kv-blocks", type=int)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    args = parser.parse_args()
    try:
        if args.command == "config":
            print("\n".join(config_info(args.config)))
        elif args.command == "idle":
            no_active_runs()
        elif args.command == "server":
            check_server(args)
        elif args.command == "owned":
            managed_service(args.port)
        elif args.command == "released":
            raise SystemExit(0 if released(args.port) else 1)
        else:
            check_runs(args.config, args.endpoint, args.cache, args.existing)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print(f"Check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
